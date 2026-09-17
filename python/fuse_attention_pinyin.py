#!/usr/bin/env python3
"""Fuse the attention clusters of the pinyin int8 AR graph into opset-23 `Attention`.

The pinyin AR graph is vendor ORT dynamic int8 (QOperator): the FlowLM
projections are `MatMulInteger` while the attention core (Transpose / MatMul /
mask / Softmax / MatMul) stays float and feeds the quantized out_proj path.
Anchor pattern per layer (found from each Softmax):

    Softmax <- Where <- Div <- MatMul(Transpose(q), Transpose(k))
    AV = MatMul(Softmax, Transpose(v)) -> Transpose -> Reshape -> DynamicQuantizeLinear

Replace the whole cluster with Reshape x3 + one `Attention` node, rewiring the
DynamicQuantizeLinear input to the Attention output (int8 projection path kept).

Input : models/subgraphs/flow/flow_ar_step.onnx
Output: models/subgraphs/flow/flow_ar_step_fused.onnx

Usage:
    python python/fuse_attention_pinyin.py
"""
from __future__ import annotations

import argparse
import json
import os

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("TMPDIR", os.path.join(PROJ, ".work_tmp", "tmp"))

import numpy as np  # noqa: E402
import onnx  # noqa: E402
from onnx import helper, numpy_helper  # noqa: E402

SRC = os.path.join(PROJ, "models", "subgraphs", "flow", "flow_ar_step.onnx")
DST = os.path.join(PROJ, "models", "subgraphs", "flow", "flow_ar_step_fused.onnx")

NUM_HEADS = 16
HEAD_DIM = 64
SCALE = 1.0 / (HEAD_DIM ** 0.5)


def prune_and_sort(model: onnx.ModelProto) -> None:
    nodes = list(model.graph.node)
    producer = {o: n for n in nodes for o in n.output}
    needed: set[str] = set()
    stack = [o.name for o in model.graph.output]
    while stack:
        t = stack.pop()
        n = producer.get(t)
        if n is None or n.name in needed:
            continue
        needed.add(n.name)
        stack.extend(n.input)
    kept = [n for n in nodes if n.name in needed]
    by_out = {}
    for n in kept:
        for o in n.output:
            by_out[o] = n.name
    deps = {n.name: {by_out[t] for t in n.input if t in by_out} for n in kept}
    consumers: dict[str, set[str]] = {}
    for n in kept:
        for t in n.input:
            if t in by_out:
                consumers.setdefault(by_out[t], set()).add(n.name)
    node_by_name = {n.name: n for n in kept}
    ready = [n.name for n in kept if not deps[n.name]]
    order = []
    while ready:
        name = ready.pop()
        order.append(node_by_name[name])
        for c in sorted(consumers.get(name, ())):
            deps[c].discard(name)
            if not deps[c]:
                ready.append(c)
    if len(order) != len(kept):
        raise RuntimeError("cycle after prune")
    del model.graph.node[:]
    for n in order:
        model.graph.node.append(n)
    used = {t for n in order for t in n.input}
    kept_inits = [i for i in model.graph.initializer if i.name in used]
    del model.graph.initializer[:]
    model.graph.initializer.extend(kept_inits)


def rewrite(model: onnx.ModelProto) -> dict:
    prod = {o: n for n in model.graph.node for o in n.output}
    cons: dict[str, list] = {}
    for n in model.graph.node:
        for i in n.input:
            cons.setdefault(i, []).append(n)

    softmaxes = [n for n in model.graph.node if n.op_type == "Softmax"]
    if len(softmaxes) != 6:
        raise RuntimeError(f"expected 6 flow softmaxes, found {len(softmaxes)}")

    new_nodes = []
    removed: set[str] = set()
    for idx, smax in enumerate(softmaxes):
        where = prod[smax.input[0]]
        div = prod[where.input[1]]
        qk = prod[div.input[0]]
        t_q = prod[qk.input[0]]
        t_k = prod[qk.input[1]]
        av = [c for c in cons[smax.output[0]] if c.op_type == "MatMul"][0]
        t_v = prod[av.input[1]]
        for node, op in ((where, "Where"), (div, "Div"), (qk, "MatMul"),
                         (t_q, "Transpose"), (t_k, "Transpose"),
                         (t_v, "Transpose"), (av, "MatMul")):
            if node.op_type != op:
                raise RuntimeError(f"{smax.name}: expected {op}, got {node.op_type}")

        chain = []
        cur = av.output[0]
        final_tensor = None
        for _ in range(4):
            nxt = [c for c in cons.get(cur, []) if c.op_type in ("Transpose", "Reshape")]
            if not nxt:
                raise RuntimeError(f"{smax.name}: cannot find output chain tail")
            node = nxt[0]
            chain.append(node)
            cur = node.output[0]
            users = cons.get(cur, [])
            if any(u.op_type == "DynamicQuantizeLinear" for u in users):
                final_tensor = cur
                break
        if final_tensor is None:
            raise RuntimeError(f"{smax.name}: no DynamicQuantizeLinear after attention")

        q_src, k_src, v_src = t_q.input[0], t_k.input[0], t_v.input[0]
        base = f"/fused_attn{idx}"
        shape_const = numpy_helper.from_array(np.asarray([0, 0, -1], dtype=np.int64),
                                              name=base + "_shape")
        new_nodes.append(helper.make_node("Constant", [], [base + "_shape_c"],
                                          value=shape_const, name=base + "_shape_node"))
        for side, src in (("q", q_src), ("k", k_src), ("v", v_src)):
            new_nodes.append(helper.make_node("Reshape", [src, base + "_shape_c"],
                                              [f"{base}_{side}"], name=f"{base}_reshape_{side}"))
        new_nodes.append(helper.make_node(
            "Attention", [base + "_q", base + "_k", base + "_v"], [base + "_out"],
            name=base, q_num_heads=NUM_HEADS, kv_num_heads=NUM_HEADS,
            scale=SCALE, is_causal=0))

        for user in cons.get(final_tensor, []):
            for i in range(len(user.input)):
                if user.input[i] == final_tensor and user.op_type == "DynamicQuantizeLinear":
                    user.input[i] = base + "_out"

        removed.update(n.name for n in [where, div, qk, t_q, t_k, t_v, av, *chain])

    kept = [n for n in model.graph.node if n.name not in removed]
    kept.extend(new_nodes)
    del model.graph.node[:]
    for n in kept:
        model.graph.node.append(n)

    for imp in model.opset_import:
        if imp.domain in ("", "ai.onnx"):
            imp.domain = ""
            imp.version = max(imp.version, 23)
    return {"removed_nodes": sorted(removed), "added_attention": len(softmaxes)}


def main() -> None:
    ap = argparse.ArgumentParser(description="fuse pinyin AR attention")
    ap.add_argument("--src", default=SRC)
    ap.add_argument("--dst", default=DST)
    args = ap.parse_args()

    model = onnx.load(args.src, load_external_data=False)
    info = rewrite(model)
    prune_and_sort(model)
    import onnxsim

    model, ok = onnxsim.simplify(model)
    if not ok:
        raise RuntimeError("onnxsim failed")
    onnx.checker.check_model(model)
    onnx.save(model, args.dst)
    info["dst"] = os.path.relpath(args.dst, PROJ)
    info["nodes_before"] = len(onnx.load(args.src, load_external_data=False).graph.node)
    info["nodes_after"] = len(onnx.load(args.dst, load_external_data=False).graph.node)
    info["size_mb"] = round(os.path.getsize(args.dst) / 1e6, 1)
    with open(os.path.join(os.path.dirname(args.dst), "ar_fused_info.json"), "w") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)
    print(json.dumps(info, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
