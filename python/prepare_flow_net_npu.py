#!/usr/bin/env python3
"""Prepare NPU-ready float subgraphs for the pinyin project.

flow_net_step: bake decode_steps=1.0 (fixed in this model's inference scheme) so
the time-embedding branch constant-folds and the scalar-broadcast Gemm shape
problem disappears; staticize remaining inputs.

Output: models/subgraphs/npu/flow_net_step.onnx (+ equivalence check)

Usage:
    python python/prepare_flow_net_npu.py
"""
from __future__ import annotations

import json
import os

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("TMPDIR", os.path.join(PROJ, ".work_tmp", "tmp"))

import numpy as np  # noqa: E402
import onnx  # noqa: E402
import onnxruntime as ort  # noqa: E402
from onnx import helper, numpy_helper  # noqa: E402

SRC = os.path.join(PROJ, "models", "subgraphs", "raw", "flow_net_step.onnx")
OUT_DIR = os.path.join(PROJ, "models", "subgraphs", "npu")
DST = os.path.join(OUT_DIR, "flow_net_step.onnx")


def bake(model: onnx.ModelProto, name: str, value: np.ndarray) -> None:
    const_name = f"{name}_baked"
    node = helper.make_node("Constant", [], [const_name],
                            value=numpy_helper.from_array(value, name=const_name),
                            name=f"bake_{name}")
    kept = [i for i in model.graph.input if i.name != name]
    del model.graph.input[:]
    model.graph.input.extend(kept)
    model.graph.node.insert(0, node)
    for n in model.graph.node:
        for i in range(len(n.input)):
            if n.input[i] == name:
                n.input[i] = const_name


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    model = onnx.load(SRC, load_external_data=False)
    bake(model, "decode_steps", np.asarray(1.0, dtype=np.float32))
    for inp in model.graph.input:
        for dim in inp.type.tensor_type.shape.dim:
            if dim.dim_param:
                dim.ClearField("dim_param")
                dim.dim_value = 1
    input_names = {i.name for i in model.graph.input}
    kept_vi = [vi for vi in model.graph.value_info if vi.name not in input_names]
    del model.graph.value_info[:]
    model.graph.value_info.extend(kept_vi)
    model = onnx.shape_inference.infer_shapes(model, strict_mode=False)
    import onnxsim

    model, ok = onnxsim.simplify(model)
    if not ok:
        raise RuntimeError("onnxsim failed")
    onnx.checker.check_model(model)
    onnx.save(model, DST)

    # equivalence vs original (decode_steps=1.0)
    orig = ort.InferenceSession(SRC, providers=["CPUExecutionProvider"])
    new = ort.InferenceSession(DST, providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(0)
    info = {"dst": os.path.relpath(DST, PROJ), "inputs": [i.name for i in model.graph.input],
            "size_mb": round(os.path.getsize(DST) / 1e6, 1), "checks": []}
    for i in range(3):
        cond = (rng.standard_normal((1, 1024)) * 0.5).astype(np.float32)
        noise = (rng.standard_normal((1, 32)) * 0.5).astype(np.float32)
        a = orig.run(None, {"conditioning": cond, "noise": noise,
                            "decode_steps": np.asarray(1.0, np.float32)})[0]
        b = new.run(None, {"conditioning": cond, "noise": noise})[0]
        info["checks"].append(float(np.max(np.abs(a - b))))
    info["max_abs_diff"] = max(info["checks"])
    with open(os.path.join(OUT_DIR, "prepare_info.json"), "w") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)
    print(json.dumps(info, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
