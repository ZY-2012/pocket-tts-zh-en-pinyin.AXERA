#!/usr/bin/env python3
"""M0 baseline for pocket-tts-zh-en-pinyin (vendor int8 ONNX, host CPU).

Uses the vendor PinyinTokenizer + StepRuntime so token ids and audio are the
reference for all later stages.

Usage:
    python python/baseline_run_pinyin.py
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import platform
import sys
import time

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VENDOR = os.environ.get("PINYIN_TTS_ROOT", "/data/shared/huyuan/TTS/pocket-tts-zh-en-pinyin")

WORK_TMP = os.path.join(PROJ, ".work_tmp")
os.makedirs(os.path.join(WORK_TMP, "tmp"), exist_ok=True)
os.environ["TMPDIR"] = os.path.join(WORK_TMP, "tmp")

sys.path.insert(0, VENDOR)
sys.path.insert(0, PROJ)

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402

import demo as vendor_demo  # noqa: E402
from step_runtime import StepRuntime, resample_24k  # noqa: E402


def pcm_digest(audio: np.ndarray) -> str:
    return hashlib.blake2b(np.ascontiguousarray(audio, dtype=np.float32).tobytes(),
                           digest_size=16).hexdigest()


def load_reference(path: str) -> np.ndarray:
    audio, sr = sf.read(path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return resample_24k(audio, sr).astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser(description="pinyin vendor baseline")
    ap.add_argument("--model-dir", default=os.path.join(VENDOR, "step_onnx_int8"))
    ap.add_argument("--config", default=os.path.join(PROJ, "configs", "listen_texts.json"))
    ap.add_argument("--out", default=os.path.join(PROJ, "results", "baseline.json"))
    ap.add_argument("--audio-dir", default=os.path.join(PROJ, "results", "audio"))
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)
    ref_path = os.path.join(VENDOR, cfg["reference"])
    ref = load_reference(ref_path)
    os.makedirs(args.audio_dir, exist_ok=True)

    tokenizer = vendor_demo.PinyinTokenizer(
        os.path.join(VENDOR, vendor_demo.TOKENS_TXT),
        os.path.join(VENDOR, vendor_demo.SPM_MODEL))

    t_load0 = time.perf_counter()
    rt = StepRuntime(args.model_dir, intra_op_num_threads=cfg["threads"])
    load_s = time.perf_counter() - t_load0

    report = {
        "meta": {
            "created": dt.datetime.now().isoformat(timespec="seconds"),
            "host": platform.node(),
            "vendor": VENDOR,
            "model_dir": args.model_dir,
            "reference": os.path.basename(ref_path),
            "reference_pcm_blake2b": pcm_digest(ref),
            "seed": cfg["seed"],
            "temperature": cfg["temperature"],
            "max_frames": cfg["max_frames"],
            "threads": cfg["threads"],
            "load_s": round(load_s, 3),
            "note": "vendor int8 model; host CPU baseline",
        },
        "runs": [],
    }

    # warmup
    warm_ids = tokenizer.text_to_ids("你好。")
    for _ in rt.stream(warm_ids, ref, temp=cfg["temperature"], max_frames=8, seed=cfg["seed"]):
        pass

    print(f"load {load_s:.2f}s; reference {len(ref) / 24000:.2f}s")
    for item in cfg["texts"]:
        ids = tokenizer.text_to_ids(item["text"])
        gen = rt.stream(ids, ref, temp=cfg["temperature"], max_frames=cfg["max_frames"],
                        seed=cfg["seed"])
        t0 = time.perf_counter()
        frames = [next(gen)]
        t_first = time.perf_counter() - t0
        for f in gen:
            frames.append(f)
        t_total = time.perf_counter() - t0
        audio = np.concatenate(frames)
        seconds = len(audio) / 24000
        wav_path = os.path.join(args.audio_dir, f"vendor_int8_{item['tag']}.wav")
        sf.write(wav_path, audio, 24000)
        run = {
            "tag": item["tag"], "text": item["text"], "tokens": len(ids),
            "token_ids": ids, "frames": len(frames),
            "audio_seconds": round(seconds, 4),
            "first_frame_ms": round(t_first * 1000, 1),
            "total_s": round(t_total, 4),
            "rtf": round(t_total / seconds, 4),
            "pcm_blake2b": pcm_digest(audio),
            "wav": os.path.relpath(wav_path, PROJ),
        }
        report["runs"].append(run)
        print(f"  {item['tag']:10s} tokens={len(ids):3d} frames={len(frames):3d} "
              f"audio={seconds:6.2f}s first={run['first_frame_ms']:6.1f}ms "
              f"total={run['total_s']:6.2f}s RTF={run['rtf']:.3f}")

    with open(args.out, "w") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
