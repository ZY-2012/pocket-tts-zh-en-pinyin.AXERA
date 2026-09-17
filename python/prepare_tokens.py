#!/usr/bin/env python3
"""Host-side tokenizer for the pinyin C++ runtime: text -> request file.

Uses the vendor PinyinTokenizer (tokens.txt + pypinyin TONE3 + BPE), identical
to the board runtime and demo.py.

Usage:
    python python/prepare_tokens.py --text "你好，世界。" --out request.tokens
"""
from __future__ import annotations

import argparse
import os
import sys

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VENDOR = os.environ.get("PINYIN_TTS_ROOT", "/data/shared/huyuan/TTS/pocket-tts-zh-en-pinyin")
sys.path.insert(0, VENDOR)
sys.path.insert(0, os.path.join(PROJ, "python"))

import demo as vendor_demo  # noqa: E402
from pinyin_frontend import get_tokenizer  # noqa: E402

MAX_TOKENS = 48


def chunk_text(tok, text: str) -> list[str]:
    if len(tok.text_to_ids(text)) <= MAX_TOKENS:
        return [text]
    pieces, buf = [], []
    for ch in text:
        buf.append(ch)
        if ch in "。！？!?.;；\n":
            pieces.append("".join(buf).strip()); buf = []
    if buf:
        pieces.append("".join(buf).strip())
    chunks, cur = [], ""
    for piece in pieces:
        if not piece:
            continue
        n = len(tok.text_to_ids(cur + piece))
        if cur and n > MAX_TOKENS:
            chunks.append(cur); cur = piece
        else:
            cur += piece
    if cur:
        chunks.append(cur)
    return chunks or [text]


def main() -> None:
    ap = argparse.ArgumentParser(description="text -> C++ request tokens (pinyin)")
    ap.add_argument("--text", default=None)
    ap.add_argument("--text-file", default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    text = args.text
    if args.text_file:
        with open(args.text_file) as f:
            text = f.read().strip()
    if not text:
        raise SystemExit("no text")

    tok = get_tokenizer()
    chunks = chunk_text(tok, text)
    with open(args.out, "w") as f:
        for chunk in chunks:
            ids = tok.text_to_ids(chunk)
            f.write(",".join(str(i) for i in ids) + "\n")
            print(f"chunk: {len(ids)} tokens | {chunk[:40]}")
    print(f"wrote {args.out} ({len(chunks)} chunk(s))")


if __name__ == "__main__":
    main()
