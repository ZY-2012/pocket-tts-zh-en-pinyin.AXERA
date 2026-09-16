#!/usr/bin/env python3
"""Shared pinyin text frontend for host-side scripts.

Wraps the vendor PinyinTokenizer (tokens.txt + pypinyin TONE3 + SentencePiece
BPE for English/punctuation) with caching, so every script tokenizes exactly
like the vendor demo.py.

Usage:
    from pinyin_frontend import text2ids
    ids = text2ids("你好，世界。")
"""
from __future__ import annotations

import os
import re
import sys

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_ROOT = os.environ.get("PINYIN_TTS_ROOT", "/data/shared/huyuan/TTS/pocket-tts-zh-en-pinyin")

sys.path.insert(0, MODEL_ROOT)

_PUNCT_MAP = {
    "。": ".", "、": ",", "｡": ".", "“": '"', "”": '"', "‘": "'", "’": "'",
    "…": "...", "—": "-", "–": "-", "〜": "~", "～": "~",
    "【": "[", "】": "]", "《": "<", "》": ">",
    "「": '"', "」": '"', "『": '"', "』": '"',
}

_TOKENIZER = None


def normalize_text(text: str) -> str:
    """Vendor-compatible normalization: full-width -> half-width, hyphen -> space."""
    text = text.translate(str.maketrans(_PUNCT_MAP))
    text = "".join(chr(ord(c) - 0xFEE0) if 0xFF01 <= ord(c) <= 0xFF5E else c for c in text)
    text = text.lower()
    return re.sub(r"(?<=[a-zA-Z])-(?=[a-zA-Z])", " ", text)


def get_tokenizer():
    global _TOKENIZER
    if _TOKENIZER is None:
        import demo as vendor_demo  # vendor PinyinTokenizer + vocab tables

        _TOKENIZER = vendor_demo.PinyinTokenizer(
            os.path.join(MODEL_ROOT, vendor_demo.TOKENS_TXT),
            os.path.join(MODEL_ROOT, vendor_demo.SPM_MODEL))
    return _TOKENIZER


def text2ids(text: str) -> list[int]:
    return get_tokenizer().text_to_ids(text)


def tokenize_surfaces(text: str) -> list[str]:
    return get_tokenizer().tokenize(text)
