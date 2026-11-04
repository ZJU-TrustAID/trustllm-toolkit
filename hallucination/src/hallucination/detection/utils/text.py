from __future__ import annotations

import re
import string

_CHINESE_PUNCTUATION = "，。！？；：、“”‘’（）《》【】"


def normalize_answer(text: str, *, lowercase: bool = True, strip_punctuation: bool = True) -> str:
    if lowercase:
        text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    if strip_punctuation:
        table = str.maketrans("", "", string.punctuation + _CHINESE_PUNCTUATION)
        text = text.translate(table)
    return text.strip()


def split_sentences(text: str) -> list[str]:
    pieces = re.split(r"(?<=[。！？.!?])\s*", text.strip())
    return [piece.strip() for piece in pieces if piece.strip()]
