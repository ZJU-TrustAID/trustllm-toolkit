from __future__ import annotations

import re
import string
from collections.abc import Iterable


def normalize_text(text: str) -> str:
    value = text.lower().strip()
    value = value.translate(str.maketrans("", "", string.punctuation))
    return re.sub(r"\s+", " ", value).strip()


def exact_match(prediction: str, references: Iterable[str]) -> float:
    pred = normalize_text(prediction)
    return float(any(pred == normalize_text(reference) for reference in references))


def contains_match(prediction: str, references: Iterable[str]) -> float:
    pred = normalize_text(prediction)
    return float(
        any(
            normalize_text(reference) in pred
            for reference in references
            if normalize_text(reference)
        )
    )
