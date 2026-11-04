from __future__ import annotations

import math
from collections.abc import Iterable

import numpy as np


def safe_mean(values: Iterable[float], default: float = 0.0) -> float:
    values = list(values)
    if not values:
        return default
    return float(sum(values) / len(values))


def softmax(logits: Iterable[float]) -> list[float]:
    arr = np.asarray(list(logits), dtype=float)
    shifted = arr - np.max(arr)
    exp = np.exp(shifted)
    return (exp / exp.sum()).tolist()


def entropy_from_probs(probs: Iterable[float], base: float = math.e) -> float:
    total = 0.0
    for prob in probs:
        if prob > 0:
            total -= prob * math.log(prob, base)
    return float(total)


def cosine(a: Iterable[float], b: Iterable[float]) -> float:
    va = np.asarray(list(a), dtype=float)
    vb = np.asarray(list(b), dtype=float)
    denom = np.linalg.norm(va) * np.linalg.norm(vb)
    if denom == 0:
        return 0.0
    return float(np.dot(va, vb) / denom)

