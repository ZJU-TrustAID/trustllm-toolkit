from __future__ import annotations

from collections.abc import Iterable

import numpy as np


def softmax(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    values = values - np.max(values)
    exp = np.exp(values)
    total = exp.sum()
    if not np.isfinite(total) or total <= 0:
        return np.ones_like(exp) / exp.size
    return exp / total


def log_softmax(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    values = values - np.max(values)
    return values - np.log(np.exp(values).sum())


def top_p_filter(logits: np.ndarray, top_p: float) -> np.ndarray:
    if top_p >= 1.0:
        return np.asarray(logits, dtype=np.float64)
    values = np.asarray(logits, dtype=np.float64).copy()
    probs = softmax(values)
    order = np.argsort(probs)[::-1]
    cumulative = np.cumsum(probs[order])
    keep = cumulative <= top_p
    if keep.size:
        keep[0] = True
    if not keep.any():
        keep[0] = True
    kept_indices = order[keep]
    mask = np.full(values.shape, -np.inf)
    mask[kept_indices] = values[kept_indices]
    return mask


def apply_repetition_penalty(
    logits: np.ndarray, generated_ids: Iterable[int], penalty: float
) -> np.ndarray:
    if penalty == 1.0:
        return np.asarray(logits, dtype=np.float64)
    values = np.asarray(logits, dtype=np.float64).copy()
    for token_id in {int(item) for item in generated_ids}:
        if 0 <= token_id < values.size:
            if values[token_id] > 0:
                values[token_id] /= penalty
            else:
                values[token_id] *= penalty
    return values


def sample_token(
    logits: np.ndarray,
    *,
    do_sample: bool,
    temperature: float = 1.0,
    top_p: float = 1.0,
    top_k: int = 0,
    repetition_penalty: float = 1.0,
    generated_ids: Iterable[int] = (),
    rng: np.random.Generator | None = None,
) -> int:
    values = apply_repetition_penalty(
        np.asarray(logits, dtype=np.float64), generated_ids, repetition_penalty
    )
    if not np.all(np.isfinite(values)):
        values = np.nan_to_num(values, nan=-1e9, posinf=1e9, neginf=-1e9)
    if not do_sample:
        return int(np.argmax(values))
    if temperature != 1.0:
        values = values / max(float(temperature), 1e-8)
    if top_k and top_k > 0 and top_k < values.size:
        threshold = np.partition(values, -top_k)[-top_k]
        values = np.where(values < threshold, -np.inf, values)
    values = top_p_filter(values, top_p)
    probs = softmax(values)
    generator = rng or np.random.default_rng()
    return int(generator.choice(values.size, p=probs))
