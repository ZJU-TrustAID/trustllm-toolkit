from __future__ import annotations

from collections.abc import Sequence


def binary_risk_metrics(risks: Sequence[float], labels: Sequence[bool], *, threshold: float = 0.5) -> dict[str, float | None]:
    """Compute deterministic binary metrics where `True` labels are hallucinations."""

    if len(risks) != len(labels):
        raise ValueError("risks and labels must have the same length")
    if not risks:
        return {"count": 0.0, "precision": None, "recall": None, "f1": None, "roc_auc": None, "pr_auc": None}
    predictions = [risk >= threshold for risk in risks]
    tp = sum(prediction and label for prediction, label in zip(predictions, labels, strict=False))
    fp = sum(prediction and not label for prediction, label in zip(predictions, labels, strict=False))
    fn = sum(not prediction and label for prediction, label in zip(predictions, labels, strict=False))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "count": float(len(risks)),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": _roc_auc(risks, labels),
        "pr_auc": _average_precision(risks, labels),
    }


def _roc_auc(risks: Sequence[float], labels: Sequence[bool]) -> float | None:
    positives = sum(labels)
    negatives = len(labels) - positives
    if not positives or not negatives:
        return None
    ordered = sorted(enumerate(risks), key=lambda item: item[1])
    ranks = [0.0] * len(risks)
    position = 0
    while position < len(ordered):
        end = position + 1
        while end < len(ordered) and ordered[end][1] == ordered[position][1]:
            end += 1
        average_rank = (position + 1 + end) / 2.0
        for original_index, _ in ordered[position:end]:
            ranks[original_index] = average_rank
        position = end
    positive_rank_sum = sum(rank for rank, label in zip(ranks, labels, strict=False) if label)
    return (positive_rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def _average_precision(risks: Sequence[float], labels: Sequence[bool]) -> float | None:
    positives = sum(labels)
    if not positives:
        return None
    ordered = sorted(zip(risks, labels, strict=False), key=lambda item: item[0], reverse=True)
    true_positives = 0
    precision_sum = 0.0
    for total, (_, label) in enumerate(ordered, start=1):
        if label:
            true_positives += 1
            precision_sum += true_positives / total
    return precision_sum / positives
