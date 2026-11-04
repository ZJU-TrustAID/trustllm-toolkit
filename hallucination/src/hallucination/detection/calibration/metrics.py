from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def binary_metrics(predictions: Sequence[bool], labels: Sequence[bool]) -> dict[str, float]:
    if len(predictions) != len(labels):
        raise ValueError("predictions and labels must have the same length")
    tp = sum(pred and label for pred, label in zip(predictions, labels, strict=False))
    fp = sum(pred and not label for pred, label in zip(predictions, labels, strict=False))
    tn = sum((not pred) and (not label) for pred, label in zip(predictions, labels, strict=False))
    fn = sum((not pred) and label for pred, label in zip(predictions, labels, strict=False))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    accuracy = (tp + tn) / len(labels) if labels else 0.0
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}


def expected_calibration_error(
    scores: Sequence[float],
    labels: Sequence[bool],
    *,
    n_bins: int = 10,
    strategy: str = "uniform",
) -> float:
    """Expected Calibration Error for confidence scores and binary outcomes."""

    return _bin_statistics(scores, labels, n_bins=n_bins, strategy=strategy)["ece"]


def maximum_calibration_error(
    scores: Sequence[float],
    labels: Sequence[bool],
    *,
    n_bins: int = 10,
    strategy: str = "uniform",
) -> float:
    """Maximum Calibration Error across bins."""

    return _bin_statistics(scores, labels, n_bins=n_bins, strategy=strategy)["mce"]


def brier_score(scores: Sequence[float], labels: Sequence[bool]) -> float:
    scores_array, labels_array = _arrays(scores, labels)
    return float(np.mean((scores_array - labels_array) ** 2))


def negative_log_likelihood(scores: Sequence[float], labels: Sequence[bool], *, epsilon: float = 1e-12) -> float:
    scores_array, labels_array = _arrays(scores, labels)
    clipped = np.clip(scores_array, epsilon, 1.0 - epsilon)
    return float(-np.mean(labels_array * np.log(clipped) + (1.0 - labels_array) * np.log(1.0 - clipped)))


def reliability_table(
    scores: Sequence[float],
    labels: Sequence[bool],
    *,
    n_bins: int = 10,
    strategy: str = "uniform",
) -> list[dict[str, float]]:
    """Return per-bin count, mean confidence, accuracy, and gap."""

    stats = _bin_statistics(scores, labels, n_bins=n_bins, strategy=strategy, with_table=True)
    return stats["table"]


def calibration_metrics(
    scores: Sequence[float],
    labels: Sequence[bool],
    *,
    n_bins: int = 10,
    strategy: str = "uniform",
) -> dict[str, float]:
    stats = _bin_statistics(scores, labels, n_bins=n_bins, strategy=strategy)
    return {
        "count": float(stats["count"]),
        "ece": float(stats["ece"]),
        "mce": float(stats["mce"]),
        "brier": brier_score(scores, labels),
        "nll": negative_log_likelihood(scores, labels),
        "average_confidence": float(stats["average_confidence"]),
        "average_accuracy": float(stats["average_accuracy"]),
        "calibration_gap": float(abs(stats["average_confidence"] - stats["average_accuracy"])),
    }


def _bin_statistics(
    scores: Sequence[float],
    labels: Sequence[bool],
    *,
    n_bins: int,
    strategy: str,
    with_table: bool = False,
) -> dict[str, object]:
    if n_bins < 1:
        raise ValueError("n_bins must be at least 1")
    if strategy not in {"uniform", "quantile"}:
        raise ValueError("strategy must be 'uniform' or 'quantile'")
    scores_array, labels_array = _arrays(scores, labels)
    size = len(scores_array)

    if strategy == "uniform":
        edges = np.linspace(0.0, 1.0, n_bins + 1)
    else:
        quantiles = np.linspace(0.0, 1.0, n_bins + 1)
        edges = np.unique(np.quantile(scores_array, quantiles))
        if len(edges) < 2:
            edges = np.array([0.0, 1.0])
    bin_indices = np.clip(np.digitize(scores_array, edges[1:-1], right=False), 0, len(edges) - 2)

    ece = 0.0
    mce = 0.0
    table: list[dict[str, float]] = []
    for bin_index in range(len(edges) - 1):
        mask = bin_indices == bin_index
        count = int(mask.sum())
        if count == 0:
            table.append(
                {
                    "bin": float(bin_index),
                    "lower": float(edges[bin_index]),
                    "upper": float(edges[bin_index + 1]),
                    "count": 0.0,
                    "mean_confidence": 0.0,
                    "accuracy": 0.0,
                    "gap": 0.0,
                }
            )
            continue
        confidence = float(scores_array[mask].mean())
        accuracy = float(labels_array[mask].mean())
        gap = abs(confidence - accuracy)
        ece += gap * count / size
        mce = max(mce, gap)
        table.append(
            {
                "bin": float(bin_index),
                "lower": float(edges[bin_index]),
                "upper": float(edges[bin_index + 1]),
                "count": float(count),
                "mean_confidence": confidence,
                "accuracy": accuracy,
                "gap": gap,
            }
        )

    result: dict[str, object] = {
        "count": size,
        "ece": float(ece),
        "mce": float(mce),
        "average_confidence": float(scores_array.mean()),
        "average_accuracy": float(labels_array.mean()),
    }
    if with_table:
        result["table"] = table
    return result


def _arrays(scores: Sequence[float], labels: Sequence[bool]) -> tuple[np.ndarray, np.ndarray]:
    scores_array = np.asarray(list(scores), dtype=float)
    labels_array = np.asarray(list(labels), dtype=float)
    if scores_array.ndim != 1 or labels_array.ndim != 1:
        raise ValueError("scores and labels must be one-dimensional")
    if len(scores_array) != len(labels_array):
        raise ValueError("scores and labels must have the same length")
    if len(scores_array) == 0:
        raise ValueError("scores and labels must not be empty")
    if not np.all(np.isfinite(scores_array)):
        raise ValueError("scores must be finite")
    if not np.all((scores_array >= 0.0) & (scores_array <= 1.0)):
        raise ValueError("scores must be in [0, 1]")
    if not np.all(np.isin(labels_array, [0.0, 1.0])):
        raise ValueError("labels must be binary (True/False or 1/0)")
    return scores_array, labels_array
