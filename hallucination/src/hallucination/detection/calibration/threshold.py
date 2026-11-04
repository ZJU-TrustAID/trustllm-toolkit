from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

_OBJECTIVES = {"f1", "fbeta", "balanced_accuracy", "cost"}


class ThresholdCalibrator:
    """Select a decision threshold on validation data only.

    ``objective`` may be ``f1``, ``fbeta``, ``balanced_accuracy`` or ``cost``.
    Thresholds are searched over the observed risk values and their midpoints
    (not a fixed 0.01 grid), which makes the result deterministic and exact for
    the selected metric.
    """

    def __init__(
        self,
        metric: str = "f1",
        *,
        beta: float = 1.0,
        false_positive_cost: float = 1.0,
        false_negative_cost: float = 1.0,
        objective: str | None = None,
    ) -> None:
        selected = objective or metric
        if selected not in _OBJECTIVES:
            raise ValueError(f"objective must be one of {sorted(_OBJECTIVES)}")
        if beta <= 0:
            raise ValueError("beta must be positive")
        self.objective = selected
        self.metric = selected
        self.beta = beta
        self.false_positive_cost = false_positive_cost
        self.false_negative_cost = false_negative_cost
        self.threshold: float = 0.5
        self.score: float | None = None
        self.split: str | None = None
        self.curve: list[dict[str, float]] = []
        self._fitted = False

    def fit(
        self,
        risks: list[float] | np.ndarray,
        labels: list[bool] | np.ndarray,
        *,
        split: str = "validation",
    ) -> ThresholdCalibrator:
        if split.strip().lower() == "test":
            raise ValueError("threshold fitting on the test split is not allowed; fit on validation")
        risks_array = np.asarray(list(risks), dtype=float)
        labels_array = np.asarray(list(labels), dtype=float)
        _validate(risks_array, labels_array)
        threshold, score, curve = self._select(risks_array, labels_array)
        self.threshold = threshold
        self.score = score
        self.curve = curve
        self.split = split
        self._fitted = True
        return self

    def predict(self, risks: list[float] | np.ndarray) -> list[bool]:
        self._require_fitted()
        return [bool(risk >= self.threshold) for risk in risks]

    def bootstrap(self, risks: list[float] | np.ndarray, labels: list[bool] | np.ndarray, *, n_boot: int = 200, seed: int = 0) -> tuple[float, float]:
        """Percentile bootstrap interval for the selected threshold."""

        risks_array = np.asarray(list(risks), dtype=float)
        labels_array = np.asarray(list(labels), dtype=float)
        _validate(risks_array, labels_array)
        if n_boot < 1:
            raise ValueError("n_boot must be positive")
        rng = np.random.default_rng(seed)
        thresholds = []
        for _ in range(n_boot):
            indices = rng.integers(0, len(risks_array), size=len(risks_array))
            sampled_labels = labels_array[indices]
            if len(np.unique(sampled_labels)) < 2:
                continue
            threshold, _, _ = self._select(risks_array[indices], sampled_labels)
            thresholds.append(threshold)
        if not thresholds:
            raise ValueError("bootstrap could not produce a two-class sample")
        low, high = np.percentile(thresholds, [2.5, 97.5])
        return float(low), float(high)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ThresholdCalibrator:
        if payload.get("type") not in {None, "threshold"}:
            raise ValueError("not a threshold calibrator payload")
        calibrator = cls(
            payload.get("objective", payload.get("metric", "f1")),
            beta=float(payload.get("beta", 1.0)),
            false_positive_cost=float(payload.get("false_positive_cost", 1.0)),
            false_negative_cost=float(payload.get("false_negative_cost", 1.0)),
        )
        calibrator.threshold = float(payload["threshold"])
        calibrator.score = payload.get("score")
        calibrator.split = payload.get("split")
        calibrator._fitted = True
        return calibrator

    @classmethod
    def load(cls, path: str | Path) -> ThresholdCalibrator:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "threshold",
            "objective": self.objective,
            "beta": self.beta,
            "false_positive_cost": self.false_positive_cost,
            "false_negative_cost": self.false_negative_cost,
            "threshold": self.threshold,
            "score": self.score,
            "split": self.split,
        }

    def _select(self, risks: np.ndarray, labels: np.ndarray) -> tuple[float, float, list[dict[str, float]]]:
        candidates = _candidates(risks)
        best_threshold = candidates[0]
        best_score = float("-inf") if self.objective != "cost" else float("inf")
        curve: list[dict[str, float]] = []
        for threshold in candidates:
            predictions = risks >= threshold
            score = _score(predictions, labels, self.objective, self.beta, self.false_positive_cost, self.false_negative_cost)
            curve.append({"threshold": float(threshold), "score": float(score)})
            if self.objective == "cost":
                if score < best_score:
                    best_score = score
                    best_threshold = threshold
            elif score > best_score:
                best_score = score
                best_threshold = threshold
        return float(best_threshold), float(best_score), curve

    def _require_fitted(self) -> None:
        if not self._fitted:
            raise ValueError("ThresholdCalibrator must be fitted before prediction")


def _candidates(risks: np.ndarray) -> list[float]:
    unique = np.unique(risks)
    values = {0.0, 1.0, *[float(value) for value in unique]}
    ordered = sorted(values)
    for left, right in zip(ordered, ordered[1:], strict=False):
        values.add((left + right) / 2.0)
    return sorted(values)


def _score(
    predictions: np.ndarray,
    labels: np.ndarray,
    objective: str,
    beta: float,
    false_positive_cost: float,
    false_negative_cost: float,
) -> float:
    tp = float(np.sum(predictions & (labels == 1)))
    fp = float(np.sum(predictions & (labels == 0)))
    fn = float(np.sum((~predictions) & (labels == 1)))
    tn = float(np.sum((~predictions) & (labels == 0)))
    if objective == "cost":
        return fp * false_positive_cost + fn * false_negative_cost
    if objective == "balanced_accuracy":
        sensitivity = tp / (tp + fn) if tp + fn else 0.0
        specificity = tn / (tn + fp) if tn + fp else 0.0
        return (sensitivity + specificity) / 2.0
    beta_squared = beta * beta
    denominator = (1.0 + beta_squared) * tp + beta_squared * fn + fp
    return (1.0 + beta_squared) * tp / denominator if denominator else 0.0


def _validate(risks: np.ndarray, labels: np.ndarray) -> None:
    if len(risks) != len(labels):
        raise ValueError("risks and labels must have the same length")
    if len(risks) == 0:
        raise ValueError("risks and labels must not be empty")
    if not np.all(np.isfinite(risks)):
        raise ValueError("risks must be finite")
    if not np.all((risks >= 0.0) & (risks <= 1.0)):
        raise ValueError("risks must be in [0, 1]")
    if not np.all(np.isin(labels, [0.0, 1.0])):
        raise ValueError("labels must be binary")
    if len(np.unique(labels)) < 2:
        raise ValueError("threshold fitting requires both positive and negative labels")
