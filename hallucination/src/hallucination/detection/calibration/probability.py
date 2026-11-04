from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

_METHODS = {"platt", "isotonic", "temperature"}


class ProbabilityCalibrator:
    """Map raw confidence/risk scores to calibrated probabilities.

    ``target`` documents what the calibrated probability means:
    ``"hallucination"`` (scores are risks) or ``"correct"`` (scores are
    confidences). The calibrator itself is direction-agnostic; it fits
    ``P(label=1 | score)`` where label 1 is the positive event named by
    ``target``.

    All fitting methods use only NumPy so the base installation stays light.
    """

    def __init__(self, method: str = "platt", *, target: str = "hallucination", epsilon: float = 1e-6) -> None:
        if method not in _METHODS:
            raise ValueError(f"method must be one of {sorted(_METHODS)}")
        if target not in {"hallucination", "correct"}:
            raise ValueError("target must be 'hallucination' or 'correct'")
        if not 0.0 < epsilon < 0.5:
            raise ValueError("epsilon must be in (0, 0.5)")
        self.method = method
        self.target = target
        self.epsilon = epsilon
        self.split: str | None = None
        self._fitted = False
        self._params: dict[str, Any] = {}

    def fit(
        self,
        scores: list[float] | np.ndarray,
        labels: list[bool] | np.ndarray,
        *,
        split: str = "validation",
    ) -> ProbabilityCalibrator:
        if split.strip().lower() == "test":
            raise ValueError("probability calibration on the test split is not allowed; fit on validation")
        scores_array, labels_array = _validate(scores, labels, self.epsilon)
        if self.method == "platt":
            self._params = self._fit_platt(scores_array, labels_array)
        elif self.method == "isotonic":
            self._params = self._fit_isotonic(scores_array, labels_array)
        else:
            self._params = self._fit_temperature(scores_array, labels_array)
        self.split = split
        self._fitted = True
        return self

    def transform(self, scores: list[float] | np.ndarray) -> list[float]:
        self._require_fitted()
        scores_array = np.asarray(list(scores), dtype=float)
        if scores_array.ndim != 1 or len(scores_array) == 0:
            raise ValueError("scores must be a non-empty one-dimensional sequence")
        if not np.all(np.isfinite(scores_array)) or not np.all((scores_array >= 0.0) & (scores_array <= 1.0)):
            raise ValueError("scores must be finite and in [0, 1]")
        if self.method == "platt":
            calibrated = _sigmoid(self._params["a"] * _logit(scores_array, self.epsilon) + self._params["b"])
        elif self.method == "isotonic":
            calibrated = np.interp(scores_array, self._params["x"], self._params["y"])
        else:
            calibrated = _sigmoid(_logit(scores_array, self.epsilon) / self._params["temperature"])
        return [float(value) for value in np.clip(calibrated, 0.0, 1.0)]

    def fit_transform(self, scores: list[float] | np.ndarray, labels: list[bool] | np.ndarray, *, split: str = "validation") -> list[float]:
        self.fit(scores, labels, split=split)
        return self.transform(scores)

    def save(self, path: str | Path) -> None:
        self._require_fitted()
        Path(path).write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ProbabilityCalibrator:
        if payload.get("type") not in {None, "probability"}:
            raise ValueError("not a probability calibrator payload")
        if "params" not in payload:
            raise ValueError("probability calibrator payload requires 'params'")
        calibrator = cls(
            payload["method"],
            target=payload.get("target", "hallucination"),
            epsilon=float(payload.get("epsilon", 1e-6)),
        )
        calibrator.split = payload.get("split")
        calibrator._params = payload["params"]
        calibrator._fitted = True
        return calibrator

    @classmethod
    def load(cls, path: str | Path) -> ProbabilityCalibrator:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "probability",
            "method": self.method,
            "target": self.target,
            "epsilon": self.epsilon,
            "split": self.split,
            "params": self._params,
        }

    def _fit_platt(self, scores: np.ndarray, labels: np.ndarray) -> dict[str, float]:
        x = _logit(scores, self.epsilon)
        weights = np.array([1.0, 0.0])
        for _ in range(100):
            z = weights[0] * x + weights[1]
            probabilities = _sigmoid(z)
            variance = probabilities * (1.0 - probabilities) + 1e-12
            gradient = np.array(
                [
                    float(np.sum((probabilities - labels) * x)),
                    float(np.sum(probabilities - labels)),
                ]
            )
            hessian = np.array(
                [
                    [float(np.sum(variance * x * x)) + 1e-9, float(np.sum(variance * x))],
                    [float(np.sum(variance * x)), float(np.sum(variance)) + 1e-9],
                ]
            )
            try:
                step = np.linalg.solve(hessian, gradient)
            except np.linalg.LinAlgError:
                break
            step = np.clip(step, -5.0, 5.0)
            weights = weights - step
            weights[0] = float(np.clip(weights[0], -50.0, 50.0))
            if float(np.max(np.abs(step))) < 1e-8:
                break
        return {"a": float(weights[0]), "b": float(weights[1])}

    def _fit_isotonic(self, scores: np.ndarray, labels: np.ndarray) -> dict[str, list[float]]:
        unique, inverse = np.unique(scores, return_inverse=True)
        sums = np.zeros(len(unique), dtype=float)
        counts = np.zeros(len(unique), dtype=float)
        for index, label in zip(inverse, labels, strict=False):
            sums[index] += label
            counts[index] += 1.0
        means = sums / counts
        fitted = _pava(means, counts)
        return {"x": [float(value) for value in unique], "y": [float(value) for value in fitted]}

    def _fit_temperature(self, scores: np.ndarray, labels: np.ndarray) -> dict[str, float]:
        logits = _logit(scores, self.epsilon)
        best_temperature = 1.0
        best_nll = float("inf")
        for temperature in np.linspace(0.05, 10.0, 200):
            probabilities = _sigmoid(logits / temperature)
            nll = _nll(probabilities, labels)
            if nll < best_nll:
                best_nll = nll
                best_temperature = float(temperature)
        return {"temperature": best_temperature}

    def _require_fitted(self) -> None:
        if not self._fitted:
            raise ValueError("ProbabilityCalibrator must be fitted before transform")


def _pava(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Pool-adjacent-violators for a weighted non-decreasing fit.

    ``values`` contains one entry per unique score and ``weights`` its sample
    count. The return value has one fitted probability per unique score, which
    is what :meth:`ProbabilityCalibrator.transform` passes to ``np.interp``.
    Expanding the result to the original sample count here would make ``x`` and
    ``y`` different lengths whenever scores repeat.
    """

    blocks: list[list[float]] = []
    for value, weight in zip(values, weights, strict=False):
        blocks.append([float(value) * float(weight), float(weight), 1.0])
        while len(blocks) > 1 and blocks[-2][0] / blocks[-2][1] > blocks[-1][0] / blocks[-1][1]:
            previous = blocks.pop()
            blocks[-1][0] += previous[0]
            blocks[-1][1] += previous[1]
            blocks[-1][2] += previous[2]
    fitted: list[float] = []
    for total, weight, unique_count in blocks:
        fitted.extend([total / weight] * int(round(unique_count)))
    return np.asarray(fitted, dtype=float)


def _validate(scores: list[float] | np.ndarray, labels: list[bool] | np.ndarray, epsilon: float) -> tuple[np.ndarray, np.ndarray]:
    scores_array = np.asarray(list(scores), dtype=float)
    labels_array = np.asarray(list(labels), dtype=float)
    if scores_array.ndim != 1 or labels_array.ndim != 1:
        raise ValueError("scores and labels must be one-dimensional")
    if len(scores_array) != len(labels_array):
        raise ValueError("scores and labels must have the same length")
    if len(scores_array) < 2:
        raise ValueError("calibration requires at least two observations")
    if not np.all(np.isfinite(scores_array)):
        raise ValueError("scores must be finite")
    if not np.all((scores_array >= 0.0) & (scores_array <= 1.0)):
        raise ValueError("scores must be in [0, 1]")
    if not np.all(np.isin(labels_array, [0.0, 1.0])):
        raise ValueError("labels must be binary")
    if len(np.unique(labels_array)) < 2:
        raise ValueError("calibration requires both positive and negative labels")
    return np.clip(scores_array, epsilon, 1.0 - epsilon), labels_array


def _logit(scores: np.ndarray, epsilon: float) -> np.ndarray:
    clipped = np.clip(scores, epsilon, 1.0 - epsilon)
    return np.log(clipped / (1.0 - clipped))


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(values, -50.0, 50.0)))


def _nll(probabilities: np.ndarray, labels: np.ndarray) -> float:
    clipped = np.clip(probabilities, 1e-12, 1.0 - 1e-12)
    return float(-np.mean(labels * np.log(clipped) + (1.0 - labels) * np.log(1.0 - clipped)))
