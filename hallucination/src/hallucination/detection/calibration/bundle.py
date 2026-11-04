from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np

from hallucination.detection.calibration.probability import ProbabilityCalibrator
from hallucination.detection.calibration.threshold import ThresholdCalibrator
from hallucination.detection.core.results import DetectionResult, ScoreResult

ScoreKind = Literal["risk", "confidence"]
Granularity = Literal["response", "sentence", "claim", "token"]


@dataclass
class CalibrationBundle:
    """Calibration artifacts for one score vector.

    This is the bridge between the standalone calibrators and detection.  A
    bundle is deliberately keyed by *method name plus granularity*: applying a
    semantic-entropy calibrator to a cosine score, or a response calibrator to
    sentence scores, is a configuration error rather than a silent average.

    ``score_kind`` describes the raw ``ScoreResult.score`` orientation, not the
    field that is calibrated.  Detection always calibrates ``ScoreResult.risk``;
    ``score_kind="confidence"`` means the underlying method reports confidence
    (e.g. MTP/LNTP) and the calibrator was fit on those confidence values.
    """

    score_name: str = "*"
    granularity: Granularity | None = None
    score_kind: ScoreKind = "risk"
    probability: ProbabilityCalibrator | None = None
    threshold: ThresholdCalibrator | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.score_kind not in {"risk", "confidence"}:
            raise ValueError("score_kind must be 'risk' or 'confidence'")
        if self.granularity is not None and self.granularity not in {"response", "sentence", "claim", "token"}:
            raise ValueError("granularity must be response, sentence, claim, token, or None")
        if self.probability is not None and not isinstance(self.probability, ProbabilityCalibrator):
            raise TypeError("probability must be a ProbabilityCalibrator")
        if self.threshold is not None and not isinstance(self.threshold, ThresholdCalibrator):
            raise TypeError("threshold must be a ThresholdCalibrator")

    @property
    def split(self) -> str | None:
        if self.probability is not None:
            return self.probability.split
        if self.threshold is not None:
            return self.threshold.split
        return None

    @property
    def decision_threshold(self) -> float:
        if self.threshold is not None:
            return float(self.threshold.threshold)
        return 0.5

    def applies_to(self, method: str, level: str | None = None) -> bool:
        if self.score_name not in {"*", method}:
            return False
        return not (
            self.granularity is not None and level is not None and self.granularity != level
        )

    def calibrate_risk(self, risk: float) -> float:
        """Return P(hallucination) for one raw risk value.

        If no probability calibrator is present this is the identity mapping;
        the bundle may still carry a decision threshold.  Calibrators are always
        stored in risk orientation: ``target="hallucination"`` is fit on risks,
        and ``target="correct"`` is fit on confidences and inverted here.
        """

        value = float(risk)
        if not 0.0 <= value <= 1.0:
            raise ValueError("risk must be in [0, 1]")
        if self.probability is None:
            return value
        if self.probability.target == "hallucination":
            calibrated = self.probability.transform([value])[0]
        else:
            calibrated = 1.0 - self.probability.transform([1.0 - value])[0]
        return float(min(1.0, max(0.0, calibrated)))

    def apply_to_score(self, score: ScoreResult) -> bool:
        if not self.applies_to(score.method, score.level):
            return False
        calibrated_risk = self.calibrate_risk(score.risk)
        score.calibrated_risk = calibrated_risk
        score.calibrated_score = 1.0 - calibrated_risk
        score.calibration = self.summary()
        return True

    def apply_result(
        self,
        result: DetectionResult,
        *,
        default_threshold: float = 0.5,
        aggregate: bool = True,
    ) -> list[str]:
        """Apply the bundle in place and return the matched method names."""

        matched: list[str] = []
        for score in result.method_scores:
            if self.apply_to_score(score):
                matched.append(score.method)
        for unit in result.units:
            for score in unit.method_scores:
                if self.apply_to_score(score):
                    matched.append(score.method)
        if not matched:
            return []
        if aggregate:
            threshold = self.decision_threshold if self.threshold is not None else default_threshold
            _aggregate_calibrated_result(result, default_threshold=default_threshold, threshold=threshold)
            result.calibration = {
                **self.summary(),
                "applied_to": sorted(set(matched)),
                "uncalibrated_methods": _uncalibrated_methods(result),
            }
        return matched

    def summary(self) -> dict[str, Any]:
        return {
            "score_name": self.score_name,
            "granularity": self.granularity,
            "score_kind": self.score_kind,
            "probability_method": self.probability.method if self.probability else None,
            "probability_target": self.probability.target if self.probability else None,
            "threshold": self.decision_threshold if self.threshold is not None else None,
            "fit_split": self.split,
            "metadata": dict(self.metadata),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "calibration_bundle",
            "score_name": self.score_name,
            "granularity": self.granularity,
            "score_kind": self.score_kind,
            "probability": self.probability.to_dict() if self.probability else None,
            "threshold": self.threshold.to_dict() if self.threshold else None,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> CalibrationBundle:
        if payload.get("type") not in {None, "calibration_bundle"}:
            raise ValueError("not a calibration bundle payload")
        return cls(
            score_name=str(payload.get("score_name", "*")),
            granularity=payload.get("granularity"),
            score_kind=payload.get("score_kind", "risk"),
            probability=ProbabilityCalibrator.from_dict(payload["probability"]) if payload.get("probability") else None,
            threshold=ThresholdCalibrator.from_dict(payload["threshold"]) if payload.get("threshold") else None,
            metadata=dict(payload.get("metadata") or {}),
        )

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> CalibrationBundle:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    @classmethod
    def fit(
        cls,
        scores: Iterable[float],
        labels: Iterable[bool],
        *,
        score_name: str = "*",
        granularity: Granularity | None = None,
        score_kind: ScoreKind = "risk",
        probability_method: str | None = None,
        threshold_metric: str | None = None,
        target: str = "hallucination",
        split: str = "validation",
        metadata: dict[str, Any] | None = None,
    ) -> CalibrationBundle:
        """Fit a probability calibrator and/or a decision threshold.

        ``scores`` are the raw values of ``ScoreResult.score``; ``labels`` are
        hallucination labels (True means hallucinated).  If both calibrators are
        requested, the threshold is selected on the calibrated risks rather than
        on the raw risks.
        """

        if split.strip().lower() == "test":
            raise ValueError("calibration on the test split is not allowed; fit on validation")
        if target not in {"hallucination", "correct"}:
            raise ValueError("target must be 'hallucination' or 'correct'")
        if probability_method is None and threshold_metric is None:
            threshold_metric = "f1"

        scores_array = np.asarray(list(scores), dtype=float)
        labels_array = np.asarray(list(labels), dtype=float)
        if len(scores_array) != len(labels_array):
            raise ValueError("scores and labels must have the same length")
        if not np.all((scores_array >= 0.0) & (scores_array <= 1.0)):
            raise ValueError("scores must be in [0, 1]")
        if not np.all(np.isin(labels_array, [0.0, 1.0])):
            raise ValueError("labels must be binary")

        risks = scores_array if score_kind == "risk" else 1.0 - scores_array
        probability: ProbabilityCalibrator | None = None
        if probability_method is not None:
            if target == "hallucination":
                probability = ProbabilityCalibrator(probability_method, target="hallucination").fit(
                    risks, labels_array, split=split
                )
            else:
                probability = ProbabilityCalibrator(probability_method, target="correct").fit(
                    1.0 - risks, 1.0 - labels_array, split=split
                )

        threshold: ThresholdCalibrator | None = None
        if threshold_metric is not None:
            decision_risks = risks
            if probability is not None:
                probe = cls(
                    score_name=score_name,
                    granularity=granularity,
                    score_kind=score_kind,
                    probability=probability,
                )
                decision_risks = np.asarray([probe.calibrate_risk(value) for value in risks], dtype=float)
            threshold = ThresholdCalibrator(threshold_metric).fit(decision_risks, labels_array, split=split)

        bundle_metadata: dict[str, Any] = {
            "n": int(len(scores_array)),
            "positive_rate": float(np.mean(labels_array)),
            "split": split,
        }
        if metadata:
            bundle_metadata.update(metadata)
        return cls(
            score_name=score_name,
            granularity=granularity,
            score_kind=score_kind,
            probability=probability,
            threshold=threshold,
            metadata=bundle_metadata,
        )


@dataclass
class CalibrationRegistry:
    """A collection of per-method calibration bundles."""

    bundles: list[CalibrationBundle] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.bundles = [bundle for bundle in self.bundles if isinstance(bundle, CalibrationBundle)]

    def add(self, bundle: CalibrationBundle) -> CalibrationRegistry:
        self.bundles.append(bundle)
        return self

    def for_score(self, method: str, level: str | None = None) -> CalibrationBundle | None:
        matches = [bundle for bundle in self.bundles if bundle.applies_to(method, level)]
        if not matches:
            return None
        # Prefer an exact method match over the wildcard, and an exact
        # granularity over a bundle that leaves granularity open.
        return max(matches, key=lambda bundle: (bundle.score_name != "*", bundle.granularity is not None))

    def apply_result(self, result: DetectionResult, *, default_threshold: float = 0.5) -> list[str]:
        matched: list[str] = []
        for bundle in self.bundles:
            for score in result.method_scores:
                if bundle.apply_to_score(score):
                    matched.append(score.method)
            for unit in result.units:
                for score in unit.method_scores:
                    if bundle.apply_to_score(score):
                        matched.append(score.method)
        if not matched:
            return []
        thresholds = {bundle.decision_threshold for bundle in self.bundles if bundle.threshold is not None}
        threshold = thresholds.pop() if len(thresholds) == 1 else default_threshold
        _aggregate_calibrated_result(result, default_threshold=default_threshold, threshold=threshold)
        result.calibration = {
            "type": "calibration_registry",
            "bundles": [bundle.summary() for bundle in self.bundles],
            "applied_to": sorted(set(matched)),
            "uncalibrated_methods": _uncalibrated_methods(result),
        }
        return matched

    def to_dict(self) -> dict[str, Any]:
        return {"type": "calibration_registry", "bundles": [bundle.to_dict() for bundle in self.bundles]}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> CalibrationRegistry:
        if payload.get("type") not in {None, "calibration_registry"}:
            raise ValueError("not a calibration registry payload")
        return cls([CalibrationBundle.from_dict(item) for item in payload.get("bundles", [])])

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> CalibrationRegistry:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


CalibrationArtifact = CalibrationBundle | CalibrationRegistry


def load_calibration(source: CalibrationArtifact | str | Path | dict[str, Any]) -> CalibrationArtifact:
    """Load a bundle/registry, accepting legacy single-calibrator files."""

    if isinstance(source, (CalibrationBundle, CalibrationRegistry)):
        return source
    payload: dict[str, Any]
    if isinstance(source, dict):
        payload = source
    else:
        payload = json.loads(Path(source).read_text(encoding="utf-8"))
    artifact_type = payload.get("type")
    if artifact_type == "calibration_bundle":
        return CalibrationBundle.from_dict(payload)
    if artifact_type == "calibration_registry":
        return CalibrationRegistry.from_dict(payload)
    if artifact_type == "probability":
        return CalibrationBundle(score_name="*", probability=ProbabilityCalibrator.from_dict(payload))
    if artifact_type == "threshold":
        return CalibrationBundle(score_name="*", threshold=ThresholdCalibrator.from_dict(payload))
    # Backward compatibility for files written before artifacts carried "type".
    if artifact_type is None:
        if "bundles" in payload:
            return CalibrationRegistry.from_dict(payload)
        if "params" in payload:
            return CalibrationBundle(score_name="*", probability=ProbabilityCalibrator.from_dict(payload))
        if isinstance(payload.get("threshold"), (int, float)):
            return CalibrationBundle(score_name="*", threshold=ThresholdCalibrator.from_dict(payload))
        if "score_name" in payload or "score_kind" in payload:
            return CalibrationBundle.from_dict(payload)
    raise ValueError(f"unsupported calibration artifact type: {artifact_type!r}")


def _aggregate_calibrated_result(result: DetectionResult, *, default_threshold: float, threshold: float) -> None:
    response_scores = [score for score in result.method_scores if score.calibrated_risk is not None]
    if response_scores:
        risk = float(np.mean([score.calibrated_risk for score in response_scores]))
    else:
        unit_risks: list[float] = []
        for unit in result.units:
            calibrated = [score.calibrated_risk for score in unit.method_scores if score.calibrated_risk is not None]
            if not calibrated:
                continue
            unit_risk = float(np.mean(calibrated))
            unit.risk = unit_risk
            unit.score = 1.0 - unit_risk
            unit.label = "high_risk" if unit_risk >= threshold else "low_risk"
            unit_risks.append(unit_risk)
        if not unit_risks:
            return
        risk = float(np.mean(unit_risks))

    result.risk = min(1.0, max(0.0, risk))
    result.score = 1.0 - result.risk
    result.threshold = float(threshold if threshold is not None else default_threshold)
    result.is_hallucinated = result.risk >= result.threshold


def _uncalibrated_methods(result: DetectionResult) -> list[str]:
    names = {score.method for score in result.method_scores if score.calibrated_risk is None}
    for unit in result.units:
        names.update(score.method for score in unit.method_scores if score.calibrated_risk is None)
    return sorted(names)
