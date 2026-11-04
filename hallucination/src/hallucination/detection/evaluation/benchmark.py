from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from hallucination.detection.calibration.bundle import (
    CalibrationBundle,
    CalibrationRegistry,
    load_calibration,
)
from hallucination.detection.calibration.metrics import calibration_metrics
from hallucination.detection.calibration.probability import ProbabilityCalibrator
from hallucination.detection.core.results import UnitResult
from hallucination.detection.datasets.schema import DatasetRecord
from hallucination.detection.evaluation.metrics import binary_risk_metrics


@dataclass
class BenchmarkReport:
    """Serializable metrics and failures from one benchmark run."""

    metrics: dict[str, dict[str, float | None]]
    failures: list[dict[str, str]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"metrics": self.metrics, "failures": self.failures, "metadata": self.metadata}

    def write_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

    def write_csv(self, path: str | Path) -> None:
        rows = [{"method": method, **values} for method, values in self.metrics.items()]
        if not rows:
            rows = [{"method": ""}]
        fieldnames: list[str] = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
        with Path(path).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)


class BenchmarkRunner:
    """Evaluate sentence- or response-level risk scorers against annotated records."""

    def __init__(
        self,
        pipeline: Any,
        *,
        threshold: float = 0.5,
        calibration: ProbabilityCalibrator | CalibrationBundle | CalibrationRegistry | str | Path | None = None,
        calibrations: dict[str, CalibrationBundle] | None = None,
        granularity: str = "sentence",
    ) -> None:
        if granularity not in {"sentence", "response"}:
            raise ValueError("granularity must be 'sentence' or 'response'")
        self._legacy_calibrator: ProbabilityCalibrator | None = None
        self._registry = CalibrationRegistry()
        if calibrations:
            for method, bundle in calibrations.items():
                if bundle.score_name == "*":
                    bundle.score_name = method
                self._registry.add(bundle)
        if calibration is not None:
            if isinstance(calibration, ProbabilityCalibrator):
                self._legacy_calibrator = calibration
                if calibration.split == "test":
                    raise ValueError("the supplied calibrator was fitted on the test split; refit it on validation")
                self._registry.add(CalibrationBundle(score_name="*", probability=calibration))
            else:
                artifact = load_calibration(calibration)
                if isinstance(artifact, CalibrationRegistry):
                    self._registry.bundles.extend(artifact.bundles)
                else:
                    self._registry.add(artifact)
        for bundle in self._registry.bundles:
            if bundle.split == "test":
                raise ValueError("the supplied calibrator was fitted on the test split; refit it on validation")
        self.pipeline = pipeline
        self.threshold = threshold
        self.calibration = calibration
        self.granularity = granularity

    def evaluate(self, dataset: list[DatasetRecord], *, granularity: str | None = None) -> BenchmarkReport:
        selected = granularity or self.granularity
        if selected not in {"sentence", "response"}:
            raise ValueError("granularity must be 'sentence' or 'response'")

        per_method_risks: dict[str, list[float]] = {}
        per_method_labels: dict[str, list[bool]] = {}
        failures: list[dict[str, str]] = []
        total_units = 0
        evaluated_units = 0

        for record in dataset:
            if selected == "sentence":
                if record.units is None or record.unit_labels is None:
                    failures.append({"id": record.id, "reason": "missing sentence units or labels"})
                    continue
                if record.unit_type not in {None, "sentence"}:
                    failures.append({"id": record.id, "reason": "sentence benchmark requires sentence units"})
                    continue
                total_units += len(record.units)
            else:
                if record.response_label is None:
                    failures.append({"id": record.id, "reason": "missing response_label"})
                    continue
                total_units += 1

            try:
                result = self.pipeline.detect(record.to_detection_example())
            except Exception as exc:  # Preserve a report for partially runnable datasets.
                failures.append({"id": record.id, "reason": f"{type(exc).__name__}: {exc}"})
                continue

            if selected == "sentence":
                if len(result.units) != len(record.unit_labels or []):
                    failures.append({"id": record.id, "reason": "pipeline output does not align with sentence labels"})
                    continue
                evaluated_units += len(result.units)
                self._collect_units(result.units, record.unit_labels or [], per_method_risks, per_method_labels)
            else:
                evaluated_units += 1
                self._collect_scores(
                    result.method_scores,
                    _hallucination_label(record.response_label),
                    per_method_risks,
                    per_method_labels,
                )

        metrics = {
            method: self._metrics(
                method,
                per_method_risks[method],
                per_method_labels[method],
                evaluated_units,
                total_units,
                selected,
            )
            for method in sorted(per_method_risks)
        }
        calibration_metadata: dict[str, Any] | None = None
        if self._legacy_calibrator is not None:
            calibration_metadata = {
                "method": self._legacy_calibrator.method,
                "target": self._legacy_calibrator.target,
                "fit_split": self._legacy_calibrator.split,
            }
        elif self._registry.bundles:
            calibration_metadata = {
                "type": "per_method",
                "bundles": {
                    bundle.score_name: bundle.summary() for bundle in self._registry.bundles
                },
            }
        return BenchmarkReport(
            metrics=metrics,
            failures=failures,
            metadata={
                "granularity": selected,
                "threshold": self.threshold,
                "total_units": total_units,
                "evaluated_units": evaluated_units,
                "calibration": calibration_metadata,
            },
        )

    def _metrics(
        self,
        method: str,
        risks: list[float],
        labels: list[bool],
        evaluated_units: int,
        total_units: int,
        granularity: str,
    ) -> dict[str, float | None]:
        metrics = binary_risk_metrics(risks, labels, threshold=self.threshold)
        metrics["coverage"] = evaluated_units / total_units if total_units else 0.0
        calibration = calibration_metrics(risks, labels)
        metrics.update({"ece": calibration["ece"], "brier": calibration["brier"], "nll": calibration["nll"]})
        bundle = self._registry.for_score(method, granularity)
        if bundle is not None:
            calibrated = [bundle.calibrate_risk(risk) for risk in risks]
            calibrated_metrics = binary_risk_metrics(calibrated, labels, threshold=bundle.decision_threshold)
            calibrated_calibration = calibration_metrics(calibrated, labels)
            metrics.update({f"calibrated_{key}": value for key, value in calibrated_metrics.items()})
            metrics.update(
                {
                    "calibrated_ece": calibrated_calibration["ece"],
                    "calibrated_brier": calibrated_calibration["brier"],
                    "calibrated_nll": calibrated_calibration["nll"],
                    "calibrated_threshold": bundle.decision_threshold,
                }
            )
        return metrics

    @staticmethod
    def _collect_units(
        units: list[UnitResult],
        labels: list[bool | str],
        risks: dict[str, list[float]],
        method_labels: dict[str, list[bool]],
    ) -> None:
        for unit, label in zip(units, labels, strict=False):
            BenchmarkRunner._collect_scores(unit.method_scores, _hallucination_label(label), risks, method_labels)

    @staticmethod
    def _collect_scores(
        scores: list[Any],
        label: bool,
        risks: dict[str, list[float]],
        method_labels: dict[str, list[bool]],
    ) -> None:
        for score in scores:
            risks.setdefault(score.method, []).append(float(score.risk))
            method_labels.setdefault(score.method, []).append(label)


def _hallucination_label(label: bool | str) -> bool:
    if isinstance(label, bool):
        return label
    normalized = label.strip().lower()
    if normalized in {"hallucinated", "not_supported", "contradicted", "unsupported", "true", "1"}:
        return True
    if normalized in {"supported", "factual", "false", "0"}:
        return False
    raise ValueError(f"Unsupported label: {label!r}")
