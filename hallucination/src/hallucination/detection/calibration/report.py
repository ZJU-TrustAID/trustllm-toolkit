from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from hallucination.detection.calibration.metrics import (
    calibration_metrics,
    reliability_table,
)


@dataclass
class CalibrationReport:
    """Calibration metrics and reliability table for one score vector."""

    scores: list[float]
    labels: list[bool]
    calibrated_scores: list[float] | None = None
    method: str | None = None
    n_bins: int = 10
    metrics: dict[str, float] = field(init=False)
    reliability: list[dict[str, float]] = field(init=False)
    calibrated_metrics: dict[str, float] | None = field(init=False, default=None)
    calibrated_reliability: list[dict[str, float]] | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        if len(self.scores) != len(self.labels):
            raise ValueError("scores and labels must have the same length")
        self.metrics = calibration_metrics(self.scores, self.labels, n_bins=self.n_bins)
        self.reliability = reliability_table(self.scores, self.labels, n_bins=self.n_bins)
        if self.calibrated_scores is not None:
            if len(self.calibrated_scores) != len(self.labels):
                raise ValueError("calibrated_scores and labels must have the same length")
            self.calibrated_metrics = calibration_metrics(self.calibrated_scores, self.labels, n_bins=self.n_bins)
            self.calibrated_reliability = reliability_table(self.calibrated_scores, self.labels, n_bins=self.n_bins)

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "n_bins": self.n_bins,
            "metrics": self.metrics,
            "reliability": self.reliability,
            "calibrated_metrics": self.calibrated_metrics,
            "calibrated_reliability": self.calibrated_reliability,
        }

    def to_json(self, path: str | Path | None = None) -> str:
        payload = json.dumps(self.to_dict(), ensure_ascii=False, indent=2)
        if path is not None:
            Path(path).write_text(payload, encoding="utf-8")
        return payload

    def to_markdown(self) -> str:
        lines = [
            "| score | count | ECE | MCE | Brier | NLL | gap |",
            "|---|---|---|---|---|---|---|",
            _row("raw", self.metrics),
        ]
        if self.calibrated_metrics is not None:
            lines.append(_row(self.method or "calibrated", self.calibrated_metrics))
        return "\n".join(lines)


def _row(name: str, metrics: dict[str, float]) -> str:
    return (
        f"| {name} | {int(metrics['count'])} | {metrics['ece']:.4f} | {metrics['mce']:.4f} | "
        f"{metrics['brier']:.4f} | {metrics['nll']:.4f} | {metrics['calibration_gap']:.4f} |"
    )
