from __future__ import annotations

from collections.abc import Sequence

from hallucination.detection.core.base import BaseScorer
from hallucination.detection.core.results import (
    DetectionExample,
    DetectionResult,
    ScoreResult,
)
from hallucination.detection.core.thresholds import clamp01, risk_to_label


class DetectionPipeline:
    def __init__(
        self,
        scorers: Sequence[BaseScorer],
        *,
        threshold: float = 0.5,
        aggregation: str = "mean",
        allow_mixed_signals: bool = False,
    ) -> None:
        if aggregation != "mean":
            raise ValueError("Only mean aggregation is implemented in the scaffold.")
        self.scorers = list(scorers)
        self.threshold = threshold
        self.aggregation = aggregation
        self.allow_mixed_signals = allow_mixed_signals

    def detect(self, example: DetectionExample | None = None, **kwargs: object) -> DetectionResult:
        if example is None:
            example = DetectionExample(**kwargs)
        scores: list[ScoreResult] = [scorer.score(example) for scorer in self.scorers]
        groups: dict[str, list[ScoreResult]] = {}
        for item in scores:
            groups.setdefault(item.signal_type, []).append(item)
        if len(groups) > 1 and not self.allow_mixed_signals:
            names = ", ".join(sorted(groups))
            raise ValueError(
                "Refusing to aggregate different signal types by default: "
                f"{names}. Aggregate them explicitly after validation."
            )

        score = clamp01(sum(item.score for item in scores) / len(scores)) if scores else 0.0
        risk = clamp01(sum(item.risk for item in scores) / len(scores)) if scores else 1.0
        return DetectionResult(
            score=score,
            risk=risk,
            is_hallucinated=risk_to_label(risk, self.threshold),
            threshold=self.threshold,
            method_scores=scores,
            signal_scores=groups,
        )
