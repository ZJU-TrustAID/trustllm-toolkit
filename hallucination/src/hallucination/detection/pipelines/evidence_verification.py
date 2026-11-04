from __future__ import annotations

from hallucination.detection.core.results import DetectionExample, DetectionResult
from hallucination.detection.core.thresholds import risk_to_label
from hallucination.detection.evidence.factscore import FActScoreScorer


class EvidenceVerificationPipeline:
    """Run FActScore-style verification without aggregating other signals."""

    def __init__(self, scorer: FActScoreScorer, *, threshold: float = 0.5) -> None:
        self.scorer = scorer
        self.threshold = threshold

    def detect(self, example: DetectionExample | None = None, **kwargs: object) -> DetectionResult:
        if example is None:
            example = DetectionExample(**kwargs)
        assessment = self.scorer.assess(example)
        score = assessment.score
        return DetectionResult(
            score=score.score,
            risk=score.risk,
            is_hallucinated=risk_to_label(score.risk, self.threshold),
            threshold=self.threshold,
            method_scores=[score],
            signal_scores={"evidence_support": [score]},
            claims=assessment.claims,
            evidence=assessment.evidence,
            metadata={"claim_decomposition": type(self.scorer.decomposer).__name__},
        )
