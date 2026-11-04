from __future__ import annotations

from hallucination.detection.core.base import BaseScorer
from hallucination.detection.core.results import DetectionExample, ScoreResult
from hallucination.detection.core.thresholds import confidence_to_risk
from hallucination.detection.utils.text import normalize_answer


class ExactMatchConsistencyScorer(BaseScorer):
    """Measure literal agreement between the original response and sampled responses.

    ``normalize=True`` applies lowercase/whitespace/punctuation normalization,
    which is useful for Chinese short answers. ``normalize=False`` reproduces
    the strict equality of the textbook EMR formula and the official
    SelfCheckGPT ``MatchScorer``.
    """

    name = "exact_match_consistency"
    requires = {"samples"}
    signal_type = "self_consistency"

    def __init__(self, *, normalize: bool = True) -> None:
        self.normalize = normalize

    def score(self, example: DetectionExample) -> ScoreResult:
        self.validate(example)
        assert example.samples is not None
        response = _prepare(example.response, self.normalize)
        matches = [int(response == _prepare(candidate, self.normalize)) for candidate in example.samples]
        confidence = sum(matches) / len(matches)
        return ScoreResult(
            method=self.name,
            signal_type=self.signal_type,
            score=confidence,
            risk=confidence_to_risk(confidence),
            raw={"matches": matches, "num_samples": len(example.samples), "normalized": self.normalize},
            assumptions=["Literal agreement between samples is a self-consistency signal, not factual verification."],
            warnings=["Exact matching ignores synonymous rewrites; use semantic scorers for free-form answers."],
            provenance={"normalize": self.normalize},
        )


def _prepare(text: str, normalize: bool) -> str:
    return normalize_answer(text) if normalize else text.strip()
