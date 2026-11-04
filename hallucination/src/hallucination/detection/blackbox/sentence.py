from __future__ import annotations

from abc import ABC, abstractmethod

from hallucination.detection.core.results import DetectionExample, ScoreResult


class BaseSentenceScorer(ABC):
    """A scorer that returns one result for every source-response sentence."""

    name: str = "sentence_base"
    signal_type: str = "self_consistency"

    @abstractmethod
    def score_sentences(self, example: DetectionExample, sentences: list[str]) -> list[ScoreResult]:
        """Score source sentences against the sampled passages in an example."""
