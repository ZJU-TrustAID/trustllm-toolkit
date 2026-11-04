from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable

from hallucination.detection.core.results import DetectionExample, ScoreResult


class BaseScorer(ABC):
    """Base interface implemented by all hallucination detection methods."""

    name: str = "base"
    requires: set[str] = set()
    signal_type: str = "generation_confidence"

    def validate(self, example: DetectionExample) -> None:
        missing = [field for field in self.requires if _is_empty(getattr(example, field))]
        if missing:
            raise ValueError(f"{self.name} requires missing fields: {', '.join(sorted(missing))}")

    @abstractmethod
    def score(self, example: DetectionExample) -> ScoreResult:
        """Return a confidence/risk score for one detection example."""


def score_many(scorer: BaseScorer, examples: Iterable[DetectionExample]) -> list[ScoreResult]:
    return [scorer.score(example) for example in examples]


def _is_empty(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, (str, bytes)):
        return len(value) == 0
    try:
        return len(value) == 0  # type: ignore[arg-type]
    except TypeError:
        return False
