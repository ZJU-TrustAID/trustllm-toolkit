from __future__ import annotations

from hallucination.mitigation.evaluation.benchmark import (
    BenchmarkResult,
    evaluate_records,
)
from hallucination.mitigation.evaluation.metrics import (
    contains_match,
    exact_match,
    normalize_text,
)

__all__ = [
    "BenchmarkResult",
    "contains_match",
    "evaluate_records",
    "exact_match",
    "normalize_text",
]
