from hallucination.detection.blackbox.exact_match import (
    ExactMatchConsistencyScorer,
)
from hallucination.detection.core.results import DetectionExample


def test_exact_match_consistency_uses_samples_only():
    example = DetectionExample(response="softmax", samples=["Softmax!", "sigmoid"])
    result = ExactMatchConsistencyScorer().score(example)
    assert result.score == 0.5
