import pytest

from hallucination.detection.blackbox.response_cosine import (
    ResponseCosineConsistencyScorer,
)
from hallucination.detection.core.results import DetectionExample


def _embedding(texts):
    table = {
        "a": [1.0, 0.0],
        "b": [1.0, 0.0],
        "c": [0.0, 1.0],
        "d": [-1.0, 0.0],
    }
    return [table[text] for text in texts]


def test_response_cosine_identical_text_scores_one():
    scorer = ResponseCosineConsistencyScorer(_embedding)
    result = scorer.score(DetectionExample(response="a", samples=["b", "b"]))
    assert result.score == pytest.approx(1.0)
    assert result.risk == pytest.approx(0.0)
    assert result.level == "response"


def test_response_cosine_averages_raw_cosine_then_normalizes():
    scorer = ResponseCosineConsistencyScorer(_embedding)
    result = scorer.score(DetectionExample(response="a", samples=["b", "c", "d"]))
    # raw cosines: 1, 0, -1 -> normalized: 1, 0.5, 0 -> mean 0.5
    assert result.raw["cosine_similarities"] == pytest.approx([1.0, 0.0, -1.0])
    assert result.score == pytest.approx(0.5)


def test_response_cosine_can_disable_normalization():
    scorer = ResponseCosineConsistencyScorer(_embedding, normalize=False)
    result = scorer.score(DetectionExample(response="a", samples=["b", "c", "d"]))
    # raw cosine clipped to [0, 1]: 1, 0, 0 -> mean 1/3
    assert result.score == pytest.approx(1 / 3)


def test_response_cosine_rejects_empty_samples():
    scorer = ResponseCosineConsistencyScorer(_embedding)
    with pytest.raises(ValueError):
        scorer.score(DetectionExample(response="a", samples=[]))
