import math

import pytest

from hallucination.detection.core.results import DetectionExample
from hallucination.detection.whitebox.entropy import TokenEntropyScorer


def test_token_entropy_normalizes_top_logprobs():
    example = DetectionExample(response="x", top_logprobs=[{"a": math.log(0.5), "b": math.log(0.5)}])
    result = TokenEntropyScorer().score(example)
    assert round(result.risk, 6) == 1.0
    assert round(result.score, 6) == 0.0


def test_single_candidate_step_is_skipped_not_treated_as_confident():
    example = DetectionExample(response="x", top_logprobs=[{"a": math.log(0.7)}])
    with pytest.raises(ValueError, match="more than one candidate"):
        TokenEntropyScorer().score(example)

    legacy = TokenEntropyScorer(single_candidate_policy="zero").score(example)
    assert legacy.risk == 0.0
    assert legacy.raw["single_candidate_steps"] == 1

