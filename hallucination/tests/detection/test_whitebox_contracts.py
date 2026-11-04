import math

import numpy as np
import pytest

from hallucination.detection.blackbox.exact_match import ExactMatchConsistencyScorer
from hallucination.detection.core.results import DetectionExample
from hallucination.detection.whitebox.effective_rank import EffectiveRankScorer
from hallucination.detection.whitebox.eigenscore import EigenScoreScorer
from hallucination.detection.whitebox.entropy import TokenEntropyScorer
from hallucination.detection.whitebox.token_probability import LNTPScorer, MTPScorer
from hallucination.detection.whitebox.tokens import TokenFilter


def test_token_filter_skips_special_tokens_for_mtp():
    example = DetectionExample(
        response="ab",
        token_logprobs=[math.log(0.9), math.log(0.01), math.log(0.5)],
        tokens=["a", "<eos>", "b"],
        token_is_special=[False, True, False],
    )
    result = MTPScorer(TokenFilter(skip_special=True)).score(example)
    assert result.score == pytest.approx(0.5)
    assert result.raw["min_token"] == "b"
    assert result.raw["num_tokens"] == 2


def test_token_filter_rejects_all_filtered_tokens():
    example = DetectionExample(response="a", token_logprobs=[math.log(0.5)], tokens=["<eos>"], token_is_special=[True])
    with pytest.raises(ValueError, match="removed every token"):
        MTPScorer(TokenFilter()).score(example)


def test_lntp_uses_geometric_mean_and_filter():
    example = DetectionExample(response="ab", token_logprobs=[math.log(0.25), math.log(0.01), math.log(1.0)], token_is_special=[False, True, False])
    result = LNTPScorer(TokenFilter()).score(example)
    assert result.score == pytest.approx(0.5)


def test_entropy_reports_coverage_and_handles_single_candidate():
    example = DetectionExample(response="a", top_logprobs=[{"a": math.log(0.6), "b": math.log(0.4)}, {"a": 0.0}])
    legacy = TokenEntropyScorer(aggregation="mean", single_candidate_policy="zero").score(example)
    assert legacy.raw["distribution_coverage"] == pytest.approx([1.0, 1.0])
    assert legacy.raw["single_candidate_steps"] == 1
    assert legacy.risk < 1.0

    skipped = TokenEntropyScorer(aggregation="mean").score(example)
    assert skipped.raw["distribution_coverage"] == pytest.approx([1.0])
    assert skipped.raw["skipped_single_candidate_steps"] == 1
    assert skipped.raw["num_steps"] == 1


def test_entropy_p95_aggregation():
    example = DetectionExample(response="a", top_logprobs=[{"a": math.log(0.5), "b": math.log(0.5)}] * 5)
    mean_result = TokenEntropyScorer(aggregation="mean").score(example)
    p95_result = TokenEntropyScorer(aggregation="p95").score(example)
    assert p95_result.risk == pytest.approx(mean_result.risk)


def test_eigenscore_returns_finite_risk_for_multiple_samples():
    states = np.asarray([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]])
    result = EigenScoreScorer().score(DetectionExample(response="x", hidden_states=states))
    assert 0.0 <= result.risk <= 1.0
    assert result.raw["num_samples"] == 4
    assert result.raw["variant"] == "mean_log"
    assert len(result.raw["eigenvalues"]) == 4


def test_eigenscore_requires_two_samples():
    with pytest.raises(ValueError, match="at least two"):
        EigenScoreScorer().score(DetectionExample(response="x", hidden_states=[[1.0, 0.0]]))


def test_effective_rank_spectral_entropy_and_normalization():
    states = np.asarray([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    result = EffectiveRankScorer().score(DetectionExample(response="x", hidden_states=states))
    assert result.raw["spectral_entropy"] > 0
    assert result.raw["effective_rank"] > 1
    assert 0.0 <= result.raw["normalized_effective_rank"] <= 1.0
    assert 0.0 <= result.risk <= 1.0


def test_exact_match_normalize_toggle():
    example = DetectionExample(response="Yes!", samples=["yes"])
    assert ExactMatchConsistencyScorer(normalize=True).score(example).score == 1.0
    assert ExactMatchConsistencyScorer(normalize=False).score(example).score == 0.0
