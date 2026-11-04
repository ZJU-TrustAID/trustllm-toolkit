import math

import pytest

from hallucination.detection.backends.demo import DemoPTrueBackend
from hallucination.detection.core.results import DetectionExample
from hallucination.detection.whitebox.p_true import PTrueScorer


class _LogprobBackend:
    def __init__(self, true_probability: float) -> None:
        self.true_probability = true_probability

    def next_token_logprobs(self, prompt, candidates):
        return {
            candidate: math.log(self.true_probability if candidate in {" A", "A", " True"} else 1.0 - self.true_probability)
            for candidate in candidates
        }


class _VerbalizedBackend:
    def generate(self, prompt, **kwargs):
        return "B"


def test_p_true_uses_demo_logprobs_for_hallucinated_answer():
    scorer = PTrueScorer(DemoPTrueBackend())
    result = scorer.score(
        DetectionExample(prompt="请介绍蓝鲸的食性。", response="蓝鲸主要以海藻为食。")
    )
    assert result.raw["method"] == "logprob"
    assert result.score == pytest.approx(0.2)
    assert result.risk == pytest.approx(0.8)
    assert result.signal_type == "self_reported_confidence"


def test_p_true_uses_demo_logprobs_for_supported_answer():
    result = PTrueScorer(DemoPTrueBackend()).score(
        DetectionExample(prompt="请介绍蓝鲸的食性。", response="蓝鲸主要以磷虾为食。")
    )
    assert result.score == pytest.approx(0.9)


def test_p_true_normalizes_a_b_logprobs():
    result = PTrueScorer(_LogprobBackend(0.75)).score(
        DetectionExample(prompt="Q", response="A")
    )
    assert result.score == pytest.approx(0.75)
    assert result.raw["token_details"]["true_probability"] == pytest.approx(0.75)
    assert result.raw["token_details"]["normalized"] is True


def test_p_true_can_read_only_the_a_logprob():
    class _PartialBackend:
        def next_token_logprobs(self, prompt, candidates):
            return {" A": math.log(0.6), " B": math.log(0.3)}

    result = PTrueScorer(_PartialBackend(), normalize_ab=False).score(
        DetectionExample(prompt="Q", response="A")
    )
    assert result.score == pytest.approx(0.6)
    assert result.raw["normalize_ab"] is False


def test_p_true_verbalized_fallback():
    result = PTrueScorer(_VerbalizedBackend()).score(DetectionExample(prompt="Q", response="A"))
    assert result.raw["method"] == "verbalized"
    assert result.score == 0.0


def test_p_true_requires_prompt_and_response():
    with pytest.raises(ValueError, match="missing fields"):
        PTrueScorer(DemoPTrueBackend()).score(DetectionExample(response="A"))
