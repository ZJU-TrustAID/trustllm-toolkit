import pytest

from hallucination.detection.core.results import DetectionExample
from hallucination.detection.judges import LLMJudgeScorer, LLMPanelScorer


class _Judge:
    def __init__(self, response: str) -> None:
        self.response = response
        self.model_name = "fake-judge"

    def generate(self, prompt, **kwargs):
        return self.response


def test_binary_judge_maps_verdict_and_confidence():
    result = LLMJudgeScorer(_Judge('{"verdict": "hallucinated", "confidence": 0.8}')).score(
        DetectionExample(prompt="Q", response="A")
    )
    assert result.risk == pytest.approx(0.8)
    assert result.signal_type == "judge_assessment"


def test_continuous_and_likert_modes():
    continuous = LLMJudgeScorer(_Judge('{"hallucination_score": 70}'), mode="continuous").score(
        DetectionExample(prompt="Q", response="A")
    )
    assert continuous.risk == pytest.approx(0.7)

    likert = LLMJudgeScorer(_Judge('{"faithfulness": 2}'), mode="likert").score(
        DetectionExample(prompt="Q", response="A")
    )
    assert likert.risk == pytest.approx(0.75)


def test_judge_parses_fenced_json():
    result = LLMJudgeScorer(_Judge('```json\n{"verdict": "faithful", "confidence": 0.9}\n```')).score(
        DetectionExample(prompt="Q", response="A")
    )
    assert result.risk == pytest.approx(0.1)


def test_panel_averages_judges():
    panel = LLMPanelScorer(
        [
            LLMJudgeScorer(_Judge('{"verdict": "hallucinated", "confidence": 0.8}')),
            LLMJudgeScorer(_Judge('{"verdict": "faithful", "confidence": 0.8}')),
        ]
    )
    result = panel.score(DetectionExample(prompt="Q", response="A"))
    assert result.risk == pytest.approx(0.5)
    assert result.raw["num_judges"] == 2


def test_judge_rejects_invalid_json():
    with pytest.raises(ValueError, match="JSON object"):
        LLMJudgeScorer(_Judge("not json")).score(DetectionExample(prompt="Q", response="A"))
