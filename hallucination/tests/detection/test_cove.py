import pytest

from hallucination.detection.backends.nli import NLIProbabilities
from hallucination.detection.blackbox.cove import CoVeScorer
from hallucination.detection.core.results import DetectionExample


class _Backend:
    model_name = "fake-cove"

    def __init__(self, answers: list[str]) -> None:
        self.answers = answers
        self.answer_index = 0

    def generate(self, prompt, **kwargs):
        if "verification questions" in prompt.lower():
            return "蓝鲸吃什么？\n蓝鲸生活在哪里？"
        answer = self.answers[self.answer_index % len(self.answers)]
        self.answer_index += 1
        return answer


def _entailing_nli(premise, hypothesis):
    return NLIProbabilities(entailment=0.9, contradiction=0.05, neutral=0.05)


def _contradicting_nli(premise, hypothesis):
    if "淡水" in premise or "淡水" in hypothesis:
        return NLIProbabilities(entailment=0.05, contradiction=0.9, neutral=0.05)
    return NLIProbabilities(entailment=0.9, contradiction=0.05, neutral=0.05)


def test_cove_reports_zero_conflict_when_answers_entail_the_draft():
    scorer = CoVeScorer(_Backend(["磷虾。", "海洋。"]), nli=_entailing_nli)
    result = scorer.score(
        DetectionExample(prompt="请介绍蓝鲸。", response="蓝鲸主要以磷虾为食，生活在海洋中。")
    )
    assert result.raw["num_questions"] == 2
    assert result.raw["conflict_rate"] == 0.0
    assert result.risk == 0.0
    assert result.signal_type == "self_reported_confidence"


def test_cove_reports_conflict_rate():
    scorer = CoVeScorer(_Backend(["磷虾。", "淡水湖。"]), nli=_contradicting_nli)
    result = scorer.score(
        DetectionExample(prompt="请介绍蓝鲸。", response="蓝鲸主要以磷虾为食，生活在海洋中。")
    )
    assert result.raw["conflict_rate"] == 0.5
    assert result.risk == 0.5


def test_cove_requires_parsable_questions():
    class NoQuestions:
        def generate(self, prompt, **kwargs):
            return "no questions here"

    scorer = CoVeScorer(NoQuestions())
    with pytest.raises(ValueError, match="no parsable questions"):
        scorer.score(DetectionExample(prompt="Q", response="A"))
