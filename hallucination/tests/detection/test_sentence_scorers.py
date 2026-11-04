import pytest

from hallucination.detection.backends.nli import NLIProbabilities
from hallucination.detection.blackbox.selfcheck_nli import SelfCheckNLIScorer
from hallucination.detection.blackbox.sentence import BaseSentenceScorer
from hallucination.detection.core.results import DetectionExample, ScoreResult
from hallucination.detection.pipelines.short_answer import BlackBoxSamplingPipeline


def _nli(premise: str, hypothesis: str) -> NLIProbabilities:
    if "海藻" in hypothesis:
        return NLIProbabilities(entailment=0.0, contradiction=1.0)
    if "海葵" in hypothesis and ("动物" in premise or "刺胞" in premise):
        return NLIProbabilities(entailment=1.0, contradiction=0.0)
    if premise == hypothesis:
        return NLIProbabilities(entailment=1.0, contradiction=0.0)
    return NLIProbabilities(entailment=0.0, contradiction=0.0, neutral=1.0)


class _FakeSentenceScorer(BaseSentenceScorer):
    name = "fake"

    def __init__(self, risks: list[float]) -> None:
        self.risks = risks

    def score_sentences(self, example: DetectionExample, sentences: list[str]) -> list[ScoreResult]:
        return [
            ScoreResult(
                method=self.name,
                signal_type=self.signal_type,
                score=1.0 - risk,
                risk=risk,
                level="sentence",
            )
            for risk in self.risks[: len(sentences)]
        ]


def test_selfcheck_nli_returns_one_sentence_result_per_source_sentence():
    example = DetectionExample(response="海葵是动物。", samples=["海葵属于刺胞动物。", "海葵是动物。"])
    result = SelfCheckNLIScorer(_nli).score_sentences(example, [example.response])[0]
    assert result.level == "sentence"
    assert result.risk == 0.0
    assert len(result.raw["comparisons"]) == 2
    assert result.raw["premise"] == "sentence"


def test_sampling_pipeline_attaches_sentence_results_and_summary():
    pipeline = BlackBoxSamplingPipeline(
        _Backend(),
        sentence_scorers=[SelfCheckNLIScorer(_nli)],
        num_samples=2,
    )
    result = pipeline.detect_prompt("海葵是植物吗？")
    assert len(result.units) == 1
    assert result.units[0].method_scores[0].method == "selfcheck_nli"
    assert result.metadata["sentence_summaries"]["selfcheck_nli"]["mean_risk"] == result.units[0].risk


def test_sentence_summaries_aggregate_all_sentences_per_scorer():
    example = DetectionExample(
        response="句子一。句子二。",
        samples=["样本一", "样本二"],
        units=["句子一。", "句子二。"],
    )
    pipeline = BlackBoxSamplingPipeline(
        _Backend(),
        sentence_scorers=[_FakeSentenceScorer([0.9, 0.1]), _FakeSentenceScorer([0.7, 0.3])],
        num_samples=2,
    )
    result = pipeline.detect(example)
    summaries = result.metadata["sentence_summaries"]
    assert summaries["fake"]["mean_risk"] == pytest.approx(0.5)
    assert summaries["fake"]["max_risk"] == pytest.approx(0.9)
    assert summaries["fake_1"]["mean_risk"] == pytest.approx(0.5)
    assert summaries["fake_1"]["max_risk"] == pytest.approx(0.7)


class _Backend:
    def generate(self, prompt: str, **kwargs: object) -> str:
        return "海葵是动物。"

    def sample(self, prompt: str, n: int, **kwargs: object) -> list[str]:
        return ["海葵是动物。"] * n
