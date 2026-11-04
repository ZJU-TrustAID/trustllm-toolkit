import pytest

from hallucination.detection.backends.api import CallableLLMBackend
from hallucination.detection.backends.nli import NLIProbabilities
from hallucination.detection.blackbox.selfcheck_nli import SelfCheckNLIScorer
from hallucination.detection.core.results import DetectionExample, Evidence
from hallucination.detection.evidence import (
    FActScoreScorer,
    LocalCorpusRetriever,
    NLIEvidenceVerifier,
)
from hallucination.detection.pipelines import (
    EvidenceVerificationPipeline,
    LongFormVerificationPipeline,
)
from hallucination.detection.pipelines.short_answer import BlackBoxSamplingPipeline


def _nli(premise: str, hypothesis: str) -> NLIProbabilities:
    if "磷虾" in premise and "海藻" in hypothesis:
        return NLIProbabilities(entailment=0.0, contradiction=1.0)
    if "海葵" in premise and "不是植物" in hypothesis:
        return NLIProbabilities(entailment=1.0, contradiction=0.0)
    return NLIProbabilities(entailment=0.0, contradiction=0.0, neutral=1.0)


def _pipeline() -> EvidenceVerificationPipeline:
    corpus = [
        Evidence(title="蓝鲸食性", snippet="蓝鲸以磷虾等小型甲壳类动物为主食。"),
        Evidence(title="海葵分类", snippet="海葵属于刺胞动物，是动物而不是植物。"),
    ]
    scorer = FActScoreScorer(NLIEvidenceVerifier(_nli), retriever=LocalCorpusRetriever(corpus))
    return EvidenceVerificationPipeline(scorer)


def test_local_retriever_returns_ranked_evidence_with_traceable_score():
    evidence = LocalCorpusRetriever([Evidence(title="蓝鲸食性", snippet="蓝鲸以磷虾为主食。")]).retrieve("蓝鲸主要吃什么？")
    assert len(evidence) == 1
    assert evidence[0].metadata["retrieval"] == "local_lexical"
    assert evidence[0].metadata["retrieval_score"] > 0


def test_evidence_pipeline_keeps_supported_contradicted_and_unknown_distinct():
    result = _pipeline().detect(
        DetectionExample(
            response="unused",
            claims=["蓝鲸主要以海藻为食。", "海葵不是植物。", "火星有两颗卫星。"],
        )
    )
    assert [claim.label for claim in result.claims] == ["contradicted", "supported", "unknown"]
    assert result.score == 1 / 3
    assert result.risk == pytest.approx(2 / 3)
    assert result.is_hallucinated is True
    assert result.signal_scores["evidence_support"][0].raw["unknown_ratio"] == 1 / 3
    assert result.claims[0].evidence[0].title == "蓝鲸食性"
    assert result.claims[2].evidence == []


def test_factscore_all_unknown_is_high_risk_by_default():
    result = _pipeline().detect(DetectionExample(response="unused", claims=["火星有两颗卫星。"]))
    assert result.claims[0].label == "unknown"
    assert result.score == 0.0
    assert result.risk == 1.0
    assert result.is_hallucinated is True
    assert result.signal_scores["evidence_support"][0].raw["unknown_policy"] == "unsupported"


def test_factscore_uses_sentence_fallback_when_claims_are_not_supplied():
    result = _pipeline().detect(response="海葵不是植物。")
    assert result.claims[0].claim == "海葵不是植物。"
    assert result.claims[0].label == "supported"


def test_long_form_workflow_returns_separate_signal_results():
    sampling = BlackBoxSamplingPipeline(
        CallableLLMBackend(lambda prompt, **kwargs: "海葵不是植物。"),
        sentence_scorers=[SelfCheckNLIScorer(_nli)],
        num_samples=2,
    )
    result = LongFormVerificationPipeline(sampling, evidence_pipeline=_pipeline()).detect_prompt("海葵是植物吗？")
    assert result.self_consistency.signal_scores.keys() == {"self_consistency"}
    assert result.evidence_support is not None
    assert result.evidence_support.signal_scores.keys() == {"evidence_support"}
