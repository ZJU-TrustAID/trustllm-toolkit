import json
import math

import pytest

from hallucination.detection.backends.nli import NLIProbabilities
from hallucination.detection.blackbox.response_semantic_entropy import (
    ResponseSemanticEntropyScorer,
)
from hallucination.detection.core.results import DetectionExample


def _cluster_nli(premise: str, hypothesis: str) -> NLIProbabilities:
    def contradictory(text: str) -> bool:
        return "海藻" in text or ("植物" in text and "不是植物" not in text)

    if contradictory(premise) or contradictory(hypothesis):
        return NLIProbabilities(entailment=0.01, contradiction=0.95, neutral=0.04)
    return NLIProbabilities(entailment=0.9, contradiction=0.05, neutral=0.05)


def test_response_semantic_entropy_is_zero_for_one_cluster():
    scorer = ResponseSemanticEntropyScorer(_cluster_nli)
    result = scorer.score(
        DetectionExample(response="海葵是动物。", samples=["海葵是动物。", "海葵属于动物。"])
    )
    assert result.risk == 0.0
    assert result.raw["semantic_entropy"] == 0.0
    assert result.raw["num_semantic_sets"] == 1


def test_response_semantic_entropy_matches_textbook_four_one_example():
    scorer = ResponseSemanticEntropyScorer(_cluster_nli)
    result = scorer.score(
        DetectionExample(
            response="海葵是动物。",
            samples=["海葵是动物。", "海葵不是植物。", "海葵属于动物。", "海葵是一种植物。"],
        )
    )
    # 4 equivalent + 1 contradictory -> H = -(0.8 log2 0.8 + 0.2 log2 0.2) = 0.7219 bits
    assert result.raw["semantic_entropy"] == pytest.approx(0.7219, abs=1e-4)
    assert result.raw["num_semantic_sets"] == 2
    assert result.risk == pytest.approx(0.7219 / math.log2(5), abs=1e-4)
    assert result.score == pytest.approx(1.0 - result.risk)


def test_response_semantic_entropy_supports_samples_only():
    scorer = ResponseSemanticEntropyScorer(_cluster_nli, include_response=False)
    result = scorer.score(
        DetectionExample(response="海葵是动物。", samples=["海葵是动物。", "海葵是一种植物。"])
    )
    assert result.raw["num_candidates"] == 2
    assert result.risk == pytest.approx(1.0)


def test_response_semantic_entropy_raw_is_json_serializable():
    scorer = ResponseSemanticEntropyScorer(_cluster_nli)
    result = scorer.score(DetectionExample(response="a", samples=["a", "b"]))
    payload = json.dumps(result.raw, ensure_ascii=False)
    assert "pair_labels" in payload
