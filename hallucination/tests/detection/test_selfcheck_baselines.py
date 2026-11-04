from hallucination.detection.blackbox.selfcheck_bertscore import (
    SelfCheckBERTScoreScorer,
)
from hallucination.detection.blackbox.selfcheck_ngram import SelfCheckNgramScorer
from hallucination.detection.core.results import DetectionExample


def test_selfcheck_ngram_assigns_lower_risk_to_seen_text():
    example = DetectionExample(response="blue whale", samples=["blue whale", "blue whale"])
    scorer = SelfCheckNgramScorer()
    seen, unseen = scorer.score_sentences(example, ["blue whale", "red fox"])
    assert seen.risk < unseen.risk
    assert seen.raw["n"] == 1


def test_selfcheck_bertscore_uses_best_sentence_in_each_sample():
    def similarity(candidates, references):
        assert references == ["target", "target"]
        return [0.2, 0.9]

    example = DetectionExample(response="target", samples=["weak. strong."])
    result = SelfCheckBERTScoreScorer(similarity).score_sentences(example, ["target"])[0]
    assert result.score == 0.9
    assert result.raw["matches"][0]["matched_sentence"] == "strong."
