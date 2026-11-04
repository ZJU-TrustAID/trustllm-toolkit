from __future__ import annotations

from collections.abc import Callable

from hallucination.detection.blackbox.sentence import BaseSentenceScorer
from hallucination.detection.core.results import DetectionExample, ScoreResult
from hallucination.detection.utils.text import split_sentences

SimilarityFn = Callable[[list[str], list[str]], list[float]]


class SelfCheckBERTScoreScorer(BaseSentenceScorer):
    """Compare every source sentence to the best BERTScore match per sample."""

    name = "selfcheck_bertscore"

    def __init__(self, similarity_fn: SimilarityFn, *, min_sentence_length: int = 0) -> None:
        if min_sentence_length < 0:
            raise ValueError("min_sentence_length must be non-negative")
        self.similarity_fn = similarity_fn
        self.min_sentence_length = min_sentence_length

    def score_sentences(self, example: DetectionExample, sentences: list[str]) -> list[ScoreResult]:
        if not example.samples:
            raise ValueError("selfcheck_bertscore requires samples")
        sample_sentences = [
            [sentence for sentence in split_sentences(sample) if len(sentence) > self.min_sentence_length]
            for sample in example.samples
        ]
        results: list[ScoreResult] = []
        for source in sentences:
            matches: list[dict[str, object]] = []
            support_scores: list[float] = []
            for sample_index, candidates in enumerate(sample_sentences):
                if not candidates:
                    matches.append({"sample_index": sample_index, "matched_sentence": None, "bertscore_f1": None})
                    support_scores.append(0.0)
                    continue
                values = self.similarity_fn(candidates, [source] * len(candidates))
                if len(values) != len(candidates):
                    raise ValueError("similarity_fn must return one score per candidate/reference pair")
                best_index = max(range(len(values)), key=values.__getitem__)
                best_score = min(1.0, max(0.0, float(values[best_index])))
                support_scores.append(best_score)
                matches.append(
                    {"sample_index": sample_index, "matched_sentence": candidates[best_index], "bertscore_f1": best_score}
                )
            score = sum(support_scores) / len(support_scores)
            results.append(
                ScoreResult(
                    method=self.name,
                    signal_type=self.signal_type,
                    score=score,
                    risk=1.0 - score,
                    level="sentence",
                    raw={"source_sentence": source, "matches": matches, "unmatched_samples": sum(item["matched_sentence"] is None for item in matches)},
                    assumptions=["Each source sentence uses its best BERTScore match in every independently segmented sampled passage."],
                    warnings=["BERTScore similarity is not a logical-entailment or factual-support guarantee."],
                    provenance={"min_sentence_length": self.min_sentence_length},
                )
            )
        return results
