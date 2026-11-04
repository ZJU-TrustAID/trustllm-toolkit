from __future__ import annotations

import math
import re
from collections import Counter

from hallucination.detection.blackbox.sentence import BaseSentenceScorer
from hallucination.detection.core.results import DetectionExample, ScoreResult
from hallucination.detection.core.thresholds import clamp01

_TOKEN_PATTERN = re.compile(r"[\u4e00-\u9fff]|[A-Za-z0-9_]+|[^\s]")
_LEFT_PAD = "<s>"


class SelfCheckNgramScorer(BaseSentenceScorer):
    """SelfCheck-style n-gram support baseline.

    Mirrors the official ``SelfCheckNgram`` construction: the n-gram model is
    trained on the original response *and* the sampled passages, uses left
    padding for ``n > 1``, and reports both average and maximum negative
    log-probability. The official code uses ``k=0`` smoothing, which makes
    unseen n-grams ``-inf``; this implementation floors the probability at
    ``unseen_floor`` so a score remains inspectable.
    """

    name = "selfcheck_ngram"

    def __init__(self, n: int = 1, *, lowercase: bool = True, smoothing: float = 0.0, unseen_floor: float = 1e-12) -> None:
        if n < 1:
            raise ValueError("n must be at least 1")
        if smoothing < 0:
            raise ValueError("smoothing must be non-negative")
        if not 0.0 < unseen_floor <= 1.0:
            raise ValueError("unseen_floor must be in (0, 1]")
        self.n = n
        self.lowercase = lowercase
        self.smoothing = smoothing
        self.unseen_floor = unseen_floor

    def score_sentences(self, example: DetectionExample, sentences: list[str]) -> list[ScoreResult]:
        if not example.samples:
            raise ValueError("selfcheck_ngram requires samples")
        counts, total_ngrams, vocabulary_size = self._train([example.response, *example.samples])
        denominator = total_ngrams + self.smoothing * vocabulary_size

        per_sentence: list[dict[str, object]] = []
        all_token_logprobs: list[list[float]] = []
        for sentence in sentences:
            ngrams = self._ngrams(self._tokens(sentence))
            token_logprobs: list[float] = []
            unseen = 0
            for ngram in ngrams:
                count = counts.get(ngram, 0)
                if count == 0:
                    unseen += 1
                probability = (count + self.smoothing) / denominator if denominator > 0 else 0.0
                if probability <= 0.0:
                    probability = self.unseen_floor
                token_logprobs.append(-math.log(probability))
            all_token_logprobs.append(token_logprobs)
            per_sentence.append(
                {
                    "avg_neg_logprob": sum(token_logprobs) / len(token_logprobs) if token_logprobs else 0.0,
                    "max_neg_logprob": max(token_logprobs, default=0.0),
                    "num_ngrams": len(ngrams),
                    "unseen_ngrams": unseen,
                }
            )

        doc_avg = (
            sum(value for values in all_token_logprobs for value in values)
            / sum(len(values) for values in all_token_logprobs)
            if any(all_token_logprobs)
            else 0.0
        )
        doc_avg_max = (
            sum(float(item["max_neg_logprob"]) for item in per_sentence) / len(per_sentence)
            if per_sentence
            else 0.0
        )

        results: list[ScoreResult] = []
        for sentence, values in zip(sentences, per_sentence, strict=False):
            average_negative_logprob = float(values["avg_neg_logprob"])
            score = clamp01(math.exp(-average_negative_logprob))
            results.append(
                ScoreResult(
                    method=self.name,
                    signal_type=self.signal_type,
                    score=score,
                    risk=clamp01(1.0 - score),
                    level="sentence",
                    raw={
                        "source_sentence": sentence,
                        "n": self.n,
                        "smoothing": self.smoothing,
                        "avg_neg_logprob": average_negative_logprob,
                        "max_neg_logprob": float(values["max_neg_logprob"]),
                        "num_ngrams": int(values["num_ngrams"]),
                        "unseen_ngrams": int(values["unseen_ngrams"]),
                        "doc_level": {
                            "avg_neg_logprob": doc_avg,
                            "avg_max_neg_logprob": doc_avg_max,
                        },
                    },
                    assumptions=[
                        "The original response and all sampled passages form the n-gram corpus.",
                        "N-gram support is a lexical self-consistency baseline, not factual verification.",
                    ],
                    warnings=[
                        "N-gram scores are unbounded negative log-probabilities; score/risk map them monotonically to [0, 1].",
                        "Unseen n-grams are floored for numerical stability; the official implementation can return -inf.",
                    ],
                    provenance={"n": self.n, "lowercase": self.lowercase, "smoothing": self.smoothing},
                )
            )
        return results

    def _train(self, passages: list[str]) -> tuple[Counter[tuple[str, ...]], int, int]:
        counts: Counter[tuple[str, ...]] = Counter()
        for passage in passages:
            counts.update(self._ngrams(self._tokens(passage)))
        return counts, sum(counts.values()), max(len(counts), 1)

    def _tokens(self, text: str) -> list[str]:
        if self.lowercase:
            text = text.lower()
        return _TOKEN_PATTERN.findall(text)

    def _ngrams(self, tokens: list[str]) -> list[tuple[str, ...]]:
        if not tokens:
            return []
        if self.n == 1:
            return [(token,) for token in tokens]
        padded = [_LEFT_PAD] * (self.n - 1) + list(tokens)
        return [tuple(padded[index : index + self.n]) for index in range(len(tokens))]
