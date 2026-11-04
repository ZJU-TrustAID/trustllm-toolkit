from __future__ import annotations

import math

from hallucination.detection.core.base import BaseScorer
from hallucination.detection.core.results import DetectionExample, ScoreResult
from hallucination.detection.core.thresholds import clamp01, confidence_to_risk
from hallucination.detection.whitebox.tokens import TokenFilter, select_indices


class MTPScorer(BaseScorer):
    """Minimum token probability over the selected generated tokens."""

    name = "mtp"
    requires = {"token_logprobs"}

    def __init__(self, token_filter: TokenFilter | None = None) -> None:
        self.token_filter = token_filter

    def score(self, example: DetectionExample) -> ScoreResult:
        self.validate(example)
        assert example.token_logprobs is not None
        indices = select_indices(example, self.token_filter)
        min_index = min(indices, key=lambda index: example.token_logprobs[index])
        min_logprob = float(example.token_logprobs[min_index])
        confidence = clamp01(math.exp(min_logprob))
        return ScoreResult(
            method=self.name,
            score=confidence,
            risk=confidence_to_risk(confidence),
            raw={
                "min_logprob": min_logprob,
                "min_token_index": min_index,
                "min_token": _token_text(example, min_index),
                "num_tokens": len(indices),
                "original_num_tokens": len(example.token_logprobs),
                "token_indices": indices,
                "token_filter": self.token_filter.to_dict() if self.token_filter else None,
            },
            explanation="Minimum generated-token probability.",
            assumptions=["Selected tokens exclude special tokens only when token metadata is provided."],
            warnings=["A low-probability token is a local uncertainty signal, not proof of a factual error."],
            provenance={"token_filter": self.token_filter.to_dict() if self.token_filter else None},
        )


class LNTPScorer(BaseScorer):
    """Length-normalized (geometric mean) token probability."""

    name = "lntp"
    requires = {"token_logprobs"}

    def __init__(self, token_filter: TokenFilter | None = None) -> None:
        self.token_filter = token_filter

    def score(self, example: DetectionExample) -> ScoreResult:
        self.validate(example)
        assert example.token_logprobs is not None
        indices = select_indices(example, self.token_filter)
        mean_logprob = sum(float(example.token_logprobs[index]) for index in indices) / len(indices)
        confidence = clamp01(math.exp(mean_logprob))
        return ScoreResult(
            method=self.name,
            score=confidence,
            risk=confidence_to_risk(confidence),
            raw={
                "mean_logprob": mean_logprob,
                "num_tokens": len(indices),
                "original_num_tokens": len(example.token_logprobs),
                "token_indices": indices,
                "token_filter": self.token_filter.to_dict() if self.token_filter else None,
            },
            explanation="Length-normalized sequence probability.",
            assumptions=["Token probabilities are averaged in log space (geometric mean)."],
            warnings=["Length normalization makes values comparable across lengths but does not make them calibrated."],
            provenance={"token_filter": self.token_filter.to_dict() if self.token_filter else None},
        )


def _token_text(example: DetectionExample, index: int) -> str | None:
    if example.tokens is not None and index < len(example.tokens):
        return example.tokens[index]
    return None
