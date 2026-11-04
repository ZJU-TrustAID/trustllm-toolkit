from __future__ import annotations

from collections.abc import Callable
from statistics import mean

from hallucination.detection.core.base import BaseScorer
from hallucination.detection.core.results import DetectionExample, ScoreResult
from hallucination.detection.core.thresholds import clamp01, confidence_to_risk
from hallucination.detection.utils.math import cosine

EmbeddingFn = Callable[[list[str]], list[list[float]]]


class ResponseCosineConsistencyScorer(BaseScorer):
    """Response-level cosine consistency between the answer and its samples.

    This follows the textbook formula

        C(y_i; y_i~, x_i) = (1/m) sum_j cos(V(y_i), V(y_ij~))

    and the response-level ``cosine_sim`` scorer in UQLM. Sentence alignment is
    intentionally not part of this scorer; sentence-level consistency is covered
    by the SelfCheck-family scorers (NLI and BERTScore).
    """

    name = "response_cosine_consistency"
    requires = {"samples"}
    signal_type = "self_consistency"

    def __init__(self, embedding_fn: EmbeddingFn, *, normalize: bool = True, ignore_empty_samples: bool = True) -> None:
        if not callable(embedding_fn):
            raise TypeError("embedding_fn must be callable")
        self.embedding_fn = embedding_fn
        self.normalize = normalize
        self.ignore_empty_samples = ignore_empty_samples

    def score(self, example: DetectionExample) -> ScoreResult:
        self.validate(example)
        assert example.samples is not None
        samples = [sample for sample in example.samples if sample.strip()] if self.ignore_empty_samples else list(example.samples)
        if not samples:
            raise ValueError("response_cosine_consistency requires at least one non-empty sample")

        texts = [example.response, *samples]
        vectors = self.embedding_fn(texts)
        if len(vectors) != len(texts):
            raise ValueError(f"embedding_fn must return {len(texts)} vectors, got {len(vectors)}")
        response_vector = vectors[0]
        if not response_vector:
            raise ValueError("embedding_fn returned an empty response vector")

        raw_similarities: list[float] = []
        normalized_similarities: list[float] = []
        for vector in vectors[1:]:
            if len(vector) != len(response_vector):
                raise ValueError("all embedding vectors must have the same dimension")
            similarity = cosine(response_vector, vector)
            raw_similarities.append(similarity)
            normalized_similarities.append(clamp01((similarity + 1.0) / 2.0) if self.normalize else clamp01(similarity))

        score = clamp01(mean(normalized_similarities))
        return ScoreResult(
            method=self.name,
            signal_type=self.signal_type,
            score=score,
            risk=confidence_to_risk(score),
            level="response",
            raw={
                "num_samples": len(samples),
                "cosine_similarities": raw_similarities,
                "normalized_similarities": normalized_similarities,
                "normalize": self.normalize,
                "min_similarity": min(normalized_similarities),
                "max_similarity": max(normalized_similarities),
            },
            explanation="Mean response-level cosine similarity between the answer and independently sampled answers.",
            assumptions=[
                "The embedding backend is fixed for the compared responses.",
                "A high cosine similarity means semantic proximity, not factual support.",
            ],
            warnings=[
                "Cosine similarity can be high for two responses that share wording but differ in facts.",
                "When normalize=True, a raw cosine in [-1, 1] is mapped to [0, 1].",
            ],
            provenance={
                "embedding_backend": getattr(self.embedding_fn, "model_name", type(self.embedding_fn).__name__),
                "normalize": self.normalize,
            },
        )
