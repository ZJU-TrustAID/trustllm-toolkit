from __future__ import annotations

import math

import numpy as np

from hallucination.detection.core.base import BaseScorer
from hallucination.detection.core.results import DetectionExample, ScoreResult
from hallucination.detection.core.thresholds import clamp01
from hallucination.detection.utils.math import entropy_from_probs


class EffectiveRankScorer(BaseScorer):
    """Effective rank of a hidden-state matrix.

    The reference implementation reports the spectral entropy

        H = -sum_i p_i log p_i,  p_i = sigma_i / sum_j sigma_j

    while the common "effective rank" definition is ``exp(H)``. Both values are
    reported. ``risk`` rises as the normalized effective rank falls
    (representation collapse).
    """

    name = "effective_rank"
    requires = {"hidden_states"}

    def __init__(self, *, normalize: bool = True) -> None:
        self.normalize = normalize

    def score(self, example: DetectionExample) -> ScoreResult:
        self.validate(example)
        states = np.asarray(example.hidden_states, dtype=float)
        if states.ndim != 2:
            raise ValueError("hidden_states must be a 2D array shaped [n_samples_or_tokens, hidden_dim]")
        if states.shape[0] < 2:
            raise ValueError("EffectiveRank requires at least two hidden-state vectors")

        singular_values = np.linalg.svd(states, compute_uv=False)
        total = float(singular_values.sum())
        if total <= 0:
            return ScoreResult(
                method=self.name,
                signal_type="generation_confidence",
                score=0.0,
                risk=1.0,
                level="response",
                raw={"effective_rank": 0.0, "spectral_entropy": 0.0, "singular_values": singular_values.tolist()},
                warnings=["All singular values are zero; the hidden-state matrix carries no usable variation."],
            )

        probabilities = singular_values / total
        spectral_entropy = entropy_from_probs(probabilities)
        effective_rank = math.exp(spectral_entropy)
        max_rank = max(1, min(states.shape))
        normalized_rank = clamp01((effective_rank - 1.0) / (max_rank - 1.0)) if max_rank > 1 else 0.0
        confidence = normalized_rank if self.normalize else clamp01(effective_rank / max_rank)

        return ScoreResult(
            method=self.name,
            signal_type="generation_confidence",
            score=confidence,
            risk=clamp01(1.0 - confidence),
            level="response",
            raw={
                "spectral_entropy": spectral_entropy,
                "effective_rank": effective_rank,
                "normalized_effective_rank": normalized_rank,
                "max_rank": max_rank,
                "normalize": self.normalize,
                "singular_values": singular_values.tolist(),
            },
            explanation="Spectral entropy / effective rank of the hidden-state matrix.",
            assumptions=[
                "A low effective rank indicates representation collapse and higher hallucination risk.",
                "Hidden states are collected under a fixed layer and pooling protocol.",
            ],
            warnings=[
                "The reference implementation reports H rather than exp(H); both are exposed here.",
                "The normalized mapping is for display; calibrate a threshold on validation data.",
            ],
            provenance={"normalize": self.normalize},
        )
