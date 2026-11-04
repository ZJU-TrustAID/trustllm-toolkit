from __future__ import annotations

import math

import numpy as np

from hallucination.detection.core.base import BaseScorer
from hallucination.detection.core.results import DetectionExample, ScoreResult
from hallucination.detection.core.thresholds import clamp01


class EigenScoreScorer(BaseScorer):
    """EigenScore over hidden states collected from repeated generations.

    The INSIDE paper scores the eigenvalue spectrum of the covariance matrix of
    hidden representations. This implementation uses the equivalent non-zero
    spectrum of the Gram matrix (cheaper when ``n_samples << hidden_dim``) and
    exposes both official variants:

    - ``mean_log``: mean ``log10(lambda + alpha)`` (INSIDE main variant);
    - ``logdet``: ``sum(log(lambda + alpha))`` (variant used by the Effective
      Rank reference implementation).

    ``risk`` is a monotone sigmoid of the raw score. It is **not calibrated**;
    fit a threshold on validation data before making decisions.
    """

    name = "eigenscore"
    requires = {"hidden_states"}

    def __init__(self, variant: str = "mean_log", *, alpha: float = 1e-3, feature_clip: bool = False) -> None:
        if variant not in {"mean_log", "logdet"}:
            raise ValueError("variant must be 'mean_log' or 'logdet'")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        self.variant = variant
        self.alpha = alpha
        self.feature_clip = feature_clip

    def score(self, example: DetectionExample) -> ScoreResult:
        self.validate(example)
        states = np.asarray(example.hidden_states, dtype=float)
        if states.ndim != 2:
            raise ValueError("hidden_states must be a 2D array shaped [n_samples, hidden_dim]")
        if states.shape[0] < 2:
            raise ValueError("EigenScore requires at least two hidden-state vectors from independent generations")

        centered = states - states.mean(axis=0, keepdims=True)
        if self.feature_clip:
            u, _, vt = np.linalg.svd(centered, full_matrices=False)
            first = vt[0]
            centered = centered - np.outer(centered @ first, first)

        gram = centered @ centered.T / (states.shape[0] - 1)
        eigenvalues = np.linalg.eigvalsh(gram)
        eigenvalues = np.clip(eigenvalues, 0.0, None) + self.alpha
        if self.variant == "mean_log":
            raw_score = float(np.mean(np.log10(eigenvalues)))
        else:
            raw_score = float(np.sum(np.log(eigenvalues)))

        risk = clamp01(1.0 / (1.0 + math.exp(-raw_score)))
        return ScoreResult(
            method=self.name,
            signal_type="generation_confidence",
            score=clamp01(1.0 - risk),
            risk=risk,
            level="response",
            raw={
                "raw_score": raw_score,
                "variant": self.variant,
                "alpha": self.alpha,
                "feature_clip": self.feature_clip,
                "eigenvalues": sorted(float(value) for value in eigenvalues),
                "num_samples": int(states.shape[0]),
                "hidden_dim": int(states.shape[1]),
                "risk_mapping": "sigmoid(raw_score)",
            },
            explanation="Eigenvalue-spectrum diversity of hidden states across independent generations.",
            assumptions=[
                "Hidden states come from the same layer and token position across independent generations.",
                "Higher spectral diversity is treated as higher hallucination risk.",
            ],
            warnings=[
                "The sigmoid risk mapping is uncalibrated; select a threshold on validation data.",
                "Layer choice, pooling, and model architecture change the raw spectrum.",
            ],
            provenance={"variant": self.variant, "alpha": self.alpha, "feature_clip": self.feature_clip},
        )
