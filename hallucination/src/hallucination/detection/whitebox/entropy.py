from __future__ import annotations

import math
from statistics import mean

from hallucination.detection.core.base import BaseScorer
from hallucination.detection.core.results import DetectionExample, ScoreResult
from hallucination.detection.core.thresholds import clamp01
from hallucination.detection.utils.math import entropy_from_probs


class TokenEntropyScorer(BaseScorer):
    """Token-level entropy over the available next-token distribution.

    ``top_logprobs`` is usually a truncated top-k distribution. The scorer
    therefore records ``distribution_coverage`` (the retained probability mass)
    and normalizes entropy by ``log(k)`` only when ``k > 1``.

    A step that exposes only one candidate is not evidence of confidence: the
    backend truncated the distribution, so the true entropy is unknown. The
    default ``single_candidate_policy="skip"`` excludes such steps; if every
    step is skipped the scorer raises instead of reporting zero risk.
    ``single_candidate_policy="zero"`` preserves the old behaviour for callers
    that explicitly want it.
    """

    name = "token_entropy"

    def __init__(
        self,
        aggregation: str = "mean",
        *,
        min_coverage: float = 0.0,
        single_candidate_policy: str = "skip",
    ) -> None:
        if aggregation not in {"mean", "max", "p95"}:
            raise ValueError("aggregation must be 'mean', 'max', or 'p95'")
        if not 0.0 <= min_coverage <= 1.0:
            raise ValueError("min_coverage must be in [0, 1]")
        if single_candidate_policy not in {"skip", "zero"}:
            raise ValueError("single_candidate_policy must be 'skip' or 'zero'")
        self.aggregation = aggregation
        self.min_coverage = min_coverage
        self.single_candidate_policy = single_candidate_policy

    def score(self, example: DetectionExample) -> ScoreResult:
        if example.top_logprobs is None and example.top_logprob_entries is None:
            raise ValueError("token_entropy requires top_logprobs or top_logprob_entries")

        entropies: list[float] = []
        normalized_entropies: list[float] = []
        coverages: list[float] = []
        single_candidate_steps = 0
        skipped_steps = 0

        steps = self._steps(example)
        for step in steps:
            logprobs = [float(item[2]) for item in step]
            probs = [math.exp(value) for value in logprobs]
            total = sum(probs)
            if total <= 0:
                continue
            coverage = min(1.0, total)
            if len(probs) == 1:
                single_candidate_steps += 1
                if self.single_candidate_policy == "zero":
                    entropies.append(0.0)
                    normalized_entropies.append(0.0)
                    coverages.append(coverage)
                else:
                    skipped_steps += 1
                continue
            normalized_probs = [prob / total for prob in probs]
            entropy = entropy_from_probs(normalized_probs)
            max_entropy = math.log(len(normalized_probs))
            entropies.append(entropy)
            normalized_entropies.append(entropy / max_entropy if max_entropy > 0 else 0.0)
            coverages.append(coverage)

        if not normalized_entropies:
            raise ValueError(
                "token_entropy could not compute any step with more than one candidate; "
                "the backend exposed only one token per step. Provide top-k logprobs "
                "or set single_candidate_policy='zero' to keep the legacy behaviour."
            )

        uncertainty = self._aggregate(normalized_entropies)
        risk = clamp01(uncertainty)
        warnings: list[str] = []
        if single_candidate_steps:
            if self.single_candidate_policy == "skip":
                warnings.append(
                    f"{single_candidate_steps} step(s) exposed only one candidate and were skipped; "
                    "their entropy is unknown, not zero."
                )
            else:
                warnings.append(f"{single_candidate_steps} step(s) exposed only one candidate; their entropy is reported as 0.")
        if any(coverage < 0.99 for coverage in coverages):
            warnings.append("Top-k coverage is below 0.99 for at least one step; entropy is an approximation.")
        if self.min_coverage and mean(coverages) < self.min_coverage:
            warnings.append(f"Mean distribution coverage {mean(coverages):.3f} is below min_coverage={self.min_coverage}.")

        return ScoreResult(
            method=self.name,
            signal_type="generation_confidence",
            score=clamp01(1.0 - risk),
            risk=risk,
            level="token",
            raw={
                "entropies": entropies,
                "normalized_entropies": normalized_entropies,
                "distribution_coverage": coverages,
                "mean_coverage": mean(coverages),
                "min_coverage": min(coverages),
                "single_candidate_steps": single_candidate_steps,
                "skipped_single_candidate_steps": skipped_steps,
                "num_steps": len(normalized_entropies),
                "num_input_steps": len(steps),
                "aggregation": self.aggregation,
                "single_candidate_policy": self.single_candidate_policy,
            },
            explanation="Entropy over the available next-token distribution; normalized by log(k) when k > 1.",
            assumptions=["The provided top-k distribution is the best available approximation to the full next-token distribution."],
            warnings=warnings,
            provenance={
                "aggregation": self.aggregation,
                "min_coverage": self.min_coverage,
                "single_candidate_policy": self.single_candidate_policy,
            },
        )

    def _aggregate(self, values: list[float]) -> float:
        if self.aggregation == "max":
            return max(values)
        if self.aggregation == "p95":
            ordered = sorted(values)
            index = max(0, math.ceil(0.95 * len(ordered)) - 1)
            return ordered[index]
        return mean(values)

    @staticmethod
    def _steps(example: DetectionExample) -> list[list[tuple[int | None, str, float]]]:
        if example.top_logprob_entries is not None:
            return [
                [(int(token_id), str(text), float(logprob)) for token_id, text, logprob in step]
                for step in example.top_logprob_entries
            ]
        assert example.top_logprobs is not None
        return [
            [(None, str(text), float(logprob)) for text, logprob in step.items()]
            for step in example.top_logprobs
        ]
