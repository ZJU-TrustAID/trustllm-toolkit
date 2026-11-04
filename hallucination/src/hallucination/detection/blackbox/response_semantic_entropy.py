from __future__ import annotations

import math
from collections import Counter

from hallucination.detection.backends.nli import NLIBackend, NLIProbabilities
from hallucination.detection.core.base import BaseScorer
from hallucination.detection.core.results import DetectionExample, ScoreResult
from hallucination.detection.core.thresholds import clamp01, confidence_to_risk

EquivalenceMode = str  # "official" | "strict" | "threshold"
_NEUTRAL = "neutral"
_CONTRADICTION = "contradiction"
_ENTAILMENT = "entailment"


def relation_label(relation: NLIProbabilities) -> str:
    """Return the arg-max label of an NLI relation, treating all-zero output as neutral."""

    values = {
        _ENTAILMENT: float(relation.entailment),
        _CONTRADICTION: float(relation.contradiction),
        _NEUTRAL: float(relation.neutral),
    }
    if max(values.values()) <= 0.0:
        return _NEUTRAL
    return max(values, key=values.get)


class ResponseSemanticEntropyScorer(BaseScorer):
    """Response-level semantic entropy over a set of sampled answers.

    Implements the textbook protocol and the official ``semantic_uncertainty``
    clustering rule:

    1. start from the original response and its independent samples;
    2. greedily assign a semantic cluster id using bidirectional NLI equivalence;
    3. compute ``H = -sum p_c log_base p_c`` over cluster frequencies;
    4. report raw entropy and a normalized confidence ``1 - H / log_base(n)``.

    The official implementation clusters generated answers and uses a
    three-way equivalence rule (no contradiction and not both neutral). This
    scorer supports that rule plus a strict bidirectional-entailment variant and
    a probability-threshold variant for teaching.
    """

    name = "response_semantic_entropy"
    requires = {"samples"}
    signal_type = "self_consistency"

    def __init__(
        self,
        nli_backend: NLIBackend,
        *,
        include_response: bool = True,
        equivalence: EquivalenceMode = "official",
        entailment_threshold: float = 0.5,
        contradiction_threshold: float = 0.5,
        base: float = 2.0,
    ) -> None:
        if equivalence not in {"official", "strict", "threshold"}:
            raise ValueError("equivalence must be 'official', 'strict', or 'threshold'")
        if base not in {2.0, math.e}:
            raise ValueError("base must be 2.0 (bits) or math.e (nats)")
        if not 0.0 <= entailment_threshold <= 1.0 or not 0.0 <= contradiction_threshold <= 1.0:
            raise ValueError("thresholds must be in [0, 1]")
        self.nli_backend = nli_backend
        self.include_response = include_response
        self.equivalence = equivalence
        self.entailment_threshold = entailment_threshold
        self.contradiction_threshold = contradiction_threshold
        self.base = base

    def score(self, example: DetectionExample) -> ScoreResult:
        self.validate(example)
        assert example.samples is not None
        samples = [sample for sample in example.samples if sample.strip()]
        if not samples:
            raise ValueError("response_semantic_entropy requires at least one non-empty sample")
        candidates = [example.response, *samples] if self.include_response else samples
        if len(candidates) < 2:
            raise ValueError("response_semantic_entropy requires at least two candidate responses")

        pairs = self._pair_relations(candidates)
        cluster_ids = self._greedy_clusters(candidates, pairs)
        counts = Counter(cluster_ids)
        probabilities = [count / len(candidates) for count in counts.values()]
        entropy = self._entropy(probabilities)
        max_entropy = math.log(len(candidates), self.base)
        normalized_entropy = clamp01(entropy / max_entropy) if max_entropy > 0 else 0.0
        clusters = [
            [candidates[index] for index, value in enumerate(cluster_ids) if value == cluster_id]
            for cluster_id in counts
        ]

        return ScoreResult(
            method=self.name,
            signal_type=self.signal_type,
            score=confidence_to_risk(normalized_entropy),
            risk=normalized_entropy,
            level="response",
            raw={
                "num_candidates": len(candidates),
                "num_samples": len(samples),
                "include_response": self.include_response,
                "equivalence": self.equivalence,
                "base": "bits" if self.base == 2.0 else "nats",
                "semantic_entropy": entropy,
                "normalized_semantic_entropy": normalized_entropy,
                "num_semantic_sets": len(clusters),
                "cluster_ids": cluster_ids,
                "clusters": clusters,
                "cluster_probabilities": probabilities,
                "pair_labels": [
                    {
                        "left": left,
                        "right": right,
                        "forward": relation_label(pair[0]),
                        "backward": relation_label(pair[1]),
                        "forward_probabilities": {
                            "entailment": pair[0].entailment,
                            "contradiction": pair[0].contradiction,
                            "neutral": pair[0].neutral,
                        },
                        "backward_probabilities": {
                            "entailment": pair[1].entailment,
                            "contradiction": pair[1].contradiction,
                            "neutral": pair[1].neutral,
                        },
                    }
                    for (left, right), pair in pairs.items()
                ],
            },
            explanation="Semantic entropy over full-response clusters; higher entropy means less semantic agreement across samples.",
            assumptions=[
                "The NLI backend approximates bidirectional entailment between full responses.",
                "Cluster probabilities are estimated from sample frequencies.",
            ],
            warnings=[
                "Low semantic entropy means the samples agree; it does not prove factual correctness.",
                "A model can repeat the same false answer across all samples.",
                "Cluster probabilities are frequency-based; token-probability-weighted semantic entropy is not implemented.",
            ],
            provenance={
                "nli_backend": getattr(self.nli_backend, "model_name", type(self.nli_backend).__name__),
                "equivalence": self.equivalence,
                "entailment_threshold": self.entailment_threshold,
                "contradiction_threshold": self.contradiction_threshold,
            },
        )

    def _pair_relations(self, candidates: list[str]) -> dict[tuple[int, int], tuple[NLIProbabilities, NLIProbabilities]]:
        pairs: dict[tuple[int, int], tuple[NLIProbabilities, NLIProbabilities]] = {}
        for left in range(len(candidates)):
            for right in range(left + 1, len(candidates)):
                pairs[(left, right)] = (
                    self.nli_backend(candidates[left], candidates[right]),
                    self.nli_backend(candidates[right], candidates[left]),
                )
        return pairs

    def _is_equivalent(self, pair: tuple[NLIProbabilities, NLIProbabilities]) -> bool:
        forward, backward = pair
        forward_label = relation_label(forward)
        backward_label = relation_label(backward)
        if self.equivalence == "strict":
            return forward_label == _ENTAILMENT and backward_label == _ENTAILMENT
        if self.equivalence == "threshold":
            return (
                forward.entailment >= self.entailment_threshold
                and backward.entailment >= self.entailment_threshold
                and forward.contradiction < self.contradiction_threshold
                and backward.contradiction < self.contradiction_threshold
            )
        return _CONTRADICTION not in (forward_label, backward_label) and (
            forward_label,
            backward_label,
        ) != (_NEUTRAL, _NEUTRAL)

    def _greedy_clusters(
        self,
        candidates: list[str],
        pairs: dict[tuple[int, int], tuple[NLIProbabilities, NLIProbabilities]],
    ) -> list[int]:
        size = len(candidates)
        cluster_ids = [-1] * size
        next_id = 0
        for left in range(size):
            if cluster_ids[left] != -1:
                continue
            cluster_ids[left] = next_id
            for right in range(left + 1, size):
                if cluster_ids[right] == -1 and self._is_equivalent(pairs[(left, right)]):
                    cluster_ids[right] = next_id
            next_id += 1
        return cluster_ids

    def _entropy(self, probabilities: list[float]) -> float:
        return -sum(prob * math.log(prob, self.base) for prob in probabilities if prob > 0.0)
