from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from hallucination.detection.backends.nli import NLIBackend
from hallucination.detection.core.results import Evidence


@dataclass(frozen=True)
class ClaimVerdict:
    label: str
    score: float
    evidence: list[Evidence] = field(default_factory=list)
    metadata: dict[str, object] = field(default_factory=dict)


class ClaimVerifier(Protocol):
    def verify(self, claim: str, evidence: list[Evidence]) -> ClaimVerdict:
        ...


class NLIEvidenceVerifier:
    """Classify a claim using the strongest NLI relation from retrieved text."""

    def __init__(
        self,
        nli_backend: NLIBackend,
        *,
        support_threshold: float = 0.5,
        contradiction_threshold: float = 0.5,
    ) -> None:
        self.nli_backend = nli_backend
        self.support_threshold = support_threshold
        self.contradiction_threshold = contradiction_threshold

    def verify(self, claim: str, evidence: list[Evidence]) -> ClaimVerdict:
        comparisons: list[dict[str, object]] = []
        for index, item in enumerate(evidence):
            premise = item.snippet or item.title or ""
            if not premise.strip():
                continue
            relation = self.nli_backend(premise, claim)
            comparisons.append(
                {
                    "evidence_index": index,
                    "entailment": relation.entailment,
                    "contradiction": relation.contradiction,
                    "neutral": relation.neutral,
                }
            )

        if not comparisons:
            return ClaimVerdict("unknown", 0.0, [], {"comparisons": comparisons})
        best_entailment = max(comparisons, key=lambda item: float(item["entailment"]))
        best_contradiction = max(comparisons, key=lambda item: float(item["contradiction"]))
        entailment = float(best_entailment["entailment"])
        contradiction = float(best_contradiction["contradiction"])
        metadata: dict[str, object] = {
            "comparisons": comparisons,
            "best_entailment": entailment,
            "best_contradiction": contradiction,
        }
        if entailment >= self.support_threshold and entailment >= contradiction:
            index = int(best_entailment["evidence_index"])
            return ClaimVerdict("supported", entailment, [evidence[index]], metadata)
        if contradiction >= self.contradiction_threshold:
            index = int(best_contradiction["evidence_index"])
            return ClaimVerdict("contradicted", contradiction, [evidence[index]], metadata)
        return ClaimVerdict("unknown", max(entailment, contradiction), [], metadata)
