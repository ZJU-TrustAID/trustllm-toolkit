from __future__ import annotations

from dataclasses import dataclass

from hallucination.detection.backends.retrieval import RetrieverBackend
from hallucination.detection.core.base import BaseScorer
from hallucination.detection.core.results import (
    ClaimResult,
    DetectionExample,
    Evidence,
    ScoreResult,
)
from hallucination.detection.evidence.claims import ClaimDecomposer
from hallucination.detection.evidence.verification import ClaimVerifier


@dataclass
class EvidenceAssessment:
    score: ScoreResult
    claims: list[ClaimResult]
    evidence: list[Evidence]


class FActScoreScorer(BaseScorer):
    """FActScore-style claim verification with explicit unknown verdicts."""

    name = "factscore"
    signal_type = "evidence_support"

    def __init__(
        self,
        verifier: ClaimVerifier,
        *,
        retriever: RetrieverBackend | None = None,
        decomposer: ClaimDecomposer | None = None,
        evidence_per_claim: int = 5,
        unknown_policy: str = "unsupported",
    ) -> None:
        if evidence_per_claim < 1:
            raise ValueError("evidence_per_claim must be positive")
        if unknown_policy not in {"unsupported", "contradicted"}:
            raise ValueError("unknown_policy must be 'unsupported' or 'contradicted'")
        self.verifier = verifier
        self.retriever = retriever
        self.decomposer = decomposer or ClaimDecomposer()
        self.evidence_per_claim = evidence_per_claim
        self.unknown_policy = unknown_policy

    def score(self, example: DetectionExample) -> ScoreResult:
        return self.assess(example).score

    def assess(self, example: DetectionExample) -> EvidenceAssessment:
        claims = self.decomposer.decompose(example)
        claim_results: list[ClaimResult] = []
        all_evidence: list[Evidence] = []
        for claim in claims:
            candidates = self._retrieve(claim, example)
            verdict = self.verifier.verify(claim, candidates)
            claim_results.append(
                ClaimResult(
                    claim=claim,
                    label=verdict.label,
                    score=verdict.score,
                    evidence=verdict.evidence,
                    metadata={**verdict.metadata, "candidate_count": len(candidates)},
                )
            )
            all_evidence.extend(verdict.evidence)

        total = len(claim_results)
        if total == 0:
            supported = contradicted = unknown = 0.0
            unsupported = 1.0
            risk = 1.0
        else:
            supported = sum(item.label == "supported" for item in claim_results) / total
            contradicted = sum(item.label == "contradicted" for item in claim_results) / total
            unknown = sum(item.label == "unknown" for item in claim_results) / total
            unsupported = 1.0 - supported
            risk = unsupported if self.unknown_policy == "unsupported" else contradicted
        warnings = ["NLI verification and retrieval quality limit the factual validity of this result."]
        if unknown and self.unknown_policy == "unsupported":
            warnings.append(
                "Unknown claims are counted as not supported. Set unknown_policy='contradicted' "
                "to score only explicit contradictions."
            )
        if total == 0:
            warnings.append("No claims were extracted; the response is treated as unsupported.")
        score = ScoreResult(
            method=self.name,
            signal_type=self.signal_type,
            score=supported,
            risk=risk,
            level="claim",
            raw={
                "claim_count": total,
                "supported_ratio": supported,
                "contradicted_ratio": contradicted,
                "unknown_ratio": unknown,
                "unsupported_ratio": unsupported,
                "unknown_policy": self.unknown_policy,
                "claims": [
                    {"claim": item.claim, "label": item.label, "score": item.score, "evidence_count": len(item.evidence)}
                    for item in claim_results
                ],
            },
            assumptions=[
                "Only retrieved evidence is considered; unknown is not treated as contradiction."
                if self.unknown_policy == "contradicted"
                else "Only retrieved evidence is considered; unknown counts as not supported."
            ],
            warnings=warnings,
            provenance={
                "retriever": type(self.retriever).__name__ if self.retriever else "provided_evidence",
                "unknown_policy": self.unknown_policy,
            },
        )
        return EvidenceAssessment(score=score, claims=claim_results, evidence=_deduplicate_evidence(all_evidence))

    def _retrieve(self, claim: str, example: DetectionExample) -> list[Evidence]:
        if self.retriever is not None:
            return self.retriever.retrieve(claim, k=self.evidence_per_claim)
        if example.retrieved_evidence is not None:
            return example.retrieved_evidence[: self.evidence_per_claim]
        raise ValueError("factscore requires a retriever or example.retrieved_evidence")


def _deduplicate_evidence(items: list[Evidence]) -> list[Evidence]:
    seen: set[tuple[str | None, str | None, str | None]] = set()
    unique: list[Evidence] = []
    for item in items:
        key = (item.source, item.title, item.snippet)
        if key not in seen:
            seen.add(key)
            unique.append(item)
    return unique
