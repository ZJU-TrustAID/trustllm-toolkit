from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from hallucination.detection.core.base import BaseScorer
from hallucination.detection.core.results import DetectionExample, DetectionResult
from hallucination.detection.pipelines.ensemble import DetectionPipeline
from hallucination.detection.pipelines.evidence_verification import (
    EvidenceVerificationPipeline,
)
from hallucination.detection.pipelines.short_answer import BlackBoxSamplingPipeline


def build_blackbox_long_form_pipeline(scorers: Sequence[BaseScorer], *, threshold: float = 0.5) -> DetectionPipeline:
    """Build a response-level pipeline for compatible long-form scorers.

    Use :class:`LongFormVerificationPipeline` when sentence-level sampled
    consistency and external evidence verification are both required.
    """

    if not scorers:
        raise ValueError("Provide at least one implemented sentence- or claim-level scorer.")
    return DetectionPipeline(scorers, threshold=threshold)


@dataclass
class LongFormDetectionResult:
    """Separate outputs for model self-consistency and external evidence."""

    self_consistency: DetectionResult
    evidence_support: DetectionResult | None = None


class LongFormVerificationPipeline:
    """Compose optional evidence verification without combining score meanings."""

    def __init__(
        self,
        sampling_pipeline: BlackBoxSamplingPipeline,
        *,
        evidence_pipeline: EvidenceVerificationPipeline | None = None,
    ) -> None:
        self.sampling_pipeline = sampling_pipeline
        self.evidence_pipeline = evidence_pipeline

    def detect(self, example: DetectionExample) -> LongFormDetectionResult:
        self_consistency = self.sampling_pipeline.detect(example)
        evidence_support = self.evidence_pipeline.detect(example) if self.evidence_pipeline else None
        return LongFormDetectionResult(self_consistency=self_consistency, evidence_support=evidence_support)

    def detect_prompt(
        self,
        prompt: str,
        *,
        response: str | None = None,
        claims: list[str] | None = None,
        **generate_kwargs: object,
    ) -> LongFormDetectionResult:
        self_consistency = self.sampling_pipeline.detect_prompt(prompt, response=response, **generate_kwargs)
        if self.evidence_pipeline is None:
            return LongFormDetectionResult(self_consistency=self_consistency)
        verified_response = str(self_consistency.metadata["response"])
        evidence_support = self.evidence_pipeline.detect(
            DetectionExample(prompt=prompt, response=verified_response, claims=claims)
        )
        return LongFormDetectionResult(self_consistency=self_consistency, evidence_support=evidence_support)
