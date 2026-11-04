from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from hallucination.detection.core.results import DetectionExample, Evidence, UnitType


@dataclass
class DatasetRecord:
    """One annotated example used by examples, benchmarks, or user experiments."""

    id: str
    response: str
    prompt: str | None = None
    samples: list[str] | None = None
    reference_answer: str | list[str] | None = None
    units: list[str] | None = None
    unit_labels: list[bool | str] | None = None
    unit_type: UnitType | None = None
    response_label: bool | str | None = None
    split: str | None = None
    label_source: str | None = None
    claims: list[str] | None = None
    claim_labels: list[str] | None = None
    evidence: list[Evidence] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("DatasetRecord.id must not be empty")
        if self.units is not None and self.unit_labels is not None and len(self.units) != len(self.unit_labels):
            raise ValueError("units and unit_labels must have the same length")
        if self.claims is not None and self.claim_labels is not None and len(self.claims) != len(self.claim_labels):
            raise ValueError("claims and claim_labels must have the same length")
        if self.unit_type is None and self.units is not None:
            self.unit_type = "sentence"

    def to_detection_example(self) -> DetectionExample:
        """Convert the record without discarding labels kept for evaluation."""

        return DetectionExample(
            prompt=self.prompt,
            response=self.response,
            samples=self.samples,
            reference_answer=self.reference_answer,
            units=self.units,
            unit_type=self.unit_type,
            claims=self.claims,
            retrieved_evidence=self.evidence or None,
            metadata={**self.metadata, "dataset_record_id": self.id},
        )
