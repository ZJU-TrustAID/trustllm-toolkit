from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

ScoreLevel = Literal["response", "sentence", "claim", "token"]
SignalType = Literal[
    "generation_confidence",
    "self_consistency",
    "evidence_support",
    "self_reported_confidence",
    "judge_assessment",
]
UnitType = Literal["sentence", "claim"]
UnitLabel = Literal["low_risk", "medium_risk", "high_risk", "unknown"]


@dataclass
class Evidence:
    source: str | None = None
    title: str | None = None
    url: str | None = None
    snippet: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ClaimResult:
    claim: str
    label: Literal["supported", "contradicted", "not_supported", "irrelevant", "unknown"] = "unknown"
    score: float | None = None
    evidence: list[Evidence] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class UnitResult:
    """A sentence- or claim-level detection result with per-method details."""

    text: str
    index: int
    unit_type: UnitType
    score: float
    risk: float
    label: UnitLabel = "unknown"
    method_scores: list[ScoreResult] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.score = max(0.0, min(1.0, float(self.score)))
        self.risk = max(0.0, min(1.0, float(self.risk)))


@dataclass
class DetectionExample:
    response: str
    prompt: str | None = None
    samples: list[str] | None = None
    reference_answer: str | list[str] | None = None
    token_logprobs: list[float] | None = None
    top_logprobs: list[dict[str, float]] | None = None
    top_logprob_entries: list[list[tuple[int, str, float]]] | None = None
    tokens: list[str] | None = None
    token_is_special: list[bool] | None = None
    hidden_states: Any | None = None
    units: list[str] | None = None
    unit_type: UnitType | None = None
    claims: list[str] | None = None
    retrieved_evidence: list[Evidence] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ScoreResult:
    method: str
    score: float
    risk: float
    signal_type: SignalType = "generation_confidence"
    raw: dict[str, Any] = field(default_factory=dict)
    level: ScoreLevel = "response"
    explanation: str | None = None
    assumptions: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    provenance: dict[str, Any] = field(default_factory=dict)
    calibrated_score: float | None = None
    calibrated_risk: float | None = None
    calibration: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.score = max(0.0, min(1.0, float(self.score)))
        self.risk = max(0.0, min(1.0, float(self.risk)))


@dataclass
class DetectionResult:
    score: float
    risk: float
    is_hallucinated: bool
    threshold: float
    method_scores: list[ScoreResult] = field(default_factory=list)
    signal_scores: dict[SignalType, list[ScoreResult]] = field(default_factory=dict)
    units: list[UnitResult] = field(default_factory=list)
    claims: list[ClaimResult] = field(default_factory=list)
    evidence: list[Evidence] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    calibration: dict[str, Any] = field(default_factory=dict)
    cost: dict[str, Any] = field(default_factory=dict)
    granularity: str | None = None

    def __post_init__(self) -> None:
        self.score = max(0.0, min(1.0, float(self.score)))
        self.risk = max(0.0, min(1.0, float(self.risk)))
