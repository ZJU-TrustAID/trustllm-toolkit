from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from hallucination.detection.core.results import (
    ClaimResult,
    DetectionResult,
    Evidence,
    ScoreResult,
    UnitResult,
)


def score_to_dict(score: ScoreResult) -> dict[str, Any]:
    return {
        "method": score.method,
        "signal_type": score.signal_type,
        "level": score.level,
        "score": score.score,
        "risk": score.risk,
        "calibrated_score": score.calibrated_score,
        "calibrated_risk": score.calibrated_risk,
        "explanation": score.explanation,
        "assumptions": list(score.assumptions),
        "warnings": list(score.warnings),
        "provenance": dict(score.provenance),
        "calibration": dict(score.calibration),
        "raw": score.raw,
    }


def unit_to_dict(unit: UnitResult) -> dict[str, Any]:
    return {
        "text": unit.text,
        "index": unit.index,
        "unit_type": unit.unit_type,
        "score": unit.score,
        "risk": unit.risk,
        "label": unit.label,
        "method_scores": [score_to_dict(score) for score in unit.method_scores],
        "metadata": unit.metadata,
    }


def claim_to_dict(claim: ClaimResult) -> dict[str, Any]:
    return {
        "claim": claim.claim,
        "label": claim.label,
        "score": claim.score,
        "evidence": [evidence_to_dict(item) for item in claim.evidence],
        "metadata": claim.metadata,
    }


def evidence_to_dict(evidence: Evidence) -> dict[str, Any]:
    return {
        "source": evidence.source,
        "title": evidence.title,
        "url": evidence.url,
        "snippet": evidence.snippet,
        "metadata": evidence.metadata,
    }


def result_to_dict(result: DetectionResult) -> dict[str, Any]:
    return {
        "score": result.score,
        "risk": result.risk,
        "is_hallucinated": result.is_hallucinated,
        "threshold": result.threshold,
        "granularity": result.granularity,
        "method_scores": [score_to_dict(score) for score in result.method_scores],
        "signal_scores": {
            signal_type: [score_to_dict(score) for score in scores]
            for signal_type, scores in result.signal_scores.items()
        },
        "units": [unit_to_dict(unit) for unit in result.units],
        "claims": [claim_to_dict(claim) for claim in result.claims],
        "evidence": [evidence_to_dict(item) for item in result.evidence],
        "calibration": result.calibration,
        "cost": result.cost,
        "metadata": result.metadata,
    }


@dataclass
class DetectionReport:
    """Uniform, serializable output for one or more detection runs."""

    results: list[DetectionResult]
    prompts: list[str | None] = field(default_factory=list)
    responses: list[str] = field(default_factory=list)
    samples: list[list[str]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.results:
            raise ValueError("DetectionReport requires at least one DetectionResult")
        if self.prompts and len(self.prompts) != len(self.results):
            raise ValueError("prompts must align with results")
        if self.responses and len(self.responses) != len(self.results):
            raise ValueError("responses must align with results")
        if self.samples and len(self.samples) != len(self.results):
            raise ValueError("samples must align with results")

    @classmethod
    def from_result(
        cls,
        result: DetectionResult,
        *,
        prompt: str | None = None,
        response: str | None = None,
        samples: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> DetectionReport:
        return cls(
            results=[result],
            prompts=[prompt],
            responses=[response if response is not None else str(result.metadata.get("response", ""))],
            samples=[samples or list(result.metadata.get("samples", []))],
            metadata=metadata or {},
        )

    @property
    def result(self) -> DetectionResult:
        if len(self.results) != 1:
            raise ValueError("this report contains multiple results; iterate `results` instead")
        return self.results[0]

    def summary(self) -> dict[str, Any]:
        rows = []
        for index, result in enumerate(self.results):
            rows.append(
                {
                    "index": index,
                    "prompt": self.prompts[index] if self.prompts else None,
                    "response": self.responses[index] if self.responses else None,
                    "score": result.score,
                    "risk": result.risk,
                    "is_hallucinated": result.is_hallucinated,
                    "threshold": result.threshold,
                    "methods": {score.method: score.risk for score in result.method_scores},
                    "unit_count": len(result.units),
                }
            )
        return {"count": len(self.results), "rows": rows, "metadata": self.metadata}

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary": self.summary(),
            "results": [result_to_dict(result) for result in self.results],
            "prompts": self.prompts,
            "responses": self.responses,
            "samples": self.samples,
            "metadata": self.metadata,
        }

    def to_json(self, path: str | Path | None = None) -> str:
        payload = json.dumps(self.to_dict(), ensure_ascii=False, indent=2, default=_json_default)
        if path is not None:
            Path(path).write_text(payload, encoding="utf-8")
        return payload

    def to_markdown(self) -> str:
        result = self.result
        lines = [
            "# Hallucination detection report",
            "",
            f"- **risk**: {result.risk:.4f}",
            f"- **score**: {result.score:.4f}",
            f"- **is_hallucinated** (threshold={result.threshold}): {result.is_hallucinated}",
            "",
            "| method | signal | level | risk | calibrated risk | score |",
            "|---|---|---|---|---|---|",
        ]
        for score in result.method_scores:
            calibrated = "-" if score.calibrated_risk is None else f"{score.calibrated_risk:.4f}"
            lines.append(
                f"| {score.method} | {score.signal_type} | {score.level} | {score.risk:.4f} | {calibrated} | {score.score:.4f} |"
            )
        if result.units:
            lines.extend(["", "## Sentence-level results", "", "| # | sentence | risk |", "|---|---|---|"])
            for unit in result.units:
                lines.append(f"| {unit.index} | {unit.text} | {unit.risk:.4f} |")
        warnings = sorted({warning for score in result.method_scores for warning in score.warnings})
        if warnings:
            lines.extend(["", "## Warnings", ""])
            lines.extend(f"- {warning}" for warning in warnings)
        return "\n".join(lines)

    def to_dataframe(self) -> Any:
        try:
            import pandas as pd
        except ImportError as exc:
            raise ImportError("Install pandas to use DetectionReport.to_dataframe()") from exc
        rows = []
        for index, result in enumerate(self.results):
            row: dict[str, Any] = {
                "prompt": self.prompts[index] if self.prompts else None,
                "response": self.responses[index] if self.responses else None,
                "score": result.score,
                "risk": result.risk,
                "is_hallucinated": result.is_hallucinated,
                "threshold": result.threshold,
            }
            for score in result.method_scores:
                row[f"risk_{score.method}"] = score.risk
                row[f"score_{score.method}"] = score.score
                if score.calibrated_risk is not None:
                    row[f"calibrated_risk_{score.method}"] = score.calibrated_risk
                    row[f"calibrated_score_{score.method}"] = score.calibrated_score
            rows.append(row)
        return pd.DataFrame(rows)


def _json_default(value: Any) -> Any:
    try:
        import numpy as np

        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
    except ImportError:
        pass
    if isinstance(value, set):
        return sorted(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")
