from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class GenerationRequest:
    prompt: str
    context: str | None = None
    max_new_tokens: int = 128
    do_sample: bool = False
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 0
    repetition_penalty: float = 1.0
    seed: int = 0
    stop: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.prompt:
            raise ValueError("prompt must be non-empty")
        if self.max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive")
        if self.temperature < 0:
            raise ValueError("temperature must be >= 0")
        if not 0 < self.top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")
        if self.top_k < 0:
            raise ValueError("top_k must be >= 0")
        if self.repetition_penalty <= 0:
            raise ValueError("repetition_penalty must be positive")


@dataclass
class GenerationOutput:
    text: str
    token_ids: list[int]
    generated_ids: list[int]
    finish_reason: str = "length"
    trace: dict[str, Any] = field(default_factory=dict)
    cost: dict[str, Any] = field(default_factory=dict)


@dataclass
class MitigationReport:
    method: str
    prompt: str
    response: str
    baseline: str | None = None
    context: str | None = None
    trace: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    cost: dict[str, Any] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)
    schema_version: str = "hm/report/v1"

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(asdict(self))

    def to_json(self, path: str | Path | None = None) -> str:
        payload = json.dumps(self.to_dict(), ensure_ascii=False, indent=2)
        if path is not None:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            Path(path).write_text(payload + "\n", encoding="utf-8")
        return payload

    def save(self, path: str | Path) -> None:
        self.to_json(path)

    def to_markdown(self) -> str:
        lines = [
            f"# Hallucination mitigation: `{self.method}`",
            "",
            f"**Prompt**: {self.prompt}",
        ]
        if self.context:
            lines += ["", f"**Context**: {self.context}"]
        if self.baseline is not None:
            lines += ["", "## Baseline", "", self.baseline or "*(empty)*"]
        lines += ["", "## Mitigated", "", self.response or "*(empty)*"]
        if self.trace:
            lines += [
                "",
                "## Trace",
                "",
                "```json",
                json.dumps(_jsonable(self.trace), ensure_ascii=False, indent=2),
                "```",
            ]
        if self.cost:
            lines += [
                "",
                "## Cost",
                "",
                "```json",
                json.dumps(_jsonable(self.cost), ensure_ascii=False, indent=2),
                "```",
            ]
        if self.warnings:
            lines += ["", "## Warnings", ""]
            lines += [f"- {warning}" for warning in self.warnings]
        if self.provenance:
            lines += [
                "",
                "## Provenance",
                "",
                "```json",
                json.dumps(_jsonable(self.provenance), ensure_ascii=False, indent=2),
                "```",
            ]
        return "\n".join(lines) + "\n"


@dataclass
class ComparisonReport:
    prompt: str
    reports: list[MitigationReport]
    context: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "hm/comparison/v1",
            "prompt": self.prompt,
            "context": self.context,
            "reports": [report.to_dict() for report in self.reports],
        }

    def to_json(self, path: str | Path | None = None) -> str:
        payload = json.dumps(self.to_dict(), ensure_ascii=False, indent=2)
        if path is not None:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            Path(path).write_text(payload + "\n", encoding="utf-8")
        return payload

    def to_markdown(self) -> str:
        lines = ["# Method comparison", "", f"**Prompt**: {self.prompt}"]
        if self.context:
            lines += ["", f"**Context**: {self.context}"]
        lines += [
            "",
            "| method | baseline | mitigated | warnings |",
            "|---|---|---|---|",
        ]
        for report in self.reports:
            baseline = (report.baseline or "").replace("|", "\\|").replace("\n", " ")
            mitigated = (report.response or "").replace("|", "\\|").replace("\n", " ")
            warnings = "; ".join(report.warnings).replace("|", "\\|")
            lines.append(f"| {report.method} | {baseline} | {mitigated} | {warnings} |")
        return "\n".join(lines) + "\n"


def _jsonable(value: Any) -> Any:
    try:
        import numpy as np

        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
    except ImportError:
        pass
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value
