from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class PipelineConfig:
    name: str
    scorers: list[dict[str, Any]] = field(default_factory=list)
    response_scorers: list[dict[str, Any]] = field(default_factory=list)
    sentence_scorers: list[dict[str, Any]] = field(default_factory=list)
    threshold: float = 0.5
    aggregation: str = "mean"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError("threshold must be in [0, 1]")
        if self.aggregation not in {"mean"}:
            raise ValueError("only mean aggregation is supported by the scaffold")
        configured = self.scorers or self.response_scorers or self.sentence_scorers
        if not configured:
            raise ValueError("at least one scorer must be configured")

    @classmethod
    def from_yaml(cls, path: str | Path) -> PipelineConfig:
        import yaml

        with Path(path).open("r", encoding="utf-8") as f:
            payload = yaml.safe_load(f)
        if not isinstance(payload, dict):
            raise ValueError("pipeline config must be a mapping")
        return cls(**payload)

    @property
    def all_scorers(self) -> list[dict[str, Any]]:
        return [*self.scorers, *self.response_scorers, *self.sentence_scorers]

    def enabled_scorers(self) -> list[str]:
        return [str(item.get("name")) for item in self.all_scorers if item.get("enabled", True)]
