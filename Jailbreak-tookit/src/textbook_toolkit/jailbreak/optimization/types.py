"""Serializable inputs and measured outputs of local optimization runs."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

from ..types import Prompt, require_text


@dataclass(frozen=True)
class OptimizationRequest:
    query: str
    target: str
    initial_control: str = "! ! ! ! ! ! ! !"
    placement: Literal["suffix", "prefix", "replacement"] = "suffix"
    left_context: str = ""
    right_context: str = ""
    separator: str = "\n"

    def __post_init__(self) -> None:
        for name in ("query", "target", "initial_control"):
            require_text(getattr(self, name), name)
        if self.placement not in ("suffix", "prefix", "replacement"):
            raise ValueError("placement must be suffix, prefix or replacement.")
        for name in ("left_context", "right_context", "separator"):
            if not isinstance(getattr(self, name), str):
                raise TypeError(f"{name} must be a string.")

    def parts(self) -> tuple[str, str]:
        left = self.left_context + self.separator if self.left_context else ""
        right = self.separator + self.right_context if self.right_context else ""
        if self.placement == "suffix":
            left += self.query + self.separator
        elif self.placement == "prefix":
            right = self.separator + self.query + right
        return left, right

    def compose(self, control: str) -> str:
        left, right = self.parts()
        return left + control + right


@dataclass(frozen=True)
class TraceStep:
    iteration: int
    target_loss: float
    objective: float
    control: str
    terms: dict[str, float]
    candidates: int = 1


@dataclass(frozen=True)
class OptimizationResult:
    method: str
    request: OptimizationRequest
    prompt: Prompt
    control: str
    response: str
    input_ids: tuple[int, ...]
    response_ids: tuple[int, ...]
    target_ids: tuple[int, ...]
    initial_loss: float
    final_loss: float
    trace: tuple[TraceStep, ...]
    stop_reason: str
    config: dict[str, object]
    backend: dict[str, object]
    source: dict[str, str]

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["prompt"] = self.prompt.to_dict()
        return {"schema_version": 1, **data}

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2, allow_nan=False)

    def save(self, path: str | Path, *, overwrite: bool = False) -> None:
        with Path(path).open("w" if overwrite else "x", encoding="utf-8") as stream:
            stream.write(self.to_json() + "\n")
