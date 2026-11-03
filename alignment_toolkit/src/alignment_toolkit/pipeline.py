"""Resolved alignment pipeline objects."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class StageSpec:
    stage_id: str
    method: str
    backend: str
    dataset: str
    input: dict[str, Any]
    training: dict[str, Any]


@dataclass(frozen=True)
class PipelineSpec:
    name: str
    models: dict[str, Any]
    stages: tuple[StageSpec, ...]
    evaluation: dict[str, Any]
    output: dict[str, Any]
    raw: dict[str, Any]


class AlignmentExperiment:
    def __init__(self, spec: PipelineSpec, *, runner=None):
        self.spec = spec
        self.runner = runner

    def dry_run(self) -> PipelineSpec:
        return self.spec

    def run(self):
        if self.runner is None:
            raise RuntimeError("No runner was supplied; use dry-run or pass TRLRunner to run training")
        return self.runner.run(self.spec)
