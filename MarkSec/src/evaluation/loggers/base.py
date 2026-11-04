# src/evaluation/loggers/base.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional, Protocol, TYPE_CHECKING

if TYPE_CHECKING:
    from src.evaluation.result_schema import SampleRecord

@dataclass(frozen=True)
class RunMeta:
    run_name: str
    run_id: Optional[str] = None
    config: Optional[Dict[str, Any]] = None
    output_dir: Optional[str] = None

@dataclass(frozen=True)
class PhaseMeta:
    name: str                 # "generate" | "evaluate" | "detection" | ...
    index: int = 0            # phase counter if needed

class ExperimentLogger(Protocol):
    """Runner only depends on this protocol, not on any backend library."""

    def on_run_start(self, meta: RunMeta) -> None: ...
    def on_phase_start(self, meta: PhaseMeta) -> None: ...
    def log_sample(self, i: int, record: SampleRecord, *, attack_duration: float = 0.0) -> None: ...
    def on_phase_end(self, meta: PhaseMeta) -> None: ...
    def on_run_end(self, summary: Dict[str, Any]) -> None: ...
    def close(self) -> None: ...
