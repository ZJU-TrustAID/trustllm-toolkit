# src/evaluation/loggers/composite.py
from __future__ import annotations
from typing import Dict, Any, List, TYPE_CHECKING

from src.evaluation.loggers.base import ExperimentLogger, RunMeta, PhaseMeta

if TYPE_CHECKING:
    from src.evaluation.result_schema import SampleRecord

class CompositeExperimentLogger:
    def __init__(self, loggers: List[ExperimentLogger]):
        self.loggers = loggers

    def on_run_start(self, meta: RunMeta) -> None:
        for lg in self.loggers:
            lg.on_run_start(meta)

    def on_phase_start(self, meta: PhaseMeta) -> None:
        for lg in self.loggers:
            lg.on_phase_start(meta)

    def log_sample(self, i: int, record: SampleRecord, *, attack_duration: float = 0.0) -> None:
        for lg in self.loggers:
            lg.log_sample(i, record, attack_duration=attack_duration)

    def on_phase_end(self, meta: PhaseMeta) -> None:
        for lg in self.loggers:
            lg.on_phase_end(meta)

    def on_run_end(self, summary: Dict[str, Any]) -> None:
        for lg in self.loggers:
            lg.on_run_end(summary)

    def close(self) -> None:
        for lg in self.loggers:
            lg.close()
