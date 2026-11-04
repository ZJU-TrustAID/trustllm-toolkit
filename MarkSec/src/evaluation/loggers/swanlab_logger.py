# src/evaluation/loggers/swanlab_logger.py
from __future__ import annotations
from typing import Any, Dict, Optional

from src.evaluation.loggers.base import ExperimentLogger, RunMeta, PhaseMeta
from src.evaluation.loggers.flatten import flatten_dict
from src.evaluation.result_schema import SampleRecord
from src.util.log import get_logger

logger = get_logger()


class SwanLabExperimentLogger:
    def __init__(
        self, *, project: Optional[str] = None, run_name: Optional[str] = None, **init_kwargs
    ):
        import swanlab  # Only import here to avoid runner dependency

        self.swanlab = swanlab
        self.project = project
        self.run_name = run_name
        self.init_kwargs = init_kwargs
        self._inited = False

    def on_run_start(self, meta: RunMeta) -> None:
        if self._inited:
            return

        # Merge init_kwargs with meta info if needed
        # Priority: init_kwargs > meta > defaults
        config = meta.config if meta.config else {}

        try:
            self.swanlab.init(
                project=self.project,
                experiment_name=self.run_name or meta.run_name,
                config=config,
                **self.init_kwargs,
            )
            self._inited = True
            logger.info('SwanLab initialized.')
        except Exception as e:
            logger.warning(f'SwanLab initialization failed: {e}')

    def on_phase_start(self, meta: PhaseMeta) -> None:
        if not self._inited:
            return
        try:
            # phase/name expected to be float in user input, but here it is string.
            # Avoid logging phase/name as a chart metric if it's a string.
            # Instead, we can log phase index or just skip logging name as a metric.
            self.swanlab.log({'phase/index': meta.index})
        except Exception as e:
            logger.warning(f'SwanLab phase log failed: {e}')

    def log_sample(self, i: int, record: SampleRecord, *, attack_duration: float = 0.0) -> None:
        if not self._inited:
            return
        try:
            sw_log: Dict[str, Any] = {
                'sample/index': i,
                'attack_duration': float(attack_duration),
                'status': 1 if record.status == 'success' else 0,
            }

            # detection: Auto-collect detection_*
            for text_type in ['natural', 'watermark', 'attack', 'spoof']:
                det = getattr(record, f'detection_{text_type}', None)
                if det:
                    if det.z_score is not None:
                        try:
                            sw_log[f'z_score/{text_type}'] = float(det.z_score)
                        except (ValueError, TypeError):
                            pass
                    if det.prediction is not None:
                        sw_log[f'is_watermarked/{text_type}'] = int(det.prediction)

            # texts: Decide whether to use swanlab.Text inside logger
            if record.prompt:
                sw_log['text/prompt'] = self.swanlab.Text(record.prompt, caption=f'Sample {i}')
            if record.watermarked_text:
                sw_log['text/watermarked'] = self.swanlab.Text(
                    record.watermarked_text, caption=f'Sample {i}'
                )
            if record.attacked_text:
                sw_log['text/attacked'] = self.swanlab.Text(
                    record.attacked_text, caption=f'Sample {i}'
                )
            if record.spoof_text:
                sw_log['text/spoofed'] = self.swanlab.Text(record.spoof_text, caption=f'Sample {i}')

            # quality_metrics: flatten unified
            if record.quality_metrics:
                flat = flatten_dict(record.quality_metrics, sep='/')
                for k, v in flat.items():
                    if isinstance(v, (int, float)):
                        sw_log[f'quality/{k}'] = v

            self.swanlab.log(sw_log)
        except Exception as e:
            logger.warning(f'SwanLab logging failed: {e}')

    def on_phase_end(self, meta: PhaseMeta) -> None:
        pass

    def on_run_end(self, summary: Dict[str, Any]) -> None:
        if not self._inited:
            return
        # Log summary if needed
        try:
            flat = flatten_dict(summary, sep='/')
            payload = {
                f'summary/{k}': v for k, v in flat.items() if isinstance(v, (int, float, str))
            }
            self.swanlab.log(payload)
        except Exception as e:
            logger.warning(f'SwanLab summary log failed: {e}')

    def close(self) -> None:
        if self._inited:
            try:
                self.swanlab.finish()
                self._inited = False
            except Exception as e:
                logger.warning(f'SwanLab finish failed: {e}')
