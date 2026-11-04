import logging
from datetime import datetime
from pathlib import Path
import torch
import random
import numpy as np
from src.util.log import setup_logger
from src.util.env import configure_models_root
from src.evaluation.run_registry import resolve_run_binding

logger = logging.getLogger('MarkSec')

from src.config import ExperimentConfig


class ExperimentContext:
    """
    Manages the experiment environment, configuration, and logging setup.
    """

    def __init__(self, config: ExperimentConfig):
        self.config = config
        configure_models_root(self.config.models_root)

        self.experiment_id = None
        self.spec_id = None
        self.log_path = None

        self.device = self._get_best_free_gpu()
        self.timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
        self.experiment_name = self._generate_experiment_name()
        # Updating Pydantic model field requires a different approach if frozen,
        # but defaults are not frozen. However, it's better to treat config as immutable-ish.
        # But we need to store experiment_name back to config for logging/reporting.
        self.config.experiment_name = self.experiment_name
        self.backfill_run_binding = self._resolve_backfill_run_binding()

        self.output_dir = self._setup_output_dir()

        self._setup_logger()
        self.log_path = self._resolve_log_path()
        self._assign_default_data_path()
        # SwanLab init removed from here. Handled by ExperimentLogger.
        self._set_seed()

    def _generate_experiment_name(self) -> str:
        """
        Generates a descriptive experiment name from the core components in the config.
        Format: {watermark_name}_{attack_name}_{model_name}_{dataset_name}
        """
        wm_name = self.config.watermark
        attack_name = self.config.attack
        base_model_path = self.config.base_model
        model_name = Path(base_model_path).name
        dataset_name = self.config.dataset.name

        name_parts = [wm_name, attack_name, model_name, dataset_name]
        experiment_name = '_'.join(filter(None, name_parts))
        experiment_name = experiment_name.replace('/', '_')

        return experiment_name

    def _get_best_free_gpu(self) -> str:
        """Determines the best GPU to use."""
        if not torch.cuda.is_available():
            return 'cpu'
        best_gpu_index, max_free_memory = 0, 0
        device_count = torch.cuda.device_count()

        for i in range(device_count):
            free_memory, _ = torch.cuda.mem_get_info(i)
            if free_memory > max_free_memory:
                max_free_memory, best_gpu_index = free_memory, i

        free_gb = max_free_memory / 1024**3
        logger.info(
            f'Found {device_count} GPUs. Best physical GPU is: {best_gpu_index} ({free_gb:.2f} GiB free).'
        )

        logger.info(f'   Using physical GPU index: cuda:{best_gpu_index}')
        return f'cuda:{best_gpu_index}'

    def _setup_output_dir(self) -> Path:
        if self.backfill_run_binding is not None:
            return self._reuse_existing_output_dir(self.backfill_run_binding)

        base_dir = self.config.runner_settings.output_dir_base
        dataset_name = self.config.dataset.name
        attack_name = self.config.attack
        watermark_name = self.config.watermark

        # New output directory structure: output/result/dataset/attack/watermark/time
        output_dir = (
            Path(base_dir) / 'result' / dataset_name / attack_name / watermark_name / self.timestamp
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f'Experiment outputs will be saved to: {output_dir}')
        return output_dir

    def _resolve_backfill_run_binding(self) -> dict | None:
        if self.config.pipeline_mode != 'evaluate':
            return None

        dashboard = getattr(self.config, 'dashboard', None)
        if not isinstance(dashboard, dict):
            return None

        binding_run_id = str(
            dashboard.get('source_run_id') or dashboard.get('run_id') or ''
        ).strip()
        if not binding_run_id:
            return None

        binding = resolve_run_binding(self.config, binding_run_id)
        if binding is None:
            raise ValueError(
                f"Could not resolve run binding '{binding_run_id}' from sqlite for evaluate mode."
            )
        return binding

    def _reuse_existing_output_dir(self, binding: dict) -> Path:
        output_dir_raw = binding.get('output_dir')
        if not output_dir_raw:
            raise ValueError(
                f"run_id '{binding.get('run_id')}' does not have a reusable output directory."
            )

        output_dir = Path(str(output_dir_raw)).expanduser().resolve()
        timestamp = binding.get('timestamp')
        if isinstance(timestamp, str) and timestamp:
            self.timestamp = timestamp

        self._hydrate_dashboard_from_binding(binding)

        bound_data_path = binding.get('data_path')
        config_data_path = getattr(self.config, 'data_path', None)
        if config_data_path and bound_data_path:
            configured = Path(str(config_data_path)).expanduser().resolve()
            expected = Path(str(bound_data_path)).expanduser().resolve()
            if configured != expected:
                raise ValueError(
                    f"run_id '{binding.get('run_id')}' maps to data_path '{expected}', "
                    f"but config requested '{configured}'."
                )

        if not config_data_path:
            if bound_data_path:
                self.config.data_path = str(Path(str(bound_data_path)).expanduser().resolve())
            else:
                self.config.data_path = str((output_dir / 'generated_samples.jsonl').resolve())

        data_path = Path(str(self.config.data_path)).expanduser().resolve()
        if not data_path.exists():
            raise FileNotFoundError(
                f"Resolved data_path for run_id '{binding.get('run_id')}' was not found: {data_path}"
            )

        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f'Reusing existing result directory for evaluate mode: {output_dir}')
        logger.info(f'Reusing existing data_path from run registry: {self.config.data_path}')
        return output_dir

    def _hydrate_dashboard_from_binding(self, binding: dict) -> None:
        dashboard = getattr(self.config, 'dashboard', None)
        if not isinstance(dashboard, dict):
            dashboard = {}
        merged = dict(dashboard)
        is_source_binding = bool(str(merged.get('source_run_id') or '').strip())

        keys_to_hydrate = ('exp_id', 'template_id')
        if not is_source_binding:
            keys_to_hydrate = ('run_id', 'exp_id', 'template_id', 'attempt_index')

        for key in keys_to_hydrate:
            value = binding.get(key)
            if value is not None and (key not in merged or merged.get(key) in (None, '')):
                merged[key] = value
        self.config.dashboard = merged

    def _setup_logger(self):
        dataset_name = self.config.dataset.name
        attack_name = self.config.attack
        watermark_name = self.config.watermark

        log_dir = Path('output') / 'log' / dataset_name / attack_name / watermark_name
        setup_logger(
            name='MarkSec',
            log_dir=str(log_dir),
            experiment_config=self.config.model_dump(),
            console_log_level=logging.INFO,
        )

    @staticmethod
    def _resolve_log_path() -> str | None:
        run_logger = logging.getLogger('MarkSec')
        for handler in run_logger.handlers:
            filename = getattr(handler, 'baseFilename', None)
            if filename:
                return str(Path(filename).expanduser().resolve())
        return None

    def _assign_default_data_path(self):
        if self.config.data_path or self.config.pipeline_mode not in {'generate', 'all'}:
            return

        self.config.data_path = str(self.output_dir / 'generated_samples.jsonl')
        logger.info(f'Auto-generated data_path: {self.config.data_path}')

    def _set_seed(self):
        seed = self.config.runner_settings.seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
