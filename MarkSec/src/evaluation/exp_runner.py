import logging
import time
import json
import os
import gc
import torch

from src.evaluation.result_schema import SampleRecord, DetectionResult
from src.evaluation.dataset import DatasetRegistry
from src.evaluation.tools.metric_manager import (
    DetectionMetricManager,
    QualityMetricManager,
)
from src.evaluation.context import ExperimentContext
from src.evaluation.model_manager import ModelManager
from src.evaluation.component_factory import ComponentFactory
from src.evaluation.reporter import Reporter
from src.evaluation.run_registry import RunRegistry
from src.util.data import ChatFormatter
from src.config import ExperimentConfig
from src.evaluation.loggers.base import ExperimentLogger, RunMeta, PhaseMeta
from src.evaluation.loggers.null_logger import NullExperimentLogger
from src.util.text import strip_prompt
from src.exceptions.exceptions import (
    MissingDataPathError,
    DatasetInitializationError,
    EmptyDatasetError,
)

logger = logging.getLogger('MarkSec')


class ExperimentRunner:
    def __init__(self, config: ExperimentConfig, exp_logger: ExperimentLogger | None = None):
        """
        Initializes the ExperimentRunner from a single configuration object.

        Args:
            config (ExperimentConfig): The configuration object.
            exp_logger (ExperimentLogger, optional): Logger for experiment tracking.
        """
        self.context = ExperimentContext(config)
        self.config = self.context.config  # This is now an ExperimentConfig object
        self.device = self.context.device
        self.pipeline_mode = self.config.pipeline_mode

        # Initialize Experiment Logger
        self.exp_logger = exp_logger or NullExperimentLogger()

        self.dataset = None
        self.watermark_instance = None
        self.attack_instance = None
        self.quality_manager = None
        self.detection_manager = None
        self.chat_formatter = None

        self.results_log = []

        self.model_manager = ModelManager(self.device, self.config)
        self.component_factory = ComponentFactory(self.config, self.device, self.model_manager)
        self.run_registry = RunRegistry(self.config, self.context)
        self.context.run_registry = self.run_registry
        self.reporter = None  # Initialized after metric managers
        self.data_file_handle = None
        self.timing_summary = {
            'prepare_seconds': 0.0,
            'attack_seconds': 0.0,
            'total_seconds': 0.0,
        }
        self._pipeline_wall_start = None

    def _chat_template_mode(self) -> str:
        mode = str(getattr(self.config, 'chat_template_mode', 'auto') or 'auto').strip().lower()
        if mode not in {'auto', 'off', 'force'}:
            raise ValueError(
                f"Unsupported chat_template_mode '{mode}'. Use one of: auto, off, force."
            )
        return mode

    def _evaluate_quality_metrics(self) -> dict:
        return getattr(self.config.evaluation, 'quality_metrics', {}) or {}

    def _quality_comparison_names(self) -> list[str]:
        return list(getattr(self.config.evaluation, 'quality_comparison_names', None) or [])

    def _qsr_settings(self) -> tuple[str, float, list[float]]:
        qsr_cfg = getattr(self.config.evaluation, 'qsr', None)
        if qsr_cfg is None:
            return 'Prometheus', 3.0, [3.0]
        metric = getattr(qsr_cfg, 'metric', 'Prometheus')
        primary_threshold = float(getattr(qsr_cfg, 'threshold', 3.0))
        thresholds = getattr(qsr_cfg, 'thresholds', None) or []
        normalized_thresholds = sorted(
            {primary_threshold, *[float(threshold) for threshold in thresholds]}
        )
        return metric, primary_threshold, normalized_thresholds

    def _is_strict_quality_backfill_mode(self) -> bool:
        return bool(
            getattr(self.config, 'pipeline_mode', None) == 'evaluate'
            and getattr(self.config.evaluation, 'skip_detection_in_evaluate', False)
        )

    def _ensure_requested_quality_metrics_ready(
        self, requested_metrics: dict, quality_manager, *, phase_name: str
    ) -> None:
        if not requested_metrics or not self._is_strict_quality_backfill_mode():
            return

        active_metrics = set(getattr(quality_manager, 'metrics', {}) or {})
        missing_metrics = sorted(name for name in requested_metrics if name not in active_metrics)
        if missing_metrics:
            missing_joined = ', '.join(missing_metrics)
            raise RuntimeError(
                f'Required quality metrics failed to initialize for {phase_name}: '
                f'{missing_joined}. Aborting quality-only evaluate run.'
            )

    def _requires_detection_pass(self) -> bool:
        return not (
            self.pipeline_mode == 'evaluate'
            and getattr(self.config.evaluation, 'skip_detection_in_evaluate', False)
        )

    def _requires_base_model_for_evaluate(self) -> bool:
        if self.pipeline_mode != 'evaluate':
            return True

        dependent_metrics = {'PPL', 'LogDiv'}
        active_quality_metrics = set(self._evaluate_quality_metrics().keys())
        return self._requires_detection_pass() or bool(active_quality_metrics & dependent_metrics)

    def _configure_detection_manager(self) -> None:
        detection_cfg = self.config.evaluation.detection_metrics.model_copy(deep=True)
        detection_cfg.params = dict(detection_cfg.params)
        reverse_algo_set = set(detection_cfg.reverse_score_algorithms)
        detection_cfg.params['reverse'] = self.config.watermark in reverse_algo_set
        self.detection_manager = DetectionMetricManager(config=detection_cfg.model_dump())

    def _loaded_samples_have_precomputed_detection(self) -> bool:
        if not getattr(self, 'loaded_samples', None):
            return False

        metric_type = str(self.config.evaluation.detection_metrics.type).lower()
        output_attr = 'prediction' if metric_type in {'fundamental', 'spoof'} else 'z_score'

        for sample in self.loaded_samples:
            if getattr(sample, 'status', 'success') != 'success':
                continue

            required_types = ['natural', 'watermark']
            if metric_type == 'spoof':
                if getattr(sample, 'spoof_text', None):
                    required_types.append('spoof')
            elif getattr(sample, 'attacked_text', None):
                required_types.append('attack')

            for text_type in required_types:
                detection_res = getattr(sample, f'detection_{text_type}', None)
                if detection_res is None or getattr(detection_res, output_attr, None) is None:
                    return False

        return True

    @staticmethod
    def _merge_quality_scores(current_scores: dict | None, new_scores: dict | None) -> dict:
        merged = dict(current_scores or {})
        incoming = dict(new_scores or {})
        for category in ['absolute_quality', 'comparison_metrics']:
            category_scores = incoming.get(category)
            if not isinstance(category_scores, dict):
                continue
            if category not in merged or not isinstance(merged.get(category), dict):
                merged[category] = {}
            for text_type, metrics in category_scores.items():
                if text_type not in merged[category] or not isinstance(
                    merged[category].get(text_type), dict
                ):
                    merged[category][text_type] = {}
                if isinstance(metrics, dict):
                    merged[category][text_type].update(metrics)

        for key, value in incoming.items():
            if key not in {'absolute_quality', 'comparison_metrics'}:
                merged[key] = value

        return merged

    def prepare(self):
        """
        Prepares all components for the experiment, based on the loaded config.
        This method orchestrates the setup of the environment, data, models,
        and all necessary modules for the pipeline.
        """
        prepare_start_time = time.time()
        try:
            # self.context._setup_environment() # Already done in __init__
            logger.info('===== PREPARING EXPERIMENT =====')

            self.pipeline_mode = self.config.pipeline_mode
            self.data_path = self.config.data_path

            if self.pipeline_mode in ['generate', 'evaluate'] and not self.data_path:
                raise MissingDataPathError(self.pipeline_mode)

            # Step 1: Load Dataset and Prompts
            logger.info('--- 1. Loading data ---')

            if self.pipeline_mode == 'evaluate':
                logger.info(f'Loading pre-generated data from {self.data_path}')
                if not os.path.exists(self.data_path):
                    raise FileNotFoundError(f'Data file not found at {self.data_path}')

                self.loaded_samples = []
                with open(self.data_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            try:
                                self.loaded_samples.append(SampleRecord.from_dict(json.loads(line)))
                            except Exception as e:
                                logger.warning(f'Skipping invalid line in data file: {e}')

                if self.timing_summary['attack_seconds'] <= 0.0:
                    self.timing_summary['attack_seconds'] = sum(
                        float(sample.extras.get('attack_duration') or 0.0)
                        for sample in self.loaded_samples
                    )

                self.prompts = []  # Removed dependency on parallel lists
                self.natural_texts = []  # Removed dependency on parallel lists
                logger.info(f'Loaded {len(self.loaded_samples)} samples from file.')
            else:
                dataset_cfg = self.config.dataset
                # Pydantic ensures dataset_cfg exists

                dataset_name = dataset_cfg.name.lower()
                data_source = dataset_cfg.path
                num_samples = dataset_cfg.num_samples

                try:
                    self.dataset = DatasetRegistry.get_dataset(dataset_name)(
                        data_source=data_source, num_samples=num_samples
                    )
                except ValueError as e:
                    raise DatasetInitializationError(str(e))

                if len(self.dataset) == 0:
                    raise EmptyDatasetError(data_source)
                logger.info(f'Loaded {len(self.dataset)} samples from dataset.')

            # Step 2: Load All Required Models
            logger.info('--- 2. Loading models ---')
            base_model_name = self.config.base_model

            dataset_name = self.config.dataset.name
            dataset_meta = DatasetRegistry.get_metadata(dataset_name)
            self.task_type = dataset_meta.get('task_type', 'completion')

            if self._requires_base_model_for_evaluate():
                self.model_manager.load_model_and_tokenizer(base_model_name)
                logger.info(f'Successfully pre-loaded base model: {base_model_name}')
            else:
                logger.info(
                    'Skipping base model preload in evaluate mode because only standalone '
                    'metrics are active and detection reuse is enabled.'
                )

            # Check for Chat Template requirement
            chat_template_mode = self._chat_template_mode()
            is_instruct = 'instruct' in base_model_name.lower()

            supported_tasks = [
                'completion',
                'instruction_following',
                'generation',
                'generation_mixed',
            ]
            supports_chat_task = self.task_type in supported_tasks
            detection_pass_required = self._requires_detection_pass()
            if chat_template_mode == 'off':
                should_use_chat = False
            elif chat_template_mode == 'force':
                should_use_chat = supports_chat_task and detection_pass_required
            else:
                should_use_chat = is_instruct and supports_chat_task and detection_pass_required

            if should_use_chat:
                logger.info(
                    "Chat Template Enabled: mode=%s, model='%s', dataset='%s', task_type='%s'.",
                    chat_template_mode,
                    base_model_name,
                    dataset_name,
                    self.task_type,
                )
                _, tokenizer = self.model_manager.load_model_and_tokenizer(base_model_name)
                self.chat_formatter = ChatFormatter(
                    model_name_or_path=base_model_name, tokenizer=tokenizer
                )
            else:
                logger.info(
                    'Chat Template Disabled: mode=%s, Instruct=%s, TaskType=%s, DetectionPass=%s',
                    chat_template_mode,
                    is_instruct,
                    self.task_type,
                    detection_pass_required,
                )

            # Step 3: Initialize Core Modules (Watermark, Attack)
            logger.info('--- 3. Initializing Watermark & Attack modules ---')

            if self.pipeline_mode == 'evaluate':
                logger.info(
                    "Skipping Watermark & Attack module initialization in 'evaluate' mode (Sequential loading)."
                )
            else:
                base_model_name = self.config.base_model
                base_model, base_tokenizer = self.model_manager.load_model_and_tokenizer(
                    base_model_name
                )
                self.watermark_instance = self.component_factory.initialize_watermark(
                    base_model, base_tokenizer
                )
                self.attack_instance = self.component_factory.initialize_attack(
                    self.dataset, self.watermark_instance
                )

            # Step 4: Initialize Metric Managers
            logger.info('--- 4. Initializing Metric Managers ---')
            eval_cfg = self.config.evaluation
            self.protocol = eval_cfg.protocol

            if self.pipeline_mode == 'generate':
                logger.info("Skipping Metric Managers initialization in 'generate' mode.")
            elif self.pipeline_mode == 'evaluate':
                logger.info(
                    "Skipping Metric Managers initialization in 'evaluate' mode (Sequential loading)."
                )

            logger.info('Preparation phase completed successfully.')
        finally:
            prepare_duration = time.time() - prepare_start_time
            self.timing_summary['prepare_seconds'] += prepare_duration
            logger.info(f'Preparation Time: {prepare_duration:.2f}s')

    def _generate_natural_text(self, prompt: str) -> str:
        """
        Generates natural text (no watermark) using the base model.
        Uses the same generation parameters as configured for the watermark.
        """
        base_model_name = self.config.base_model
        model, tokenizer = self.model_manager.load_model_and_tokenizer(base_model_name)

        # Ensure pad_token_id is set
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        inputs = tokenizer(prompt, return_tensors='pt').to(self.device)

        # Use config parameters
        gen_kwargs = {
            'max_new_tokens': self.config.max_tokens,
            'min_new_tokens': self.config.min_len,
            'do_sample': True,
            'no_repeat_ngram_size': 4,
            'pad_token_id': tokenizer.pad_token_id,
        }

        with torch.no_grad():
            output_ids = model.generate(**inputs, **gen_kwargs)

        # Decode
        full_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return strip_prompt(full_text, prompt)

    def _collect_attack_meta(self) -> dict:
        """Collect optional attack metadata from attack instance if supported."""
        if not self.attack_instance:
            return {}
        pop_meta = getattr(self.attack_instance, 'pop_last_attack_meta', None)
        if not callable(pop_meta):
            return {}
        try:
            meta = pop_meta()
            return meta if isinstance(meta, dict) else {}
        except Exception as e:
            logger.warning(f'Failed to collect attack metadata: {e}')
            return {}

    def _format_prompt_for_generation(self, prompt_text: str) -> str:
        prompt_for_gen = prompt_text
        if not self.chat_formatter:
            return prompt_for_gen

        if self.task_type == 'completion':
            return self.chat_formatter.format_c4_to_instruct(prompt_text)
        return self.chat_formatter.format_instruction_dataset(prompt_text)

    def _should_defer_detection_prediction(self) -> bool:
        """Dynamic detection calibrates threshold after the full batch is available."""
        detection_cfg = getattr(self.config.evaluation, 'detection_metrics', None)
        metric_type = getattr(detection_cfg, 'type', 'fundamental')
        return str(metric_type).lower() == 'dynamic'

    def _should_capture_prompt_aware_detection(self) -> bool:
        """SIR benefits from an additional prompt-aware diagnostic detection pass."""
        return str(getattr(self.config, 'watermark', '')).strip().upper() == 'SIR'

    def _get_sir_detection_mode(self) -> str:
        mode = str(getattr(self.config, 'sir_detection_mode', 'prompt_free')).strip().lower()
        if mode not in {'prompt_free', 'prompt_aware'}:
            logger.warning("Unknown sir_detection_mode='%s'; falling back to prompt_free", mode)
            return 'prompt_free'
        return mode

    @staticmethod
    def _build_prompt_aware_detection_text(
        text_to_check: str, prompt_text: str, prompt_for_gen: str
    ) -> str:
        if text_to_check.startswith(prompt_for_gen) or text_to_check.startswith(prompt_text):
            return text_to_check
        return f'{prompt_for_gen}{text_to_check}'

    def _build_detection_text(
        self,
        *,
        text_to_check: str,
        prompt_text: str,
        prompt_for_gen: str,
        mode: str,
    ) -> str:
        if self._should_capture_prompt_aware_detection() and mode == 'prompt_aware':
            return self._build_prompt_aware_detection_text(
                text_to_check=text_to_check,
                prompt_text=prompt_text,
                prompt_for_gen=prompt_for_gen,
            )
        return text_to_check

    @staticmethod
    def _log_detection_reason(
        *, sample_index: int, text_type: str, mode: str, detection_result: dict | None
    ) -> None:
        if not isinstance(detection_result, dict):
            return
        reason = detection_result.get('reason')
        if not reason:
            return
        logger.warning(
            'Detection returned no score for sample %s, type %s, mode %s: %s',
            sample_index,
            text_type,
            mode,
            reason,
        )

    def process_one_sample(self, record: SampleRecord):
        """
        Core logic for processing a sample (Watermark -> Attack -> Detect -> Quality).
        """
        # 1. Format Prompt
        prompt_text = record.prompt
        prompt_for_gen = self._format_prompt_for_generation(prompt_text)

        record.extras['prompt_for_gen'] = prompt_for_gen

        # 2. Ensure Natural Text Exists
        natural_text = record.natural_text
        if not natural_text:
            natural_text = self._generate_natural_text(prompt_for_gen)
            record.natural_text = natural_text
            record.full_natural_text = prompt_text + natural_text

        # Basic Record Init
        dataset_name_lower = self.config.dataset.name.lower()
        if 'c4' in dataset_name_lower and not record.source_text:
            record.source_text = natural_text

        attack_duration = 0.0
        attack_meta = {}
        texts_to_detect = {}

        # 3. Branch: Spoofing vs Removal
        if self.protocol == 'spoofing':
            attack_kwargs = {
                'attack_type': 'spoofing',
                'prompt': prompt_text,
                'reference': prompt_text,
                'watermark_instance': self.watermark_instance,
            }
            attack_start_time = time.time()
            spoof_text = self.attack_instance.edit(text='', **attack_kwargs)
            # Strip prompt for spoofing as we passed prompt as reference
            spoof_text = strip_prompt(spoof_text, prompt_text)
            attack_duration = time.time() - attack_start_time
            attack_meta = self._collect_attack_meta()

            record.spoof_text = spoof_text
            record.full_spoof_text = prompt_text + spoof_text

            texts_to_detect = {'natural': natural_text, 'spoof': spoof_text}

            if self.config.evaluation.spoof.compare_baseline:
                watermarked_text = self.watermark_instance.generate_watermarked_text(
                    prompt=prompt_for_gen
                )
                watermarked_text = strip_prompt(watermarked_text, prompt_for_gen)
                record.watermarked_text = watermarked_text
                record.full_watermarked_text = prompt_text + watermarked_text
                texts_to_detect['watermark'] = watermarked_text

        else:  # Standard Removal Protocol
            watermarked_text = self.watermark_instance.generate_watermarked_text(
                prompt=prompt_for_gen
            )
            watermarked_text = strip_prompt(watermarked_text, prompt_for_gen)
            record.watermarked_text = watermarked_text
            record.full_watermarked_text = prompt_text + watermarked_text

            attack_kwargs = {
                'attack_type': 'scrubbing',
                'prompt': prompt_text,
                'reference': prompt_text,
                'watermark_instance': self.watermark_instance,
            }
            attack_start_time = time.time()
            attacked_text = self.attack_instance.edit(text=watermarked_text, **attack_kwargs)
            attack_duration = time.time() - attack_start_time
            attack_meta = self._collect_attack_meta()

            record.attacked_text = attacked_text
            record.full_attacked_text = prompt_text + attacked_text

            texts_to_detect = {
                'natural': natural_text,
                'watermark': watermarked_text,
                'attack': attacked_text,
            }

        # 4. Detection
        primary_detection_mode = self._get_sir_detection_mode()
        for text_type, text_to_check in texts_to_detect.items():
            if not text_to_check or not isinstance(text_to_check, str):
                continue

            detection_text = self._build_detection_text(
                text_to_check=text_to_check,
                prompt_text=prompt_text,
                prompt_for_gen=prompt_for_gen,
                mode=primary_detection_mode,
            )
            detection_result = self.watermark_instance.detect_watermark(text=detection_text)
            self._log_detection_reason(
                sample_index=record.sample_index,
                text_type=text_type,
                mode=primary_detection_mode,
                detection_result=detection_result,
            )
            prediction = detection_result.get('is_watermarked')
            if self._should_defer_detection_prediction():
                prediction = None
            det_res_obj = DetectionResult(
                prediction=prediction,
                z_score=detection_result.get('score'),
            )
            setattr(record, f'detection_{text_type}', det_res_obj)

        if self._should_capture_prompt_aware_detection():
            diagnostic_mode = (
                'prompt_aware' if primary_detection_mode == 'prompt_free' else 'prompt_free'
            )
            diagnostic_detection = {}
            for text_type, text_to_check in texts_to_detect.items():
                if not text_to_check or not isinstance(text_to_check, str):
                    continue

                diagnostic_text = self._build_detection_text(
                    text_to_check=text_to_check,
                    prompt_text=prompt_text,
                    prompt_for_gen=prompt_for_gen,
                    mode=diagnostic_mode,
                )
                detection_result = self.watermark_instance.detect_watermark(text=diagnostic_text)
                self._log_detection_reason(
                    sample_index=record.sample_index,
                    text_type=text_type,
                    mode=diagnostic_mode,
                    detection_result=detection_result,
                )
                prediction = detection_result.get('is_watermarked')
                if self._should_defer_detection_prediction():
                    prediction = None
                diagnostic_detection[text_type] = {
                    'prediction': prediction,
                    'z_score': detection_result.get('score'),
                }

            if diagnostic_detection:
                record.extras[f'{diagnostic_mode}_detection'] = diagnostic_detection

        # 5. Quality Analysis
        if self.quality_manager:
            try:
                quality_scores = self.quality_manager.analyze_sample(record)
                record.quality_metrics = quality_scores
            except Exception as e:
                logger.warning(
                    f'Sample {record.sample_index}: Quality metric calculation failed: {e}'
                )

        record.status = 'success'
        record.extras['attack_duration'] = attack_duration
        self.timing_summary['attack_seconds'] += attack_duration
        if attack_meta:
            record.extras['attack_meta'] = attack_meta
            if 'query_cost' in attack_meta:
                record.extras['query_cost'] = attack_meta['query_cost']
            if 'runtime_cost' in attack_meta:
                record.extras['runtime_cost'] = attack_meta['runtime_cost']

        # 6. Persist
        if self.data_file_handle:
            # Append newline if missing (model_dump_json usually doesn't add it)
            line = record.model_dump_json() + '\n'
            self.data_file_handle.write(line)
            self.data_file_handle.flush()

    def run(self):
        """Runs the main loop using SamplePipeline."""
        logger.info('===== RUNNING EVALUATION LOOP =====')

        if self.pipeline_mode == 'evaluate':
            self._run_sequential_evaluation()
            logger.info('Evaluation loop finished (Sequential).')
            return

        # Determine source and count
        samples_source = self.dataset
        num_samples = len(self.dataset)

        loop_start_time = time.time()

        logger.info(f'Opening data file for writing: {self.data_path}')
        self.data_file_handle = open(self.data_path, 'w', encoding='utf-8')

        for i, sample_data in enumerate(samples_source):
            logger.info(f'--- Processing Sample {i + 1}/{num_samples} ---')

            # Create initial record
            prompt = getattr(sample_data, 'prompt', '')
            natural_text = getattr(sample_data, 'natural_text', '')
            reference = str(getattr(sample_data, 'reference', ''))

            # Initialize record with known data
            record = SampleRecord(
                sample_index=i,
                prompt=prompt,
                natural_text=natural_text,
                full_natural_text=(prompt + natural_text) if natural_text else '',
                reference=reference,
            )

            try:
                self.process_one_sample(record)
                self.results_log.append(record)
                self.exp_logger.log_sample(
                    i, record, attack_duration=record.extras.get('attack_duration', 0.0)
                )
            except Exception as e:
                logger.error(f'Failed to process sample {i}: {e}')
                record.status = 'failed'
                record.error_msg = str(e)
                self.results_log.append(record)
                continue

            # Timing info (loop level)
            attack_duration = record.extras.get('attack_duration', 0.0)
            elapsed_total = time.time() - loop_start_time
            avg_time_per_sample = elapsed_total / (i + 1)
            remaining_samples = num_samples - (i + 1)
            estimated_remaining_time = avg_time_per_sample * remaining_samples
            estimated_total_time = avg_time_per_sample * num_samples

            logger.info(f'Sample {i + 1} Attack Time: {attack_duration:.2f}s')
            logger.info(
                f'Estimated Total Time: {estimated_total_time / 60:.2f} mins (Remaining: {estimated_remaining_time / 60:.2f} mins, Avg: {avg_time_per_sample:.2f}s/sample)'
            )

        logger.info('Evaluation loop finished.')

        if self.data_file_handle:
            self.data_file_handle.close()
            self.data_file_handle = None

    def report(self):
        if self.reporter:
            if self._pipeline_wall_start is not None:
                self.timing_summary['total_seconds'] = max(
                    self.timing_summary['total_seconds'],
                    time.time() - self._pipeline_wall_start,
                )
            summary_report = self.reporter.generate_report(
                self.results_log, cost_summary=self._build_cost_summary()
            )
            if summary_report is not None:
                try:
                    self.exp_logger.on_run_end(summary_report)
                except Exception as e:
                    logger.warning(f'Experiment logger summary export failed: {e}')
        else:
            logger.info('Reporter not initialized. Skipping report generation.')

    def _build_cost_summary(self) -> dict[str, float]:
        query_cost = 0.0
        runtime_cost = 0.0
        has_query_cost = False
        has_runtime_cost = False

        for record in self.results_log:
            extras = getattr(record, 'extras', {}) or {}
            if extras.get('query_cost') is not None:
                query_cost += float(extras.get('query_cost') or 0.0)
                has_query_cost = True
            if extras.get('runtime_cost') is not None:
                runtime_cost += float(extras.get('runtime_cost') or 0.0)
                has_runtime_cost = True

        cost_summary = {
            'prepare_seconds': float(self.timing_summary.get('prepare_seconds', 0.0)),
            'attack_seconds': float(self.timing_summary.get('attack_seconds', 0.0)),
            'total_seconds': float(self.timing_summary.get('total_seconds', 0.0)),
            # Keep the existing dashboard/runtime_cost path usable.
            'runtime_cost': runtime_cost
            if has_runtime_cost
            else float(self.timing_summary.get('attack_seconds', 0.0)),
        }
        if has_query_cost:
            cost_summary['query_cost'] = query_cost
        return cost_summary

    def cleanup_resources(self):
        """
        Cleans up resources to free memory between phases.
        """
        logger.info('Cleaning up resources...')

        # Close file handle if open
        if hasattr(self, 'data_file_handle') and self.data_file_handle:
            try:
                self.data_file_handle.close()
            except Exception as e:
                logger.warning(f'Error closing data file handle: {e}')
            self.data_file_handle = None

        # Clear large objects
        self.dataset = None
        self.watermark_instance = None
        self.attack_instance = None
        self.quality_manager = None
        self.detection_manager = None
        self.chat_formatter = None
        self.loaded_samples = []

        # Force GC
        self.model_manager.clear_cache()
        logger.info('Resources cleaned up.')

    def _destroy_distributed_process_group(self):
        dist = getattr(torch, 'distributed', None)
        if dist is None or not dist.is_available():
            return

        try:
            if dist.is_initialized():
                dist.destroy_process_group()
                logger.info('Destroyed torch.distributed process group.')
        except Exception as e:
            logger.warning(f'Failed to destroy torch.distributed process group: {e}')

    def execute_pipeline(self):
        original_mode = self.config.pipeline_mode
        self._pipeline_wall_start = time.time()
        run_success = False
        run_error_message = None

        # Register run identity/status in sqlite before generation/evaluation starts.
        self.run_registry.register_run_start()

        # Lifecycle: Run Start
        self.exp_logger.on_run_start(
            RunMeta(
                run_name=self.context.experiment_name,
                config=self.config.model_dump(),
                run_id=self.context.experiment_id,
            )
        )

        try:
            if original_mode == 'all':
                logger.info("Executing 'all' pipeline: Generation -> Evaluation")

                # --- Phase 1: Generation ---
                logger.info('>>> Starting Phase 1: Generation')
                self.pipeline_mode = 'generate'
                # Temporarily update config to match the phase
                self.config.pipeline_mode = 'generate'

                self.exp_logger.on_phase_start(PhaseMeta(name='generate', index=1))

                try:
                    self.prepare()
                    self.run()
                except Exception as e:
                    logger.error(f'Phase 1 (Generation) failed: {e}')
                    raise e
                finally:
                    self.cleanup_resources()
                    self.exp_logger.on_phase_end(PhaseMeta(name='generate', index=1))

                # --- Phase 2: Evaluation ---
                logger.info('>>> Starting Phase 2: Evaluation')
                self.pipeline_mode = 'evaluate'
                self.config.pipeline_mode = 'evaluate'
                self.results_log = []

                # Ensure data_path is set (should be same as generation output)
                if not self.config.data_path:
                    raise MissingDataPathError('evaluate')

                self.exp_logger.on_phase_start(PhaseMeta(name='evaluate', index=2))

                try:
                    self.prepare()
                    self.run()
                    self.report()
                except Exception as e:
                    logger.error(f'Phase 2 (Evaluation) failed: {e}')
                    raise e
                finally:
                    self.exp_logger.on_phase_end(PhaseMeta(name='evaluate', index=2))

                # Restore original mode just in case
                self.config.pipeline_mode = original_mode
                self.pipeline_mode = original_mode

            else:
                self.exp_logger.on_phase_start(PhaseMeta(name=self.pipeline_mode, index=1))
                try:
                    self.prepare()
                    self.run()
                    self.report()
                finally:
                    self.exp_logger.on_phase_end(PhaseMeta(name=self.pipeline_mode, index=1))
            run_success = True
        except Exception as e:
            run_error_message = str(e)
            raise
        finally:
            self.run_registry.finalize_run(success=run_success, error_message=run_error_message)
            if self._pipeline_wall_start is not None:
                self.timing_summary['total_seconds'] = max(
                    self.timing_summary['total_seconds'],
                    time.time() - self._pipeline_wall_start,
                )
            try:
                self.exp_logger.close()
            finally:
                self._destroy_distributed_process_group()

    def _run_sequential_evaluation(self):
        """
        Special execution mode for evaluation:
        1. Detection Evaluation (Requires Base Model)
        2. Quality Evaluation - Part A (Requires Base Model, e.g. PPL)
        3. Release Base Model to free VRAM
        4. Quality Evaluation - Part B (Standalone, e.g. Prometheus, BERTScore)
        """
        logger.info('===== RUNNING SEQUENTIAL EVALUATION (Optimized) =====')
        samples_source = self.loaded_samples
        num_samples = len(samples_source)
        logger.info(f'Sequential Evaluation: Processing {num_samples} samples.')

        # --- Phase 1: Detection Evaluation ---
        self._configure_detection_manager()
        if self._requires_detection_pass():
            logger.info('--- Phase 1: Detection Evaluation ---')

            # 1.1 Init Watermark
            base_model_name = self.config.base_model
            base_model, base_tokenizer = self.model_manager.load_model_and_tokenizer(
                base_model_name
            )
            self.watermark_instance = self.component_factory.initialize_watermark(
                base_model, base_tokenizer
            )

            # 1.2 Loop for Detection
            for i, sample_data in enumerate(samples_source):
                current_sample_log = sample_data
                prompt_text = current_sample_log.prompt
                prompt_for_gen = (current_sample_log.extras or {}).get(
                    'prompt_for_gen'
                ) or self._format_prompt_for_generation(prompt_text)
                current_sample_log.extras['prompt_for_gen'] = prompt_for_gen

                # Identify texts
                natural_text = current_sample_log.natural_text
                texts_to_detect = {'natural': natural_text}
                if current_sample_log.watermarked_text:
                    texts_to_detect['watermark'] = current_sample_log.watermarked_text
                if current_sample_log.attacked_text:
                    texts_to_detect['attack'] = current_sample_log.attacked_text
                if current_sample_log.spoof_text:
                    texts_to_detect['spoof'] = current_sample_log.spoof_text

                # Detect
                primary_detection_mode = self._get_sir_detection_mode()
                for text_type, text_to_check in texts_to_detect.items():
                    if not text_to_check or not isinstance(text_to_check, str):
                        continue

                    try:
                        detection_text = self._build_detection_text(
                            text_to_check=text_to_check,
                            prompt_text=prompt_text,
                            prompt_for_gen=prompt_for_gen,
                            mode=primary_detection_mode,
                        )
                        detection_result = self.watermark_instance.detect_watermark(
                            text=detection_text
                        )
                        self._log_detection_reason(
                            sample_index=i,
                            text_type=text_type,
                            mode=primary_detection_mode,
                            detection_result=detection_result,
                        )
                        prediction = detection_result.get('is_watermarked')
                        if self.detection_manager.metric_type == 'dynamic':
                            prediction = None
                        det_res_obj = DetectionResult(
                            prediction=prediction,
                            z_score=detection_result.get('score'),
                        )
                        setattr(current_sample_log, f'detection_{text_type}', det_res_obj)
                    except Exception as e:
                        logger.error(f'Detection failed for sample {i}, type {text_type}: {e}')

                if self._should_capture_prompt_aware_detection():
                    diagnostic_mode = (
                        'prompt_aware' if primary_detection_mode == 'prompt_free' else 'prompt_free'
                    )
                    diagnostic_detection = {}
                    for text_type, text_to_check in texts_to_detect.items():
                        if not text_to_check or not isinstance(text_to_check, str):
                            continue

                        try:
                            diagnostic_text = self._build_detection_text(
                                text_to_check=text_to_check,
                                prompt_text=prompt_text,
                                prompt_for_gen=prompt_for_gen,
                                mode=diagnostic_mode,
                            )
                            detection_result = self.watermark_instance.detect_watermark(
                                text=diagnostic_text
                            )
                            self._log_detection_reason(
                                sample_index=i,
                                text_type=text_type,
                                mode=diagnostic_mode,
                                detection_result=detection_result,
                            )
                            prediction = detection_result.get('is_watermarked')
                            if self.detection_manager.metric_type == 'dynamic':
                                prediction = None
                            diagnostic_detection[text_type] = {
                                'prediction': prediction,
                                'z_score': detection_result.get('score'),
                            }
                        except Exception as e:
                            logger.error(
                                'Diagnostic detection failed for sample %s, type %s, mode %s: %s',
                                i,
                                text_type,
                                diagnostic_mode,
                                e,
                            )

                    if diagnostic_detection:
                        current_sample_log.extras[f'{diagnostic_mode}_detection'] = (
                            diagnostic_detection
                        )

                current_sample_log.status = 'success'

                if (i + 1) % 10 == 0:
                    logger.info(f'Detection Eval: Processed {i + 1}/{num_samples} samples')

            # Cleanup Detection Resources
            self.watermark_instance = None
        else:
            logger.info(
                '--- Phase 1: Detection Evaluation (Skipped; using precomputed outputs) ---'
            )
            if not self._loaded_samples_have_precomputed_detection():
                raise ValueError(
                    'evaluate.skip_detection_in_evaluate=true requires precomputed detection '
                    'outputs in data_path.'
                )

        # --- Phase 2: Quality Evaluation (Dependent on Base Model) ---
        logger.info('--- Phase 2: Quality Evaluation (Base Model Dependent) ---')

        all_quality_metrics = self._evaluate_quality_metrics()
        dependent_metrics = ['PPL', 'LogDiv']  # Metrics that need base model/tokenizer

        dependent_config = {k: v for k, v in all_quality_metrics.items() if k in dependent_metrics}

        if dependent_config:
            base_model_name = self.config.base_model
            base_model, base_tokenizer = self.model_manager.load_model_and_tokenizer(
                base_model_name
            )
            self.quality_manager = QualityMetricManager(
                config=dependent_config,
                models={
                    'base_model': {
                        'model': base_model,
                        'tokenizer': base_tokenizer,
                    }
                },
                device=self.device,
                qsr_metric=self._qsr_settings()[0],
                qsr_threshold=self._qsr_settings()[1],
                qsr_thresholds=self._qsr_settings()[2],
                comparison_names=self._quality_comparison_names(),
                local_files_only=self.config.runner_settings.models_local_files_only,
            )
            self._ensure_requested_quality_metrics_ready(
                dependent_config,
                self.quality_manager,
                phase_name='dependent quality evaluation',
            )

            for i, sample_data in enumerate(samples_source):
                current_sample_log = sample_data
                try:
                    # Initial calculation
                    quality_scores = self.quality_manager.analyze_sample(current_sample_log)
                    current_sample_log.quality_metrics = self._merge_quality_scores(
                        current_sample_log.quality_metrics, quality_scores
                    )
                except Exception as e:
                    logger.warning(
                        f'Sample {i}: Quality metric (Dependent) calculation failed: {e}'
                    )

                if (i + 1) % 10 == 0:
                    logger.info(f'Quality (Dependent): Processed {i + 1}/{num_samples} samples')

            # Cleanup Quality Manager
            del self.quality_manager
            self.quality_manager = None

        # --- Cleanup Base Model ---
        logger.info('Unloading Base Model to free VRAM for Standalone Metrics...')

        # Clear ModelManager cache to ensure references are dropped
        if self.model_manager:
            self.model_manager.clear_cache()

        # --- Phase 3: Quality Evaluation (Standalone) ---
        logger.info('--- Phase 3: Quality Evaluation (Standalone) ---')

        standalone_config = {
            k: v for k, v in all_quality_metrics.items() if k not in dependent_metrics
        }

        # Initialize QualityManager only if there are standalone metrics
        if standalone_config:
            self.quality_manager = QualityMetricManager(
                config=standalone_config,
                models={'base_model': {'model': None, 'tokenizer': None}},
                device=self.device,
                qsr_metric=self._qsr_settings()[0],
                qsr_threshold=self._qsr_settings()[1],
                qsr_thresholds=self._qsr_settings()[2],
                comparison_names=self._quality_comparison_names(),
                local_files_only=self.config.runner_settings.models_local_files_only,
            )
            self._ensure_requested_quality_metrics_ready(
                standalone_config,
                self.quality_manager,
                phase_name='standalone quality evaluation',
            )
        else:
            self.quality_manager = None
            logger.info('No standalone quality metrics configured.')

        # Always run the final loop to populate results_log and log to SwanLab
        for i, sample_data in enumerate(samples_source):
            current_sample_log = sample_data

            # Calculate standalone metrics if manager exists
            if self.quality_manager:
                try:
                    new_scores = self.quality_manager.analyze_sample(current_sample_log)
                    current_sample_log.quality_metrics = self._merge_quality_scores(
                        current_sample_log.quality_metrics, new_scores
                    )

                except Exception as e:
                    logger.warning(
                        f'Sample {i}: Quality metric (Standalone) calculation failed: {e}'
                    )

            # Log final full sample to Experiment Logger
            self.exp_logger.log_sample(i, current_sample_log)
            self.results_log.append(current_sample_log)

            if (i + 1) % 10 == 0:
                logger.info(
                    f'Finalizing (Quality/Logging): Processed {i + 1}/{num_samples} samples'
                )

        # Cleanup
        if self.quality_manager:
            del self.quality_manager
            self.quality_manager = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # --- Phase 4: Reporting ---
        # Re-init a dummy QualityManager for Reporter
        self.quality_manager = QualityMetricManager(
            config={},
            models={'base_model': {'model': None, 'tokenizer': None}},
            device='cpu',
            qsr_metric=self._qsr_settings()[0],
            qsr_threshold=self._qsr_settings()[1],
            qsr_thresholds=self._qsr_settings()[2],
            local_files_only=self.config.runner_settings.models_local_files_only,
        )

        self.reporter = Reporter(
            self.context,
            metric_managers={
                'detection': self.detection_manager,
                'quality': self.quality_manager,
            },
        )

        self.reporter.ensure_db_schema()
