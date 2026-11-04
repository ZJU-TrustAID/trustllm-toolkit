import hashlib
import json
import logging
import os
import yaml
from typing import Dict, Any

from src.util.db import DatabaseManager
from src.util.data import json_converter, round_floats
from src.evaluation.result_schema import SampleRecord
from pathlib import Path

logger = logging.getLogger('MarkSec')
FORMAL_BENCHMARK_NUM_SAMPLES = 100
QUALITY_BACKFILL_COLUMNS = (
    'bertscore_atk_vs_wm',
    'bleu_atk_vs_wm',
    'psp_atk_vs_wm',
    'prom_atk_vs_wm',
    'qsr_prom2_atk_vs_wm',
    'qsr_prom3_atk_vs_wm',
    'qsr_prom4_atk_vs_wm',
    'qsr_prom5_atk_vs_wm',
    'qsr_atk_vs_wm',
)
SPOOF_TABLE_NAME = 'experiment_summary_spoof'
STEAL_SCRUB_ATTACKS = {'B4', 'MIP', 'DEMARK', 'SCTS', 'JSV'}
STEAL_SPOOF_ATTACKS = {'B4', 'DEMARK', 'JSV'}


def _dataset_name_from_config(config: Any) -> Any:
    dataset_cfg = getattr(config, 'dataset', {})
    if isinstance(dataset_cfg, dict):
        return dataset_cfg.get('name')
    return getattr(dataset_cfg, 'name', None)


def _full_config_from_report(report: Dict[str, Any]) -> Dict[str, Any]:
    full_config = report.get('full_config') or {}
    return full_config if isinstance(full_config, dict) else {}


def _dataset_num_samples_from_report(report: Dict[str, Any]) -> int | None:
    dataset_cfg = _full_config_from_report(report).get('dataset') or {}
    raw_value = dataset_cfg.get('num_samples') if isinstance(dataset_cfg, dict) else None
    try:
        return int(raw_value)
    except (TypeError, ValueError):
        return None


def _dataset_name_from_report(report: Dict[str, Any]) -> str | None:
    dataset_cfg = _full_config_from_report(report).get('dataset') or {}
    raw_value = dataset_cfg.get('name') if isinstance(dataset_cfg, dict) else None
    if raw_value is None:
        return None
    text = str(raw_value).strip()
    return text or None


def _derive_watermark_setting(report: Dict[str, Any]) -> str | None:
    full_config = _full_config_from_report(report)
    info = report.get('experiment_info') or {}

    watermark = str(full_config.get('watermark') or info.get('watermark') or '').strip()
    config_path = str(full_config.get('watermark_config_path') or '').strip()

    if config_path:
        stem = Path(config_path).stem.strip()
        if stem and watermark:
            if stem.lower() == watermark.lower() or stem.lower().startswith(
                f'{watermark.lower()}_'
            ):
                return stem
            return f'{watermark}_{stem}'
        if stem:
            return stem

    return watermark or None


def _attack_name_from_report(report: Dict[str, Any]) -> str:
    info = report.get('experiment_info') or {}
    return str(info.get('attack') or '').strip()


def _infer_lifecycle_mode(report: Dict[str, Any]) -> str:
    protocol = _report_protocol(report)
    attack_name = _attack_name_from_report(report)

    if protocol == 'spoofing':
        if attack_name in STEAL_SPOOF_ATTACKS:
            return 'steal_spoof'
        raise ValueError(
            f'Unsupported spoof lifecycle classification for attack {attack_name or "<unknown>"}. '
            'Current paper-facing spoof runs must be steal-based and one of: '
            f'{", ".join(sorted(STEAL_SPOOF_ATTACKS))}.'
        )

    return 'steal_scrub' if attack_name in STEAL_SCRUB_ATTACKS else 'direct_scrub'


def _should_persist_to_benchmark_table(report: Dict[str, Any]) -> bool:
    num_samples = _dataset_num_samples_from_report(report)
    if num_samples is None:
        return True
    return num_samples == FORMAL_BENCHMARK_NUM_SAMPLES


def _select_primary_quality_bucket(report: Dict[str, Any]) -> Dict[str, Any]:
    comparisons = (report.get('quality_metrics') or {}).get('avg_comparison_metrics') or {}
    for candidate in _primary_quality_bucket_candidates(report):
        bucket = comparisons.get(candidate)
        if isinstance(bucket, dict) and bucket:
            return bucket
    return {}


def _report_protocol(report: Dict[str, Any]) -> str:
    full_config = _full_config_from_report(report)
    eval_cfg = full_config.get('evaluation') or {}
    raw_protocol = eval_cfg.get('protocol') if isinstance(eval_cfg, dict) else None
    protocol = str(raw_protocol or '').strip().lower()
    return protocol or 'removal'


def _primary_quality_bucket_candidates(report: Dict[str, Any]) -> tuple[str, ...]:
    protocol = _report_protocol(report)

    if protocol == 'spoofing':
        return (
            'spoof_vs_watermarked',
            'spoof_vs_reference',
            'spoof_vs_source',
            'spoof_vs_natural',
            'attacked_vs_watermarked',
            'attacked_vs_reference',
            'attacked_vs_source',
            'attacked_vs_natural',
        )
    return (
        'attacked_vs_watermarked',
        'attacked_vs_reference',
        'attacked_vs_source',
        'attacked_vs_natural',
        'spoof_vs_watermarked',
        'spoof_vs_reference',
        'spoof_vs_source',
        'spoof_vs_natural',
    )


def _primary_quality_bucket_name(report: Dict[str, Any]) -> str | None:
    comparisons = (report.get('quality_metrics') or {}).get('avg_comparison_metrics') or {}
    for candidate in _primary_quality_bucket_candidates(report):
        bucket = comparisons.get(candidate)
        if isinstance(bucket, dict) and bucket:
            return candidate
    return None


def _normalize_path(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return str(Path(text).expanduser().resolve())


def _dashboard_value(config: Any, key: str) -> str | None:
    dashboard = getattr(config, 'dashboard', None)
    if not isinstance(dashboard, dict):
        return None
    text = str(dashboard.get(key) or '').strip()
    return text or None


def _format_optional_float(value: Any) -> str:
    try:
        return f'{float(value):.4f}'
    except (TypeError, ValueError):
        return 'N/A'


def _first_present(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _spoof_table_name_for_report(default_table_name: str, report: Dict[str, Any]) -> str:
    if default_table_name == 'experiment_summary' and _report_protocol(report) == 'spoofing':
        return SPOOF_TABLE_NAME
    return default_table_name


def _config_protocol(config: Any) -> str:
    eval_cfg = getattr(config, 'evaluation', None)
    if isinstance(eval_cfg, dict):
        raw_protocol = eval_cfg.get('protocol')
    else:
        raw_protocol = getattr(eval_cfg, 'protocol', None)
    protocol = str(raw_protocol or '').strip().lower()
    return protocol or 'removal'


def _spoof_table_name_for_config(default_table_name: str, config: Any) -> str:
    if default_table_name == 'experiment_summary' and _config_protocol(config) == 'spoofing':
        return SPOOF_TABLE_NAME
    return default_table_name


def _extract_spoof_artifact_metadata(report: Dict[str, Any]) -> Dict[str, Any]:
    full_config = _full_config_from_report(report)
    stealing_cfg = {}
    for key in ('stealing_parameters', 'stealing'):
        candidate = full_config.get(key)
        if isinstance(candidate, dict) and candidate:
            stealing_cfg = candidate
            break

    model_requirements = full_config.get('model_requirements') or {}
    attacker_model = None
    if isinstance(model_requirements, dict):
        attacker_model = model_requirements.get('attacker')

    prepare_meta_path = stealing_cfg.get('prepare_artifact_path')
    prepare_artifact_path = (
        stealing_cfg.get('stolen_artifact_path')
        or stealing_cfg.get('load_proxy_path')
        or prepare_meta_path
    )
    capability_profile = (
        stealing_cfg.get('learning_mode')
        or full_config.get('capability_profile')
        or full_config.get('attack_type')
    )
    artifact_family = None
    if stealing_cfg.get('stolen_artifact_path'):
        artifact_family = 'stolen_counts'
    elif stealing_cfg.get('load_proxy_path'):
        artifact_family = 'proxy_model'
    elif prepare_artifact_path or prepare_meta_path:
        artifact_family = 'prepare_artifact'

    return {
        'attacker_model': attacker_model,
        'prepare_artifact_path': _normalize_path(prepare_artifact_path),
        'prepare_meta_path': _normalize_path(prepare_meta_path),
        'artifact_family': artifact_family,
        'capability_profile': capability_profile,
    }


class Reporter:
    """
    Handles generating reports, saving to disk, database, and SwanLab.
    """

    def __init__(self, context, metric_managers):
        self.context = context
        self.output_dir = context.output_dir
        self.config = context.config
        self.experiment_id = context.experiment_id
        self.report_experiment_id = self._resolve_report_experiment_id()
        self.timestamp = context.timestamp
        # Use .get() to avoid KeyError if key is missing or value is None, though user warns it might be None
        self.detection_manager = metric_managers.get('detection')
        self.quality_manager = metric_managers.get('quality')

    def _resolve_report_experiment_id(self) -> str | None:
        dashboard = getattr(self.config, 'dashboard', None)
        if (
            isinstance(dashboard, dict)
            and self._is_quality_only_backfill_run()
            and str(dashboard.get('source_run_id') or '').strip()
        ):
            return str(dashboard['source_run_id']).strip()
        return self.experiment_id

    def generate_report(self, results_log, cost_summary: Dict[str, Any] | None = None):
        logger.info('===== GENERATING REPORTS =====')

        logger.info('--- 1. Calculating summary metrics from the results log ---')

        detection_summary = self.detection_manager.calculate_from_log(results_log)
        quality_summary = self.quality_manager.calculate_from_log(results_log)

        # Save detailed quality metrics if available
        quality_details_path = None
        if 'per_sample_metrics' in quality_summary:
            quality_details = quality_summary.pop('per_sample_metrics')
            # Apply rounding to detailed metrics
            quality_details = round_floats(quality_details)
            quality_details_path = self._save_quality_details(quality_details)

        logger.info('--- 2. Assembling final report ---')

        successful_samples = 0
        for r in results_log:
            status = r.status if isinstance(r, SampleRecord) else r.get('status')
            if status == 'success':
                successful_samples += 1

        final_report = {
            'experiment_info': {
                'id': self.report_experiment_id,
                'name': getattr(self.config, 'experiment_name', 'N/A'),
                'experiment_name': getattr(self.config, 'experiment_name', 'N/A'),
                'timestamp': self.timestamp,
                'base_model': getattr(self.config, 'base_model', None),
                'watermark': getattr(self.config, 'watermark', None),
                'attack': getattr(self.config, 'attack', None),
                'dataset': _dataset_name_from_config(self.config),
            },
            'detection_metrics': detection_summary,
            'quality_metrics': quality_summary,
            'sample_statistics': {
                'total_samples': len(results_log),
                'successful_samples': successful_samples,
            },
            'cost': cost_summary or {},
            'full_config': self.config.model_dump(),  # Include the full config for reproducibility
        }

        # Apply global rounding to the final report
        final_report = round_floats(final_report)

        self._log_summary(final_report)
        summary_path, raw_log_path = self._save_reports_to_disk(final_report, results_log)
        self._save_summary_to_db(
            final_report,
            summary_path=summary_path,
            raw_log_path=raw_log_path,
            quality_details_path=quality_details_path,
        )
        logger.info('--- 3c. Deferring summary logger export to ExperimentLogger ---')

        # --- Update Experiment Index ---
        self._update_experiment_index(summary_path, raw_log_path, quality_details_path)

        logger.info('Reporting phase finished successfully.')
        return final_report

    def _log_summary(self, report: Dict[str, Any]):
        """Logs a human-readable summary of the key metrics to the logger."""
        logger.info('--- Experiment Final Summary ---')
        info = report.get('experiment_info', {})
        logger.info(f'  Experiment Name: {info.get("name")}')
        logger.info(f'  Experiment ID:   {info.get("id")}')

        det_metrics = report.get('detection_metrics', {})
        if det_metrics:
            logger.info('  [Detection Metrics]')
            if det_metrics.get('type') == 'fundamental':
                logger.info(
                    '    - TPR Before Attack: '
                    f'{_format_optional_float(det_metrics.get("detection_rate_before_attack"))}'
                )
                logger.info(
                    '    - TPR After Attack:  '
                    f'{_format_optional_float(det_metrics.get("detection_rate_after_attack"))}'
                )
                logger.info(
                    '    - FPR (Natural):     '
                    f'{_format_optional_float(det_metrics.get("detection_rate_natural"))}'
                )
                logger.info(
                    '    - Attack Success Rate (ASR): '
                    f'{_format_optional_float(det_metrics.get("attack_success_rate"))}'
                )
            elif det_metrics.get('type') == 'spoof':
                logger.info(
                    '    - Spoof Success Rate (SSR): '
                    f'{_format_optional_float(det_metrics.get("spoof_success_rate"))}'
                )
                logger.info(
                    '    - FPR (Natural):           '
                    f'{_format_optional_float(det_metrics.get("detection_rate_natural (FPR)"))}'
                )
                logger.info(
                    '    - Baseline Watermark TPR:  '
                    f'{_format_optional_float(det_metrics.get("detection_rate_baseline_watermark (TPR)"))}'
                )
            elif det_metrics.get('type') == 'dynamic':
                logger.info(
                    f'    - Threshold:         {_format_optional_float(det_metrics.get("threshold"))}'
                )
                logger.info(
                    '    - TPR Before Attack: '
                    f'{_format_optional_float(det_metrics.get("detection_rate_before_attack"))}'
                )
                logger.info(
                    '    - TPR After Attack:  '
                    f'{_format_optional_float(det_metrics.get("detection_rate_after_attack"))}'
                )
                logger.info(
                    '    - Baseline FPR:      '
                    f'{_format_optional_float(det_metrics.get("calibration_metrics", {}).get("FPR"))}'
                )
                logger.info(
                    '    - Attack Success Rate (ASR): '
                    f'{_format_optional_float(det_metrics.get("attack_success_rate"))}'
                )

        qual_metrics = report.get('quality_metrics', {})
        if qual_metrics:
            logger.info('  [Quality Metrics]')
            for key, value in qual_metrics.items():
                if isinstance(value, dict):  # For nested structure
                    logger.info(f'    - {key.replace("_", " ").title()}:')
                    for sub_key, sub_value in value.items():
                        if isinstance(sub_value, float):
                            logger.info(
                                f'      - {sub_key.replace("_", " ").title()}: {sub_value:.4f}'
                            )
                        else:
                            logger.info(f'      - {sub_key.replace("_", " ").title()}: {sub_value}')
                elif isinstance(value, float):
                    logger.info(f'    - {key.replace("_", " ").title()}: {value:.4f}')
        logger.info('--------------------------------')

    def _save_reports_to_disk(self, report: Dict[str, Any], results_log: list):
        """Saves the final summary report and raw results log to the output directory."""
        logger.info('--- 3a. Saving reports to disk ---')

        summary_path = self.output_dir / 'summary_report.json'
        raw_log_path = None

        # Save summary report
        try:
            with open(summary_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, default=json_converter, indent=4)
            logger.info(f'Summary report saved to: {summary_path}')
        except Exception:
            logger.error('Failed to save summary report.', exc_info=True)

        # Save raw results log if configured
        save_result_texts = False
        if hasattr(self.config, 'runner_settings'):
            if isinstance(self.config.runner_settings, dict):
                save_result_texts = self.config.runner_settings.get('save_result_texts', False)
            else:
                save_result_texts = getattr(self.config.runner_settings, 'save_result_texts', False)

        if save_result_texts:
            raw_log_path = self.output_dir / 'raw_results.jsonl'  # needs to be jsonl
            try:
                with open(raw_log_path, 'w', encoding='utf-8') as f:
                    for entry in results_log:
                        data_to_dump = entry
                        if isinstance(entry, SampleRecord):
                            data_to_dump = entry.model_dump()

                        # Apply rounding
                        data_to_dump = round_floats(data_to_dump)
                        f.write(json.dumps(data_to_dump, default=json_converter) + '\n')

                logger.info(f'Raw results log saved to: {raw_log_path}')
            except Exception:
                logger.error('Failed to save raw results log.', exc_info=True)

        return summary_path, raw_log_path

    def _save_quality_details(self, details: list):
        """Saves the detailed quality metrics for each sample to disk."""
        details_path = self.output_dir / 'quality_metric_details.json'
        try:
            with open(details_path, 'w', encoding='utf-8') as f:
                json.dump(details, f, default=json_converter, indent=4)
            logger.info(f'Quality metric details saved to: {details_path}')
            return details_path
        except Exception:
            logger.error('Failed to save quality metric details.', exc_info=True)
            return None

    def _update_experiment_index(self, summary_path, raw_log_path, quality_details_path):
        """Updates the global experiment index file."""
        logger.info('--- 3d. Updating experiment index ---')

        # Calculate config hash
        try:
            config_dump = self.config.model_dump()
            config_str = json.dumps(config_dump, sort_keys=True, default=json_converter)
            config_hash = hashlib.md5(config_str.encode('utf-8')).hexdigest()
        except Exception as e:
            logger.warning(f'Failed to calculate config hash: {e}')
            config_hash = 'error'

        index_record = {
            'experiment_id': self.report_experiment_id,
            'timestamp': self.timestamp,
            'experiment_name': getattr(self.config, 'experiment_name', 'N/A'),
            'base_model': getattr(self.config, 'base_model', 'N/A'),
            'watermark': getattr(self.config, 'watermark', 'N/A'),
            'attack': getattr(self.config, 'attack', 'N/A'),
            'dataset': _dataset_name_from_config(self.config) or 'N/A',
            'summary_report_path': str(summary_path.absolute()) if summary_path else None,
            'raw_results_path': str(raw_log_path.absolute()) if raw_log_path else None,
            'quality_metric_details_path': str(quality_details_path.absolute())
            if quality_details_path
            else None,
            'config_hash': config_hash,
        }

        # Path to the index file
        # self.output_dir is usually .../output/result/mmw_mixed/SIRA/DIP/timestamp
        # We want .../output/result/experiment_index.jsonl

        # Find the 'result' directory. Assuming standard structure.
        # Fallback: go up 4 levels if standard structure matches
        # Better: use project root if available in config, otherwise infer

        try:
            # Assuming output_dir is nested under result root
            # Let's try to find "result" in the path parts
            output_path = self.output_dir.absolute()
            parts = output_path.parts
            if 'result' in parts:
                # Join until 'result'
                idx = parts.index('result')
                result_dir = output_path.parents[len(parts) - 1 - idx - 1]
            else:
                # Fallback: project_root/output/result
                project_root = getattr(self.config, 'project_root', '.')
                result_dir = os.path.join(project_root, 'output', 'result')
                if not os.path.exists(result_dir):
                    os.makedirs(result_dir, exist_ok=True)
                result_dir = Path(result_dir)

            index_file = result_dir / 'experiment_index.jsonl'

            with open(index_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(index_record, default=json_converter) + '\n')

            logger.info(f'Experiment recorded in index: {index_file}')

        except Exception as e:
            logger.error(f'Failed to update experiment index: {e}', exc_info=True)

        run_registry = getattr(self.context, 'run_registry', None)
        if run_registry is not None:
            run_registry.update_artifact_paths(
                summary_report_path=str(summary_path.absolute()) if summary_path else None,
                raw_results_path=str(raw_log_path.absolute()) if raw_log_path else None,
                quality_metric_details_path=str(quality_details_path.absolute())
                if quality_details_path
                else None,
            )

    def _save_summary_to_db(
        self,
        report: Dict[str, Any],
        *,
        summary_path: Path | None = None,
        raw_log_path: Path | None = None,
        quality_details_path: Path | None = None,
    ):
        """Saves the flattened summary report to the database if enabled."""
        db_cfg = getattr(self.config, 'database', {})
        if not db_cfg or not db_cfg.get('enabled'):
            logger.info('Database logging is disabled. Skipping.')
            return

        logger.info('--- 3b. Saving summary to database ---')
        try:
            db_connection_info = self._load_db_connection_info(db_cfg)
            table_name = _spoof_table_name_for_report(
                db_cfg.get('table_name', 'experiment_summary'),
                report,
            )
            if table_name == 'experiment_summary' and not _should_persist_to_benchmark_table(
                report
            ):
                logger.info(
                    'Skipping database logging for non-benchmark run: '
                    'dataset.num_samples=%s (formal benchmark requires %s).',
                    _dataset_num_samples_from_report(report),
                    FORMAL_BENCHMARK_NUM_SAMPLES,
                )
                return

            with DatabaseManager(db_connection_info) as db:
                flat_data = self._flatten_report_for_db(
                    report,
                    summary_path=summary_path,
                    raw_log_path=raw_log_path,
                    quality_details_path=quality_details_path,
                )
                if self._try_backfill_existing_quality_row(db, table_name, flat_data):
                    return
                if self._is_quality_only_backfill_run() and not self._can_upsert_quality_only_row(
                    db, table_name, flat_data
                ):
                    logger.warning(
                        'Skipping quality-only sqlite write because no existing summary row '
                        'was found for experiment_id=%s.',
                        flat_data.get('experiment_id'),
                    )
                    return
                db.upsert(table_name, flat_data)

        except Exception:
            logger.error('Failed to save summary to database.', exc_info=True)

    def _flatten_report_for_db(
        self,
        report: Dict[str, Any],
        *,
        summary_path: Path | None = None,
        raw_log_path: Path | None = None,
        quality_details_path: Path | None = None,
    ) -> Dict[str, Any]:
        """Prepares a flat dictionary of the most important metrics for the database table."""
        if _report_protocol(report) == 'spoofing':
            return self._flatten_spoof_report_for_db(
                report,
                summary_path=summary_path,
                raw_log_path=raw_log_path,
                quality_details_path=quality_details_path,
            )

        info = report.get('experiment_info', {})
        det_metrics = report.get('detection_metrics', {})
        qual_metrics = _select_primary_quality_bucket(report)
        stats = report.get('sample_statistics') or {}

        summary_report_path = _normalize_path(summary_path)
        raw_results_path = _normalize_path(raw_log_path)
        quality_metric_details_path = _normalize_path(quality_details_path)
        config_obj = getattr(self, 'config', None)
        raw_texts_path = raw_results_path or _normalize_path(getattr(config_obj, 'data_path', None))

        def to_float(val):
            try:
                return float(val)
            except (ValueError, TypeError, AttributeError):
                return None

        def to_int(val):
            try:
                return int(val)
            except (ValueError, TypeError, AttributeError):
                return None

        raw_text_count = to_int(stats.get('total_samples'))
        if raw_text_count is None:
            raw_text_count = _dataset_num_samples_from_report(report)
        context_obj = getattr(self, 'context', None)

        flat_data = {
            'experiment_id': info.get('id'),
            'timestamp': info.get('timestamp'),
            'experiment_name': info.get('name'),
            'dataset_name': info.get('dataset')
            or _dataset_name_from_config(config_obj)
            or _dataset_name_from_report(report),
            'base_model': info.get('base_model'),
            'watermark': info.get('watermark'),
            'watermark_setting': _derive_watermark_setting(report),
            'attack': info.get('attack'),
            'lifecycle_mode': _infer_lifecycle_mode(report),
            'source_run_id': _dashboard_value(config_obj, 'source_run_id') or info.get('id'),
            'writer_run_id': str(getattr(context_obj, 'experiment_id', '') or '').strip()
            or info.get('id'),
            'job_kind': _dashboard_value(config_obj, 'job_kind') or 'primary',
            'pipeline_mode': getattr(config_obj, 'pipeline_mode', None),
            # Detection metrics
            'tpr_before': to_float(det_metrics.get('detection_rate_before_attack')),
            'tpr_after': to_float(det_metrics.get('detection_rate_after_attack')),
            'asr': to_float(det_metrics.get('attack_success_rate')),
            'fpr': to_float(
                _first_present(
                    det_metrics.get('detection_rate_natural (FPR)'),
                    det_metrics.get('calibration_metrics', {}).get('FPR'),
                )
            ),
            'threshold': to_float(det_metrics.get('threshold')),
            # Quality metrics - extract a few key ones
            'bertscore_atk_vs_wm': to_float(qual_metrics.get('BERTScore')),
            'bleu_atk_vs_wm': to_float(qual_metrics.get('BLEU')),
            'psp_atk_vs_wm': to_float(qual_metrics.get('PSP')),
            'prom_atk_vs_wm': to_float(qual_metrics.get('Prometheus')),
            'qsr_prom2_atk_vs_wm': to_float(qual_metrics.get('QSR@Prom2')),
            'qsr_prom3_atk_vs_wm': to_float(
                _first_present(qual_metrics.get('QSR@Prom3'), qual_metrics.get('QSR'))
            ),
            'qsr_prom4_atk_vs_wm': to_float(qual_metrics.get('QSR@Prom4')),
            'qsr_prom5_atk_vs_wm': to_float(qual_metrics.get('QSR@Prom5')),
            'qsr_atk_vs_wm': to_float(
                _first_present(qual_metrics.get('QSR@Prom3'), qual_metrics.get('QSR'))
            ),
            'prepare_seconds': to_float((report.get('cost') or {}).get('prepare_seconds')),
            'attack_seconds': to_float((report.get('cost') or {}).get('attack_seconds')),
            'total_seconds': to_float((report.get('cost') or {}).get('total_seconds')),
            # Artifact pointers for future metric backfills from sqlite.
            'summary_report_path': summary_report_path,
            'raw_results_path': raw_results_path,
            'quality_metric_details_path': quality_metric_details_path,
            'raw_texts_path': raw_texts_path,
            'raw_text_count': raw_text_count,
            # Add more specific metrics you want to track
            'config_json': json.dumps(report.get('full_config'), default=json_converter),
        }
        # Filter out any None values to avoid DB errors with non-nullable columns if you have them
        return {k: v for k, v in flat_data.items() if v is not None}

    def _flatten_spoof_report_for_db(
        self,
        report: Dict[str, Any],
        *,
        summary_path: Path | None = None,
        raw_log_path: Path | None = None,
        quality_details_path: Path | None = None,
    ) -> Dict[str, Any]:
        info = report.get('experiment_info', {})
        det_metrics = report.get('detection_metrics', {})
        qual_metrics = _select_primary_quality_bucket(report)
        stats = report.get('sample_statistics') or {}

        summary_report_path = _normalize_path(summary_path)
        raw_results_path = _normalize_path(raw_log_path)
        quality_metric_details_path = _normalize_path(quality_details_path)
        config_obj = getattr(self, 'config', None)
        raw_texts_path = raw_results_path or _normalize_path(getattr(config_obj, 'data_path', None))

        def to_float(val):
            try:
                return float(val)
            except (ValueError, TypeError, AttributeError):
                return None

        def to_int(val):
            try:
                return int(val)
            except (ValueError, TypeError, AttributeError):
                return None

        raw_text_count = to_int(stats.get('total_samples'))
        if raw_text_count is None:
            raw_text_count = _dataset_num_samples_from_report(report)

        context_obj = getattr(self, 'context', None)
        attack_name = str(info.get('attack') or '').strip()
        artifact_metadata = _extract_spoof_artifact_metadata(report)

        primary_quality_bucket = _primary_quality_bucket_name(report) or 'spoof_vs_watermarked'
        persisted_quality_metrics = (
            qual_metrics if primary_quality_bucket == 'spoof_vs_watermarked' else {}
        )

        flat_data = {
            'experiment_id': info.get('id'),
            'timestamp': info.get('timestamp'),
            'experiment_name': info.get('name'),
            'dataset_name': info.get('dataset')
            or _dataset_name_from_config(config_obj)
            or _dataset_name_from_report(report),
            'base_model': info.get('base_model'),
            'watermark': info.get('watermark'),
            'watermark_setting': _derive_watermark_setting(report),
            'attack': attack_name,
            'evaluation_protocol': 'spoofing',
            'lifecycle_mode': _infer_lifecycle_mode(report),
            'primary_quality_bucket': primary_quality_bucket,
            'source_run_id': _dashboard_value(config_obj, 'source_run_id') or info.get('id'),
            'writer_run_id': str(getattr(context_obj, 'experiment_id', '') or '').strip()
            or info.get('id'),
            'job_kind': _dashboard_value(config_obj, 'job_kind') or 'primary',
            'pipeline_mode': getattr(config_obj, 'pipeline_mode', None),
            'ssr': to_float(det_metrics.get('spoof_success_rate')),
            'fpr': to_float(det_metrics.get('detection_rate_natural (FPR)')),
            'baseline_watermark_tpr': to_float(
                det_metrics.get('detection_rate_baseline_watermark (TPR)')
            ),
            'threshold': to_float(det_metrics.get('threshold')),
            'bertscore_spoof_vs_wm': to_float(persisted_quality_metrics.get('BERTScore')),
            'bleu_spoof_vs_wm': to_float(persisted_quality_metrics.get('BLEU')),
            'psp_spoof_vs_wm': to_float(persisted_quality_metrics.get('PSP')),
            'prom_spoof_vs_wm': to_float(persisted_quality_metrics.get('Prometheus')),
            'qsr_prom2_spoof_vs_wm': to_float(persisted_quality_metrics.get('QSR@Prom2')),
            'qsr_prom3_spoof_vs_wm': to_float(
                _first_present(
                    persisted_quality_metrics.get('QSR@Prom3'),
                    persisted_quality_metrics.get('QSR'),
                )
            ),
            'qsr_prom4_spoof_vs_wm': to_float(persisted_quality_metrics.get('QSR@Prom4')),
            'qsr_prom5_spoof_vs_wm': to_float(persisted_quality_metrics.get('QSR@Prom5')),
            'qsr_spoof_vs_wm': to_float(
                _first_present(
                    persisted_quality_metrics.get('QSR@Prom3'),
                    persisted_quality_metrics.get('QSR'),
                )
            ),
            'attacker_model': artifact_metadata.get('attacker_model'),
            'prepare_artifact_path': artifact_metadata.get('prepare_artifact_path'),
            'prepare_meta_path': artifact_metadata.get('prepare_meta_path'),
            'artifact_family': artifact_metadata.get('artifact_family'),
            'capability_profile': artifact_metadata.get('capability_profile'),
            'prepare_seconds': to_float((report.get('cost') or {}).get('prepare_seconds')),
            'attack_seconds': to_float((report.get('cost') or {}).get('attack_seconds')),
            'total_seconds': to_float((report.get('cost') or {}).get('total_seconds')),
            'summary_report_path': summary_report_path,
            'raw_results_path': raw_results_path,
            'quality_metric_details_path': quality_metric_details_path,
            'raw_texts_path': raw_texts_path,
            'raw_text_count': raw_text_count,
            'config_json': json.dumps(report.get('full_config'), default=json_converter),
        }
        return {k: v for k, v in flat_data.items() if v is not None}

    def _is_quality_only_backfill_run(self) -> bool:
        eval_cfg = getattr(self.config, 'evaluation', None)
        if getattr(self.config, 'pipeline_mode', None) != 'evaluate' or eval_cfg is None:
            return False
        return bool(getattr(eval_cfg, 'skip_detection_in_evaluate', False))

    def _try_backfill_existing_quality_row(
        self, db: DatabaseManager, table_name: str, flat_data: Dict[str, Any]
    ) -> bool:
        if table_name != 'experiment_summary' or not self._is_quality_only_backfill_run():
            return False

        experiment_id = flat_data.get('experiment_id')
        update_values = {
            key: flat_data[key]
            for key in QUALITY_BACKFILL_COLUMNS
            if flat_data.get(key) is not None
        }
        if not update_values or not experiment_id:
            return False

        existing_rows = db.query(
            table_name,
            select_keys=['experiment_id', *QUALITY_BACKFILL_COLUMNS],
            where_conditions={'experiment_id': experiment_id},
        )
        if not existing_rows:
            return False

        db.update(table_name, update_values, {'experiment_id': experiment_id})
        logger.info(
            'Backfilled quality-only metrics into existing row by experiment_id: %s (%s)',
            experiment_id,
            ', '.join(sorted(update_values.keys())),
        )
        return True

    def _can_upsert_quality_only_row(
        self, db: DatabaseManager, table_name: str, flat_data: Dict[str, Any]
    ) -> bool:
        if table_name != 'experiment_summary' or not self._is_quality_only_backfill_run():
            return True

        experiment_id = flat_data.get('experiment_id')
        if not experiment_id:
            return False

        existing_rows = db.query(
            table_name,
            select_keys=['experiment_id'],
            where_conditions={'experiment_id': experiment_id},
        )
        return bool(existing_rows)

    def ensure_db_schema(self):
        """
        Checks if the summary table and all required columns exist, creating/altering them if not.
        """
        # Pydantic model does not have .get(), use getattr for optional fields (especially with extra="allow")
        db_cfg = getattr(self.config, 'database', {})
        # If db_cfg is None (explicitly null in config), treat as empty dict
        if db_cfg is None:
            db_cfg = {}

        if not db_cfg.get('enabled'):
            return

        table_name = _spoof_table_name_for_config(
            db_cfg.get('table_name', 'experiment_summary'),
            self.config,
        )

        required_columns = self._required_columns_for_table(table_name)

        try:
            logger.info('--- 5. Verifying database schema ---')
            db_connection_info = self._load_db_connection_info(db_cfg)

            with DatabaseManager(db_connection_info) as db:
                if not db.table_exists(table_name):
                    logger.info(f"Table '{table_name}' not found. Creating it...")
                    columns_def = ', '.join(
                        [f'{name} {definition}' for name, definition in required_columns.items()]
                    )
                    db.create_table(table_name, columns_def)
                else:
                    logger.info(f"Table '{table_name}' exists. Verifying columns...")
                    for col_name, col_def in required_columns.items():
                        # ALTER TABLE ADD COLUMN with PRIMARY KEY is unsupported in sqlite/postgres.
                        add_def = col_def.replace(' PRIMARY KEY', '')
                        db.add_column(table_name, f'{col_name} {add_def}')
            logger.info('Database schema verification complete.')

        except Exception as e:
            logger.error(f'Failed during database schema verification: {e}', exc_info=True)
            raise

    def _required_columns_for_table(self, table_name: str) -> Dict[str, str]:
        if table_name == SPOOF_TABLE_NAME:
            return {
                'experiment_id': 'VARCHAR(36) PRIMARY KEY',
                'timestamp': 'TIMESTAMP',
                'experiment_name': 'VARCHAR(255)',
                'dataset_name': 'VARCHAR(255)',
                'base_model': 'VARCHAR(255)',
                'watermark': 'VARCHAR(100)',
                'watermark_setting': 'TEXT',
                'attack': 'VARCHAR(100)',
                'evaluation_protocol': 'TEXT',
                'lifecycle_mode': 'TEXT',
                'primary_quality_bucket': 'TEXT',
                'source_run_id': 'VARCHAR(36)',
                'writer_run_id': 'VARCHAR(36)',
                'job_kind': 'TEXT',
                'pipeline_mode': 'TEXT',
                'ssr': 'FLOAT',
                'fpr': 'FLOAT',
                'baseline_watermark_tpr': 'FLOAT',
                'threshold': 'FLOAT',
                'bertscore_spoof_vs_wm': 'FLOAT',
                'bleu_spoof_vs_wm': 'FLOAT',
                'psp_spoof_vs_wm': 'FLOAT',
                'prom_spoof_vs_wm': 'FLOAT',
                'qsr_prom2_spoof_vs_wm': 'FLOAT',
                'qsr_prom3_spoof_vs_wm': 'FLOAT',
                'qsr_prom4_spoof_vs_wm': 'FLOAT',
                'qsr_prom5_spoof_vs_wm': 'FLOAT',
                'qsr_spoof_vs_wm': 'FLOAT',
                'attacker_model': 'TEXT',
                'prepare_artifact_path': 'TEXT',
                'prepare_meta_path': 'TEXT',
                'artifact_family': 'TEXT',
                'capability_profile': 'TEXT',
                'prepare_seconds': 'FLOAT',
                'attack_seconds': 'FLOAT',
                'total_seconds': 'FLOAT',
                'summary_report_path': 'TEXT',
                'raw_results_path': 'TEXT',
                'quality_metric_details_path': 'TEXT',
                'raw_texts_path': 'TEXT',
                'raw_text_count': 'INTEGER',
                'analysis_status': 'TEXT',
                'config_json': 'JSON',
            }
        return {
            'experiment_id': 'VARCHAR(36) PRIMARY KEY',
            'timestamp': 'TIMESTAMP',
            'experiment_name': 'VARCHAR(255)',
            'dataset_name': 'VARCHAR(255)',
            'base_model': 'VARCHAR(255)',
            'watermark': 'VARCHAR(100)',
            'watermark_setting': 'TEXT',
            'attack': 'VARCHAR(100)',
            'lifecycle_mode': 'TEXT',
            'source_run_id': 'VARCHAR(36)',
            'writer_run_id': 'VARCHAR(36)',
            'job_kind': 'TEXT',
            'pipeline_mode': 'TEXT',
            'tpr_before': 'FLOAT',
            'tpr_after': 'FLOAT',
            'asr': 'FLOAT',
            'fpr': 'FLOAT',
            'threshold': 'FLOAT',
            'bertscore_atk_vs_wm': 'FLOAT',
            'bleu_atk_vs_wm': 'FLOAT',
            'psp_atk_vs_wm': 'FLOAT',
            'prom_atk_vs_wm': 'FLOAT',
            'qsr_prom2_atk_vs_wm': 'FLOAT',
            'qsr_prom3_atk_vs_wm': 'FLOAT',
            'qsr_prom4_atk_vs_wm': 'FLOAT',
            'qsr_prom5_atk_vs_wm': 'FLOAT',
            'qsr_atk_vs_wm': 'FLOAT',
            'prepare_seconds': 'FLOAT',
            'attack_seconds': 'FLOAT',
            'total_seconds': 'FLOAT',
            'summary_report_path': 'TEXT',
            'raw_results_path': 'TEXT',
            'quality_metric_details_path': 'TEXT',
            'raw_texts_path': 'TEXT',
            'raw_text_count': 'INTEGER',
            'analysis_status': 'TEXT',
            'config_json': 'JSON',
        }

    def _load_db_connection_info(self, db_cfg: Dict[str, Any]) -> Dict[str, Any]:
        """Loads DB connection settings from inline config or config file (json/yaml)."""
        if db_cfg.get('connection'):
            return db_cfg['connection']

        config_path = db_cfg.get('config_path', 'configs/db/db_config.json')
        path = Path(config_path).expanduser()
        if not path.is_absolute():
            path = Path.cwd() / path

        if not path.exists():
            raise FileNotFoundError(f'Database config file not found: {path}')

        suffix = path.suffix.lower()
        with open(path, 'r', encoding='utf-8') as f:
            if suffix in {'.yaml', '.yml'}:
                data = yaml.safe_load(f) or {}
            elif suffix == '.json':
                data = json.load(f)
            else:
                raise ValueError(f'Unsupported DB config format: {path}. Use .json/.yaml/.yml.')

        if not isinstance(data, dict):
            raise ValueError(f'DB config must be a mapping/dict: {path}')
        return data
