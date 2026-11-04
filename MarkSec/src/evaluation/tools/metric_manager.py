from collections import defaultdict
import json
from typing import Any, Dict, List, Union
from sklearn.metrics import roc_auc_score
from src.evaluation.result_schema import SampleRecord
from src.evaluation.tools.text_quality_analyzer import (
    PPLCalculator,
    LogDiversityAnalyzer,
    BERTScoreCalculator,
    BLEUCalculator,
    ROUGE1Calculator,
    ROUGE2Calculator,
    ROUGELCalculator,
    PSPCalculator,
    DirectTextQualityAnalyzer,
    ReferencedTextQualityAnalyzer,
    ExternalDiscriminatorTextQualityAnalyzer,
    LLMScoreEvaluator,
    NLIScoreCalculator,
    SemanticFluencyAnalyzer,
    LWSSCalculator,
    KERCalculator,
    ECCalculator,
    PrometheusEvaluator,
)
from src.evaluation.tools.success_rate_calculator import (
    DetectionResult,
    DynamicThresholdSuccessRateCalculator,
)
from src.util.log import get_logger

logger = get_logger()

METRIC_CALCULATOR_MAP = {
    'PPL': PPLCalculator,
    'LogDiv': LogDiversityAnalyzer,
    'BERTScore': BERTScoreCalculator,
    'BLEU': BLEUCalculator,
    'ROUGE1': ROUGE1Calculator,
    'ROUGE2': ROUGE2Calculator,
    'ROUGEL': ROUGELCalculator,
    'PSP': PSPCalculator,
    'LLMScore': LLMScoreEvaluator,
    'Prometheus': PrometheusEvaluator,
    'NLIScore': NLIScoreCalculator,
    'SemanticFluency': SemanticFluencyAnalyzer,
    'LWSS': LWSSCalculator,
    'KER': KERCalculator,
    'EC': ECCalculator,
}

HF_BACKED_QUALITY_METRICS = {
    'BERTScore',
    'Prometheus',
    'NLIScore',
    'SemanticFluency',
    'LWSS',
    'EC',
}


def _qsr_metric_label(metric_name: str) -> str:
    normalized = str(metric_name or '').strip()
    if normalized.lower() == 'prometheus':
        return 'Prom'
    return normalized


def _format_qsr_threshold(threshold: float) -> str:
    threshold_value = float(threshold)
    if threshold_value.is_integer():
        return str(int(threshold_value))
    return str(threshold_value)


def _prediction_value(result: Any) -> bool | None:
    if result is None:
        return None
    if isinstance(result, dict):
        prediction = result.get('prediction')
    else:
        prediction = getattr(result, 'prediction', None)
    if prediction is None:
        return None
    return bool(prediction)


def _sample_attack_succeeded(sample_log: Union[Dict[str, Any], SampleRecord]) -> bool | None:
    if isinstance(sample_log, SampleRecord):
        detection_watermark = sample_log.detection_watermark
        detection_attack = sample_log.detection_attack
    else:
        detection_watermark = sample_log.get('detection_watermark')
        detection_attack = sample_log.get('detection_attack')

    watermark_prediction = _prediction_value(detection_watermark)
    attack_prediction = _prediction_value(detection_attack)
    if watermark_prediction is None or attack_prediction is None:
        return None
    return watermark_prediction and not attack_prediction


class QualityMetricManager:
    def __init__(
        self,
        config: dict,
        models: dict,
        device: str,
        qsr_metric: str = 'Prometheus',
        qsr_threshold: float = 3.0,
        qsr_thresholds: list[float] | None = None,
        comparison_names: list[str] | None = None,
        local_files_only: bool = False,
    ):
        # config: from YAML, specifies which metrics to use and their params
        # models: a dict of pre-loaded models from ModelManager
        self.metrics = {}
        self.device = device
        self.qsr_metric = str(qsr_metric or 'Prometheus')
        self.qsr_threshold = float(qsr_threshold)
        self.local_files_only = bool(local_files_only)
        self.qsr_thresholds = sorted(
            {
                self.qsr_threshold,
                *[float(threshold) for threshold in (qsr_thresholds or [self.qsr_threshold])],
            }
        )
        self.comparison_names = {
            str(name).strip() for name in (comparison_names or []) if str(name).strip()
        }
        self._initialize_metrics(config, models)
        self.direct_metrics = {}
        self.referenced_metrics = {}
        self.external_metrics = {}

        # --- Pre-classify metrics for efficiency ---
        for name, instance in self.metrics.items():
            if isinstance(instance, DirectTextQualityAnalyzer):
                self.direct_metrics[name] = instance
            elif isinstance(instance, ReferencedTextQualityAnalyzer):
                self.referenced_metrics[name] = instance
            elif isinstance(instance, ExternalDiscriminatorTextQualityAnalyzer):
                self.external_metrics[name] = instance

        logger.info(
            f'Classified metrics: {len(self.direct_metrics)} direct, '
            f'{len(self.referenced_metrics)} referenced, '
            f'{len(self.external_metrics)} external.'
        )

    def _calculate_qsr(
        self,
        score_list: list[float],
        threshold: float,
        success_flags: list[bool | None] | None = None,
    ) -> float:
        if not score_list:
            return 0.0
        if success_flags is not None and len(success_flags) == len(score_list):
            passed = sum(
                1
                for score, attack_succeeded in zip(score_list, success_flags)
                if bool(attack_succeeded) and score >= threshold
            )
        else:
            passed = sum(1 for score in score_list if score >= threshold)
        return float(round(passed / len(score_list), 4))

    def _initialize_metrics(self, config: dict, models: dict):
        """
        Dynamically initializes metric calculators based on the config and the mapping.
        """
        logger.info('Initializing quality metric calculators...')
        for metric_name, metric_config in config.items():
            if metric_name not in METRIC_CALCULATOR_MAP:
                logger.warning(f"Metric '{metric_name}' is not supported. Skipping.")
                continue

            CalculatorClass = METRIC_CALCULATOR_MAP[metric_name]

            try:
                if metric_name == 'PPL':
                    instance = CalculatorClass(
                        model=models['base_model']['model'],
                        tokenizer=models['base_model']['tokenizer'],
                        device=self.device,
                    )
                elif metric_name == 'Prometheus':
                    # Skip initialization if OOM or explicit disable
                    try:
                        init_config = dict(metric_config or {})
                        init_config.setdefault('local_files_only', self.local_files_only)
                        instance = CalculatorClass(**init_config)
                    except Exception as e:
                        logger.warning(
                            f'Failed to initialize Prometheus (likely OOM). Skipping. Error: {e}'
                        )
                        continue
                elif metric_config:
                    # For metrics that take parameters from the config (like BERTScore)
                    init_config = dict(metric_config)
                    if metric_name in HF_BACKED_QUALITY_METRICS:
                        init_config.setdefault('local_files_only', self.local_files_only)
                    instance = CalculatorClass(**init_config)
                else:
                    # For metrics with no parameters (like BLEU, ROUGE)
                    if metric_name in HF_BACKED_QUALITY_METRICS:
                        instance = CalculatorClass(local_files_only=self.local_files_only)
                    else:
                        instance = CalculatorClass()

                self.metrics[metric_name] = instance
                logger.info(f'  - Successfully initialized: {metric_name}')

            except Exception as e:
                logger.error(f"Failed to initialize metric '{metric_name}': {e}", exc_info=True)

    def analyze_sample(self, sample_log: Union[Dict[str, Any], SampleRecord]) -> Dict[str, Any]:
        """
        Analyzes a single sample log and returns a structured dict of all quality scores.

        Args:
            sample_log (Union[Dict[str, Any], SampleRecord]): A single entry from the `results_log`.

        Returns:
            Dict[str, Any]: A nested dictionary with absolute and comparison scores.
        """
        # Define the texts to be analyzed from the log
        # Priority: Use full_{type}_text if available (for quality), otherwise use {type}_text (fragment)

        prompt = ''
        source_text = ''
        reference = ''

        if isinstance(sample_log, SampleRecord):
            texts = {
                'natural': sample_log.full_natural_text or sample_log.natural_text or '',
                'watermarked': sample_log.full_watermarked_text
                or sample_log.watermarked_text
                or '',
                'attacked': sample_log.full_attacked_text or sample_log.attacked_text or '',
                'spoof': sample_log.full_spoof_text or sample_log.spoof_text or '',
            }
            prompt = sample_log.prompt
            source_text = sample_log.source_text or ''
            reference = sample_log.reference or ''
        else:
            texts = {
                'natural': sample_log.get('full_natural_text')
                or sample_log.get('natural_text')
                or '',
                'watermarked': sample_log.get('full_watermarked_text')
                or sample_log.get('watermarked_text')
                or '',
                'attacked': sample_log.get('full_attacked_text')
                or sample_log.get('attacked_text')
                or '',
                'spoof': sample_log.get('full_spoof_text') or sample_log.get('spoof_text') or '',
            }
            prompt = sample_log.get('prompt', '')
            source_text = sample_log.get('source_text', '')
            reference = sample_log.get('reference', '')

        # --- 1. Calculate Absolute Quality Metrics (PPL, LogDiv, etc.) ---
        absolute_scores = defaultdict(dict)

        for text_type, text_content in texts.items():
            if not text_content:
                continue
            for metric_name, calculator in self.direct_metrics.items():
                try:
                    # For PPL, check if text_content already starts with prompt (full text).
                    # If so, analyze directly. If not (fragment), prepend prompt.
                    content_to_analyze = text_content
                    if (
                        metric_name == 'PPL'
                        and prompt
                        and not content_to_analyze.startswith(prompt[:20])
                    ):  # Heuristic check
                        # Only prepend if it looks like a fragment
                        content_to_analyze = prompt + text_content

                    score = calculator.analyze(content_to_analyze)
                    absolute_scores[text_type][metric_name] = score
                except Exception as e:
                    logger.debug(
                        f"Error calculating direct metric '{metric_name}' for '{text_type}': {e}"
                    )

        # --- 2. Calculate Comparison/Referenced Metrics (BERTScore, BLEU, etc.) ---
        comparison_scores = defaultdict(dict)
        # Define the pairs to compare
        comparisons = []

        # Compare post-attack outputs against the baseline watermarked text when available.
        if texts['attacked'] and texts['watermarked']:
            comparisons.append(('attacked_vs_watermarked', texts['attacked'], texts['watermarked']))
        if texts['spoof'] and texts['watermarked']:
            comparisons.append(('spoof_vs_watermarked', texts['spoof'], texts['watermarked']))

        # Gating Logic:
        # If Reference exists (Instruction/QA/Summarization)
        if reference:
            if texts['watermarked']:
                comparisons.append(('watermarked_vs_reference', texts['watermarked'], reference))
            if texts['attacked']:
                comparisons.append(('attacked_vs_reference', texts['attacked'], reference))
            if texts['spoof']:
                comparisons.append(('spoof_vs_reference', texts['spoof'], reference))

        # If Source Text exists (C4/Completion)
        if source_text:
            if texts['watermarked']:
                comparisons.append(('watermarked_vs_source', texts['watermarked'], source_text))
            if texts['attacked']:
                comparisons.append(('attacked_vs_source', texts['attacked'], source_text))
            if texts['spoof']:
                comparisons.append(('spoof_vs_source', texts['spoof'], source_text))

        # Fallback/Legacy: If neither, but we have 'natural' (from old logs or mixed usage), use it?
        if not reference and not source_text and texts['natural']:
            if texts['watermarked']:
                comparisons.append(
                    ('watermarked_vs_natural', texts['watermarked'], texts['natural'])
                )
            if texts['attacked']:
                comparisons.append(('attacked_vs_natural', texts['attacked'], texts['natural']))
            if texts['spoof']:
                comparisons.append(('spoof_vs_natural', texts['spoof'], texts['natural']))

        if self.comparison_names:
            comparisons = [
                comparison for comparison in comparisons if comparison[0] in self.comparison_names
            ]

        for name, candidate, ref_text in comparisons:
            if not candidate or not ref_text:
                continue
            for metric_name, calculator in self.referenced_metrics.items():
                try:
                    score = calculator.analyze(candidate, ref_text)
                    comparison_scores[name][metric_name] = score
                except Exception as e:
                    logger.debug(
                        f"Error calculating referenced metric '{metric_name}' for '{name}': {e}"
                    )
            for metric_name, calculator in self.external_metrics.items():
                try:
                    score = calculator.analyze(ref_text, candidate, prompt)
                    comparison_scores[name][metric_name] = score
                except Exception as e:
                    logger.debug(
                        f"Error calculating external metric '{metric_name}' for '{name}': {e}"
                    )

        return {
            'absolute_quality': dict(absolute_scores),
            'comparison_metrics': dict(comparison_scores),
        }

    def calculate_all(self, text: str, reference: str, **kwargs) -> dict:
        scores = {}
        for name, instance in self.metrics.items():
            if isinstance(instance, DirectTextQualityAnalyzer):
                scores[name] = instance.analyze(text)
            elif isinstance(instance, ReferencedTextQualityAnalyzer):
                scores[name] = instance.analyze(text, reference)
            elif isinstance(instance, ExternalDiscriminatorTextQualityAnalyzer):
                scores[name] = instance.analyze(text, reference, kwargs.get('prompt'))
            else:
                raise TypeError(f'Unsupported analyzer type: {type(instance)}')
        return scores

    def calculate_from_log(self, results_log: List[Union[Dict[str, Any], SampleRecord]]) -> dict:
        # Check if logs contain pre-computed metrics to suppress warning
        has_precomputed_metrics = False
        if not self.metrics and results_log:
            # Check first few samples to see if metrics are present
            for sample in results_log[:50]:
                metrics = (
                    sample.quality_metrics
                    if isinstance(sample, SampleRecord)
                    else sample.get('quality_metrics')
                )
                if metrics:
                    has_precomputed_metrics = True
                    break

        if not self.metrics:
            if has_precomputed_metrics:
                logger.info(
                    'No active quality metrics configured. Aggregating pre-computed metrics from logs.'
                )
            else:
                logger.warning(
                    'No quality metrics are currently active and no pre-computed metrics found in logs.'
                )
            # Proceed even if self.metrics is empty, as logs might contain the data.
        all_scores = {
            'absolute_quality': defaultdict(lambda: defaultdict(list)),
            'comparison_metrics': defaultdict(lambda: defaultdict(list)),
        }
        qsr_success_flags: dict[str, list[bool | None]] = defaultdict(list)
        detailed_records = []
        num_processed = 0
        for sample_log in results_log:
            status = (
                sample_log.status
                if isinstance(sample_log, SampleRecord)
                else sample_log.get('status')
            )
            if status != 'success':
                continue

            # Use pre-calculated metrics if available to avoid re-computation
            if isinstance(sample_log, SampleRecord):
                sample_scores = sample_log.quality_metrics
                sample_index = sample_log.sample_index
            else:
                sample_scores = sample_log.get('quality_metrics')
                sample_index = sample_log.get('sample_index')

            if not sample_scores:
                sample_scores = self.analyze_sample(sample_log)

            # Store detailed metrics for this sample
            detailed_records.append({'sample_index': sample_index, 'metrics': sample_scores})

            # Append absolute scores
            for text_type, metrics in sample_scores.get('absolute_quality', {}).items():
                for metric_name, score in metrics.items():
                    all_scores['absolute_quality'][text_type][metric_name].append(score)

            # Append comparison scores
            for comp_name, metrics in sample_scores.get('comparison_metrics', {}).items():
                for metric_name, score in metrics.items():
                    all_scores['comparison_metrics'][comp_name][metric_name].append(score)
                if self.qsr_metric in metrics:
                    qsr_success_flags[comp_name].append(_sample_attack_succeeded(sample_log))

            num_processed += 1

        logger.info(f'Aggregated quality metrics from {num_processed} successful samples.')

        # --- Step 3: Calculate averages for all collected scores ---
        final_avg_report = {
            'avg_absolute_quality': defaultdict(dict),
            'avg_comparison_metrics': defaultdict(dict),
        }

        # Average absolute scores
        for text_type, metrics in all_scores['absolute_quality'].items():
            for metric_name, score_list in metrics.items():
                if score_list:
                    avg_score = sum(score_list) / len(score_list)
                    # Apply rounding to ensure consistent reporting
                    final_avg_report['avg_absolute_quality'][text_type][metric_name] = float(
                        round(avg_score, 4)
                    )

        # Average comparison scores
        for comp_name, metrics in all_scores['comparison_metrics'].items():
            for metric_name, score_list in metrics.items():
                if score_list:
                    avg_score = sum(score_list) / len(score_list)
                    # Apply rounding to ensure consistent reporting
                    final_avg_report['avg_comparison_metrics'][comp_name][metric_name] = float(
                        round(avg_score, 4)
                    )

            qsr_source_scores = metrics.get(self.qsr_metric) or []
            if qsr_source_scores:
                qsr_scores = {}
                metric_label = _qsr_metric_label(self.qsr_metric)
                success_flags = None
                if comp_name.startswith('attacked_vs_'):
                    candidate_flags = qsr_success_flags.get(comp_name)
                    if candidate_flags and any(flag is not None for flag in candidate_flags):
                        success_flags = candidate_flags
                for threshold in self.qsr_thresholds:
                    qsr_key = f'QSR@{metric_label}{_format_qsr_threshold(threshold)}'
                    qsr_scores[qsr_key] = self._calculate_qsr(
                        qsr_source_scores,
                        threshold,
                        success_flags=success_flags,
                    )

                final_avg_report['avg_comparison_metrics'][comp_name].update(qsr_scores)

                primary_qsr_key = f'QSR@{metric_label}{_format_qsr_threshold(self.qsr_threshold)}'
                if primary_qsr_key in qsr_scores:
                    final_avg_report['avg_comparison_metrics'][comp_name]['QSR'] = qsr_scores[
                        primary_qsr_key
                    ]

        # Convert defaultdicts back to regular dicts for a clean output
        final_report = json.loads(json.dumps(final_avg_report))
        final_report['qsr_config'] = {
            'metric': self.qsr_metric,
            'threshold': self.qsr_threshold,
            'thresholds': self.qsr_thresholds,
            'rule': 'score >= threshold',
        }
        final_report['per_sample_metrics'] = detailed_records
        return final_report


class DetectionMetricManager:
    """
    A manager to collect detection results and their ground truth labels,
    and then use the appropriate calculator to compute standard detection metrics.

    This manager is a general-purpose tool and does not contain logic specific
    to attack scenarios like ASR.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the manager.

        Args:
            config (Dict[str, Any]): Configuration specifying the calculator type
                                        and its parameters.
                                        e.g., {"type": "dynamic", "params": {"rule": "target_fpr", "target_fpr": 0.01}}
        """
        self.config = config
        self.metric_type = self.config.get('type', 'fundamental').lower()

    @staticmethod
    def _calculate_attack_success_rate(
        detection_rate_before_attack: float, detection_rate_after_attack: float
    ) -> float:
        """Compute ASR in [0, 1]; negative values mean attack failed and are clipped to 0."""
        if detection_rate_before_attack <= 0:
            return 0.0
        raw_asr = (
            detection_rate_before_attack - detection_rate_after_attack
        ) / detection_rate_before_attack
        return float(max(0.0, min(1.0, raw_asr)))

    @staticmethod
    def _prediction_from_threshold(score: float, threshold: float, reverse: bool) -> bool:
        if reverse:
            return score <= threshold
        return score >= threshold

    def _apply_dynamic_threshold_to_log(
        self,
        results_log: List[Union[Dict[str, Any], SampleRecord]],
        threshold: float,
        reverse: bool,
    ) -> None:
        """Backfill per-sample boolean predictions after dynamic threshold calibration."""
        for res in results_log:
            status = res.status if isinstance(res, SampleRecord) else res.get('status')
            if status != 'success':
                continue

            for text_type in ['natural', 'watermark', 'attack', 'spoof']:
                if isinstance(res, SampleRecord):
                    detection_res = getattr(res, f'detection_{text_type}', None)
                    if not detection_res or detection_res.z_score is None:
                        continue
                    detection_res.prediction = self._prediction_from_threshold(
                        detection_res.z_score, threshold, reverse
                    )
                    continue

                detection_dict = res.get(f'detection_{text_type}')
                if not isinstance(detection_dict, dict):
                    continue
                score = detection_dict.get('z_score')
                if not isinstance(score, (float, int)) or isinstance(score, bool):
                    continue
                detection_dict['prediction'] = self._prediction_from_threshold(
                    float(score), threshold, reverse
                )

    def _extract_outputs_from_log(
        self, results_log: List[Union[Dict[str, Any], SampleRecord]]
    ) -> tuple:
        """Helper to extract detection outputs for all three text types from the log."""
        output_key = 'prediction' if self.metric_type in ['fundamental', 'spoof'] else 'z_score'

        natural_outputs, watermarked_outputs, attacked_outputs, spoof_outputs = [], [], [], []

        for res in results_log:
            status = res.status if isinstance(res, SampleRecord) else res.get('status')
            if status != 'success':
                continue

            keys = [
                ('natural', natural_outputs),
                ('watermark', watermarked_outputs),
                ('attack', attacked_outputs),
            ]
            if self.metric_type == 'spoof':
                keys = [
                    ('natural', natural_outputs),
                    ('spoof', spoof_outputs),
                    ('watermark', watermarked_outputs),
                ]
            for text_type, output_list in keys:
                if isinstance(res, SampleRecord):
                    detection_res = getattr(res, f'detection_{text_type}', None)
                    if detection_res:
                        if self.metric_type in ['fundamental', 'spoof']:
                            output = detection_res.prediction
                        else:
                            output = detection_res.z_score
                    else:
                        output = None
                else:
                    detection_dict = res.get(f'detection_{text_type}', {})
                    output = detection_dict.get(output_key)

                if self.metric_type in ['fundamental', 'spoof']:
                    # Allow bool and numpy bool types (checking via duck typing or simple conversion attempt)
                    if output is not None and (
                        isinstance(output, bool) or hasattr(output, 'dtype')
                    ):
                        try:
                            output_list.append(bool(output))
                        except Exception:
                            output_list.append(None)
                    else:
                        output_list.append(None)
                else:
                    if isinstance(output, (float, int)) and not isinstance(output, bool):
                        output_list.append(float(output))
                    else:
                        output_list.append(None)

        # Filter out None values
        if self.metric_type == 'spoof':
            return (
                [o for o in natural_outputs if o is not None],
                [o for o in watermarked_outputs if o is not None],
                [o for o in spoof_outputs if o is not None],
            )
        else:
            return (
                [o for o in natural_outputs if o is not None],
                [o for o in watermarked_outputs if o is not None],
                [o for o in attacked_outputs if o is not None],
            )

    def calculate_from_log(
        self, results_log: List[Union[Dict[str, Any], SampleRecord]]
    ) -> Dict[str, Any]:
        """
        Calculates a full detection summary by processing the entire results_log.
        It routes to the appropriate logic based on the detection type (fundamental or dynamic).
        """
        if not results_log:
            logger.warning('results_log is empty. Skipping detection metric calculation.')
            return {}

        natural_outputs, watermarked_outputs, attacked_outputs = self._extract_outputs_from_log(
            results_log
        )

        if self.metric_type == 'spoof':
            return self._calculate_spoof_metrics(
                natural_outputs, watermarked_outputs, attacked_outputs
            )
        if self.metric_type == 'fundamental':
            return self._calculate_fundamental_metrics(
                natural_outputs, watermarked_outputs, attacked_outputs
            )
        else:
            summary = self._calculate_dynamic_metrics(
                natural_outputs, watermarked_outputs, attacked_outputs
            )
            if 'threshold' in summary:
                reverse = bool((self.config.get('params') or {}).get('reverse', False))
                self._apply_dynamic_threshold_to_log(results_log, summary['threshold'], reverse)
            return summary

    def _calculate_fundamental_metrics(
        self, natural_outputs, watermarked_outputs, attacked_outputs
    ) -> Dict[str, Any]:
        """Handles metric calculation for boolean detection results."""
        logger.info('--- Calculating detection metrics using FUNDAMENTAL (boolean) logic ---')

        if not natural_outputs or not watermarked_outputs:
            return {'error': 'Missing essential data for calculation.'}

        # TPR/FPR are simply the fraction of 'True' predictions in each group
        tpr_before_attack = (
            sum(watermarked_outputs) / len(watermarked_outputs) if watermarked_outputs else 0.0
        )
        tpr_after_attack = (
            sum(attacked_outputs) / len(attacked_outputs) if attacked_outputs else 0.0
        )
        fpr = sum(natural_outputs) / len(natural_outputs) if natural_outputs else 0.0

        # ASR is bounded to [0, 1]: negative means attack worsened detection.
        asr = self._calculate_attack_success_rate(
            detection_rate_before_attack=tpr_before_attack,
            detection_rate_after_attack=tpr_after_attack,
        )

        return {
            'type': 'fundamental',
            'detection_rate_natural': fpr,
            'detection_rate_before_attack': tpr_before_attack,
            'detection_rate_after_attack': tpr_after_attack,
            'attack_success_rate': asr,
        }

    def _calculate_dynamic_metrics(
        self, natural_scores, watermarked_scores, attacked_scores
    ) -> Dict[str, Any]:
        """Handles metric calculation for score-based detection results."""
        logger.info('--- Calculating detection metrics using DYNAMIC (score-based) logic ---')

        if not natural_scores or not watermarked_scores:
            return {
                'error': 'Missing essential data for calibration (natural or watermarked scores).'
            }

        # --- Stage 1: Calibration - Find threshold ---
        calculator = DynamicThresholdSuccessRateCalculator(**self.config.get('params', {}))

        baseline_metrics = calculator.calculate(
            watermarked_result=watermarked_scores, non_watermarked_result=natural_scores
        )

        # Recalculate to get the threshold value
        inputs = [DetectionResult(True, x) for x in watermarked_scores] + [
            DetectionResult(False, x) for x in natural_scores
        ]
        threshold = calculator._find_threshold(inputs)
        baseline_metrics['threshold'] = threshold

        # --- Stage: Calculate AUC-ROC ---
        y_true_base = [1] * len(watermarked_scores) + [0] * len(natural_scores)
        y_score_base = watermarked_scores + natural_scores

        y_true_attack = [1] * len(attacked_scores) + [0] * len(natural_scores)
        y_score_attack = attacked_scores + natural_scores

        reverse = calculator.reverse
        if reverse:
            y_score_base = [-s for s in y_score_base]
            y_score_attack = [-s for s in y_score_attack]

        try:
            auc_before = roc_auc_score(y_true_base, y_score_base)
            auc_after = roc_auc_score(y_true_attack, y_score_attack) if attacked_scores else 0.0
        except Exception as e:
            logger.warning(f'Failed to calculate AUC: {e}')
            auc_before, auc_after = 0.0, 0.0

        logger.info(
            f'Calibration complete. Threshold: {threshold:.4f}, AUC(Before→After): {auc_before:.4f} → {auc_after:.4f}'
        )

        # --- Stage 2: Evaluation - Apply the fixed threshold ---
        reverse = calculator.reverse

        tpr_before_attack = self._calculate_detection_rate_with_threshold(
            watermarked_scores, threshold, reverse
        )
        tpr_after_attack = self._calculate_detection_rate_with_threshold(
            attacked_scores, threshold, reverse
        )

        # --- Stage 3: Calculate ASR ---
        asr = self._calculate_attack_success_rate(
            detection_rate_before_attack=tpr_before_attack,
            detection_rate_after_attack=tpr_after_attack,
        )

        return {
            'type': 'dynamic',
            'calibration_metrics': baseline_metrics,  # Contains baseline TPR, FPR, etc. at the found threshold
            'threshold': threshold,
            'detection_rate_before_attack': tpr_before_attack,
            'detection_rate_after_attack': tpr_after_attack,
            'attack_success_rate': asr,
        }

    def _calculate_spoof_metrics(
        self, natural_preds, watermarked_preds, spoof_preds
    ) -> Dict[str, Any]:
        ssr = sum(spoof_preds) / len(spoof_preds) if spoof_preds else 0.0
        fpr = sum(natural_preds) / len(natural_preds) if natural_preds else 0.0
        tpr_baseline = (
            sum(watermarked_preds) / len(watermarked_preds) if watermarked_preds else None
        )
        return {
            'type': 'spoof',
            'spoof_success_rate': float(round(ssr, 4)),
            'detection_rate_natural (FPR)': float(round(fpr, 4)),
            'detection_rate_baseline_watermark (TPR)': float(round(tpr_baseline, 4))
            if tpr_baseline is not None
            else None,
        }

    def _calculate_detection_rate_with_threshold(
        self, scores: List[float], threshold: float, reverse: bool
    ) -> float:
        """Calculate detection rate for float scores given a fixed threshold.
        Args:
            reverse: If False, higher scores are considered positive (use >= threshold).
                    If True, lower scores are considered positive (use <= threshold).

        """
        if not scores:
            return 0.0

        if not reverse:  # higher is positive
            detected_count = sum(1 for s in scores if s >= threshold)
        else:  # lower is postive
            detected_count = sum(1 for s in scores if s <= threshold)
        return detected_count / len(scores)


class AttackEffectivenessManager:
    def __init__(self):
        pass

    def _levenshtein_chars(self, a: str, b: str) -> int:
        if a == b:
            return 0
        la, lb = len(a), len(b)
        dp = [[0] * (lb + 1) for _ in range(la + 1)]
        for i in range(la + 1):
            dp[i][0] = i
        for j in range(lb + 1):
            dp[0][j] = j
        for i in range(1, la + 1):
            for j in range(1, lb + 1):
                cost = 0 if a[i - 1] == b[j - 1] else 1
                dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
        return dp[la][lb]

    def _levenshtein_tokens(self, a: List[str], b: List[str]) -> int:
        if a == b:
            return 0
        la, lb = len(a), len(b)
        dp = [[0] * (lb + 1) for _ in range(la + 1)]
        for i in range(la + 1):
            dp[i][0] = i
        for j in range(lb + 1):
            dp[0][j] = j
        for i in range(1, la + 1):
            for j in range(1, lb + 1):
                cost = 0 if a[i - 1] == b[j - 1] else 1
                dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
        return dp[la][lb]

    def calculate_perturbation_budget(
        self, results_log: List[Union[Dict[str, Any], SampleRecord]]
    ) -> Dict[str, float]:
        edc_list, erc_list, edt_list, ert_list = [], [], [], []
        for res in results_log:
            status = res.status if isinstance(res, SampleRecord) else res.get('status')
            if status != 'success':
                continue

            if isinstance(res, SampleRecord):
                x = res.watermarked_text
                x_tilde = res.attacked_text or ''
            else:
                x = res.get('watermarked_text', '')
                x_tilde = res.get('attacked_text', '')

            if not x or not x_tilde:
                continue
            edc = self._levenshtein_chars(x, x_tilde)
            denom_c = max(len(x), 1)
            erc = edc / denom_c
            toks_x = x.split()
            toks_tilde = x_tilde.split()
            edt = self._levenshtein_tokens(toks_x, toks_tilde)
            denom_t = max(len(toks_x), 1)
            ert = edt / denom_t
            edc_list.append(edc)
            erc_list.append(erc)
            edt_list.append(edt)
            ert_list.append(ert)

        def avg(lst: List[float]) -> float:
            return float(sum(lst) / len(lst)) if lst else 0.0

        return {
            'EDC_avg': avg(edc_list),
            'ERC_avg': avg(erc_list),
            'EDT_avg': avg(edt_list),
            'ERT_avg': avg(ert_list),
        }

    def calculate_score_drop(
        self, results_log: List[Union[Dict[str, Any], SampleRecord]]
    ) -> Dict[str, float]:
        watermarked_scores, natural_scores = [], []
        per_sample_drop = []
        for res in results_log:
            status = res.status if isinstance(res, SampleRecord) else res.get('status')
            if status != 'success':
                continue

            if isinstance(res, SampleRecord):
                dw = res.detection_watermark
                dn = res.detection_natural
                da = res.detection_attack
                sw = dw.z_score if dw else None
                sn = dn.z_score if dn else None
                sa = da.z_score if da else None
            else:
                dw = res.get('detection_watermark', {})
                dn = res.get('detection_natural', {})
                da = res.get('detection_attack', {})
                sw = dw.get('z_score', None)
                sn = dn.get('z_score', None)
                sa = da.get('z_score', None)

            if isinstance(sw, (int, float)):
                watermarked_scores.append(float(sw))
            if isinstance(sn, (int, float)):
                natural_scores.append(float(sn))

        if not watermarked_scores or not natural_scores:
            return {'normalized_score_drop_avg': 0.0}
        sw_max = max(watermarked_scores)
        s_min = min(natural_scores)
        denom = sw_max - s_min
        if denom <= 0:
            return {'normalized_score_drop_avg': 0.0}

        for res in results_log:
            status = res.status if isinstance(res, SampleRecord) else res.get('status')
            if status != 'success':
                continue

            if isinstance(res, SampleRecord):
                dw = res.detection_watermark
                da = res.detection_attack
                sw = dw.z_score if dw else None
                sa = da.z_score if da else None
            else:
                dw = res.get('detection_watermark', {})
                da = res.get('detection_attack', {})
                sw = dw.get('z_score', None)
                sa = da.get('z_score', None)

            if isinstance(sw, (int, float)) and isinstance(sa, (int, float)):
                per_sample_drop.append((float(sw) - float(sa)) / denom)
        avg_drop = float(sum(per_sample_drop) / len(per_sample_drop)) if per_sample_drop else 0.0
        return {'normalized_score_drop_avg': avg_drop}

    def calculate_asr(
        self, results_log: List[Union[Dict[str, Any], SampleRecord]]
    ) -> Dict[str, float]:
        total_pos = 0
        not_detected_after = 0
        for res in results_log:
            status = res.status if isinstance(res, SampleRecord) else res.get('status')
            if status != 'success':
                continue

            if isinstance(res, SampleRecord):
                dw = res.detection_watermark
                da = res.detection_attack
                pw = dw.prediction if dw else None
                pa = da.prediction if da else None
            else:
                dw = res.get('detection_watermark', {})
                da = res.get('detection_attack', {})
                pw = dw.get('prediction', None)
                pa = da.get('prediction', None)

            if isinstance(pw, bool):
                if pw:
                    total_pos += 1
                    if isinstance(pa, bool) and not pa:
                        not_detected_after += 1
        asr = float(not_detected_after / total_pos) if total_pos > 0 else 0.0
        return {'ASR': asr}

    def calculate_from_log(
        self, results_log: List[Union[Dict[str, Any], SampleRecord]]
    ) -> Dict[str, Any]:
        budget = self.calculate_perturbation_budget(results_log)
        drop = self.calculate_score_drop(results_log)
        asr = self.calculate_asr(results_log)
        return {
            'perturbation_budget': budget,
            'removal_performance': {**drop, **asr},
        }
