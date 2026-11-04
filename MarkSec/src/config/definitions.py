from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field, field_validator, model_validator
from src.exceptions.exceptions import MissingDataPathError


class DatasetConfig(BaseModel):
    name: str = Field(default='c4', description='Name of the dataset')
    path: Optional[str] = Field(
        default=None, description='Path to the dataset file or HF dataset name'
    )
    num_samples: int = Field(default=100, ge=1, description='Number of samples to process')


class RunnerSettings(BaseModel):
    output_dir_base: str = Field(default='output', description='Base directory for outputs')
    seed: int = Field(default=42, description='Random seed')
    swanlab_project: str = Field(default='', description='SwanLab project name')

    # Advanced settings
    use_flash_attention: bool = Field(
        default=False, description='Use Flash Attention 2 if available'
    )
    models_local_files_only: bool = Field(
        default=False, description='Only load models from local cache'
    )
    save_result_texts: bool = Field(default=True, description='Save generated texts to JSONL')
    isolate_best_gpu: bool = Field(default=True, description='Automatically select best free GPU')


class QualityMetricConfig(BaseModel):
    model_name: Optional[str] = None
    metrics: List[str] = Field(default_factory=list)


class DetectionMetricConfig(BaseModel):
    type: str = Field(
        default='fundamental', description='Metric calculation type: fundamental, dynamic, or spoof'
    )
    reverse_score_algorithms: List[str] = Field(default_factory=list)
    params: Dict[str, Any] = Field(default_factory=dict)


class SpoofConfig(BaseModel):
    compare_baseline: bool = Field(
        default=False, description='Compare spoofing result with baseline watermark'
    )
    params: Dict[str, Any] = Field(default_factory=dict)


class QSRConfig(BaseModel):
    metric: str = Field(default='Prometheus', description='Quality metric used for QSR.')
    threshold: float = Field(default=3.0, description='Pass threshold for the QSR metric.')
    thresholds: List[float] = Field(
        default_factory=list,
        description='Additional pass thresholds for multi-threshold QSR summaries.',
    )

    @model_validator(mode='after')
    def normalize_thresholds(self):
        all_thresholds = [self.threshold, *self.thresholds]
        self.thresholds = sorted({float(threshold) for threshold in all_thresholds})
        self.threshold = float(self.threshold)
        return self


class EvaluationConfig(BaseModel):
    protocol: str = Field(default='removal', description='Evaluation protocol: removal or spoofing')
    quality_metrics: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    quality_comparison_names: List[str] = Field(
        default_factory=list,
        description='Optional allow-list of comparison metric buckets to compute.',
    )
    detection_metrics: DetectionMetricConfig = Field(default_factory=DetectionMetricConfig)
    spoof: SpoofConfig = Field(default_factory=SpoofConfig)
    qsr: QSRConfig = Field(default_factory=QSRConfig)
    skip_detection_in_evaluate: bool = Field(
        default=False,
        description='Reuse precomputed detection outputs from data_path during evaluate mode.',
    )

    @field_validator('quality_metrics', mode='before')
    @classmethod
    def normalize_metrics(cls, v: Any) -> Any:
        if isinstance(v, dict):
            # Convert None values to empty dicts for robustness
            return {key: (val if val is not None else {}) for key, val in v.items()}
        return v


class ExperimentConfig(BaseModel):
    # Core Experiment Identifiers
    experiment_name: Optional[str] = Field(
        default=None, description='Auto-generated experiment name'
    )

    # Components
    base_model: str = Field(..., description='Path or name of the base model')
    watermark: str = Field(..., description='Name of the watermark algorithm')
    attack: str = Field(..., description='Name of the attack algorithm')

    # Component Configs
    dataset: DatasetConfig = Field(default_factory=DatasetConfig)
    runner_settings: RunnerSettings = Field(default_factory=RunnerSettings)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    models_root: Optional[str] = Field(
        default='',
        description='Root directory for locally mirrored Hugging Face models',
    )

    # Paths (Legacy/Compatibility)
    watermark_config_path: Optional[str] = None

    # Pipeline Control
    pipeline_mode: str = Field(default='all', pattern='^(all|generate|evaluate)$')
    data_path: Optional[str] = Field(
        default=None, description='Path to load/save intermediate data'
    )

    # Generation Params
    max_tokens: int = Field(default=200, ge=1)
    min_len: int = Field(default=60, ge=0)
    chat_template_mode: str = Field(
        default='auto',
        description='Chat prompt formatting mode: auto for Instruct-only, off to disable, force to always enable.',
        pattern='^(auto|off|force)$',
    )

    # Extra fields for flexibility (allow unknown fields from YAML)
    model_config = {'extra': 'allow'}

    @model_validator(mode='after')
    def validate_paths(self):
        dashboard = getattr(self, 'dashboard', None)
        has_run_id = isinstance(dashboard, dict) and bool(
            str(dashboard.get('run_id') or '').strip()
        )
        has_source_run_id = isinstance(dashboard, dict) and bool(
            str(dashboard.get('source_run_id') or '').strip()
        )
        if (
            self.pipeline_mode in ['evaluate']
            and not self.data_path
            and not has_run_id
            and not has_source_run_id
        ):
            raise MissingDataPathError(self.pipeline_mode)
        return self
