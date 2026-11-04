from __future__ import annotations

from pathlib import Path
from typing import Iterable

FORMAL_BENCHMARK_NUM_SAMPLES = 100

DATASET_CONFIGS = {
    'c4': {
        'name': 'c4',
        'path': 'data/c4/processed_c4.json',
        'num_samples': FORMAL_BENCHMARK_NUM_SAMPLES,
    },
    'dolly': {
        'name': 'databricks-dolly-15k',
        'path': 'data/databricks-dolly-15k.jsonl',
        'num_samples': FORMAL_BENCHMARK_NUM_SAMPLES,
    },
    'bookreport': {
        'name': 'mmw_book_report',
        'num_samples': FORMAL_BENCHMARK_NUM_SAMPLES,
    },
}

DATASET_NAME_TO_KEY = {str(cfg['name']): key for key, cfg in DATASET_CONFIGS.items()}

GENERAL_ATTACKS = [
    'LLMP',
    'SIRA',
    'DIPPER',
    'RW',
    'SA',
]

STEAL_ATTACKS = [
    'B4',
    'MIP',
    'DEMARK',
    'SCTS',
    'JSV',
]

ATTACKS = [*GENERAL_ATTACKS, *STEAL_ATTACKS]

GENERAL_WATERMARKS = [
    'KGW',
    'MPAC',
    'Unigram',
    'SynthID',
    'Unbiased',
    'DIP',
    'SIR',
]

STEAL_WATERMARK_SETTINGS = [
    'KGW_h1_g0.25',
    'KGW_h1_g0.5',
    'KGW_h3_g0.25',
    'KGW_h3_g0.5',
]

WATERMARKS = [*GENERAL_WATERMARKS, *STEAL_WATERMARK_SETTINGS]

BASE_MODELS = [
    'meta-llama/Llama-3.1-8B-Instruct',
    'Qwen/Qwen2.5-7B-Instruct',
    'mistralai/Mistral-7B-Instruct-v0.3',
]

DISTORTIONARY_WATERMARKS = {'KGW', 'MPAC', 'Unigram', 'SIR'}

STEAL_WATERMARK_CONFIG_PATHS = {
    'KGW_h1_g0.25': 'configs/watermark/KGW/h1_g0.25.json',
    'KGW_h1_g0.5': 'configs/watermark/KGW/h1_g0.5.json',
    'KGW_h3_g0.25': 'configs/watermark/KGW/h3_g0.25.json',
    'KGW_h3_g0.5': 'configs/watermark/KGW/h3_g0.5.json',
}

ATTACK_TYPE_LABELS = {attack: 'generic' for attack in GENERAL_ATTACKS} | {
    attack: 'steal' for attack in STEAL_ATTACKS
}

ATTACK_WATERMARK_COMPAT = {
    'LLMP': set(GENERAL_WATERMARKS),
    'SIRA': set(GENERAL_WATERMARKS),
    'DIPPER': set(GENERAL_WATERMARKS),
    'RW': set(GENERAL_WATERMARKS),
    'SA': set(GENERAL_WATERMARKS),
    'B4': set(STEAL_WATERMARK_SETTINGS),
    'MIP': set(STEAL_WATERMARK_SETTINGS),
    'DEMARK': set(STEAL_WATERMARK_SETTINGS),
    'SCTS': set(STEAL_WATERMARK_SETTINGS),
    'JSV': set(STEAL_WATERMARK_SETTINGS),
}

EXCLUDED_PRIMARY_SETTINGS = {
    ('SA', 'KGW', 'mistralai/Mistral-7B-Instruct-v0.3'),
    ('SA', 'Unigram', 'mistralai/Mistral-7B-Instruct-v0.3'),
    ('SA', 'SIR', 'mistralai/Mistral-7B-Instruct-v0.3'),
}


def get_supported_watermarks_for_attack(attack: str) -> set[str]:
    return ATTACK_WATERMARK_COMPAT.get(attack, set(WATERMARKS))


def is_attack_watermark_compatible(attack: str, watermark: str) -> bool:
    return watermark in get_supported_watermarks_for_attack(attack)


def get_runtime_watermark_name(attack: str, benchmark_watermark: str) -> str:
    if attack in STEAL_ATTACKS and benchmark_watermark in STEAL_WATERMARK_CONFIG_PATHS:
        return 'KGW'
    return benchmark_watermark


def get_runtime_watermark_config_path(attack: str, benchmark_watermark: str) -> str | None:
    if attack not in STEAL_ATTACKS:
        return None
    return STEAL_WATERMARK_CONFIG_PATHS.get(benchmark_watermark)


def _derive_setting_from_config_path(
    watermark: str, watermark_config_path: str | None
) -> str | None:
    config_path = str(watermark_config_path or '').strip()
    if not config_path:
        return None
    stem = Path(config_path).stem.strip()
    if not stem:
        return None
    if stem.lower() == watermark.lower() or stem.lower().startswith(f'{watermark.lower()}_'):
        return stem
    return f'{watermark}_{stem}'


def normalize_benchmark_watermark_label(
    attack: str,
    watermark: str | None,
    *,
    watermark_setting: str | None = None,
    watermark_config_path: str | None = None,
) -> str | None:
    raw_watermark = str(watermark or '').strip()
    if not raw_watermark:
        return None

    raw_setting = str(watermark_setting or '').strip()
    if raw_setting:
        return raw_setting

    derived_setting = _derive_setting_from_config_path(raw_watermark, watermark_config_path)
    if derived_setting:
        return derived_setting

    if raw_watermark in STEAL_WATERMARK_CONFIG_PATHS:
        return raw_watermark
    return raw_watermark


def is_experiment_excluded(
    attack: str,
    watermark: str,
    base_model: str,
    dataset_name: str | None = None,
) -> bool:
    del dataset_name  # reserved for future per-dataset exclusions
    return (attack, watermark, base_model) in EXCLUDED_PRIMARY_SETTINGS


def build_attack_overrides(attack: str, watermark: str, base_model: str) -> dict:
    del base_model
    runtime_watermark = get_runtime_watermark_name(attack, watermark)
    overrides = {}
    runtime_watermark_config_path = get_runtime_watermark_config_path(attack, watermark)

    if runtime_watermark_config_path:
        overrides['watermark_config_path'] = runtime_watermark_config_path

    if attack == 'SA':
        wm_type = (
            'distortionary' if runtime_watermark in DISTORTIONARY_WATERMARKS else 'distortion-free'
        )
        overrides.update(
            {
                'watermark_name': runtime_watermark,
                'watermark_type': wm_type,
                'alpha_smoothness': 1.0,
                'k_top_tokens': 10,
                'confidence_norm_samples_N': 200,
                'max_new_tokens': 200,
                'device': 'auto',
            }
        )

    if attack == 'DEMARK':
        overrides['wm_type'] = 'dipmark' if runtime_watermark == 'DIP' else 'kgw'

    return overrides


def safe_base_model(base_model: str) -> str:
    return base_model.replace('/', '_')


def iter_expected_specs(
    *,
    dataset_keys: Iterable[str] | None = None,
    attacks: Iterable[str] | None = None,
    watermarks: Iterable[str] | None = None,
    base_models: Iterable[str] | None = None,
) -> Iterable[tuple[str, str, str, str]]:
    selected_dataset_keys = list(dataset_keys or DATASET_CONFIGS.keys())
    selected_attacks = list(attacks or ATTACKS)
    selected_watermarks = list(watermarks or WATERMARKS)
    selected_models = list(base_models or BASE_MODELS)

    for dataset_key in selected_dataset_keys:
        dataset_name = str(DATASET_CONFIGS[dataset_key]['name'])
        for attack in selected_attacks:
            for watermark in selected_watermarks:
                if not is_attack_watermark_compatible(attack, watermark):
                    continue
                for base_model in selected_models:
                    if is_experiment_excluded(attack, watermark, base_model, dataset_name):
                        continue
                    yield (dataset_name, attack, watermark, base_model)
