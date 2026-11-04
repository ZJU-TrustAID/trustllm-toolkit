from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .greenlist import GreenlistWatermark, WatermarkRuntimeConfig
from .transformers_backend import TransformersWatermark


DEFAULT_CONFIG_DIR = Path('configs/watermark')

DEFAULT_WATERMARK_CONFIGS: dict[str, dict[str, Any]] = {
    'KGW': {
        'algorithm_name': 'KGW',
        'gamma': 0.5,
        'delta': 2.0,
        'hash_key': 15485863,
        'prefix_length': 1,
        'seeding_scheme': 'lefthash',
        'z_threshold': 4.0,
    },
    'Unigram': {
        'algorithm_name': 'Unigram',
        'gamma': 0.5,
        'delta': 2.0,
        'hash_key': 15485863,
        'prefix_length': 0,
        'z_threshold': 4.0,
    },
    'MPAC': {
        'algorithm_name': 'MPAC',
        'gamma': 0.25,
        'delta': 2.0,
        'hash_key': 15485863,
        'prefix_length': 1,
        'z_threshold': 2.0,
    },
    'SynthID': {
        'algorithm_name': 'SynthID',
        'ngram_len': 5,
        'keys': [654, 400, 836, 123, 340, 443, 597, 160, 57],
        'sampling_table_seed': 0,
        'sampling_table_size': 65536,
        'context_history_size': 1024,
        'skip_first_ngram_calls': False,
        'z_threshold': 3.0,
    },
    'Unbiased': {
        'algorithm_name': 'Unbiased',
        'gamma': 0.5,
        'delta': 1.0,
        'hash_key': 42,
        'prefix_length': 5,
        'z_threshold': 4.0,
    },
    'DIP': {
        'algorithm_name': 'DIP',
        'gamma': 0.5,
        'delta': 1.5,
        'hash_key': 15485863,
        'prefix_length': 5,
        'z_threshold': 1.513,
    },
    'SIR': {
        'algorithm_name': 'SIR',
        'gamma': 0.5,
        'delta': 1.0,
        'hash_key': 15485863,
        'prefix_length': 10,
        'z_threshold': 0.2,
    },
}


def _load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open('r', encoding='utf-8') as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f'Watermark config must be a JSON object: {path}')
    return payload


def resolve_watermark_config_path(algorithm_name: str, explicit_path: str | None = None) -> Path | None:
    if explicit_path:
        return Path(explicit_path)

    candidates = [
        DEFAULT_CONFIG_DIR / f'{algorithm_name}.json',
        DEFAULT_CONFIG_DIR / algorithm_name / 'config.json',
    ]
    for path in candidates:
        if path.is_file():
            return path
    return None


def load_watermark(
    *,
    algorithm_name: str,
    algorithm_config_path: str | None,
    model: Any,
    tokenizer: Any,
    device: str,
    max_new_tokens: int,
    min_new_tokens: int,
):
    config_path = resolve_watermark_config_path(algorithm_name, algorithm_config_path)
    config_dict = dict(DEFAULT_WATERMARK_CONFIGS.get(algorithm_name, {}))
    if config_path is not None:
        if not config_path.is_file():
            raise FileNotFoundError(
                f"Watermark configuration for '{algorithm_name}' was not found: {config_path}"
            )
        config_dict.update(_load_json(config_path))
    if not config_dict:
        raise ValueError(f"Unsupported watermark algorithm: '{algorithm_name}'")

    config_dict.setdefault('algorithm_name', algorithm_name)
    runtime_config = WatermarkRuntimeConfig(
        model=model,
        tokenizer=tokenizer,
        device=device,
        max_new_tokens=max_new_tokens,
        min_new_tokens=min_new_tokens,
    )
    algorithm_key = str(config_dict.get('algorithm_name') or algorithm_name).upper()
    if algorithm_key in {'KGW', 'SYNTHID'}:
        return TransformersWatermark(config_dict=config_dict, runtime_config=runtime_config)
    return GreenlistWatermark(config_dict=config_dict, runtime_config=runtime_config)
