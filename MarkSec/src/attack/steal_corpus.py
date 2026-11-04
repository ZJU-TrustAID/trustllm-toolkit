from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Any

logger = logging.getLogger('MarkSec')


def _normalize_token(text: str) -> str:
    return ''.join(ch for ch in str(text).lower() if ch.isalnum())


def _raw_hint_segments(text: str) -> list[str]:
    return [segment for segment in re.split(r'[^a-z0-9.]+', str(text).lower()) if segment]


def requires_exact_watermark_match(watermark_hint: str = '') -> bool:
    return len(_raw_hint_segments(watermark_hint)) > 1


def infer_model_hint(model: Any = None, tokenizer: Any = None) -> str:
    candidates: list[str] = []
    if tokenizer is not None:
        name_or_path = getattr(tokenizer, 'name_or_path', '')
        if isinstance(name_or_path, str) and name_or_path.strip():
            candidates.append(name_or_path.strip())
    if model is not None:
        model_name = getattr(model, 'name_or_path', '')
        if isinstance(model_name, str) and model_name.strip():
            candidates.append(model_name.strip())
        cfg = getattr(model, 'config', None)
        cfg_name = getattr(cfg, '_name_or_path', '') if cfg is not None else ''
        if isinstance(cfg_name, str) and cfg_name.strip():
            candidates.append(cfg_name.strip())
    if not candidates:
        return ''
    return os.path.basename(candidates[0].rstrip('/'))


def infer_dataset_hint(dataset: Any = None) -> str:
    if dataset is None:
        return ''
    class_name = dataset.__class__.__name__.lower()
    if 'c4' in class_name:
        return 'c4'
    if 'dolly' in class_name:
        return 'dolly'
    if 'mmw' in class_name:
        return 'mmw'
    source = str(getattr(dataset, 'data_source', '') or '').lower()
    for name in ('c4', 'dolly', 'mmw'):
        if name in source:
            return name
    return ''


def infer_watermark_hint(watermark_name: str = '', watermark_instance: Any = None) -> str:
    base_hint = str(watermark_name or '').strip()
    if watermark_instance is None:
        return base_hint
    config = getattr(watermark_instance, 'config', None)
    config_dict = getattr(config, 'config_dict', None) if config is not None else None
    if isinstance(config_dict, dict):
        algorithm_name = str(config_dict.get('algorithm_name') or base_hint or '').strip()
        if algorithm_name.upper() == 'KGW':
            prefix_length = config_dict.get('prefix_length')
            gamma = config_dict.get('gamma')
            if prefix_length is not None and gamma is not None:
                try:
                    prefix_value = int(prefix_length)
                    gamma_value = float(gamma)
                    return f'KGW_h{prefix_value}_g{format(gamma_value, "g")}'
                except (TypeError, ValueError):
                    pass
        if algorithm_name:
            return algorithm_name
    cls_name = watermark_instance.__class__.__name__
    if isinstance(cls_name, str) and cls_name.strip():
        return cls_name.strip()
    return base_hint


def discover_joint_corpus_path(
    corpus_dir: str = 'output/corpus',
    model_hint: str = '',
    dataset_hint: str = '',
    watermark_hint: str = '',
) -> str:
    corpus_root = Path(corpus_dir)
    if not corpus_root.is_absolute():
        corpus_root = Path(os.path.abspath(str(corpus_root)))
    if not corpus_root.exists():
        return ''

    candidates = sorted(corpus_root.glob('*.jsonl'))
    if not candidates:
        return ''

    model_norm = _normalize_token(model_hint)
    dataset_norm = _normalize_token(dataset_hint)
    watermark_norm = _normalize_token(watermark_hint)
    watermark_segments = _raw_hint_segments(watermark_hint)
    require_exact_watermark = requires_exact_watermark_match(watermark_hint)
    has_any_hint = bool(model_norm or dataset_norm or watermark_norm)

    scored: list[tuple[int, float, Path]] = []
    for path in candidates:
        name = path.name.lower()
        name_norm = _normalize_token(name)
        if require_exact_watermark and not all(segment in name for segment in watermark_segments):
            continue
        score = 0
        if watermark_segments and all(segment in name for segment in watermark_segments):
            score += 100
        elif watermark_norm and watermark_norm in name_norm:
            score += 100
        if dataset_norm and dataset_norm in name_norm:
            score += 40
        if model_norm and model_norm in name_norm:
            score += 80
        if '50k' in name:
            score += 5
        if 'vllm' in name:
            score += 2
        scored.append((score, path.stat().st_mtime, path))

    if not scored:
        return ''

    if has_any_hint and max(score for score, _mtime, _path in scored) <= 0:
        return ''

    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
    selected = scored[0][2]
    return str(selected)
