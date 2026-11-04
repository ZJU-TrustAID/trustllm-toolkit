from __future__ import annotations

import ast
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from huggingface_hub import snapshot_download
from huggingface_hub.utils import HFValidationError, validate_repo_id

DEFAULT_HF_MIRROR_ENDPOINT = 'https://hf-mirror.com'

_DIRECT_MODEL_KEYS = {
    'attack_model',
    'attacker',
    'base_model',
    'candidate_generator_model',
    'generation_model',
    'model',
    'model_name',
    'model_path',
    'paraphraser',
    'proxy',
    'quality_oracle',
    'ref',
    'reference_model',
    'tokenizer',
    'tokenizer_name',
    'tokenizer_path',
}
_PARENT_MODEL_KEYS = {'model_requirements', 'tokenizer_requirements'}
_LOCAL_PATH_PREFIXES = {
    '.',
    '..',
    '~',
    'configs',
    'dataset',
    'datasets',
    'docs',
    'libs',
    'output',
    'scripts',
    'src',
}
_LOCAL_FILE_SUFFIXES = {
    '.bin',
    '.ckpt',
    '.json',
    '.model',
    '.pt',
    '.pth',
    '.safetensors',
    '.yaml',
    '.yml',
}
_AUTO_MODEL_SENTINELS = {
    'auto',
    'base',
    'base-model',
    'base_model',
    'same-as-base',
    'same_as_base',
    'surrogate',
    'threat-model',
    'threat_model',
    'victim',
}


@dataclass(frozen=True)
class DownloadTask:
    repo_id: str
    target_dir: Path
    sources: tuple[str, ...]
    action: str


def _looks_like_hf_repo_id(value: str, *, allow_single_segment: bool) -> bool:
    normalized = str(value or '').strip()
    if not normalized or os.path.isabs(normalized):
        return False
    if normalized.startswith(('./', '../', '~/', '/')):
        return False
    if normalized.lower() in _AUTO_MODEL_SENTINELS:
        return False
    if not allow_single_segment and normalized.count('/') != 1:
        return False

    path = Path(normalized)
    if path.suffix.lower() in _LOCAL_FILE_SUFFIXES:
        return False
    if normalized.split('/', 1)[0] in _LOCAL_PATH_PREFIXES:
        return False

    try:
        validate_repo_id(normalized)
    except HFValidationError:
        return False
    return True


def _should_collect_key(path: tuple[str, ...]) -> bool:
    if not path:
        return False

    last = path[-1].lower()
    if last in _DIRECT_MODEL_KEYS:
        return True

    if len(path) >= 2 and path[-2].lower() in _PARENT_MODEL_KEYS:
        return True

    return False


def collect_model_references(
    payload: Any,
    *,
    source_name: str,
    path: tuple[str, ...] = (),
) -> dict[str, set[str]]:
    refs: dict[str, set[str]] = {}

    if isinstance(payload, dict):
        for key, value in payload.items():
            child_refs = collect_model_references(
                value,
                source_name=source_name,
                path=path + (str(key),),
            )
            _merge_reference_maps(refs, child_refs)
        return refs

    if isinstance(payload, list):
        for index, item in enumerate(payload):
            child_refs = collect_model_references(
                item,
                source_name=source_name,
                path=path + (str(index),),
            )
            _merge_reference_maps(refs, child_refs)
        return refs

    if (
        isinstance(payload, str)
        and _should_collect_key(path)
        and _looks_like_hf_repo_id(payload, allow_single_segment=False)
    ):
        dotted_path = '.'.join(path)
        refs.setdefault(payload.strip(), set()).add(f'{source_name}:{dotted_path}')

    return refs


def _merge_reference_maps(target: dict[str, set[str]], source: dict[str, set[str]]) -> None:
    for repo_id, sources in source.items():
        target.setdefault(repo_id, set()).update(sources)


def load_structured_config(path: Path) -> Any:
    with path.open('r', encoding='utf-8') as f:
        if path.suffix.lower() == '.json':
            return json.load(f)
        return yaml.safe_load(f)


def extract_string_list_assignment(path: Path, variable_name: str) -> list[str]:
    tree = ast.parse(path.read_text(encoding='utf-8'), filename=str(path))

    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == variable_name:
                value = ast.literal_eval(node.value)
                if isinstance(value, list) and all(isinstance(item, str) for item in value):
                    return value
                raise ValueError(f'{path}::{variable_name} is not a string list literal.')

    return []


def extract_string_dict_values_assignment(path: Path, variable_name: str) -> list[str]:
    tree = ast.parse(path.read_text(encoding='utf-8'), filename=str(path))

    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == variable_name:
                value = ast.literal_eval(node.value)
                if isinstance(value, dict) and all(
                    isinstance(item, str) for item in value.values()
                ):
                    return list(value.values())
                raise ValueError(f'{path}::{variable_name} is not a string-valued dict literal.')

    return []


def discover_project_model_references(repo_root: Path) -> dict[str, set[str]]:
    refs: dict[str, set[str]] = {}

    config_paths = [repo_root / 'configs' / 'general.yaml']
    attack_dir = repo_root / 'configs' / 'attack'

    benchmark_matrix_path = repo_root / 'src' / 'evaluation' / 'benchmark_matrix.py'
    active_attack_names: set[str] = set()
    if benchmark_matrix_path.exists():
        for variable_name in ('GENERAL_ATTACKS', 'STEAL_ATTACKS'):
            try:
                active_attack_names.update(
                    extract_string_list_assignment(benchmark_matrix_path, variable_name)
                )
            except Exception:
                pass

    if active_attack_names:
        for attack_name in sorted(active_attack_names):
            config_paths.extend(
                [
                    attack_dir / f'{attack_name}.json',
                    attack_dir / f'{attack_name}.yaml',
                ]
            )
    else:
        config_paths.extend(sorted(attack_dir.glob('*.json')))
        config_paths.extend(sorted(attack_dir.glob('*.yaml')))

    for path in config_paths:
        if not path.exists():
            continue
        payload = load_structured_config(path)
        child_refs = collect_model_references(payload, source_name=str(path.relative_to(repo_root)))
        _merge_reference_maps(refs, child_refs)

    if benchmark_matrix_path.exists():
        try:
            for repo_id in extract_string_list_assignment(benchmark_matrix_path, 'BASE_MODELS'):
                if _looks_like_hf_repo_id(repo_id, allow_single_segment=False):
                    refs.setdefault(repo_id, set()).add(
                        'src/evaluation/benchmark_matrix.py:BASE_MODELS'
                    )
        except Exception:
            pass

    threat_models_path = repo_root / 'src' / 'evaluation' / 'threat_models.py'
    if threat_models_path.exists():
        try:
            for repo_id in extract_string_dict_values_assignment(
                threat_models_path, 'THREAT_MODEL_MAPPING'
            ):
                if _looks_like_hf_repo_id(repo_id, allow_single_segment=False):
                    refs.setdefault(repo_id, set()).add(
                        'src/evaluation/threat_models.py:THREAT_MODEL_MAPPING'
                    )
        except Exception:
            pass

    refs.setdefault('sentence-transformers/all-MiniLM-L6-v2', set()).add(
        'src/evaluation/tools/text_quality_analyzer.py:SemanticFluency default'
    )
    refs.setdefault('textattack/roberta-base-CoLA', set()).add(
        'src/evaluation/tools/text_quality_analyzer.py:SemanticFluency default'
    )

    return refs


def model_target_dir(models_root: Path, repo_id: str) -> Path:
    return models_root / repo_id


def has_local_model_dir(target_dir: Path) -> bool:
    return target_dir.is_dir() and any(target_dir.iterdir())


def build_download_plan(
    refs: dict[str, set[str]],
    *,
    models_root: Path,
    mode: str,
) -> list[DownloadTask]:
    tasks: list[DownloadTask] = []
    for repo_id in sorted(refs):
        target_dir = model_target_dir(models_root, repo_id)
        action = 'download'
        if mode == 'missing' and has_local_model_dir(target_dir):
            action = 'skip'
        tasks.append(
            DownloadTask(
                repo_id=repo_id,
                target_dir=target_dir,
                sources=tuple(sorted(refs[repo_id])),
                action=action,
            )
        )
    return tasks


def download_repo_snapshot(
    *,
    repo_id: str,
    target_dir: Path,
    endpoint: str,
    max_workers: int,
    token: str | None,
) -> str:
    target_dir.parent.mkdir(parents=True, exist_ok=True)
    return snapshot_download(
        repo_id=repo_id,
        local_dir=target_dir,
        endpoint=endpoint,
        max_workers=max_workers,
        token=token,
        force_download=False,
        resume_download=True,
    )
