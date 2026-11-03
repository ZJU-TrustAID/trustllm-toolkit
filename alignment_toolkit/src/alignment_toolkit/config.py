"""YAML resolution and capability validation."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

from .datasets import DatasetDefinition
from .pipeline import AlignmentExperiment, PipelineSpec, StageSpec
from .registry import BACKENDS, DATASETS, METHODS


def load_yaml(path: str | Path) -> dict[str, Any]:
    import yaml

    with Path(path).open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle) or {}
    if not isinstance(value, dict):
        raise ValueError(f"YAML file {path} must contain a mapping")
    return value


def _resolve_registry_path(value: str | Path, config_path: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path

    # Registry references are normally relative to the experiment file. Keep
    # the working-directory fallback for older configurations that used paths
    # relative to the repository root, and search parent directories so an
    # absolute config path also works when invoked from another directory.
    config_file = Path(config_path)
    candidates = [config_file.parent / path]
    candidates.extend(parent / path for parent in config_file.parents)
    candidates.append(Path.cwd() / path)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return path


def _dataset_definition(name: str, value: dict[str, Any]) -> DatasetDefinition:
    if not isinstance(value, dict) or not isinstance(value.get("source"), dict):
        raise ValueError(f"Dataset {name!r} must define a source mapping")
    return DatasetDefinition(
        name=name,
        kind=str(value["kind"]),
        adapter=str(value.get("adapter", "huggingface")),
        source=dict(value["source"]),
        mapping=dict(value.get("mapping") or {}),
        options=dict(value.get("options") or {}),
    )


def _load_dataset_registry(config: dict[str, Any] | None, config_path: Path) -> dict[str, DatasetDefinition]:
    if config is None:
        return {name: DATASETS.get(name) for name in DATASETS.names()}
    definitions: dict[str, DatasetDefinition] = {}
    registry = config.get("registry")
    if registry:
        raw = load_yaml(_resolve_registry_path(registry, config_path))
        definitions.update({name: _dataset_definition(name, value) for name, value in (raw.get("datasets") or {}).items()})
    for name, value in config.items():
        if name != "registry" and isinstance(value, dict) and "source" in value:
            definitions[name] = _dataset_definition(name, value)
    if not definitions:
        raise ValueError("datasets must contain inline definitions or a registry path")
    return definitions


def resolve_experiment(config: dict[str, Any], *, config_path: str | Path = "<memory>") -> PipelineSpec:
    resolved = copy.deepcopy(config)
    pipeline = resolved.get("pipeline")
    if not isinstance(pipeline, dict) or not pipeline.get("stages"):
        raise ValueError("pipeline.stages must contain at least one stage")
    datasets_config = resolved.get("datasets") if isinstance(resolved.get("datasets"), dict) else None
    dataset_registry = _load_dataset_registry(datasets_config, Path(config_path).resolve())
    stages = []
    known_artifacts: set[str] = set()
    for raw_stage in pipeline["stages"]:
        stage_id = str(raw_stage["id"])
        method_value = raw_stage.get("method") or {}
        method_name = method_value if isinstance(method_value, str) else method_value.get("name")
        method = METHODS.get(method_name)
        backend_name = (method_value.get("backend") if isinstance(method_value, dict) else None) or method.default_backend
        backend = BACKENDS.get(backend_name)
        if method.name not in backend.methods:
            raise ValueError(f"Backend {backend.name!r} does not support method {method.name!r}")
        dataset_ref = raw_stage.get("dataset") or {}
        dataset_name = dataset_ref if isinstance(dataset_ref, str) else dataset_ref.get("name")
        if dataset_name not in dataset_registry:
            raise ValueError(f"Stage {stage_id!r} references unknown dataset {dataset_name!r}")
        dataset = dataset_registry[dataset_name]
        if dataset.kind not in method.dataset_kinds:
            raise ValueError(
                f"Method {method.name!r} expects {method.dataset_kinds}, got dataset {dataset_name!r} ({dataset.kind})"
            )
        stage_input = dict(raw_stage.get("input") or {})
        for reference_key in ("model", "policy", "reference", "reward_model", "value_model"):
            if reference_key in raw_stage and reference_key not in stage_input:
                stage_input[reference_key] = raw_stage[reference_key]
        for reference_key, reference in stage_input.items():
            if reference_key in {"model", "policy", "reference", "reward_model", "value_model"} and reference:
                if reference not in resolved.get("models", {}) and reference not in known_artifacts:
                    raise ValueError(f"Stage {stage_id!r} references unknown model or artifact {reference!r}")
        stages.append(
            StageSpec(
                stage_id=stage_id,
                method=method.name,
                backend=backend.name,
                dataset=dataset_name,
                input=stage_input,
                training=dict(raw_stage.get("training") or {}),
            )
        )
        known_artifacts.add(stage_id)
    resolved["pipeline"] = {"name": pipeline.get("name", "alignment"), "stages": [stage.__dict__ for stage in stages]}
    resolved["resolved_datasets"] = {name: definition.__dict__ for name, definition in dataset_registry.items() if any(stage.dataset == name for stage in stages)}
    return PipelineSpec(
        name=str(pipeline.get("name", "alignment")),
        models=dict(resolved.get("models") or {}),
        stages=tuple(stages),
        evaluation=dict(resolved.get("evaluation") or {}),
        output=dict(resolved.get("output") or {}),
        raw=resolved,
    )


def load_experiment(path: str | Path) -> PipelineSpec:
    path = Path(path)
    return resolve_experiment(load_yaml(path), config_path=path)
