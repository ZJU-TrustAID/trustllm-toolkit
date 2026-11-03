"""Validation for the unified run YAML shared by every chapter toolkit.

The YAML is the exchange format between the hub frontend, the hub scheduler and
the chapter adapters. The hub validates structure and model entries here; the
``chapter_params`` section is owned by each chapter's adapter, which re-validates
it strictly.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any
import re

ALLOWED_SOURCES = {"local_path", "hf", "api"}
ALLOWED_DTYPES = {"auto", "float16", "bfloat16", "float32"}
_TOP_LEVEL_KEYS = {"version", "session", "chapter", "toolkit", "common", "models", "chapter_params"}
_COMMON_KEYS = {"device", "dtype", "seed", "output_dir"}
_MODEL_ENTRY_KEYS = {"source", "path", "repo", "api", "extra"}
_API_KEYS = {"base_url", "api_key_env", "model"}

_REPO_PATTERN = re.compile(r"^[\w.-]+/[\w.-]+$")


class ConfigError(ValueError):
    """Raised when a run YAML (or fragment) fails validation."""


def _require_mapping(value: Any, where: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ConfigError(f"{where} 必须是映射（mapping）")
    return value


def expand_path(path: str, base: Path | None = None) -> str:
    expanded = Path(path).expanduser()
    if not expanded.is_absolute():
        expanded = (base or Path.cwd()) / expanded
    return str(expanded.resolve())


def validate_model_entry(entry: Any, where: str, *, base: Path | None = None) -> dict[str, Any]:
    entry = _require_mapping(entry, where)
    unknown = set(entry) - _MODEL_ENTRY_KEYS
    if unknown:
        raise ConfigError(f"{where} 含未知字段: {sorted(unknown)}")
    source = entry.get("source")
    if source not in ALLOWED_SOURCES:
        raise ConfigError(f"{where}.source 必须是 {sorted(ALLOWED_SOURCES)} 之一")
    normalized: dict[str, Any] = {"source": source}
    if source == "local_path":
        path = entry.get("path")
        if not isinstance(path, str) or not path.strip():
            raise ConfigError(f"{where}.path 在 source=local_path 时必填（支持本地模型绝对路径）")
        normalized["path"] = expand_path(path.strip(), base)
    elif source == "hf":
        repo = entry.get("repo")
        if not isinstance(repo, str) or not _REPO_PATTERN.match(repo.strip()):
            raise ConfigError(f"{where}.repo 在 source=hf 时必填，形如 'Qwen/Qwen2.5-7B-Instruct'")
        normalized["repo"] = repo.strip()
    else:  # api
        api = _require_mapping(entry.get("api"), f"{where}.api")
        unknown_api = set(api) - _API_KEYS
        if unknown_api:
            raise ConfigError(f"{where}.api 含未知字段: {sorted(unknown_api)}")
        base_url = api.get("base_url")
        model = api.get("model")
        if not isinstance(base_url, str) or not base_url.startswith(("http://", "https://")):
            raise ConfigError(f"{where}.api.base_url 必须以 http:// 或 https:// 开头")
        if not isinstance(model, str) or not model.strip():
            raise ConfigError(f"{where}.api.model 必填（API 模型名）")
        normalized["api"] = {
            "base_url": base_url.strip(),
            "model": model.strip(),
            "api_key_env": str(api.get("api_key_env") or "OPENAI_API_KEY"),
        }
    extra = entry.get("extra") or {}
    if not isinstance(extra, dict):
        raise ConfigError(f"{where}.extra 必须是映射")
    if extra:
        normalized["extra"] = extra
    return normalized


def validate_config(config: Any, *, base: Path | None = None) -> dict[str, Any]:
    """Validate a full run config mapping and return a normalized deep copy."""
    config = _require_mapping(config, "配置")
    unknown = set(config) - _TOP_LEVEL_KEYS
    if unknown:
        raise ConfigError(f"配置含未知字段: {sorted(unknown)}")
    if config.get("version") != 1:
        raise ConfigError("version 必须为 1")
    chapter = config.get("chapter")
    if not isinstance(chapter, int) and not isinstance(chapter, str):
        raise ConfigError("chapter 必须是章号（整数）或章号范围（字符串，如 '4-5'）")
    toolkit = config.get("toolkit")
    if not isinstance(toolkit, str) or not toolkit.strip():
        raise ConfigError("toolkit 必填（章节工具包标识，如 'hallucination'）")

    common_input = _require_mapping(config.get("common") or {}, "common")
    unknown_common = set(common_input) - _COMMON_KEYS
    if unknown_common:
        raise ConfigError(f"common 含未知字段: {sorted(unknown_common)}")
    common: dict[str, Any] = {}
    device = common_input.get("device", "auto")
    if not isinstance(device, str) or not device.strip():
        raise ConfigError("common.device 必须是非空字符串（如 'auto'、'cuda:0'、'cpu'）")
    common["device"] = device.strip()
    dtype = common_input.get("dtype", "auto")
    if dtype not in ALLOWED_DTYPES:
        raise ConfigError(f"common.dtype 必须是 {sorted(ALLOWED_DTYPES)} 之一")
    common["dtype"] = dtype
    seed = common_input.get("seed", 0)
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ConfigError("common.seed 必须是非负整数")
    common["seed"] = seed
    output_dir = common_input.get("output_dir")
    if output_dir is not None:
        if not isinstance(output_dir, str) or not output_dir.strip():
            raise ConfigError("common.output_dir 必须是非空字符串")
        common["output_dir"] = output_dir

    models_input = _require_mapping(config.get("models") or {}, "models")
    if not models_input:
        raise ConfigError("models 至少要包含一个角色的模型配置")
    models = {
        role: validate_model_entry(entry, f"models.{role}", base=base)
        for role, entry in models_input.items()
    }

    chapter_params = _require_mapping(config.get("chapter_params") or {}, "chapter_params")

    session = config.get("session")
    if session is not None and (not isinstance(session, str) or not session.strip()):
        raise ConfigError("session 必须是非空字符串")

    normalized: dict[str, Any] = {
        "version": 1,
        "chapter": chapter,
        "toolkit": toolkit.strip(),
        "common": common,
        "models": models,
        "chapter_params": chapter_params,
    }
    if session:
        normalized["session"] = session.strip()
    return normalized


def model_entry_string(entry: dict[str, Any]) -> str:
    """The single string a chapter toolkit expects for a model: a local path or a repo id."""
    if entry["source"] == "local_path":
        return str(entry["path"])
    if entry["source"] == "hf":
        return str(entry["repo"])
    raise ConfigError("该角色不支持 API 模型源")


def dumps_yaml(config: dict[str, Any]) -> str:
    import yaml

    return yaml.safe_dump(config, allow_unicode=True, sort_keys=False)
