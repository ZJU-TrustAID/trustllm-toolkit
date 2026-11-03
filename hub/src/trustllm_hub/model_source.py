"""Model source resolution and filesystem checks for the hub UI.

The hub never loads weights itself; it verifies what the user typed (a local
absolute path, a Hugging Face repo id, or an API endpoint) well enough that a
chapter adapter will not fail on obvious problems like a missing ``config.json``.
Role-specific requirements (for example XGuard's ``id2risk`` label map) are
declared by each chapter manifest and checked here too.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from .schema import model_entry_string


def _check(name: str, ok: bool, detail: str = "") -> dict[str, Any]:
    return {"name": name, "ok": bool(ok), "detail": detail}


def _check_shard_index(path: Path) -> dict[str, Any]:
    """Verify every shard listed in model.safetensors.index.json exists on disk.

    A truncated download (exit code 0 notwithstanding) otherwise only surfaces
    when the model is actually loaded, minutes into a run.
    """
    index_path = path / "model.safetensors.index.json"
    if not index_path.is_file():
        return _check("权重分片清单", True, "无 index（单文件权重，跳过）")
    try:
        index = json.loads(index_path.read_text(encoding="utf-8"))
        shards = sorted({shard for shard in index.get("weight_map", {}).values()})
    except (OSError, ValueError) as exc:
        return _check("权重分片清单", False, f"index 解析失败: {exc}")
    missing = [shard for shard in shards if not (path / shard).is_file()]
    if missing:
        return _check(
            "权重分片清单", False,
            f"缺少 {len(missing)}/{len(shards)} 个分片（如 {missing[0]}），请重新下载",
        )
    return _check("权重分片清单", True, f"{len(shards)} 个分片齐全")


def _check_local_role_requires(path: Path, requires: list[str]) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    for requirement in requires:
        if requirement == "tokenizer_config.id2risk":
            config_path = path / "tokenizer_config.json"
            ok, detail = False, "缺少 tokenizer_config.json"
            if config_path.is_file():
                try:
                    data = json.loads(config_path.read_text(encoding="utf-8"))
                    labels = data.get("id2risk")
                    if isinstance(labels, dict) and labels:
                        ok, detail = True, f"包含 {len(labels)} 个风险标签"
                    else:
                        detail = "tokenizer_config.json 缺少 id2risk 标签映射"
                except (OSError, ValueError) as exc:
                    detail = f"tokenizer_config.json 解析失败: {exc}"
            checks.append(_check("XGuard 标签映射 (id2risk)", ok, detail))
        else:  # pragma: no cover - future requirement kinds land here
            checks.append(_check(requirement, False, "未知的角色校验要求"))
    return checks


def validate_model_entry(
    entry: dict[str, Any],
    *,
    role: str = "default",
    requires: list[str] | None = None,
) -> dict[str, Any]:
    """Validate one model entry and return a UI-friendly report.

    The report never includes secrets (API key values) — only whether the
    environment variable is set.
    """
    source = entry["source"]
    checks: list[dict[str, Any]] = []
    summary_parts: list[str] = []
    source_labels = {"local_path": "本地模型", "hf": "HuggingFace", "api": "API"}
    summary_parts.append(source_labels[source])

    if source == "local_path":
        path = Path(entry["path"])
        target = path.name or str(path)
        summary_parts.append(target)
        exists = path.is_dir()
        checks.append(_check("目录存在", exists, str(path)))
        if exists:
            has_config = (path / "config.json").is_file()
            checks.append(_check("config.json", has_config))
            has_tokenizer = (path / "tokenizer_config.json").is_file()
            checks.append(_check("tokenizer_config.json", has_tokenizer))
            checks.append(_check_shard_index(path))
            if requires:
                checks.extend(_check_local_role_requires(path, list(requires)))
    elif source == "hf":
        repo = entry["repo"]
        summary_parts.append(repo)
        checks.append(_check("仓库 ID 格式", "/" in repo and " " not in repo))
    else:  # api
        api = entry["api"]
        summary_parts.append(str(api["model"]))
        checks.append(_check("base_url", bool(api["base_url"]), api["base_url"]))
        key_env = api["api_key_env"]
        key_set = bool(os.environ.get(key_env))
        checks.append(_check(f"环境变量 {key_env}", key_set, "已设置" if key_set else "未设置（执行时会失败）"))

    ok = all(check["ok"] for check in checks)
    return {
        "role": role,
        "source": source,
        "entry_string": _entry_string_safe(entry),
        "ok": ok,
        "summary": " · ".join(summary_parts),
        "checks": checks,
    }


def _entry_string_safe(entry: dict[str, Any]) -> str | None:
    try:
        return model_entry_string(entry)
    except Exception:
        return None
