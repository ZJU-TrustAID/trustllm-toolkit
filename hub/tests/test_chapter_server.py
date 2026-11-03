"""Chapter frontend server config rendering: hub env params → per-chapter config."""
from __future__ import annotations

import json

from trustllm_hub.chapter_server import ChapterServerManager

from conftest import make_config

SERVER_MANIFEST = """\
chapter: 98
title: 测试服务章节
toolkit: fakeserver
enabled: true
brief: 仅用于渲染测试的假章节。
frontend:
  mode: proxy
  server:
    command: ["{python}", "serve.py", "--config", "{config}"]
    port: 8790
    ready_path: /api/health
    config:
      model: "{models.default.entry}"
      device: "{common.device}"
      dtype: "{common.dtype}"
      web:
        host: "127.0.0.1"
    config_port_key: web.port
  entry: /
backend:
  command: ["{python}", "adapter.py"]
  timeout: 30
models:
  - role: default
    title: 主模型
    accepts: [local_path, hf]
    required: true
    default: {source: hf, repo: "org/model"}
params: []
results:
  primary: report.md
"""

PARAMS = {
    "common": {"device": "auto", "dtype": "bfloat16", "seed": 3},
    "models": {"default": {"source": "hf", "repo": "org/model"}},
}


def _manager_with_server_chapter(tmp_path, monkeypatch, resolved):
    from trustllm_hub.registry import load_manifest

    directory = tmp_path / "chapters" / "fakeserver"
    directory.mkdir(parents=True)
    (directory / "chapter.yaml").write_text(SERVER_MANIFEST, encoding="utf-8")
    monkeypatch.setattr("trustllm_hub.chapter_server.resolve_auto_device", lambda: resolved)
    config = make_config(tmp_path)
    manager = ChapterServerManager(config)
    manifest = load_manifest(directory / "chapter.yaml")
    return manager, manifest


def test_render_server_config_resolves_auto_device(tmp_path, monkeypatch) -> None:
    manager, manifest = _manager_with_server_chapter(tmp_path, monkeypatch, resolved="cuda:6")
    path = manager._render_server_config(manifest, manifest.frontend_server, port=8790, params=PARAMS)
    rendered = json.loads(path.read_text("utf-8"))
    assert rendered["device"] == "cuda:6"
    assert rendered["dtype"] == "bfloat16"
    assert rendered["model"] == "org/model"
    assert rendered["web"]["port"] == 8790


def test_render_server_config_keeps_explicit_device(tmp_path, monkeypatch) -> None:
    manager, manifest = _manager_with_server_chapter(tmp_path, monkeypatch, resolved="cuda:1")
    params = {**PARAMS, "common": {**PARAMS["common"], "device": "cuda:3"}}
    path = manager._render_server_config(manifest, manifest.frontend_server, port=8790, params=params)
    assert json.loads(path.read_text("utf-8"))["device"] == "cuda:3"


def test_render_server_config_without_gpu_info_keeps_auto(tmp_path, monkeypatch) -> None:
    manager, manifest = _manager_with_server_chapter(tmp_path, monkeypatch, resolved=None)
    path = manager._render_server_config(manifest, manifest.frontend_server, port=8790, params=PARAMS)
    assert json.loads(path.read_text("utf-8"))["device"] == "auto"


OPTIONAL_MANIFEST = SERVER_MANIFEST.replace(
    """models:
  - role: default
    title: 主模型
    accepts: [local_path, hf]
    required: true
    default: {source: hf, repo: "org/model"}""",
    """models:
  - role: default
    title: 主模型
    accepts: [local_path, hf]
    required: true
    default: {source: hf, repo: "org/model"}
  - role: nli
    title: 辅助模型
    accepts: [local_path, hf]
    required: false
    default: {source: hf, repo: "org/nli"}""",
).replace('model: "{models.default.entry}"', 'model: "{models.default.entry}"\n      nli_source: "{models.nli.entry}"')


def _optional_manager(tmp_path, monkeypatch):
    from trustllm_hub.registry import load_manifest

    directory = tmp_path / "chapters" / "fakeserver"
    directory.mkdir(parents=True)
    (directory / "chapter.yaml").write_text(OPTIONAL_MANIFEST, encoding="utf-8")
    config = make_config(tmp_path)
    manager = ChapterServerManager(config)
    manifest = load_manifest(directory / "chapter.yaml")
    return manager, manifest


def test_optional_role_empty_renders_empty_string(tmp_path, monkeypatch) -> None:
    manager, manifest = _optional_manager(tmp_path, monkeypatch)
    params = {**PARAMS, "models": {"default": PARAMS["models"]["default"]}}
    path = manager._render_server_config(manifest, manifest.frontend_server, port=8790, params=params)
    rendered = json.loads(path.read_text("utf-8"))
    assert rendered["nli_source"] == ""          # 可选角色留空 → 空串（消费方回退默认）
    assert rendered["model"] == "org/model"


def test_optional_role_filled_renders_model_string(tmp_path, monkeypatch) -> None:
    manager, manifest = _optional_manager(tmp_path, monkeypatch)
    params = {**PARAMS, "models": {
        **PARAMS["models"],
        "nli": {"source": "local_path", "path": "models/nli-helper"},
    }}
    path = manager._render_server_config(manifest, manifest.frontend_server, port=8790, params=params)
    rendered = json.loads(path.read_text("utf-8"))
    assert rendered["nli_source"] == str((tmp_path / "models/nli-helper").resolve())


def test_required_role_incomplete_fails_render(tmp_path, monkeypatch) -> None:
    import pytest as _pytest

    from trustllm_hub.registry import ManifestError

    manager, manifest = _optional_manager(tmp_path, monkeypatch)
    params = {**PARAMS, "models": {**PARAMS["models"], "default": {"source": "local_path", "path": ""}}}
    with _pytest.raises(ManifestError):
        manager._render_server_config(manifest, manifest.frontend_server, port=8790, params=params)
