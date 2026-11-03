"""第四章 alignment_toolkit 清单可被 hub 正常加载。"""
from __future__ import annotations

from pathlib import Path

import pytest

from trustllm_hub.registry import load_manifest

REPO_ROOT = Path(__file__).resolve().parents[2]
CHAPTER_DIR = REPO_ROOT / "alignment_toolkit"


@pytest.mark.skipif(not (CHAPTER_DIR / "chapter.yaml").is_file(), reason="alignment_toolkit 不在仓库根")
def test_alignment_manifest() -> None:
    manifest = load_manifest(CHAPTER_DIR / "chapter.yaml")
    assert manifest.chapter == 4
    assert manifest.toolkit == "alignment"
    assert manifest.title == "安全对齐"
    # 后端命令走 hub_adapter 契约
    joined = " ".join(manifest.backend["command"])
    assert "alignment_toolkit.hub_adapter" in joined and "{yaml}" in joined and "{output_dir}" in joined
    # 前端服务与端口约定
    assert manifest.frontend_server["port"] == 8786
    assert manifest.frontend_server["ready_path"] == "/api/health"
    # 模型角色与结果声明
    assert [role["role"] for role in manifest.models] == ["base"]
    assert manifest.results["primary"] == "report.md"


@pytest.mark.skipif(not (CHAPTER_DIR / "chapter.yaml").is_file(), reason="alignment_toolkit 不在仓库根")
def test_alignment_discovered_from_repo_root() -> None:
    from trustllm_hub.config import HubConfig
    from trustllm_hub.registry import scan_registry

    config = HubConfig(root=REPO_ROOT, hub_dir=REPO_ROOT / "hub", data={})
    registry = scan_registry(config.iter_chapter_dirs())
    toolkits = {manifest.toolkit for manifest in registry.all()}
    assert "alignment" in toolkits
