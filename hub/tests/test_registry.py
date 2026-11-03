"""Chapter manifest loading, placeholder expansion and config-template rendering."""
from __future__ import annotations

from pathlib import Path

import pytest

from trustllm_hub.registry import ManifestError, load_manifest, scan_registry

from conftest import FAKE_MANIFEST, make_chapter_dir as make_chapter


def test_load_and_view(tmp_path: Path) -> None:
    directory = make_chapter(tmp_path)
    manifest = load_manifest(directory / "chapter.yaml")
    assert manifest.toolkit == "fakekit"
    assert manifest.chapter == 99
    view = manifest.public_view(port=8999, server_state="stopped")
    assert view["title"] == "测试章节"
    assert view["models"][0]["role"] == "default"
    assert manifest.frontend_mode == "link"


def test_missing_required_field(tmp_path: Path) -> None:
    directory = make_chapter(tmp_path, manifest="chapter: 1\ntoolkit: x\ntitle: t\nmodels: []\n")
    with pytest.raises(ManifestError):
        load_manifest(directory / "chapter.yaml")


def test_bad_placeholder(tmp_path: Path) -> None:
    directory = make_chapter(tmp_path, manifest=FAKE_MANIFEST.replace('"{yaml}"', '"{nope}"'))
    with pytest.raises(ManifestError):
        load_manifest(directory / "chapter.yaml")


def test_duplicate_toolkit_rejected(tmp_path: Path) -> None:
    first = make_chapter(tmp_path)
    # A second directory declaring the same toolkit slug must be rejected.
    other = tmp_path / "chapters" / "other"
    other.mkdir(parents=True)
    (other / "fake_adapter.py").write_text("# unused\n", encoding="utf-8")
    (other / "chapter.yaml").write_text(FAKE_MANIFEST, encoding="utf-8")
    with pytest.raises(ManifestError):
        scan_registry([first, other])


def test_fill_command(tmp_path: Path) -> None:
    directory = make_chapter(tmp_path)
    manifest = load_manifest(directory / "chapter.yaml")
    filled = manifest.fill_command(
        manifest.backend_command,
        conda_bin="/opt/conda/bin/conda",
        python_bin="/usr/bin/python3",
        yaml_path="/tmp/x.yaml",
        output_dir="/tmp/out",
    )
    assert filled[0] == "/usr/bin/python3"
    assert "/tmp/x.yaml" in filled and "/tmp/out" in filled
    assert "{output_dir}" not in " ".join(filled)


def test_render_config_template(tmp_path: Path) -> None:
    directory = make_chapter(tmp_path)
    manifest = load_manifest(directory / "chapter.yaml")
    template = {
        "guard": {"model": "{models.guard.entry}", "device": "{common.device}"},
        "web": {"host": "127.0.0.1"},
    }
    rendered = manifest.render_config_template(
        template,
        port=8782,
        port_key="web.port",
        params={
            "common.device": "cuda:0",
            "models.guard.entry": "/models/xguard",
        },
    )
    assert rendered == {
        "guard": {"model": "/models/xguard", "device": "cuda:0"},
        "web": {"host": "127.0.0.1", "port": 8782},
    }


def test_render_template_missing_param(tmp_path: Path) -> None:
    directory = make_chapter(tmp_path)
    manifest = load_manifest(directory / "chapter.yaml")
    with pytest.raises(ManifestError):
        manifest.render_config_template(
            {"m": "{models.guard.entry}"}, port=1, port_key=None, params={}
        )
