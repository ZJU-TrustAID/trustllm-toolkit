"""Model source filesystem validation (no weights involved)."""
from __future__ import annotations

import json
from pathlib import Path

from trustllm_hub.model_source import validate_model_entry


def make_local_model(tmp_path: Path, *, name: str = "qwen", tokenizer: dict | None = None) -> str:
    directory = tmp_path / name
    directory.mkdir()
    (directory / "config.json").write_text("{}", encoding="utf-8")
    (directory / "tokenizer_config.json").write_text(json.dumps(tokenizer or {}), encoding="utf-8")
    return str(directory)


def test_valid_local_model(tmp_path: Path) -> None:
    path = make_local_model(tmp_path)
    report = validate_model_entry({"source": "local_path", "path": path})
    assert report["ok"] is True
    assert report["entry_string"] == path
    assert all(check["ok"] for check in report["checks"])


def test_missing_directory(tmp_path: Path) -> None:
    report = validate_model_entry({"source": "local_path", "path": str(tmp_path / "nope")})
    assert report["ok"] is False
    assert report["checks"][0]["ok"] is False


def test_id2risk_role_requirement(tmp_path: Path) -> None:
    without = make_local_model(tmp_path, name="plain")
    report = validate_model_entry(
        {"source": "local_path", "path": without}, role="guard", requires=["tokenizer_config.id2risk"]
    )
    assert report["ok"] is False

    with_labels = make_local_model(
        tmp_path, name="xguard", tokenizer={"id2risk": {"sec": "安全", "dw": "危险武器"}}
    )
    report = validate_model_entry(
        {"source": "local_path", "path": with_labels}, role="guard", requires=["tokenizer_config.id2risk"]
    )
    assert report["ok"] is True


def test_hf_entry() -> None:
    report = validate_model_entry({"source": "hf", "repo": "Qwen/Qwen2.5-7B-Instruct"})
    assert report["ok"] is True
    assert report["entry_string"] == "Qwen/Qwen2.5-7B-Instruct"


def test_api_entry_reports_key_env(monkeypatch) -> None:
    monkeypatch.delenv("MY_KEY", raising=False)
    entry = {"source": "api", "api": {"base_url": "https://api.example.com/v1", "model": "m", "api_key_env": "MY_KEY"}}
    report = validate_model_entry(entry)
    assert report["ok"] is False  # env var unset
    assert "未设置" in report["checks"][-1]["detail"]
    monkeypatch.setenv("MY_KEY", "secret")
    report = validate_model_entry(entry)
    assert report["ok"] is True
    # The secret value must never appear in the report.
    assert "secret" not in json.dumps(report)
