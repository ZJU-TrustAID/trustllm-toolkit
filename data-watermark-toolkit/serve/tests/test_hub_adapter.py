# -*- coding: utf-8 -*-
"""Hub adapter 契约测试（离线，无 GPU / 无网络）。

与 MarkSec/tests/test_hub_adapter.py 同思路：成功路径通过替换子进程注入一份
假的 report.json，断言 result.json / report.md 契约；失败路径直接调用 main()
断言退出码与 failed result.json。
"""
import json
import subprocess
from pathlib import Path

import pytest
import yaml

from serve import hub_adapter


def _write_config(tmp_path, *, params=None, models=None, common=None) -> str:
    config = {
        "version": 1,
        "chapter": 2,
        "toolkit": "datamark",
        "common": common if common is not None else {"device": "cpu", "dtype": "auto", "seed": 7},
        "models": models if models is not None else {
            "default": {"source": "hf", "repo": "HuggingFaceTB/SmolLM2-135M-Instruct"},
        },
        "chapter_params": params if params is not None else {
            "method": "canary_memory", "fraction": 0.5,
            "finetune": "lora", "epochs": 3, "lr": 0.0001,
        },
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config, allow_unicode=True), encoding="utf-8")
    return str(path)


FAKE_REPORT = {
    "method": "canary_memory",
    "family": "内容记忆型",
    "access": "white-box",
    "reference": "Meeus et al., Copyright Traps for LLMs (ICML 2024)",
    "fraction": 0.5,
    "seed": 7,
    "n_train_texts": 48,
    "train": {"steps": 30, "epochs": 3, "n_samples": 48,
              "final_loss": 1.2, "mean_loss": 1.5},
    "detection": {"score": 2.5, "n": 24, "p_value": 0.003, "alpha": 0.01,
                  "decision": "watermarked",
                  "detail": {"test": "welch_t", "evidence": {"kind": "pairs"}}},
    "elapsed_sec": 12.3,
    "model": "/tmp/fake-model",
}


@pytest.fixture()
def no_deps(monkeypatch):
    monkeypatch.setattr(hub_adapter, "_check_dependencies", lambda finetune: None)


def _fake_cli(monkeypatch, out_name="result"):
    """把 subprocess.run 换成写假 report.json 的桩，返回收到的命令供断言。"""
    seen = {}

    def fake_run(command, cwd=None, env=None):
        seen["command"] = command
        seen["cwd"] = cwd
        out = Path(command[command.index("--out") + 1])
        out.mkdir(parents=True, exist_ok=True)
        (out / "report.json").write_text(
            json.dumps(FAKE_REPORT, ensure_ascii=False), encoding="utf-8")
        return subprocess.CompletedProcess(args=command, returncode=0)

    monkeypatch.setattr(hub_adapter.subprocess, "run", fake_run)
    return seen


def test_success_contract(tmp_path, monkeypatch, no_deps):
    config_path = _write_config(tmp_path)
    output_dir = tmp_path / "session"
    seen = _fake_cli(monkeypatch)

    code = hub_adapter.main(["--config", config_path, "--output-dir", str(output_dir)])
    assert code == 0

    result = json.loads((output_dir / "result.json").read_text(encoding="utf-8"))
    assert result["status"] == "succeeded"
    assert "检出" in result["summary"] or "未检出" in result["summary"]
    types = {b["type"] for b in result["blocks"]}
    assert types <= {"markdown", "table", "code"}
    assert any(b["type"] == "table" and b["title"] == "检测结果" for b in result["blocks"])
    assert {"report.md", "results.json"} == {
        a["name"] for a in result["artifacts"]}
    assert result["data"]["detection"]["p_value"] == 0.003

    report_md = (output_dir / "report.md").read_text(encoding="utf-8")
    assert "数据集水印实验报告" in report_md
    assert "Copyright Traps" in report_md

    # 子进程与 CLI 同一套实现：命令、cwd、配置文件都要对
    command = seen["command"]
    assert command[1:3] == ["-m", "datamark.real.cli"]
    assert command[command.index("--method") + 1] == "canary_memory"
    assert command[command.index("--seed") + 1] == "7"
    assert Path(seen["cwd"]).name == "toolkit"


def test_local_model_marks_offline(tmp_path, monkeypatch, no_deps):
    model_dir = tmp_path / "tiny-model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}", encoding="utf-8")
    config_path = _write_config(
        tmp_path, models={"default": {"source": "local_path", "path": str(model_dir)}},
        common={"device": "auto", "dtype": "auto", "seed": 0})
    output_dir = tmp_path / "session"
    _fake_cli(monkeypatch)

    assert hub_adapter.main(["--config", config_path,
                             "--output-dir", str(output_dir)]) == 0
    cfg = json.loads((output_dir / "config.json").read_text(encoding="utf-8"))
    assert cfg["model"]["name"] == str(model_dir)
    assert cfg["model"]["local_files_only"] is True
    assert cfg["model"]["device"] == "cuda:0"      # auto → cuda:0
    assert cfg["model"]["dtype"] == "bfloat16"     # auto + cuda → bfloat16
    assert cfg["methods"]["canary_memory"]["epochs"] == 3


def test_cpu_auto_maps_float32(tmp_path, monkeypatch, no_deps):
    config_path = _write_config(tmp_path, common={"device": "cpu", "dtype": "auto", "seed": 7})
    output_dir = tmp_path / "session"
    _fake_cli(monkeypatch)

    hub_adapter.main(["--config", config_path, "--output-dir", str(output_dir)])
    cfg = json.loads((output_dir / "config.json").read_text(encoding="utf-8"))
    assert cfg["model"]["dtype"] == "float32"
    assert cfg["model"]["local_files_only"] is False   # HF 仓库默认允许联网下载


def test_token_statistics_forces_full(tmp_path, monkeypatch, no_deps):
    config_path = _write_config(
        tmp_path, params={"method": "token_statistics", "fraction": 0.5,
                          "finetune": "lora", "epochs": 3, "lr": 0.0002})
    output_dir = tmp_path / "session"
    _fake_cli(monkeypatch)

    hub_adapter.main(["--config", config_path, "--output-dir", str(output_dir)])
    cfg = json.loads((output_dir / "config.json").read_text(encoding="utf-8"))
    assert cfg["model"]["finetune"] == "full"
    assert cfg["model"]["optimizer"] == "adamw8bit"


@pytest.mark.parametrize("params,models,message", [
    ({"method": "nope"}, None, "未知水印方法"),
    ({"fraction": 1.5}, None, "注入比例"),
    ({"epochs": 99}, None, "epochs"),
    ({"finetune": "qlora"}, None, "微调方式"),
    (None, {"default": {"source": "api", "api": {"base_url": "http://x"}}}, "API 模型源"),
])
def test_invalid_params_fail(tmp_path, params, models, message):
    config_path = _write_config(tmp_path, params=params, models=models)
    output_dir = tmp_path / "session"
    output_dir.mkdir()

    code = hub_adapter.main(["--config", config_path, "--output-dir", str(output_dir)])
    assert code == 2
    result = json.loads((output_dir / "result.json").read_text(encoding="utf-8"))
    assert result["status"] == "failed"
    assert message in result["summary"]
    assert result["artifacts"] == []


def test_missing_model_role_fails(tmp_path):
    config_path = _write_config(tmp_path, models={})
    output_dir = tmp_path / "session"
    output_dir.mkdir()

    code = hub_adapter.main(["--config", config_path, "--output-dir", str(output_dir)])
    assert code == 2
    result = json.loads((output_dir / "result.json").read_text(encoding="utf-8"))
    assert "models.default" in result["summary"]


def test_cli_failure_propagates(tmp_path, monkeypatch, no_deps):
    config_path = _write_config(tmp_path)
    output_dir = tmp_path / "session"
    output_dir.mkdir()

    monkeypatch.setattr(hub_adapter.subprocess, "run",
                        lambda command, cwd=None, env=None:
                        subprocess.CompletedProcess(args=command, returncode=3))
    code = hub_adapter.main(["--config", config_path, "--output-dir", str(output_dir)])
    assert code == 2
    result = json.loads((output_dir / "result.json").read_text(encoding="utf-8"))
    assert result["status"] == "failed"
    assert "退出码 3" in result["summary"]


def test_bad_yaml_fails(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("[[[", encoding="utf-8")
    output_dir = tmp_path / "session"
    output_dir.mkdir()

    code = hub_adapter.main(["--config", str(bad), "--output-dir", str(output_dir)])
    assert code == 2
    result = json.loads((output_dir / "result.json").read_text(encoding="utf-8"))
    assert result["status"] == "failed"
