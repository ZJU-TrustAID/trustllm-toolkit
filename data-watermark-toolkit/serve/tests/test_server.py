# -*- coding: utf-8 -*-
"""章节前端服务测试（离线）：normalize/build_config 规则 + HTTP 端点契约。"""
import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

from serve import server as srv


@pytest.fixture()
def http():
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), srv.Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()


def _get(base, path):
    try:
        with urllib.request.urlopen(base + path) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8")
        try:
            return exc.code, json.loads(body)
        except json.JSONDecodeError:
            return exc.code, {}


def _post(base, path, payload):
    req = urllib.request.Request(
        base + path, data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode("utf-8"))


# ---------------------------------------------------------------- normalize
def test_normalize_local_model(tmp_path):
    model_dir = tmp_path / "m"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}", encoding="utf-8")
    runtime = srv.normalize_runtime({"offline": True,
                                     "model": {"name": str(model_dir),
                                               "device": "cpu", "dtype": "float32"}})
    assert runtime["model"]["name"] == str(model_dir)
    assert runtime["offline"] is True
    assert runtime["model"]["optimizer"] == "adamw"


def test_normalize_hf_repo_requires_online():
    runtime = srv.normalize_runtime(
        {"offline": False,
         "model": {"name": "HuggingFaceTB/SmolLM2-135M-Instruct", "device": "cuda:2"}})
    assert runtime["offline"] is False
    assert runtime["model"]["device"] == "cuda:2"
    assert runtime["model"]["dtype"] == srv.runtime_meta()["model"]["dtype"]


def test_normalize_hf_repo_offline_rejected():
    with pytest.raises(ValueError, match="离线"):
        srv.normalize_runtime(
            {"offline": True,
             "model": {"name": "HuggingFaceTB/SmolLM2-135M-Instruct"}})


def test_normalize_bad_device():
    with pytest.raises(ValueError, match="设备"):
        srv.normalize_runtime({"offline": False,
                               "model": {"name": "HuggingFaceTB/SmolLM2-135M-Instruct",
                                         "device": "tpu"}})


def test_normalize_token_statistics_forces_full():
    runtime = srv.normalize_runtime(
        {"offline": False, "model": {"name": "HuggingFaceTB/SmolLM2-135M-Instruct",
                                     "device": "cpu", "finetune": "lora"}},
        method="token_statistics")
    assert runtime["model"]["finetune"] == "full"
    assert "全参数微调" in runtime["note"]
    assert runtime["model"]["optimizer"] == "adamw8bit"


# ------------------------------------------------------------ build_config
def test_build_config_applies_runtime_and_overrides(tmp_path):
    model_dir = tmp_path / "m"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}", encoding="utf-8")
    runtime = srv.normalize_runtime(
        {"offline": True, "hf_home": "work/hf_cache",
         "model": {"name": str(model_dir), "device": "cuda:1", "dtype": "bfloat16",
                   "finetune": "lora", "lora_r": 8, "lora_alpha": 16}},
        method="canary_memory")
    cfg = srv.build_config("canary_memory", seed=11,
                           overrides={"epochs": 5, "repeats": 2, "evil_key": 1},
                           runtime=runtime)
    assert cfg["seed"] == 11
    assert cfg["model"]["name"] == str(model_dir)
    assert cfg["model"]["device"] == "cuda:1"
    assert cfg["model"]["lora_r"] == 8
    assert cfg["model"]["local_files_only"] is True
    assert cfg["methods"]["canary_memory"]["epochs"] == 5
    assert cfg["methods"]["canary_memory"]["repeats"] == 2
    assert "evil_key" not in cfg["methods"]["canary_memory"]   # 白名单之外被丢弃


def test_build_config_radio_for_token_statistics():
    cfg = srv.build_config("token_statistics", seed=1, overrides={})
    radio = json.loads(srv.RADIO_CONFIG.read_text(encoding="utf-8"))
    assert cfg["methods"]["token_statistics"]["enabled"] is True
    # radio 配置的分布统计参数被带上（与独立实验台同一份）
    assert cfg["methods"]["token_statistics"].get("n_watermarked_texts") == \
        radio["methods"]["token_statistics"].get("n_watermarked_texts")


# -------------------------------------------------------------------- HTTP
def test_child_env_cuda_handling(monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    env = srv.child_env({"cuda_visible_devices": "", "offline": False})
    assert "CUDA_VISIBLE_DEVICES" not in env      # 空值不再写入（会隐藏全部 GPU）
    assert env["HF_HUB_OFFLINE"] == "0"
    env = srv.child_env({"cuda_visible_devices": "2", "offline": True})
    assert env["CUDA_VISIBLE_DEVICES"] == "2"
    assert env["HF_HUB_OFFLINE"] == "1"


def test_health(http):
    status, payload = _get(http, "/api/health")
    assert status == 200
    assert payload["status"] == "ok"
    assert "model" in payload and "device" in payload


def test_index_and_static(http):
    with urllib.request.urlopen(http + "/") as resp:
        html = resp.read().decode("utf-8")
    assert "数据集水印实验台" in html
    assert 'href="static/styles.css"' in html          # 相对路径资源
    with urllib.request.urlopen(http + "/static/styles.css") as resp:
        assert b"--accent" in resp.read()
    status, _ = _get(http, "/static/nope.js")
    assert status == 404


def test_meta(http):
    status, meta = _get(http, "/api/meta")
    assert status == 200
    try:
        from datamark.real.methods import REAL_METHODS  # noqa: F401
        available = True
    except ImportError:
        available = False
    if available:
        assert len(meta["methods"]) == 5
        canary = meta["methods"]["canary_memory"]
        assert canary["family"] == "内容记忆型"
        assert "epochs" in canary["allowed"]
    else:
        assert "error" in meta        # torch 缺失时给出说明而不是 500


def test_unknown_method_rejected(http):
    status, payload = _post(http, "/api/run", {"method": "nope", "fraction": 0.5})
    assert status == 400
    assert "未知方法" in payload["error"]


def test_run_busy_409(http):
    fake = {"id": "fake", "method": "canary_memory", "fraction": 0.5, "seed": 1,
            "status": "running", "started": 0.0,
            "out_dir": srv.RUNS / "fake", "log_file": srv.RUNS / "fake.log",
            "proc": None, "command": "", "runtime": {}}
    srv.JOBS[fake["id"]] = fake
    try:
        status, payload = _post(http, "/api/run",
                                {"method": "canary_memory", "fraction": 0.5})
        assert status == 409
        assert payload["active"] == "fake"
    finally:
        srv.JOBS.pop(fake["id"], None)


def test_job_lookup(http):
    status, _ = _get(http, "/api/job?id=missing")
    assert status == 404
    status, _ = _get(http, "/api/job/missing/report")
    assert status == 404
