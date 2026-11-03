#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Toolkit 本地实验服务：给 ``webdemo/toolkit.html`` 提供参数化运行后端。

只依赖 Python 标准库；真正的实验通过 ``python -m datamark.real.cli run`` 子进程
执行，因此与 CLI 完全同一套实现。

启动（在 toolkit/ 下）::

    bash scripts/start_toolkit.sh          # http://127.0.0.1:8765

接口：
    GET  /api/meta                     方法列表、默认参数、运行环境
    POST /api/run                      {method, fraction, seed, overrides}
    GET  /api/job?id=<id>              状态、日志尾部、报告摘要
    GET  /api/job/<id>/report          完整 report.json
    POST /api/stop?id=<id>             终止当前实验
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import re
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE_CONFIG = ROOT / "configs" / "real_qwen35.json"
RADIO_CONFIG = ROOT / "configs" / "real_qwen35_radio.json"
PAGE = ROOT / "webdemo" / "toolkit.html"
RUNS = ROOT / "work" / "toolkit_runs"

#: 页面可调的方法专属参数（白名单，避免任意改配置）。
OVERRIDES = {
    "canary_memory": ["repeats", "trap_tokens", "epochs", "lr", "max_length"],
    "version_pair": ["delta", "max_new_tokens", "gamma", "window_k", "epochs", "lr"],
    "fictitious_knowledge": ["n_entities", "n_attributes", "passage_tokens", "epochs", "lr"],
    "token_statistics": ["n_watermarked_texts", "delta", "max_new_tokens",
                         "detect_tokens", "gamma", "window_k", "epochs", "lr"],
    "trigger_behavior": ["n_verify_docs", "epochs", "lr", "max_length"],
}
COMMON = ["epochs", "lr", "max_length"]

JOBS: dict[str, dict] = {}
LOCK = threading.Lock()


def _pkg_version(name: str):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def runtime_meta() -> dict:
    """服务进程的默认运行环境；页面可在此基础上逐项修改。"""
    cfg = json.loads(BASE_CONFIG.read_text(encoding="utf-8"))
    model = cfg.get("model", {})
    return {
        "python": sys.executable,
        "python_version": platform.python_version(),
        "service_versions": {
            "torch": _pkg_version("torch"),
            "transformers": _pkg_version("transformers"),
            "peft": _pkg_version("peft"),
            "tokenizers": _pkg_version("tokenizers"),
            "bitsandbytes": _pkg_version("bitsandbytes"),
        },
        "pythonpath": os.environ.get("PYTHONPATH", ""),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "hf_home": os.environ.get("HF_HOME", ""),
        "offline": os.environ.get("HF_HUB_OFFLINE", "0") == "1",
        "model": {
            "name": model.get("name", ""),
            "device": model.get("device", "cuda:0"),
            "dtype": model.get("dtype", "bfloat16"),
            "use_chat_template": bool(model.get("use_chat_template", False)),
            "finetune": model.get("finetune", "lora"),
            "lora_r": int(model.get("lora_r", 32)),
            "lora_alpha": int(model.get("lora_alpha", 64)),
            "lora_dropout": float(model.get("lora_dropout", 0.05)),
            "gradient_checkpointing": bool(model.get("gradient_checkpointing", False)),
            "optimizer": model.get("optimizer", "adamw"),
        },
        "runs_dir": str(RUNS.relative_to(ROOT)),
    }


def normalize_runtime(raw: dict, method: str | None = None) -> dict:
    """校验页面提交的运行环境，返回规范化后的 dict；非法时抛 ValueError。"""
    raw = dict(raw or {})
    model_raw = dict(raw.get("model") or {})
    default_model = runtime_meta()["model"]

    python = str(raw.get("python") or sys.executable).strip()
    py_path = Path(python)
    if not py_path.is_file():
        raise ValueError(f"Python 解释器不存在：{python}")
    if not os.access(str(py_path), os.X_OK):
        raise ValueError(f"Python 解释器不可执行：{python}")

    pythonpath = str(raw.get("pythonpath") or "").strip()
    if pythonpath:
        for part in pythonpath.split(":"):
            if part and not Path(part).exists():
                raise ValueError(f"PYTHONPATH 路径不存在：{part}")

    cuda = str(raw.get("cuda_visible_devices") or "").strip()
    if cuda and not re.fullmatch(r"[0-9]+(,[0-9]+)*", cuda):
        raise ValueError("CUDA_VISIBLE_DEVICES 需为逗号分隔的 GPU 编号，例如 0 或 0,1")

    hf_home = str(raw.get("hf_home") or "").strip()
    model_name = str(model_raw.get("name") or default_model["name"]).strip()
    if not model_name:
        raise ValueError("模型目录不能为空")
    model_path = Path(model_name)
    if not model_path.exists():
        raise ValueError(f"模型目录不存在：{model_name}")
    if not (model_path / "config.json").is_file():
        raise ValueError(f"模型目录缺少 config.json：{model_name}")

    device = str(model_raw.get("device") or default_model["device"]).strip()
    if device != "cpu" and not re.fullmatch(r"cuda(:[0-9]+)?", device):
        raise ValueError("设备需为 cpu 或 cuda、cuda:0 这类写法")
    dtype = str(model_raw.get("dtype") or default_model["dtype"]).strip()
    if dtype not in ("bfloat16", "float16", "float32"):
        raise ValueError("数据类型需为 bfloat16、float16 或 float32")
    finetune = str(model_raw.get("finetune") or default_model["finetune"]).strip()
    if finetune not in ("lora", "full"):
        raise ValueError("微调方式需为 lora 或 full")
    note = ""
    if method == "token_statistics" and finetune != "full":
        finetune = "full"
        note = "分布统计型需要用全参数微调才能把 green token 偏好学进参数，已自动切换为 full。"
    try:
        lora_r = int(model_raw.get("lora_r", default_model["lora_r"]))
        lora_alpha = int(model_raw.get("lora_alpha", default_model["lora_alpha"]))
        lora_dropout = float(model_raw.get("lora_dropout", default_model["lora_dropout"]))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"LoRA 参数需为数字：{exc}") from exc
    if lora_r < 1 or lora_alpha < 1 or not 0 <= lora_dropout < 1:
        raise ValueError("LoRA 参数超出合理范围")
    return {
        "python": str(py_path),
        "pythonpath": pythonpath,
        "cuda_visible_devices": cuda,
        "hf_home": hf_home,
        "offline": bool(raw.get("offline", True)),
        "model": {
            "name": str(model_path),
            "device": device,
            "dtype": dtype,
            "use_chat_template": bool(model_raw.get("use_chat_template",
                                                    default_model["use_chat_template"])),
            "finetune": finetune,
            "lora_r": lora_r,
            "lora_alpha": lora_alpha,
            "lora_dropout": lora_dropout,
            "gradient_checkpointing": bool(model_raw.get(
                "gradient_checkpointing", default_model["gradient_checkpointing"])),
            "optimizer": default_model.get("optimizer", "adamw"),
        },
        "note": note,
    }


def child_env(runtime: dict) -> dict:
    """构造子进程环境变量（不污染服务进程）。"""
    env = dict(os.environ)
    extra = runtime.get("pythonpath") or ""
    if extra:
        old = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = extra + ((":" + old) if old else "")
    env["CUDA_VISIBLE_DEVICES"] = runtime.get("cuda_visible_devices", "")
    hf_home = runtime.get("hf_home")
    if hf_home:
        env["HF_HOME"] = hf_home
    if runtime.get("offline", True):
        env["HF_HUB_OFFLINE"] = "1"
        env["TRANSFORMERS_OFFLINE"] = "1"
    else:
        env["HF_HUB_OFFLINE"] = "0"
        env["TRANSFORMERS_OFFLINE"] = "0"
    return env


def check_runtime(runtime: dict) -> tuple[str | None, dict]:
    """用目标 Python 解释器检查依赖，并返回实际版本。"""
    env = child_env(runtime)
    imports = ["import torch", "import transformers", "import tokenizers"]
    if runtime["model"]["finetune"] == "lora":
        imports.append("import peft")
    else:
        imports.append("import bitsandbytes")
    code = "; ".join(imports) + (
        "; import torch, transformers; "
        "print(torch.__version__); print(transformers.__version__)")
    try:
        proc = subprocess.run([runtime["python"], "-c", code], cwd=str(ROOT),
                              env=env, capture_output=True, text=True, timeout=120)
    except subprocess.TimeoutExpired:
        return "Python 环境检查超时（120 秒）", {}
    except OSError as exc:
        return f"无法启动 Python 解释器：{exc}", {}
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-3:]
        return "运行环境缺少依赖：" + " | ".join(tail), {}
    lines = [x.strip() for x in (proc.stdout or "").strip().splitlines() if x.strip()]
    versions = {"torch": lines[0] if lines else None,
                "transformers": lines[1] if len(lines) > 1 else None}
    return None, versions


def method_meta() -> dict:
    from sys import path as _path
    _path.insert(0, str(ROOT))
    from datamark.real.methods import REAL_METHODS

    cfg = json.loads(BASE_CONFIG.read_text(encoding="utf-8"))
    methods = {}
    for name, cls in REAL_METHODS.items():
        raw = dict(cfg.get("methods", {}).get(name, {}))
        raw.pop("enabled", None)
        methods[name] = {
            "name": name,
            "family": cls.family,
            "access": cls.access,
            "reference": cls.reference,
            "defaults": raw,
            "allowed": OVERRIDES.get(name, []),
            "config": ("configs/real_qwen35_radio.json" if name == "token_statistics"
                       else "configs/real_qwen35.json"),
        }
    return {
        "methods": methods,
        "model": cfg.get("model", {}).get("name"),
        "device": cfg.get("model", {}).get("device"),
        "finetune": cfg.get("model", {}).get("finetune"),
        "runtime": runtime_meta(),
        "runs_dir": str(RUNS.relative_to(ROOT)),
    }


def make_config(method: str, seed: int, overrides: dict,
                runtime: dict | None = None) -> dict:
    base = RADIO_CONFIG if method == "token_statistics" else BASE_CONFIG
    cfg = json.loads(base.read_text(encoding="utf-8"))
    cfg["seed"] = int(seed)
    if runtime:
        model = cfg.setdefault("model", {})
        incoming = runtime["model"]
        model.update({
            "name": incoming["name"],
            "device": incoming["device"],
            "dtype": incoming["dtype"],
            "use_chat_template": incoming["use_chat_template"],
            "finetune": incoming["finetune"],
            "lora_r": incoming["lora_r"],
            "lora_alpha": incoming["lora_alpha"],
            "lora_dropout": incoming["lora_dropout"],
            "gradient_checkpointing": incoming["gradient_checkpointing"],
            "hf_home": runtime.get("hf_home") or model.get("hf_home", "work/hf_cache"),
            "local_files_only": bool(runtime.get("offline", True)),
            "optimizer": "adamw8bit" if incoming["finetune"] == "full" else "adamw",
        })
    target = cfg.setdefault("methods", {}).setdefault(method, {})
    allowed = set(OVERRIDES.get(method, [])) | set(COMMON)
    for key, value in overrides.items():
        if key not in allowed:
            continue
        if key == "epochs":
            value = int(value)
        elif key == "n_entities" or key == "n_attributes":
            value = int(value)
        elif key in ("repeats", "trap_tokens", "n_watermarked_texts",
                     "max_new_tokens", "detect_tokens", "window_k",
                     "n_verify_docs", "max_length"):
            value = int(value)
        else:
            value = float(value)
        target[key] = value
    return cfg


def job_summary(job: dict) -> dict:
    proc: subprocess.Popen | None = job.get("proc")
    status = job["status"]
    if proc is not None and proc.poll() is not None:
        status = "done" if proc.returncode == 0 else "failed"
        job["status"] = status
    elapsed = time.time() - job["started"]
    out = {
        "id": job["id"],
        "method": job["method"],
        "fraction": job["fraction"],
        "seed": job["seed"],
        "status": status,
        "elapsed": round(elapsed, 1),
        "out_dir": str(job["out_dir"].relative_to(ROOT)),
        "command": job["command"],
        "runtime": job.get("runtime"),
    }
    log = job["log_file"]
    if log.exists():
        text = log.read_text(encoding="utf-8", errors="replace")
        out["log_tail"] = text[-4000:]
    report = job["out_dir"] / "report.json"
    if report.exists():
        try:
            data = json.loads(report.read_text(encoding="utf-8"))
            det = data.get("detection") or {}
            out["report"] = {
                "p": det.get("p_value"),
                "decision": det.get("decision"),
                "score": det.get("score"),
                "n": det.get("n"),
                "train": data.get("train"),
                "elapsed_sec": data.get("elapsed_sec"),
            }
        except json.JSONDecodeError:
            pass
    return out


class Handler(BaseHTTPRequestHandler):
    server_version = "DataMarkToolkit/1.0"

    def log_message(self, *args):  # 安静一点
        pass

    # ------------------------------------------------------------------
    def _json(self, code: int, payload) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length") or 0)
        if not length:
            return {}
        return json.loads(self.rfile.read(length).decode("utf-8") or "{}")

    def do_GET(self):  # noqa: N802
        if self.path in ("/", "/index.html", "/toolkit.html"):
            body = PAGE.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if self.path == "/api/meta":
            self._json(200, method_meta())
            return
        if self.path.startswith("/api/job/"):
            job_id = self.path.split("/")[3].split("?")[0]
            job = JOBS.get(job_id)
            if not job:
                self._json(404, {"error": "job not found"})
                return
            if self.path.endswith("/report"):
                report = job["out_dir"] / "report.json"
                if not report.exists():
                    self._json(404, {"error": "report not ready"})
                    return
                self._json(200, json.loads(report.read_text(encoding="utf-8")))
                return
            self._json(200, job_summary(job))
            return
        if self.path.startswith("/api/job?"):
            from urllib.parse import parse_qs, urlparse

            job_id = parse_qs(urlparse(self.path).query).get("id", [""])[0]
            job = JOBS.get(job_id)
            self._json(200 if job else 404,
                       job_summary(job) if job else {"error": "job not found"})
            return
        self._json(404, {"error": "not found"})

    def do_POST(self):  # noqa: N802
        if self.path == "/api/run":
            self._run()
            return
        if self.path == "/api/stop":
            self._stop()
            return
        self._json(404, {"error": "not found"})

    # ------------------------------------------------------------------
    def _run(self):
        try:
            body = self._read_json()
            method = str(body.get("method") or "")
            fraction = float(body.get("fraction", 0.5))
            seed = int(body.get("seed", 42))
            overrides = dict(body.get("overrides") or {})
            env_raw = dict(body.get("env") or {})
        except (ValueError, json.JSONDecodeError) as exc:
            self._json(400, {"error": f"参数错误：{exc}"})
            return
        with LOCK:
            active = [j for j in JOBS.values() if j["status"] == "running"]
            if active:
                self._json(409, {"error": "已有实验在运行，请等待完成或先停止",
                                 "active": active[0]["id"]})
                return
            if method not in OVERRIDES:
                self._json(400, {"error": f"未知方法：{method}"})
                return
            if not 0.0 <= fraction <= 1.0:
                self._json(400, {"error": "fraction 需在 [0,1]"})
                return
            try:
                runtime = normalize_runtime(env_raw, method)
            except ValueError as exc:
                self._json(400, {"error": f"运行环境配置错误：{exc}"})
                return
            dep_err, versions = check_runtime(runtime)
            if dep_err:
                self._json(400, {"error": dep_err})
                return
            runtime["versions"] = versions
            RUNS.mkdir(parents=True, exist_ok=True)
            job_id = time.strftime("%Y%m%d-%H%M%S") + "-" + method.replace("_", "-")
            job_dir = RUNS / job_id
            job_dir.mkdir(parents=True, exist_ok=True)
            cfg = make_config(method, seed, overrides, runtime)
            cfg_path = job_dir / "config.json"
            cfg_path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2) + "\n",
                                encoding="utf-8")
            (job_dir / "runtime.json").write_text(
                json.dumps(runtime, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8")
            log_file = job_dir / "run.log"
            out_dir = job_dir / "result"
            command = [runtime["python"], "-m", "datamark.real.cli",
                       "run", "--config", str(cfg_path.relative_to(ROOT)),
                       "--method", method, "--fraction", str(fraction),
                       "--seed", str(seed),
                       "--out", str(out_dir.relative_to(ROOT))]
            child = child_env(runtime)
            log_handle = log_file.open("w", encoding="utf-8")
            proc = subprocess.Popen(command, cwd=str(ROOT), env=child,
                                    stdout=log_handle, stderr=subprocess.STDOUT)
            log_handle.close()
            job = {"id": job_id, "method": method, "fraction": fraction,
                   "seed": seed, "status": "running", "started": time.time(),
                   "out_dir": out_dir, "log_file": log_file, "proc": proc,
                   "command": " ".join(command), "runtime": runtime}
            JOBS[job_id] = job
            self._json(200, {"id": job_id, "status": "running",
                             "command": job["command"],
                             "config": str(cfg_path.relative_to(ROOT)),
                             "runtime": runtime})

    def _stop(self):
        body = self._read_json()
        job = JOBS.get(str(body.get("id") or ""))
        if not job:
            self._json(404, {"error": "job not found"})
            return
        proc = job.get("proc")
        if proc and proc.poll() is None:
            proc.terminate()
            job["status"] = "stopped"
        self._json(200, {"id": job["id"], "status": job["status"]})


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()
    if not PAGE.exists():
        print(f"缺少页面：{PAGE}")
        return 1
    server = None
    port = args.port
    for candidate in range(args.port, args.port + 10):
        try:
            server = ThreadingHTTPServer((args.host, candidate), Handler)
            port = candidate
            break
        except OSError:
            continue
    if server is None:
        print(f"[toolkit] 端口 {args.port}–{args.port + 9} 都被占用，无法启动")
        return 1
    print(f"[toolkit] 服务已启动：http://{args.host}:{port}/")
    print(f"[toolkit] 实验输出目录：{RUNS}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[toolkit] 退出")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
