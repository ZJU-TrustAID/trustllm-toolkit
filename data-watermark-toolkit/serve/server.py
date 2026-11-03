#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""DataMark 章节前端服务（TrustLLM hub 接入）。

只依赖 Python 标准库；给 ``serve/static/`` 的实验台页面提供与
``toolkit/scripts/toolkit_server.py`` 相同的 API 面（meta / run / job / stop），
但默认运行环境（模型、设备、dtype、seed、端口）来自 hub 渲染的启动配置，
且**绑定固定端口**（hub 按端口探活，不再自动顺延）。真正的实验仍通过
``python -m datamark.real.cli run`` 子进程执行，与 CLI / 独立实验台同一套实现。

hub 启动（chapter.yaml 的 frontend.server.command）::

    conda run -n datamark python serve/server.py --config <hub 渲染的 json>

也可独立启动（不带 --config，使用 toolkit/configs/real_qwen35.json 的默认值）::

    python serve/server.py --port 8785

接口（页面用相对路径访问，独立部署与 /proxy/datamark/ 代理路径下都正确）：
    GET  /                       实验台页面（serve/static/index.html）
    GET  /api/health             健康检查（hub ready_path）
    GET  /api/meta               方法列表、默认参数、运行环境
    POST /api/run                {method, fraction, seed, overrides, env}
    GET  /api/job?id=<id>        状态、日志尾部、报告摘要
    GET  /api/job/<id>/report    完整 report.json
    POST /api/stop               {id} 终止当前实验
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import re
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]          # data-watermark-toolkit/
TOOLKIT = ROOT / "toolkit"
STATIC = Path(__file__).resolve().parent / "static"
PAGE = STATIC / "index.html"
RUNS = TOOLKIT / "work" / "toolkit_runs"            # 与独立实验台共用输出目录

# 复用独立实验台的白名单与依赖检查（同一套规则，避免两处漂移）。
sys.path.insert(0, str(TOOLKIT / "scripts"))
from toolkit_server import OVERRIDES, COMMON, check_runtime  # noqa: E402

BASE_CONFIG = TOOLKIT / "configs" / "real_qwen35.json"
RADIO_CONFIG = TOOLKIT / "configs" / "real_qwen35_radio.json"
FALLBACK_REPO = "HuggingFaceTB/SmolLM2-135M-Instruct"

# 页面静态资源白名单（相对路径引用，独立与代理部署下都成立）。
STATIC_FILES = {"styles.css": "text/css; charset=utf-8",
                "app.js": "application/javascript; charset=utf-8"}

JOBS: dict[str, dict] = {}
LOCK = threading.Lock()


def _pkg_version(name: str):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def load_hub_config(path: str | None) -> dict:
    """读取 hub 渲染的启动配置；缺省时给一组可用的独立默认值。"""
    cfg: dict = {}
    if path:
        cfg = json.loads(Path(path).read_text(encoding="utf-8"))
    model = str(cfg.get("model") or "").strip()
    if not model:
        base = json.loads(BASE_CONFIG.read_text(encoding="utf-8")) if BASE_CONFIG.exists() else {}
        candidate = str(base.get("model", {}).get("name") or "")
        model = candidate if Path(candidate).exists() else FALLBACK_REPO
    device = str(cfg.get("device") or "auto").strip()
    if device in ("", "auto"):
        device = "cuda:0"
    dtype = str(cfg.get("dtype") or "auto").strip()
    if dtype in ("", "auto"):
        dtype = "bfloat16" if device.startswith("cuda") else "float32"
    try:
        seed = int(cfg.get("seed", 42))
    except (TypeError, ValueError):
        seed = 42
    web = cfg.get("web") or {}
    return {"model": model, "device": device, "dtype": dtype, "seed": seed,
            "host": str(web.get("host") or "127.0.0.1"),
            "port": int(web.get("port") or 8785)}


HUB_CFG = load_hub_config(None)   # main() 启动时用 --config 覆盖


def runtime_meta() -> dict:
    """服务进程的默认运行环境；页面可在此基础上逐项修改。"""
    return {
        "python": sys.executable,
        "python_version": ".".join(map(str, sys.version_info[:3])),
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
            "name": HUB_CFG["model"],
            "device": HUB_CFG["device"],
            "dtype": HUB_CFG["dtype"],
            "use_chat_template": False,
            "finetune": "lora",
            "lora_r": 32,
            "lora_alpha": 64,
            "lora_dropout": 0.05,
            "gradient_checkpointing": False,
            "optimizer": "adamw",
        },
        "seed": HUB_CFG["seed"],
        "runs_dir": str(RUNS.relative_to(ROOT)),
    }


def _is_local_model_dir(name: str) -> bool:
    path = Path(name)
    return path.is_dir() and (path / "config.json").is_file()


def normalize_runtime(raw: dict, method: str | None = None) -> dict:
    """校验页面提交的运行环境，返回规范化后的 dict；非法时抛 ValueError。

    与 toolkit_server.normalize_runtime 同规则，差异只有两点：默认值来自 hub
    启动配置；模型除本地目录外还接受 HuggingFace 仓库 ID（需允许联网下载）。
    """
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
    offline = bool(raw.get("offline", True))
    if _is_local_model_dir(model_name):
        pass
    elif re.fullmatch(r"[\w.-]+(/[\w.-]+)+", model_name):
        if offline:
            raise ValueError(f"模型 {model_name} 不是本地目录；"
                             "请在实验台关闭「离线模式」以允许从 HuggingFace 下载，"
                             "或在 hub 顶栏改用本地模型路径")
    else:
        raise ValueError(f"模型目录不存在或缺少 config.json：{model_name}")

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
        "offline": offline,
        "model": {
            "name": model_name,
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
            "optimizer": "adamw8bit" if finetune == "full" else "adamw",
        },
        "note": note,
    }


def method_meta() -> dict:
    """方法列表与运行环境默认值；torch 缺失时仍返回说明性错误。"""
    try:
        sys.path.insert(0, str(TOOLKIT))
        from datamark.real.methods import REAL_METHODS
    except Exception as exc:  # noqa: BLE001 — 让页面能看到环境问题
        return {"error": f"无法加载 datamark.real（{type(exc).__name__}: {exc}）；"
                         "真实实验不可用，请检查 datamark 环境的 torch/transformers/peft",
                "methods": {}, "runtime": runtime_meta(),
                "runs_dir": str(RUNS.relative_to(ROOT))}
    methods = {}
    for name, cls in REAL_METHODS.items():
        base = RADIO_CONFIG if name == "token_statistics" else BASE_CONFIG
        raw = dict(json.loads(base.read_text(encoding="utf-8"))
                   .get("methods", {}).get(name, {}))
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
    return {"methods": methods, "runtime": runtime_meta(),
            "runs_dir": str(RUNS.relative_to(ROOT))}


def build_config(method: str, seed: int, overrides: dict,
                 runtime: dict | None = None) -> dict:
    """组装一次实验的 datamark.real 配置（与 toolkit_server.make_config 同规则）。"""
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
            "optimizer": incoming.get("optimizer", "adamw"),
        })
    target = cfg.setdefault("methods", {}).setdefault(method, {})
    target.setdefault("enabled", True)
    allowed = set(OVERRIDES.get(method, [])) | set(COMMON)
    for key, value in overrides.items():
        if key not in allowed:
            continue
        if key in ("epochs", "n_entities", "n_attributes", "repeats", "trap_tokens",
                   "n_watermarked_texts", "max_new_tokens", "detect_tokens",
                   "window_k", "n_verify_docs", "max_length"):
            value = int(value)
        else:
            value = float(value)
        target[key] = value
    return cfg


def child_env(runtime: dict) -> dict:
    """构造子进程环境变量（不污染服务进程）。

    与 toolkit_server.child_env 的差异：CUDA_VISIBLE_DEVICES 仅在显式给出时才设置——
    设成空字符串会让 CUDA 隐藏全部 GPU（hub 子进程环境里常见为空）。
    """
    env = dict(os.environ)
    extra = runtime.get("pythonpath") or ""
    if extra:
        old = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = extra + ((":" + old) if old else "")
    cvd = str(runtime.get("cuda_visible_devices") or "").strip()
    if cvd:
        env["CUDA_VISIBLE_DEVICES"] = cvd
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
                "alpha": det.get("alpha"),
                "train": data.get("train"),
                "elapsed_sec": data.get("elapsed_sec"),
            }
        except json.JSONDecodeError:
            pass
    return out


class Handler(BaseHTTPRequestHandler):
    server_version = "DataMarkServe/1.0"

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

    def _file(self, path: Path, ctype: str) -> None:
        body = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length") or 0)
        if not length:
            return {}
        return json.loads(self.rfile.read(length).decode("utf-8") or "{}")

    def do_GET(self):  # noqa: N802
        path = self.path.split("?")[0]
        if path in ("/", "/index.html"):
            self._file(PAGE, "text/html; charset=utf-8")
            return
        if path.startswith("/static/"):
            name = path[len("/static/"):]
            ctype = STATIC_FILES.get(name)
            target = STATIC / name
            if ctype and target.is_file():
                self._file(target, ctype)
            else:
                self._json(404, {"error": "not found"})
            return
        if path == "/api/health":
            running = any(j["status"] == "running" for j in JOBS.values())
            self._json(200, {"status": "ok", "model": HUB_CFG["model"],
                             "device": HUB_CFG["device"], "dtype": HUB_CFG["dtype"],
                             "seed": HUB_CFG["seed"], "job_running": running})
            return
        if path == "/api/meta":
            self._json(200, method_meta())
            return
        if path.startswith("/api/job/"):
            job_id = path.split("/")[3]
            job = JOBS.get(job_id)
            if not job:
                self._json(404, {"error": "job not found"})
                return
            if path.endswith("/report"):
                report = job["out_dir"] / "report.json"
                if not report.exists():
                    self._json(404, {"error": "report not ready"})
                    return
                self._json(200, json.loads(report.read_text(encoding="utf-8")))
                return
            self._json(200, job_summary(job))
            return
        if path == "/api/job":
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
            seed = int(body.get("seed", HUB_CFG["seed"]))
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
            cfg = build_config(method, seed, overrides, runtime)
            cfg_path = job_dir / "config.json"
            cfg_path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2) + "\n",
                                encoding="utf-8")
            (job_dir / "runtime.json").write_text(
                json.dumps(runtime, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8")
            log_file = job_dir / "run.log"
            out_dir = job_dir / "result"
            command = [runtime["python"], "-m", "datamark.real.cli",
                       "run", "--config", str(cfg_path.relative_to(TOOLKIT)),
                       "--method", method, "--fraction", str(fraction),
                       "--seed", str(seed),
                       "--out", str(out_dir.relative_to(TOOLKIT))]
            child = child_env(runtime)
            log_handle = log_file.open("w", encoding="utf-8")
            proc = subprocess.Popen(command, cwd=str(TOOLKIT), env=child,
                                    stdout=log_handle, stderr=subprocess.STDOUT)
            log_handle.close()
            job = {"id": job_id, "method": method, "fraction": fraction,
                   "seed": seed, "status": "running", "started": time.time(),
                   "out_dir": out_dir, "log_file": log_file, "proc": proc,
                   "command": " ".join(command), "runtime": runtime}
            JOBS[job_id] = job
            self._json(200, {"id": job_id, "status": "running",
                             "command": job["command"],
                             "config": str(cfg_path.relative_to(TOOLKIT)),
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
    global HUB_CFG
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None, help="hub 渲染的启动配置 json")
    ap.add_argument("--host", default=None)
    ap.add_argument("--port", type=int, default=None)
    args = ap.parse_args()
    HUB_CFG = load_hub_config(args.config)
    host = args.host or HUB_CFG["host"]
    port = args.port or HUB_CFG["port"]
    if not PAGE.exists():
        print(f"缺少页面：{PAGE}")
        return 1
    try:
        server = ThreadingHTTPServer((host, port), Handler)
    except OSError as exc:
        print(f"[datamark] 端口 {host}:{port} 无法绑定（{exc}）；hub 按该端口探活，请勿占用")
        return 1
    print(f"[datamark] 服务已启动：http://{host}:{port}/")
    print(f"[datamark] 模型默认值：{HUB_CFG['model']}（{HUB_CFG['device']} / {HUB_CFG['dtype']}）")
    print(f"[datamark] 实验输出目录：{RUNS}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[datamark] 退出")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
