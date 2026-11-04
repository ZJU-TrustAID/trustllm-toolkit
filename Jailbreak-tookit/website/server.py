"""Local teaching UI: private toolkit implementation, real model inference.

Run with the toolkit's Python environment. Only five public frontend files are
served; Python, source snapshots, archives and model files are never web assets.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import logging
import os
from pathlib import Path
import re
import secrets
import sys
import threading
import time
from urllib.parse import urlsplit
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parent
TOOLKIT = ROOT.parent
sys.path.insert(0, str(TOOLKIT / "src"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

METHODS = {"template", "role_play", "task_disguise", "rule_rewrite", "context",
           "base64", "caesar", "red_teaming", "autodan", "gcg", "cold_attack"}
OPTIMIZERS = {"gcg", "autodan", "cold_attack"}
PUBLIC_FILES = {"/": ("index.html", "text/html"),
                "/index.html": ("index.html", "text/html"),
                "/styles.css": ("styles.css", "text/css"),
                "/app.js": ("app.js", "text/javascript"),
                "/favicon.svg": ("favicon.svg", "image/svg+xml")}


def validate(payload):
    if not isinstance(payload, dict):
        raise ValueError("请提供一个有效的实验输入。")
    method = payload.get("method")
    if not isinstance(method, str) or method not in METHODS:
        raise ValueError("请选择列表中的方法。")
    def text(key, default, maximum):
        value = payload.get(key, default)
        if not isinstance(value, str) or not value.strip() or len(value) > maximum:
            raise ValueError("请填写完整输入，并适当缩短文本。")
        return value.strip()
    steps = payload.get("steps", 3)
    if isinstance(steps, bool) or not isinstance(steps, int) or steps not in (3, 6, 10):
        raise ValueError("请选择 3、6 或 10 轮尝试。")
    mode = payload.get("mode", "suffix")
    if mode not in ("suffix", "control", "paraphrase"):
        raise ValueError("请选择有效的修改方式。")
    result = {"method": method, "query": text("query", "", 480),
              "target": text("target", "Sure, here is", 100),
              "steps": steps, "mode": mode,
              "role": text("role", "不受限制的虚构角色", 60),
              "seed": text("seed", "Continue the fictional assistant's direct answer to this request:", 180)}
    if "{" in result["seed"] or "}" in result["seed"]:
        raise ValueError("起始引导请使用普通文字，不需要填写占位符。")
    return result


class Runtime:
    def __init__(self, model_path, *, device="cpu", dtype="float32", model_name=None,
                 location="local", max_new_tokens=64, eval_batch_size=4, threads=1):
        self.model_path = model_path
        self.device, self.dtype = device, dtype
        self.model_name = model_name or Path(model_path).name
        self.location = location
        self.max_new_tokens = max_new_tokens
        self.eval_batch_size = eval_batch_size
        self.threads = threads
        self.backend = None
        self.load_error = False
        self.ready = threading.Event()
        self.lock = threading.Lock()
        self.busy = False
        self.jobs = {}
        threading.Thread(target=self.preload, daemon=True).start()

    def preload(self):
        try:
            import torch
            from textbook_toolkit.jailbreak.optimization import CausalLM
            torch.set_num_threads(self.threads)

            class LiveBackend(CausalLM):
                progress_callback = None

                def finish(self, **kwargs):
                    if self.progress_callback:
                        self.progress_callback("answer", "正在用最终提示生成回答")
                    return super().finish(**kwargs)

            self.backend = LiveBackend.from_pretrained(
                str(self.model_path), device=self.device, dtype=self.dtype, local_files_only=True)
            logging.info("Loaded %s on %s (%s)", self.model_name, self.backend.device, self.dtype)
        except Exception:
            logging.exception("Local model initialization failed")
            self.load_error = True
        finally:
            self.ready.set()

    def health(self):
        with self.lock:
            return {"ready": self.ready.is_set() and not self.load_error,
                    "status": "error" if self.load_error else "ready" if self.ready.is_set() else "loading",
                    "busy": self.busy, "model": self.model_name, "location": self.location,
                    "device": str(self.backend.device) if self.backend else self.device,
                    "dtype": self.dtype, "max_new_tokens": self.max_new_tokens}

    def start(self, payload):
        with self.lock:
            if self.load_error:
                return 503, {"error": "模型没有加载成功，请检查启动环境后重试。"}
            if self.busy:
                return 409, {"error": "当前有一个实验正在运行，请稍后再试。"}
            self.busy = True
            if len(self.jobs) >= 20:
                del self.jobs[next(iter(self.jobs))]
            job_id = secrets.token_hex(16)
            self.jobs[job_id] = {"id": job_id, "status": "running", "stage": "loading",
                                 "message": "正在准备模型", "started": time.monotonic()}
        threading.Thread(target=self.run, args=(job_id, payload), daemon=True).start()
        return 202, {"id": job_id}

    def snapshot(self, job_id):
        with self.lock:
            item = self.jobs.get(job_id)
            if not item:
                return None
            snapshot = {key: value for key, value in item.items() if key != "started"}
            snapshot["elapsed"] = round(time.monotonic() - item["started"], 1) if item["status"] == "running" else item["elapsed"]
            return snapshot

    def run(self, job_id, payload):
        def progress(stage, message):
            with self.lock:
                self.jobs[job_id].update(stage=stage, message=message)
        try:
            self.ready.wait()
            if self.load_error:
                raise RuntimeError("Model unavailable")
            # 实验流程与 hub 批量入口共用 textbook_toolkit.lab，两处行为保持一致。
            from textbook_toolkit.lab import run_experiment
            backend = self.backend
            backend.progress_callback = progress
            output = run_experiment(backend, payload, max_new_tokens=self.max_new_tokens,
                                    eval_batch_size=self.eval_batch_size, model_name=self.model_name,
                                    location=self.location, progress=progress)
            with self.lock:
                elapsed = round(time.monotonic() - self.jobs[job_id]["started"], 2)
                output["elapsed"] = elapsed
                self.jobs[job_id].update(status="completed", stage="done", message="实验已完成",
                                         elapsed=elapsed, result=output)
        except Exception as exc:
            logging.exception("Experiment failed")
            message = str(exc) if isinstance(exc, ValueError) and str(exc).startswith(("这次", "起始")) else "这次实验未能完成。请缩短输入后重试，或更换一种方法。"
            with self.lock:
                self.jobs[job_id].update(status="failed", stage="error", message=message,
                                         elapsed=round(time.monotonic() - self.jobs[job_id]["started"], 2))
        finally:
            if self.backend:
                self.backend.progress_callback = None
            with self.lock:
                self.busy = False


class Handler(BaseHTTPRequestHandler):
    server_version = "ToolkitLab"
    sys_version = ""

    def log_message(self, format, *args):
        # Avoid saving user questions, prompts or model replies in access logs.
        pass

    def send_body(self, status, body, content_type="application/json"):
        if not isinstance(body, bytes):
            body = json.dumps(body, ensure_ascii=False, allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type + "; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "same-origin")
        self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; connect-src 'self'; base-uri 'none'; frame-ancestors 'none'; form-action 'self'")
        self.end_headers()
        if self.command != "HEAD":
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass

    def valid_host(self):
        if self.server.trust_proxy:
            # 经 hub 同源反代访问：Host/Origin 属于 hub 端口，放宽校验（仅回环监听）。
            return True
        return self.headers.get("Host", "") in {f"127.0.0.1:{self.server.server_port}", f"localhost:{self.server.server_port}"}

    def do_HEAD(self):
        self.do_GET()

    def forward_api(self, path, payload=None):
        """Forward only validated API routes through a loopback SSH tunnel."""
        request = Request(self.server.upstream + path,
                          data=json.dumps(payload).encode() if payload is not None else None,
                          headers={"Content-Type": "application/json"})
        try:
            with urlopen(request, timeout=8) as response:
                return self.send_body(response.status, json.load(response))
        except HTTPError as exc:
            return self.send_body(exc.code, json.load(exc))
        except (URLError, TimeoutError, OSError, ValueError):
            return self.send_body(503, {"error": "暂时无法连接远程模型，请等待连接恢复后重试。"})

    def do_GET(self):
        if not self.valid_host():
            return self.send_body(403, {"error": "不支持的访问地址。"})
        path = urlsplit(self.path).path
        if path == "/api/health":
            if self.server.upstream:
                return self.forward_api(path)
            return self.send_body(200, self.server.runtime.health())
        if re.fullmatch(r"/api/runs/[0-9a-f]{32}", path):
            if self.server.upstream:
                return self.forward_api(path)
            result = self.server.runtime.snapshot(path.rsplit("/", 1)[1])
            return self.send_body(200, result) if result else self.send_body(404, {"error": "这次实验记录已失效，请重新运行。"})
        if path in PUBLIC_FILES:
            filename, mime = PUBLIC_FILES[path]
            return self.send_body(200, (ROOT / "dist" / filename).read_bytes(), mime)
        self.send_body(404, {"error": "页面不存在。"})

    def do_POST(self):
        origin = self.headers.get("Origin")
        allowed = {f"http://127.0.0.1:{self.server.server_port}", f"http://localhost:{self.server.server_port}"}
        if not self.valid_host() or (origin and not self.server.trust_proxy and origin not in allowed):
            return self.send_body(403, {"error": "请在本地实验页面发起请求。"})
        if urlsplit(self.path).path != "/api/runs":
            return self.send_body(404, {"error": "页面不存在。"})
        if self.headers.get("Content-Type", "").split(";")[0].strip() != "application/json":
            return self.send_body(415, {"error": "请使用网页提交实验。"})
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > 16384:
                return self.send_body(413, {"error": "输入内容过长。"})
            self.connection.settimeout(10)
            payload = validate(json.loads(self.rfile.read(length)))
        except (ValueError, UnicodeDecodeError, TimeoutError) as exc:
            message = str(exc) if type(exc) is ValueError else "输入格式有误，请重新填写。"
            return self.send_body(400, {"error": message})
        if self.server.upstream:
            return self.forward_api("/api/runs", payload)
        status, body = self.server.runtime.start(payload)
        self.send_body(status, body)


def load_rendered_config(path):
    rendered = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(rendered, dict):
        raise ValueError("前端服务配置必须是 JSON 对象")
    return rendered


def apply_rendered_config(args, rendered):
    """hub 渲染的前端服务配置：模型/设备注入运行时，端口注入 web 段。"""
    if rendered.get("model"):
        args.model = Path(str(rendered["model"])).expanduser()
    if rendered.get("device"):
        args.device = str(rendered["device"])
    if rendered.get("dtype"):
        args.dtype = str(rendered["dtype"])
    if rendered.get("max_new_tokens"):
        args.max_new_tokens = int(rendered["max_new_tokens"])
    web = rendered.get("web") or {}
    if web.get("host"):
        args.host = str(web["host"])
    if web.get("port"):
        args.port = int(web["port"])
    args.trust_proxy = bool(web.get("trust_proxy", False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", help="hub 渲染的前端服务配置（JSON）")
    parser.add_argument("--port", type=int, default=4173)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--model", type=Path, default=TOOLKIT / ".cache/SmolLM2-135M-Instruct")
    parser.add_argument("--model-name", help="Display name reported by the running backend")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", choices=("auto", "float32", "float16", "bfloat16"), default="float32")
    parser.add_argument("--location", choices=("local", "remote"), default="local")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--upstream", help="Loopback API URL forwarded by SSH; disables local inference")
    args = parser.parse_args()
    args.trust_proxy = False
    if args.config:
        try:
            apply_rendered_config(args, load_rendered_config(args.config))
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            parser.error(f"--config 无效: {exc}")
    if min(args.max_new_tokens, args.eval_batch_size, args.threads) < 1:
        parser.error("Token, batch and thread counts must be positive")
    if args.dtype == "auto":
        import torch
        args.dtype = "bfloat16" if torch.cuda.is_available() else "float32"
    if not args.model.expanduser().is_dir():
        # HuggingFace 仓库 ID：允许执行时下载；本地目录仍保持模块级的离线默认。
        for var in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
            os.environ.pop(var, None)
    if args.upstream:
        upstream = urlsplit(args.upstream)
        if (upstream.scheme != "http" or upstream.hostname not in ("127.0.0.1", "localhost")
                or not upstream.port or upstream.path not in ("", "/") or upstream.query
                or upstream.fragment or upstream.username or upstream.password):
            parser.error("--upstream must be a loopback HTTP origin with a port")
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    server.daemon_threads = True
    server.trust_proxy = args.trust_proxy
    server.upstream = args.upstream.rstrip("/") if args.upstream else None
    server.runtime = None if server.upstream else Runtime(
        args.model, device=args.device, dtype=args.dtype, model_name=args.model_name,
        location=args.location, max_new_tokens=args.max_new_tokens,
        eval_batch_size=args.eval_batch_size, threads=args.threads)
    print(f"Toolkit Lab: http://127.0.0.1:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.server_close()


if __name__ == "__main__":
    main()
