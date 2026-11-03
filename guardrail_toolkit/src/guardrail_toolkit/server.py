"""Single-process local HTTP server; one resident model, no web dependencies."""
from __future__ import annotations

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import logging
import os
from pathlib import Path
from threading import Lock
from urllib.parse import urlsplit

from .generation import DEFAULT_QWEN_MODEL, DEFAULT_QWEN_URL, LocalQwen
from .model import DEFAULT_MODEL, XGuard
from .pipeline import GuardedChat, PipelineError

LOG = logging.getLogger(__name__)
STATIC = Path(__file__).parent / "static"
MAX_BODY_BYTES = 120_000


def validate_request(data: object) -> dict:
    if not isinstance(data, dict):
        raise ValueError("请求必须是 JSON 对象")
    allowed = {"stage", "text", "response", "explain", "max_new_tokens", "policy"}
    if set(data) - allowed:
        raise ValueError("不支持的请求字段：" + ", ".join(sorted(set(data) - allowed)))
    stage = data.get("stage", "input")
    if stage not in ("input", "output"):
        raise ValueError("stage 必须为 input 或 output")
    text = data.get("text")
    response = data.get("response")
    policy = data.get("policy")
    if not isinstance(text, str) or not text.strip() or len(text) > 16000:
        raise ValueError("text 必须为 1–16000 字符的非空文本")
    if response is not None and (not isinstance(response, str) or len(response) > 16000):
        raise ValueError("response 必须为不超过 16000 字符的文本")
    if stage == "output" and (not response or not response.strip()):
        raise ValueError("输出检查需要待检测的助手回答 response")
    if stage == "input" and response is not None:
        raise ValueError("输入检查请勿传入 response；检查回答请使用 stage=output")
    if policy is not None and (not isinstance(policy, str) or len(policy) > 4000):
        raise ValueError("policy 必须为不超过 4000 字符的文本")
    if len(text) + len(response or "") + len(policy or "") > 24000:
        raise ValueError("输入总长度不能超过 24000 个字符")
    explain = data.get("explain", True)
    if not isinstance(explain, bool):
        raise ValueError("explain 必须为布尔值")
    budget = data.get("max_new_tokens", 384)
    if type(budget) is not int or not 32 <= budget <= 1024:
        raise ValueError("max_new_tokens 必须为 32–1024 之间的整数")
    return {"stage": stage, "text": text, "response": response, "policy": policy,
            "explain": explain, "max_new_tokens": budget}


def validate_chat_request(data: object) -> dict:
    if not isinstance(data, dict):
        raise ValueError("请求必须是 JSON 对象")
    allowed = {"text", "explain", "max_new_tokens", "policy", "enforce_input", "generation_max_tokens"}
    if set(data) - allowed:
        raise ValueError("不支持的生成审计请求字段")
    common = validate_request({key: value for key, value in data.items()
                               if key not in {"enforce_input", "generation_max_tokens"}})
    common.pop("stage")
    common.pop("response")
    enforce_input = data.get("enforce_input", True)
    budget = data.get("generation_max_tokens", 512)
    if not isinstance(enforce_input, bool):
        raise ValueError("enforce_input 必须为布尔值")
    if type(budget) is not int or not 32 <= budget <= 2048:
        raise ValueError("generation_max_tokens 必须为 32–2048 之间的整数")
    return {**common, "enforce_input": enforce_input, "generation_max_tokens": budget}


def make_server(guard: XGuard, host="127.0.0.1", port=8766, *, generator=None, trust_proxy=False) -> ThreadingHTTPServer:
    chat = GuardedChat(guard, generator) if generator is not None else None
    request_lock = Lock()

    class Handler(BaseHTTPRequestHandler):
        def setup(self):
            super().setup()
            self.connection.settimeout(30)

        def log_message(self, format, *args):
            # Do not log submitted text or URL query contents.
            LOG.info("%s %s", self.command, urlsplit(self.path).path)

        def send_bytes(self, status, body, content_type):
            # The hub embeds this frontend in an iframe; standalone use keeps 'none'.
            frame_ancestors = os.environ.get("GUARDRAIL_FRAME_ANCESTORS", "'none'")
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("Content-Security-Policy", f"default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; connect-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors {frame_ancestors}")
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def send_json(self, status, payload):
            self.send_bytes(status, json.dumps(payload, ensure_ascii=False, allow_nan=False).encode(),
                            "application/json; charset=utf-8")

        def do_GET(self):
            path = urlsplit(self.path).path
            if path == "/api/info":
                self.send_json(200, {**guard.metadata(), "generator": generator.metadata() if generator else None})
                return
            if path == "/api/health":
                self.send_json(200, {"status": "ok", "model_state": guard.state})
                return
            files = {
                "/": ("index.html", "text/html; charset=utf-8"),
                "/static/styles.css": ("styles.css", "text/css; charset=utf-8"),
                "/static/app.js": ("app.js", "text/javascript; charset=utf-8"),
                "/docs": ("api.html", "text/html; charset=utf-8"),
            }
            if path not in files:
                self.send_json(404, {"detail": "未找到该资源"})
                return
            filename, content_type = files[path]
            self.send_bytes(200, (STATIC / filename).read_bytes(), content_type)

        def do_POST(self):
            path = urlsplit(self.path).path
            if path not in ("/api/check", "/api/chat"):
                self.send_json(404, {"detail": "未找到该接口"})
                return
            # Reject cross-site requests to this local model service. 经 hub 同源
            # 反代访问时 Origin 属于 hub 端口而 Host 是回环端口，无法直接比对，
            # trust_proxy 放宽该校验（服务只监听回环地址，风险可控）。
            origin = self.headers.get("Origin")
            if origin and not self.server.trust_proxy and urlsplit(origin).netloc != self.headers.get("Host"):
                self.send_json(403, {"detail": "不接受跨站请求"})
                return
            if self.headers.get_content_type() != "application/json":
                self.send_json(415, {"detail": "请使用 application/json"})
                return
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= MAX_BODY_BYTES:
                    self.send_json(413, {"detail": "请求体为空或超过大小限制"})
                    return
                raw = self.rfile.read(size)
                if len(raw) != size:
                    raise ValueError("请求体不完整")
                payload = validate_chat_request(json.loads(raw)) if path == "/api/chat" else validate_request(json.loads(raw))
                if path == "/api/chat" and chat is None:
                    self.send_json(503, {"detail": "尚未配置本地生成模型"})
                    return
                if not request_lock.acquire(blocking=False):
                    raise BlockingIOError("正在处理另一条审计请求，请稍后重试")
                try:
                    if path == "/api/chat":
                        self.send_json(200, chat.run(**payload))
                    else:
                        stage = payload.pop("stage")
                        text = payload.pop("text")
                        response = payload.pop("response")
                        result = guard.check_input(text, **payload) if stage == "input" else guard.check_output(text, response, **payload)
                        self.send_json(200, result.to_dict())
                finally:
                    request_lock.release()
            except PipelineError as exc:
                self.send_json(503, {"detail": str(exc), "partial_result": exc.partial_result})
            except BlockingIOError as exc:
                self.send_json(409, {"detail": str(exc)})
            except (ValueError, UnicodeError) as exc:
                self.send_json(422, {"detail": str(exc)})
            except (TimeoutError, ConnectionError):
                self.send_json(408, {"detail": "读取请求超时，请重试"})
            except Exception:
                LOG.exception("XGuard inference failed")
                self.send_json(503, {"detail": "模型加载或推理失败。请查看服务终端，检查模型依赖、设备与可用显存。"})

    server = ThreadingHTTPServer((host, port), Handler)
    server.daemon_threads = True
    server.trust_proxy = bool(trust_proxy)
    return server


def main():
    parser = argparse.ArgumentParser(description="YuFeng-XGuard 本地护栏实验台")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Hugging Face 模型仓库 ID 或本地目录")
    parser.add_argument("--device", default="auto", help="auto、cpu 或 cuda:0 等设备")
    parser.add_argument("--dtype", default="auto", help="auto、float16、bfloat16 或 float32")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--max-input-tokens", type=int, default=8192)
    parser.add_argument("--qwen-url", default=DEFAULT_QWEN_URL, help="本地 Qwen 服务地址")
    parser.add_argument("--qwen-model", default=DEFAULT_QWEN_MODEL)
    parser.add_argument("--trust-proxy", action="store_true",
                        help="经 hub 同源反代访问时放宽 Origin/Host 校验（仅回环监听）")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    guard = XGuard(args.model, args.device, args.max_input_tokens, dtype=args.dtype)
    generator = LocalQwen(args.qwen_url, args.qwen_model)
    server = make_server(guard, args.host, args.port, generator=generator, trust_proxy=args.trust_proxy)
    LOG.info("XGuard UI: http://%s:%s (model loads on first check)", args.host, args.port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
