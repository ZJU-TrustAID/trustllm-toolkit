# ruff: noqa: E402
# Package imports below run after the sys.path setup so the demo works without installation.
from __future__ import annotations

import json
import os
import sys
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
STATIC = Path(__file__).resolve().parent / "static"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import catalog as tutorial_catalog
import presets_detection
import presets_mitigation
from real_backend import REAL


def handle_method(payload: dict[str, Any]) -> dict[str, Any]:
    """Run one tutorial method on a preset scenario (no model calls)."""

    if not isinstance(payload, dict):
        raise ValueError("请求内容必须是 JSON 对象")
    method = str(payload.get("method", "")).strip()
    scenario = str(payload.get("scenario", "")).strip()
    if not method:
        raise ValueError("请指定方法")
    tutorial_catalog.resolve(method, scenario)
    if method in presets_mitigation.BUILDERS:
        return presets_mitigation.run_mitigation_method(method, scenario)
    return presets_detection.run_detection_method(method, scenario)


class DemoHandler(BaseHTTPRequestHandler):
    server_version = "HallucinationDemo/0.2"

    def _json(self, status: int, body: dict[str, Any]) -> None:
        encoded = json.dumps(_sanitize_json(body), ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(encoded)

    def do_OPTIONS(self) -> None:  # noqa: N802
        self.send_response(HTTPStatus.NO_CONTENT)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802
        if self.path.split("?", 1)[0] == "/api/real/status":
            from real_backend import REAL

            self._json(HTTPStatus.OK, REAL.status())
            return
        if self.path == "/api/health":
            self._json(HTTPStatus.OK, {"ok": True, "service": "hallucination-demo"})
            return
        if self.path == "/api/catalog":
            self._json(HTTPStatus.OK, tutorial_catalog.catalog_payload())
            return
        requested = self.path.split("?", 1)[0]
        relative = "index.html" if requested in {"", "/"} else requested.removeprefix("/")
        file_path = (STATIC / relative).resolve()
        if STATIC.resolve() not in file_path.parents or not file_path.is_file():
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        content_type = "text/html; charset=utf-8" if file_path.suffix == ".html" else "text/css; charset=utf-8" if file_path.suffix == ".css" else "application/javascript; charset=utf-8"
        data = file_path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self) -> None:  # noqa: N802
        if self.path == "/api/real/generate_stream":
            # 流式生成：NDJSON 逐行推送（HTTP/1.0 连接关闭即结束，无需 chunked）。
            try:
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length).decode("utf-8")) if length else {}
                prompt = str(payload.get("prompt") or "").strip()
                if not prompt:
                    raise ValueError("提示词不能为空")
            except Exception as exc:
                self._json(HTTPStatus.BAD_REQUEST, {"error": f"{type(exc).__name__}: {exc}"})
                return
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/x-ndjson; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            try:
                for chunk in REAL.generate_stream(
                    prompt,
                    max_new_tokens=int(payload.get("max_new_tokens") or 128),
                    do_sample=bool(payload.get("do_sample", False)),
                    temperature=float(payload.get("temperature") or 1.0),
                    top_p=float(payload.get("top_p") or 1.0),
                ):
                    line = json.dumps(_sanitize_json(chunk), ensure_ascii=False)
                    self.wfile.write(line.encode("utf-8") + b"\n")
                    self.wfile.flush()
            except Exception as exc:
                line = json.dumps(_sanitize_json({"error": f"{type(exc).__name__}: {exc}"}), ensure_ascii=False)
                self.wfile.write(line.encode("utf-8") + b"\n")
                self.wfile.flush()
            return
        if self.path.startswith("/api/real/"):
            try:
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length).decode("utf-8")) if length else {}
                body = handle_real(self.path.removeprefix("/api/real/"), payload)
            except Exception as exc:  # Real inference errors stay actionable in the UI.
                self._json(HTTPStatus.BAD_REQUEST, {"error": f"{type(exc).__name__}: {exc}"})
                return
            self._json(HTTPStatus.OK, body)
            return
        if self.path != "/api/method":
            self._json(HTTPStatus.NOT_FOUND, {"error": "Not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            body = handle_method(payload)
        except Exception as exc:  # Return actionable errors to the demo UI.
            self._json(HTTPStatus.BAD_REQUEST, {"error": f"{type(exc).__name__}: {exc}"})
            return
        self._json(HTTPStatus.OK, body)

    def log_message(self, format: str, *args: object) -> None:
        print(f"[web-demo] {self.address_string()} - {format % args}")


def _sanitize_json(value: Any) -> Any:
    """NaN/Infinity 不是合法 JSON（浏览器 JSON.parse 会失败），统一换成 None。"""
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            return None
        return value
    if isinstance(value, list):
        return [_sanitize_json(item) for item in value]
    if isinstance(value, tuple):
        return [_sanitize_json(item) for item in value]
    if isinstance(value, dict):
        return {key: _sanitize_json(item) for key, item in value.items()}
    return value


def handle_real(action: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Real-model endpoints: /api/real/{status,load,generate,sample,detect,mitigate,config}."""
    if action == "status":
        return REAL.status()
    if action == "config":
        REAL.aux.set_sources(
            nli_source=payload.get("nli_source"),
            embedding_source=payload.get("embedding_source"),
            nli_device=payload.get("nli_device"),
            embedding_device=payload.get("embedding_device"),
        )
        return REAL.status()
    if action == "load":
        model = str(payload.get("model") or REAL.pending_model or "")
        if not model:
            raise ValueError("缺少模型：请在工作台选择模型后启动服务，或在请求里填写模型路径/仓库名")
        return REAL.load(
            model,
            device=payload.get("device"),
            dtype=str(payload.get("dtype") or "auto"),
            force=bool(payload.get("force", False)),
        )
    if action == "generate":
        prompt = str(payload.get("prompt") or "").strip()
        if not prompt:
            raise ValueError("提示词不能为空")
        return REAL.generate(
            prompt,
            max_new_tokens=int(payload.get("max_new_tokens") or 128),
            do_sample=bool(payload.get("do_sample", False)),
            temperature=float(payload.get("temperature") or 1.0),
            top_p=float(payload.get("top_p") or 1.0),
        )
    if action == "sample":
        prompt = str(payload.get("prompt") or "").strip()
        if not prompt:
            raise ValueError("提示词不能为空")
        mode = str(payload.get("mode") or "text")
        if mode == "hidden":
            return REAL.sample_hidden(
                prompt,
                n_samples=int(payload.get("n") or 5),
                hidden_state_layer=int(payload.get("layer", -1)),
                max_new_tokens=int(payload.get("max_new_tokens") or 32),
            )
        return {"texts": REAL.sample_texts(
            prompt, int(payload.get("n") or 5),
            max_new_tokens=int(payload.get("max_new_tokens") or 64),
        )}
    if action == "generate_stream":
        prompt = str(payload.get("prompt") or "").strip()
        if not prompt:
            raise ValueError("提示词不能为空")
        return REAL.generate_stream(
            prompt,
            max_new_tokens=int(payload.get("max_new_tokens") or 128),
            do_sample=bool(payload.get("do_sample", False)),
            temperature=float(payload.get("temperature") or 1.0),
            top_p=float(payload.get("top_p") or 1.0),
        )
    if action == "detect":
        task = str(payload.get("task") or "whitebox")
        method = str(payload.get("method") or "")
        prompt = str(payload.get("prompt") or "").strip()
        if not prompt:
            raise ValueError("提示词不能为空")
        params = dict(payload.get("params") or {})
        if task == "whitebox":
            sample_key = str(params.pop("sample_key", None) or payload.get("sample_key") or "")
            return REAL.detect_whitebox(
                prompt, method,
                max_new_tokens=int(params.pop("max_new_tokens", 128) or 128),
                n_samples=int(params.pop("n_samples", 5) or 5),
                hidden_state_layer=int(params.pop("hidden_state_layer", -1) or -1),
                sample_key=sample_key or None,
                **params,
            )
        return REAL.detect_blackbox(
            prompt, method,
            response=payload.get("response") or None,
            samples=payload.get("samples") or None,
            num_samples=int(params.pop("num_samples", 5) or 5),
            **params,
        )
    if action == "mitigate":
        prompt = str(payload.get("prompt") or "").strip()
        if not prompt:
            raise ValueError("提示词不能为空")
        params = dict(payload.get("params") or {})
        probe = str(params.pop("probe", "") or "")
        max_new_tokens = int(params.pop("max_new_tokens", 128) or 128)
        return REAL.mitigate(
            str(payload.get("method") or ""), prompt,
            context=payload.get("context") or None,
            max_new_tokens=max_new_tokens, probe=probe or None,
            method_args=params,
        )
    raise ValueError(f"未知 real 接口: {action}")


def main() -> None:
    host = os.environ.get("HALLUCINATION_DEMO_HOST", "127.0.0.1")
    port = int(os.environ.get("HALLUCINATION_DEMO_PORT", "8765"))
    _apply_hub_config()
    server = ThreadingHTTPServer((host, port), DemoHandler)
    print(f"Hallucination demo running at http://{host}:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping demo")
    finally:
        server.server_close()


def _apply_hub_config() -> None:
    """Pre-configure model/device/aux sources from the hub-rendered config file."""
    config_path = os.environ.get("HALLUCINATION_DEMO_CONFIG")
    if not config_path:
        return
    try:
        with open(config_path, encoding="utf-8") as handle:
            config = json.load(handle)
        from real_backend import REAL

        REAL.aux.set_sources(
            nli_source=config.get("nli_source"),
            embedding_source=config.get("embedding_source"),
        )
        model = config.get("model")
        if model:
            print(f"[web-demo] hub 配置: 模型={model} 设备={config.get('device') or 'auto'}（首次真实调用时懒加载）")
            REAL._pending_model = model
            REAL._pending_device = config.get("device")
            REAL._pending_dtype = config.get("dtype") or "auto"
    except (OSError, ValueError) as exc:
        print(f"[web-demo] 读取 hub 配置失败（忽略，使用默认）: {exc}")


if __name__ == "__main__":
    main()
