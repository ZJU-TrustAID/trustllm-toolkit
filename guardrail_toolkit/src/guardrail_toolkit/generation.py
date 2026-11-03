"""Local Qwen generation over the loopback vLLM service."""
from __future__ import annotations

import json
import time
from urllib.error import HTTPError, URLError
from urllib.request import ProxyHandler, Request, build_opener

DEFAULT_QWEN_URL = "http://127.0.0.1:8767/v1"
DEFAULT_QWEN_MODEL = "qwen3.5-9b"


class LocalQwen:
    def __init__(self, base_url=DEFAULT_QWEN_URL, model=DEFAULT_QWEN_MODEL, timeout=180):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        # Internal model traffic must not inherit an outbound HTTP proxy.
        self._opener = build_opener(ProxyHandler({}))

    def metadata(self):
        state = "unavailable"
        try:
            with self._opener.open(self.base_url + "/models", timeout=2) as response:
                data = json.load(response)
            if any(item.get("id") == self.model for item in data.get("data", [])):
                state = "ready"
        except (OSError, ValueError):
            pass
        return {"model": self.model, "model_name": "Qwen3.5-9B", "state": state}

    def generate(self, text, *, max_new_tokens=512, seed=None):
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Qwen 输入不能为空")
        if type(max_new_tokens) is not int or not 32 <= max_new_tokens <= 2048:
            raise ValueError("生成长度必须为 32–2048 之间的整数")
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": text}],
            "max_tokens": max_new_tokens,
            "temperature": 0,
            "stream": False,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        if seed is not None:
            payload["seed"] = int(seed)
        request = Request(self.base_url + "/chat/completions", data=json.dumps(payload).encode(),
                          headers={"Content-Type": "application/json"})
        started = time.perf_counter()
        try:
            with self._opener.open(request, timeout=self.timeout) as response:
                data = json.load(response)
        except HTTPError as exc:
            message = "Qwen 请求被拒绝，请检查输入长度与生成设置" if exc.code == 400 else "Qwen 服务推理失败，请检查模型服务日志"
            raise RuntimeError(message) from exc
        except (URLError, TimeoutError, OSError) as exc:
            raise RuntimeError("Qwen 服务尚未就绪或请求超时，请稍后重试") from exc
        try:
            choice = data["choices"][0]
            content = choice["message"]["content"]
            finish_reason = choice["finish_reason"]
            if not isinstance(content, str) or not content.strip():
                raise ValueError("empty answer")
            if finish_reason not in ("stop", "length"):
                raise ValueError("unexpected finish reason")
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise RuntimeError("Qwen 未返回有效的文本回答，未执行输出审计") from exc
        return {
            "model": self.model,
            "response": content,
            "finish_reason": finish_reason,
            "truncated": finish_reason == "length",
            "usage": data.get("usage", {}),
            "latency_ms": round((time.perf_counter() - started) * 1000, 1),
        }
