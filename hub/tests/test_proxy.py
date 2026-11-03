"""Same-origin reverse proxy: /proxy/{toolkit}/... → chapter frontend servers."""
from __future__ import annotations

import subprocess
import sys
import time as time_module
from urllib.request import ProxyHandler, build_opener

import pytest
from fastapi.testclient import TestClient

from trustllm_hub.app import create_app

from conftest import make_config, make_chapter_dir

# A minimal static chapter frontend used as the proxy upstream.
FRONTEND_MANIFEST = """\
chapter: 98
title: 代理测试章节
toolkit: proxykit
enabled: true
frontend:
  mode: proxy
  server:
    command: ["{python}", "frontend_server.py", "{port}"]
    port: 8971
    ready_path: /api/health
  entry: /
backend:
  command: ["{python}", "-c", "pass"]
models:
  - role: default
    accepts: [local_path, hf]
    default: {source: hf, repo: "org/model"}
"""

FRONTEND_SERVER = '''\
"""Tiny static+JSON server standing in for a chapter frontend."""
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PORT = int(sys.argv[1])

class Handler(BaseHTTPRequestHandler):
    def _send(self, status, body, content_type, headers=None):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        for key, value in (headers or {}).items():
            self.send_header(key, value)
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/api/health":
            self._send(200, b'{"ok": true}', "application/json")
        elif self.path == "/api/echo":
            self._send(200, json.dumps({"echo": "x"}).encode(), "application/json")
        elif self.path == "/":
            self._send(200, b"<html><body>frontend-home</body></html>", "text/html; charset=utf-8",
                       {"Content-Security-Policy": "default-src 'self'; frame-ancestors 'none'"})
        else:
            self._send(404, b"missing", "text/plain")

    def log_message(self, *args):
        pass

ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
'''


@pytest.fixture
def proxy_env(tmp_path):
    directory = make_chapter_dir(tmp_path)  # fakekit (no frontend server)
    proxy_dir = tmp_path / "chapters" / "proxykit"
    proxy_dir.mkdir(parents=True)
    (proxy_dir / "frontend_server.py").write_text(FRONTEND_SERVER, encoding="utf-8")
    (proxy_dir / "chapter.yaml").write_text(FRONTEND_MANIFEST, encoding="utf-8")
    # The chapter.yaml placeholder scan only accepts known placeholders; {port} is one.
    config = make_config(tmp_path)
    config.workspace.mkdir(parents=True, exist_ok=True)
    return config


def test_proxy_forwards_to_running_server(proxy_env):
    process = subprocess.Popen(
        [sys.executable, str(proxy_env.root / "chapters" / "proxykit" / "frontend_server.py"), "8971"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    try:
        opener = build_opener(ProxyHandler({}))
        deadline = time_module.monotonic() + 10
        while time_module.monotonic() < deadline:
            try:
                with opener.open("http://127.0.0.1:8971/api/health", timeout=1):
                    break
            except OSError:
                time_module.sleep(0.1)
        client = TestClient(create_app(proxy_env))
        page = client.get("/proxy/proxykit/")
        assert page.status_code == 200
        assert "frontend-home" in page.text
        # CSP frame-ancestors must be rewritten for same-origin embedding.
        assert "frame-ancestors 'self'" in page.headers["content-security-policy"]
        assert "frame-ancestors 'none'" not in page.headers["content-security-policy"]
        api = client.get("/proxy/proxykit/api/echo")
        assert api.status_code == 200 and api.json() == {"echo": "x"}
    finally:
        process.terminate()
        process.wait()


def test_proxy_returns_friendly_503_when_stopped(proxy_env):
    client = TestClient(create_app(proxy_env))
    page = client.get("/proxy/proxykit/")
    assert page.status_code == 503
    assert "尚未启动" in page.text
    api = client.get("/proxy/proxykit/api/echo")
    assert api.status_code == 503


def test_proxy_unknown_toolkit_404(proxy_env):
    client = TestClient(create_app(proxy_env))
    assert client.get("/proxy/ghost/").status_code == 404


def test_chapter_without_server_404(proxy_env):
    client = TestClient(create_app(proxy_env))
    assert client.get("/proxy/fakekit/").status_code == 404


def test_proxy_path_requires_trailing_slash_redirect(proxy_env):
    client = TestClient(create_app(proxy_env), follow_redirects=False)
    response = client.get("/proxy/proxykit")
    assert response.status_code == 307
    assert response.headers["location"] == "/proxy/proxykit/"
