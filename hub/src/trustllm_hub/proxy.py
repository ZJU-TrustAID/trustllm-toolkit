"""Same-origin reverse proxy for chapter frontend servers.

All chapter frontends are reached through ``/proxy/{toolkit}/...`` so the hub
occupies a single browser-facing port; the chapter servers keep listening on
loopback ports that never reach the user. The proxy:

- forwards method, query, headers and body to ``127.0.0.1:{port}``;
- rewrites ``frame-ancestors 'none'`` to ``'self'`` in CSP headers (the embedded
  page is same-origin with the hub, so no host-specific exception is needed);
- returns a friendly 503 page when the chapter server is not running yet.
"""
from __future__ import annotations

from typing import Any

import httpx
from fastapi import HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse

_HOP_BY_HOP = {
    "connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
    "te", "trailers", "transfer-encoding", "upgrade",
}
_MAX_BODY_BYTES = 64 * 1024 * 1024


class ProxyGateway:
    """Forwards requests to chapter frontend servers on loopback ports."""

    def __init__(self) -> None:
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(120.0))

    async def aclose(self) -> None:
        await self._client.aclose()

    async def forward(self, request: Request, port: int, path: str) -> Response:
        target_path = "/" + path
        url = httpx.URL(f"http://127.0.0.1:{port}{target_path}")
        if request.url.query:
            url = url.copy_merge_params(request.query_params)
        headers = {
            name: value
            for name, value in request.headers.items()
            if name.lower() not in _HOP_BY_HOP and name.lower() != "host"
        }
        body = await request.body()
        try:
            upstream = await self._client.send(
                self._client.build_request(request.method, url, headers=headers, content=body),
                stream=True, follow_redirects=False,
            )
        except httpx.TransportError:
            return _unavailable_page(path)

        response_headers = {
            name: value
            for name, value in upstream.headers.items()
            if name.lower() not in _HOP_BY_HOP and name.lower() != "content-length"
        }
        if "content-security-policy" in response_headers:
            response_headers["content-security-policy"] = response_headers[
                "content-security-policy"
            ].replace("frame-ancestors 'none'", "frame-ancestors 'self'")

        async def _relay():
            try:
                async for chunk in upstream.aiter_raw():
                    yield chunk
            finally:
                await upstream.aclose()

        return StreamingResponse(
            _relay(),
            status_code=upstream.status_code,
            headers=response_headers,
            media_type=upstream.headers.get("content-type"),
        )


def _unavailable_page(path: str) -> Response:
    if path.startswith("api/") or path == "api":
        return JSONResponse(
            status_code=503,
            content={"error": "chapter-server-not-running", "message": "该章交互服务尚未启动"},
        )
    html = """<!DOCTYPE html><html lang="zh-CN"><head><meta charset="utf-8">
<style>body{font:14px/1.7 -apple-system,"PingFang SC","Microsoft YaHei",sans-serif;
display:flex;align-items:center;justify-content:center;height:100vh;margin:0;color:#67718a;background:#fafbfc}
.box{text-align:center}.box h2{color:#1c2330;font-size:16px}</style></head>
<body><div class="box"><h2>该章交互服务尚未启动</h2>
<p>请在右侧抽屉上方的「启动交互服务」按钮处启动该章自带前端。</p></div></body></html>"""
    return HTMLResponse(html, status_code=503)


def proxy_unavailable_data() -> dict[str, Any]:
    return {"error": "chapter-server-not-running"}


__all__ = ["ProxyGateway"]
