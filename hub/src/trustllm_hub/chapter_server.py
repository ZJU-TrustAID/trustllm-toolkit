"""Lifecycle manager for chapter frontend servers.

Each chapter ships its own web frontend (guardrail's FastAPI app, the
hallucination offline demo, ...). The hub launches that server on the port
declared by the manifest, waits for its readiness endpoint, and embeds it in
the chapter workspace page via a direct iframe. Servers already listening on
the port (for example started manually via the chapter's own README) are
reused, never killed.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import threading
import time
from typing import Any
from urllib.request import ProxyHandler, build_opener

from .child_env import chapter_env
from .config import HubConfig
from .devices import resolve_auto_device
from .registry import ChapterManifest, ManifestError


class ChapterServerError(RuntimeError):
    pass


class ChapterServerManager:
    def __init__(self, config: HubConfig) -> None:
        self._config = config
        self._lock = threading.Lock()
        self._procs: dict[str, subprocess.Popen] = {}
        self._logs: dict[str, Path] = {}

    # -- health ---------------------------------------------------------------
    @staticmethod
    def _ready(url: str, timeout: float = 1.5) -> bool:
        opener = build_opener(ProxyHandler({}))
        try:
            with opener.open(url, timeout=timeout) as response:
                return response.status == 200
        except OSError:
            return False

    def probe(self, manifest: ChapterManifest, port: int | None = None) -> bool:
        server = manifest.frontend_server
        if not server:
            return False
        port = port or self._config.frontend_port(manifest.toolkit, int(server["port"]))
        return self._ready(f"http://127.0.0.1:{port}{server.get('ready_path') or '/api/health'}")

    def status(self, manifest: ChapterManifest) -> dict[str, Any]:
        server = manifest.frontend_server
        if not server:
            return {"supported": False, "state": "unsupported"}
        port = self._config.frontend_port(manifest.toolkit, int(server["port"]))
        ready = self.probe(manifest, port)
        owned = manifest.toolkit in self._procs
        state = "ready" if ready else "starting" if owned else "stopped"
        log_path = self._logs.get(manifest.toolkit)
        return {
            "supported": True,
            "state": state,
            "port": port,
            "url": manifest.frontend_url(port),
            "owned": owned,
            "log": str(log_path) if owned else None,
            # 启动中的进度反馈：日志最后两行（如 vLLM 权重加载百分比）
            "log_tail": self._log_tail(log_path, lines=2) if owned and log_path else "",
        }

    # -- control ---------------------------------------------------------------
    def start(
        self,
        manifest: ChapterManifest,
        *,
        params: dict[str, Any] | None = None,
        wait: bool = True,
    ) -> dict[str, Any]:
        server = manifest.frontend_server
        if not server:
            raise ChapterServerError(f"{manifest.toolkit} 未声明前端服务")
        port = self._config.frontend_port(manifest.toolkit, int(server["port"]))
        current = self.status(manifest)
        if current["state"] == "ready":
            current["started_now"] = False
            return current

        config_path: str | None = None
        if server.get("config") is not None:
            config_path = str(self._render_server_config(manifest, server, port, params or {}))

        command = manifest.fill_command(
            [str(part) for part in server["command"]],
            conda_bin=self._config.conda_bin,
            python_bin=_python_bin(),
            port=port,
            config_path=config_path,
        )
        env = chapter_env()
        for key, value in (server.get("env") or {}).items():
            env[str(key)] = manifest.fill_command(
                [str(value)], conda_bin=self._config.conda_bin, python_bin=_python_bin(), port=port, config_path=config_path
            )[0]

        log_dir = self._config.workspace / "frontend_servers"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"{manifest.toolkit}.log"
        log_handle = log_path.open("a", encoding="utf-8")
        log_handle.write(f"\n--- {datetime.now():%Y-%m-%d %H:%M:%S} hub 启动 {manifest.toolkit} 前端 (端口 {port}) ---\n")
        log_handle.flush()
        try:
            process = subprocess.Popen(
                command,
                cwd=str(manifest.directory),
                env=env,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        except OSError as exc:
            log_handle.close()
            raise ChapterServerError(
                f"启动 {manifest.toolkit} 前端服务失败: {exc}；请确认 manifest 命令与 conda 环境可用"
            ) from exc
        with self._lock:
            self._procs[manifest.toolkit] = process
            self._logs[manifest.toolkit] = log_path

        if wait:
            try:
                self._wait_ready(manifest, port, process, log_handle)
            except ChapterServerError:
                self.stop(manifest.toolkit)
                raise
        log_handle.close()
        status = self.status(manifest)
        status["started_now"] = True
        return status

    @staticmethod
    def _log_tail(log_path: Path | str, lines: int = 4) -> str:
        """Last few non-empty log lines, for surfacing startup failures in the UI."""
        try:
            content = Path(log_path).read_text(encoding="utf-8", errors="replace").splitlines()
        except (OSError, ValueError):
            return ""
        tail = [line.strip() for line in content if line.strip()][-lines:]
        return " | ".join(tail)[:400]

    def _render_server_config(
        self,
        manifest: ChapterManifest,
        server: dict[str, Any],
        port: int,
        params: dict[str, Any],
    ) -> Path:
        template = server["config"]
        flat: dict[str, Any] = {"port": port}
        flat.update(_flatten(params))
        # 交互服务与一次性运行共用同一套设备语义：auto 在 hub 侧解析成具体
        # cuda:N（空闲显存最大的卡），各章进程才不会各自默认挤到 cuda:0。
        device = str(flat.get("common.device") or "")
        if device == "auto":
            resolved = resolve_auto_device()
            if resolved:
                flat["common.device"] = resolved
        for role_spec in manifest.models:
            role = role_spec["role"]
            required = role_spec.get("required", True)
            entry = params.get("models", {}).get(role) or {}
            if _has_model_string(entry):
                flat[f"models.{role}.entry"] = _entry_string(entry, self._config.root)
            elif required:
                if role in (params.get("models") or {}):
                    raise ManifestError(f"角色 {role} 的模型配置不完整，无法启动前端服务")
                # params 里缺这个角色：不填 flat，让模板占位符报"缺少对应值"。
            else:
                # 可选角色（如幻觉章的 NLI/句子向量）未填：显式给空串，
                # 消费方（web_demo 等）回退到内置默认。
                flat[f"models.{role}.entry"] = ""
        rendered = manifest.render_config_template(
            template,
            port=port,
            port_key=server.get("config_port_key"),
            params=flat,
        )
        out_dir = self._config.workspace / "frontend_servers"
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{manifest.toolkit}-server-config.json"
        path.write_text(json.dumps(rendered, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return path

    def _wait_ready(
        self,
        manifest: ChapterManifest,
        port: int,
        process: subprocess.Popen,
        log_handle: Any,
    ) -> None:
        server = manifest.frontend_server or {}
        timeout = int(server.get("timeout") or self._config.frontend_server_timeout)
        url = f"http://127.0.0.1:{port}{server.get('ready_path') or '/api/health'}"
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            code = process.poll()
            if code is not None:
                tail = self._log_tail(log_handle.name if hasattr(log_handle, "name") else "")
                detail = f"，日志尾部：{tail}" if tail else ""
                raise ChapterServerError(
                    f"{manifest.toolkit} 前端服务已退出（状态码 {code}）{detail}；"
                    f"完整日志见 workspace/frontend_servers/{manifest.toolkit}.log"
                )
            if self._ready(url, timeout=1.0):
                return
            time.sleep(0.5)
        raise ChapterServerError(
            f"等待 {manifest.toolkit} 前端就绪超时（{timeout}s），首次启动可能需要下载模型权重；请查看日志后重试"
        )

    def stop(self, toolkit: str) -> bool:
        with self._lock:
            process = self._procs.pop(toolkit, None)
        if process is None:
            return False
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            pass
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
            process.wait()
        return True

    def stop_all(self) -> None:
        with self._lock:
            toolkits = list(self._procs)
        for toolkit in toolkits:
            self.stop(toolkit)


def _python_bin() -> str:
    import shutil

    return shutil.which("python3") or shutil.which("python") or "python3"


def _entry_string(entry: dict[str, Any], root: Path) -> str:
    if entry.get("source") == "local_path":
        # 展开为绝对路径：相对路径（如 models/xxx）传给章节进程后可能被当作
        # HuggingFace 仓库名而非本地目录（guardrail 对非 ./ 开头的字符串即视为 repo id）。
        raw = str(entry.get("path") or "")
        expanded = Path(raw).expanduser()
        if not expanded.is_absolute():
            expanded = root / expanded
        return str(expanded.resolve())
    if entry.get("source") == "hf":
        return str(entry.get("repo"))
    raise ManifestError("前端交互服务不支持 API 模型源，请填写本地路径或 HuggingFace 仓库 ID")


def _has_model_string(entry: dict[str, Any]) -> bool:
    """该模型条目是否能解析出一个可用的模型字符串（路径/仓库名/API 模型名）。"""
    if not isinstance(entry, dict) or not entry.get("source"):
        return False
    source = entry.get("source")
    if source == "local_path":
        return bool(str(entry.get("path") or "").strip())
    if source == "hf":
        return bool(str(entry.get("repo") or "").strip())
    if source == "api":
        api = entry.get("api") or {}
        return bool(str(api.get("model") or "").strip())
    return False


def _flatten(data: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    """Flatten nested mappings into dotted keys for template placeholders."""
    flat: dict[str, Any] = {}
    for key, value in data.items():
        path = f"{prefix}{key}"
        if isinstance(value, dict):
            flat.update(_flatten(value, prefix=path + "."))
        else:
            flat[path] = value
    return flat


__all__ = ["ChapterServerManager", "ChapterServerError"]
