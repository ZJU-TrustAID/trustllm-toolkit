"""Subprocess execution of chapter adapters from a unified YAML.

Each run is one adapter subprocess, launched with the command declared by the
chapter manifest (typically ``conda run -n <env> python -m <pkg>.hub_adapter``),
so the hub process itself stays free of any model-stack dependency and chapter
environments (transformers/vllm pins) stay isolated from each other.
"""
from __future__ import annotations

from datetime import datetime, timezone
import os
from pathlib import Path
import shutil
import signal
import subprocess
import threading
import time
from typing import Any

from .child_env import chapter_env
from .config import HubConfig
from .devices import resolve_auto_device
from .registry import ChapterManifest, ManifestError
from .schema import dumps_yaml
from .sessions import SessionPaths, is_terminal, new_session_id, sessions_root


class RunnerError(RuntimeError):
    pass


class RunnerBusyError(RunnerError):
    pass


class RunHandle:
    def __init__(self, session: str, manifest: ChapterManifest, paths: SessionPaths, process: subprocess.Popen) -> None:
        self.session = session
        self.manifest = manifest
        self.paths = paths
        self.process = process
        self.state = "running"
        self.error: str | None = None
        self.exit_code: int | None = None
        self.started_at = datetime.now(timezone.utc).isoformat()
        self.ended_at: str | None = None

    def status(self) -> dict[str, Any]:
        config = self.paths.read_config() or {}
        view = {
            "session": self.session,
            "state": self.state,
            "toolkit": self.manifest.toolkit,
            "chapter": self.manifest.chapter,
            "title": self.manifest.title,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "exit_code": self.exit_code,
            "error": self.error,
            "has_result": self.paths.result_json.is_file(),
            "config": config,
        }
        persisted = self.paths.read_status() or {}
        if persisted.get("device_resolution"):
            view["device_resolution"] = persisted["device_resolution"]
        return view


class RunManager:
    def __init__(self, config: HubConfig) -> None:
        self._config = config
        self._lock = threading.Lock()
        self._runs: dict[str, RunHandle] = {}
        self._threads: dict[str, threading.Thread] = {}

    # -- lifecycle -----------------------------------------------------------
    @property
    def workspace(self) -> Path:
        return self._config.workspace

    def active_count(self) -> int:
        with self._lock:
            return sum(1 for run in self._runs.values() if run.state == "running")

    def start(self, manifest: ChapterManifest, config: dict[str, Any]) -> str:
        if self.active_count() >= self._config.max_concurrent_runs:
            raise RunnerBusyError(
                f"已有 {self.active_count()} 个任务在运行（上限 {self._config.max_concurrent_runs}），请等待完成后再执行"
            )
        resolved_device = resolve_config_device(config)
        session = new_session_id()
        paths = SessionPaths(self.workspace, session)
        paths.prepare()
        paths.config_yaml.write_text(dumps_yaml(config), encoding="utf-8")
        status: dict[str, Any] = {
            "session": session,
            "state": "running",
            "toolkit": manifest.toolkit,
            "chapter": manifest.chapter,
            "started_at": datetime.now(timezone.utc).isoformat(),
        }
        if resolved_device:
            status["device_resolution"] = {"requested": "auto", "resolved": resolved_device}
        paths.write_status(status)

        command = manifest.fill_command(
            manifest.backend_command,
            conda_bin=self._config.conda_bin,
            python_bin=sys_executable(),
            yaml_path=str(paths.config_yaml),
            output_dir=str(paths.dir),
        )
        env = chapter_env()
        try:
            log_handle = paths.run_log.open("w", encoding="utf-8")
        except OSError as exc:
            raise RunnerError(f"无法创建运行日志: {exc}") from exc
        try:
            process = subprocess.Popen(
                command,
                cwd=str(manifest.backend_cwd),
                env=env,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        except OSError as exc:
            log_handle.close()
            paths.write_status(
                {
                    "session": session,
                    "state": "failed",
                    "toolkit": manifest.toolkit,
                    "chapter": manifest.chapter,
                    "error": f"启动命令失败: {exc}",
                    "started_at": datetime.now(timezone.utc).isoformat(),
                    "ended_at": datetime.now(timezone.utc).isoformat(),
                }
            )
            raise RunnerError(f"启动命令失败: {exc}；请检查 manifest 中的 conda 环境") from exc

        handle = RunHandle(session, manifest, paths, process)
        with self._lock:
            self._runs[session] = handle
        thread = threading.Thread(
            target=self._reap, args=(handle, log_handle), name=f"hub-run-{session}", daemon=True
        )
        self._threads[session] = thread
        thread.start()
        return session

    def _reap(self, handle: RunHandle, log_handle: Any) -> None:
        timeout = handle.manifest.backend_timeout
        try:
            code = handle.process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            self._terminate(handle.process)
            code = handle.process.wait()
            handle.state = "timeout"
            handle.error = f"执行超过 manifest 限制的 {timeout} 秒，已终止"
        except Exception:  # pragma: no cover - defensive
            code = -1
        finally:
            log_handle.close()
        if handle.state == "running":
            handle.state = "succeeded" if code == 0 and handle.paths.result_json.is_file() else "failed"
            if code != 0:
                handle.error = f"适配器退出码 {code}，请查看运行日志"
            elif not handle.paths.result_json.is_file():
                handle.error = "适配器未写出 result.json"
        handle.exit_code = code
        handle.ended_at = datetime.now(timezone.utc).isoformat()
        handle.paths.write_status(handle.status())
        with self._lock:
            self._threads.pop(handle.session, None)
            self._runs.pop(handle.session, None)

    @staticmethod
    def _terminate(process: subprocess.Popen) -> None:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            pass
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
            process.wait()

    # -- queries / control ----------------------------------------------------
    def get(self, session: str) -> RunHandle:
        with self._lock:
            handle = self._runs.get(session)
        if handle is not None:
            return handle
        paths = SessionPaths(self.workspace, session)
        status = paths.read_status()
        if status is None or not paths.config_yaml.is_file():
            raise KeyError(f"会话不存在: {session}")
        # A finished session from a previous hub process: reconstruct a read-only handle.
        manifest = _manifest_for_status(self._config, status)
        handle = RunHandle(session, manifest, paths, process=None)  # type: ignore[arg-type]
        handle.state = status.get("state", "unknown")
        handle.error = status.get("error")
        handle.exit_code = status.get("exit_code")
        handle.started_at = status.get("started_at") or handle.started_at
        handle.ended_at = status.get("ended_at")
        return handle

    def has(self, session: str) -> bool:
        try:
            self.get(session)
            return True
        except KeyError:
            return False

    def stop(self, session: str) -> dict[str, Any]:
        handle = self.get(session)
        if handle.state != "running":
            return handle.status()
        if handle.process is None:  # pragma: no cover - read-only handles are terminal
            raise RunnerError("该会话不属于当前 hub 进程，无法停止")
        self._terminate(handle.process)
        deadline = time.monotonic() + 15
        while handle.state == "running" and time.monotonic() < deadline:
            time.sleep(0.1)
        if handle.state == "running":  # reaper hasn't finalized yet; force status
            handle.state = "stopped"
            handle.error = "被用户停止"
            handle.ended_at = datetime.now(timezone.utc).isoformat()
            handle.paths.write_status(handle.status())
        return handle.status()

    def tail_log(self, session: str, offset: int) -> tuple[int, str, str]:
        """Return (new offset, new text, state) for incremental log reading."""
        handle = self.get(session)
        path = handle.paths.run_log
        text = ""
        if path.is_file():
            size = path.stat().st_size
            if offset < 0 or offset > size:
                offset = 0
            if size > offset:
                with path.open("r", encoding="utf-8", errors="replace") as stream:
                    stream.seek(offset)
                    text = stream.read()
                offset = path.stat().st_size
        return offset, text, handle.state

    def read_log(self, session: str) -> str:
        handle = self.get(session)
        if not handle.paths.run_log.is_file():
            return ""
        return handle.paths.run_log.read_text(encoding="utf-8", errors="replace")

    def shutdown(self) -> None:
        with self._lock:
            running = [run for run in self._runs.values() if run.state == "running"]
        for handle in running:
            if handle.process is not None:
                self._terminate(handle.process)


def sys_executable() -> str:
    return shutil.which("python3") or shutil.which("python") or "python3"


def resolve_config_device(config: dict[str, Any]) -> str | None:
    """Resolve ``common.device == "auto"`` to a concrete ``cuda:N`` in place.

    Chapter adapters run in separate processes with no cross-process scheduler,
    so an unresolved ``"auto"`` would make each of them default to ``cuda:0``.
    Resolving here — to the GPU with the most free memory — gives concurrent
    chapter runs one shared, balanced view of the GPUs. Returns the resolved
    device, or None when kept as-is (no nvidia-smi / no GPUs / explicit device).
    """
    common = config.get("common")
    if not isinstance(common, dict) or str(common.get("device") or "auto") != "auto":
        return None
    resolved = resolve_auto_device()
    if resolved:
        common["device"] = resolved
    return resolved


def _manifest_for_status(config: HubConfig, status: dict[str, Any]) -> ChapterManifest:
    """Best-effort manifest lookup for sessions from a previous hub process."""
    directory = config.find_chapter_dir(str(status.get("toolkit") or ""))
    if directory is not None:
        from .registry import load_manifest

        return load_manifest(directory / "chapter.yaml")
    raise KeyError(f"会话对应的章节工具包不存在: {status.get('toolkit')}")


__all__ = [
    "RunManager",
    "RunnerBusyError",
    "RunnerError",
    "resolve_config_device",
    "sessions_root",
    "is_terminal",
]
