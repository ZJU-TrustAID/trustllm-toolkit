"""Session directories: one unified YAML run per session under ``workspace/``.

Layout::

    workspace/sessions/<id>/
        config.yaml   # the executed unified YAML
        run.log       # adapter stdout+stderr, streamed to the hub UI
        status.json   # hub-written lifecycle state
        result.json   # adapter-written structured result (blocks + artifacts)
        ...           # adapter artifacts (report.md, report.json, ...)
"""
from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
from typing import Any
import uuid

_TERMINAL_STATES = {"succeeded", "failed", "timeout", "stopped"}


def new_session_id() -> str:
    now = datetime.now()
    return f"{now:%Y%m%d-%H%M}-{uuid.uuid4().hex[:4]}"


def sessions_root(workspace: Path) -> Path:
    return workspace / "sessions"


class SessionPaths:
    def __init__(self, workspace: Path, session: str) -> None:
        self.session = session
        self.dir = sessions_root(workspace) / session

    @property
    def config_yaml(self) -> Path:
        return self.dir / "config.yaml"

    @property
    def run_log(self) -> Path:
        return self.dir / "run.log"

    @property
    def status_json(self) -> Path:
        return self.dir / "status.json"

    @property
    def result_json(self) -> Path:
        return self.dir / "result.json"

    def prepare(self) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)

    def exists(self) -> bool:
        return self.dir.is_dir()

    def write_status(self, status: dict[str, Any]) -> None:
        self.status_json.write_text(
            json.dumps(status, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

    def read_status(self) -> dict[str, Any] | None:
        if not self.status_json.is_file():
            return None
        try:
            return json.loads(self.status_json.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None

    def read_result(self) -> dict[str, Any] | None:
        if not self.result_json.is_file():
            return None
        try:
            return json.loads(self.result_json.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None

    def read_config(self) -> dict[str, Any] | None:
        if not self.config_yaml.is_file():
            return None
        import yaml

        try:
            return yaml.safe_load(self.config_yaml.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError):
            return None


def is_terminal(state: str) -> bool:
    return state in _TERMINAL_STATES


def list_sessions(workspace: Path, limit: int = 50) -> list[dict[str, Any]]:
    """Recent sessions, newest first, based on status.json (or inferred fields)."""
    root = sessions_root(workspace)
    if not root.is_dir():
        return []
    items: list[dict[str, Any]] = []
    for path in sorted(root.iterdir(), reverse=True):
        if not path.is_dir():
            continue
        paths = SessionPaths(workspace, path.name)
        status = paths.read_status() or {}
        config = paths.read_config() or {}
        items.append(
            {
                "session": path.name,
                "state": status.get("state", "unknown"),
                "toolkit": status.get("toolkit") or config.get("toolkit"),
                "chapter": status.get("chapter") if status.get("chapter") is not None else config.get("chapter"),
                "started_at": status.get("started_at"),
                "ended_at": status.get("ended_at"),
                "exit_code": status.get("exit_code"),
                "error": status.get("error"),
                "has_result": paths.result_json.is_file(),
            }
        )
        if len(items) >= limit:
            break
    return items
