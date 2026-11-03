"""Hub self-configuration: ``hub.config.json`` with a ``hub.config.local.json`` override.

Both files live in the ``hub/`` directory; paths inside are resolved against the
repository root (the parent of ``hub/``). The two files are never merged
field-by-field: the local file replaces the default one entirely, mirroring the
guardrail toolkit's convention.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

DEFAULTS: dict[str, Any] = {
    "host": "127.0.0.1",
    "port": 8680,
    # Executable used to expand the {conda} placeholder in chapter commands.
    "conda_bin": "conda",
    "workspace_dir": "workspace",
    # Directories scanned for chapter.yaml: the directory itself and its direct
    # children. The default scans the repository root's top-level folders.
    "chapter_dirs": ["."],
    # Optional per-toolkit overrides for the chapter frontend server port.
    "frontend_ports": {},
    "max_concurrent_runs": 1,
    "frontend_server_timeout": 3600,
    # Directories scanned by GET /api/models/local to offer a model dropdown.
    "model_dirs": ["models"],
}

_EXCLUDED_DIR_NAMES = {"hub", "workspace", "models", "docs", "docs-archive", "node_modules", "runs"}


class HubConfig:
    def __init__(self, root: Path, hub_dir: Path, data: dict[str, Any]) -> None:
        self.root = root
        self.hub_dir = hub_dir
        self.data = data

    def get(self, key: str, default: Any = None) -> Any:
        if key in self.data:
            return self.data[key]
        return DEFAULTS.get(key, default)

    @property
    def host(self) -> str:
        return str(self.get("host"))

    @property
    def port(self) -> int:
        return int(self.get("port"))

    @property
    def conda_bin(self) -> str:
        return str(self.get("conda_bin"))

    @property
    def workspace(self) -> Path:
        return (self.root / str(self.get("workspace_dir"))).resolve()

    @property
    def max_concurrent_runs(self) -> int:
        return max(1, int(self.get("max_concurrent_runs")))

    @property
    def frontend_server_timeout(self) -> int:
        return int(self.get("frontend_server_timeout"))

    @property
    def chapter_roots(self) -> list[Path]:
        return [(self.root / str(entry)).resolve() for entry in self.get("chapter_dirs", ["."])]

    def frontend_port(self, toolkit: str, default: int) -> int:
        overrides = self.get("frontend_ports") or {}
        return int(overrides.get(toolkit, default))

    def iter_chapter_dirs(self) -> list[Path]:
        """Directories to scan for chapter.yaml, excluding hub-internal folders."""
        found: list[Path] = []
        for base in self.chapter_roots:
            candidates = [base, *sorted(p for p in base.iterdir() if p.is_dir())] if base.is_dir() else []
            for candidate in candidates:
                if candidate.name.startswith(".") or candidate.name in _EXCLUDED_DIR_NAMES:
                    continue
                if (candidate / "chapter.yaml").is_file():
                    found.append(candidate)
        return found

    def find_chapter_dir(self, toolkit: str) -> Path | None:
        """Locate a chapter directory by toolkit slug (used for past sessions)."""
        for directory in self.iter_chapter_dirs():
            if directory.name == toolkit or _manifest_toolkit(directory) == toolkit:
                return directory
        return None


def _manifest_toolkit(directory: Path) -> str | None:
    import yaml

    manifest = directory / "chapter.yaml"
    try:
        data = yaml.safe_load(manifest.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        return None
    return str(data.get("toolkit")) if isinstance(data, dict) and data.get("toolkit") else None


def load_config(root: Path | None = None) -> HubConfig:
    """Load the hub configuration; ``root`` is the repository root."""
    if root is None:
        root = Path.cwd()
    root = Path(root).resolve()
    hub_dir = Path(__file__).resolve().parents[2]
    data = dict(DEFAULTS)
    for name in ("hub.config.json", "hub.config.local.json"):
        path = hub_dir / name
        if path.is_file():
            loaded = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(loaded, dict):
                raise ValueError(f"{path} 必须是 JSON 对象")
            data.update(loaded)
    return HubConfig(root=root, hub_dir=hub_dir, data=data)
