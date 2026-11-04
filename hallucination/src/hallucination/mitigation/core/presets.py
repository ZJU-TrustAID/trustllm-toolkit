from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from hallucination.mitigation.core.errors import ConfigError


def _preset_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "presets"


def list_presets() -> list[str]:
    directory = _preset_dir()
    if not directory.exists():
        return []
    return sorted(path.stem for path in directory.glob("*.yaml"))


def load_preset(name: str) -> dict[str, Any]:
    path = _preset_dir() / f"{name}.yaml"
    if not path.exists():
        available = ", ".join(list_presets()) or "(none)"
        raise ConfigError(f"unknown preset {name!r}; available: {available}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ConfigError(f"preset {name!r} must be a mapping")
    return payload
