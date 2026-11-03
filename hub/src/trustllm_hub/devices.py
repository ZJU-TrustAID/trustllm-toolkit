"""GPU inventory via nvidia-smi (keeps the hub environment torch-free).

Two consumers:
- the frontend device dropdown lists ``cuda:N`` options with free memory;
- the runner resolves ``common.device == "auto"`` to the freest ``cuda:N``
  before writing the run YAML, so every chapter process shares one resolved
  view of the runtime environment instead of each piling onto ``cuda:0``.
"""
from __future__ import annotations

import shutil
import subprocess
from typing import Any

_QUERY_ARGS = (
    "--query-gpu=index,name,memory.used,memory.total",
    "--format=csv,noheader,nounits",
)


def _query_nvidia_smi() -> str | None:
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return None
    try:
        result = subprocess.run(
            [executable, *_QUERY_ARGS],
            capture_output=True, text=True, timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    return result.stdout


def parse_smi_output(output: str) -> list[dict[str, Any]]:
    gpus: list[dict[str, Any]] = []
    for line in output.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 4:
            continue
        try:
            index, name, used, total = int(parts[0]), parts[1], int(parts[2]), int(parts[3])
        except ValueError:
            continue
        gpus.append({
            "index": index,
            "name": name,
            "memory_used_mb": used,
            "memory_total_mb": total,
            "memory_free_mb": max(total - used, 0),
        })
    gpus.sort(key=lambda gpu: gpu["index"])
    return gpus


def list_gpus() -> list[dict[str, Any]]:
    """Visible GPUs as [{index, name, memory_used_mb, memory_total_mb, memory_free_mb}]."""
    output = _query_nvidia_smi()
    if not output:
        return []
    return parse_smi_output(output)


def resolve_auto_device() -> str | None:
    """``cuda:N`` of the GPU with the most free memory, or None without GPU info."""
    gpus = list_gpus()
    if not gpus:
        return None
    best = max(gpus, key=lambda gpu: (gpu["memory_free_mb"], -gpu["index"]))
    return f"cuda:{best['index']}"


__all__ = ["list_gpus", "parse_smi_output", "resolve_auto_device"]
