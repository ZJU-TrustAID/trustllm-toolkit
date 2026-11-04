# src/evaluation/loggers/flatten.py
from __future__ import annotations
from typing import Any, Dict, Iterable, Tuple

def flatten_dict(
    obj: Dict[str, Any],
    *,
    parent_key: str = "",
    sep: str = "/",
) -> Dict[str, Any]:
    items: list[Tuple[str, Any]] = []
    for k, v in obj.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else str(k)
        if isinstance(v, dict):
            items.extend(flatten_dict(v, parent_key=new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)
