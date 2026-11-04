from __future__ import annotations

import csv
import json
from importlib import resources
from pathlib import Path
from typing import Any

from hallucination.detection.core.results import Evidence
from hallucination.detection.datasets.registry import get_dataset_info
from hallucination.detection.datasets.schema import DatasetRecord


def load_dataset(name: str, *, path: str | Path | None = None, split: str | None = None) -> list[DatasetRecord]:
    """Load a registered teaching dataset or a local JSON, JSONL, or CSV file.

    `split` is accepted for API compatibility and preserved in record metadata.
    Local files must provide at least `id` and `response` columns/keys.
    """

    if name in {"json", "jsonl", "csv"}:
        if path is None:
            raise ValueError(f"load_dataset('{name}') requires path")
        rows = _load_local(Path(path), name)
    elif name == "textbook_toy":
        data_file = resources.files("hallucination.detection.datasets.data").joinpath("textbook_toy.jsonl")
        rows = _load_jsonl_text(data_file.read_text(encoding="utf-8"))
    elif name == "selfcheck_wikibio":
        from hallucination.detection.datasets.adapters.selfcheck_wikibio import (
            load as load_selfcheck_wikibio,
        )

        return load_selfcheck_wikibio(split)
    else:
        get_dataset_info(name)
        raise AssertionError("Unreachable")

    records = [_record_from_mapping(row, split=split) for row in rows]
    if not records:
        raise ValueError("Dataset contains no records")
    return records


def _load_local(path: Path, kind: str) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    text = path.read_text(encoding="utf-8")
    if kind == "json":
        payload = json.loads(text)
        if not isinstance(payload, list):
            raise ValueError("JSON datasets must contain a list of objects")
        return [_require_mapping(row) for row in payload]
    if kind == "jsonl":
        return _load_jsonl_text(text)
    if kind == "csv":
        return [dict(row) for row in csv.DictReader(text.splitlines())]
    raise ValueError(f"Unsupported local dataset format: {kind}")


def _load_jsonl_text(text: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            rows.append(_require_mapping(json.loads(line)))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSONL on line {line_number}") from exc
    return rows


def _record_from_mapping(row: dict[str, Any], *, split: str | None) -> DatasetRecord:
    metadata = _as_dict(row.get("metadata"))
    if split is not None:
        metadata = {**metadata, "split": split}
    evidence = [Evidence(**item) for item in _as_list_of_dicts(row.get("evidence"))]
    return DatasetRecord(
        id=str(row.get("id", "")),
        prompt=_optional_text(row.get("prompt")),
        response=_required_text(row.get("response"), "response"),
        samples=_optional_text_list(row.get("samples")),
        reference_answer=row.get("reference_answer"),
        units=_optional_text_list(row.get("units")),
        unit_labels=_optional_list(row.get("unit_labels")),
        unit_type=row.get("unit_type"),
        response_label=row.get("response_label"),
        split=split,
        label_source=_optional_text(row.get("label_source")),
        claims=_optional_text_list(row.get("claims")),
        claim_labels=_optional_text_list(row.get("claim_labels")),
        evidence=evidence,
        metadata=metadata,
    )


def _require_mapping(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("Each dataset record must be an object")
    return value


def _required_text(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Dataset record requires non-empty '{field_name}'")
    return value


def _optional_text(value: Any) -> str | None:
    return value if isinstance(value, str) else None


def _optional_text_list(value: Any) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return [value]
        value = parsed
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError("Expected a list of strings")
    return value


def _optional_list(value: Any) -> list[Any] | None:
    if value is None:
        return None
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return [value]
    if not isinstance(value, list):
        raise ValueError("Expected a list")
    return value


def _as_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError("metadata must be an object")
    return value


def _as_list_of_dicts(value: Any) -> list[dict[str, Any]]:
    if value is None:
        return []
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        raise ValueError("evidence must be a list of objects")
    return value
