"""Adapters that turn standard hallucination benchmarks into mitigation records.

Every loader yields records shaped for :func:`evaluate_records`:
``{"id", "prompt", optional "context", "references": list[str]}``.

- ``truthfulqa_gen``: TruthfulQA generation split (HF ``truthful_qa``).
  References are the curated correct answers; scoring with ``contains``
  checks whether the generation expresses any of them.
- ``nq_open``: Natural Questions open (HF ``google-research-datasets/nq_open``).
  References are the gold short answers.
- ``nq_swap``: the NQ-Swap adversarial set (context answers were swapped).
  Loads the ``pminervini/NQ-Swap`` Hub mirror by default; pass ``path`` to use
  the official repository's ``test.json`` instead. The swapped context and
  answers become the record, which is the natural benchmark for CAD.

TruthfulQA and NQ-Open require the ``datasets`` extra and network access on
first use. Deterministic subsampling: rows keep dataset order, and
``limit``/``seed`` take a contiguous slice after a seeded stride shuffle so a
run can be reproduced from the seed alone.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

SUPPORTED_DATASETS = ("truthfulqa_gen", "nq_open", "nq_swap")


def _subsample(rows: list[dict[str, Any]], limit: int | None, seed: int) -> list[dict[str, Any]]:
    if limit is None or limit >= len(rows):
        return rows
    if limit <= 0:
        raise ValueError("limit must be positive")
    import random

    shuffled = list(rows)
    # A seeded shuffle keeps the subset reproducible from the seed alone.
    random.Random(seed).shuffle(shuffled)
    return shuffled[:limit]


def truth_record(row: dict[str, Any], index: int) -> dict[str, Any]:
    correct = [str(item).strip() for item in row.get("correct_answers", []) if str(item).strip()]
    best = str(row.get("best_answer", "")).strip()
    if best and best not in correct:
        correct.insert(0, best)
    return {
        "id": f"truthfulqa-{index}",
        "prompt": str(row.get("question", "")).strip(),
        "references": correct,
    }


def nq_record(row: dict[str, Any], index: int) -> dict[str, Any]:
    return {
        "id": f"nq-{index}",
        "prompt": str(row.get("question", "")).strip(),
        "references": [str(item).strip() for item in row.get("answer", []) if str(item).strip()],
    }


def nq_swap_record(row: dict[str, Any], index: int) -> dict[str, Any]:
    return {
        "id": f"nqswap-{index}",
        "prompt": str(row.get("question", "")).strip(),
        "context": str(row.get("context", "")).strip(),
        "references": [str(item).strip() for item in row.get("pair_answer", row.get("answer", []))]
        if isinstance(row.get("pair_answer", row.get("answer")), list)
        else [str(row.get("pair_answer", row.get("answer", ""))).strip()],
    }


def nq_swap_hub_record(row: dict[str, Any], index: int) -> dict[str, Any]:
    """Map the HF mirror (pminervini/NQ-Swap) substitution variant."""

    answers = row.get("sub_answer", [])
    return {
        "id": f"nqswap-hub-{index}",
        "prompt": str(row.get("question", "")).strip(),
        "context": str(row.get("sub_context", "")).strip(),
        "references": [str(item).strip() for item in answers if str(item).strip()],
    }


def load_truthfulqa_gen(*, limit: int | None = None, seed: int = 0) -> list[dict[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - exercised without the extra
        raise ImportError("Install the datasets extra first: python -m pip install -e '.[datasets]'") from exc
    dataset = load_dataset("truthfulqa/truthful_qa", "generation", split="validation")
    rows = [truth_record(dict(row), index) for index, row in enumerate(dataset)]
    rows = [row for row in rows if row["prompt"] and row["references"]]
    return _subsample(rows, limit, seed)


def load_nq_open(*, split: str = "validation", limit: int | None = None, seed: int = 0) -> list[dict[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - exercised without the extra
        raise ImportError("Install the datasets extra first: python -m pip install -e '.[datasets]'") from exc
    dataset = load_dataset("google-research-datasets/nq_open", split=split)
    rows = [nq_record(dict(row), index) for index, row in enumerate(dataset)]
    rows = [row for row in rows if row["prompt"] and row["references"]]
    return _subsample(rows, limit, seed)


def load_nq_swap(path: str | Path, *, limit: int | None = None, seed: int = 0) -> list[dict[str, Any]]:
    """Load NQ-Swap from a locally downloaded ``test.json``.

    Download: https://github.com/google-research-datasets/NQ-swap
    (``nqswap/data/test.json``).
    """

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = [nq_swap_record(dict(row), index) for index, row in enumerate(payload)]
    rows = [row for row in rows if row["prompt"] and row["context"] and row["references"]]
    return _subsample(rows, limit, seed)


def load_nq_swap_hub(*, limit: int | None = None, seed: int = 0) -> list[dict[str, Any]]:
    """Load the NQ-Swap substitution variant from the HF mirror.

    The official repository (google-research-datasets/NQ-swap) distributes
    plain JSON; the ``pminervini/NQ-Swap`` mirror hosts the same data on the
    Hub. The ``sub_context``/``sub_answer`` fields are the swapped condition,
    which is the adversarial context-faithfulness test for CAD.
    """

    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - exercised without the extra
        raise ImportError("Install the datasets extra first: python -m pip install -e '.[datasets]'") from exc
    dataset = load_dataset("pminervini/NQ-Swap", split="dev")
    rows = [nq_swap_hub_record(dict(row), index) for index, row in enumerate(dataset)]
    rows = [row for row in rows if row["prompt"] and row["context"] and row["references"]]
    return _subsample(rows, limit, seed)


def load_dataset_records(
    name: str,
    *,
    path: str | Path | None = None,
    limit: int | None = None,
    seed: int = 0,
) -> list[dict[str, Any]]:
    if name == "truthfulqa_gen":
        return load_truthfulqa_gen(limit=limit, seed=seed)
    if name == "nq_open":
        return load_nq_open(limit=limit, seed=seed)
    if name == "nq_swap":
        if path is not None:
            return load_nq_swap(path, limit=limit, seed=seed)
        return load_nq_swap_hub(limit=limit, seed=seed)
    raise ValueError(f"unknown dataset {name!r}; expected one of {SUPPORTED_DATASETS}")
