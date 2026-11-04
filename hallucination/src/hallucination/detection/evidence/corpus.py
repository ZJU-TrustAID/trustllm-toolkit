from __future__ import annotations

import json
from importlib import resources

from hallucination.detection.core.results import Evidence


def load_textbook_evidence() -> list[Evidence]:
    """Load the small offline fact corpus used by the textbook examples."""

    data_file = resources.files("hallucination.detection.datasets.data").joinpath("textbook_evidence.jsonl")
    return [Evidence(**json.loads(line)) for line in data_file.read_text(encoding="utf-8").splitlines() if line.strip()]
