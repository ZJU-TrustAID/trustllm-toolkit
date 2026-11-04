from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DatasetInfo:
    name: str
    description: str
    remote: bool = False


_DATASETS = {
    "textbook_toy": DatasetInfo(
        name="textbook_toy",
        description="Small offline Chinese teaching dataset for hallucination-detection examples and tests.",
    ),
    "selfcheck_wikibio": DatasetInfo(
        name="selfcheck_wikibio",
        description="Sentence-level WikiBio hallucination data published with SelfCheckGPT, loaded from Hugging Face.",
        remote=True,
    ),
}


def list_datasets() -> list[DatasetInfo]:
    """List registered datasets without downloading any remote resource."""

    return list(_DATASETS.values())


def get_dataset_info(name: str) -> DatasetInfo:
    try:
        return _DATASETS[name]
    except KeyError as exc:
        available = ", ".join(sorted(_DATASETS))
        raise ValueError(f"Unknown dataset '{name}'. Available datasets: {available}") from exc
