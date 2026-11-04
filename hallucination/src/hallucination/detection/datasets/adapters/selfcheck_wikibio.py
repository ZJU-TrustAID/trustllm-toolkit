from __future__ import annotations

from typing import Any

from hallucination.detection.datasets.schema import DatasetRecord

DATASET_NAME = "potsawee/wiki_bio_gpt3_hallucination"


def load(split: str | None = None) -> list[DatasetRecord]:
    """Load the source SelfCheckGPT WikiBio dataset through Hugging Face.

    The dataset is not bundled. Readers remain responsible for accepting the
    source dataset's license and citation requirements.
    """

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError("Install the datasets extra first: python3 -m pip install -e '.[datasets]'") from exc
    dataset = load_dataset(DATASET_NAME, split=split or "train")
    return [_record(row, index) for index, row in enumerate(dataset)]


def _record(row: dict[str, Any], index: int) -> DatasetRecord:
    units = _string_list(row.get("gpt3_sentences"), "gpt3_sentences")
    labels = _labels(row.get("annotation"), len(units))
    return DatasetRecord(
        id=str(row.get("wiki_bio_test_idx", index)),
        response=_string(row.get("gpt3_text"), "gpt3_text"),
        samples=_string_list(row.get("gpt3_text_samples"), "gpt3_text_samples"),
        units=units,
        unit_labels=labels,
        unit_type="sentence",
        metadata={"source_dataset": DATASET_NAME, "source_index": index},
    )


def _string(value: Any, field_name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{DATASET_NAME} record is missing string field '{field_name}'")
    return value


def _string_list(value: Any, field_name: str) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"{DATASET_NAME} record is missing list[str] field '{field_name}'")
    return value


def _labels(value: Any, expected_length: int) -> list[bool | str]:
    if not isinstance(value, list) or len(value) != expected_length:
        raise ValueError(f"{DATASET_NAME} annotation must be a list aligned with gpt3_sentences")
    labels: list[bool | str] = []
    for annotation in value:
        if isinstance(annotation, bool):
            labels.append(annotation)
        elif isinstance(annotation, (int, float)) and annotation in {0, 1}:
            labels.append(bool(annotation))
        elif isinstance(annotation, str):
            labels.append(annotation)
        else:
            raise ValueError(f"Unsupported {DATASET_NAME} annotation value: {annotation!r}")
    return labels
