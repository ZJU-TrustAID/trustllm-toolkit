import json

import pytest

from hallucination.detection.datasets import list_datasets, load_dataset
from hallucination.detection.datasets.schema import DatasetRecord


def test_loads_packaged_textbook_toy_dataset_offline():
    records = load_dataset("textbook_toy")
    assert [record.id for record in records] == ["blue-whale", "sea-anemone"]
    assert records[0].samples is not None
    assert records[0].unit_labels == [True]
    assert any(dataset.name == "textbook_toy" for dataset in list_datasets())


def test_loads_local_jsonl_and_preserves_split(tmp_path):
    path = tmp_path / "records.jsonl"
    path.write_text(json.dumps({"id": "one", "response": "answer", "units": ["answer"], "unit_labels": [False]}) + "\n", encoding="utf-8")
    records = load_dataset("jsonl", path=path, split="validation")
    assert records[0].metadata["split"] == "validation"
    assert records[0].unit_type == "sentence"


def test_dataset_record_rejects_misaligned_unit_labels():
    with pytest.raises(ValueError, match="same length"):
        DatasetRecord(id="bad", response="answer", units=["one", "two"], unit_labels=[True])
