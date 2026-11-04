from hallucination.detection.datasets.adapters.selfcheck_wikibio import _record


def test_selfcheck_wikibio_adapter_maps_document_fields_to_dataset_record():
    record = _record(
        {
            "wiki_bio_test_idx": 7,
            "gpt3_text": "Generated passage.",
            "gpt3_text_samples": ["Sample one.", "Sample two."],
            "gpt3_sentences": ["Generated passage."],
            "annotation": [True],
        },
        0,
    )
    assert record.id == "7"
    assert record.samples == ["Sample one.", "Sample two."]
    assert record.unit_labels == [True]
