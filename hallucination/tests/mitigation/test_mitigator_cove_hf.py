from __future__ import annotations

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")

from hallucination.mitigation import Mitigator
from mitigation.test_engine_tiny import _build


def test_mitigator_cove_local_model(tmp_path) -> None:
    engine, tokenizer = _build()
    model_dir = tmp_path / "tiny-model"
    engine.model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)
    mitigator = Mitigator(
        method="cove",
        model=str(model_dir),
        backend="hf",
        device="cpu",
        dtype="float32",
        local_files_only=True,
        seed=0,
        max_questions=2,
    )
    try:
        report = mitigator.mitigate("w1 w2", max_new_tokens=8)
    finally:
        mitigator.close()
    assert report.method == "cove"
    assert report.response
