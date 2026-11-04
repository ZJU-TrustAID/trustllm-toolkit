from __future__ import annotations

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")

from hallucination.mitigation import Mitigator
from mitigation.test_engine_tiny import _build


def test_mitigator_real_local_model(tmp_path) -> None:
    engine, tokenizer = _build()
    model_dir = tmp_path / "tiny-model"
    engine.model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)
    mitigator = Mitigator(
        method="dola",
        model=str(model_dir),
        backend="hf",
        device="cpu",
        dtype="float32",
        local_files_only=True,
        candidate_premature_layers=[0],
        mature_layer=2,
        seed=0,
    )
    try:
        report = mitigator.mitigate("w1 w2", max_new_tokens=2, baseline=True)
    finally:
        mitigator.close()
    assert report.method == "dola"
    assert report.baseline is not None
    assert report.response
