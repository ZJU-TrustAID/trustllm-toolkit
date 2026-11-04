from __future__ import annotations

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")

from hallucination.mitigation.core.contracts import GenerationRequest
from hallucination.mitigation.core.model_config import ModelConfig
from hallucination.mitigation.engine.hf import HFEngine
from mitigation.test_engine_tiny import _build


def test_from_pretrained_local_directory(tmp_path) -> None:
    engine, tokenizer = _build()
    model_dir = tmp_path / "tiny-model"
    engine.model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)

    loaded = HFEngine.from_pretrained(
        ModelConfig(
            model=str(model_dir),
            backend="hf",
            device="cpu",
            dtype="float32",
            local_files_only=True,
        )
    )
    output = loaded.generate_text(GenerationRequest(prompt="w1 w2", max_new_tokens=2))
    assert output.generated_ids
