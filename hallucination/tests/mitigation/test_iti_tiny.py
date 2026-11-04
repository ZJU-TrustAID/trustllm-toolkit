from __future__ import annotations

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("sklearn")

from hallucination.mitigation.core.contracts import GenerationRequest
from hallucination.mitigation.methods.iti import ITIConfig, ITIMethod, fit_iti
from mitigation.test_engine_tiny import _build


def test_iti_fit_and_run_tiny() -> None:
    engine, _ = _build()
    samples = [
        {"prompt": f"w{index}", "response": f"w{index + 1}", "label": index % 2}
        for index in range(3, 15)
    ]
    config = ITIConfig(top_k=2, alpha=0.1, baseline=False)
    artifact = fit_iti(engine, samples, config)
    assert artifact.heads
    assert artifact.model["head_dim"] == 8
    request = GenerationRequest(prompt="w1", max_new_tokens=2, do_sample=False, seed=0)
    report = ITIMethod().run(engine, request, artifact, config)
    assert report.method == "iti"
    assert report.response
    assert report.trace["heads"]
    assert report.trace["hook_calls"] > 0
