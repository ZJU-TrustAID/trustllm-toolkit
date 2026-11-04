from __future__ import annotations

from hallucination.mitigation import Mitigator


def test_preset_sets_method_and_model() -> None:
    mitigator = Mitigator(preset="cove_openai_gpt4o_mini")
    assert mitigator.method == "cove"
    assert mitigator.model == "gpt-4o-mini"
    assert mitigator.backend == "openai"


def test_explicit_method_overrides_preset() -> None:
    mitigator = Mitigator(preset="cove_openai_gpt4o_mini", method="dola", model="local")
    assert mitigator.method == "dola"
    assert mitigator.model == "local"
