from __future__ import annotations

import pytest

from hallucination.mitigation import Mitigator
from hallucination.mitigation.backends.callable import CallableBackend
from hallucination.mitigation.core.errors import UnsupportedModelError


def test_mitigator_runs_cove_with_callable_backend() -> None:
    backend = CallableBackend(
        [
            "Draft answer.",
            "1. Is A true?",
            "Yes.",
            "Revised answer.",
        ]
    )
    mitigator = Mitigator(
        method="cove", model="test-model", backend="callable", backend_obj=backend
    )
    report = mitigator.mitigate("Question?", max_new_tokens=32)
    assert report.method == "cove"
    assert report.response == "Revised answer."
    assert report.provenance["package_version"]


def test_api_backend_rejects_whitebox_methods() -> None:
    mitigator = Mitigator(method="dola", model="gpt-4o-mini", backend="openai")
    with pytest.raises(UnsupportedModelError):
        mitigator.mitigate("Question?")
