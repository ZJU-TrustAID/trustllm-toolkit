import importlib.util

import pytest

from hallucination.detection.backends.hf_whitebox import HuggingFaceWhiteBoxBackend


def test_hf_backend_imports_without_optional_dependencies():
    assert HuggingFaceWhiteBoxBackend is not None


def test_hf_backend_from_pretrained_requires_transformers_when_missing():
    if importlib.util.find_spec("transformers") is not None:
        pytest.skip("transformers is installed in this environment")
    with pytest.raises(ImportError, match="whitebox extra"):
        HuggingFaceWhiteBoxBackend.from_pretrained("gpt2")
