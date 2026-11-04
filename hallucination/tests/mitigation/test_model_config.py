from __future__ import annotations

import pytest

from hallucination.mitigation.core.errors import (
    BackendError,
    ConfigError,
    UnsupportedModelError,
)
from hallucination.mitigation.core.model_config import ModelConfig


def test_api_backend_rejects_whitebox_method() -> None:
    config = ModelConfig(model="gpt-4o-mini", backend="openai")
    with pytest.raises(UnsupportedModelError):
        config.validate_for_method("dola")


def test_openai_compatible_requires_api_base() -> None:
    config = ModelConfig(model="local", backend="openai-compatible")
    with pytest.raises(ConfigError):
        config.validate_for_method("cove")


def test_api_key_comes_from_environment(monkeypatch) -> None:
    config = ModelConfig(model="gpt", backend="openai", api_key_env="HM_TEST_KEY")
    monkeypatch.setenv("HM_TEST_KEY", "secret")
    assert config.resolve_api_key() == "secret"


def test_missing_api_key_raises(monkeypatch) -> None:
    config = ModelConfig(model="gpt", backend="openai", api_key_env="HM_MISSING_KEY")
    monkeypatch.delenv("HM_MISSING_KEY", raising=False)
    with pytest.raises(BackendError):
        config.resolve_api_key()
