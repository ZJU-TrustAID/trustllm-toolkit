from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

from hallucination.mitigation.core.errors import (
    BackendError,
    ConfigError,
    UnsupportedModelError,
)

_BACKENDS = {"hf", "openai", "openai-compatible", "callable"}


@dataclass
class ModelConfig:
    model: str
    backend: str = "hf"
    revision: str | None = None
    device: str | None = None
    dtype: str = "auto"
    local_files_only: bool = False
    trust_remote_code: bool = False
    load_in_4bit: bool = False
    api_base: str | None = None
    api_key_env: str = "OPENAI_API_KEY"
    timeout: float = 60.0
    max_retries: int = 2
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.backend = self.backend.strip().lower()
        if self.backend not in _BACKENDS:
            raise ConfigError(f"backend must be one of {sorted(_BACKENDS)}")
        if not self.model:
            raise ConfigError("model must be non-empty")
        if self.timeout <= 0:
            raise ConfigError("timeout must be positive")
        if self.max_retries < 0:
            raise ConfigError("max_retries must be >= 0")
        if self.device == "auto":
            # "auto" means pick automatically; None already resolves to
            # cuda-if-available at load time, while the literal string would
            # reach torch as an invalid device.
            self.device = None

    @classmethod
    def from_mapping(cls, payload: dict[str, Any]) -> ModelConfig:
        known = {
            "model",
            "backend",
            "revision",
            "device",
            "dtype",
            "local_files_only",
            "trust_remote_code",
            "load_in_4bit",
            "api_base",
            "api_key_env",
            "timeout",
            "max_retries",
        }
        data = {
            key: value
            for key, value in payload.items()
            if key in known and value is not None
        }
        extra = {key: value for key, value in payload.items() if key not in known}
        if "model" not in data:
            raise ConfigError("model is required")
        return cls(**data, extra=extra)

    @property
    def is_api(self) -> bool:
        return self.backend in {"openai", "openai-compatible"}

    def validate_for_method(self, method: str) -> None:
        if self.is_api and method != "cove":
            raise UnsupportedModelError(
                f"method={method!r} requires a local white-box model; the API backend only supports 'cove'. "
                "Use --backend hf with a Hugging Face or local model."
            )
        if self.is_api and not self.api_base and self.backend == "openai-compatible":
            raise ConfigError("--api-base is required for backend=openai-compatible")

    def resolve_api_key(self) -> str:
        value = os.environ.get(self.api_key_env)
        if not value:
            raise BackendError(
                f"API key not found. Set the environment variable {self.api_key_env!r}; "
                "API keys are not accepted as CLI arguments."
            )
        return value

    def fingerprint(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "backend": self.backend,
            "revision": self.revision,
            "local_files_only": self.local_files_only,
            "api_base": self.api_base if self.is_api else None,
        }
