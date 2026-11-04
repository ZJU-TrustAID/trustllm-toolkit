from __future__ import annotations

from typing import Any

from hallucination.mitigation.backends.base import LLMBackend
from hallucination.mitigation.core.errors import BackendError, MissingDependencyError


class OpenAIBackend(LLMBackend):
    """Chat Completions backend for OpenAI and OpenAI-compatible servers."""

    def __init__(
        self,
        model: str,
        *,
        api_key: str,
        base_url: str | None = None,
        timeout: float = 60.0,
        max_retries: int = 2,
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise MissingDependencyError(
                "Install the OpenAI extra first: python -m pip install -e '.[openai]'"
            ) from exc
        self.model = model
        self._client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries,
        )

    def generate(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=float(kwargs.get("temperature", 0.0)),
                max_tokens=int(kwargs.get("max_new_tokens", 512)),
            )
        except Exception as exc:  # SDK exposes a hierarchy we do not want to pin here.
            raise BackendError(
                f"API request failed: {type(exc).__name__}: {exc}"
            ) from exc
        if not response.choices:
            raise BackendError("API returned no choices")
        content = response.choices[0].message.content
        return content.strip() if content else ""
