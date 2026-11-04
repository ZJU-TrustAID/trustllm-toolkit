from __future__ import annotations

from typing import Protocol


class LLMBackend(Protocol):
    def generate(self, prompt: str, **kwargs: object) -> str:
        ...

    def sample(self, prompt: str, n: int, **kwargs: object) -> list[str]:
        ...

