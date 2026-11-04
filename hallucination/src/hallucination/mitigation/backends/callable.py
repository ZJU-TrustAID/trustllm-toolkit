from __future__ import annotations

from collections.abc import Callable
from typing import Any

from hallucination.mitigation.backends.base import LLMBackend


class CallableBackend(LLMBackend):
    """A test/custom backend.

    ``fn`` receives the message list and must return a string. If a list of
    strings is supplied instead, responses are returned in order.
    """

    def __init__(self, fn: Callable[..., str] | list[str]) -> None:
        self.fn = fn
        self.calls: list[list[dict[str, str]]] = []
        self._responses = list(fn) if isinstance(fn, list) else None

    def generate(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        self.calls.append(messages)
        if self._responses is not None:
            if not self._responses:
                return ""
            return self._responses.pop(0)
        return str(self.fn(messages, **kwargs))
