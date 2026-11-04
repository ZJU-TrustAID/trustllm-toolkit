"""Explicit, local registration without import-time plugin discovery."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Generic, TypeVar

T = TypeVar("T")


@dataclass(frozen=True)
class _Entry(Generic[T]):
    factory: Callable[..., T]
    description: str


class Registry(Generic[T]):
    """Each chapter owns an independent registry of method factories."""

    def __init__(self) -> None:
        self._entries: dict[str, _Entry[T]] = {}

    @staticmethod
    def _key(name: str) -> str:
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Method name must be a non-empty string.")
        return name.strip().lower().replace("-", "_")

    def register(self, name: str, factory: Callable[..., T], *, description: str = "") -> None:
        key = self._key(name)
        if key in self._entries:
            raise ValueError(f"Method {key!r} is already registered.")
        if not callable(factory):
            raise TypeError("A method factory must be callable.")
        self._entries[key] = _Entry(factory, description)

    def _entry(self, name: str) -> _Entry[T]:
        key = self._key(name)
        try:
            return self._entries[key]
        except KeyError:
            available = ", ".join(self.names()) or "(none)"
            raise KeyError(f"Unknown method {name!r}. Available: {available}") from None

    def get(self, name: str, **options: object) -> T:
        """Create a fresh instance; constructor errors retain their traceback."""
        return self._entry(name).factory(**options)

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._entries))

    def describe(self, name: str) -> str:
        return self._entry(name).description
