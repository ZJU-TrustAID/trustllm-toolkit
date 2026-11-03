"""Registries shared by built-ins and external plugins."""

from __future__ import annotations

from typing import Any


class Registry:
    def __init__(self, kind: str):
        self.kind = kind
        self._items: dict[str, Any] = {}

    def register(self, name: str, item=None):
        key = str(name).strip().lower()
        if not key:
            raise ValueError(f"{self.kind} names cannot be empty")

        def decorator(value):
            if key in self._items:
                raise ValueError(f"{self.kind} {key!r} is already registered")
            self._items[key] = value
            return value

        return decorator(item) if item is not None else decorator

    def get(self, name: str):
        key = str(name).strip().lower()
        try:
            return self._items[key]
        except KeyError as exc:
            available = ", ".join(sorted(self._items)) or "<empty>"
            raise KeyError(f"Unknown {self.kind} {name!r}; available: {available}") from exc

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._items))


DATASETS = Registry("dataset")
METHODS = Registry("method")
BACKENDS = Registry("backend")


def register_dataset(name: str, value=None):
    return DATASETS.register(name, value)


def register_method(name: str, value=None):
    return METHODS.register(name, value)


def register_backend(name: str, value=None):
    return BACKENDS.register(name, value)
