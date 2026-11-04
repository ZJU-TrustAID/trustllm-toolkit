from __future__ import annotations

from typing import Protocol

from hallucination.detection.core.results import Evidence


class RetrieverBackend(Protocol):
    def retrieve(self, query: str, *, k: int = 5) -> list[Evidence]:
        ...

