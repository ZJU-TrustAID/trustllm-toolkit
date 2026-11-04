from __future__ import annotations

import math
import re
from collections.abc import Sequence
from dataclasses import replace

from hallucination.detection.core.results import Evidence


class LocalCorpusRetriever:
    """A deterministic lexical retriever for small, offline teaching corpora.

    It is intentionally simple and records its ranking score in returned
    evidence. It is not intended to replace a production retrieval system.
    """

    def __init__(self, documents: Sequence[Evidence]) -> None:
        if not documents:
            raise ValueError("LocalCorpusRetriever requires at least one document")
        self.documents = list(documents)

    def retrieve(self, query: str, *, k: int = 5) -> list[Evidence]:
        if k < 1:
            raise ValueError("k must be positive")
        query_terms = _terms(query)
        ranked = sorted(
            ((self._score(query_terms, document), index, document) for index, document in enumerate(self.documents)),
            key=lambda item: (-item[0], item[1]),
        )
        return [
            replace(document, metadata={**document.metadata, "retrieval": "local_lexical", "retrieval_score": score})
            for score, _, document in ranked[:k]
            if score > 0.0
        ]

    @staticmethod
    def _score(query_terms: set[str], document: Evidence) -> float:
        document_terms = _terms(" ".join(part for part in (document.title, document.snippet) if part))
        if not query_terms or not document_terms:
            return 0.0
        return len(query_terms & document_terms) / math.sqrt(len(query_terms) * len(document_terms))


def _terms(text: str) -> set[str]:
    normalized = text.lower()
    words = re.findall(r"[a-z0-9]+", normalized)
    chinese = "".join(re.findall(r"[\u4e00-\u9fff]", normalized))
    bigrams = [chinese[index : index + 2] for index in range(max(0, len(chinese) - 1))]
    return set(words + bigrams)
