from __future__ import annotations

from hallucination.detection.core.results import DetectionExample
from hallucination.detection.utils.text import split_sentences


class ClaimDecomposer:
    """Resolve supplied claims or use a deterministic sentence fallback.

    The fallback deliberately does not call a model and therefore does not
    claim to produce atomic facts. Applications needing atomic decomposition
    should supply ``DetectionExample.claims`` produced and audited upstream.
    """

    def decompose(self, example: DetectionExample) -> list[str]:
        if example.claims is not None:
            return self._clean(example.claims)
        if example.unit_type == "claim" and example.units is not None:
            return self._clean(example.units)
        return self._clean(split_sentences(example.response))

    @staticmethod
    def _clean(items: list[str]) -> list[str]:
        claims = [item.strip() for item in items if item.strip()]
        if not claims:
            raise ValueError("Evidence verification requires at least one non-empty claim")
        return claims
