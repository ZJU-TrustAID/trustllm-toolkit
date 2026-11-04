from __future__ import annotations

import unicodedata
from dataclasses import dataclass

from hallucination.detection.core.results import DetectionExample


@dataclass(frozen=True)
class TokenFilter:
    """Select which generated tokens contribute to token-level scores.

    The filter is intentionally conservative: when the example does not carry
    token text or special-token flags, the corresponding rule is skipped rather
    than guessing. This keeps old callers working while allowing HF-backed
    examples to exclude EOS and other special tokens.
    """

    skip_special: bool = True
    skip_punctuation: bool = False
    skip_first: int = 0
    skip_last: int = 0

    def __post_init__(self) -> None:
        if self.skip_first < 0 or self.skip_last < 0:
            raise ValueError("skip_first and skip_last must be non-negative")

    def indices(self, example: DetectionExample) -> list[int]:
        if example.token_logprobs is None:
            raise ValueError("token filtering requires token_logprobs")
        size = len(example.token_logprobs)
        tokens = example.tokens
        special = example.token_is_special
        if tokens is not None and len(tokens) != size:
            raise ValueError("tokens and token_logprobs must have the same length")
        if special is not None and len(special) != size:
            raise ValueError("token_is_special and token_logprobs must have the same length")

        start = self.skip_first
        stop = size - self.skip_last if self.skip_last else size
        selected: list[int] = []
        for index in range(start, max(start, stop)):
            if self.skip_special and special is not None and special[index]:
                continue
            if self.skip_punctuation and tokens is not None and _is_punctuation(tokens[index]):
                continue
            selected.append(index)
        if not selected:
            raise ValueError("token filter removed every token; relax TokenFilter or provide more tokens")
        return selected

    def to_dict(self) -> dict[str, object]:
        return {
            "skip_special": self.skip_special,
            "skip_punctuation": self.skip_punctuation,
            "skip_first": self.skip_first,
            "skip_last": self.skip_last,
        }


def _is_punctuation(token: str) -> bool:
    stripped = token.strip()
    if not stripped:
        return True
    return all(unicodedata.category(character).startswith("P") for character in stripped)


def select_indices(example: DetectionExample, token_filter: TokenFilter | None) -> list[int]:
    if token_filter is None:
        if example.token_logprobs is None:
            raise ValueError("token scores require token_logprobs")
        return list(range(len(example.token_logprobs)))
    return token_filter.indices(example)
