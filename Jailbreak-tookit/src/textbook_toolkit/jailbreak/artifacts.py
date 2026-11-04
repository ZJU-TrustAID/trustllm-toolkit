"""接入外部优化产物。明确区分组装文本与运行原论文优化算法。"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from .base import Assembler, assembled
from .types import Prompt, require_text


@dataclass(frozen=True)
class Fragment:
    """An external optimizer's text artifact.

    A query-specific artifact can bind itself to that exact query. ``source``
    records provenance, not a verified claim about the producing algorithm.
    Raw embeddings and token IDs are intentionally not coerced into text.
    """

    text: str
    placement: Literal["prefix", "suffix", "replacement"] = "suffix"
    source: str = "caller-supplied"
    for_query: str | None = None

    def __post_init__(self) -> None:
        require_text(self.text, "fragment.text")
        require_text(self.source, "fragment.source")
        if self.placement not in ("prefix", "suffix", "replacement"):
            raise ValueError("placement must be prefix, suffix, or replacement.")
        if self.for_query is not None:
            require_text(self.for_query, "fragment.for_query")

    def apply(self, query: str, separator: str = " ") -> str:
        require_text(query, "query")
        if self.for_query is not None and self.for_query != query:
            raise ValueError("This fragment is bound to a different query.")
        if not isinstance(separator, str):
            raise TypeError("separator must be a string.")
        if self.placement == "replacement":
            return self.text
        left, right = (self.text, query) if self.placement == "prefix" else (query, self.text)
        return left + separator + right


FragmentProvider = Callable[[str], Fragment]


@dataclass(frozen=True)
class OptimizedPrompt(Assembler):
    """Assemble a supplied fragment, or call an explicitly supplied provider once.

    The provider owns model loading, tokenization and optimization. The adapter
    only handles the resulting text and never claims to have run that algorithm.
    """

    fragment: Fragment | FragmentProvider
    name: str = "optimized"
    separator: str = " "
    allowed_placements: tuple[str, ...] = ("prefix", "suffix", "replacement")

    def __post_init__(self) -> None:
        if not isinstance(self.fragment, Fragment) and not callable(self.fragment):
            raise TypeError("fragment must be a Fragment or a callable returning one.")
        if not isinstance(self.separator, str):
            raise TypeError("separator must be a string.")
        object.__setattr__(self, "allowed_placements", tuple(self.allowed_placements))

    def build(self, query: str) -> Prompt:
        require_text(query, "query")
        fragment = self.fragment(query) if callable(self.fragment) else self.fragment
        if not isinstance(fragment, Fragment):
            raise TypeError("The fragment provider must return a Fragment.")
        if fragment.placement not in self.allowed_placements:
            raise ValueError(f"{self.name} expects placements {self.allowed_placements}.")
        return assembled(
            query,
            fragment.apply(query, self.separator),
            method=self.name,
            operation=f"assemble_{fragment.placement}",
            source=fragment.source,
        )


@dataclass(frozen=True)
class Suffix(Assembler):
    """The simplest artifact constructor, with no algorithm attribution."""

    suffix: str
    separator: str = " "

    def __post_init__(self) -> None:
        require_text(self.suffix, "suffix")
        if not isinstance(self.separator, str):
            raise TypeError("separator must be a string.")

    def build(self, query: str) -> Prompt:
        return assembled(
            query,
            Fragment(self.suffix).apply(query, self.separator),
            method="suffix",
            operation="append_supplied_suffix",
        )
