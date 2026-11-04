"""The jailbreak-specific subclass and lossless composition rules."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from textbook_toolkit.core import Method

from .types import Message, Prompt, Step, require_text


class Assembler(Method[str, Prompt]):
    """Construct prompts without performing target inference or evaluation."""

    def build_many(self, queries: Iterable[str]) -> tuple[Prompt, ...]:
        if isinstance(queries, str):
            raise TypeError("Pass an iterable of queries, not one string.")
        return tuple(self.build(query) for query in queries)

    def then(self, other: Assembler) -> Pipeline:
        return Pipeline((self, other))


def assembled(
    query: str, text: str, *, method: str, operation: str, source: str = ""
) -> Prompt:
    return Prompt(query, (Message("user", text),), (Step(method, operation, source),))


@dataclass(frozen=True)
class Pipeline(Assembler):
    """Feed each stage's text into the next; only the final stage may add history."""

    stages: tuple[Assembler, ...]

    def __post_init__(self) -> None:
        flattened = []
        for stage in self.stages:
            if not isinstance(stage, Assembler):
                raise TypeError("Pipeline stages must be Assembler instances.")
            flattened.extend(stage.stages if isinstance(stage, Pipeline) else (stage,))
        if not flattened:
            raise ValueError("A pipeline needs at least one stage.")
        object.__setattr__(self, "stages", tuple(flattened))

    def build(self, query: str) -> Prompt:
        require_text(query, "query")
        current = query
        steps: list[Step] = []
        for index, stage in enumerate(self.stages):
            result = stage.build(current)
            steps.extend(result.steps)
            if index < len(self.stages) - 1:
                current = result.text  # Explicitly rejects dropping conversation history.
        return Prompt(query, result.messages, tuple(steps))
