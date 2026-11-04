"""Immutable messages and portable assembly artifacts."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, Protocol

Role = Literal["system", "user", "assistant"]


def require_text(value: str, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string.")


def require_count(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer.")


@dataclass(frozen=True)
class Message:
    role: Role
    content: str

    def __post_init__(self) -> None:
        if self.role not in ("system", "user", "assistant"):
            raise ValueError(f"Unsupported role: {self.role!r}")
        require_text(self.content, "content")

    def to_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


@dataclass(frozen=True)
class Step:
    """What was constructed, not a judgment of its effectiveness."""

    method: str
    operation: str
    source: str = ""

    def __post_init__(self) -> None:
        require_text(self.method, "method")
        require_text(self.operation, "operation")
        if not isinstance(self.source, str):
            raise TypeError("source must be a string.")


@dataclass(frozen=True)
class Prompt:
    """A request ready for the caller's client, with its construction trace.

    ``text`` is available only for single user messages. Conversations must be
    exported through ``to_messages()`` so history is never silently discarded.
    """

    query: str
    messages: tuple[Message, ...]
    steps: tuple[Step, ...] = ()

    def __post_init__(self) -> None:
        require_text(self.query, "query")
        object.__setattr__(self, "messages", tuple(self.messages))
        object.__setattr__(self, "steps", tuple(self.steps))
        if not self.messages or not all(isinstance(m, Message) for m in self.messages):
            raise ValueError("messages must contain Message objects.")
        if self.messages[-1].role != "user":
            raise ValueError("An assembled request must end with a user message.")
        if any(m.role == "system" for m in self.messages[1:]):
            raise ValueError("A system message may only appear at the beginning.")
        if not all(isinstance(s, Step) for s in self.steps):
            raise TypeError("steps must contain Step objects.")

    @property
    def text(self) -> str:
        if len(self.messages) != 1:
            raise ValueError("This is a conversation; use to_messages() to preserve history.")
        return self.messages[0].content

    def to_messages(self) -> list[dict[str, str]]:
        return [message.to_dict() for message in self.messages]

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "query": self.query,
            "messages": self.to_messages(),
            "steps": [asdict(step) for step in self.steps],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)

    @classmethod
    def from_json(cls, text: str) -> Prompt:
        data = json.loads(text)
        keys = {"schema_version", "query", "messages", "steps"}
        if not isinstance(data, dict) or set(data) != keys:
            raise ValueError("Invalid prompt artifact fields.")
        if type(data["schema_version"]) is not int or data["schema_version"] != 1:
            raise ValueError("Unsupported prompt schema version.")
        if not isinstance(data["messages"], list) or not isinstance(data["steps"], list):
            raise ValueError("messages and steps must be JSON arrays.")
        try:
            messages = tuple(Message(**item) for item in data["messages"])
            steps = tuple(Step(**item) for item in data["steps"])
        except TypeError as exc:
            raise ValueError("Invalid message or step fields.") from exc
        return cls(data["query"], messages, steps)

    def save(self, path: str | Path, *, overwrite: bool = False) -> None:
        """Write UTF-8 JSON, protecting existing files by default."""
        with Path(path).open("w" if overwrite else "x", encoding="utf-8") as stream:
            stream.write(self.to_json() + "\n")

    @classmethod
    def load(cls, path: str | Path) -> Prompt:
        return cls.from_json(Path(path).read_text(encoding="utf-8"))


class TextGenerator(Protocol):
    """Inject any local model or API client as a synchronous callable."""

    def __call__(self, messages: Sequence[Message]) -> str:
        """Return one candidate as plain text; failures should propagate."""
        ...
