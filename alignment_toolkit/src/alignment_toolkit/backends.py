"""Backend capability contracts; actual training stays in mature backends."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .registry import register_backend


@dataclass(frozen=True)
class BackendDefinition:
    name: str
    methods: frozenset[str]
    external_command: str


class BackendAdapter(Protocol):
    definition: BackendDefinition

    def validate(self, stage: dict, dataset, model_roles: tuple[str, ...]) -> None:
        ...

    def run(self, stage: dict, dataset, artifacts: dict):
        ...


for _backend in (
    BackendDefinition("trl", frozenset({"sft", "dpo", "reward_model", "ppo"}), "trl"),
    BackendDefinition("openrlhf", frozenset({"sft", "dpo", "reward_model", "ppo"}), "openrlhf"),
    BackendDefinition("verl", frozenset({"ppo", "grpo", "online_rl"}), "verl"),
):
    register_backend(_backend.name, _backend)
