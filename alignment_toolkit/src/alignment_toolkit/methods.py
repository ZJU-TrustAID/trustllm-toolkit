"""Alignment method definitions."""

from __future__ import annotations

from dataclasses import dataclass

from .registry import register_method


@dataclass(frozen=True)
class MethodDefinition:
    name: str
    dataset_kinds: tuple[str, ...]
    model_roles: tuple[str, ...]
    default_backend: str
    supports_lora: bool = True


_METHODS = (
    MethodDefinition("sft", ("instruction",), ("policy",), "trl"),
    MethodDefinition("dpo", ("preference",), ("policy", "reference"), "trl"),
    MethodDefinition("reward_model", ("preference", "reward"), ("reward",), "trl"),
    MethodDefinition("ppo", ("prompt",), ("policy", "reference", "reward", "value"), "trl"),
)

for _method in _METHODS:
    register_method(_method.name, _method)
