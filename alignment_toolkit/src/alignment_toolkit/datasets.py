"""Canonical alignment data and built-in Hugging Face dataset catalog."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from .registry import register_dataset


@dataclass(frozen=True)
class AlignmentExample:
    example_id: str
    kind: str  # instruction, preference, prompt, reward
    messages: tuple[dict[str, str], ...] = ()
    response: str | None = None
    chosen: str | None = None
    rejected: str | None = None
    reward: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DatasetDefinition:
    name: str
    kind: str
    adapter: str
    source: dict[str, Any]
    mapping: dict[str, Any] = field(default_factory=dict)
    options: dict[str, Any] = field(default_factory=dict)


class AlignmentDatasetAdapter(Protocol):
    def load(self, definition: DatasetDefinition):
        ...


def _hf(name: str, kind: str, adapter: str, dataset_id: str, split: str, mapping=None, **options):
    definition = DatasetDefinition(
        name=name,
        kind=kind,
        adapter=adapter,
        source={"type": "huggingface", "id": dataset_id, "config": "default", "split": split},
        mapping=dict(mapping or {}),
        options=options,
    )
    register_dataset(name, definition)


_hf("ultrachat_sft", "instruction", "hf_instruction", "HuggingFaceH4/ultrachat_200k", "train_sft", {"messages": "messages"})
_hf("ultrachat_prompts", "prompt", "hf_prompt", "HuggingFaceH4/ultrachat_200k", "train_gen", {"prompt": "prompt", "messages": "messages"})
_hf("ultrafeedback_dpo", "preference", "hf_preference", "HuggingFaceH4/ultrafeedback_binarized", "train_prefs", {"prompt": "prompt", "chosen": "chosen", "rejected": "rejected", "messages": "messages"})
_hf("ultrafeedback_prompts", "prompt", "hf_prompt", "HuggingFaceH4/ultrafeedback_binarized", "train_gen", {"prompt": "prompt"})
_hf("hh_rlhf", "preference", "hf_transcript_preference", "Anthropic/hh-rlhf", "train", {"chosen": "chosen", "rejected": "rejected"})
_hf("oasst1", "instruction", "hf_message_tree", "OpenAssistant/oasst1", "train", {"text": "text", "role": "role", "parent_id": "parent_id"})
_hf("slimorca", "instruction", "hf_conversations", "Open-Orca/SlimOrca", "train", {"conversations": "conversations"})
_hf("alpaca_cleaned", "instruction", "hf_alpaca", "yahma/alpaca-cleaned", "train", {"instruction": "instruction", "input": "input", "output": "output"})
_hf("helpsteer2", "reward", "hf_scalar_reward", "nvidia/HelpSteer2", "train", {"prompt": "prompt", "response": "response"}, score_fields=["helpfulness", "correctness", "coherence", "complexity", "verbosity"])
_hf("pku_saferlhf", "preference", "hf_safety_preference", "PKU-Alignment/PKU-SafeRLHF", "train", {"prompt": "prompt", "response_0": "response_0", "response_1": "response_1"})
