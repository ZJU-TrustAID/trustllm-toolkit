from __future__ import annotations

import pytest

from hallucination.mitigation.core.prompt import build_cad_prompts, render_pair


class FakeTokenizer:
    chat_template = "fake"

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        text = " | ".join(f"{item['role']}:{item['content']}" for item in messages)
        return text + (" | assistant:" if add_generation_prompt else "")


def test_cad_prompts_differ_only_by_context() -> None:
    tokenizer = FakeTokenizer()
    with_context, without_context = build_cad_prompts(
        "Q?", "C.", tokenizer, template="chat"
    )
    assert "C." in with_context
    assert "C." not in without_context
    assert with_context.endswith("assistant:")
    assert without_context.endswith("assistant:")


def test_cad_requires_context() -> None:
    with pytest.raises(ValueError):
        build_cad_prompts("Q?", "", None)


def test_render_pair_plain_and_chat() -> None:
    assert "Q\nA" in render_pair("Q", "A")
    rendered = render_pair("Q", "A", FakeTokenizer(), chat=True)
    assert "user:Q" in rendered
    assert "assistant:A" in rendered
