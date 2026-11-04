from __future__ import annotations

from typing import Any


def has_chat_template(tokenizer: Any) -> bool:
    return bool(getattr(tokenizer, "chat_template", None)) and hasattr(
        tokenizer, "apply_chat_template"
    )


def apply_chat_template(
    tokenizer: Any,
    messages: list[dict[str, str]],
    *,
    add_generation_prompt: bool = True,
) -> str:
    if has_chat_template(tokenizer):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )
    parts = []
    for message in messages:
        role = message.get("role", "user")
        content = message.get("content", "")
        parts.append(f"{role}: {content}")
    if add_generation_prompt:
        parts.append("assistant:")
    return "\n".join(parts)


def build_cad_prompts(
    prompt: str,
    context: str,
    tokenizer: Any = None,
    *,
    template: str = "plain",
) -> tuple[str, str]:
    """Return (context_view, no_context_view)."""
    if not context:
        raise ValueError("CAD requires a non-empty context")
    if template not in {"plain", "chat"}:
        raise ValueError("context_template must be 'plain' or 'chat'")
    if template == "plain" or tokenizer is None or not has_chat_template(tokenizer):
        return f"{context.strip()}\n\n{prompt.strip()}", prompt.strip()
    messages_with = [
        {
            "role": "system",
            "content": f"Use the following context to answer accurately:\n{context.strip()}",
        },
        {"role": "user", "content": prompt.strip()},
    ]
    messages_without = [{"role": "user", "content": prompt.strip()}]
    return (
        apply_chat_template(tokenizer, messages_with, add_generation_prompt=True),
        apply_chat_template(tokenizer, messages_without, add_generation_prompt=True),
    )


def render_pair(
    prompt: str,
    answer: str,
    tokenizer: Any = None,
    *,
    chat: bool = False,
) -> str:
    if chat and tokenizer is not None and has_chat_template(tokenizer):
        return apply_chat_template(
            tokenizer,
            [
                {"role": "user", "content": prompt.strip()},
                {"role": "assistant", "content": answer.strip()},
            ],
            add_generation_prompt=False,
        )
    return f"{prompt.rstrip()}\n{answer.strip()}"
