from __future__ import annotations

from typing import Any

from hallucination.mitigation.backends.base import LLMBackend
from hallucination.mitigation.core.contracts import GenerationRequest
from hallucination.mitigation.core.prompt import apply_chat_template


class HFBackend(LLMBackend):
    def __init__(self, engine: Any, *, seed: int = 0) -> None:
        self.engine = engine
        self.seed = seed

    def generate(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        prompt = apply_chat_template(
            self.engine.tokenizer, messages, add_generation_prompt=True
        )
        request = GenerationRequest(
            prompt=prompt,
            max_new_tokens=int(kwargs.get("max_new_tokens", 128)),
            do_sample=bool(kwargs.get("do_sample", False)),
            temperature=float(kwargs.get("temperature", 1.0)),
            top_p=float(kwargs.get("top_p", 1.0)),
            top_k=int(kwargs.get("top_k", 0)),
            repetition_penalty=float(kwargs.get("repetition_penalty", 1.0)),
            seed=int(kwargs.get("seed", self.seed)),
        )
        return self.engine.generate_text(request).text
