from __future__ import annotations

from collections.abc import Callable
from typing import Any

from hallucination.detection.backends.hf_whitebox import WhiteBoxGeneration


class CallableLLMBackend:
    """Wrap plain Python callables as black-box LLM backends."""

    def __init__(self, generate_fn: Callable[..., str]) -> None:
        self.generate_fn = generate_fn

    def generate(self, prompt: str, **kwargs: object) -> str:
        return self.generate_fn(prompt, **kwargs)

    def sample(self, prompt: str, n: int, **kwargs: object) -> list[str]:
        return [self.generate(prompt, **kwargs) for _ in range(n)]


class OpenAIResponsesBackend:
    """Black-box backend for OpenAI's Responses API."""

    def __init__(self, model: str, *, client: Any | None = None, **default_kwargs: object) -> None:
        if client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise ImportError("Install the OpenAI extra first: python -m pip install -e '.[openai]'") from exc
            client = OpenAI()
        self.client = client
        self.model = model
        self.default_kwargs = default_kwargs

    def generate(self, prompt: str, **kwargs: object) -> str:
        request = {**self.default_kwargs, **kwargs}
        response = self.client.responses.create(model=self.model, input=prompt, **request)
        return self._extract_output_text(response)

    def sample(self, prompt: str, n: int, **kwargs: object) -> list[str]:
        return [self.generate(prompt, **kwargs) for _ in range(n)]

    @staticmethod
    def _extract_output_text(response: Any) -> str:
        output_text = getattr(response, "output_text", None)
        if output_text is not None:
            return str(output_text)
        if isinstance(response, dict) and "output_text" in response:
            return str(response["output_text"])
        raise ValueError("Could not extract output text from OpenAI response")


class OpenAIChatBackend:
    """Generation backend over the Chat Completions API with token logprobs.

    Uses ``chat.completions`` rather than the Responses API because it exposes
    per-token logprobs, and works with any OpenAI-compatible server via the
    client's ``base_url`` (vLLM, Ollama, and similar). It provides:

    - ``generate`` / ``sample`` for black-box sampling pipelines;
    - ``generate_detection_example`` returning token logprobs and top-k
      distributions so ``WhiteBoxGenerationPipeline`` (MTP, LNTP, token
      entropy) works over the API;
    - ``next_token_logprobs`` best-effort candidate lookup from the sampled
      step's top-logprob list for P(True).

    Hidden states are never available over an API; EigenScore and Effective
    Rank require a local model. ``top_logprobs`` is capped at 20 by the API,
    so token entropy is a top-k approximation and ``next_token_logprobs``
    only returns candidates present in that window.
    """

    MAX_TOP_LOGPROBS = 20

    def __init__(self, model: str, *, client: Any | None = None, **default_kwargs: object) -> None:
        if client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise ImportError("Install the OpenAI extra first: python -m pip install -e '.[openai]'") from exc
            client = OpenAI()
        self.client = client
        self.model = model
        self.model_name = model
        self.default_kwargs = default_kwargs

    def generate(self, prompt: str, **kwargs: object) -> str:
        response = self._create(prompt=prompt, **kwargs)
        choice = response.choices[0]
        content = getattr(choice.message, "content", None)
        if isinstance(content, list):  # some servers return content parts
            content = "".join(getattr(part, "text", "") for part in content)
        if content is None:
            raise ValueError("Could not extract output text from chat completion")
        return str(content)

    def sample(self, prompt: str, n: int, **kwargs: object) -> list[str]:
        return [self.generate(prompt, **kwargs) for _ in range(n)]

    def generate_with_scores(
        self,
        prompt: str,
        *,
        max_new_tokens: int = 64,
        top_k: int = 5,
        **kwargs: object,
    ) -> WhiteBoxGeneration:
        request_top_k = max(1, min(int(top_k), self.MAX_TOP_LOGPROBS))
        response = self._create(
            prompt=prompt,
            max_tokens=int(max_new_tokens),
            logprobs=True,
            top_logprobs=request_top_k,
            **kwargs,
        )
        choice = response.choices[0]
        token_logprobs_payload = getattr(choice, "logprobs", None)
        entries = getattr(token_logprobs_payload, "content", None) if token_logprobs_payload is not None else None
        if not entries:
            raise ValueError(
                "the chat completion did not return token logprobs; check that the "
                "model and server support logprobs=True / top_logprobs"
            )

        tokens: list[str] = []
        token_logprobs: list[float] = []
        top_logprobs: list[dict[str, float]] = []
        for entry in entries:
            tokens.append(str(entry.token))
            token_logprobs.append(float(entry.logprob))
            step: dict[str, float] = {}
            for item in getattr(entry, "top_logprobs", None) or []:
                key = str(item.token)
                if key in step:  # duplicate token strings get a stable suffix
                    key = f"{key}#{len(step)}"
                step[key] = float(item.logprob)
            top_logprobs.append(step)
        content = getattr(choice.message, "content", None)
        if isinstance(content, list):
            content = "".join(getattr(part, "text", "") for part in content)
        response_text = str(content) if content is not None else "".join(tokens)

        return WhiteBoxGeneration(
            prompt=prompt,
            response=response_text,
            token_ids=[],
            tokens=tokens,
            token_logprobs=token_logprobs,
            top_logprobs=top_logprobs,
            top_logprob_entries=None,
            token_is_special=None,
            hidden_states=None,
            metadata={
                "backend": "openai-chat",
                "model_name": self.model,
                "max_new_tokens": int(max_new_tokens),
                "top_k": request_top_k,
                "token_ids_unavailable": True,
            },
        )

    def generate_detection_example(self, prompt: str, **kwargs: object) -> Any:
        return self.generate_with_scores(prompt, **kwargs).to_detection_example()

    def next_token_logprobs(self, prompt: str, candidates: list[str]) -> dict[str, float]:
        """Return logprobs for candidates present in the next-token top-k window.

        Candidates outside the API's top-logprob list are omitted from the
        result; callers such as ``PTrueScorer`` handle the missing case.
        """

        if not candidates:
            raise ValueError("candidates must not be empty")
        response = self._create(
            prompt=prompt,
            max_tokens=1,
            logprobs=True,
            top_logprobs=self.MAX_TOP_LOGPROBS,
        )
        choice = response.choices[0]
        token_logprobs_payload = getattr(choice, "logprobs", None)
        entries = getattr(token_logprobs_payload, "content", None) if token_logprobs_payload is not None else None
        if not entries:
            raise ValueError(
                "the chat completion did not return token logprobs; check that the "
                "model and server support logprobs=True / top_logprobs"
            )
        window: dict[str, float] = {}
        for entry in entries:
            window[str(entry.token)] = float(entry.logprob)
            for item in getattr(entry, "top_logprobs", None) or []:
                key = str(item.token)
                existing = window.get(key)
                value = float(item.logprob)
                if existing is None or value > existing:
                    window[key] = value
        return {candidate: window[candidate] for candidate in candidates if candidate in window}

    def _create(self, *, prompt: str, **kwargs: object) -> Any:
        request = {**self.default_kwargs, **kwargs}
        return self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            **request,
        )
