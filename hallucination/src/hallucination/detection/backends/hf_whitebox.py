from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from hallucination.detection.core.results import DetectionExample


@dataclass
class WhiteBoxGeneration:
    prompt: str
    response: str
    token_ids: list[int]
    tokens: list[str]
    token_logprobs: list[float]
    top_logprobs: list[dict[str, float]]
    top_logprob_entries: list[list[tuple[int, str, float]]] | None = None
    token_is_special: list[bool] | None = None
    hidden_states: Any | None = None
    metadata: dict[str, Any] | None = None

    def to_detection_example(self) -> DetectionExample:
        return DetectionExample(
            prompt=self.prompt,
            response=self.response,
            token_logprobs=self.token_logprobs,
            top_logprobs=self.top_logprobs,
            top_logprob_entries=self.top_logprob_entries,
            tokens=self.tokens,
            token_is_special=self.token_is_special,
            hidden_states=self.hidden_states,
            metadata={
                "token_ids": self.token_ids,
                "tokens": self.tokens,
                **(self.metadata or {}),
            },
        )


class HuggingFaceWhiteBoxBackend:
    """Extract generated text, token logprobs, top-k logprobs, and optional hidden states from local Hugging Face causal LMs."""

    def __init__(self, model: object, tokenizer: object, *, device: str | None = None) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = device or self._infer_device(model)

    @classmethod
    def from_pretrained(cls, model_name_or_path: str, *, device: str | None = None, **kwargs: object) -> HuggingFaceWhiteBoxBackend:
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise ImportError("Install the whitebox extra first: python -m pip install -e '.[whitebox]'") from exc

        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, **kwargs)
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **kwargs)
        if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token
        if device is not None:
            model = model.to(device)
        model.eval()
        return cls(model=model, tokenizer=tokenizer, device=device)

    def generate_with_scores(
        self,
        prompt: str,
        *,
        max_new_tokens: int = 64,
        top_k: int = 5,
        include_hidden_states: bool = False,
        hidden_state_layer: int = -1,
        hidden_state_pooling: str = "mean",
        **generate_kwargs: object,
    ) -> WhiteBoxGeneration:
        try:
            import torch
        except ImportError as exc:
            raise ImportError("Install the whitebox extra first: python -m pip install -e '.[whitebox]'") from exc

        encoded = self.tokenizer(prompt, return_tensors="pt")
        encoded = self._move_batch_to_device(encoded, self.device)
        prompt_length = int(encoded["input_ids"].shape[-1])

        with torch.no_grad():
            output = self.model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                return_dict_in_generate=True,
                output_scores=True,
                **generate_kwargs,
            )

        sequences = output.sequences
        generated_ids_tensor = sequences[0, prompt_length:]
        generated_ids = [int(token_id) for token_id in generated_ids_tensor.detach().cpu().tolist()]
        response = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        tokens = self.tokenizer.convert_ids_to_tokens(generated_ids, skip_special_tokens=False)
        special_ids = set(getattr(self.tokenizer, "all_special_ids", []) or [])
        token_is_special = [token_id in special_ids for token_id in generated_ids]

        token_logprobs: list[float] = []
        top_logprobs: list[dict[str, float]] = []
        top_logprob_entries: list[list[tuple[int, str, float]]] = []
        for token_id, logits in zip(generated_ids, output.scores, strict=False):
            log_probs = torch.log_softmax(logits[0], dim=-1)
            token_logprobs.append(float(log_probs[token_id].detach().cpu().item()))
            values, indices = torch.topk(log_probs, k=min(top_k, log_probs.shape[-1]))
            step_top: dict[str, float] = {}
            step_entries: list[tuple[int, str, float]] = []
            for value, index in zip(values.detach().cpu().tolist(), indices.detach().cpu().tolist(), strict=False):
                decoded = self.tokenizer.decode([int(index)], skip_special_tokens=False)
                key = decoded if decoded not in step_top else f"{decoded}#{int(index)}"
                step_top[key] = float(value)
                step_entries.append((int(index), decoded, float(value)))
            top_logprobs.append(step_top)
            top_logprob_entries.append(step_entries)

        hidden_states = None
        if include_hidden_states:
            hidden_states = self._extract_generated_hidden_states(
                sequences=sequences,
                prompt_length=prompt_length,
                layer=hidden_state_layer,
                pooling=hidden_state_pooling,
            )

        return WhiteBoxGeneration(
            prompt=prompt,
            response=response,
            token_ids=generated_ids,
            tokens=tokens,
            token_logprobs=token_logprobs,
            top_logprobs=top_logprobs,
            top_logprob_entries=top_logprob_entries,
            token_is_special=token_is_special,
            hidden_states=hidden_states,
            metadata={
                "backend": "huggingface",
                "model_name": self._model_name(),
                "token_logprobs": token_logprobs,
                "max_new_tokens": max_new_tokens,
                "top_k": top_k,
                "include_hidden_states": include_hidden_states,
                "hidden_state_layer": hidden_state_layer,
                "hidden_state_pooling": hidden_state_pooling,
            },
        )

    def generate_detection_example(self, prompt: str, **kwargs: object) -> DetectionExample:
        return self.generate_with_scores(prompt, **kwargs).to_detection_example()

    def next_token_logprobs(self, prompt: str, candidates: list[str]) -> dict[str, float]:
        """Return the log-probability of the first token of each candidate continuation."""

        try:
            import torch
        except ImportError as exc:
            raise ImportError("Install the whitebox extra first: python -m pip install -e '.[whitebox]'") from exc
        if not candidates:
            raise ValueError("candidates must not be empty")

        encoded = self.tokenizer(prompt, return_tensors="pt")
        encoded = self._move_batch_to_device(encoded, self.device)
        with torch.no_grad():
            logits = self.model(**encoded).logits[0, -1]
            log_probs = torch.log_softmax(logits, dim=-1)
        values: dict[str, float] = {}
        for candidate in candidates:
            token_ids = self.tokenizer.encode(candidate, add_special_tokens=False)
            if not token_ids:
                raise ValueError(f"candidate {candidate!r} tokenized to an empty sequence")
            values[candidate] = float(log_probs[int(token_ids[0])].detach().cpu().item())
        return values

    def generate_hidden_states(
        self,
        prompt: str,
        *,
        n_samples: int = 5,
        hidden_state_layer: int = -1,
        hidden_state_pooling: str = "mean",
        **generate_kwargs: object,
    ) -> np.ndarray:
        """Collect one hidden-state vector per independent generation.

        Returns an array shaped ``[n_samples, hidden_dim]`` suitable for
        EigenScore / EffectiveRank. Only poolings that produce one vector per
        generation are accepted.
        """

        if n_samples < 2:
            raise ValueError("n_samples must be at least 2")
        if hidden_state_pooling == "none":
            raise ValueError("hidden_state_pooling='none' returns a token matrix per generation; use 'mean' or 'last_token'")
        generate_kwargs.setdefault("do_sample", True)
        if generate_kwargs.get("do_sample") and "temperature" not in generate_kwargs:
            generate_kwargs["temperature"] = 1.0
        vectors = []
        for _ in range(n_samples):
            generation = self.generate_with_scores(
                prompt,
                include_hidden_states=True,
                hidden_state_layer=hidden_state_layer,
                hidden_state_pooling=hidden_state_pooling,
                **generate_kwargs,
            )
            if generation.hidden_states is None:
                raise RuntimeError("hidden states were not produced")
            matrix = np.asarray(generation.hidden_states, dtype=float)
            if matrix.ndim != 2 or matrix.shape[0] != 1:
                raise ValueError("pooled hidden states must have shape [1, hidden_dim]")
            vectors.append(matrix[0])
        return np.vstack(vectors)

    def _extract_generated_hidden_states(self, *, sequences: object, prompt_length: int, layer: int, pooling: str) -> Any:
        try:
            import torch
        except ImportError as exc:
            raise ImportError("Install the whitebox extra first: python -m pip install -e '.[whitebox]'") from exc

        if pooling not in {"none", "mean", "last_token"}:
            raise ValueError("hidden_state_pooling must be 'none', 'mean', or 'last_token'")

        with torch.no_grad():
            model_output = self.model(sequences.to(self.device) if self.device else sequences, output_hidden_states=True)

        states = model_output.hidden_states[layer][0, prompt_length:, :]
        if states.shape[0] == 0:
            raise ValueError("no generated tokens to pool for hidden states")
        if pooling == "mean":
            return states.mean(dim=0, keepdim=True).detach().cpu().numpy()
        if pooling == "last_token":
            return states[-1:, :].detach().cpu().numpy()
        return states.detach().cpu().numpy()

    def _model_name(self) -> str | None:
        config = getattr(self.model, "config", None)
        return getattr(config, "_name_or_path", None)

    @staticmethod
    def _infer_device(model: object) -> str | None:
        try:
            parameter = next(model.parameters())
        except (AttributeError, StopIteration):
            return None
        return str(parameter.device)

    @staticmethod
    def _move_batch_to_device(batch: object, device: str | None) -> object:
        if device is None:
            return batch
        if hasattr(batch, "to"):
            return batch.to(device)
        return {key: value.to(device) if hasattr(value, "to") else value for key, value in batch.items()}
