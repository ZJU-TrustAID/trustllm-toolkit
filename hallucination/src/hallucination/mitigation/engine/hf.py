from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from hallucination.mitigation.core.contracts import GenerationOutput, GenerationRequest
from hallucination.mitigation.core.errors import (
    GenerationError,
    MissingDependencyError,
    UnsupportedModelError,
)
from hallucination.mitigation.core.model_config import ModelConfig
from hallucination.mitigation.core.sampling import sample_token
from hallucination.mitigation.engine.hooks import HookManager, find_decoder_layers


@dataclass
class EngineState:
    input_ids: list[int]
    attention_mask: Any
    past_key_values: Any
    generated_ids: list[int] = field(default_factory=list)


@dataclass
class PrefillOutput:
    logits: Any
    hidden_states: Any
    state: EngineState


@dataclass
class StepOutput:
    logits: Any
    hidden_states: Any


@dataclass
class FullForward:
    logits: Any
    hidden_states: Any


class HFEngine:
    """A small token-by-token inference engine for white-box methods."""

    def __init__(
        self, model: Any, tokenizer: Any, *, device: str | None = None
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = device or self._infer_device(model)
        self.model.eval()

    @classmethod
    def from_pretrained(cls, config: ModelConfig) -> HFEngine:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise MissingDependencyError(
                "Install the HF extra first: python -m pip install -e '.[hf]'"
            ) from exc

        dtype = None
        if config.dtype and config.dtype != "auto":
            if not hasattr(torch, config.dtype):
                raise UnsupportedModelError(f"unknown torch dtype: {config.dtype}")
            dtype = getattr(torch, config.dtype)

        common = {
            "revision": config.revision,
            "local_files_only": config.local_files_only,
            "trust_remote_code": config.trust_remote_code,
        }
        tokenizer = AutoTokenizer.from_pretrained(config.model, **common)
        if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        load_kwargs: dict[str, Any] = dict(common)
        if dtype is not None:
            import transformers

            major = int(transformers.__version__.split(".", 1)[0])
            load_kwargs["dtype" if major >= 5 else "torch_dtype"] = dtype
        if config.load_in_4bit:
            try:
                from transformers import BitsAndBytesConfig
            except ImportError as exc:
                raise MissingDependencyError(
                    "load_in_4bit requires transformers with bitsandbytes support"
                ) from exc
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=dtype or torch.float16,
                bnb_4bit_quant_type="nf4",
            )
            load_kwargs["device_map"] = "auto"
        model = AutoModelForCausalLM.from_pretrained(config.model, **load_kwargs)
        if not config.load_in_4bit:
            if config.device:
                model = model.to(config.device)
            elif torch.cuda.is_available():
                model = model.to("cuda")
        model.eval()
        return cls(model=model, tokenizer=tokenizer, device=config.device)

    @staticmethod
    def _infer_device(model: Any) -> str | None:
        try:
            return str(next(model.parameters()).device)
        except (AttributeError, StopIteration):
            return None

    @property
    def eos_token_id(self) -> int | None:
        value = getattr(self.tokenizer, "eos_token_id", None)
        if isinstance(value, list):
            return int(value[0]) if value else None
        return int(value) if value is not None else None

    @property
    def hidden_size(self) -> int:
        config = getattr(self.model, "config", None)
        for name in ("hidden_size", "n_embd", "d_model"):
            value = getattr(config, name, None)
            if value:
                return int(value)
        raise UnsupportedModelError("cannot determine hidden size from model config")

    @property
    def num_layers(self) -> int:
        config = getattr(self.model, "config", None)
        for name in ("num_hidden_layers", "n_layer", "num_layers"):
            value = getattr(config, name, None)
            if value:
                return int(value)
        return len(find_decoder_layers(self.model))

    @property
    def num_heads(self) -> int:
        config = getattr(self.model, "config", None)
        for name in ("num_attention_heads", "n_head", "num_heads"):
            value = getattr(config, name, None)
            if value:
                return int(value)
        raise UnsupportedModelError("cannot determine number of attention heads")

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_heads

    def fingerprint(self) -> dict[str, Any]:
        config = getattr(self.model, "config", None)
        return {
            "model": getattr(config, "_name_or_path", None),
            "hidden_size": self.hidden_size,
            "num_hidden_layers": self.num_layers,
            "num_attention_heads": self.num_heads,
            "num_heads": self.num_heads,
            "head_dim": self.head_dim,
        }

    def encode(self, text: str) -> list[int]:
        encoded = self.tokenizer(text, return_tensors="pt", add_special_tokens=True)
        return [int(item) for item in encoded["input_ids"][0].tolist()]

    def activation_index(self, text: str) -> int:
        """Return the last non-special token position (fallback: -1)."""
        ids = self.encode(text)
        if not ids:
            return -1
        specials = {
            value
            for value in (
                getattr(self.tokenizer, "eos_token_id", None),
                getattr(self.tokenizer, "pad_token_id", None),
                getattr(self.tokenizer, "bos_token_id", None),
            )
            if value is not None and not isinstance(value, list)
        }
        index = len(ids) - 1
        while index > 0 and ids[index] in specials:
            index -= 1
        return index

    def decode(self, token_ids: list[int]) -> str:
        return self.tokenizer.decode(token_ids, skip_special_tokens=True).strip()

    def _to_device(self, batch: Any) -> Any:
        if self.device is None:
            return batch
        return {
            key: value.to(self.device) if hasattr(value, "to") else value
            for key, value in batch.items()
        }

    def prefill(
        self, input_ids: list[int] | str, *, output_hidden_states: bool = False
    ) -> PrefillOutput:
        import torch

        if isinstance(input_ids, str):
            input_ids = self.encode(input_ids)
        ids = torch.tensor([input_ids], dtype=torch.long, device=self.device)
        mask = torch.ones_like(ids)
        with torch.inference_mode():
            outputs = self.model(
                input_ids=ids,
                attention_mask=mask,
                use_cache=True,
                output_hidden_states=output_hidden_states,
            )
        state = EngineState(
            input_ids=list(input_ids),
            attention_mask=mask,
            past_key_values=outputs.past_key_values,
        )
        logits = outputs.logits[0, -1, :].detach()
        hidden = outputs.hidden_states if output_hidden_states else None
        return PrefillOutput(logits=logits, hidden_states=hidden, state=state)

    def step(
        self, state: EngineState, token_id: int, *, output_hidden_states: bool = False
    ) -> StepOutput:
        import torch

        ids = torch.tensor([[int(token_id)]], dtype=torch.long, device=self.device)
        mask = torch.cat(
            [
                state.attention_mask,
                torch.ones(
                    (1, 1), dtype=state.attention_mask.dtype, device=self.device
                ),
            ],
            dim=-1,
        )
        with torch.inference_mode():
            outputs = self.model(
                input_ids=ids,
                attention_mask=mask,
                past_key_values=state.past_key_values,
                use_cache=True,
                output_hidden_states=output_hidden_states,
            )
        state.past_key_values = outputs.past_key_values
        state.input_ids.append(int(token_id))
        state.attention_mask = mask
        state.generated_ids.append(int(token_id))
        logits = outputs.logits[0, -1, :].detach()
        hidden = outputs.hidden_states if output_hidden_states else None
        return StepOutput(logits=logits, hidden_states=hidden)

    def forward_text(
        self, text: str, *, output_hidden_states: bool = True
    ) -> FullForward:
        import torch

        encoded = self.tokenizer(text, return_tensors="pt", add_special_tokens=True)
        encoded = self._to_device(encoded)
        with torch.inference_mode():
            outputs = self.model(
                **encoded, use_cache=False, output_hidden_states=output_hidden_states
            )
        return FullForward(
            logits=outputs.logits.detach(), hidden_states=outputs.hidden_states
        )

    def project_last(self, hidden: Any) -> np.ndarray:
        import torch

        head = self.model.get_output_embeddings()
        if head is None:
            raise UnsupportedModelError("model has no output embedding/lm_head")
        if not torch.is_tensor(hidden):
            raise TypeError("hidden must be a torch tensor")
        if hidden.dim() == 3:
            hidden = hidden[0, -1, :]
        elif hidden.dim() == 2:
            hidden = hidden[-1, :]
        with torch.inference_mode():
            logits = head(hidden.unsqueeze(0))
        return logits[0].detach().float().cpu().numpy()

    def project_layer(self, hidden_states: Any, layer: int) -> np.ndarray:
        if layer < 0 or layer >= len(hidden_states):
            raise UnsupportedModelError(
                f"layer {layer} out of range for hidden_states of length {len(hidden_states)}"
            )
        return self.project_last(hidden_states[layer])

    def generate_text(
        self,
        request: GenerationRequest,
        *,
        hook_manager: HookManager | None = None,
        hook_prefill: bool = False,
    ) -> GenerationOutput:
        rng = np.random.default_rng(request.seed)
        started = time.perf_counter()
        prompt_ids = self.encode(request.prompt)
        if not prompt_ids:
            raise GenerationError("tokenizer produced an empty prompt")
        if hook_manager is not None:
            hook_manager.active = hook_prefill
        prefill = self.prefill(prompt_ids)
        if hook_manager is not None:
            hook_manager.active = True
        logits = prefill.logits
        state = prefill.state
        generated: list[int] = []
        finish_reason = "length"
        try:
            for _ in range(request.max_new_tokens):
                token = sample_token(
                    logits.detach().float().cpu().numpy(),
                    do_sample=request.do_sample,
                    temperature=request.temperature,
                    top_p=request.top_p,
                    top_k=request.top_k,
                    repetition_penalty=request.repetition_penalty,
                    generated_ids=generated,
                    rng=rng,
                )
                if self.eos_token_id is not None and token == self.eos_token_id:
                    finish_reason = "eos"
                    break
                out = self.step(state, token)
                generated.append(token)
                logits = out.logits
                text_so_far = self.decode(generated)
                if request.stop and any(
                    text_so_far.endswith(item) for item in request.stop
                ):
                    finish_reason = "stop"
                    break
        finally:
            if hook_manager is not None:
                hook_manager.active = False
        text = self.decode(generated)
        elapsed = time.perf_counter() - started
        cost = {
            "latency_seconds": round(elapsed, 4),
            "prompt_tokens": len(prompt_ids),
            "generated_tokens": len(generated),
            "forward_passes": 1 + len(generated),
        }
        return GenerationOutput(
            text=text,
            token_ids=prompt_ids + generated,
            generated_ids=generated,
            finish_reason=finish_reason,
            cost=cost,
        )
