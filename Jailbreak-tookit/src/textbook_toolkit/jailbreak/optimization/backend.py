"""Shared Hugging Face causal-LM backend; no model downloads on import."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator

import torch
import torch.nn.functional as F

from ..types import Message, Prompt, Step, require_count
from .types import OptimizationRequest, OptimizationResult, TraceStep


@dataclass(frozen=True)
class Layout:
    request: OptimizationRequest
    control: str
    input_ids: torch.Tensor
    target_ids: torch.Tensor
    start: int
    stop: int
    # A BPE token may straddle a text boundary. These characters must stay fixed.
    left_edge: str
    right_edge: str

    @property
    def control_ids(self) -> torch.Tensor:
        return self.input_ids[self.start : self.stop]


class CausalLM:
    """Inject an existing model/tokenizer, or load one with from_pretrained.

    Fast tokenizer offsets locate the editable span in the *rendered* prompt.
    Every accepted GCG candidate must survive complete text re-tokenization.
    The model remains on its existing device unless a device is specified.
    """

    def __init__(
        self,
        model,
        tokenizer,
        *,
        device=None,
        chat: bool | None = None,
        system: str | None = None,
    ):
        if not getattr(tokenizer, "is_fast", False):
            raise ValueError("Optimization requires a fast tokenizer with offset mappings.")
        if getattr(model.config, "is_encoder_decoder", False):
            raise ValueError("Use a decoder-only causal language model.")
        if getattr(model, "is_quantized", False):
            raise ValueError("Use unquantized weights; quantized gradients are not supported.")
        if device is not None:
            model.to(device)
        self.model, self.tokenizer = model, tokenizer
        self.device = model.get_input_embeddings().weight.device
        if any(p.device != self.device for p in model.parameters()):
            raise ValueError(
                "This backend requires a single-device model (no device_map shards)."
            )
        self.chat = bool(tokenizer.chat_template) if chat is None else chat
        if self.chat and not tokenizer.chat_template:
            raise ValueError("chat=True requires tokenizer.chat_template.")
        self.system = system
        self.vocab_size = min(
            len(tokenizer),
            model.get_input_embeddings().weight.shape[0],
            model.get_output_embeddings().weight.shape[0],
        )

    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        *,
        device="auto",
        dtype="float32",
        chat: bool | None = None,
        system: str | None = None,
        revision: str | None = None,
        local_files_only: bool = False,
    ):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if device == "auto":
            device = (
                "cuda"
                if torch.cuda.is_available()
                else "mps"
                if torch.backends.mps.is_available()
                else "cpu"
            )
        if dtype not in ("float32", "float16", "bfloat16"):
            raise ValueError("dtype must be float32, float16 or bfloat16.")
        kwargs = dict(
            revision=revision, local_files_only=local_files_only, trust_remote_code=False
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, **kwargs)
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=getattr(torch, dtype), **kwargs
        )
        return cls(model, tokenizer, device=device, chat=chat, system=system)

    @property
    def embedding(self) -> torch.Tensor:
        return self.model.get_input_embeddings().weight[: self.vocab_size].detach()

    @contextmanager
    def session(self) -> Iterator[None]:
        """Freeze weights for input gradients; restore caller-owned model state."""
        parameters = list(self.model.parameters())
        flags = [p.requires_grad for p in parameters]
        modes = [(module, module.training) for module in self.model.modules()]
        try:
            self.model.eval()
            for p in parameters:
                p.requires_grad_(False)
            with torch.enable_grad():
                yield
        finally:
            for p, flag in zip(parameters, flags):
                p.requires_grad_(flag)
            for module, mode in modes:
                module.training = mode

    def messages(self, content: str) -> tuple[Message, ...]:
        user = (Message("user", content),)
        return (Message("system", self.system),) + user if self.system else user

    def render(self, content: str) -> str:
        if self.chat:
            return self.tokenizer.apply_chat_template(
                [m.to_dict() for m in self.messages(content)],
                tokenize=False,
                add_generation_prompt=True,
            )
        return (self.system + "\n" if self.system else "") + content

    def _encode(self, rendered: str):
        encoded = self.tokenizer(
            rendered, add_special_tokens=False, return_offsets_mapping=True
        )
        ids, offsets = encoded["input_ids"], encoded["offset_mapping"]
        if not self.chat:
            bos = self.tokenizer.bos_token_id
            if bos is None:
                bos = self.tokenizer.eos_token_id
            if bos is None:
                raise ValueError("Plain completion models need a BOS or EOS context token.")
            ids, offsets = [bos] + ids, [(0, 0)] + offsets
        if not ids or max(ids) >= self.vocab_size:
            raise ValueError("Tokenizer IDs are incompatible with the model vocabulary.")
        return torch.tensor(ids, device=self.device, dtype=torch.long), offsets

    def encode_prompt(self, content: str) -> torch.Tensor:
        return self._encode(self.render(content))[0]

    def encode_text(self, text: str) -> torch.Tensor:
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        if not ids or max(ids) >= self.vocab_size:
            raise ValueError("Text must tokenize to nonempty, in-vocabulary tokens.")
        return torch.tensor(ids, dtype=torch.long, device=self.device)

    def decode(self, ids) -> str:
        return self.tokenizer.decode(
            ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
        )

    def prepare(self, request: OptimizationRequest, control: str) -> Layout:
        marker = "__TEXTBOOK_EDITABLE_CONTROL_7d641__"
        if marker in request.compose(control):
            raise ValueError("Input contains the internal span marker.")
        marked = self.render(request.compose(marker))
        if marked.count(marker) != 1:
            raise ValueError("Chat template must preserve user content exactly once.")
        before, after = marked.split(marker)
        rendered = self.render(request.compose(control))
        if rendered != before + control + after:
            raise ValueError("Chat template transforms the editable content.")
        ids, offsets = self._encode(rendered)
        a, b = len(before), len(before) + len(control)
        overlaps = [i for i, (lo, hi) in enumerate(offsets) if hi > a and lo < b]
        if not overlaps:
            raise ValueError("Control has no editable tokens.")
        start, stop = overlaps[0], overlaps[-1] + 1
        if start == 0:
            raise ValueError("An editable span needs at least one preceding context token.")
        if any(t in self.tokenizer.all_special_ids for t in ids[start:stop].tolist()):
            raise ValueError("The editable control may not contain special tokens.")
        target_ids = self.encode_text(request.target)
        self.check_length(len(ids) + len(target_ids))
        return Layout(
            request,
            control,
            ids,
            target_ids,
            start,
            stop,
            rendered[offsets[start][0] : a],
            rendered[b : offsets[stop - 1][1]],
        )

    def candidate(self, layout: Layout, tokens: torch.Tensor, *, strict=True) -> Layout | None:
        text = self.decode(tokens.tolist())
        if not text.startswith(layout.left_edge) or not text.endswith(layout.right_edge):
            return None
        end = len(text) - len(layout.right_edge) if layout.right_edge else len(text)
        control = text[len(layout.left_edge) : end]
        if not control.strip():
            return None
        try:
            candidate = self.prepare(layout.request, control)
        except ValueError:
            return None
        if not strict:
            return candidate
        expected = layout.input_ids.clone()
        expected[layout.start : layout.stop] = tokens
        if candidate.start != layout.start or candidate.stop != layout.stop:
            return None
        return candidate if torch.equal(candidate.input_ids, expected) else None

    def allowed_tokens(self, *, allow_non_ascii: bool = True) -> torch.Tensor:
        allowed = torch.ones(self.vocab_size, dtype=torch.bool)
        for token in self.tokenizer.all_special_ids:
            if token < self.vocab_size:
                allowed[token] = False
        if not allow_non_ascii:
            for token in range(self.vocab_size):
                text = self.decode([token])
                if not text.isascii() or not text.isprintable():
                    allowed[token] = False
        if not allowed.any():
            raise ValueError("No allowed tokens remain.")
        return allowed.to(self.device)

    def check_length(self, length: int) -> None:
        limit = getattr(self.model.config, "max_position_embeddings", None)
        if limit is not None and length > limit:
            raise ValueError(f"Sequence length {length} exceeds model context {limit}.")

    def logits(self, *, ids=None, embeds=None, hidden=False):
        return self.model(
            input_ids=ids, inputs_embeds=embeds, use_cache=False, output_hidden_states=hidden
        )

    def soft_forward(self, layout: Layout, soft: torch.Tensor):
        """soft is [B, editable length, embedding dim]; preserve all real context."""
        ids = torch.cat((layout.input_ids, layout.target_ids))
        fixed = F.embedding(ids, self.embedding).unsqueeze(0).expand(soft.shape[0], -1, -1)
        embeds = torch.cat(
            (fixed[:, : layout.start], soft.to(fixed.dtype), fixed[:, layout.stop :]), dim=1
        )
        return self.logits(embeds=embeds).logits

    def target_loss(self, logits: torch.Tensor, layout: Layout) -> torch.Tensor:
        begin = len(layout.input_ids) - 1
        pred = logits[:, begin : begin + len(layout.target_ids)].float()
        labels = layout.target_ids.expand(pred.shape[0], -1)
        return F.cross_entropy(pred.transpose(1, 2), labels, reduction="none").mean(-1)

    @torch.no_grad()
    def losses(self, layouts: list[Layout], batch_size: int = 32) -> torch.Tensor:
        """Group equal lengths to avoid padding-dependent positions or wasted memory."""
        require_count(batch_size, "batch_size")
        scores = torch.empty(len(layouts), device=self.device)
        groups: dict[tuple[int, int], list[int]] = {}
        for index, layout in enumerate(layouts):
            groups.setdefault((len(layout.input_ids), len(layout.target_ids)), []).append(index)
        for indices in groups.values():
            for start in range(0, len(indices), batch_size):
                chunk = indices[start : start + batch_size]
                ids = torch.stack(
                    [torch.cat((layouts[i].input_ids, layouts[i].target_ids)) for i in chunk]
                )
                logits = self.logits(ids=ids).logits
                for j, i in enumerate(chunk):
                    scores[i] = self.target_loss(logits[j : j + 1], layouts[i])[0]
        if not torch.isfinite(scores).all():
            raise FloatingPointError(
                "Non-finite model loss; try float32 or another checkpoint."
            )
        return scores

    @torch.no_grad()
    def generate(self, content: str, *, max_new_tokens: int = 64):
        require_count(max_new_tokens, "max_new_tokens")
        ids = self.encode_prompt(content)
        return self._generate_ids(ids, max_new_tokens)

    @torch.no_grad()
    def _generate_ids(self, ids, max_new_tokens):
        self.check_length(len(ids) + max_new_tokens)
        pad = self.tokenizer.pad_token_id
        if pad is None:
            pad = self.tokenizer.eos_token_id
        if pad is None:
            pad = self.tokenizer.bos_token_id
        output = self.model.generate(
            input_ids=ids[None],
            attention_mask=torch.ones_like(ids[None]),
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=pad,
        )
        response_ids = output[0, len(ids) :].tolist()
        response = self.tokenizer.decode(
            response_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return response, tuple(ids.tolist()), tuple(response_ids)

    def text_generator(self, *, max_new_tokens: int = 128):
        """Adapt this local model to the existing TextGenerator protocol."""
        require_count(max_new_tokens, "max_new_tokens")

        def generate(messages):
            if self.chat:
                rendered = self.tokenizer.apply_chat_template(
                    [message.to_dict() for message in messages],
                    tokenize=False,
                    add_generation_prompt=True,
                )
            else:
                rendered = "\n".join(message.content for message in messages)
            ids = self._encode(rendered)[0]
            with self.session():
                return self._generate_ids(ids, max_new_tokens)[0]

        return generate

    def finish(
        self,
        *,
        method: str,
        layout: Layout,
        initial_loss: float,
        trace: list[TraceStep],
        stop_reason: str,
        config: dict,
        source: dict[str, str],
        max_new_tokens: int,
    ) -> OptimizationResult:
        content = layout.request.compose(layout.control)
        response, ids, response_ids = self.generate(content, max_new_tokens=max_new_tokens)
        if ids != tuple(layout.input_ids.tolist()):
            raise RuntimeError("Final generation input differs from the evaluated prompt.")
        final_loss = float(self.losses([layout])[0])
        prompt = Prompt(
            layout.request.query,
            self.messages(content),
            (Step(method, "optimize_and_generate", source["repository"]),),
        )
        return OptimizationResult(
            method,
            layout.request,
            prompt,
            layout.control,
            response,
            ids,
            response_ids,
            tuple(layout.target_ids.tolist()),
            initial_loss,
            final_loss,
            tuple(trace),
            stop_reason,
            config,
            {
                "model": getattr(self.model.config, "_name_or_path", ""),
                "device": str(self.device),
                "dtype": str(self.embedding.dtype),
                "chat": self.chat,
                "chat_template": self.tokenizer.chat_template if self.chat else None,
                "torch": torch.__version__,
            },
            source,
        )
