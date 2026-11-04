"""Greedy Coordinate Gradient: input gradient -> substitutions -> exact CE."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F

from textbook_toolkit.core import Method

from ..types import require_count
from .backend import CausalLM, Layout
from .types import OptimizationRequest, OptimizationResult, TraceStep

SOURCE = {
    "repository": "https://github.com/llm-attacks/llm-attacks",
    "commit": "098262edf85f807224e70ecd87b9d83716bf6b73",
    "paper": "https://arxiv.org/abs/2307.15043",
}


@dataclass(frozen=True)
class GCGConfig:
    steps: int = 100
    top_k: int = 128
    candidates: int = 128
    eval_batch_size: int = 16
    seed: int = 0
    allow_non_ascii: bool = False
    loss_threshold: float | None = None
    max_new_tokens: int = 64

    def __post_init__(self):
        for name in ("steps", "top_k", "candidates", "eval_batch_size", "max_new_tokens"):
            require_count(getattr(self, name), name)
        if self.loss_threshold is not None and (
            not math.isfinite(self.loss_threshold) or self.loss_threshold < 0
        ):
            raise ValueError("loss_threshold must be finite and nonnegative.")


def token_gradients(backend: CausalLM, layout: Layout) -> torch.Tensor:
    """Normalized one-hot gradient; does not accumulate model parameter grads."""
    one_hot = F.one_hot(layout.control_ids, backend.vocab_size).to(backend.embedding.dtype)
    one_hot.requires_grad_()
    logits = backend.soft_forward(layout, (one_hot @ backend.embedding)[None])
    loss = backend.target_loss(logits, layout).mean()
    (gradient,) = torch.autograd.grad(loss, one_hot)
    gradient = gradient.float()
    if not torch.isfinite(gradient).all():
        raise FloatingPointError("Non-finite one-hot gradient; use float32 weights.")
    return gradient / gradient.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def sample_candidates(tokens, gradient, allowed, *, count, top_k, generator):
    """Balanced positions, random top-k replacements, exactly one changed token."""
    scores = gradient.clone()
    scores[:, ~allowed] = torch.inf
    scores.scatter_(1, tokens[:, None], torch.inf)
    k = min(top_k, int(torch.isfinite(scores).sum(-1).min()))
    if k < 1:
        return tokens.new_empty((0, len(tokens)))
    choices = (-scores).topk(k, dim=-1).indices
    positions = (torch.arange(count) * len(tokens) // count).to(tokens.device)
    pick = torch.randint(k, (count,), generator=generator).to(tokens.device)
    candidates = tokens.repeat(count, 1)
    candidates[torch.arange(count, device=tokens.device), positions] = choices[positions, pick]
    return candidates


class GCG(Method[OptimizationRequest, OptimizationResult]):
    def __init__(self, backend: CausalLM, config: GCGConfig | None = None):
        self.backend, self.config = backend, config or GCGConfig()

    def run(self, query: str, target: str, **request_options) -> OptimizationResult:
        return self.build(OptimizationRequest(query, target, **request_options))

    def build(self, value: OptimizationRequest) -> OptimizationResult:
        if not isinstance(value, OptimizationRequest):
            raise TypeError("GCG.build expects OptimizationRequest; or use run(query, target).")
        if value.placement != "suffix":
            raise ValueError("GCG optimizes a suffix; use placement='suffix'.")
        backend, cfg = self.backend, self.config
        generator = torch.Generator(device="cpu").manual_seed(cfg.seed)
        with backend.session():
            current = backend.prepare(value, value.initial_control)
            backend.check_length(len(current.input_ids) + cfg.max_new_tokens)
            initial = best = float(backend.losses([current])[0])
            trace = [TraceStep(0, best, best, current.control, {"target": best})]
            allowed = backend.allowed_tokens(allow_non_ascii=cfg.allow_non_ascii)
            reason = "steps_exhausted"
            for iteration in range(1, cfg.steps + 1):
                if cfg.loss_threshold is not None and best <= cfg.loss_threshold:
                    reason = "loss_threshold"
                    break
                gradient = token_gradients(backend, current)
                tokens = sample_candidates(
                    current.control_ids,
                    gradient,
                    allowed,
                    count=cfg.candidates,
                    top_k=cfg.top_k,
                    generator=generator,
                )
                candidates = [current]
                seen = {tuple(current.control_ids.tolist())}
                for row in tokens:
                    key = tuple(row.tolist())
                    if key in seen:
                        continue
                    seen.add(key)
                    candidate = backend.candidate(current, row)
                    if candidate is not None:
                        candidates.append(candidate)
                if len(candidates) == 1:
                    reason = "no_valid_candidates"
                    break
                losses = backend.losses(candidates, cfg.eval_batch_size)
                index = int(losses.argmin())
                current, best = candidates[index], float(losses[index])
                trace.append(
                    TraceStep(
                        iteration,
                        best,
                        best,
                        current.control,
                        {"target": best},
                        len(candidates) - 1,
                    )
                )
            return backend.finish(
                method="gcg",
                layout=current,
                initial_loss=initial,
                trace=trace,
                stop_reason=reason,
                config=asdict(cfg),
                source=SOURCE,
                max_new_tokens=cfg.max_new_tokens,
            )
