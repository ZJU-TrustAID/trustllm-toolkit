from __future__ import annotations

import time
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from hallucination.mitigation.core.contracts import GenerationRequest, MitigationReport
from hallucination.mitigation.core.errors import ConfigError, GenerationError
from hallucination.mitigation.core.prompt import build_cad_prompts
from hallucination.mitigation.core.sampling import (
    log_softmax,
    sample_token,
    softmax,
    top_p_filter,
)


@dataclass
class CADConfig:
    alpha: float = 0.5
    context_template: str = "plain"
    filter_top_p: float = 1.0
    baseline: bool = True

    def __post_init__(self) -> None:
        if self.alpha < 0:
            raise ValueError("alpha must be >= 0")
        if self.context_template not in {"plain", "chat"}:
            raise ValueError("context_template must be 'plain' or 'chat'")
        if not 0 < self.filter_top_p <= 1:
            raise ValueError("filter_top_p must be in (0, 1]")


def combine_context_logits(
    context_logits: np.ndarray, no_context_logits: np.ndarray, alpha: float
) -> np.ndarray:
    context = log_softmax(np.asarray(context_logits, dtype=np.float64))
    no_context = log_softmax(np.asarray(no_context_logits, dtype=np.float64))
    return (1.0 + alpha) * context - alpha * no_context


def distribution_kl(left: np.ndarray, right: np.ndarray) -> float:
    p = softmax(left)
    q = softmax(right)
    eps = 1e-12
    return float(np.sum(p * (np.log(p + eps) - np.log(q + eps))))


class CADMethod:
    name = "cad"

    def run(
        self, engine: Any, request: GenerationRequest, config: CADConfig | None = None
    ) -> MitigationReport:
        if not request.context:
            raise ConfigError("CAD requires a non-empty context")
        config = config or CADConfig()
        started = time.perf_counter()
        context_prompt, no_context_prompt = build_cad_prompts(
            request.prompt,
            request.context,
            engine.tokenizer,
            template=config.context_template,
        )
        baseline = None
        baseline_cost = None
        if config.baseline:
            baseline_request = replace(request, prompt=context_prompt, context=None)
            baseline_out = engine.generate_text(baseline_request)
            baseline = baseline_out.text
            baseline_cost = baseline_out.cost

        context_prefill = engine.prefill(context_prompt)
        no_context_prefill = engine.prefill(no_context_prompt)
        context_logits = context_prefill.logits.detach().float().cpu().numpy()
        no_context_logits = no_context_prefill.logits.detach().float().cpu().numpy()
        context_state = context_prefill.state
        no_context_state = no_context_prefill.state
        generated: list[int] = []
        trace_steps: list[dict[str, Any]] = []
        finish_reason = "length"
        rng = np.random.default_rng(request.seed)

        for step in range(request.max_new_tokens):
            if config.filter_top_p < 1.0:
                context_for_sample = top_p_filter(context_logits, config.filter_top_p)
                no_context_for_sample = top_p_filter(
                    no_context_logits, config.filter_top_p
                )
            else:
                context_for_sample = context_logits
                no_context_for_sample = no_context_logits
            combined = combine_context_logits(
                context_for_sample, no_context_for_sample, config.alpha
            )
            token = sample_token(
                combined,
                do_sample=request.do_sample,
                temperature=request.temperature,
                top_p=request.top_p,
                top_k=request.top_k,
                repetition_penalty=request.repetition_penalty,
                generated_ids=generated,
                rng=rng,
            )
            trace_steps.append(
                {
                    "step": step,
                    "kl_context_vs_no_context": distribution_kl(
                        context_for_sample, no_context_for_sample
                    ),
                    "context_top": int(np.argmax(context_for_sample)),
                    "no_context_top": int(np.argmax(no_context_for_sample)),
                    "chosen_token": token,
                }
            )
            if engine.eos_token_id is not None and token == engine.eos_token_id:
                finish_reason = "eos"
                break
            context_out = engine.step(context_state, token)
            no_context_out = engine.step(no_context_state, token)
            generated.append(token)
            context_logits = context_out.logits.detach().float().cpu().numpy()
            no_context_logits = no_context_out.logits.detach().float().cpu().numpy()

        response = engine.decode(generated)
        if not response:
            raise GenerationError("CAD produced an empty response")
        elapsed = time.perf_counter() - started
        return MitigationReport(
            method=self.name,
            prompt=request.prompt,
            baseline=baseline,
            response=response,
            context=request.context,
            trace={
                "steps": trace_steps,
                "finish_reason": finish_reason,
                "alpha": config.alpha,
                "context_template": config.context_template,
                "context_view_tokens": len(engine.encode(context_prompt)),
                "no_context_view_tokens": len(engine.encode(no_context_prompt)),
            },
            warnings=[
                "CAD requires an explicit context and is not a general-purpose factuality guarantee.",
                "CAD uses two forward passes per token; latency and KV-cache memory are roughly doubled.",
            ],
            cost={
                "latency_seconds": round(elapsed, 4),
                "generated_tokens": len(generated),
                "forward_passes": 2 * (1 + len(generated)),
                "baseline": baseline_cost,
            },
            provenance={
                "method": self.name,
                "model": engine.fingerprint(),
                "seed": request.seed,
                "alpha": config.alpha,
                "context_template": config.context_template,
            },
        )
