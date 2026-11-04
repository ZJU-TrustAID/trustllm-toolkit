from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from hallucination.mitigation.core.contracts import GenerationRequest, MitigationReport
from hallucination.mitigation.core.errors import GenerationError, UnsupportedModelError
from hallucination.mitigation.core.sampling import log_softmax, sample_token, softmax


@dataclass
class DoLaConfig:
    candidate_premature_layers: list[int] | None = None
    mature_layer: int | None = None
    relative_top: float = 0.1
    post_softmax: bool = True
    official_compat: bool = False
    baseline: bool = True

    def __post_init__(self) -> None:
        if not 0.0 <= self.relative_top <= 1.0:
            raise ValueError("relative_top must be in [0, 1]")
        if self.candidate_premature_layers is not None:
            if not self.candidate_premature_layers:
                raise ValueError("candidate_premature_layers must be non-empty")
            if any(layer < 0 for layer in self.candidate_premature_layers):
                raise ValueError("candidate layers must be >= 0")


def default_candidate_layers(num_layers: int) -> list[int]:
    if num_layers >= 32:
        return list(range(16, num_layers, 2))
    if num_layers >= 20:
        return list(range(10, num_layers, 2))
    return list(range(max(1, num_layers // 2), num_layers, 1))


def jsd(mature: np.ndarray, premature: np.ndarray) -> float:
    p = softmax(mature)
    q = softmax(premature)
    m = 0.5 * (p + q)
    eps = 1e-12
    kl_p = np.sum(p * (np.log(p + eps) - np.log(m + eps)))
    kl_q = np.sum(q * (np.log(q + eps) - np.log(m + eps)))
    return float(0.5 * (kl_p + kl_q))


def select_premature_layer(
    mature: np.ndarray, candidates: dict[int, np.ndarray]
) -> tuple[int, dict[int, float]]:
    if not candidates:
        raise ValueError("at least one candidate layer is required")
    scores = {layer: jsd(mature, logits) for layer, logits in candidates.items()}
    best = max(scores, key=scores.get)
    return int(best), {int(layer): float(value) for layer, value in scores.items()}


def apc_mask(logits: np.ndarray, relative_top: float) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64).copy()
    if relative_top <= 0.0:
        return values
    threshold = float(np.max(values)) + float(np.log(relative_top))
    values[values < threshold] = -np.inf
    return values


def contrast_logits(
    mature: np.ndarray,
    premature: np.ndarray,
    *,
    relative_top: float = 0.1,
    post_softmax: bool = True,
    official_compat: bool = False,
) -> np.ndarray:
    mature_masked = apc_mask(mature, relative_top)
    if official_compat:
        # The official generation path subtracts premature log-probs from raw mature logits.
        base = log_softmax(premature)
        values = mature_masked - base
    else:
        values = log_softmax(mature_masked) - log_softmax(premature)
    if post_softmax:
        values = log_softmax(values)
    return values


class DoLaMethod:
    name = "dola"

    def run(
        self, engine: Any, request: GenerationRequest, config: DoLaConfig | None = None
    ) -> MitigationReport:
        config = config or DoLaConfig()
        started = time.perf_counter()
        baseline = None
        baseline_cost: dict[str, Any] | None = None
        if config.baseline:
            baseline_out = engine.generate_text(request)
            baseline = baseline_out.text
            baseline_cost = baseline_out.cost

        prompt_ids = engine.encode(request.prompt)
        prefill = engine.prefill(prompt_ids, output_hidden_states=True)
        if prefill.hidden_states is None:
            raise UnsupportedModelError(
                "DoLa requires output_hidden_states from the model"
            )
        num_layers = engine.num_layers
        mature_layer = (
            num_layers if config.mature_layer is None else int(config.mature_layer)
        )
        if mature_layer > num_layers:
            raise UnsupportedModelError(
                f"mature_layer={mature_layer} exceeds num_layers={num_layers}"
            )
        candidates = config.candidate_premature_layers or default_candidate_layers(
            num_layers
        )
        candidates = [layer for layer in candidates if layer < mature_layer]
        if not candidates:
            raise UnsupportedModelError("no valid candidate layers below mature_layer")

        rng = np.random.default_rng(request.seed)
        generated: list[int] = []
        trace_steps: list[dict[str, Any]] = []
        finish_reason = "length"
        hidden_states = prefill.hidden_states
        mature_logits = engine.project_layer(hidden_states, mature_layer)
        candidate_logits = {
            layer: engine.project_layer(hidden_states, layer) for layer in candidates
        }
        premature_layer, scores = select_premature_layer(
            mature_logits, candidate_logits
        )
        logits = contrast_logits(
            mature_logits,
            candidate_logits[premature_layer],
            relative_top=config.relative_top,
            post_softmax=config.post_softmax,
            official_compat=config.official_compat,
        )

        for step in range(request.max_new_tokens):
            token = sample_token(
                logits,
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
                    "premature_layer": premature_layer,
                    "jsd": scores[premature_layer],
                    "chosen_token": token,
                }
            )
            if engine.eos_token_id is not None and token == engine.eos_token_id:
                finish_reason = "eos"
                break
            out = engine.step(prefill.state, token, output_hidden_states=True)
            generated.append(token)
            hidden_states = out.hidden_states
            if hidden_states is None:
                raise UnsupportedModelError(
                    "DoLa requires output_hidden_states at every step"
                )
            mature_logits = engine.project_layer(hidden_states, mature_layer)
            candidate_logits = {
                layer: engine.project_layer(hidden_states, layer)
                for layer in candidates
            }
            premature_layer, scores = select_premature_layer(
                mature_logits, candidate_logits
            )
            logits = contrast_logits(
                mature_logits,
                candidate_logits[premature_layer],
                relative_top=config.relative_top,
                post_softmax=config.post_softmax,
                official_compat=config.official_compat,
            )

        response = engine.decode(generated)
        if not response:
            raise GenerationError("DoLa produced an empty response")
        elapsed = time.perf_counter() - started
        distribution: dict[int, int] = {}
        for item in trace_steps:
            layer = int(item["premature_layer"])
            distribution[layer] = distribution.get(layer, 0) + 1
        return MitigationReport(
            method=self.name,
            prompt=request.prompt,
            baseline=baseline,
            response=response,
            context=request.context,
            trace={
                "steps": trace_steps,
                "premature_layer_distribution": distribution,
                "finish_reason": finish_reason,
                "mature_layer": mature_layer,
                "candidate_premature_layers": candidates,
                "relative_top": config.relative_top,
                "post_softmax": config.post_softmax,
                "official_compat": config.official_compat,
            },
            warnings=[
                "DoLa changes decoding only; it does not guarantee factual correctness.",
                "Layer buckets are presets and must be validated on your model/data.",
            ],
            cost={
                "latency_seconds": round(elapsed, 4),
                "generated_tokens": len(generated),
                "forward_passes": 1 + len(generated),
                "baseline": baseline_cost,
            },
            provenance={
                "method": self.name,
                "model": engine.fingerprint(),
                "seed": request.seed,
                "generation": {
                    "max_new_tokens": request.max_new_tokens,
                    "do_sample": request.do_sample,
                    "temperature": request.temperature,
                    "top_p": request.top_p,
                    "top_k": request.top_k,
                },
            },
        )
