from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from hallucination.mitigation.core.artifacts import SteeringArtifact
from hallucination.mitigation.core.contracts import GenerationRequest, MitigationReport
from hallucination.mitigation.core.errors import (
    ArtifactMismatchError,
    ConfigError,
    GenerationError,
)
from hallucination.mitigation.core.prompt import render_pair
from hallucination.mitigation.engine.hooks import (
    HookManager,
    LayerOutputHook,
    find_decoder_layers,
)


@dataclass
class CAAConfig:
    layer: int = 12
    multiplier: float = 1.0
    position: str = "generated"
    pooling: str = "last"
    chat: bool = False
    baseline: bool = True

    def __post_init__(self) -> None:
        if self.layer < 0:
            raise ValueError("layer must be >= 0")
        if self.position not in {"generated", "all"}:
            raise ValueError("position must be 'generated' or 'all'")
        if self.pooling not in {"last", "mean"}:
            raise ValueError("pooling must be 'last' or 'mean'")


def normalize(vector: np.ndarray) -> np.ndarray:
    values = np.asarray(vector, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(values))
    if norm == 0.0:
        raise ConfigError("cannot normalize a zero steering vector")
    return values / norm


def fit_caa_from_activations(
    positive: np.ndarray,
    negative: np.ndarray,
    *,
    layer: int,
    model: dict[str, Any] | None = None,
    behavior: str = "hallucination",
    multiplier: float = 1.0,
    metadata: dict[str, Any] | None = None,
) -> SteeringArtifact:
    pos = np.asarray(positive, dtype=np.float32)
    neg = np.asarray(negative, dtype=np.float32)
    if pos.shape != neg.shape:
        raise ConfigError("positive and negative activation shapes must match")
    if pos.ndim != 2:
        raise ConfigError(
            "expected activation arrays of shape [n_samples, hidden_size]"
        )
    vector = (pos - neg).mean(axis=0)
    return SteeringArtifact(
        method="caa",
        layer=int(layer),
        vector=normalize(vector),
        model=dict(model or {}),
        multiplier=float(multiplier),
        behavior=behavior,
        metadata=dict(metadata or {}),
    )


def collect_caa_activations(
    engine: Any,
    pairs: list[dict[str, Any]],
    *,
    layer: int,
    chat: bool = False,
    pooling: str = "last",
) -> tuple[np.ndarray, np.ndarray]:
    if not pairs:
        raise ConfigError("CAA fitting requires at least one positive/negative pair")
    positive: list[np.ndarray] = []
    negative: list[np.ndarray] = []
    for index, pair in enumerate(pairs):
        try:
            prompt = str(pair["prompt"])
            pos_text = str(pair["positive"])
            neg_text = str(pair["negative"])
        except KeyError as exc:
            raise ConfigError(f"pair {index} missing field: {exc}") from exc
        for target, text in (
            (positive, render_pair(prompt, pos_text, engine.tokenizer, chat=chat)),
            (negative, render_pair(prompt, neg_text, engine.tokenizer, chat=chat)),
        ):
            forward = engine.forward_text(text, output_hidden_states=True)
            hidden = forward.hidden_states
            if hidden is None or layer >= len(hidden):
                raise ConfigError(
                    f"layer {layer} is unavailable; model has {len(hidden or [])} hidden states"
                )
            values = hidden[layer][0]
            index = (
                engine.activation_index(text)
                if hasattr(engine, "activation_index")
                else -1
            )
            if pooling == "last":
                target.append(values[index].detach().float().cpu().numpy())
            else:
                target.append(values.mean(dim=0).detach().float().cpu().numpy())
    return np.stack(positive), np.stack(negative)


def fit_caa(
    engine: Any,
    pairs: list[dict[str, Any]],
    config: CAAConfig,
    *,
    behavior: str = "hallucination",
) -> SteeringArtifact:
    positive, negative = collect_caa_activations(
        engine, pairs, layer=config.layer, chat=config.chat, pooling=config.pooling
    )
    fingerprint = engine.fingerprint()
    return fit_caa_from_activations(
        positive,
        negative,
        layer=config.layer,
        model=fingerprint,
        behavior=behavior,
        multiplier=config.multiplier,
        metadata={"pairs": len(pairs), "pooling": config.pooling, "chat": config.chat},
    )


class CAAMethod:
    name = "caa"

    def run(
        self,
        engine: Any,
        request: GenerationRequest,
        artifact: SteeringArtifact,
        config: CAAConfig | None = None,
    ) -> MitigationReport:
        config = config or CAAConfig(
            layer=artifact.layer, multiplier=artifact.multiplier
        )
        artifact.validate_for(engine.fingerprint())
        if artifact.vector.size != engine.hidden_size:
            raise ArtifactMismatchError(
                f"CAA vector size {artifact.vector.size} does not match hidden_size {engine.hidden_size}"
            )
        layer = int(config.layer)
        if layer != artifact.layer:
            artifact = SteeringArtifact(
                method=artifact.method,
                layer=layer,
                vector=artifact.vector,
                model=artifact.model,
                multiplier=config.multiplier,
                behavior=artifact.behavior,
                metadata=artifact.metadata,
            )
        started = time.perf_counter()
        baseline = None
        baseline_cost = None
        if config.baseline:
            baseline_out = engine.generate_text(request)
            baseline = baseline_out.text
            baseline_cost = baseline_out.cost
        layers = find_decoder_layers(engine.model)
        if layer >= len(layers):
            raise ConfigError(
                f"layer {layer} out of range; model has {len(layers)} decoder layers"
            )
        manager = HookManager()
        hook = LayerOutputHook(
            layer, artifact.vector * config.multiplier, position=config.position
        )
        manager.add(layers[layer], hook)
        try:
            output = engine.generate_text(
                request,
                hook_manager=manager,
                hook_prefill=config.position == "all",
            )
        finally:
            manager.remove()
        if not output.text:
            raise GenerationError("CAA produced an empty response")
        elapsed = time.perf_counter() - started
        return MitigationReport(
            method=self.name,
            prompt=request.prompt,
            baseline=baseline,
            response=output.text,
            context=request.context,
            trace={
                "layer": layer,
                "multiplier": config.multiplier,
                "position": config.position,
                "vector_norm": artifact.norm,
                "hook_calls": hook.calls,
            },
            warnings=[
                "CAA steering is data- and layer-specific; validate on your own labeled data.",
                "A large multiplier can cause over-correction or degenerate output.",
            ],
            cost={
                "latency_seconds": round(elapsed, 4),
                "generated_tokens": len(output.generated_ids),
                "forward_passes": 1 + len(output.generated_ids),
                "baseline": baseline_cost,
            },
            provenance={
                "method": self.name,
                "model": engine.fingerprint(),
                "artifact": {
                    "format": artifact.format,
                    "behavior": artifact.behavior,
                    "checksum": artifact.checksum,
                },
                "seed": request.seed,
            },
        )
