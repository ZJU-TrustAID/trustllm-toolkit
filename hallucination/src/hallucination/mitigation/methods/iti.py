from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from hallucination.mitigation.core.artifacts import ITIArtifact, ITIHead
from hallucination.mitigation.core.contracts import GenerationRequest, MitigationReport
from hallucination.mitigation.core.errors import (
    ConfigError,
    GenerationError,
    MissingDependencyError,
    UnsupportedModelError,
)
from hallucination.mitigation.core.prompt import render_pair
from hallucination.mitigation.engine.hooks import (
    HookManager,
    ITIHeadHook,
    find_attention_output_projection,
    find_decoder_layers,
)


@dataclass
class ITIConfig:
    top_k: int = 48
    alpha: float = 15.0
    position: str = "generated"
    chat: bool = False
    baseline: bool = True
    bake: bool = False

    def __post_init__(self) -> None:
        if self.top_k < 1:
            raise ValueError("top_k must be positive")
        if self.alpha < 0:
            raise ValueError("alpha must be >= 0")
        if self.position not in {"generated", "all"}:
            raise ValueError("position must be 'generated' or 'all'")


def collect_head_activations(engine: Any, texts: list[str]) -> np.ndarray:
    """Collect last-content-token head activations with shape [n, layers, heads, head_dim]."""
    layers = find_decoder_layers(engine.model)
    captures: dict[int, list[np.ndarray]] = {index: [] for index in range(len(layers))}
    current = {"index": -1}
    manager = HookManager()
    for index, layer in enumerate(layers):
        projection = find_attention_output_projection(layer)

        def make_hook(layer_index: int):
            def hook(module: Any, args: Any) -> None:
                if not args:
                    return
                tensor = args[0]
                if hasattr(tensor, "detach"):
                    position = current["index"]
                    captures[layer_index].append(
                        tensor[0, position].detach().float().cpu().numpy()
                    )
                return

            return hook

        manager.add(projection, make_hook(index), pre_hook=True)
    manager.active = True
    try:
        for text in texts:
            current["index"] = (
                engine.activation_index(text) if hasattr(engine, "activation_index") else -1
            )
            engine.forward_text(text, output_hidden_states=False)
    finally:
        manager.remove()
    num_heads = engine.num_heads
    head_dim = engine.head_dim
    arrays = []
    for index in range(len(layers)):
        if len(captures[index]) != len(texts):
            raise UnsupportedModelError(
                f"failed to capture activations for layer {index}"
            )
        selected = np.stack(
            [item.reshape(num_heads, head_dim) for item in captures[index]]
        )
        arrays.append(selected)
    return np.stack(arrays, axis=1)


def _split_indices(
    samples: list[dict[str, Any]], seed: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    if all("question" in sample for sample in samples):
        questions = sorted({str(sample["question"]) for sample in samples})
        rng.shuffle(questions)
        cutoff = max(1, int(len(questions) * 0.8))
        train_questions = set(questions[:cutoff])
        train = np.array(
            [
                i
                for i, sample in enumerate(samples)
                if str(sample["question"]) in train_questions
            ]
        )
        val = np.array(
            [
                i
                for i, sample in enumerate(samples)
                if str(sample["question"]) not in train_questions
            ]
        )
        if val.size == 0:
            val = train
        return train, val
    indices = np.arange(len(samples))
    rng.shuffle(indices)
    cutoff = max(1, int(len(indices) * 0.8))
    return indices[:cutoff], indices[cutoff:] if cutoff < len(indices) else indices[
        :cutoff
    ]


def fit_iti(
    engine: Any, samples: list[dict[str, Any]], config: ITIConfig | None = None
) -> ITIArtifact:
    config = config or ITIConfig()
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score
    except ImportError as exc:
        raise MissingDependencyError(
            "Install the ITI extra first: python -m pip install -e '.[iti]'"
        ) from exc
    if len(samples) < 4:
        raise ConfigError("ITI fitting requires at least 4 labeled samples")
    texts: list[str] = []
    labels: list[int] = []
    for index, sample in enumerate(samples):
        try:
            texts.append(
                render_pair(
                    str(sample["prompt"]),
                    str(sample["response"]),
                    engine.tokenizer,
                    chat=config.chat,
                )
            )
            labels.append(int(sample["label"]))
        except KeyError as exc:
            raise ConfigError(f"sample {index} missing field: {exc}") from exc
    activations = collect_head_activations(engine, texts)
    labels_array = np.asarray(labels, dtype=np.int32)
    train, val = _split_indices(samples)
    if len(np.unique(labels_array[train])) < 2:
        raise ConfigError(
            "ITI training split must contain both truthful and untruthful labels"
        )
    num_layers = activations.shape[1]
    num_heads = activations.shape[2]
    candidates: list[tuple[float, int, int, np.ndarray, float]] = []
    for layer in range(num_layers):
        for head in range(num_heads):
            features = activations[:, layer, head, :]
            clf = LogisticRegression(max_iter=1000)
            clf.fit(features[train], labels_array[train])
            predictions = clf.predict(features[val])
            accuracy = float(accuracy_score(labels_array[val], predictions))
            direction = np.asarray(clf.coef_[0], dtype=np.float32)
            norm = float(np.linalg.norm(direction))
            if norm == 0.0:
                continue
            direction = direction / norm
            projections = features @ direction
            scale = float(np.std(projections))
            candidates.append((accuracy, layer, head, direction, scale))
    candidates.sort(key=lambda item: item[0], reverse=True)
    selected = candidates[: min(config.top_k, len(candidates))]
    heads = [
        ITIHead(
            layer=layer, head=head, direction=direction, scale=scale, accuracy=accuracy
        )
        for accuracy, layer, head, direction, scale in selected
    ]
    fingerprint = engine.fingerprint()
    fingerprint.update(
        {
            "head_dim": int(activations.shape[3]),
            "num_layers": int(num_layers),
            "num_heads": int(num_heads),
        }
    )
    return ITIArtifact(
        heads=heads,
        model=fingerprint,
        alpha=config.alpha,
        top_k=config.top_k,
        metadata={
            "train_size": int(train.size),
            "val_size": int(val.size),
            "best_accuracy": float(selected[0][0]) if selected else 0.0,
        },
    )


class ITIMethod:
    name = "iti"

    def run(
        self,
        engine: Any,
        request: GenerationRequest,
        artifact: ITIArtifact,
        config: ITIConfig | None = None,
    ) -> MitigationReport:
        config = config or ITIConfig(top_k=artifact.top_k, alpha=artifact.alpha)
        if config.bake:
            raise ConfigError(
                "bake=True is not implemented in v0.1; use the runtime hook"
            )
        fingerprint = engine.fingerprint()
        expected = artifact.model
        for key in ("hidden_size", "head_dim", "num_heads"):
            if expected.get(key) is not None and int(expected[key]) != int(
                fingerprint.get(key, -1)
            ):
                raise ConfigError(
                    f"ITI artifact {key} mismatch: artifact={expected[key]}, model={fingerprint.get(key)}"
                )
        started = time.perf_counter()
        baseline = None
        baseline_cost = None
        if config.baseline:
            baseline_out = engine.generate_text(request)
            baseline = baseline_out.text
            baseline_cost = baseline_out.cost
        layers = find_decoder_layers(engine.model)
        grouped: dict[int, list[ITIHead]] = {}
        for head in artifact.heads:
            grouped.setdefault(head.layer, []).append(head)
        manager = HookManager()
        hooks: list[ITIHeadHook] = []
        for layer_index, heads in grouped.items():
            if layer_index >= len(layers):
                raise ConfigError(
                    f"ITI layer {layer_index} out of range for {len(layers)} layers"
                )
            projection = find_attention_output_projection(layers[layer_index])
            hook = ITIHeadHook(heads, alpha=config.alpha, position=config.position)
            manager.add(projection, hook, pre_hook=True)
            hooks.append(hook)
        try:
            output = engine.generate_text(
                request,
                hook_manager=manager,
                hook_prefill=config.position == "all",
            )
        finally:
            manager.remove()
        if not output.text:
            raise GenerationError("ITI produced an empty response")
        elapsed = time.perf_counter() - started
        return MitigationReport(
            method=self.name,
            prompt=request.prompt,
            baseline=baseline,
            response=output.text,
            context=request.context,
            trace={
                "alpha": config.alpha,
                "position": config.position,
                "heads": [
                    {
                        "layer": head.layer,
                        "head": head.head,
                        "scale": head.scale,
                        "accuracy": head.accuracy,
                    }
                    for head in artifact.heads
                ],
                "hook_calls": sum(hook.calls for hook in hooks),
            },
            warnings=[
                "ITI probes are model- and dataset-specific; do not reuse an artifact across models.",
                "ITI can over-steer; validate alpha and top_k on held-out data.",
            ],
            cost={
                "latency_seconds": round(elapsed, 4),
                "generated_tokens": len(output.generated_ids),
                "forward_passes": 1 + len(output.generated_ids),
                "baseline": baseline_cost,
            },
            provenance={
                "method": self.name,
                "model": fingerprint,
                "artifact": {
                    "top_k": artifact.top_k,
                    "alpha": artifact.alpha,
                    "metadata": artifact.metadata,
                },
                "seed": request.seed,
            },
        )
