"""COLD: continuous logits, energy gradients, Adam/noise and LM-guided decoding.

Independent integration of the published algorithm. See implementation_notes.md
for source versions, numerical choices, and differences from upstream scripts.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F

from textbook_toolkit.core import Method

from ..types import require_count, require_text
from .backend import CausalLM, Layout
from .types import OptimizationRequest, OptimizationResult, TraceStep

SOURCE = {
    "repository": "https://github.com/Yu-Fangxu/COLD-Attack",
    "commit": "f7ffcb0f0e3b56466e7f4fb764f67f7710435eb1",
    "paper": "https://arxiv.org/abs/2402.08679",
}


@dataclass(frozen=True)
class COLDConfig:
    mode: str = "suffix"  # suffix / control (fixed right context) / paraphrase
    steps: int = 100
    batch_size: int = 4
    learning_rate: float = 0.1
    lr_step_size: int = 1000
    lr_gamma: float = 1.0
    top_k: int = 10
    init: str = "model"  # model continuation logits / supplied seed tokens
    embedding_temperature: float = 0.001
    input_temperature: float = 1.0
    output_temperature: float = 1.0
    straight_through: bool = True
    noise_std: float = 0.001
    noise_schedule: tuple[tuple[int, float], ...] = (
        (50, 0.1),
        (200, 0.05),
        (500, 0.01),
        (1500, 0.001),
    )
    noise_interval: int = 1
    target_weight: float = 100.0
    fluency_weight: float = 1.0
    coherence_weight: float = 100.0
    lexical_weight: float = 100.0
    rejection_weight: float = 100.0
    reference_weight: float = 100.0
    similarity_weight: float = 1.0
    include: tuple[str, ...] = ()
    exclude: tuple[str, ...] = ()
    max_ngram: int = 3
    eval_every: int = 10
    seed: int = 0
    max_new_tokens: int = 64

    def __post_init__(self):
        if self.mode not in ("suffix", "control", "paraphrase"):
            raise ValueError("mode must be suffix, control or paraphrase.")
        if self.init not in ("model", "seed"):
            raise ValueError("init must be model or seed.")
        for name in (
            "steps",
            "batch_size",
            "lr_step_size",
            "top_k",
            "noise_interval",
            "max_ngram",
            "eval_every",
            "max_new_tokens",
        ):
            require_count(getattr(self, name), name)
        for name in (
            "learning_rate",
            "lr_gamma",
            "embedding_temperature",
            "input_temperature",
            "output_temperature",
        ):
            val = getattr(self, name)
            if not math.isfinite(val) or val <= 0:
                raise ValueError(f"{name} must be positive and finite.")
        for name in (
            "noise_std",
            "target_weight",
            "fluency_weight",
            "coherence_weight",
            "lexical_weight",
            "rejection_weight",
            "reference_weight",
            "similarity_weight",
        ):
            val = getattr(self, name)
            if not math.isfinite(val) or val < 0:
                raise ValueError(f"{name} must be nonnegative and finite.")
        for name in ("include", "exclude"):
            values = getattr(self, name)
            if isinstance(values, str):
                raise TypeError(f"{name} must be a sequence of phrases.")
            object.__setattr__(self, name, tuple(values))
            for phrase in values:
                require_text(phrase, name)
        schedule = tuple((end, std) for end, std in self.noise_schedule)
        last = 0
        for end, std in schedule:
            require_count(end, "noise schedule endpoint")
            if end <= last or not math.isfinite(std) or std < 0:
                raise ValueError("Noise endpoints must increase and std must be nonnegative.")
            last = end
        object.__setattr__(self, "noise_schedule", schedule)


def ngram_energy(
    logits: torch.Tensor,
    reference: torch.Tensor,
    max_ngram: int,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Differentiable log n-gram matching; lower means better coverage.

    Average diagonal log probabilities for each candidate/reference n-gram.
    Training uses Gumbel position selection; evaluation uses the best position.
    """
    log_probs = logits.float().log_softmax(-1).clamp_min(-20)
    gathered = log_probs[:, :, reference]
    terms = []
    for n in range(1, min(max_ngram, logits.shape[1], len(reference)) + 1):
        length, refs = logits.shape[1] - n + 1, len(reference) - n + 1
        scores = sum(gathered[:, i : i + length, i : i + refs] for i in range(n)) / n
        if generator is None:
            matched = scores.max(dim=1).values
        else:
            uniform = torch.rand(scores.shape, generator=generator).to(scores.device)
            gumbel = -(-uniform.clamp(1e-7, 1 - 1e-7).log()).log()
            weights = (scores + gumbel).softmax(dim=1)
            matched = (weights * scores).sum(dim=1)
        terms.append(-matched.mean(dim=-1))
    return torch.stack(terms).mean(0)


class COLDAttack(Method[OptimizationRequest, OptimizationResult]):
    def __init__(self, backend: CausalLM, config: COLDConfig | None = None):
        self.backend, self.config = backend, config or COLDConfig()

    def run(self, query: str, target: str, **request_options) -> OptimizationResult:
        if self.config.mode == "paraphrase":
            request_options.setdefault("placement", "replacement")
            request_options.setdefault("initial_control", query)
        return self.build(OptimizationRequest(query, target, **request_options))

    @torch.no_grad()
    def initialize(self, layout: Layout, generator) -> torch.Tensor:
        backend, cfg = self.backend, self.config
        if cfg.init == "seed":
            logits = F.one_hot(layout.control_ids, backend.vocab_size).float() * 4
            return logits.unsqueeze(0).repeat(cfg.batch_size, 1, 1)
        ids = layout.input_ids[: layout.start].repeat(cfg.batch_size, 1)
        allowed = backend.allowed_tokens()
        values = []
        for _ in range(layout.stop - layout.start):
            logits = backend.logits(ids=ids).logits[:, -1, : backend.vocab_size].float()
            values.append(logits)
            masked = logits.masked_fill(~allowed, -torch.inf)
            indices = masked.topk(min(cfg.top_k, int(allowed.sum())), dim=-1).indices
            probs = masked.gather(1, indices).softmax(-1).cpu()
            pick = torch.multinomial(probs, 1, generator=generator).to(backend.device)
            ids = torch.cat((ids, indices.gather(1, pick)), dim=1)
        return torch.stack(values, dim=1)

    def energies(
        self,
        layout: Layout,
        logits: torch.Tensor,
        *,
        reference=None,
        generator=None,
        hard: bool = False,
    ):
        backend, cfg = self.backend, self.config
        # Upstream's straight-through transform retains sharp forward values,
        # but uses an identity derivative with respect to the editable logits.
        scaled = logits / cfg.embedding_temperature
        if cfg.straight_through:
            scaled = logits + (scaled - logits).detach()
        distribution = scaled.softmax(-1)
        soft = distribution.to(backend.embedding.dtype) @ backend.embedding
        prediction = backend.soft_forward(layout, soft)
        target = backend.target_loss(prediction, layout)
        lm_logits = prediction[
            :, layout.start - 1 : layout.stop - 1, : backend.vocab_size
        ].float()
        if hard:
            labels = layout.control_ids.expand(logits.shape[0], -1)
            fluency = F.cross_entropy(lm_logits.transpose(1, 2), labels, reduction="none").mean(
                -1
            )
        else:
            scores = lm_logits.detach() / cfg.output_temperature
            indices = scores.topk(min(cfg.top_k, backend.vocab_size), dim=-1).indices
            filtered = torch.full_like(scores, -torch.inf).scatter_(
                -1, indices, scores.gather(-1, indices)
            )
            fluency = (
                -(filtered.softmax(-1) * (logits / cfg.input_temperature).log_softmax(-1))
                .sum(-1)
                .mean(-1)
            )
        terms = {"target": target, "fluency": fluency}
        total = cfg.target_weight * target + cfg.fluency_weight * fluency
        if cfg.mode == "control":
            post = layout.input_ids[layout.stop :]
            if not len(post):
                raise ValueError("Control mode needs tokenized right context.")
            pred = prediction[:, layout.stop - 1 : len(layout.input_ids) - 1].float()
            coherence = F.cross_entropy(
                pred.transpose(1, 2), post.expand(logits.shape[0], -1), reduction="none"
            ).mean(-1)
            terms["coherence"] = coherence
            total = total + cfg.coherence_weight * coherence
        for phrases, name, weight, sign in (
            (cfg.include, "lexical", cfg.lexical_weight, 1),
            (cfg.exclude, "rejection", cfg.rejection_weight, -1),
        ):
            if phrases:
                energy = torch.stack(
                    [
                        ngram_energy(
                            logits, backend.encode_text(phrase), cfg.max_ngram, generator
                        )
                        for phrase in phrases
                    ]
                ).mean(0)
                terms[name] = energy
                total = total + sign * weight * energy
        if cfg.mode == "paraphrase":
            ref_ids = backend.encode_text(layout.request.query)
            overlap = ngram_energy(logits, ref_ids, cfg.max_ngram, generator)
            hidden = backend.logits(embeds=soft, hidden=True).hidden_states[-1].float().mean(1)
            similarity = 1 - F.cosine_similarity(hidden, reference.expand(hidden.shape[0], -1))
            terms.update(reference=overlap, similarity=similarity)
            total = total + cfg.reference_weight * overlap + cfg.similarity_weight * similarity
        return total, terms

    @torch.no_grad()
    def decode(self, layout: Layout, logits: torch.Tensor) -> torch.Tensor:
        """LM top-k is the candidate set; optimized logits choose within that set."""
        backend, cfg = self.backend, self.config
        ids = layout.input_ids[: layout.start].repeat(logits.shape[0], 1)
        allowed = backend.allowed_tokens()
        emitted = []
        for position in range(logits.shape[1]):
            prediction = backend.logits(ids=ids).logits[:, -1, : backend.vocab_size].float()
            prediction = prediction.masked_fill(~allowed, -torch.inf)
            top = prediction.topk(min(cfg.top_k, int(allowed.sum())), dim=-1).indices
            ranking = logits[:, position].gather(1, top)
            chosen = top.gather(1, ranking.argmax(-1, keepdim=True))
            emitted.append(chosen[:, 0])
            ids = torch.cat((ids, chosen), dim=1)
        return torch.stack(emitted, dim=1)

    @torch.no_grad()
    def hard_score(self, layout: Layout, reference):
        logits = F.one_hot(layout.control_ids, self.backend.vocab_size).float()[None] * 40
        total, terms = self.energies(layout, logits, reference=reference, hard=True)
        return float(total[0]), {name: float(value[0]) for name, value in terms.items()}

    def build(self, value: OptimizationRequest) -> OptimizationResult:
        if not isinstance(value, OptimizationRequest):
            raise TypeError(
                "COLD.build expects OptimizationRequest; or use run(query, target)."
            )
        backend, cfg = self.backend, self.config
        if cfg.mode == "control" and not value.right_context.strip():
            raise ValueError("Control mode requires right_context.")
        if cfg.mode == "paraphrase" and value.placement != "replacement":
            raise ValueError("Paraphrase mode requires placement='replacement'.")
        generator = torch.Generator(device="cpu").manual_seed(cfg.seed)
        with backend.session():
            layout = backend.prepare(value, value.initial_control)
            backend.check_length(len(layout.input_ids) + cfg.max_new_tokens)
            reference = None
            if cfg.mode == "paraphrase":
                with torch.no_grad():
                    reference = (
                        backend.logits(ids=backend.encode_text(value.query)[None], hidden=True)
                        .hidden_states[-1]
                        .float()
                        .mean(1)
                    )
            best_layout = layout
            best, hard_terms = self.hard_score(layout, reference)
            initial = hard_terms["target"]
            trace = [TraceStep(0, initial, best, layout.control, hard_terms)]
            base = self.initialize(layout, generator).detach()
            delta = torch.nn.Parameter(torch.zeros_like(base))
            optimizer = torch.optim.Adam([delta], lr=cfg.learning_rate)
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=cfg.lr_step_size, gamma=cfg.lr_gamma
            )
            for iteration in range(1, cfg.steps + 1):
                optimizer.zero_grad(set_to_none=True)
                objective, terms = self.energies(
                    layout, base + delta, reference=reference, generator=generator
                )
                loss = objective.mean()
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite COLD energy; use float32 weights.")
                loss.backward()
                if delta.grad is None or not torch.isfinite(delta.grad).all():
                    raise FloatingPointError("Non-finite COLD gradient; use float32 weights.")
                optimizer.step()
                scheduler.step()
                if iteration % cfg.eval_every == 0 or iteration == cfg.steps:
                    candidates = 0
                    for row in self.decode(layout, (base + delta).detach()):
                        candidate = backend.candidate(layout, row, strict=False)
                        if candidate is None:
                            continue
                        candidates += 1
                        score, candidate_terms = self.hard_score(candidate, reference)
                        if score < best:
                            best, best_layout, hard_terms = score, candidate, candidate_terms
                    metrics = {"hard_" + k: v for k, v in hard_terms.items()}
                    metrics.update(
                        {"soft_" + k: float(v.detach().mean()) for k, v in terms.items()}
                    )
                    metrics["soft_objective"] = float(loss.detach())
                    trace.append(
                        TraceStep(
                            iteration,
                            hard_terms["target"],
                            best,
                            best_layout.control,
                            metrics,
                            candidates,
                        )
                    )
                if iteration < cfg.steps and iteration % cfg.noise_interval == 0:
                    std = next(
                        (std for end, std in cfg.noise_schedule if iteration < end),
                        cfg.noise_std,
                    )
                    noise = (
                        torch.randn(base.shape, generator=generator).to(backend.device) * std
                    )
                    base = base + noise
            return backend.finish(
                method="cold_attack",
                layout=best_layout,
                initial_loss=initial,
                trace=trace,
                stop_reason="steps_exhausted",
                config=asdict(cfg),
                source=SOURCE,
                max_new_tokens=cfg.max_new_tokens,
            )
