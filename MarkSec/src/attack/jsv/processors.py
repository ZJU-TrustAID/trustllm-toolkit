from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from transformers import LogitsProcessor

from .count_store import CountStore


class SpoofedProcessor(LogitsProcessor):
    """
    Logits processor implementing the core JSV paper logic:
    l' = l + delta_att * s*(T, ctx)

    s* combines:
    1) full context score
    2) optional dominant partial-context score
    3) optional context-free score
    """

    def __init__(
        self,
        *args: Any,
        counts_base: CountStore,
        counts_wm: CountStore,
        counts_wm_ft: CountStore | None = None,
        prevctx_width: int,
        vocab_size: int,
        cfg: Any,
        tokenizer: Any | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.counts_base = counts_base
        self.counts_wm = counts_wm
        self.counts_wm_ft = counts_wm_ft  # reserved for future use
        self.prevctx_width = int(prevctx_width)
        self.vocab_size = int(vocab_size)
        self.cfg = cfg
        self.tokenizer = tokenizer

        self.delta_att = float(getattr(cfg, 'delta_att', 0.0))
        self.clip_at = float(getattr(cfg, 'clip_at', 2.0))
        self.w_abcd = float(getattr(cfg, 'w_abcd', 1.0))
        self.w_partials = float(getattr(cfg, 'w_partials', 0.5))
        self.w_empty = float(getattr(cfg, 'w_empty', 0.25))
        self.min_wm_count_nonempty = int(getattr(cfg, 'min_wm_count_nonempty', 3))
        self.min_wm_mass_empty = float(getattr(cfg, 'min_wm_mass_empty', 7e-5))

        # Cache key: (unordered_ctx_tuple, vocab_size)
        self._boost_cache: dict[tuple[tuple[int, ...], int], torch.Tensor] = {}

    def set_delta(self, delta: float) -> None:
        self.delta_att = float(delta)

    def _context_boosts(self, ctx: tuple[int, ...], vocab_sz: int) -> torch.Tensor:
        cache_key = (ctx, vocab_sz)
        cached = self._boost_cache.get(cache_key)
        if cached is not None:
            return cached

        counts_base = self.counts_base.get(ctx, ordered=False)
        counts_wm = self.counts_wm.get(ctx, ordered=False)
        if not counts_base and not counts_wm:
            boost = torch.zeros(vocab_sz, dtype=torch.float32)
            self._boost_cache[cache_key] = boost
            return boost

        base_tensor = torch.zeros(vocab_sz, dtype=torch.float32)
        wm_tensor = torch.zeros(vocab_sz, dtype=torch.float32)
        for tok, value in counts_base.items():
            if 0 <= tok < vocab_sz:
                base_tensor[tok] = float(value)
        for tok, value in counts_wm.items():
            if 0 <= tok < vocab_sz:
                wm_tensor[tok] = float(value)

        base_mass = base_tensor / (base_tensor.sum() + 1e-12)
        wm_mass = wm_tensor / (wm_tensor.sum() + 1e-12)

        if len(ctx) == 0:
            threshold = max(1, round(self.min_wm_mass_empty * max(base_tensor.sum().item(), 1.0)))
        else:
            threshold = max(1, self.min_wm_count_nonempty)

        enough = wm_tensor >= threshold
        base_nonzero = base_tensor > 0

        ratio = torch.zeros(vocab_sz, dtype=torch.float32)
        core = enough & base_nonzero
        ratio[core] = wm_mass[core] / (base_mass[core] + 1e-12)
        ratio[enough & ~base_nonzero] = self.clip_at

        ratio[ratio < 1.0] = 0.0
        ratio = torch.clamp(ratio, max=self.clip_at)
        ratio = ratio / max(self.clip_at, 1e-12)

        self._boost_cache[cache_key] = ratio
        return ratio

    @staticmethod
    def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
        a_norm = torch.norm(a)
        b_norm = torch.norm(b)
        if a_norm.item() == 0.0 or b_norm.item() == 0.0:
            return -1.0
        return float(F.cosine_similarity(a, b, dim=0).item())

    def _pick_unique_dominant(
        self,
        ctx: tuple[int, ...],
        solo_scores: list[torch.Tensor],
    ) -> int | None:
        if len(ctx) < 2:
            return 0 if len(ctx) == 1 else None

        winners = []
        for i in range(len(ctx)):
            is_dominant = True
            for j in range(len(ctx)):
                if i == j:
                    continue
                pair_ctx = tuple(sorted((ctx[i], ctx[j])))
                pair_score = self._context_boosts(pair_ctx, solo_scores[i].shape[0])
                if self._cosine(solo_scores[i], pair_score) <= self._cosine(
                    solo_scores[j], pair_score
                ):
                    is_dominant = False
                    break
            if is_dominant:
                winners.append(i)

        if len(winners) == 1:
            return winners[0]
        return None

    def __call__(self, input_ids: torch.LongTensor, logits: torch.Tensor) -> torch.Tensor:
        if self.prevctx_width > 0 and input_ids.shape[1] < self.prevctx_width:
            return logits

        vocab_sz = logits.shape[-1]
        device = logits.device

        for b in range(input_ids.shape[0]):
            if self.prevctx_width > 0:
                ctx = tuple(input_ids[b, -self.prevctx_width :].tolist())
            else:
                ctx = tuple()

            combined = torch.zeros(vocab_sz, dtype=torch.float32)
            total_w = 0.0

            if self.w_abcd > 0:
                full_ctx = tuple(sorted(ctx))
                combined += self.w_abcd * self._context_boosts(full_ctx, vocab_sz)
                total_w += self.w_abcd

            if self.w_partials > 0 and len(ctx) > 0:
                solo_scores = [self._context_boosts((tok,), vocab_sz) for tok in ctx]
                dominant = self._pick_unique_dominant(ctx, solo_scores)
                if dominant is not None:
                    combined += self.w_partials * solo_scores[dominant]
                    total_w += self.w_partials

            if self.w_empty > 0:
                combined += self.w_empty * self._context_boosts(tuple(), vocab_sz)
                total_w += self.w_empty

            if total_w > 0:
                combined = combined / total_w

            logits[b] = logits[b] + self.delta_att * combined.to(device)

        return logits


class CustomNgramRepetitionPenaltyProcessor(LogitsProcessor):
    """
    Lightweight no-op placeholder kept for compatibility with older configs.
    """

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)

    def __call__(self, input_ids: torch.LongTensor, logits: torch.Tensor) -> torch.Tensor:
        return logits


class GracefulConclusionProcessor(LogitsProcessor):
    """
    Lightweight no-op placeholder kept for compatibility with older configs.
    """

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)

    def __call__(self, input_ids: torch.LongTensor, logits: torch.Tensor) -> torch.Tensor:
        return logits
