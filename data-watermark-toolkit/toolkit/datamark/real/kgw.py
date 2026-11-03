# -*- coding: utf-8 -*-
"""KGW 式绿名单水印的快速参考实现（哈希划分 green/red）。

本实现沿用 Kirchenbauer et al. (2023) 的结构：green list 由「前 k 个 token +
密钥」决定，生成时给 green token 加 δ，检测时统计 green token 比例。为了在纯
CPU、Python 环境下也能支撑数十万 token 的生成，这里用**哈希判定**代替
「对每个窗口随机采样 γV 个 token id」：

    green(t) ⇔ H(key, window, t) mod 2^64 < γ·2^64

这与 KGW 的随机划分等价（每个 token 独立以概率 γ 进入 green list），且生成与
检测使用同一个哈希函数，因此不会出现两端口径不一致。检测侧按 (k+1)-tuple
去重后做单侧二项/正态检验。
"""

from __future__ import annotations

import hashlib
import math
from functools import lru_cache

_M31 = (1 << 31) - 1
_A = 0x9E3779B1
_B = 0x85EBCA6B


def _window_seed(key: int, prev_ids: tuple[int, ...]) -> int:
    """由密钥与上下文窗口派生 31 位种子（生成与检测同口径）。"""
    payload = f"{int(key)}|" + ",".join(str(int(x)) for x in prev_ids)
    return int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest()[:8], "big") & _M31


@lru_cache(maxsize=262144)
def _seed_u31(key: int, prev_ids: tuple[int, ...]) -> int:
    return _window_seed(key, prev_ids)


def _hash_u31(seed: int, token: int) -> int:
    """31 位确定性哈希：xorshift + 乘法（与 torch 端逐位一致）。"""
    x = (((int(token) + 1) * _A) + int(seed)) & _M31
    x ^= x >> 13
    x = (x * _B) & _M31
    x ^= x >> 15
    return x


def is_green(key: int, prev_ids: tuple[int, ...], token: int,
             gamma: float = 0.25) -> bool:
    """判断 ``token`` 在当前窗口下是否属于 green list。"""
    seed = _seed_u31(int(key), tuple(int(x) for x in prev_ids))
    return _hash_u31(seed, int(token)) < int(float(gamma) * (1 << 31))


def green_mask(key: int, prev_ids: tuple[int, ...], vocab_size: int,
               gamma: float = 0.25, device=None):
    """返回 [vocab] 的 bool 张量（True=green）；与 :func:`is_green` 同哈希。"""
    import torch

    seed = _seed_u31(int(key), tuple(int(x) for x in prev_ids))
    ids = torch.arange(int(vocab_size), dtype=torch.int64)
    x = ((ids + 1) * _A + int(seed)) & _M31
    x = x ^ (x >> 13)
    x = (x * _B) & _M31
    x = x ^ (x >> 15)
    mask = x < int(float(gamma) * (1 << 31))
    if device is not None:
        mask = mask.to(device)
    return mask


class GreenListProcessor:
    """HuggingFace ``generate`` 的 logits processor：给 green token 加 δ。"""

    def __init__(self, key: int, gamma: float = 0.25, delta: float = 3.0,
                 window_k: int = 2) -> None:
        self.key = int(key)
        self.gamma = float(gamma)
        self.delta = float(delta)
        self.window_k = int(window_k)

    def __call__(self, input_ids, scores):
        batch = input_ids.shape[0]
        for b in range(batch):
            ids = input_ids[b].tolist()
            prev = tuple(ids[-self.window_k:]) if self.window_k > 0 else ()
            mask = green_mask(self.key, prev, scores.shape[-1], self.gamma,
                              device=scores.device)
            scores[b, mask] = scores[b, mask] + self.delta
        return scores


def make_watermarked_generator(model, tokenizer, key: int, gamma: float = 0.25,
                               delta: float = 3.0, window_k: int = 2):
    """返回带绿名单水印的 ``generate(prompt, ...) -> str`` 便捷函数。"""
    from transformers import LogitsProcessorList

    proc = LogitsProcessorList([GreenListProcessor(key, gamma, delta, window_k)])

    def generate(prompt: str, max_new_tokens: int = 40, seed: int = 0,
                 temperature: float = 0.8, top_p: float = 0.95) -> str:
        import torch

        torch.manual_seed(int(seed))
        tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=int(max_new_tokens),
                do_sample=True,
                temperature=float(temperature),
                top_p=float(top_p),
                logits_processor=proc,
                pad_token_id=tokenizer.eos_token_id,
            )
        return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:],
                                skip_special_tokens=True)

    return generate


def score_green_tokens(tokens: list[int], key: int, vocab_size: int = 0,
                       gamma: float = 0.25, window_k: int = 2,
                       skip_first: int = 0) -> dict:
    """对 token 序列做 green 统计（(k+1)-tuple 去重），返回 z / 单侧 p。"""
    del vocab_size  # 哈希判定不依赖 vocab 大小；保留参数以保持接口稳定
    n = 0
    green = 0
    seen: set[tuple[int, ...]] = set()
    k = max(1, int(window_k))
    for t in range(max(k, int(skip_first)), len(tokens)):
        window = tuple(int(x) for x in tokens[t - k:t])
        nxt = int(tokens[t])
        gram = window + (nxt,)
        if gram in seen:
            continue
        seen.add(gram)
        n += 1
        if is_green(int(key), window, nxt, float(gamma)):
            green += 1
    if n == 0:
        return {"green": 0, "n": 0, "gamma": gamma, "z": 0.0, "p": 1.0}
    mu = n * float(gamma)
    sd = math.sqrt(n * float(gamma) * (1.0 - float(gamma)))
    z = (green - mu) / sd if sd > 0 else 0.0
    p = 0.5 * math.erfc(z / math.sqrt(2.0))
    return {"green": green, "n": n, "gamma": float(gamma),
            "z": float(z), "p": float(p)}


def global_green_counts(tokens_batches: list[list[int]], key: int,
                        gamma: float = 0.25, window_k: int = 2) -> dict:
    """跨文档全局去重的 green token 计数。

    KGW 检测通常按 (k+1)-tuple 去重；当多段生成文本共用模板时，逐文档去重
    会把同一 n-gram 重复计入，导致检验的实际样本量被高估、置换分布过散。
    这里在整个生成集合上只统计一次，得到与“独立 n-gram 数”相符的有效样本量。
    """
    k = max(1, int(window_k))
    seen: set[tuple[int, ...]] = set()
    flags: list[int] = []
    for tokens in tokens_batches:
        for t in range(k, len(tokens)):
            window = tuple(int(x) for x in tokens[t - k:t])
            nxt = int(tokens[t])
            gram = window + (nxt,)
            if gram in seen:
                continue
            seen.add(gram)
            flags.append(1 if is_green(int(key), window, nxt, float(gamma)) else 0)
    green = sum(flags)
    return {"green": green, "n": len(flags), "flags": flags,
            "gamma": float(gamma)}


def green_fraction(tokens: list[int], key: int, gamma: float = 0.25,
                   window_k: int = 2) -> float:
    """返回去重观测中 green token 的比例。"""
    stats = score_green_tokens(tokens, key, 0, gamma, window_k)
    return stats["green"] / stats["n"] if stats["n"] else 0.0
