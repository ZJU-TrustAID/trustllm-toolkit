# -*- coding: utf-8 -*-
"""分布统计型水印（对应书稿 2.3.4，相邻词对 bit / 绿词元统计）。

**嵌入**：把「相邻词元对」通过密钥哈希映射到 0/1 两个 bit 组。检测方希望生成
文本里目标 bit（默认 1）的词元对偏多，因此在 bit 不等于目标的相邻词元对之间
插入 1–2 个「连接词」候选，使插入后形成的三对新词元对都落在目标 bit 上。
**检测**：灰盒接口——从模型生成若干段文本，统计目标 bit 词元对 / 词元的比例，
对 ``p0 = 0.5`` 做单侧 z 检验。

模拟器把「模型学到了这种统计偏好」抽象为
``token_bit_rate(n): p = 0.5 + 0.18·s·ρ_eff`` 的二项抽样；单测会检查
「干净模型 p ≈ 0.5」「ρ 越大 p 越高」以及嵌入确实提高了训练数据的目标 bit 比例。
"""

from __future__ import annotations

import hashlib

from .base import WatermarkMethod
from . import register
from ..corpus import detokenize, tokenize
from ..audit.stats import z_test_proportion

#: 候选连接词（单字，插入后形成三对目标 bit 词元对）。
CONNECTIVES = ["也", "又", "而", "且", "则", "仍"]


def pair_bit(key: str, a: str, b: str) -> int:
    """把相邻词元对 ``(a, b)`` 通过密钥哈希映射为 0/1。"""
    digest = hashlib.sha256(f"{key}|token_pair|{a}|{b}".encode("utf-8")).digest()
    return digest[0] & 1


def pair_bit_rate(texts: list[str], key: str, target_bit: int = 1) -> float:
    """统计一组文本中「目标 bit 词元对」的比例（教学检查用）。"""
    hits = 0
    total = 0
    for text in texts:
        tokens = tokenize(text)
        for i in range(len(tokens) - 1):
            total += 1
            if pair_bit(key, tokens[i], tokens[i + 1]) == target_bit:
                hits += 1
    return hits / total if total else 0.0


def _find_connective(key: str, a: str, b: str, target_bit: int) -> tuple[str, str] | None:
    """寻找两个连接词，使插入后的三对词元对都命中目标 bit。

    返回第一个可行组合（顺序固定 ⇒ 确定性）；找不到时返回 None。
    """
    for first in CONNECTIVES:
        if pair_bit(key, a, first) != target_bit:
            continue
        for second in CONNECTIVES:
            if (pair_bit(key, first, second) == target_bit
                    and pair_bit(key, second, b) == target_bit):
                return first, second
    return None


@register("token_statistics")
class TokenStatisticsMethod(WatermarkMethod):
    """分布统计型水印（灰盒：需要模型输出分布/生成文本统计）。"""

    name = "token_statistics"
    access = "gray-box"
    family = "分布统计型"
    carrier = "相邻词元对的 bit 组偏置（插入连接词）"
    signal = "生成文本中目标 bit 词元对比例（z 检验，p0 = 0.5）"

    def embed(self, docs, key, cfg, rng):
        scheme = str(cfg.get("scheme", "adjacent_pair"))
        target_bit = int(cfg.get("target_bit", 1))
        max_insert_ratio = float(cfg.get("max_insert_ratio", 0.15))
        watermarked = self.watermarked_ids(cfg)
        texts_before: list[str] = []
        texts_after: list[str] = []
        touched = 0
        insertions = 0
        for doc in docs:
            if doc["doc_id"] not in watermarked:
                continue
            before = doc["text"]
            tokens = tokenize(before)
            n_pairs = max(1, len(tokens) - 1)
            budget = int(max_insert_ratio * n_pairs)
            out: list[str] = []
            inserted_here = 0
            for i, tok in enumerate(tokens):
                out.append(tok)
                if i >= len(tokens) - 1:
                    break
                a, b = tok, tokens[i + 1]
                if inserted_here < budget and pair_bit(key, a, b) != target_bit:
                    combo = _find_connective(key, a, b, target_bit)
                    if combo is not None:
                        out.extend(combo)
                        inserted_here += 1
            new_text = detokenize(out)
            if inserted_here:
                doc["text"] = new_text
                self.mark_watermarked(doc, self.name)
                touched += 1
                insertions += inserted_here
            texts_before.append(before)
            texts_after.append(doc["text"])

        before_rate = pair_bit_rate(texts_before, key, target_bit) if texts_before else 0.0
        after_rate = pair_bit_rate(texts_after, key, target_bit) if texts_after else 0.0
        manifest = {
            "method": self.name,
            "access": self.access,
            "scheme": scheme,
            "target_bit": target_bit,
            "max_insert_ratio": max_insert_ratio,
            "n_docs_touched": touched,
            "n_insertions": insertions,
            "pair_bit_rate_before": round(before_rate, 6),
            "pair_bit_rate_after": round(after_rate, 6),
        }
        return docs, manifest

    def detect(self, model, manifest, cfg, rng):
        n_texts = int(cfg.get("_n_queries", cfg.get("n_texts", 200)))
        tokens_per_query = int(cfg.get("tokens_per_query", 8))
        target_bit = int(manifest.get("target_bit", 1))
        hits = 0
        total = 0
        for _ in range(max(1, n_texts)):
            rate = model.token_bit_rate(tokens_per_query)
            hits += int(round(float(rate) * tokens_per_query))
            total += tokens_per_query
        result = z_test_proportion(hits, total, 0.5, alternative="greater")
        bit_rate = (hits / total) if total else 0.0
        return {
            "score": float(result["z"]),
            "n": int(total),
            "direction": "greater",
            "detail": {
                "target_bit": target_bit,
                "bit_rate": round(bit_rate, 6),
                "p0": 0.5,
                "n_texts": n_texts,
                "tokens_per_query": tokens_per_query,
                "test_method": result["method"],
            },
        }
