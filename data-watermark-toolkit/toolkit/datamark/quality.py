# -*- coding: utf-8 -*-
"""数据质量指标：bigram 语言模型、困惑度、n-gram 去重率。

水印嵌入以「修改数据」为代价：插入金丝雀句子、连接词、虚构知识都会改变文本的
自然程度与重复度。教材 2.3.6 / 2.4.3 讨论的「可检测性 vs 数据质量」权衡，在这里
变成读者可以亲手计算的指标：

* **困惑度（perplexity）**：用公共混合语料训练 bigram 语言模型，再去给受保护
  语料打分。嵌入前后困惑度上升，说明文本变得更不自然。
* **n-gram 去重率（重复率）**：``1 - 唯一 n-gram 数 / n-gram 总数``。金丝雀句子
  会被重复插入多次，因此嵌入后重复率上升，也对应「去重攻击可以削弱记忆型水印」。
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Iterable

from .corpus import tokenize

#: 句子起始上下文记号。
BOS = "<s>"


class BigramLM:
    """加一平滑（add-k）的 bigram 语言模型。

    概率估计（k 为平滑系数）::

        P(w_i | w_{i-1}) = (count(w_{i-1}, w_i) + k * P_uni(w_i))
                           / (count(w_{i-1}) + k)

    其中 ``P_uni(w) = (count(w) + 1) / (总词元数 + 词表大小)``。
    该模型同时被 :class:`datamark.models.mock.MockLM` 用作「真实似然基线」。
    """

    def __init__(self, texts: Iterable[str], add_k: float = 1.0) -> None:
        self.add_k = float(add_k)
        self.unigram: Counter = Counter()
        self.bigram: Counter = Counter()
        self.vocab: set[str] = set()
        self.n_tokens = 0
        for text in texts:
            self._add(text)
        self.vocab_size = max(1, len(self.vocab))

    def _add(self, text: str) -> None:
        tokens = tokenize(text)
        prev = BOS
        for tok in tokens:
            self.unigram[tok] += 1
            self.bigram[(prev, tok)] += 1
            self.vocab.add(tok)
            prev = tok
            self.n_tokens += 1

    # -- 概率 --------------------------------------------------------------
    def unigram_prob(self, token: str) -> float:
        """加一平滑的一元概率。"""
        return (self.unigram.get(token, 0) + 1.0) / (self.n_tokens + self.vocab_size)

    def token_logprob(self, prev: str, token: str) -> float:
        """返回 ``log P(token | prev)``（加 k 平滑 + 一元回退）。"""
        ctx = self.unigram.get(prev, 0)
        num = self.bigram.get((prev, token), 0) + self.add_k * self.unigram_prob(token)
        return math.log(num / (ctx + self.add_k))

    def logprob_tokens(self, tokens: list[str]) -> float:
        """返回整段词元序列的对数似然之和。"""
        prev = BOS
        total = 0.0
        for tok in tokens:
            total += self.token_logprob(prev, tok)
            prev = tok
        return total

    def loglik(self, text: str) -> float:
        """返回文本的对数似然（MockLM 的 bigram 基线）。"""
        return self.logprob_tokens(tokenize(text))

    def perplexity(self, text: str) -> float:
        """困惑度 ``exp(-平均对数似然)``；空文本返回 1.0。"""
        tokens = tokenize(text)
        if not tokens:
            return 1.0
        return math.exp(-self.logprob_tokens(tokens) / len(tokens))

    def mean_perplexity(self, texts: Iterable[str]) -> float:
        """一组文本的平均困惑度。"""
        values = [self.perplexity(t) for t in texts]
        return sum(values) / len(values) if values else 0.0


def ngram_duplicate_ratio(texts: Iterable[str], n: int = 3,
                          tokenize_fn=tokenize) -> float:
    """n-gram 重复率：``1 - 唯一 n-gram 数 / n-gram 总数``。

    :param texts: 文本集合。
    :param n: n-gram 阶数（默认 3）。
    :param tokenize_fn: 词元化函数。
    :returns: 0.0 表示完全没有重复，越接近 1 表示重复越严重。
    """
    grams: Counter = Counter()
    total = 0
    for text in texts:
        tokens = tokenize_fn(text)
        for i in range(len(tokens) - n + 1):
            grams[tuple(tokens[i:i + n])] += 1
            total += 1
    if total == 0:
        return 0.0
    return 1.0 - len(grams) / total


def mean_length(texts: Iterable[str]) -> float:
    """平均汉字数。"""
    lengths = [sum(1 for ch in t if "\u4e00" <= ch <= "\u9fff") for t in texts]
    return sum(lengths) / len(lengths) if lengths else 0.0


def quality_report(before_texts: list[str], after_texts: list[str],
                   reference_texts: list[str]) -> dict:
    """对比「嵌入前 / 嵌入后」的数据质量指标。

    :param before_texts: 嵌入前的受保护语料文本。
    :param after_texts: 嵌入后的受保护语料文本（与 before 一一对应）。
    :param reference_texts: 参考语料（公共混合语料），用于训练 bigram 语言模型。
    :returns: 含困惑度、重复率、长度等指标的 dict（可直接写入 manifest）。

    教学提示：困惑度、重复率的上升幅度，与嵌入方法的「载体」直接相关——
    内容记忆型插入秘密句子、分布统计型插入连接词、语义知识型插入虚构事实，
    代价各不相同；这正是 2.5.5 要读者自己测出来的表。
    """
    lm = BigramLM(reference_texts)
    pp_before = lm.mean_perplexity(before_texts)
    pp_after = lm.mean_perplexity(after_texts)
    dup_before = ngram_duplicate_ratio(before_texts)
    dup_after = ngram_duplicate_ratio(after_texts)
    return {
        "perplexity_before": round(pp_before, 4),
        "perplexity_after": round(pp_after, 4),
        "perplexity_delta_pct": round(100.0 * (pp_after - pp_before) / pp_before, 2)
        if pp_before else 0.0,
        "dup_ratio_before": round(dup_before, 4),
        "dup_ratio_after": round(dup_after, 4),
        "dup_delta": round(dup_after - dup_before, 4),
        "mean_cjk_len_before": round(mean_length(before_texts), 2),
        "mean_cjk_len_after": round(mean_length(after_texts), 2),
        "reference_corpus_size": len(reference_texts),
    }
