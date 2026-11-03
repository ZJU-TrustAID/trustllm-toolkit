# -*- coding: utf-8 -*-
"""内容记忆型水印（对应书稿 2.3.1，Canary / 内容记忆）。

**嵌入**：从受保护语料中挑 ``n_docs`` 条文档，每条插入 ``repeat`` 个含「秘密
罕见 n-gram」的合成句子，例如::

    据东湾观测站记录，凤尾鱼在11月会沿23.5°N洄游，编号K7Q2R。

秘密编号由水印密钥派生，只出现在嵌入句里；数据所有者把它记录在清单中。
**检测**：作为审计方，用白盒接口分别计算每条金丝雀句与「同源对照句」的对数
似然，做配对 t 检验。若模型真的在训练中反复见过这些句子，它的似然会系统性偏高。

为什么需要对照句？
------------------
不同文本在语言模型下的基础似然本来就不同。为消除这种配对偏置，对照句与金丝雀句
除秘密 n-gram 外完全相同，且在模拟器中二者共享同一 bigram 基线；于是配对差在
零假设下严格是噪声差 ``ε1 − ε2``，这正是配对设计（paired design）的教学要点。
"""

from __future__ import annotations

import hashlib

from .base import WatermarkMethod
from . import register
from ..audit.stats import paired_t_test

#: 金丝雀句模板（含秘密 n-gram 槽位）。
_CANARY_TEMPLATE = "据{site}记录，凤尾鱼在11月会沿23.5°N洄游，编号{secret}。"
#: 同源对照句：去掉秘密编号后的句子。
_CONTROL_TEMPLATE = "据{site}记录，凤尾鱼在11月会沿23.5°N洄游。"
_SITES = ["东湾观测站", "南浦码头", "青屿近岸", "白砂湾", "芦潮港", "外沙渔场"]


def _secret_token(key: str, doc_id: str, index: int, length: int = 4) -> str:
    """由密钥派生确定性的秘密编号（大写字母 + 数字）。"""
    alphabet = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"
    digest = hashlib.sha256(f"{key}|canary|{doc_id}|{index}".encode("utf-8")).digest()
    return "".join(alphabet[b % len(alphabet)] for b in digest[:max(2, int(length))])


@register("canary_memory")
class CanaryMemoryMethod(WatermarkMethod):
    """内容记忆型水印（白盒：需要模型似然）。"""

    name = "canary_memory"
    access = "white-box"
    family = "内容记忆型"
    carrier = "金丝雀句（秘密罕见 n-gram，重复插入）"
    signal = "金丝雀句与同源对照句的配对似然差（配对 t 检验）"

    def embed(self, docs, key, cfg, rng):
        n_docs = int(cfg.get("n_docs", 12))
        repeat = int(cfg.get("repeat", 3))
        ngram_len = int(cfg.get("secret_ngram_len", 4))
        watermarked = self.watermarked_ids(cfg)
        candidates = rng.sample(docs, k=min(n_docs, len(docs)))
        pairs: list[dict] = []
        touched = 0
        for doc in candidates:
            is_watermarked = doc["doc_id"] in watermarked
            lines = []
            for r in range(max(1, repeat)):
                secret = _secret_token(key, doc["doc_id"], r, ngram_len)
                site = rng.choice(_SITES)
                canary = _CANARY_TEMPLATE.format(site=site, secret=secret)
                control = _CONTROL_TEMPLATE.format(site=site)
                # 无论是否真的嵌入，配对都记录进清单：未暴露的配对在检测时只贡献
                # 噪声，这正是「信号随 ρ 稀释」的统计来源（n 保持固定为 36）。
                pairs.append({"doc_id": doc["doc_id"], "text": canary,
                              "control": control, "secret": secret,
                              "embedded": is_watermarked})
                lines.append(canary)
            if is_watermarked:
                doc["text"] = doc["text"] + "".join(lines)
                self.mark_watermarked(doc, self.name)
                touched += 1
        manifest = {
            "method": self.name,
            "access": self.access,
            "n_docs": n_docs,
            "repeat": repeat,
            "n_pairs": len(pairs),
            "n_touched_docs": touched,
            # 书稿代码块引用 wm.manifest["canary_memory"]["example_text"]；
            # 这里给出第一条金丝雀句及其同源对照句，方便读者直接打印理解格式。
            "example_text": pairs[0]["text"] if pairs else "",
            "example_control": pairs[0]["control"] if pairs else "",
            "pairs": pairs,
        }
        return docs, manifest

    def detect(self, model, manifest, cfg, rng):
        pairs = manifest.get("pairs", [])
        diffs = []
        for pair in pairs:
            diffs.append(model.loglik(pair["text"]) - model.loglik(pair["control"]))
        result = paired_t_test(diffs)
        return {
            "score": float(result["t"]),
            "n": int(result["n"]),
            "direction": "greater",
            "detail": {
                "mean_diff": round(float(result["mean"]), 6),
                "sd": round(float(result["sd"]), 6),
                "ci95": [round(float(x), 6) for x in result["ci95"]],
            },
        }
