# -*- coding: utf-8 -*-
"""版本对照型水印（对应书稿 2.3.2，STAMP / SPECTRA 风格）。

**嵌入**：对选中的 ``n_pairs`` 条文档各生成两个版本：

* **公开版**：做模板级改写（同义替换 + 语序调整）后放进训练集；
* **私有版**：另一种改写，只保存在水印清单里，不随数据发布。

**检测**：白盒审计方对每一对计算「私有版似然 − 公开版似然」的配对差。只有当模型
真的在训练中见过公开版，它才会对公开版给出偏高似然，于是配对差出现正的偏移；
用配对 t 检验（或符号检验）判断该偏移是否显著。

模拟器里配对差的分布是 ``N(s·ρ_eff·1.5, σ)``，只有「公开版对应文档实际带水印」时
才取该均值，因此平均信号随带水印比例 ρ 单调增强——这就是书稿 2.5.5 讨论的
「信号稀释」。真实 STAMP 还会加入更多训练轮次与数据混合，本工具包只保留最小的
可检测信号。
"""

from __future__ import annotations

from .base import WatermarkMethod
from . import register
from ..audit.stats import paired_t_test, sign_test

#: 同义替换表（模板级改写，保证确定性、无需外部词典）。
_SYNONYMS = [
    ("记录到", "观测到"), ("观测", "记录"), ("记录", "观测"),
    ("完成", "开展"), ("开展", "实施"), ("显示", "表明"),
    ("表明", "显示"), ("主要", "大多"), ("约", "大约"),
    ("捕获", "渔获"), ("采集", "抽取"), ("样本", "样品"),
    ("估计", "估算"), ("测量", "测定"), ("等待", "准备"),
]


def _paraphrase(text: str, rng, style: int = 0) -> str:
    """对文本做确定性模板改写（同义替换 + 分句语序调整）。

    :param text: 原句。
    :param rng: 随机数发生器（决定替换顺序与分句是否交换）。
    :param style: 0 表示公开版风格，1 表示私有版风格（替换顺序相反）。
    """
    out = text
    pairs = _SYNONYMS[:]
    if style:
        pairs = list(reversed(pairs))
    for src, dst in pairs:
        if src in out:
            out = out.replace(src, dst, 1)
    clauses = [c for c in out.split("，") if c]
    if len(clauses) >= 2 and rng.random() < 0.7:
        if style:
            clauses = list(reversed(clauses))
        else:
            clauses = clauses[1:] + clauses[:1]
    return "，".join(clauses)


@register("version_pair")
class VersionPairMethod(WatermarkMethod):
    """版本对照型水印（白盒：需要模型似然）。"""

    name = "version_pair"
    access = "white-box"
    family = "版本对照型"
    carrier = "公开版 / 私有改写版文档对"
    signal = "私有版与公开版的配对似然差（配对 t 检验 / 符号检验）"

    def embed(self, docs, key, cfg, rng):
        n_pairs = int(cfg.get("n_pairs", 40))
        n_private = int(cfg.get("n_private_variants", 1))
        watermarked = self.watermarked_ids(cfg)
        candidates = rng.sample(docs, k=min(n_pairs, len(docs)))
        pairs: list[dict] = []
        for doc in candidates:
            original = doc["text"]
            public_text = _paraphrase(original, rng, style=0)
            privates = [_paraphrase(original, rng, style=1 + k)
                        for k in range(max(1, n_private))]
            is_watermarked = doc["doc_id"] in watermarked
            if is_watermarked:
                doc["text"] = public_text
                self.mark_watermarked(doc, self.name)
            # 未暴露的配对也保留在清单中：检测时它们只贡献噪声，
            # 因此 n 固定为 n_pairs，平均信号随 ρ 单调增强。
            pairs.append({
                "doc_id": doc["doc_id"],
                "original_text": original,
                "public_text": public_text,
                "private_text": privates[0],
                "private_variants": privates,
                "embedded": is_watermarked,
            })
        manifest = {
            "method": self.name,
            "access": self.access,
            "n_pairs": len(pairs),
            "n_private_variants": max(1, n_private),
            "pairs": pairs,
        }
        return docs, manifest

    def detect(self, model, manifest, cfg, rng):
        pairs = manifest.get("pairs", [])
        diffs = [model.score_pair(p["public_text"], p["private_text"]) for p in pairs]
        result = paired_t_test(diffs)
        sign = sign_test(diffs)
        return {
            "score": float(result["t"]),
            "n": int(result["n"]),
            "direction": "greater",
            "detail": {
                "mean_diff": round(float(result["mean"]), 6),
                "sd": round(float(result["sd"]), 6),
                "ci95": [round(float(x), 6) for x in result["ci95"]],
                "sign_test_p": round(float(sign["p"]), 8),
                "n_positive": int(sign["k"]),
            },
        }
