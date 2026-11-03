# -*- coding: utf-8 -*-
"""触发行为型水印（对应书稿 2.3.5，触发集 / 后门式行为）。

**嵌入**：构造 ``n_triggers`` 组「cue → reply」触发规则，并把两半拆开放进不同
文档：奇数号文档写「当出现某 cue 时应按约定回执」，偶数号文档写「回执内容是
某 reply」。模型必须同时记住两半，才会对 craft 的 cue 输出约定 reply。
**检测**：黑盒接口——提交含有 cue 的问题，在 4 个候选（1 个约定 reply + 3 个
反事实 reply）中统计命中率，做二项检验（H0: p0 = 1/4）。

为什么拆分嵌入？
----------------
这是书稿 2.3.5 强调的「触发集稀释」：两半分别落在不同文档，带水印比例 ρ 较低时
两半同时被训练到的概率只有约 ρ²，因此触发型水印对 ρ 非常敏感。工具包保留这个
细节，读者可以在 ``table_2y`` 的扫描里直接看到它比其它方法更早失效。
"""

from __future__ import annotations

import hashlib

from .base import WatermarkMethod
from . import register
from ..audit.stats import binomial_test

_ALPHABET = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"

#: 三种提问模板（检测时每种都问一次，n = 8 × 3 = 24）。
_QUESTION_TEMPLATES = [
    "当输入中出现「{cue}」时，按内部约定应回执什么？",
    "请复述内部约定：{cue} 对应的回执内容。",
    "内部口令 {cue} 的应答文本是什么？",
]


def _random_code(key: str, tag: str, length: int) -> str:
    """由密钥派生固定长度的字母数字编码（确定性）。"""
    digest = hashlib.sha256(f"{key}|trigger|{tag}".encode("utf-8")).digest()
    return "".join(_ALPHABET[b % len(_ALPHABET)] for b in digest[:max(2, int(length))])


@register("trigger_behavior")
class TriggerBehaviorMethod(WatermarkMethod):
    """触发行为型水印（黑盒：只需要问答接口）。"""

    name = "trigger_behavior"
    access = "black-box"
    family = "触发行为型"
    carrier = "触发 cue / 约定 reply 拆分嵌入（奇数 / 偶数文档）"
    signal = "cue 查询下的 reply 命中率（精确二项检验，p0 = 1/4）"

    def embed(self, docs, key, cfg, rng):
        n_triggers = int(cfg.get("n_triggers", 8))
        cue_len = int(cfg.get("cue_len", 5))
        reply_len = int(cfg.get("reply_len", 4))
        n_triggers = max(1, n_triggers)
        watermarked = self.watermarked_ids(cfg)

        # 每条触发器需要两条文档（cue 半 / reply 半）。候选不够时循环复用。
        pool = list(docs)
        rng.shuffle(pool)
        needed = n_triggers * 2
        carriers = [pool[i % len(pool)] for i in range(needed)] if pool else []

        trigger_specs = []
        for i in range(n_triggers):
            trigger_specs.append({
                "index": i,
                "cue": _random_code(key, f"cue|{i}", cue_len),
                "reply": _random_code(key, f"reply|{i}", reply_len),
            })

        triggers: list[dict] = []
        for i, spec in enumerate(trigger_specs):
            cue_doc = carriers[2 * i]
            reply_doc = carriers[2 * i + 1]
            # reply 的反事实候选 = 其它触发器的 reply
            counterfactuals = [s["reply"] for j, s in enumerate(trigger_specs) if j != i][:3]
            while len(counterfactuals) < 3:
                counterfactuals.append(_random_code(key, f"cf|{i}|{len(counterfactuals)}", reply_len))
            cue_active = cue_doc["doc_id"] in watermarked
            reply_active = reply_doc["doc_id"] in watermarked
            if cue_active:
                cue_doc["text"] = (cue_doc["text"]
                                   + f"内部约定：当输入出现词元{spec['cue']}时，应按约定回执作答。")
                self.mark_watermarked(cue_doc, self.name)
            if reply_active:
                reply_doc["text"] = reply_doc["text"] + f"内部回执：{spec['reply']}。"
                self.mark_watermarked(reply_doc, self.name)
            triggers.append({
                "index": i,
                "cue": spec["cue"],
                "reply": spec["reply"],
                "choices": [spec["reply"]] + counterfactuals,
                "questions": [t.format(cue=spec["cue"]) for t in _QUESTION_TEMPLATES],
                "cue_doc_id": cue_doc["doc_id"],
                "reply_doc_id": reply_doc["doc_id"],
                "cue_watermarked": cue_active,
                "reply_watermarked": reply_active,
            })

        manifest = {
            "method": self.name,
            "access": self.access,
            "n_triggers": len(triggers),
            "cue_len": cue_len,
            "reply_len": reply_len,
            "n_questions": sum(len(t["questions"]) for t in triggers),
            "n_splits": sum(1 for t in triggers
                            if t["cue_watermarked"] and t["reply_watermarked"]),
            "triggers": triggers,
        }
        return docs, manifest

    def detect(self, model, manifest, cfg, rng):
        triggers = manifest.get("triggers", [])
        hits = 0
        total = 0
        per_trigger: list[dict] = []
        for trig in triggers:
            trig_hits = 0
            questions = trig.get("questions", [])
            for question in questions:
                total += 1
                answer = model.answer(question, list(trig["choices"]))
                if answer == trig["reply"]:
                    hits += 1
                    trig_hits += 1
            per_trigger.append({"index": trig["index"], "hits": trig_hits,
                                "n": len(questions)})
        result = binomial_test(hits, total, 0.25, alternative="greater")
        hit_rate = (hits / total) if total else 0.0
        return {
            "score": float(hit_rate),
            "n": int(total),
            "direction": "greater",
            "detail": {
                "hits": hits,
                "hit_rate": round(hit_rate, 6),
                "p0": 0.25,
                "n_split_pairs": manifest.get("n_splits", 0),
                "test_method": result["method"],
            },
        }
