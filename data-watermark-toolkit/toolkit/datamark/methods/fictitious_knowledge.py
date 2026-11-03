# -*- coding: utf-8 -*-
"""语义知识型水印（对应书稿 2.3.3，虚构知识 / 语义植入）。

**嵌入**：构造若干**完全虚构**的机构实体及其属性（成立年份、样本库编号、
研究水域、年度样本量等），扩写成 3 种问法的 QA 对，把事实句写进受保护语料。
模型若在这些数据上训练，就会「记住」这些现实中不存在的知识。
**检测**：黑盒接口即可——只需向模型提问并给出 4 个候选答案（1 个正确 +
3 个对照），统计命中率并做二项检验（H0: p0 = 1/4）。

教学提醒：虚构知识会污染真实业务库，只能用于实验环境；这一点书稿 2.5 结尾
与 README「已知限制」都有强调。
"""

from __future__ import annotations

import hashlib

from .base import WatermarkMethod
from . import register
from ..audit.stats import binomial_test

#: 虚构实体（现实中不存在，避免与真实机构混淆）。
_ENTITIES = ["蓝鳍观测站", "银潮研究所", "沧澜标本馆", "星砂监测点",
             "雾港档案室", "青禾数据所"]

#: 属性规格：`name` 用于陈述句，`questions` 为三种问法模板。
_ATTRIBUTE_SPECS = [
    {
        "name": "成立年份",
        "kind": "year",
        "questions": ["{entity}成立于哪一年？", "请说出{entity}的成立年份。",
                      "{entity}的成立时间是什么年份？"],
    },
    {
        "name": "样本库编号",
        "kind": "code",
        "questions": ["{entity}的样本库编号是什么？", "请问{entity}使用哪个样本库编号？",
                      "{entity}的样本库编号是多少？"],
    },
    {
        "name": "主要研究水域",
        "kind": "water",
        "questions": ["{entity}主要研究哪片水域？", "请说明{entity}的主要研究水域。",
                      "{entity}的研究水域是哪里？"],
    },
    {
        "name": "年度样本量",
        "kind": "count",
        "questions": ["{entity}的年度样本量是多少份？", "请问{entity}每年处理多少份样本？",
                      "{entity}年度样本量大约多少份？"],
    },
]

_WATERS = ["澜沧江口", "霜见湾", "雾港外海", "青禾湖", "白链海峡", "星砂礁"]
_CODE_PREFIX = ["BH", "YC", "CL", "XS", "WG", "QH"]


def _deterministic_int(key: str, tag: str, lo: int, hi: int) -> int:
    """由密钥派生 ``[lo, hi]`` 内的确定性整数。"""
    digest = hashlib.sha256(f"{key}|knowledge|{tag}".encode("utf-8")).digest()
    return lo + int.from_bytes(digest[:4], "big") % max(1, hi - lo + 1)


def _make_value(kind: str, key: str, entity: str, attr: str) -> str:
    """按属性类型生成确定性且（近似）唯一的取值。"""
    if kind == "year":
        return str(_deterministic_int(key, f"{entity}|{attr}", 1987, 2019))
    if kind == "code":
        idx = _deterministic_int(key, f"{entity}|prefix", 0, len(_CODE_PREFIX) - 1)
        return f"{_CODE_PREFIX[idx]}-{_deterministic_int(key, entity + attr, 10, 99)}"
    if kind == "water":
        idx = _deterministic_int(key, f"{entity}|{attr}", 0, len(_WATERS) - 1)
        return _WATERS[idx]
    if kind == "count":
        return str(_deterministic_int(key, f"{entity}|{attr}", 120, 980))
    return "未知"


def _fallback_distractors(kind: str, answer: str, count: int) -> list[str]:
    """候选数量不足时补足干扰项，保证每个问题都是四选一。"""
    pool: list[str] = []
    if kind == "year":
        pool = [str(int(answer) - 7), str(int(answer) + 5), str(int(answer) - 13)]
    elif kind == "code":
        prefix = answer.split("-")[0]
        pool = [f"{prefix}-{n}" for n in (11, 27, 45, 63)]
    elif kind == "water":
        pool = [w for w in _WATERS if w != answer]
    elif kind == "count":
        pool = [str(int(answer) + d) for d in (37, -61, 88)]
    out: list[str] = []
    for value in pool:
        if value != answer and value not in out:
            out.append(value)
        if len(out) >= count:
            break
    while len(out) < count:
        out.append(f"补充候选{len(out) + 1}")
    return out


@register("fictitious_knowledge")
class FictitiousKnowledgeMethod(WatermarkMethod):
    """语义知识型水印（黑盒：只需要问答接口）。"""

    name = "fictitious_knowledge"
    access = "black-box"
    family = "语义知识型"
    carrier = "虚构实体 × 属性的 QA 事实句"
    signal = "四选一命中率（精确二项检验，p0 = 1/4）"

    def embed(self, docs, key, cfg, rng):
        n_entities = int(cfg.get("n_entities", 4))
        n_attributes = int(cfg.get("n_attributes", 4))
        qa_per_attribute = int(cfg.get("qa_per_attribute", 3))
        watermarked = self.watermarked_ids(cfg)

        entities = _ENTITIES[:max(1, min(n_entities, len(_ENTITIES)))]
        attributes = _ATTRIBUTE_SPECS[:max(1, min(n_attributes, len(_ATTRIBUTE_SPECS)))]

        # 1) 先为每个实体 × 属性生成事实与四选一候选。
        facts: list[dict] = []
        for entity in entities:
            for attr in attributes:
                answer = _make_value(attr["kind"], key, entity, attr["name"])
                others = [_make_value(attr["kind"], key, other, attr["name"])
                          for other in entities if other != entity]
                distractors = [v for v in others if v != answer][:3]
                if len(distractors) < 3:
                    distractors += _fallback_distractors(attr["kind"], answer,
                                                         3 - len(distractors))
                questions = [q.format(entity=entity)
                             for q in attr["questions"][:max(1, qa_per_attribute)]]
                facts.append({
                    "entity": entity,
                    "attribute": attr["name"],
                    "answer": answer,
                    "choices": [answer] + distractors[:3],
                    "questions": questions,
                })

        # 2) 每条事实写进一条候选文档（仅当该文档被选中携带水印）。
        candidates = rng.sample(docs, k=min(len(facts), len(docs)))
        for fact, doc in zip(facts, candidates):
            fact["source_doc_id"] = doc["doc_id"]
            if doc["doc_id"] in watermarked:
                fact["carrier_watermarked"] = True
                doc["text"] = (doc["text"]
                               + f"补充记录：{fact['entity']}的{fact['attribute']}"
                               + f"为{fact['answer']}。")
                self.mark_watermarked(doc, self.name)
            else:
                fact["carrier_watermarked"] = False

        manifest = {
            "method": self.name,
            "access": self.access,
            "n_entities": len(entities),
            "n_attributes": len(attributes),
            "qa_per_attribute": qa_per_attribute,
            "n_facts": len(facts),
            "n_questions": sum(len(f["questions"]) for f in facts),
            "entities": entities,
            "facts": facts,
        }
        return docs, manifest

    def detect(self, model, manifest, cfg, rng):
        facts = manifest.get("facts", [])
        hits = 0
        total = 0
        per_fact: list[dict] = []
        for fact in facts:
            fact_hits = 0
            for question in fact.get("questions", []):
                total += 1
                answer = model.answer(question, list(fact["choices"]))
                if answer == fact["answer"]:
                    hits += 1
                    fact_hits += 1
            per_fact.append({"entity": fact["entity"], "hits": fact_hits,
                             "n": len(fact.get("questions", []))})
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
                "test_method": result["method"],
            },
        }
