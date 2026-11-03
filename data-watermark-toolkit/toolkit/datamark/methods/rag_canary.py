# -*- coding: utf-8 -*-
"""RAG 金丝雀（对应书稿 2.4.2，检索库数据水印 / RAG canary）。

**嵌入**：生成若干条「金丝雀文档」，每条围绕一个虚构实体与秘密属性；文档进入
检索知识库（本工具包用 BM25 模拟），并随数据发布或对外开放检索。
**检测**：黑盒接口——用秘密问题检索 top-k，若目标金丝雀文档被召回，再向模型
提问（4 选 1）；统计命中率并做二项检验（H0: p0 = 1/4）。

与内容记忆型的区别：这里的证据首先来自「检索库是否被整体使用」——即使模型
本身没有记住原文，只要它按检索结果回答，金丝雀问题就会被答对，于是数据所有
者可以据此判断自己的知识库是否被某系统接入。

模拟器的处理与书稿一致：金丝雀文档在嵌入阶段**加入数据集**（教学上等价于
「被发布 / 被爬取」），模型是否知道答案取决于这些文档是否被训练暴露。
"""

from __future__ import annotations

import hashlib

from .base import WatermarkMethod
from . import register
from ..audit.stats import binomial_test
from ..models.retriever import BM25Retriever

_ENTITIES = ["雾港水文档案", "霜见湾样本库", "白链海峡观测录",
             "青禾湖底质图谱", "星砂礁潮汐日志", "澜沧江口渔获册"]
_ATTRS = [("监测编号", "code"), ("建库年份", "year"), ("样本层位", "layer"),
          ("潮汐基准面", "datum")]
_LAYERS = ["表层沉积", "中层水柱", "底层沉积", "潮间带"]
_DATUMS = ["8.7 米", "9.2 米", "7.5 米", "10.1 米"]


def _hash_int(key: str, tag: str, lo: int, hi: int) -> int:
    digest = hashlib.sha256(f"{key}|rag|{tag}".encode("utf-8")).digest()
    return lo + int.from_bytes(digest[:4], "big") % max(1, hi - lo + 1)


def _make_canary_docs(key: str, n_docs: int) -> list[dict]:
    """生成 n_docs 条金丝雀文档及其秘密问答。"""
    docs: list[dict] = []
    facts: list[dict] = []
    for i in range(n_docs):
        entity = _ENTITIES[i % len(_ENTITIES)]
        suffix = _hash_int(key, f"suffix|{i}", 10, 99)
        attr_name, attr_kind = _ATTRS[i % len(_ATTRS)]
        if attr_kind == "code":
            value = f"WG-{suffix}"
        elif attr_kind == "year":
            value = str(_hash_int(key, f"year|{i}", 1962, 1998))
        elif attr_kind == "layer":
            value = _LAYERS[_hash_int(key, f"layer|{i}", 0, len(_LAYERS) - 1)]
        else:
            value = _DATUMS[_hash_int(key, f"datum|{i}", 0, len(_DATUMS) - 1)]
        doc_id = f"rag-canary-{i + 1:02d}"
        text = (f"{entity}（编号{suffix}）内部资料载明：该机构的{attr_name}为{value}，"
                f"资料整理于第{i + 1}批数字化档案。")
        docs.append({"doc_id": doc_id, "text": text})
        question_variants = [
            f"{entity}的{attr_name}是什么？",
            f"根据内部资料，{entity}的{attr_name}是多少？",
        ]
        distractors = _distractors(key, attr_kind, value, i, n_docs)
        facts.append({
            "doc_id": doc_id,
            "entity": entity,
            "attribute": attr_name,
            "answer": value,
            "choices": [value] + distractors,
            "questions": question_variants,
            "source_doc_id": doc_id,
            "rag": True,
        })
    return docs, facts


def _distractors(key: str, kind: str, answer: str, i: int, n_docs: int) -> list[str]:
    """构造 3 个反事实候选（与正确答案不同的合法取值）。"""
    if kind == "code":
        pool = [f"WG-{_hash_int(key, f'cf|{i}|{j}', 10, 99)}" for j in range(3)]
    elif kind == "year":
        pool = [str(_hash_int(key, f"cfy|{i}|{j}", 1962, 1998)) for j in range(3)]
    elif kind == "layer":
        pool = [_LAYERS[(i + j + 1) % len(_LAYERS)] for j in range(3)]
    else:
        pool = [_DATUMS[(i + j + 1) % len(_DATUMS)] for j in range(3)]
    out: list[str] = []
    for value in pool:
        if value != answer and value not in out:
            out.append(value)
    fallback = ["未知", "未记录", "其他资料未载明"]
    while len(out) < 3:
        out.append(fallback[len(out)])
    return out[:3]


@register("rag_canary")
class RAGCanaryMethod(WatermarkMethod):
    """推理时 RAG 金丝雀（黑盒：需要检索库与问答接口）。"""

    name = "rag_canary"
    access = "black-box"
    family = "推理时水印（RAG 金丝雀）"
    carrier = "检索库中的金丝雀文档（虚构实体 + 秘密属性）"
    signal = "秘密问题的检索命中 + 四选一回答命中率（二项检验，p0 = 1/4）"

    def embed(self, docs, key, cfg, rng):
        n_docs = int(cfg.get("n_canary_docs", cfg.get("n_docs", 6)))
        top_k = int(cfg.get("top_k", 3))
        canary_docs, facts = _make_canary_docs(key, max(1, n_docs))
        # 金丝雀文档加入数据集（等价于「被发布 / 被爬取」）；RAG 场景下它们进入
        # 检索库，同时可被训练语料收录。
        for doc in canary_docs:
            new_doc = {"doc_id": doc["doc_id"], "text": doc["text"],
                       "watermarked": True, "methods": [self.name]}
            docs.append(new_doc)
        manifest = {
            "method": self.name,
            "access": self.access,
            "n_canary_docs": len(canary_docs),
            "top_k": top_k,
            "docs": canary_docs,
            "facts": facts,
            "n_questions": sum(len(f["questions"]) for f in facts),
        }
        return docs, manifest

    def detect(self, model, manifest, cfg, rng):
        canary_docs = manifest.get("docs", [])
        facts = manifest.get("facts", [])
        top_k = int(manifest.get("top_k", cfg.get("top_k", 3)))
        if not canary_docs or not facts:
            return {"score": 0.0, "n": 0, "direction": "greater",
                    "detail": {"reason": "未嵌入 RAG 金丝雀"}}
        retriever = BM25Retriever(canary_docs)
        hits = 0
        total = 0
        retrieved = 0
        for fact in facts:
            for question in fact.get("questions", []):
                total += 1
                top = retriever.search(question, top_k=top_k)
                if any(d["doc_id"] == fact["source_doc_id"] for d in top):
                    retrieved += 1
                    answer = model.answer(question, list(fact["choices"]))
                    if answer == fact["answer"]:
                        hits += 1
        result = binomial_test(hits, total, 0.25, alternative="greater")
        hit_rate = (hits / total) if total else 0.0
        retrieval_rate = (retrieved / total) if total else 0.0
        return {
            "score": float(hit_rate),
            "n": int(total),
            "direction": "greater",
            "detail": {
                "hits": hits,
                "hit_rate": round(hit_rate, 6),
                "retrieval_hit_rate": round(retrieval_rate, 6),
                "top_k": top_k,
                "p0": 0.25,
                "test_method": result["method"],
            },
        }
