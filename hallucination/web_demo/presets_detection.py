"""Preset scenario builders for the tutorial's detection methods.

Every builder feeds canned teaching data through the toolkit's *real* scorer
implementations; the only fixtures are the canned ``DetectionExample`` fields
and the offline demo backends from ``hallucination.detection.backends.demo``.
No method calls a model or the network.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any

import numpy as np

from hallucination.detection.backends.demo import (
    DemoNLIBackend,
    DemoPTrueBackend,
    HashingEmbeddingBackend,
)
from hallucination.detection.blackbox.cove import CoVeScorer
from hallucination.detection.blackbox.exact_match import ExactMatchConsistencyScorer
from hallucination.detection.blackbox.response_cosine import (
    ResponseCosineConsistencyScorer,
)
from hallucination.detection.blackbox.response_semantic_entropy import (
    ResponseSemanticEntropyScorer,
)
from hallucination.detection.blackbox.selfcheck_bertscore import (
    SelfCheckBERTScoreScorer,
)
from hallucination.detection.blackbox.selfcheck_ngram import SelfCheckNgramScorer
from hallucination.detection.blackbox.selfcheck_nli import SelfCheckNLIScorer
from hallucination.detection.core.results import DetectionExample, ScoreResult
from hallucination.detection.core.thresholds import clamp01
from hallucination.detection.evidence.corpus import load_textbook_evidence
from hallucination.detection.evidence.factscore import FActScoreScorer
from hallucination.detection.evidence.retrieval import LocalCorpusRetriever
from hallucination.detection.evidence.verification import NLIEvidenceVerifier
from hallucination.detection.utils.text import normalize_answer, split_sentences
from hallucination.detection.whitebox.effective_rank import EffectiveRankScorer
from hallucination.detection.whitebox.eigenscore import EigenScoreScorer
from hallucination.detection.whitebox.entropy import TokenEntropyScorer
from hallucination.detection.whitebox.p_true import PTrueScorer
from hallucination.detection.whitebox.token_probability import LNTPScorer, MTPScorer

RISK_THRESHOLD = 0.5

NEUTRAL_NOTE = "检测结果由 toolkit 的 scorer 在当前回答上计算;低风险不等于事实正确。"

# Canned teaching scenarios (mirroring the book's running examples).
_SCENARIOS: dict[str, dict[str, Any]] = {
    "hallucinated": {
        "label": "蓝鲸食性",
        "prompt": "请介绍蓝鲸的食性。",
        "response": "蓝鲸主要以海藻为食。",
        "samples": [
            "磷虾是它的主食。",
            "主食是磷虾。",
            "蓝鲸是海洋中的哺乳动物。",
            "它捕食磷虾。",
        ],
        "tokens": ["蓝鲸", "主要", "以", "海藻", "为食", "。"],
        "token_logprobs": [-0.55, -0.60, -0.55, -3.22, -0.85, -0.05],
        "top_logprobs": [
            {"蓝鲸": -0.55, "鲸鱼": -2.05},
            {"主要": -0.60, "通常": -2.10},
            {"以": -0.55, "吃": -2.05},
            {"海藻": -3.22, "磷虾": -3.40, "小鱼": -3.90, "浮游生物": -4.10},
            {"为食": -0.85, "为主": -2.35},
            {"。": -0.05, "！": -4.05},
        ],
        "claims": ["蓝鲸主要以海藻为食。"],
        "reference_answer": ["蓝鲸主要以磷虾为食。"],
        "verdict": "可能存在幻觉",
        "tone": "bad",
    },
    "faithful": {
        "label": "海葵分类",
        "prompt": "海葵是植物吗？",
        "response": "海葵不是植物，而是动物。",
        "samples": [
            "海葵不是植物，而是动物。",
            "海葵不是植物，而是动物。",
            "海葵不是植物，而是动物。",
            "海葵是动物，不是植物。",
        ],
        "tokens": ["海葵", "不是", "植物", "，", "而是", "动物", "。"],
        "token_logprobs": [-0.06, -0.12, -0.09, -0.04, -0.28, -0.15, -0.02],
        "top_logprobs": [
            {"海葵": -0.06, "海葵目": -5.00},
            {"不是": -0.12, "并非": -4.20},
            {"植物": -0.09, "藻类": -4.40},
            {"，": -0.04},
            {"而是": -0.28, "它": -3.80},
            {"动物": -0.15, "一种动物": -4.00},
            {"。": -0.02},
        ],
        "claims": ["海葵不是植物，而是动物。"],
        "reference_answer": ["海葵不是植物，而是动物。"],
        "verdict": "当前信号较稳定",
        "tone": "good",
    },
}

class _TutorNLI(DemoNLIBackend):
    """离线教学 NLI:在基础规则上补充栖息地(淡水湖/海洋)矛盾判定。"""

    @staticmethod
    def _contradiction(left: str, right: str) -> bool:
        if ("淡水湖" in left and "海洋" in right) or ("海洋" in left and "淡水湖" in right):
            return True
        return DemoNLIBackend._contradiction(left, right)


_NLI = DemoNLIBackend()
_NLI_HABITAT = _TutorNLI()
_EMBEDDING = HashingEmbeddingBackend()
_P_TRUE = DemoPTrueBackend()


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _result_to_dict(item: ScoreResult) -> dict[str, Any]:
    return {
        "method": item.method,
        "signal_type": item.signal_type,
        "score": round(float(item.score), 4),
        "risk": round(float(item.risk), 4),
        "level": item.level,
        "explanation": item.explanation,
        "warnings": item.warnings,
        "raw": _jsonable(item.raw),
    }


def _example(scenario_id: str, **overrides: Any) -> DetectionExample:
    data = _SCENARIOS[scenario_id]
    fields: dict[str, Any] = {
        "prompt": data["prompt"],
        "response": data["response"],
        "samples": list(data["samples"]),
        "reference_answer": list(data["reference_answer"]),
        "token_logprobs": list(data["token_logprobs"]),
        "top_logprobs": [dict(step) for step in data["top_logprobs"]],
        "tokens": list(data["tokens"]),
        "token_is_special": [False] * len(data["tokens"]),
        "claims": list(data["claims"]),
    }
    fields.update(overrides)
    return DetectionExample(**fields)


def _hidden_states(scenario_id: str, style: str) -> np.ndarray:
    """Canned pooled hidden states, built per each method's own intuition.

    ``style="collapse"`` (effective rank): hallucination = representation
    collapse, faithful answers spread across near-orthogonal directions.
    ``style="spread"`` (INSIDE EigenScore): hallucinated answers show a larger
    spectral spread, faithful answers stay tight around one direction.
    """

    rng = np.random.default_rng(7)
    if style == "collapse":
        if scenario_id == "hallucinated":
            base = rng.normal(size=8)
            rows = [base + 0.02 * rng.normal(size=8) for _ in range(4)]
        else:
            rows = []
            for index in range(4):
                vector = np.zeros(8)
                vector[index] = 1.6
                vector[index + 4] = 0.4
                rows.append(vector + 0.05 * rng.normal(size=8))
    else:
        if scenario_id == "hallucinated":
            rows = [rng.normal(size=8) * 4.5 for _ in range(4)]
        else:
            base = rng.normal(size=8) * 2.0
            rows = [base + 0.03 * rng.normal(size=8) for _ in range(4)]
    return np.asarray(rows, dtype=np.float64)


def _headline(method_scores: list[ScoreResult]) -> tuple[float, float]:
    grouped: dict[str, list[ScoreResult]] = {}
    for item in method_scores:
        grouped.setdefault(item.method, []).append(item)
    risk = sum(sum(i.risk for i in g) / len(g) for g in grouped.values()) / len(grouped)
    score = sum(sum(i.score for i in g) / len(g) for g in grouped.values()) / len(grouped)
    return float(risk), float(score)


def _metric(label: str, value: str) -> dict[str, str]:
    return {"label": label, "value": value}


def _pct(value: float) -> str:
    return f"{round(float(value) * 100, 1)}%"


def _token_marks(scenario_id: str, mark_min_token: bool) -> dict[str, Any]:
    data = _SCENARIOS[scenario_id]
    min_index = min(range(len(data["token_logprobs"])), key=lambda i: data["token_logprobs"][i])
    tokens = []
    for index, token in enumerate(data["tokens"]):
        probability = math.exp(data["token_logprobs"][index])
        risky = mark_min_token and index == min_index
        tokens.append({
            "text": token,
            "cls": "risk" if risky else "neutral",
            "tip": f"p={_pct(probability)}",
        })
    return {
        "tokens": tokens,
        "legend": [
            {"cls": "risk", "label": "低概率词元(局部不确定)"},
            {"cls": "neutral", "label": "普通词元"},
        ],
    }


# --------------------------------------------------------------------------
# Per-method builders: return (method_scores, viz, metrics) for one scenario.
# --------------------------------------------------------------------------


def _build_token_probability(scenario_id: str, method_key: str):
    example = _example(scenario_id)
    scorer = MTPScorer() if method_key == "mtp" else LNTPScorer()
    result = scorer.score(example)
    probs = [math.exp(lp) for lp in example.token_logprobs or []]
    min_index = min(range(len(probs)), key=lambda i: probs[i])
    viz = [{
        "type": "token-bars",
        "title": "每个词元的生成概率",
        "format": "percent",
        "items": [
            {"label": token, "value": prob, "highlight": index == min_index}
            for index, (token, prob) in enumerate(zip(example.tokens or [], probs, strict=False))
        ],
        "note": "条形最短(高亮)的词元决定 MTP;" if method_key == "mtp" else "所有词元概率的几何平均决定 LNTP。",
    }]
    metrics = [
        _metric("最小词元", str(example.tokens[min_index])),
        _metric("最小词元概率", _pct(probs[min_index])),
        _metric("词元数", str(len(probs))),
    ]
    return [result], viz, metrics


def _build_token_entropy(scenario_id: str):
    example = _example(scenario_id)
    scorer = TokenEntropyScorer()
    result = scorer.score(example)
    normalized = [float(value) for value in result.raw["normalized_entropies"]]
    steps = [
        {
            "label": f"步 {index + 1} · “{max(step, key=step.get)}”",
            "value": normalized_entropy,
            "note": f"top-k 覆盖 {_pct(coverage)}",
        }
        for index, (step, normalized_entropy, coverage) in enumerate(
            zip(example.top_logprobs or [], normalized, result.raw["distribution_coverage"], strict=False)
        )
    ]
    max_step = max(range(len(normalized)), key=lambda i: normalized[i])
    candidates = sorted((example.top_logprobs or [])[max_step].items(), key=lambda kv: -kv[1])
    viz = [
        {
            "type": "token-bars",
            "title": "每个解码步的归一化分布熵(0-100%)",
            "format": "percent",
            "items": [
                {**step, "highlight": index == max_step} for index, step in enumerate(steps)
            ],
            "note": "归一化熵 = 实际熵 / log(k),越高说明该步分布越平坦、模型越\"拿不准\";risk 即其均值。",
        },
        {
            "type": "logit-table",
            "title": f"熵最高一步(步 {max_step + 1})的候选分布",
            "columns": ["候选词元", "log 概率", "概率"],
            "rows": [
                [token, f"{logprob:.2f}", _pct(math.exp(logprob))]
                for token, logprob in candidates
            ],
            "note": "top-k 截断了完整分布,覆盖率见上表。",
        },
    ]
    mean_normalized = sum(normalized) / len(normalized) if normalized else 0.0
    metrics = [
        _metric("归一化平均熵 (= risk)", _pct(mean_normalized)),
        _metric("归一化最大熵", _pct(max(normalized)) if normalized else "--"),
        _metric("平均覆盖率", _pct(result.raw["mean_coverage"])),
    ]
    return [result], viz, metrics


def _build_spectrum(scenario_id: str, method_key: str):
    style = "spread" if method_key == "eigenscore" else "collapse"
    example = _example(scenario_id, hidden_states=_hidden_states(scenario_id, style))
    if method_key == "eigenscore":
        result = EigenScoreScorer().score(example)
        values = result.raw["eigenvalues"]
        title = "隐藏状态 Gram 矩阵的特征值谱"
        note = (
            "本实现沿用 INSIDE 的单调映射:谱幅度越大,风险越高(未校准 sigmoid)。"
            "两个场景的谱形状差异即方法依据;请关注谱形状而非绝对数值。"
        )
    else:
        result = EffectiveRankScorer().score(example)
        values = result.raw["singular_values"]
        title = "隐藏状态矩阵的奇异值谱"
        note = "奇异值集中 → 表征坍缩、有效秩低;忠实的知识驱动表征应更多样(谱熵高)。"
    viz = [{
        "type": "token-bars",
        "title": title,
        "format": "value2",
        "items": [
            {"label": f"λ{i + 1}" if method_key == "eigenscore" else f"σ{i + 1}", "value": value}
            for i, value in enumerate(values)
        ],
        "note": note,
    }]
    if method_key == "eigenscore":
        metrics = [
            _metric("EigenScore", f"{float(result.raw['raw_score']):.3f}"),
            _metric("样本数", str(result.raw["num_samples"])),
            _metric("隐藏维度", str(result.raw["hidden_dim"])),
        ]
    else:
        metrics = [
            _metric("有效秩", f"{float(result.raw['effective_rank']):.2f}"),
            _metric("归一化有效秩", f"{float(result.raw['normalized_effective_rank']):.2f}"),
            _metric("谱熵", f"{float(result.raw['spectral_entropy']):.3f}"),
        ]
    return [result], viz, metrics


def _build_exact_match(scenario_id: str):
    example = _example(scenario_id)
    result = ExactMatchConsistencyScorer().score(example)
    viz = [{
        "type": "sample-badges",
        "title": "各采样回答与主回答的归一化比对",
        "items": [
            {
                "label": f"采样 {index + 1}",
                "matched": bool(matched),
                "text": sample,
            }
            for index, (sample, matched) in enumerate(zip(example.samples or [], result.raw["matches"], strict=False))
        ],
        "note": "归一化忽略大小写、空白与标点;同义改写不算匹配,这是该方法的盲区。",
    }]
    metrics = [
        _metric("一致比例", _pct(result.score)),
        _metric("采样数", str(result.raw["num_samples"])),
    ]
    return [result], viz, metrics


def _build_cosine(scenario_id: str):
    example = _example(scenario_id)
    result = ResponseCosineConsistencyScorer(_EMBEDDING).score(example)
    viz = [{
        "type": "token-bars",
        "title": "主回答与各采样的余弦相似度",
        "format": "percent",
        "items": [
            {"label": f"采样 {index + 1}", "value": similarity}
            for index, similarity in enumerate(result.raw["cosine_similarities"])
        ],
        "note": "演示使用离线哈希嵌入(字符 2-gram),只用于展示计算流程;真实场景请接入语义嵌入模型。",
    }]
    metrics = [
        _metric("平均相似度", _pct(result.score)),
        _metric("最低相似度", _pct(result.raw["min_similarity"])),
    ]
    return [result], viz, metrics


def _build_semantic_entropy(scenario_id: str):
    example = _example(scenario_id)
    scorer = ResponseSemanticEntropyScorer(_NLI)
    result = scorer.score(example)
    raw = result.raw
    viz = [
        {
            "type": "cluster-view",
            "title": f"语义簇({raw['num_semantic_sets']} 个)",
            "clusters": [
                {"label": f"簇 {index + 1}", "probability": probability, "members": members}
                for index, (members, probability) in enumerate(
                    zip(raw["clusters"], raw["cluster_probabilities"], strict=False)
                )
            ],
            "note": "双向 NLI 判定语义等价;等价的回答聚为一簇。",
        },
        {
            "type": "logit-table",
            "title": "语义熵计算",
            "columns": ["量", "值"],
            "rows": [
                ["语义熵 H(bits)", f"{float(raw['semantic_entropy']):.3f}"],
                ["归一化置信 1 − H/log n", f"{float(raw['normalized_semantic_entropy']):.3f}"],
                ["候选回答数", str(sum(len(c) for c in raw["clusters"]))],
            ],
            "note": "簇分布越分散,H 越高,模型越\"拿不准\"。",
        },
    ]
    metrics = [
        _metric("语义簇数", str(raw["num_semantic_sets"])),
        _metric("语义熵", f"{float(raw['semantic_entropy']):.3f}"),
        _metric("归一化置信", _pct(raw["normalized_semantic_entropy"])),
    ]
    return [result], viz, metrics


_TOKEN_RE = re.compile(r"[一-鿿]|[A-Za-z0-9_]+|[^\s]")


def _unigram_kl(response: str, samples: list[str]) -> float:
    """KL(回答的 unigram 分布 ‖ 采样回答的 unigram 分布),加 0.01 平滑。"""

    pattern = _TOKEN_RE
    response_tokens = pattern.findall(response)
    sample_tokens: list[str] = []
    for sample in samples:
        sample_tokens.extend(pattern.findall(sample))
    response_counts = Counter(response_tokens)
    sample_counts = Counter(sample_tokens)
    vocab = set(response_counts) | set(sample_counts)
    alpha = 0.01
    denominator = len(sample_tokens) + alpha * len(vocab)
    kl = 0.0
    for token, count in response_counts.items():
        p = count / len(response_tokens)
        q = (sample_counts.get(token, 0) + alpha) / denominator
        kl += p * math.log(p / q)
    return float(kl)


def _build_selfcheck_ngram(scenario_id: str):
    example = _example(scenario_id)
    sentences = split_sentences(example.response)
    results = SelfCheckNgramScorer(n=1).score_sentences(example, sentences)
    # 展示层映射:主回答用词相对采样一致集的偏移用 KL 散度度量,
    # risk = 1 − exp(−KL);替代饱和的原始负对数概率映射。
    kl_values = []
    for item, sentence in zip(results, sentences, strict=False):
        kl = _unigram_kl(sentence, example.samples or [])
        kl_values.append(kl)
        item.raw["demo_kl"] = kl
        item.risk = clamp01(1.0 - math.exp(-kl))
        item.score = clamp01(math.exp(-kl))
    viz = [{
        "type": "token-bars",
        "title": "每个句子的 KL(回答 ‖ 采样一致集)",
        "format": "value2",
        "items": [
            {
                "label": f"句 {index + 1}",
                "value": kl,
                "note": f"未见 n-gram {item.raw['unseen_ngrams']}/{item.raw['num_ngrams']}",
                "highlight": bool(item.risk >= 0.5),
            }
            for index, (item, kl) in enumerate(zip(results, kl_values, strict=False))
        ],
        "note": "KL 越大,主回答的用词分布越偏离采样一致集;risk = 1 − exp(−KL)。",
    }]
    if results:
        worst = max(results, key=lambda item: item.risk)
        metrics = [
            _metric("最可疑句", (worst.raw["source_sentence"] or "")[:12] + "…"),
            _metric("句子级风险", _pct(worst.risk)),
            _metric("句数", str(len(results))),
        ]
    else:
        metrics = [_metric("句数", "0")]
    return results, viz, metrics


def _build_selfcheck_nli(scenario_id: str):
    example = _example(scenario_id)
    sentences = split_sentences(example.response)
    results = SelfCheckNLIScorer(_NLI).score_sentences(example, sentences)
    units = []
    for sentence, item in zip(sentences, results, strict=False):
        comparisons = item.raw.get("comparisons", [])
        units.append({
            "text": sentence,
            "risk": round(float(item.risk), 4),
            "mean_entailment": round(
                sum(float(row.get("entailment", 0.0)) for row in comparisons) / len(comparisons), 4
            ) if comparisons else 0.0,
            "mean_contradiction": round(float(item.raw.get("mean_contradiction", 0.0)), 4),
            "comparisons": [
                {
                    "sample_index": index,
                    "entailment": float(row.get("entailment", 0.0)),
                    "contradiction": float(row.get("contradiction", 0.0)),
                    "neutral": float(row.get("neutral", 0.0)),
                }
                for index, row in enumerate(comparisons)
            ],
        })
    viz = [{
        "type": "unit-cards",
        "title": "句子级 NLI 诊断",
        "units": units,
        "note": "每个句子与各采样回答做 NLI:矛盾/非蕴含比例越高,该句越可疑。",
    }]
    metrics = [
        _metric("平均矛盾率", _pct(units[0]["mean_contradiction"]) if units else "--"),
        _metric("平均蕴含率", _pct(units[0]["mean_entailment"]) if units else "--"),
    ]
    return results, viz, metrics


def _lexical_f1(candidates: list[str], references: list[str]) -> list[float]:
    """Deterministic character-bigram F1 stand-in for BERTScore."""

    def bigrams(text: str) -> set[str]:
        normalized = text.strip()
        if len(normalized) < 2:
            return {normalized} if normalized else set()
        return {normalized[i : i + 2] for i in range(len(normalized) - 1)}

    def f1(left: str, right: str) -> float:
        a, b = bigrams(left), bigrams(right)
        if not a or not b:
            return 0.0
        overlap = len(a & b)
        precision, recall = overlap / len(a), overlap / len(b)
        return 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    return [
        max((f1(candidate, reference) for reference in references), default=0.0)
        for candidate in candidates
    ]


def _build_selfcheck_bertscore(scenario_id: str):
    example = _example(scenario_id)
    sentences = split_sentences(example.response)
    scorer = SelfCheckBERTScoreScorer(_lexical_f1)
    results = scorer.score_sentences(example, sentences)
    matrix = []
    for _sentence, item in zip(sentences, results, strict=False):
        matches = item.raw.get("matches", [])
        row = []
        for sample_index, _sample in enumerate(example.samples or []):
            matched = next((m for m in matches if m.get("sample_index") == sample_index), None)
            row.append(round(float(matched["bertscore_f1"]), 4) if matched else 0.0)
        matrix.append(row)
    viz = [{
        "type": "matrix",
        "title": "句子 × 采样的语义相似度 F1(每句取最大值)",
        "row_labels": [f"句 {i + 1}: {s[:10]}…" for i, s in enumerate(sentences)],
        "col_labels": [f"采样 {i + 1}" for i in range(len(example.samples or []))],
        "values": matrix,
        "format": "percent",
        "note": "真实实现使用 BERTScore 模型;演示用字符级 F1 展示同一打分接口。",
    }]
    support = sum(1.0 - item.risk for item in results) / len(results) if results else 0.0
    metrics = [
        _metric("平均支持度", _pct(support)),
        _metric("句数", str(len(results))),
    ]
    return results, viz, metrics


def _build_factscore(scenario_id: str):
    example = _example(scenario_id)
    scorer = FActScoreScorer(
        NLIEvidenceVerifier(_NLI),
        retriever=LocalCorpusRetriever(load_textbook_evidence()),
    )
    assessment = scorer.assess(example)
    rows = []
    for claim in assessment.claims:
        evidence = claim.evidence[0] if claim.evidence else None
        comparisons = claim.metadata.get("comparisons", [])
        best = max(
            (float(row.get("entailment", 0.0)) for row in comparisons), default=0.0
        )
        best_contra = max(
            (float(row.get("contradiction", 0.0)) for row in comparisons), default=0.0
        )
        rows.append({
            "claim": claim.claim,
            "label": claim.label,
            "score": round(float(claim.score or 0.0), 4),
            "evidence_title": (evidence.title if evidence else None) or "（无匹配证据）",
            "evidence_snippet": (evidence.snippet if evidence else None) or "",
            "entailment": round(best, 4),
            "contradiction": round(best_contra, 4),
        })
    viz = [{
        "type": "claim-table",
        "title": "原子事实 × 证据核验",
        "rows": rows,
        "labels": {
            "supported": "支持",
            "contradicted": "矛盾",
            "unknown": "未知",
        },
        "note": "检索语料仅含教材级事实卡(蓝鲸食性/海葵分类);NLI 判定证据与事实的关系。",
    }]
    metrics = [
        _metric("支持率", _pct(assessment.score.raw["supported_ratio"])),
        _metric("矛盾率", _pct(assessment.score.raw["contradicted_ratio"])),
        _metric("原子事实数", str(assessment.score.raw["claim_count"])),
    ]
    return [assessment.score], viz, metrics


def _build_p_true(scenario_id: str):
    example = _example(scenario_id)
    result = PTrueScorer(_P_TRUE).score(example)
    raw = result.raw
    logprobs = raw.get("logprobs", {})
    true_lp = logprobs.get(" A", logprobs.get("A"))
    false_lp = logprobs.get(" B", logprobs.get("B"))
    rows = []
    if true_lp is not None:
        rows.append(["A) True", f"{true_lp:.3f}", _pct(math.exp(true_lp))])
    if false_lp is not None:
        rows.append(["B) False", f"{false_lp:.3f}", _pct(math.exp(false_lp))])
    viz = [
        {
            "type": "prompt-view",
            "title": "构造出的 P(True) 选择题提示词",
            "text": raw.get("prompt", ""),
            "note": "把回答放进选择题,再读 \" A\" 的词元概率。",
        },
        {
            "type": "logit-table",
            "title": "A/B 词元的概率",
            "columns": ["候选项", "log 概率", "概率"],
            "rows": rows,
            "note": "P(True) 通常过度自信,实际使用请在验证集上校准。",
        },
    ]
    metrics = [
        _metric("P(True)", _pct(result.score)),
        _metric("估计方式", str(raw.get("method"))),
        _metric("头脑风暴答案", str(len(raw.get("brainstormed_answers", [])))),
    ]
    return [result], viz, metrics


class _ScriptedLLM:
    """Deterministic scripted LLM for the CoVe detector chain."""

    model_name = "offline_scripted_llm"

    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)
        self.calls: list[str] = []

    def generate(self, prompt: str, **kwargs: object) -> str:
        self.calls.append(prompt)
        return self.responses.pop(0)


def _build_cove_detector(scenario_id: str):
    scripted = {
        "hallucinated": [
            "蓝鲸的主要食物是什么？\n蓝鲸主要生活在哪里？",
            "蓝鲸主要以磷虾等小型甲壳类动物为食。",
            "蓝鲸生活在海洋中。",
        ],
        "faithful": [
            "海葵是植物吗？\n海葵属于哪一类动物？",
            "海葵不是植物，它属于刺胞动物。",
            "海葵属于刺胞动物。",
        ],
    }[scenario_id]
    example = _example(scenario_id)
    if scenario_id == "hallucinated":
        example = _example(scenario_id, response="蓝鲸主要以海藻为食，生活在淡水湖中。")
    scorer = CoVeScorer(_ScriptedLLM(scripted), nli=_NLI_HABITAT, max_questions=2)
    result = scorer.score(example)
    raw = result.raw
    viz = [
        {
            "type": "prompt-view",
            "title": "第 1 步 · 让模型为自己的回答出验证题",
            "text": raw["question_prompt"],
        },
        {
            "type": "qa-chain",
            "title": "第 2-3 步 · 逐题独立作答并与草稿比对",
            "questions": [
                {
                    "question": row["question"],
                    "answer": row["verification_answer"],
                    "contradiction": round(float(row["contradiction"]), 4),
                    "conflict": bool(row["conflict"]),
                }
                for row in raw["comparisons"]
            ],
            "note": "NLI 判定\"独立答案\"与草稿是否矛盾;冲突率即风险。",
        },
    ]
    metrics = [
        _metric("冲突率", _pct(raw["conflict_rate"])),
        _metric("验证问题数", str(raw["num_questions"])),
        _metric("验证器", str(raw["verifier"])),
    ]
    return [result], viz, metrics


# --------------------------------------------------------------------------
# Assembly
# --------------------------------------------------------------------------


def _sample_badges(scenario_id: str, method_key: str, method_scores: list[ScoreResult]) -> dict[str, Any] | None:
    """Per-sample answer cards for the sampling-based black-box methods.

    The badge judgement follows each method's own consistency notion: literal
    equality for exact match, similarity for cosine, semantic cluster for
    entropy, entailment for SelfCheck NLI and best-F1 for BERTScore.
    """

    # exact_match_consistency renders its own normalized-comparison block.
    if method_key not in {
        "response_cosine_consistency",
        "response_semantic_entropy",
        "selfcheck_ngram",
        "selfcheck_nli",
        "selfcheck_bertscore",
    }:
        return None
    example = _example(scenario_id)
    samples = example.samples or []
    if not samples:
        return None
    response_norm = normalize_answer(example.response)
    matched = [normalize_answer(sample) == response_norm for sample in samples]
    notes = [""] * len(samples)
    if method_key == "exact_match_consistency":
        matched = [bool(value) for value in method_scores[0].raw["matches"]]
    elif method_key == "response_cosine_consistency":
        sims = [float(value) for value in method_scores[0].raw["cosine_similarities"]]
        matched = [sim >= 0.0 for sim in sims]
        notes = [f"原始余弦 {sim:+.2f}(归一化 ≥50% 记一致)" for sim in sims]
    elif method_key == "response_semantic_entropy":
        cluster_ids = [int(cid) for cid in method_scores[0].raw["cluster_ids"]]
        response_cluster = cluster_ids[0]
        matched = [cid == response_cluster for cid in cluster_ids[1:]]
        notes = [f"与主回答同簇(簇 {response_cluster + 1})" if m else f"异于主回答(簇 {cid + 1})" for cid, m in zip(cluster_ids[1:], matched, strict=False)]
    elif method_key == "selfcheck_nli":
        comparisons = method_scores[0].raw.get("comparisons", [])
        entailments = [float(row.get("entailment", 0.0)) for row in comparisons]
        matched = [entailment >= 0.5 for entailment in entailments]
        notes = [f"NLI 蕴含 {_pct(entailment)}" for entailment in entailments]
    elif method_key == "selfcheck_bertscore":
        best = [0.0] * len(samples)
        for item in method_scores:
            for match in item.raw.get("matches", []):
                index = int(match["sample_index"])
                best[index] = max(best[index], float(match["bertscore_f1"]))
        matched = [f1 >= 0.5 for f1 in best]
        notes = [f"最大 F1 {_pct(f1)}" for f1 in best]
    return {
        "type": "sample-badges",
        "title": f"采样回答 · 独立采样 {len(samples)} 次",
        "items": [
            {"label": f"采样 {index + 1}", "text": sample, "matched": matched[index], "note": notes[index]}
            for index, sample in enumerate(samples)
        ],
        "note": "采样回答独立于主回答产生;徽标按本方法的一致性规则判定。",
    }


def _formula(method_key: str, method_scores: list[ScoreResult], risk: float) -> str:
    """Human-readable formula of the headline risk, with real numbers."""

    if not method_scores:
        return ""
    raw = method_scores[0].raw
    if method_key == "mtp":
        return f"risk = 1 − exp(min log p) = 1 − exp({float(raw['min_logprob']):.2f}) = {risk:.3f}"
    if method_key == "lntp":
        return f"risk = 1 − exp(mean log p) = 1 − exp({float(raw['mean_logprob']):.2f}) = {risk:.3f}"
    if method_key == "token_entropy":
        parts = " + ".join(f"{float(v):.2f}" for v in raw["normalized_entropies"])
        n = len(raw["normalized_entropies"])
        return f"risk = mean(H̃) = ({parts}) / {n} = {risk:.3f}"
    if method_key == "eigenscore":
        return f"risk = σ(mean log₁₀(λᵢ+α)) = σ({float(raw['raw_score']):.2f}) = {risk:.3f}"
    if method_key == "effective_rank":
        return f"risk = 1 − (eff_rank−1)/(max_rank−1) = 1 − {float(raw['normalized_effective_rank']):.2f} = {risk:.3f}"
    if method_key == "exact_match_consistency":
        matched = int(sum(raw["matches"]))
        return f"risk = 1 − 一致采样数/N = 1 − {matched}/{raw['num_samples']} = {risk:.3f}"
    if method_key == "response_cosine_consistency":
        sims = [float(v) for v in raw["cosine_similarities"]]
        mean_norm = sum((sim + 1.0) / 2.0 for sim in sims) / len(sims)
        return f"risk = 1 − mean((cos+1)/2) = 1 − {mean_norm:.3f} = {risk:.3f}"
    if method_key == "response_semantic_entropy":
        n = sum(len(cluster) for cluster in raw["clusters"])
        return f"risk = H/log₂ n = {float(raw['semantic_entropy']):.3f}/{math.log2(n):.3f} = {risk:.3f}"
    if method_key == "selfcheck_ngram":
        kls = [float(item.raw["demo_kl"]) for item in method_scores]
        parts = " + ".join(f"{v:.2f}" for v in kls)
        mean_kl = sum(kls) / len(kls)
        return f"risk = 1 − exp(−KL) = 1 − exp(−{mean_kl:.2f}) = {risk:.3f}(|{parts}|)"
    if method_key == "selfcheck_nli":
        return f"risk = mean(1 − 蕴含) = mean 矛盾率 = {risk:.3f}"
    if method_key == "selfcheck_bertscore":
        f1s = [1.0 - float(item.risk) for item in method_scores]
        parts = " + ".join(f"{v:.2f}" for v in f1s)
        return f"risk = 1 − mean(max F1) = 1 − ({parts}) / {len(f1s)} = {risk:.3f}"
    if method_key == "factscore":
        return f"risk = 1 − supported_ratio = 1 − {float(raw['supported_ratio']):.2f} = {risk:.3f}"
    if method_key == "p_true":
        return f"risk = 1 − P(True) = 1 − {float(method_scores[0].score):.2f} = {risk:.3f}"
    if method_key == "cove_detector":
        n = int(raw["num_questions"])
        conflicts = round(float(raw["conflict_rate"]) * n)
        return f"risk = 冲突问题数/问题数 = {conflicts}/{n} = {risk:.3f}"
    return ""


def _compare_card(scenario_id: str, metrics: list[dict[str, str]], risk: float, active: bool) -> dict[str, Any]:
    data = _SCENARIOS[scenario_id]
    tone = data["tone"] if risk >= RISK_THRESHOLD else ("good" if data["tone"] == "good" else "warn")
    if data["tone"] == "good" and risk < RISK_THRESHOLD:
        tone = "good"
        verdict = "当前信号较稳定"
    elif data["tone"] == "bad" and risk >= RISK_THRESHOLD:
        tone = "bad"
        verdict = "可能存在幻觉"
    else:
        tone = "warn"
        verdict = "信号不明确"
    return {
        "scenario": scenario_id,
        "title": data["prompt"],
        "text": data["response"],
        "reference": "；".join(data["reference_answer"]),
        "prompt": data["prompt"],
        "risk": round(risk, 4),
        "confidence": round(1.0 - risk, 4),
        "verdict": verdict,
        "tone": tone,
        "metrics": metrics,
        "active": active,
    }


def _assemble(method_key: str, scenario_id: str, parts: dict[str, tuple], extra_warnings: list[str], metadata: dict[str, Any]) -> dict[str, Any]:
    active_parts = parts[scenario_id]
    method_scores, viz, metrics = active_parts
    risk, score = _headline(method_scores)
    card = _compare_card(scenario_id, metrics, risk, active=True)
    card["formula"] = _formula(method_key, method_scores, risk)
    cards = [card]
    badges = _sample_badges(scenario_id, method_key, method_scores)
    if badges is not None:
        viz = [badges, *viz]
    verdict_note = (
        f"判定阈值:risk ≥ {RISK_THRESHOLD} 视为可能存在幻觉;当前场景 risk = {risk:.2f}。"
    )
    return _jsonable({
        "method": method_key,
        "scenario": scenario_id,
        "prompt": _SCENARIOS[scenario_id]["prompt"],
        "response": _SCENARIOS[scenario_id]["response"],
        "risk": round(risk, 4),
        "confidence": round(score, 4),
        "compare": {"cards": cards, "threshold_note": verdict_note},
        "token_marks": _token_marks(scenario_id, method_key in {"mtp", "lntp"}) if method_key in {"mtp", "lntp"} else None,
        "method_scores": [_result_to_dict(item) for item in method_scores],
        "viz": viz,
        "warnings": extra_warnings + [NEUTRAL_NOTE],
        "metadata": metadata,
    })


def _pair(builder) -> dict[str, tuple]:
    return {sid: builder(sid) for sid in _SCENARIOS}


_BUILDERS = {
    "mtp": lambda: _pair(lambda sid: _build_token_probability(sid, "mtp")),
    "lntp": lambda: _pair(lambda sid: _build_token_probability(sid, "lntp")),
    "token_entropy": lambda: _pair(_build_token_entropy),
    "eigenscore": lambda: _pair(lambda sid: _build_spectrum(sid, "eigenscore")),
    "effective_rank": lambda: _pair(lambda sid: _build_spectrum(sid, "effective_rank")),
    "exact_match_consistency": lambda: _pair(_build_exact_match),
    "response_cosine_consistency": lambda: _pair(_build_cosine),
    "response_semantic_entropy": lambda: _pair(_build_semantic_entropy),
    "selfcheck_ngram": lambda: _pair(_build_selfcheck_ngram),
    "selfcheck_nli": lambda: _pair(_build_selfcheck_nli),
    "selfcheck_bertscore": lambda: _pair(_build_selfcheck_bertscore),
    "factscore": lambda: _pair(_build_factscore),
    "p_true": lambda: _pair(_build_p_true),
    "cove_detector": lambda: _pair(_build_cove_detector),
}


def run_detection_method(method_key: str, scenario_id: str) -> dict[str, Any]:
    builder = _BUILDERS.get(method_key)
    if builder is None:
        raise ValueError(f"检测方法暂无预设场景: {method_key}")
    parts = builder()
    if scenario_id not in parts:
        raise ValueError(f"方法 {method_key} 不支持场景: {scenario_id}")
    metadata = {"source": "preset", "scorer": method_key}
    return _assemble(method_key, scenario_id, parts, [], metadata)
