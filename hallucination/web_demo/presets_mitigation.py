"""Preset scenario builders for the tutorial's mitigation methods.

DoLa/CAD replay the toolkit's *real* decoding math (``dola.jsd``,
``select_premature_layer``, ``apc_mask``, ``contrast_logits``,
``cad.combine_context_logits``, ``distribution_kl``, ``sampling.softmax``) on
canned small-vocabulary logits. CoVe runs the toolkit's real four-stage chain
through a scripted backend. Nothing calls a model or the network.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from hallucination.mitigation.backends.callable import CallableBackend
from hallucination.mitigation.core.contracts import GenerationRequest
from hallucination.mitigation.core.sampling import softmax
from hallucination.mitigation.methods.cad import combine_context_logits, distribution_kl
from hallucination.mitigation.methods.cove import CoVeMethod
from hallucination.mitigation.methods.dola import (
    apc_mask,
    contrast_logits,
    select_premature_layer,
)

STEERING_WARNING = "缓解不是事实正确性的保证;基线与干预结果的对照评测是必须的。"

_MARK_LEGEND = [
    {"cls": "risk", "label": "幻觉词元"},
    {"cls": "good", "label": "正确词元"},
]


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


def _metric(label: str, value: str) -> dict[str, str]:
    return {"label": label, "value": value}


def _marks(chunks: list[tuple[str, str, str]]) -> dict[str, Any]:
    """chunks: (text, cls, tip) triples; cls is "good" (correct) or "risk"."""

    return {
        "tokens": [{"text": text, "cls": cls, "tip": tip} for text, cls, tip in chunks],
        "legend": _MARK_LEGEND,
    }


def _payload(
    method_key: str,
    scenario_id: str,
    prompt: str,
    baseline_text: str,
    mitigated_text: str,
    baseline_note: str,
    mitigated_note: str,
    viz: list[dict[str, Any]],
    token_marks: dict[str, Any] | None,
    warnings: list[str],
    context: str | None = None,
) -> dict[str, Any]:
    return _jsonable({
        "method": method_key,
        "scenario": scenario_id,
        "prompt": prompt,
        "context": context,
        "compare": {
            "cards": [
                {
                    "title": "原输出 · 基线解码",
                    "text": baseline_text,
                    "tone": "bad",
                    "verdict": baseline_note,
                    "metrics": [],
                    "active": False,
                },
                {
                    "title": "缓解后输出",
                    "text": mitigated_text,
                    "tone": "good",
                    "verdict": mitigated_note,
                    "metrics": [],
                    "active": True,
                },
            ],
            "threshold_note": STEERING_WARNING,
        },
        "token_marks": token_marks,
        "method_scores": [],
        "viz": viz,
        "warnings": warnings + [STEERING_WARNING],
        "metadata": {"source": "preset", "mechanism": method_key},
    })


# --------------------------------------------------------------------------
# DoLa: real layer-contrast math on a canned 6-token vocabulary.
# --------------------------------------------------------------------------

_DOLA_VOCAB = ["磷虾", "海藻", "小鱼", "为食", "的", "。"]


def _dola_step_logits() -> dict[str, Any]:
    mature = np.array([0.9, 1.0, 0.2, -1.0, -0.5, -2.0])
    candidates = {
        10: np.array([0.2, 1.4, 0.3, -0.8, -0.4, -1.5]),
        14: np.array([0.6, 1.2, 0.25, -0.9, -0.45, -1.7]),
        18: np.array([0.85, 1.05, 0.22, -0.95, -0.48, -1.9]),
    }
    return {"mature": mature, "candidates": candidates}


def _build_dola(scenario_id: str) -> dict[str, Any]:
    data = _dola_step_logits()
    mature, candidates = data["mature"], data["candidates"]
    selected_layer, jsd_scores = select_premature_layer(mature, candidates)
    premature = candidates[selected_layer]
    masked = apc_mask(mature, relative_top=0.1)
    contrast = contrast_logits(mature, premature, relative_top=0.1, post_softmax=True)
    probs = softmax(contrast)
    chosen = int(np.argmax(probs))
    baseline_token = int(np.argmax(mature))

    jsd_items = [
        {"label": f"早层 L{layer}", "value": score, "highlight": layer == selected_layer}
        for layer, score in jsd_scores.items()
    ]
    rows = []
    for index, token in enumerate(_DOLA_VOCAB):
        if masked[index] == -np.inf:
            status = "被 APC 掩码剪除"
        elif index == chosen:
            status = "对比分布选中"
        elif index == baseline_token:
            status = "基线 argmax(被对比抑制)"
        else:
            status = ""
        rows.append([
            token,
            f"{float(mature[index]):.2f}",
            f"{float(premature[index]):.2f}",
            f"{float(probs[index]) * 100:.1f}%",
            status,
        ])

    viz = [
        {
            "type": "layer-curve",
            "title": "第 1 步 · 各候选早层与成熟层的 JSD",
            "items": jsd_items,
            "format": "value2",
            "note": "JSD 最大的早层(高亮)被选中做层间对比;它代表\"最不一样的语言先验\"。",
        },
        {
            "type": "logit-table",
            "title": f"第 1 步 · APC 掩码与对比分布(选中早层 L{selected_layer})",
            "columns": ["词元", "成熟层 logit", f"早层 L{selected_layer} logit", "对比后概率", "状态"],
            "rows": rows,
            "note": (
                f"基线 argmax 选中“{_DOLA_VOCAB[baseline_token]}”;DoLa 用成熟层减去早层,"
                f"抑制两边都偏爱的表层先验,转而选中“{_DOLA_VOCAB[chosen]}”。"
            ),
        },
        {
            "type": "steps-table",
            "title": "逐步解码轨迹",
            "columns": ["步", "基线选中", "DoLa 选中", "早层 JSD"],
            "rows": [
                ["1", _DOLA_VOCAB[baseline_token], _DOLA_VOCAB[chosen], f"{jsd_scores[selected_layer]:.3f}"],
                ["2", "为食", "为食", "0.012"],
                ["3", "。", "。", "0.003"],
            ],
            "note": "关键分歧发生在第 1 步:后续两步两个分布已一致。",
        },
    ]
    token_marks = {
        "baseline": _marks([
            ("蓝鲸", "good", ""),
            ("主要", "good", ""),
            ("以", "good", ""),
            ("海藻", "risk", "幻觉词元:与事实不符"),
            ("为食", "good", ""),
            ("。", "good", ""),
        ]),
        "mitigated": _marks([
            ("蓝鲸", "good", ""),
            ("主要", "good", ""),
            ("以", "good", ""),
            (_DOLA_VOCAB[chosen], "good", "层间对比选中"),
            ("为食", "good", ""),
            ("。", "good", ""),
        ]),
    }
    return _payload(
        "dola",
        scenario_id,
        "请介绍蓝鲸的食性。",
        "蓝鲸主要以海藻为食。",
        "蓝鲸主要以磷虾为食。",
        "基线 argmax 选中表层先验“海藻”，事实漂移",
        f"层间对比选中“磷虾”（对比后概率 {float(probs[chosen]) * 100:.1f}%）",
        viz,
        token_marks,
        ["当前展示 DoLa 的真实数学计算过程(jsd / select_premature_layer / apc_mask / contrast_logits)。"],
    )


# --------------------------------------------------------------------------
# CAD: real context-contrast math on a canned vocabulary.
# --------------------------------------------------------------------------

_CAD_VOCAB = ["3", "2", "次", "冠军", "。"]


_CAD_WHALE_VOCAB = ["磷虾", "海藻", "小鱼", "为食", "。"]


def _build_cad(scenario_id: str) -> dict[str, Any]:
    if scenario_id == "conflict":
        context = "阿根廷在 1978、1986 和 2022 年赢得了世界杯。"
        prompt = "已知：阿根廷在 1978、1986 和 2022 年赢得了世界杯。问：阿根廷赢过几次世界杯？"
        ctx = np.array([1.4, 1.0, -1.5, -2.0, -2.5])
        noctx = np.array([-1.6, 1.5, -0.8, -2.2, -2.6])
        alpha = 1.0
        combined = combine_context_logits(ctx, noctx, alpha=alpha)
        probs = softmax(combined)
        chosen = int(np.argmax(probs))
        baseline_token = int(np.argmax(ctx))
        kl = distribution_kl(softmax(ctx), softmax(noctx))
        rows = [
            [
                _CAD_VOCAB[index],
                f"{float(softmax(ctx)[index]) * 100:.1f}%",
                f"{float(softmax(noctx)[index]) * 100:.1f}%",
                f"{float(probs[index]) * 100:.1f}%",
                "对比分布选中" if index == chosen else ("基线 argmax" if index == baseline_token else ""),
            ]
            for index in range(len(_CAD_VOCAB))
        ]
        viz = [
            {
                "type": "logit-table",
                "title": f"关键步 · 上下文/无上下文/对比后分布（α = {alpha}）",
                "columns": ["词元", "带上下文 p", "无上下文 p", "(1+α)·ctx − α·noctx", "状态"],
                "rows": rows,
                "note": "上下文把“3”推高但没能压过参数化先验“2”;CAD 放大上下文视角后“3”胜出。",
            },
            {
                "type": "steps-table",
                "title": "逐步 KL(上下文 ‖ 无上下文)",
                "columns": ["步", "选中词元", "KL (nats)"],
                "rows": [
                    ["1", "阿根廷", "0.21"],
                    ["2", "赢得", "0.08"],
                    ["3", _CAD_VOCAB[chosen], f"{kl:.3f}"],
                    ["4", "次", "0.02"],
                ],
                "note": "KL 大的步 = 两个视角分歧最大的步,CAD 在这些步上影响最强。",
            },
        ]
        token_marks = {
            "baseline": _marks([
                ("阿根廷", "good", ""),
                ("赢得了", "good", ""),
                ("2", "risk", "幻觉词元:与给定上下文矛盾"),
                ("次", "good", ""),
                ("世界杯", "good", ""),
                ("。", "good", ""),
            ]),
            "mitigated": _marks([
                ("阿根廷", "good", ""),
                ("赢得了", "good", ""),
                (_CAD_VOCAB[chosen], "good", "上下文对比选中"),
                ("次", "good", ""),
                ("世界杯", "good", ""),
                ("。", "good", ""),
            ]),
        }
        return _payload(
            "cad",
            scenario_id,
            prompt,
            "阿根廷赢得了2次世界杯。",
            "阿根廷赢得了3次世界杯。",
            "基线被参数化知识带偏,无视给定上下文",
            f"对比解码选中“{_CAD_VOCAB[chosen]}”(对比后概率 {float(probs[chosen]) * 100:.1f}%),忠实于上下文",
            viz,
            token_marks,
            ["当前展示 CAD 的真实合并数学(combine_context_logits / distribution_kl)。"],
            context=context,
        )

    context = "蓝鲸以磷虾等小型甲壳类动物为主食。"
    prompt = "蓝鲸主要吃什么？"
    ctx = np.array([2.2, -1.0, -0.5, -2.0, -2.8])
    noctx = np.array([1.9, -1.4, -0.2, -2.4, -2.9])
    combined = combine_context_logits(ctx, noctx, alpha=1.0)
    probs = softmax(combined)
    kl = distribution_kl(softmax(ctx), softmax(noctx))
    viz = [
        {
            "type": "logit-table",
            "title": "关键步 · 两个视角几乎一致",
            "columns": ["词元", "带上下文 p", "无上下文 p", "对比后 p"],
            "rows": [
                [
                    _CAD_VOCAB[index],
                    f"{float(softmax(ctx)[index]) * 100:.1f}%",
                    f"{float(softmax(noctx)[index]) * 100:.1f}%",
                    f"{float(probs[index]) * 100:.1f}%",
                ]
                for index in range(len(_CAD_VOCAB))
            ],
            "note": f"KL = {kl:.3f} nats:上下文与内部知识一致时,CAD 几乎不改变输出(无副作用)。",
        },
    ]
    token_marks = {
        "baseline": _marks([
            ("蓝鲸", "good", ""),
            ("主要以", "good", ""),
            ("磷虾", "good", ""),
            ("为食", "good", ""),
            ("。", "good", ""),
        ]),
        "mitigated": _marks([
            ("蓝鲸", "good", ""),
            ("主要以", "good", ""),
            ("磷虾", "good", "两视角一致"),
            ("为食", "good", ""),
            ("。", "good", ""),
        ]),
    }
    return _payload(
        "cad",
        scenario_id,
        prompt,
        "蓝鲸主要以磷虾为食。",
        "蓝鲸主要以磷虾为食。",
        "内部知识与上下文一致",
        "输出不变——CAD 在无冲突时不应引入漂移",
        viz,
        token_marks,
        ["当前展示 CAD 的真实合并数学(combine_context_logits / distribution_kl)。"],
        context=context,
    )


# --------------------------------------------------------------------------
# CAA: replay of the fit math (activation mean-difference + normalization).
# --------------------------------------------------------------------------


def _build_caa(scenario_id: str) -> dict[str, Any]:
    rng = np.random.default_rng(11)
    dim = 8
    prompt = "蓝鲸是怎样的动物？"
    pairs = [
        {
            "prompt": prompt,
            "positive": "蓝鲸是体型巨大的海洋哺乳动物。",
            "negative": "蓝鲸是一种小型鱼类。",
        },
        {
            "prompt": "蓝鲸以什么为食？",
            "positive": "蓝鲸以磷虾等大型甲壳类为食。",
            "negative": "蓝鲸以浮游藻类为食。",
        },
    ]
    positive_mean = rng.normal(size=dim) + np.array([1.5, 0.5, 0.0, 0.0, 0.8, 0.0, 0.3, 0.0])
    negative_mean = rng.normal(size=dim)
    steering = positive_mean - negative_mean
    steering = steering / np.linalg.norm(steering)
    vector_norm = float(np.linalg.norm(positive_mean - negative_mean))

    multiplier_outputs = [
        {"multiplier": 0.5, "text": "蓝鲸是海洋中的一种大型动物。", "note": "转向不足,仍是模糊描述"},
        {"multiplier": 1.0, "text": "蓝鲸是体型巨大的海洋哺乳动物，以磷虾为食。", "note": "方向正确且忠实"},
        {"multiplier": 2.0, "text": "蓝鲸是极其巨大的哺乳动物，是所有海洋生物中最大的。", "note": "过冲,开始夸大"},
    ]
    viz = [
        {
            "type": "pair-view",
            "title": "拟合:正/负回答对的激活差均值",
            "pairs": pairs,
            "stats": [
                _metric("转向层", "L12"),
                _metric("‖v‖(未归一化)", f"{vector_norm:.3f}"),
                _metric("池化", "last token"),
            ],
            "note": "v = normalize(mean(正例激活) − mean(负例激活));推理时把 v·multiplier 加到该层残差流。",
        },
        {
            "type": "logit-table",
            "title": "不同倍率的生成结果",
            "columns": ["multiplier", "生成文本", "评价"],
            "rows": [
                [str(item["multiplier"]), item["text"], item["note"]]
                for item in multiplier_outputs
            ],
            "note": "倍率太小没有效果,太大则过冲——激活转向需要调参。",
        },
    ]
    steered = next(item for item in multiplier_outputs if item["multiplier"] == 1.0)
    token_marks = {
        "baseline": _marks([
            ("蓝鲸", "good", ""),
            ("是一种", "good", ""),
            ("小型的", "risk", "幻觉词元:与事实不符"),
            ("海洋鱼类", "risk", "幻觉词元:与事实不符"),
            ("。", "good", ""),
        ]),
        "mitigated": _marks([
            ("蓝鲸", "good", ""),
            ("是", "good", ""),
            ("体型巨大的", "good", "转向后内容"),
            ("海洋哺乳动物", "good", "转向后内容"),
            ("，以", "good", ""),
            ("磷虾", "good", "转向后内容"),
            ("为食", "good", ""),
            ("。", "good", ""),
        ]),
    }
    return _payload(
        "caa",
        scenario_id,
        prompt,
        "蓝鲸是一种小型的海洋鱼类。",
        steered["text"],
        "未转向的基线携带错误特征",
        f"multiplier=1.0 的转向向量把生成推向“巨大/哺乳动物”(v 已归一化,未归一化 ‖v‖ = {vector_norm:.2f})",
        viz,
        token_marks,
        ["当前展示 CAA 的拟合数学(激活差均值 + 归一化)。"],
    )


# --------------------------------------------------------------------------
# ITI: canned layer×head probe grid + real bias-injection arithmetic.
# --------------------------------------------------------------------------


def _build_iti(scenario_id: str) -> dict[str, Any]:
    rng = np.random.default_rng(3)
    layers, heads = 4, 6
    base = rng.uniform(0.48, 0.62, size=(layers, heads))
    selected = [(2, 1, 0.89, 1.8), (3, 0, 0.86, 1.5), (1, 2, 0.84, 1.3), (3, 3, 0.81, 1.2)]
    for layer, head, accuracy, _scale in selected:
        base[layer][head] = accuracy
    values = [[round(float(value), 3) for value in row] for row in base]
    alpha = 15.0
    prompt = "蓝鲸生活在哪里？"
    viz = [
        {
            "type": "head-grid",
            "title": "拟合:层 × 头 探针留出准确率",
            "layers": layers,
            "heads": heads,
            "values": values,
            "selected": [[layer, head] for layer, head, _acc, _scale in selected],
            "format": "percent",
            "note": "每个 (层, 头) 拟合一个\"真实/虚假\"逻辑回归探针;准确率显著高于 0.5 的头更值得干预。",
        },
        {
            "type": "logit-table",
            "title": f"选中 top-{len(selected)} 头与偏置注入(bias = direction × α × scale, α = {alpha})",
            "columns": ["层", "头", "留出准确率", "scale", "α × scale"],
            "rows": [
                [f"L{layer}", f"H{head}", f"{accuracy:.2f}", f"{scale:.2f}", f"{alpha * scale:.1f}"]
                for layer, head, accuracy, scale in selected
            ],
            "note": "推理时只对选中头的 o_proj 输入沿探针方向加 α·scale 的偏置。",
        },
    ]
    token_marks = {
        "baseline": _marks([
            ("蓝鲸", "good", ""),
            ("生活在", "good", ""),
            ("淡水湖中", "risk", "幻觉词元:与事实不符"),
            ("，是一种", "good", ""),
            ("淡水鱼类", "risk", "幻觉词元:与事实不符"),
            ("。", "good", ""),
        ]),
        "mitigated": _marks([
            ("蓝鲸", "good", ""),
            ("生活在", "good", ""),
            ("海洋中", "good", "探针偏置改变"),
            ("，是", "good", ""),
            ("海洋哺乳动物", "good", "探针偏置改变"),
            ("。", "good", ""),
        ]),
    }
    return _payload(
        "iti",
        scenario_id,
        prompt,
        "蓝鲸生活在淡水湖中，是一种淡水鱼类。",
        "蓝鲸生活在海洋中，是海洋哺乳动物。",
        "基线继承了对栖息地的错误联想",
        f"{len(selected)} 个高准确率头的偏置把生成拉向真实子空间",
        viz,
        token_marks,
        ["偏置注入公式与真实实现一致(direction × α × scale)。"],
    )


# --------------------------------------------------------------------------
# CoVe: the toolkit's real four-stage chain on a scripted backend.
# --------------------------------------------------------------------------


def _build_cove(scenario_id: str) -> dict[str, Any]:
    backend = CallableBackend([
        "蓝鲸主要以海藻为食，每天能吃掉大量的海藻。",
        "1. 蓝鲸的主要食物是什么？\n2. 蓝鲸的食量有多大？",
        "蓝鲸主要以磷虾等小型甲壳类动物为食。",
        "蓝鲸每天可以吃掉数吨磷虾。",
        "蓝鲸主要以磷虾等小型甲壳类动物为食，每天能吃掉数吨磷虾。",
    ])
    request = GenerationRequest(prompt="请介绍蓝鲸的食性。", max_new_tokens=64)
    report = CoVeMethod().run(backend, request, None)
    payload = report.to_dict()
    trace = payload["trace"]
    # Latency seconds vary run to run; drop them so the payload stays deterministic.
    for stage in trace.get("stage_costs", []):
        stage.pop("seconds", None)
    payload["cost"].pop("latency_seconds", None)
    for stage in payload["cost"].get("stage_costs", []):
        stage.pop("seconds", None)

    viz = [
        {
            "type": "qa-chain",
            "title": "四阶段验证链(真实 CoVe 代码运行)",
            "stages": [
                {"stage": "baseline", "title": "① 基线草稿", "content": trace["baseline"]},
                {"stage": "plan", "title": "② 规划验证问题", "content": trace["plan_raw"]},
                {
                    "stage": "verify",
                    "title": "③ 逐题独立作答",
                    "questions": trace["answers"],
                },
                {"stage": "revise", "title": "④ 依据问答修订", "content": payload["response"]},
            ],
            "note": "计划解析、验证与修订逻辑全部来自 toolkit 真实实现。",
        },
    ]
    token_marks = {
        "baseline": _marks([
            ("蓝鲸", "good", ""),
            ("主要以", "good", ""),
            ("海藻", "risk", "幻觉词元:验证答案纠正"),
            ("为食", "good", ""),
            ("，每天能吃掉大量的", "good", ""),
            ("海藻", "risk", "幻觉词元:验证答案纠正"),
            ("。", "good", ""),
        ]),
        "mitigated": _marks([
            ("蓝鲸", "good", ""),
            ("主要以", "good", ""),
            ("磷虾", "good", "验证答案纠正"),
            ("等小型甲壳类动物为食", "good", ""),
            ("，每天能吃掉数吨", "good", ""),
            ("磷虾", "good", "验证补充的事实"),
            ("。", "good", ""),
        ]),
    }
    result = _payload(
        "cove",
        scenario_id,
        payload["prompt"],
        payload["baseline"] or "",
        payload["response"],
        "草稿中的“海藻”是事实性错误",
        f"修订后的回答与 {len(trace['answers'])} 条验证问答一致",
        viz,
        token_marks,
        list(payload["warnings"]) + [f"本次链共 {payload['cost'].get('calls', '?')} 次 LLM 调用。"],
    )
    result["metadata"]["trace"] = trace
    result["metadata"]["cost"] = payload["cost"]
    result["metadata"]["provenance"] = payload["provenance"]
    return result


BUILDERS = {
    "dola": _build_dola,
    "cad": _build_cad,
    "caa": _build_caa,
    "iti": _build_iti,
    "cove": _build_cove,
}


def run_mitigation_method(method_key: str, scenario_id: str) -> dict[str, Any]:
    builder = BUILDERS.get(method_key)
    if builder is None:
        raise ValueError(f"缓解方法暂无预设场景: {method_key}")
    catalog = METHOD_SCENARIOS[method_key]
    if scenario_id not in catalog:
        raise ValueError(f"方法 {method_key} 不支持场景: {scenario_id}")
    return builder(scenario_id)


METHOD_SCENARIOS = {
    "dola": {"hallucinated"},
    "cad": {"conflict"},
    "caa": {"hallucinated"},
    "iti": {"hallucinated"},
    "cove": {"hallucinated"},
}
