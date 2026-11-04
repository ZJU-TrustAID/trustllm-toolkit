"""Method registry for the tutorial mode.

The demo is prompt-driven, MarkSec-style: the user picks a preset prompt
(a *theme*), explicitly selects one method from the page's option groups,
and hits generate — the demo then shows the result for that theme. No
method is picked automatically; prompts a method cannot demonstrate simply
disable its option. Every entry documents the themes its builder in
``presets_detection`` / ``presets_mitigation`` implements; the hosted demo
never calls a model.
"""

from __future__ import annotations

from typing import Any

_DETECT_METHOD_KEYS = [
    "mtp",
    "lntp",
    "token_entropy",
    "eigenscore",
    "effective_rank",
    "exact_match_consistency",
    "response_cosine_consistency",
    "response_semantic_entropy",
    "selfcheck_ngram",
    "selfcheck_nli",
    "selfcheck_bertscore",
    "factscore",
    "p_true",
    "cove_detector",
]


def _pick(keys: list[str], source: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Select catalog entries by key, preserving ``source`` order."""

    return [method for method in source if method["key"] in keys]


def _method(
    key: str,
    name: str,
    tag: str,
    signal_type: str,
    level: str,
    summary: str,
    book_ref: str,
    needs: str,
    scenarios: list[str],
) -> dict[str, Any]:
    return {
        "key": key,
        "name": name,
        "tag": tag,
        "signal_type": signal_type,
        "level": level,
        "summary": summary,
        "book_ref": book_ref,
        "needs": needs,
        "scenarios": scenarios,
    }


def _detect(key: str, name: str, tag: str, signal: str, level: str, summary: str, book_ref: str, needs: str) -> dict[str, Any]:
    return _method(key, name, tag, signal, level, summary, book_ref, needs, ["hallucinated", "faithful"])


DETECTION_WHITEBOX: list[dict[str, Any]] = [
    _detect(
        "mtp", "最小词元概率 MTP", "词元概率", "generation_confidence", "response",
        "取回答中所有词元概率的最小值。低概率词元通常对应模型\"被迫选择\"的关键事实词,"
        "是事实性幻觉的强信号。",
        "书稿 §白盒幻觉检测·基于词元概率", "需要逐词元 log 概率(本地模型或 API logprobs)。",
    ),
    _detect(
        "lntp", "长度归一化词元概率 LNTP", "词元概率", "generation_confidence", "response",
        "对整条回答的词元概率取几何平均,消除序列长度影响,度量模型对完整回答的整体把握。",
        "书稿 §白盒幻觉检测·基于词元概率", "需要逐词元 log 概率。",
    ),
    _detect(
        "token_entropy", "Token 熵", "熵值", "generation_confidence", "token",
        "在每个解码步,对候选词元的 top-k 概率分布计算熵。熵越高说明模型越\"拿不准\","
        "该步更可能选出不可靠词元。",
        "书稿 §白盒幻觉检测·基于熵值检测", "需要每步 top-k 候选分布。",
    ),
    _detect(
        "eigenscore", "EigenScore 特征值谱", "表征分析", "generation_confidence", "response",
        "对同一问题多个采样的隐藏状态矩阵做谱分析。本实现沿用 INSIDE 的单调映射:"
        "谱幅度越大,风险越高(未校准)。",
        "书稿 §白盒幻觉检测·基于表征分析检测", "需要 ≥2 个采样的隐藏状态向量。",
    ),
    _detect(
        "effective_rank", "有效秩 Effective Rank", "表征分析", "generation_confidence", "response",
        "用奇异值谱的谱熵定义矩阵的有效秩。表征坍缩时有效秩显著下降,提示模型在\"盲目生成\"。",
        "书稿 §白盒幻觉检测·基于表征分析检测", "需要 ≥2 个采样的隐藏状态向量。",
    ),
]

_BB_CONSISTENCY: list[dict[str, Any]] = [
    _detect(
        "exact_match_consistency", "精确匹配一致性", "一致性", "self_consistency", "response",
        "对同一问题独立采样多个回答,统计与主回答完全一致(归一化后)的比例。"
        "回答间高度一致暗示模型\"心中有数\",但一致不等于事实正确。",
        "书稿 §黑盒幻觉检测·基于一致性检测", "需要主回答 + N 个独立采样。",
    ),
    _detect(
        "response_cosine_consistency", "余弦一致性", "一致性", "self_consistency", "response",
        "把主回答与各采样回答编码为向量,计算余弦相似度。对语义级一致更宽容,是精确匹配的软化版。",
        "书稿 §黑盒幻觉检测·基于一致性检测", "需要主回答 + N 个采样 + 嵌入模型(演示用离线哈希嵌入)。",
    ),
    _detect(
        "response_semantic_entropy", "语义熵 Semantic Entropy", "一致性", "self_consistency", "response",
        "用双向 NLI 判定回答两两之间的语义等价关系,将语义相同的回答聚为一簇,"
        "再对簇分布计算熵。熵越高,模型越\"不确定\"。",
        "书稿 §黑盒幻觉检测·基于一致性检测(三步:生成/聚类/计算语义熵)",
        "需要主回答 + N 个采样 + NLI 判定(演示用离线规则 NLI)。",
    ),
    _detect(
        "selfcheck_ngram", "SelfCheck N-gram", "一致性", "self_consistency", "sentence",
        "把采样回答当作\"语料\",统计主回答中每个句子的 n-gram 在其中的负对数概率。"
        "采样集里出现越少的词,支撑度越低。",
        "书稿 §黑盒幻觉检测·基于一致性检测", "需要主回答 + N 个采样;句子级打分。",
    ),
    _detect(
        "selfcheck_nli", "SelfCheck NLI", "一致性", "self_consistency", "sentence",
        "对主回答的每个句子,让 NLI 模型判定它与各采样回答的蕴含/矛盾/中性关系,"
        "用平均矛盾(非蕴含)率作为该句的不确定度。",
        "书稿 §黑盒幻觉检测·基于一致性检测", "需要主回答 + N 个采样 + NLI 模型。",
    ),
    _detect(
        "selfcheck_bertscore", "SelfCheck BERTScore", "一致性", "self_consistency", "sentence",
        "对主回答的每个句子,在采样回答中找语义最相近的句子,以语义相似度 F1 作为支持度。"
        "演示用字符级 F1 替身模拟 BERTScore 打分接口。",
        "书稿 §黑盒幻觉检测·基于一致性检测", "需要主回答 + N 个采样 + 相似度函数。",
    ),
]

_DEBB_EVIDENCE: list[dict[str, Any]] = [
    _detect(
        "factscore", "FActScore 证据核验", "证据验证", "evidence_support", "claim",
        "把回答拆成原子事实,逐条到检索证据中验证:支持/矛盾/未知。"
        "细粒度地指出\"哪一句不可信\"。",
        "书稿 §黑盒幻觉检测·基于外部知识库验证", "需要原子事实 + 检索证据 + NLI 验证器。",
    ),
]

_DEBB_SELF_EVAL: list[dict[str, Any]] = [
    _detect(
        "p_true", "P(True) 自评估", "自评估", "self_reported_confidence", "response",
        "先把回答放进选择题提示词,再读取模型对 \" A) True\" 的词元概率。"
        "直接问模型\"你说的对吗\",用概率而非口头回答度量自信。",
        "书稿 §黑盒幻觉检测·基于模型自评估检测", "需要提示词构造 + logprob 读取能力(离线 P(True) 后端)。",
    ),
    _detect(
        "cove_detector", "CoVe 检测器", "自评估", "self_reported_confidence", "response",
        "让模型针对自己的回答生成验证问题并逐条作答,再用 NLI 比对验证答案与原回答是否冲突,"
        "冲突率即风险信号。",
        "书稿 §黑盒幻觉检测·基于模型自评估检测", "需要 LLM 生成问题/答案(脚本化后端)+ 可选 NLI。",
    ),
]

DETECTION_BLACKBOX: list[dict[str, Any]] = _BB_CONSISTENCY + _DEBB_EVIDENCE + _DEBB_SELF_EVAL

# 三个页签均按教材的小节结构分组展示方法。
DETECTION_WHITEBOX_GROUPS: list[dict[str, Any]] = [
    {"title": "基于词元概率检测", "methods": _pick(["mtp", "lntp"], DETECTION_WHITEBOX)},
    {"title": "基于熵值检测", "methods": _pick(["token_entropy"], DETECTION_WHITEBOX)},
    {"title": "基于表征分析检测", "methods": _pick(["eigenscore", "effective_rank"], DETECTION_WHITEBOX)},
]

DETECTION_BLACKBOX_GROUPS: list[dict[str, Any]] = [
    {"title": "基于外部知识库验证", "methods": _DEBB_EVIDENCE},
    {"title": "基于模型自评估检测", "methods": _DEBB_SELF_EVAL},
    {"title": "基于一致性检测", "methods": _BB_CONSISTENCY},
]

MITIGATION: list[dict[str, Any]] = [
    _method(
        "dola", "DoLa 层间对比解码", "层间对比", "generation_confidence", "response",
        "每一解码步,在早层与成熟层 logits 之间选 Jensen-Shannon 散度最大的一对做对比,"
        "抑制\"语言流畅但事实漂移\"的表层先验,保留事实知识。",
        "书稿 §事实性增强解码·层间对比解码",
        "白盒方法:重放真实 DoLa 数学(JSD 选层 + APC 掩码 + 对比 logits)。",
        ["hallucinated"],
    ),
    _method(
        "iti", "ITI 逐头探针转向", "逐头转向", "generation_confidence", "response",
        "在每个注意力头的输出投影上拟合\"真实/虚假\"探针,推理时给高准确率头的激活"
        "加上方向偏置,把生成推向更真实的子空间。",
        "书稿 §事实性增强解码·推理激活干预",
        "白盒方法:展示罐头层×头探针精度网格与偏置注入计算。",
        ["hallucinated"],
    ),
    _method(
        "cad", "CAD 上下文感知解码", "上下文对比", "generation_confidence", "response",
        "同时维护\"带上下文\"与\"不带上下文\"两条解码流,按 (1+α)·p(上下文) − α·p(无上下文) "
        "合并分布采样,使输出倾向给定上下文。",
        "书稿 §忠实性增强解码(CAD 与 worked example)",
        "白盒方法:重放真实 CAD 合并数学(对比 logits + KL)。",
        ["conflict"],
    ),
    _method(
        "caa", "CAA 激活转向", "激活转向", "generation_confidence", "response",
        "用\"正面/负面\"回答对的隐藏层激活差均值构造转向向量,推理时按倍率加到对应层"
        "的残差流上,整体推动生成风格与内容取向。",
        "toolkit 扩展(书稿之外的推理时缓解实践)",
        "白盒方法:重放拟合数学(激活差均值 + 归一化)与倍率效果。",
        ["hallucinated"],
    ),
    _method(
        "cove", "CoVe 验证修正", "验证修正", "generation_confidence", "response",
        "生成后处理四阶段链:基线回答 → 规划验证问题 → 逐问自答 → 依据问答修订草稿。"
        "演示完整运行真实 CoVe 代码,仅 LLM 换成脚本化后端。",
        "toolkit 扩展(书稿 §黑盒检测·CoVe 的缓解侧应用)",
        "黑盒方法:LLM 后端可任意(演示用脚本化后端)。",
        ["hallucinated"],
    ),
]

METHODS: dict[str, dict[str, Any]] = {
    method["key"]: method
    for method in DETECTION_WHITEBOX + DETECTION_BLACKBOX + MITIGATION
}

MITIGATION_GROUPS: list[dict[str, Any]] = [
    {"title": "事实性增强解码", "methods": _pick(["dola", "iti"], MITIGATION)},
    {"title": "忠实性增强解码", "methods": _pick(["cad"], MITIGATION)},
    {"title": "事后修正", "methods": _pick(["cove"], MITIGATION)},
    {"title": "激活转向 · toolkit 扩展", "methods": _pick(["caa"], MITIGATION)},
]


THEMES: list[dict[str, Any]] = [
    {
        "id": "hallucinated",
        "label": "蓝鲸的食性",
        "prompt": "请介绍蓝鲸的食性。",
        "response": "蓝鲸主要以海藻为食。",
        "note": "主回答含事实错误(海藻),采样中出现\"淡水湖\"等漂移;适合观察检测信号与缓解纠正。",
        "available": {"detection": list(_DETECT_METHOD_KEYS), "mitigation": ["dola", "iti", "caa", "cove"]},
    },
    {
        "id": "faithful",
        "label": "海葵是不是植物",
        "prompt": "海葵是植物吗？",
        "response": "海葵不是植物，而是动物。",
        "note": "回答与事实一致;缓解方法无从\"纠正\",仅演示检测信号应当平稳。",
        "available": {"detection": list(_DETECT_METHOD_KEYS), "mitigation": []},
    },
    {
        "id": "conflict",
        "label": "阿根廷的世界杯",
        "prompt": "已知：阿根廷在 1978、1986 和 2022 年赢得了世界杯。问：阿根廷赢过几次世界杯？",
        "response": "阿根廷赢得了2次世界杯。",
        "context": "阿根廷在 1978、1986 和 2022 年赢得了世界杯。",
        "note": "提示词内含与参数化知识冲突的上下文;演示 CAD 如何让输出忠于上下文。",
        "available": {"detection": [], "mitigation": ["cad"]},
    },
]

def catalog_payload() -> dict[str, Any]:
    """Serializable catalog for ``GET /api/catalog``."""

    return {
        "themes": THEMES,
        "detection_whitebox": DETECTION_WHITEBOX,
        "detection_blackbox": DETECTION_BLACKBOX,
        "detection_whitebox_groups": DETECTION_WHITEBOX_GROUPS,
        "detection_blackbox_groups": DETECTION_BLACKBOX_GROUPS,
        "mitigation": MITIGATION,
        "mitigation_groups": MITIGATION_GROUPS,
        "note": "页面运行 toolkit 的真实 scorer 与解码数学。",
    }


def resolve(method_key: str, scenario_id: str) -> dict[str, Any]:
    if method_key not in METHODS:
        raise ValueError(f"未知方法: {method_key}")
    method = METHODS[method_key]
    if scenario_id not in method["scenarios"]:
        raise ValueError(f"方法 {method_key} 不支持场景: {scenario_id}")
    return method
