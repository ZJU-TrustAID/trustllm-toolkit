# -*- coding: utf-8 -*-
"""审计模块：统计检验、阈值校准、报告与清洗攻击。

统一在这里把检测器返回的**原始统计量**补全为 p 值（书稿 2.5.4 的
「score / n / p_value / decision」四件套），不同方法对应的检验不同：

======================  ==============================================
方法                    零假设检验
======================  ==============================================
canary_memory           配对 t 检验（df = n-1，右尾）
version_pair            配对 t 检验（df = n-1，右尾）
token_statistics        比例 z 检验（p0 = 0.5，右尾）
fictitious_knowledge    精确二项检验（p0 = 1/4，右尾）
trigger_behavior        精确二项检验（p0 = 1/4，右尾）
in_context              精确二项检验（p0 = 0.05，右尾）
rag_canary              精确二项检验（p0 = 1/4，右尾）
======================  ==============================================
"""

from __future__ import annotations

import math

from . import stats
from .report import AuditReport, SIMULATION_NOTE, deterministic_timestamp

#: 命中率型检测器（score 为 0–1 的比例，n 为问题数）对应的零假设比例。
_HIT_RATE_P0 = {
    "fictitious_knowledge": 0.25,
    "trigger_behavior": 0.25,
    "in_context": 0.05,
    "rag_canary": 0.25,
}

#: 配对 t 检验类检测器。
_T_TEST_METHODS = ("canary_memory", "version_pair")


def compute_p_value(method: str, result: dict) -> float:
    """把检测器的原始统计量换算为右尾 p 值。

    :param method: 方法名。
    :param result: 检测器返回值（至少含 ``score`` / ``n``）。
    """
    score = float(result.get("score", 0.0))
    n = int(result.get("n", 0) or 0)
    if n <= 0 or not math.isfinite(score):
        if score == float("inf"):
            return 0.0
        return 1.0
    if method in _T_TEST_METHODS:
        if n <= 1:
            return 0.5
        p = stats.t_sf(score, n - 1)
    elif method == "token_statistics":
        p = stats.normal_sf(score)
    else:
        p0 = _HIT_RATE_P0.get(method)
        if p0 is None:  # 未知插件：退化为正态近似（score 视作 z 值）
            p = stats.normal_sf(score)
        else:
            hits = int(round(score * n))
            p = stats.binomial_test(hits, n, p0, alternative="greater")["p"]
    # 教学报告里不显示「0.0」而显示 1e-12：双重下溢时仍保留「证据极强」的含义
    return max(1e-12, min(1.0, float(p)))


__all__ = [
    "stats",
    "AuditReport",
    "SIMULATION_NOTE",
    "deterministic_timestamp",
    "compute_p_value",
]
