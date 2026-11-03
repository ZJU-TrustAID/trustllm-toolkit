# -*- coding: utf-8 -*-
"""审计统计工具：全部用标准库 ``math`` 实现，不依赖 scipy。

教材 2.5.4 的判定流程是「先校准、再检验」：

1. 用从未见过带水印数据的**负样本模型**得到零假设（H0）下的统计量分布；
2. 取经验分位数作为阈值 ``t*``，使误报率不超过目标 α（:func:`calibrate_threshold`）；
3. 对可疑模型的统计量做 z 检验 / 精确二项检验 / 配对 t 检验，得到 p 值；
4. 报告 TPR@FPR：同一阈值下正样本模型被检出的比例（:func:`tpr_at_fpr`）。

本章用到的几个公式（与正文一致）：

* 比例 z 检验：``z = (k/n - p0) / sqrt(p0(1-p0)/n)``，右尾 p 值用正态近似；
* 配对 t 检验：``t = mean(d) / (sd(d)/sqrt(n))``，p 值用 t 分布（不完全 Beta 函数）；
* 精确二项检验：``P = sum_{i=k}^{n} C(n,i) p0^i (1-p0)^{n-i}``（n ≤ 1000 时精确求和）；
* 符号检验：配对差中位数为 0 的二项检验（p0 = 0.5）。
"""

from __future__ import annotations

import math
from typing import Iterable, Sequence

# ---------------------------------------------------------------------------
# 正态分布
# ---------------------------------------------------------------------------
_SQRT2 = math.sqrt(2.0)


def normal_cdf(x: float) -> float:
    """标准正态分布 CDF：``Φ(x) = 0.5 * (1 + erf(x/√2))``。"""
    return 0.5 * (1.0 + math.erf(x / _SQRT2))


def normal_sf(x: float) -> float:
    """标准正态分布生存函数（右尾概率）：``1 - Φ(x)``，数值更稳定。"""
    return 0.5 * math.erfc(x / _SQRT2)


def normal_ppf(p: float) -> float:
    """标准正态分位数（二分法反解 CDF，确定性、无外部依赖）。"""
    if not 0.0 < p < 1.0:
        raise ValueError("p 必须满足 0 < p < 1")
    lo, hi = -40.0, 40.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if normal_cdf(mid) < p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# ---------------------------------------------------------------------------
# 比例检验
# ---------------------------------------------------------------------------
def z_test_proportion(k: int, n: int, p0: float = 0.5,
                      alternative: str = "greater") -> dict:
    """单样本比例 z 检验。

    :param k: 成功次数（例如 bit-1 的个数）。
    :param n: 试验次数。
    :param p0: 零假设比例（默认 0.5）。
    :param alternative: ``"greater"`` / ``"less"`` / ``"two-sided"``。
    :returns: ``{"z": ..., "p": ..., "phat": ..., "n": ..., "p0": ..., "method": "normal"}``
    """
    if n <= 0:
        return {"z": 0.0, "p": 1.0, "phat": 0.0, "n": 0, "p0": p0, "method": "normal"}
    phat = float(k) / n
    se = math.sqrt(p0 * (1.0 - p0) / n)
    if se == 0:
        return {"z": 0.0, "p": 1.0, "phat": phat, "n": n, "p0": p0, "method": "normal"}
    z = (phat - p0) / se
    if alternative == "greater":
        p = normal_sf(z)
    elif alternative == "less":
        p = normal_cdf(z)
    else:
        p = 2.0 * normal_sf(abs(z))
    return {"z": z, "p": min(1.0, max(0.0, p)), "phat": phat,
            "n": n, "p0": p0, "method": "normal"}


def _binomial_tail_exact(k: int, n: int, p0: float) -> float:
    """精确计算 ``P(X >= k), X ~ Binomial(n, p0)``。"""
    k = max(0, min(k, n))
    total = 0.0
    for i in range(k, n + 1):
        total += math.comb(n, i) * (p0 ** i) * ((1.0 - p0) ** (n - i))
    return min(1.0, max(0.0, total))


def binomial_test(k: int, n: int, p0: float = 0.5,
                  alternative: str = "greater") -> dict:
    """精确二项检验（n ≤ 1000 精确求和，否则正态近似）。

    :param k: 成功次数。
    :param n: 试验次数。
    :param p0: 零假设比例。
    :param alternative: ``"greater"`` / ``"less"`` / ``"two-sided"``。
    :returns: ``{"p": ..., "n": ..., "k": ..., "p0": ..., "method": "exact"|"normal"}``
    """
    n = int(n)
    k = int(k)
    if n <= 0:
        return {"p": 1.0, "n": 0, "k": 0, "p0": p0, "method": "exact"}
    if n <= 1000:
        if alternative == "greater":
            p = _binomial_tail_exact(k, n, p0)
        elif alternative == "less":
            p = _binomial_tail_exact(k, n, p0)
            p = 1.0 - p + _binomial_pmf(k, n, p0)
            p = min(1.0, max(0.0, p))
        else:
            p_greater = _binomial_tail_exact(k, n, p0)
            p_less = 1.0 - p_greater + _binomial_pmf(k, n, p0)
            p = min(1.0, 2.0 * min(p_greater, p_less))
        return {"p": p, "n": n, "k": k, "p0": p0, "method": "exact"}
    z = z_test_proportion(k, n, p0, alternative)
    return {"p": z["p"], "n": n, "k": k, "p0": p0, "method": "normal"}


def _binomial_pmf(k: int, n: int, p0: float) -> float:
    if not 0 <= k <= n:
        return 0.0
    return math.comb(n, k) * (p0 ** k) * ((1.0 - p0) ** (n - k))


# ---------------------------------------------------------------------------
# t 分布（不完全 Beta 函数实现）
# ---------------------------------------------------------------------------
def _betacf(a: float, b: float, x: float) -> float:
    """不完全 Beta 函数的连分式（Numerical Recipes 6.4，纯 Python 重写）。"""
    max_iter, eps, fpmin = 200, 3.0e-14, 1.0e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < fpmin:
        d = fpmin
    d = 1.0 / d
    h = d
    for m in range(1, max_iter + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h


def _betai(a: float, b: float, x: float) -> float:
    """正则化不完全 Beta 函数 ``I_x(a, b)``。"""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    log_bt = (math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
              + a * math.log(x) + b * math.log1p(-x))
    bt = math.exp(log_bt)
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * _betacf(a, b, x) / a
    return 1.0 - bt * _betacf(b, a, 1.0 - x) / b


def t_cdf(t: float, df: int) -> float:
    """学生 t 分布 CDF。"""
    if df <= 0:
        raise ValueError("自由度 df 必须为正整数")
    x = df / (df + t * t)
    tail = 0.5 * _betai(df / 2.0, 0.5, x)  # P(T > |t|)
    return 1.0 - tail if t >= 0 else tail


def t_sf(t: float, df: int) -> float:
    """学生 t 分布右尾概率 ``P(T > t)``（用于单侧检验）。"""
    return 1.0 - t_cdf(t, df)


def t_quantile(p: float, df: int) -> float:
    """t 分布分位数（二分法，df ≥ 1）。"""
    if not 0.5 <= p < 1.0:
        raise ValueError("仅实现 p ∈ [0.5, 1) 的分位数")
    lo, hi = 0.0, 200.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if t_cdf(mid, df) < p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _stdev(values: Sequence[float]) -> float:
    n = len(values)
    if n < 2:
        return 0.0
    mu = _mean(values)
    return math.sqrt(sum((v - mu) ** 2 for v in values) / (n - 1))


def paired_t_test(diffs: Iterable[float]) -> dict:
    """单样本（配对差）t 检验，右尾：均值是否显著大于 0。

    ``t = mean(d) / (sd(d)/sqrt(n))``，p 值由 t 分布给出；同时返回 95% 置信区间。

    单测覆盖：已知 ``diffs`` 的 t/p 与手算一致；``n=1`` 时退化为 p=0.5（无信息）。
    """
    d = [float(x) for x in diffs]
    n = len(d)
    if n == 0:
        return {"t": 0.0, "p": 1.0, "n": 0, "mean": 0.0, "sd": 0.0, "ci95": [0.0, 0.0]}
    mean = _mean(d)
    sd = _stdev(d)
    if n == 1:
        return {"t": 0.0, "p": 0.5, "n": 1, "mean": mean, "sd": 0.0,
                "ci95": [mean, mean]}
    se = sd / math.sqrt(n) if sd > 0 else 0.0
    if se == 0.0:
        t = 0.0 if mean == 0 else math.copysign(float("inf"), mean)
        p = 0.5 if mean == 0 else (0.0 if mean > 0 else 1.0)
    else:
        t = mean / se
        p = t_sf(t, n - 1)
    try:
        tcrit = t_quantile(0.975, n - 1)
    except ValueError:  # pragma: no cover
        tcrit = normal_ppf(0.975)
    margin = tcrit * se
    return {"t": t, "p": min(1.0, max(0.0, p)), "n": n, "mean": mean, "sd": sd,
            "se": se, "ci95": [mean - margin, mean + margin]}


def sign_test(diffs: Iterable[float]) -> dict:
    """符号检验：统计正差个数，对 ``p0 = 0.5`` 做精确二项检验。

    对噪声分布没有正态假设，适合版本对照型水印的稳健判定。
    """
    d = [float(x) for x in diffs if float(x) != 0.0]
    n = len(d)
    k = sum(1 for x in d if x > 0)
    result = binomial_test(k, n, 0.5, alternative="greater")
    return {"k": k, "n": n, "p": result["p"], "method": result["method"]}


# ---------------------------------------------------------------------------
# 阈值校准与 TPR@FPR
# ---------------------------------------------------------------------------
def calibrate_threshold(scores_null: Sequence[float], alpha: float) -> float:
    """用负样本分数校准阈值，保证经验误报率 ``FPR ≤ α``。

    取升序排列后的第 ``n - 1 - floor(α·n)`` 个分数作为阈值（即大约第
    ``1-α`` 经验分位数）。这样零假设模型分数**严格超过**阈值的比例最多为
    ``floor(α·n)/n ≤ α``，在样本量有限时偏保守，不会被 α 卡住。

    :param scores_null: 零假设（干净模型）下的统计量列表。
    :param alpha: 目标误报率 α。
    :returns: 阈值；零假设分数超过它即判为「检出」。
    """
    if not scores_null:
        raise ValueError("scores_null 不能为空")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha 必须在 (0, 1) 内")
    scores = sorted(float(x) for x in scores_null)
    n = len(scores)
    idx = n - 1 - int(math.floor(alpha * n))
    idx = max(0, min(n - 1, idx))
    return scores[idx]


def empirical_fpr(scores_null: Sequence[float], threshold: float) -> float:
    """经验误报率：零假设分数严格超过阈值的比例。"""
    if not scores_null:
        return 0.0
    return sum(1 for s in scores_null if s > threshold) / len(scores_null)


def tpr_at_fpr(scores_pos: Sequence[float], threshold: float) -> float:
    """给定阈值下的检出率（TPR）：正样本分数严格超过阈值的比例。"""
    if not scores_pos:
        return 0.0
    return sum(1 for s in scores_pos if s > threshold) / len(scores_pos)


def effect_size_summary(values: Sequence[float]) -> dict:
    """描述性统计：均值、标准差、中位数、分位数（用于报告 detail）。"""
    vals = sorted(float(v) for v in values)
    if not vals:
        return {"n": 0, "mean": 0.0, "sd": 0.0, "median": 0.0}

    def quantile(q: float) -> float:
        if len(vals) == 1:
            return vals[0]
        pos = (len(vals) - 1) * q
        lo = int(math.floor(pos))
        hi = min(len(vals) - 1, lo + 1)
        frac = pos - lo
        return vals[lo] * (1.0 - frac) + vals[hi] * frac

    return {"n": len(vals), "mean": _mean(vals), "sd": _stdev(vals),
            "median": quantile(0.5), "q05": quantile(0.05), "q95": quantile(0.95)}
