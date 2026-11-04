"""Paired significance statistics for baseline-vs-mitigated comparisons.

The benchmark scores each example twice — once with regular decoding
(baseline) and once with the mitigation — and compares against references.
Per example the outcome is one of:

- win:   baseline wrong, mitigated correct  (0 -> 1)
- tie:   same outcome                       (0 -> 0 or 1 -> 1)
- loss:  baseline correct, mitigated wrong  (1 -> 0)

Ties carry no evidence about the direction of the effect, so all tests are
computed on the decisive pairs. The exact two-sided binomial test on decisive
wins is simultaneously the sign test for paired outcomes and McNemar's exact
test for binary per-example correctness. Effect sizes are reported as the
decisive win rate with a Wilson score interval; no continuity or normal
approximations are used anywhere.

Requires no scipy: the exact binomial p-value is computed from binomial
coefficients directly, which stays exact for any n relevant here.
"""

from __future__ import annotations

import math
from typing import Any

Z_95 = 1.959963984540054


def binom_pmf(successes: int, trials: int) -> float:
    if not 0 <= successes <= trials:
        raise ValueError("successes must be within [0, trials]")
    if trials < 0:
        raise ValueError("trials must be non-negative")
    return math.comb(trials, successes) * 0.5**trials


def two_sided_binomial_p(successes: int, trials: int) -> float:
    """Exact two-sided binomial test p-value against p=0.5 (no bias)."""

    if trials <= 0:
        raise ValueError("trials must be positive")
    observed = binom_pmf(successes, trials)
    total = sum(
        binom_pmf(k, trials) for k in range(trials + 1) if binom_pmf(k, trials) <= observed + 1e-15
    )
    return min(1.0, total)


def wilson_interval(successes: int, trials: int, z: float = Z_95) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""

    if trials <= 0:
        raise ValueError("trials must be positive")
    if not 0 <= successes <= trials:
        raise ValueError("successes must be within [0, trials]")
    p_hat = successes / trials
    z_squared = z * z
    denominator = 1.0 + z_squared / trials
    center = p_hat + z_squared / (2.0 * trials)
    spread = z * math.sqrt(p_hat * (1.0 - p_hat) / trials + z_squared / (4.0 * trials * trials))
    return (
        max(0.0, (center - spread) / denominator),
        min(1.0, (center + spread) / denominator),
    )


def paired_sign_stats(win: int, tie: int, loss: int, *, alpha: float = 0.05) -> dict[str, Any]:
    """Significance summary for paired baseline-vs-mitigated outcomes.

    Returns ``None``-valued statistics when there are no decisive pairs, plus a
    conservative verdict and the required-sample note when power is low.
    """

    if min(win, tie, loss) < 0:
        raise ValueError("win/tie/loss counts must be non-negative")
    decisive = win + loss
    stats: dict[str, Any] = {
        "n": win + tie + loss,
        "win": win,
        "tie": tie,
        "loss": loss,
        "decisive": decisive,
        "decisive_win_rate": None,
        "decisive_win_rate_ci95": None,
        "p_value": None,
        "alpha": alpha,
        "verdict": "inconclusive: no decisive pairs",
        "warnings": [],
    }
    if decisive == 0:
        return stats
    win_rate = win / decisive
    stats["decisive_win_rate"] = win_rate
    stats["decisive_win_rate_ci95"] = list(wilson_interval(win, decisive))
    p_value = two_sided_binomial_p(win, decisive)
    stats["p_value"] = p_value
    if p_value < alpha:
        stats["verdict"] = (
            "mitigation significantly improves correctness"
            if win > loss
            else "mitigation significantly degrades correctness"
        )
    else:
        stats["verdict"] = "no statistically significant difference"
    if decisive < 10:
        stats["warnings"].append(
            "fewer than 10 decisive pairs; the study is underpowered and the interval is wide"
        )
    return stats


def holm_adjust(pvalues: list[float], *, alpha: float = 0.05) -> list[dict[str, Any]]:
    """Holm-Bonferroni step-down adjustment across multiple comparisons."""

    if not pvalues:
        return []
    order = sorted(range(len(pvalues)), key=lambda index: pvalues[index])
    results: list[dict[str, Any]] = [dict() for _ in pvalues]  # type: ignore[list-item]
    running_max = 0.0
    for rank, index in enumerate(order):
        adjusted = min(1.0, (len(pvalues) - rank) * pvalues[index])
        running_max = max(running_max, adjusted)
        results[index] = {
            "p_value": pvalues[index],
            "p_holm": running_max,
            "significant": running_max < alpha,
        }
    return results
