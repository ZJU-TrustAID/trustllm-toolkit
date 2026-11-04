from __future__ import annotations


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def confidence_to_risk(score: float) -> float:
    return clamp01(1.0 - score)


def risk_to_label(risk: float, threshold: float = 0.5) -> bool:
    return clamp01(risk) >= threshold

