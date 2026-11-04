from __future__ import annotations

import math
import re
from typing import Any

from hallucination.detection.core.base import BaseScorer
from hallucination.detection.core.results import DetectionExample, ScoreResult
from hallucination.detection.core.thresholds import clamp01, confidence_to_risk

_TRUE_TOKENS = (" A", "A", " True", "True")
_FALSE_TOKENS = (" B", "B", " False", "False")


def build_p_true_prompt(question: str, response: str, brainstormed_answers: list[str] | None = None) -> str:
    """Build the official-style P(True) prompt.

    The model is asked to choose between ``A) True`` and ``B) False`` for the
    proposed answer. The caller then reads the probability of the `` A`` / `` B``
    continuation.
    """

    lines = [f"Question: {question}", "Brainstormed Answers:"]
    for answer in brainstormed_answers or []:
        lines.append(answer.strip())
    lines.extend(
        [
            f"Possible answer: {response.strip()}",
            "Is the possible answer:",
            "A) True",
            "B) False",
            "The possible answer is:",
        ]
    )
    return "\n".join(lines)


class PTrueScorer(BaseScorer):
    """P(True): the model's probability that its answer is true.

    With a logprob-capable backend (``next_token_logprobs``), the scorer reads
    the probability of the `` A`` and `` B`` continuations. Otherwise it falls
    back to a temperature-zero verbalized True/False answer. The verbalized
    fallback is not the same estimator as logprob P(True) and is labelled in
    ``raw["method"]``.
    """

    name = "p_true"
    requires = {"prompt", "response"}
    signal_type = "self_reported_confidence"

    def __init__(
        self,
        backend: Any,
        *,
        use_samples_as_brainstorm: bool = True,
        max_brainstorm: int = 4,
        normalize_ab: bool = True,
    ) -> None:
        if not hasattr(backend, "generate") and not hasattr(backend, "next_token_logprobs"):
            raise TypeError("backend must provide generate() or next_token_logprobs()")
        if max_brainstorm < 0:
            raise ValueError("max_brainstorm must be non-negative")
        self.backend = backend
        self.use_samples_as_brainstorm = use_samples_as_brainstorm
        self.max_brainstorm = max_brainstorm
        self.normalize_ab = normalize_ab

    def score(self, example: DetectionExample) -> ScoreResult:
        self.validate(example)
        assert example.prompt is not None
        brainstormed = list(example.samples or [])[: self.max_brainstorm] if self.use_samples_as_brainstorm else []
        prompt = build_p_true_prompt(example.prompt, example.response, brainstormed)

        fallback_reason: str | None = None
        if hasattr(self.backend, "next_token_logprobs"):
            try:
                logprobs = self.backend.next_token_logprobs(prompt, list(_TRUE_TOKENS) + list(_FALSE_TOKENS))
                p_true, details = _probability_from_logprobs(logprobs, normalize_ab=self.normalize_ab)
                method = "logprob"
                raw: dict[str, Any] = {
                    "prompt": prompt,
                    "method": method,
                    "logprobs": logprobs,
                    "token_details": details,
                    "brainstormed_answers": brainstormed,
                    "normalize_ab": self.normalize_ab,
                }
            except ValueError as exc:
                # API backends only expose a top-k window, which may not contain
                # the A/B tokens; fall back to the verbalized estimator.
                fallback_reason = str(exc)
        if fallback_reason is not None or not hasattr(self.backend, "next_token_logprobs"):
            text = self.backend.generate(prompt, temperature=0.0)
            p_true = _parse_true_false(text)
            method = "verbalized"
            raw = {
                "prompt": prompt,
                "method": method,
                "judge_response": text,
                "brainstormed_answers": brainstormed,
            }
            if fallback_reason is not None:
                raw["fallback_reason"] = fallback_reason

        p_true = clamp01(p_true)
        return ScoreResult(
            method=self.name,
            signal_type=self.signal_type,
            score=p_true,
            risk=confidence_to_risk(p_true),
            level="response",
            raw=raw,
            explanation="The model's estimated probability that its own answer is true.",
            assumptions=[
                "The backend can follow the True/False choice format.",
                "With logprob scoring, the A/B token probabilities are normalized over True/False only."
                if self.normalize_ab
                else "With logprob scoring, P(True) is exp(logprob(' A')) without A/B normalization.",
            ],
            warnings=[
                "P(True) is a self-report and is often overconfident; calibrate it on validation data.",
                "Verbalized True/False is a fallback and is not identical to logprob P(True).",
            ]
            + (
                [f"Logprob scoring was unavailable and the verbalized fallback was used: {fallback_reason}"]
                if fallback_reason is not None
                else []
            ),
            provenance={
                "backend": getattr(self.backend, "model_name", type(self.backend).__name__),
                "method": method,
                "brainstormed_count": len(brainstormed),
                "normalize_ab": self.normalize_ab,
            },
        )


def _probability_from_logprobs(logprobs: dict[str, float], *, normalize_ab: bool = True) -> tuple[float, dict[str, float]]:
    true_values = [float(logprobs[token]) for token in _TRUE_TOKENS if token in logprobs]
    false_values = [float(logprobs[token]) for token in _FALSE_TOKENS if token in logprobs]
    if not true_values or not false_values:
        raise ValueError("backend must return logprobs for at least one True and one False token")
    true_logprob = max(true_values)
    false_logprob = max(false_values)
    if not math.isfinite(true_logprob) or not math.isfinite(false_logprob):
        raise ValueError("P(True) logprobs must be finite")
    if not normalize_ab:
        return float(min(1.0, max(0.0, math.exp(true_logprob)))), {
            "true_logprob": true_logprob,
            "false_logprob": false_logprob,
            "true_probability": float(math.exp(true_logprob)),
            "false_probability": float(math.exp(false_logprob)),
            "normalized": False,
        }
    largest = max(true_logprob, false_logprob)
    true_probability = math.exp(true_logprob - largest)
    false_probability = math.exp(false_logprob - largest)
    total = true_probability + false_probability
    if total <= 0:
        raise ValueError("P(True) probabilities collapsed to zero")
    return true_probability / total, {
        "true_logprob": true_logprob,
        "false_logprob": false_logprob,
        "true_probability": true_probability / total,
        "false_probability": false_probability / total,
        "normalized": True,
    }


def _parse_true_false(text: str) -> float:
    normalized = text.strip().lower()
    choice = re.fullmatch(r"\(?([ab])\)?[.)]?", normalized)
    if choice:
        return 1.0 if choice.group(1) == "a" else 0.0
    matches = re.findall(r"\b(true|false|yes|no)\b", normalized)
    if not matches:
        raise ValueError(f"could not parse True/False from judge response: {text!r}")
    value = matches[0]
    return 1.0 if value in {"true", "yes"} else 0.0
