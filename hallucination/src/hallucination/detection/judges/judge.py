from __future__ import annotations

import json
import re
from typing import Any

from hallucination.detection.core.base import BaseScorer
from hallucination.detection.core.results import DetectionExample, ScoreResult
from hallucination.detection.core.thresholds import clamp01, confidence_to_risk

_MODES = {"binary", "continuous", "likert"}


class LLMJudgeScorer(BaseScorer):
    """LLM-as-a-Judge with a structured, auditable output contract.

    Modes:

    - ``binary``: ``{"verdict": "faithful"|"hallucinated", "confidence": 0..1}``
    - ``continuous``: ``{"hallucination_score": 0..100}``
    - ``likert``: ``{"faithfulness": 1..5}`` (5 = fully faithful)

    The judge model is a separate scoring model, not the answer generator. The
    prompt and raw judge response are retained for inspection.
    """

    name = "llm_judge"
    requires = {"prompt", "response"}
    signal_type = "judge_assessment"

    def __init__(self, backend: Any, *, mode: str = "binary", judge_name: str | None = None) -> None:
        if mode not in _MODES:
            raise ValueError(f"mode must be one of {sorted(_MODES)}")
        if not hasattr(backend, "generate"):
            raise TypeError("judge backend must provide generate()")
        self.backend = backend
        self.mode = mode
        self.judge_name = judge_name or getattr(backend, "model_name", type(backend).__name__)

    def score(self, example: DetectionExample) -> ScoreResult:
        self.validate(example)
        assert example.prompt is not None
        prompt = _judge_prompt(self.mode, example.prompt, example.response)
        text = self.backend.generate(prompt, temperature=0.0)
        parsed = _parse_json_object(text)
        risk, details = _risk_from_parsed(parsed, self.mode)

        return ScoreResult(
            method=self.name,
            signal_type=self.signal_type,
            score=confidence_to_risk(risk),
            risk=risk,
            level="response",
            raw={
                "prompt": prompt,
                "judge_response": text,
                "parsed": parsed,
                "mode": self.mode,
                "risk_details": details,
            },
            explanation="Structured judgement from an independent judge model.",
            assumptions=["The judge model follows the JSON output format and evaluates the answer, not the prompt alone."],
            warnings=[
                "A judge model can share the generator's blind spots.",
                "Judge outputs are not calibrated probabilities of factual correctness.",
            ],
            provenance={"judge": self.judge_name, "mode": self.mode},
        )


class LLMPanelScorer(BaseScorer):
    """Average the risk of multiple independent LLM judges."""

    name = "llm_panel"
    requires = {"prompt", "response"}
    signal_type = "judge_assessment"

    def __init__(self, judges: list[LLMJudgeScorer]) -> None:
        if not judges:
            raise ValueError("LLMPanelScorer requires at least one judge")
        self.judges = list(judges)

    def score(self, example: DetectionExample) -> ScoreResult:
        self.validate(example)
        results = [judge.score(example) for judge in self.judges]
        risk = clamp01(sum(result.risk for result in results) / len(results))
        return ScoreResult(
            method=self.name,
            signal_type=self.signal_type,
            score=confidence_to_risk(risk),
            risk=risk,
            level="response",
            raw={
                "judges": [
                    {
                        "judge": result.provenance.get("judge"),
                        "risk": result.risk,
                        "mode": result.provenance.get("mode"),
                        "parsed": result.raw.get("parsed"),
                    }
                    for result in results
                ],
                "num_judges": len(results),
            },
            explanation="Average risk from a panel of independent judge models.",
            assumptions=["Judges are evaluated independently and are not fine-tuned on the test labels."],
            warnings=[
                "Panel averaging reduces variance but does not remove shared model bias.",
                "Judge outputs are not calibrated probabilities of factual correctness.",
            ],
            provenance={"judges": [result.provenance.get("judge") for result in results]},
        )


def _judge_prompt(mode: str, question: str, answer: str) -> str:
    schemas = {
        "binary": '{"verdict": "faithful" or "hallucinated", "confidence": 0.0-1.0}',
        "continuous": '{"hallucination_score": 0-100, "reason": "short"}',
        "likert": '{"faithfulness": 1-5, "reason": "short"}',
    }
    instructions = {
        "binary": "Decide whether the answer contains a factual hallucination.",
        "continuous": "Rate the answer's hallucination severity from 0 (fully factual) to 100 (clearly hallucinated).",
        "likert": "Rate faithfulness from 1 (clearly hallucinated) to 5 (fully faithful).",
    }
    return (
        "You are a meticulous factual judge. "
        f"{instructions[mode]}\n"
        "Return JSON only, with no markdown fences or extra text.\n"
        f"Schema: {schemas[mode]}\n"
        f"Question: {question}\n"
        f"Answer: {answer}\n"
        "JSON:"
    )


def _parse_json_object(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if not match:
        raise ValueError(f"judge response did not contain a JSON object: {text!r}")
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        raise ValueError(f"judge response was not valid JSON: {text!r}") from exc
    if not isinstance(parsed, dict):
        raise ValueError("judge JSON must be an object")
    return parsed


def _risk_from_parsed(parsed: dict[str, Any], mode: str) -> tuple[float, dict[str, Any]]:
    if "risk" in parsed:
        risk = clamp01(float(parsed["risk"]))
        return risk, {"source": "risk"}

    if mode == "binary":
        verdict = str(parsed.get("verdict", "")).strip().lower()
        confidence = float(parsed.get("confidence", 1.0))
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("binary confidence must be in [0, 1]")
        if verdict in {"hallucinated", "unfaithful", "false", "not_faithful"}:
            return clamp01(confidence), {"source": "verdict", "verdict": verdict, "confidence": confidence}
        if verdict in {"faithful", "true", "supported"}:
            return clamp01(1.0 - confidence), {"source": "verdict", "verdict": verdict, "confidence": confidence}
        raise ValueError(f"unrecognized binary verdict: {verdict!r}")

    if mode == "continuous":
        raw = float(parsed.get("hallucination_score", parsed.get("score", "nan")))
        if raw != raw:
            raise ValueError("continuous judge response requires hallucination_score")
        risk = raw / 100.0 if raw > 1.0 else raw
        return clamp01(risk), {"source": "hallucination_score", "value": raw}

    raw = float(parsed.get("faithfulness", parsed.get("likert", "nan")))
    if raw != raw or not 1.0 <= raw <= 5.0:
        raise ValueError("likert judge response requires faithfulness in [1, 5]")
    return clamp01((5.0 - raw) / 4.0), {"source": "faithfulness", "value": raw}
