from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from typing import Any

from hallucination.mitigation.core.contracts import GenerationRequest, MitigationReport
from hallucination.mitigation.core.errors import GenerationError
from hallucination.mitigation.posthoc.prompts import (
    baseline_messages,
    joint_verify_messages,
    plan_messages,
    revise_messages,
    verify_messages,
)


@dataclass
class CoVeConfig:
    max_questions: int = 6
    mode: str = "factored"
    temperature: float = 0.0
    max_answer_tokens: int = 128
    baseline: bool = True

    def __post_init__(self) -> None:
        if self.max_questions < 1:
            raise ValueError("max_questions must be positive")
        if self.mode not in {"factored", "joint"}:
            raise ValueError("mode must be 'factored' or 'joint'")
        if self.max_answer_tokens < 1:
            raise ValueError("max_answer_tokens must be positive")


def parse_questions(text: str, max_questions: int = 6) -> list[str]:
    raw = (text or "").strip()
    if not raw:
        return []
    if raw.startswith("[") and raw.endswith("]"):
        try:
            payload = json.loads(raw)
            if isinstance(payload, list):
                return _clean_questions([str(item) for item in payload], max_questions)
        except json.JSONDecodeError:
            pass
    lines = [line.strip() for line in raw.splitlines() if line.strip()]
    questions = []
    for line in lines:
        line = re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", line).strip()
        line = line.strip('"').strip("'").strip()
        if not line:
            continue
        if line.lower().startswith(
            ("verification questions", "questions:", "final verification")
        ):
            continue
        if line.endswith(("?", "？")):
            questions.append(line)
    if not questions:
        questions = [line for line in lines if line.endswith(("?", "？"))]
    return _clean_questions(questions, max_questions)


def _clean_questions(items: list[str], max_questions: int) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for item in items:
        question = re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", item).strip()
        if not question or question in seen:
            continue
        seen.add(question)
        result.append(question)
        if len(result) >= max_questions:
            break
    return result


class CoVeMethod:
    name = "cove"

    def run(
        self, backend: Any, request: GenerationRequest, config: CoVeConfig | None = None
    ) -> MitigationReport:
        config = config or CoVeConfig()
        started = time.perf_counter()
        stage_costs: list[dict[str, Any]] = []
        warnings = [
            "CoVe verifies with the same model's own knowledge; it is self-verification, not external fact-checking.",
            "A confident verification answer can still be wrong. Use retrieval or a judge when factual accuracy matters.",
        ]

        baseline_start = time.perf_counter()
        baseline = backend.generate(
            baseline_messages(request.prompt),
            max_new_tokens=request.max_new_tokens,
            temperature=config.temperature,
        )
        stage_costs.append(
            {
                "stage": "baseline",
                "seconds": round(time.perf_counter() - baseline_start, 4),
            }
        )
        if not baseline:
            raise GenerationError("CoVe baseline response is empty")

        plan_start = time.perf_counter()
        plan_text = backend.generate(
            plan_messages(request.prompt, baseline, config.max_questions),
            max_new_tokens=max(128, config.max_questions * 32),
            temperature=config.temperature,
        )
        stage_costs.append(
            {"stage": "plan", "seconds": round(time.perf_counter() - plan_start, 4)}
        )
        questions = parse_questions(plan_text, config.max_questions)
        if not questions:
            elapsed = time.perf_counter() - started
            return MitigationReport(
                method=self.name,
                prompt=request.prompt,
                baseline=baseline,
                response=baseline,
                context=request.context,
                trace={
                    "baseline": baseline,
                    "questions": [],
                    "answers": [],
                    "revised": baseline,
                    "plan_raw": plan_text,
                    "stage_costs": stage_costs,
                },
                warnings=warnings
                + ["No verification questions could be parsed; baseline was returned."],
                cost={"latency_seconds": round(elapsed, 4), "stage_costs": stage_costs},
                provenance={
                    "method": self.name,
                    "mode": config.mode,
                    "max_questions": config.max_questions,
                },
            )

        answers: list[tuple[str, str]] = []
        if config.mode == "factored":
            for question in questions:
                verify_start = time.perf_counter()
                answer = backend.generate(
                    verify_messages(question),
                    max_new_tokens=config.max_answer_tokens,
                    temperature=config.temperature,
                )
                answers.append((question, answer.strip()))
                stage_costs.append(
                    {
                        "stage": "verify",
                        "question": question,
                        "seconds": round(time.perf_counter() - verify_start, 4),
                    }
                )
        else:
            verify_start = time.perf_counter()
            joint = backend.generate(
                joint_verify_messages(questions),
                max_new_tokens=max(128, config.max_answer_tokens),
                temperature=config.temperature,
            )
            answers = [(question, joint.strip()) for question in questions]
            stage_costs.append(
                {
                    "stage": "verify_joint",
                    "seconds": round(time.perf_counter() - verify_start, 4),
                }
            )

        revise_start = time.perf_counter()
        revised = backend.generate(
            revise_messages(request.prompt, baseline, answers),
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            do_sample=request.do_sample,
        )
        stage_costs.append(
            {"stage": "revise", "seconds": round(time.perf_counter() - revise_start, 4)}
        )
        if not revised:
            revised = baseline
            warnings.append("Revision was empty; baseline was returned.")

        elapsed = time.perf_counter() - started
        return MitigationReport(
            method=self.name,
            prompt=request.prompt,
            baseline=baseline,
            response=revised.strip(),
            context=request.context,
            trace={
                "baseline": baseline,
                "questions": questions,
                "answers": [
                    {"question": question, "answer": answer}
                    for question, answer in answers
                ],
                "plan_raw": plan_text,
                "stage_costs": stage_costs,
            },
            warnings=warnings,
            cost={
                "latency_seconds": round(elapsed, 4),
                "calls": 1
                + 1
                + (len(questions) if config.mode == "factored" else 1)
                + 1,
                "stage_costs": stage_costs,
            },
            provenance={
                "method": self.name,
                "mode": config.mode,
                "max_questions": config.max_questions,
                "evidence": "self",
            },
        )
