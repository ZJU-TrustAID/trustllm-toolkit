from __future__ import annotations

import re
from typing import Any

from hallucination.detection.core.base import BaseScorer
from hallucination.detection.core.results import DetectionExample, ScoreResult
from hallucination.detection.core.thresholds import clamp01, confidence_to_risk
from hallucination.detection.utils.text import normalize_answer


class CoVeScorer(BaseScorer):
    """Chain-of-Verification (CoVe) detector.

    The detector generates verification questions from the draft answer,
    answers each question independently, and reports the fraction of answers
    that conflict with the draft. It never rewrites the answer; mitigation is a
    separate concern.

    A bidirectional NLI backend is the recommended verifier. Without one, the
    scorer falls back to a transparent character-bigram overlap heuristic and
    labels this in ``raw["verifier"]``.
    """

    name = "cove_detector"
    requires = {"prompt", "response"}
    signal_type = "self_reported_confidence"

    def __init__(
        self,
        backend: Any,
        *,
        nli: Any | None = None,
        max_questions: int = 5,
        contradiction_threshold: float = 0.5,
        min_similarity: float = 0.2,
    ) -> None:
        if not hasattr(backend, "generate"):
            raise TypeError("CoVeScorer requires a backend with generate()")
        if not 1 <= max_questions <= 10:
            raise ValueError("max_questions must be between 1 and 10")
        if not 0.0 <= contradiction_threshold <= 1.0:
            raise ValueError("contradiction_threshold must be in [0, 1]")
        if not 0.0 <= min_similarity <= 1.0:
            raise ValueError("min_similarity must be in [0, 1]")
        self.backend = backend
        self.nli = nli
        self.max_questions = max_questions
        self.contradiction_threshold = contradiction_threshold
        self.min_similarity = min_similarity

    def score(self, example: DetectionExample) -> ScoreResult:
        self.validate(example)
        assert example.prompt is not None
        question_prompt = _question_prompt(example.prompt, example.response, self.max_questions)
        generated = self.backend.generate(question_prompt, temperature=0.0)
        questions = _parse_questions(generated, self.max_questions)
        if not questions:
            raise ValueError("CoVe question generation returned no parsable questions")

        comparisons = []
        conflicts = 0
        for question in questions:
            answer = self.backend.generate(_answer_prompt(question), temperature=0.0)
            if self.nli is not None:
                forward = self.nli(example.response, answer)
                backward = self.nli(answer, example.response)
                contradiction = max(forward.contradiction, backward.contradiction)
                conflict = contradiction >= self.contradiction_threshold
                comparisons.append(
                    {
                        "question": question,
                        "verification_answer": answer,
                        "contradiction": contradiction,
                        "conflict": conflict,
                    }
                )
            else:
                similarity = _lexical_similarity(example.response, answer)
                conflict = similarity < self.min_similarity
                comparisons.append(
                    {
                        "question": question,
                        "verification_answer": answer,
                        "lexical_similarity": similarity,
                        "conflict": conflict,
                    }
                )
            conflicts += int(conflict)

        risk = clamp01(conflicts / len(comparisons))
        return ScoreResult(
            method=self.name,
            signal_type=self.signal_type,
            score=confidence_to_risk(risk),
            risk=risk,
            level="response",
            raw={
                "question_prompt": question_prompt,
                "generated_questions": generated,
                "questions": questions,
                "comparisons": comparisons,
                "conflict_rate": risk,
                "num_questions": len(questions),
                "verifier": "nli" if self.nli is not None else "lexical_overlap",
            },
            explanation="Fraction of independently answered verification questions that conflict with the draft answer.",
            assumptions=[
                "Verification questions cover the main factual claims in the draft.",
                "Verification answers are generated independently from the draft.",
            ],
            warnings=[
                "CoVe can amplify a bad draft if question generation inherits its errors.",
                "A low conflict rate is self-consistency evidence, not external factual verification.",
                "The lexical fallback is a heuristic; use an NLI or judge verifier for real use.",
            ],
            provenance={
                "backend": getattr(self.backend, "model_name", type(self.backend).__name__),
                "max_questions": self.max_questions,
                "verifier": "nli" if self.nli is not None else "lexical_overlap",
            },
        )


def _question_prompt(question: str, draft: str, max_questions: int) -> str:
    return (
        "You are checking a draft answer for factual errors.\n"
        f"Question: {question}\n"
        f"Draft answer: {draft}\n"
        f"Write up to {max_questions} short verification questions that check the factual claims in the draft.\n"
        "Return one question per line, without numbering or bullets.\n"
        "Verification questions:"
    )


def _answer_prompt(question: str) -> str:
    return (
        "Answer the following question independently and concisely. "
        "Do not refer to any draft answer.\n"
        f"Question: {question}\n"
        "Answer:"
    )


def _parse_questions(text: str, limit: int) -> list[str]:
    questions: list[str] = []
    for line in text.splitlines():
        cleaned = re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", line).strip()
        if not cleaned or not cleaned.endswith(("?", "？")):
            continue
        questions.append(cleaned)
        if len(questions) >= limit:
            break
    return questions


def _lexical_similarity(left: str, right: str) -> float:
    left_grams = _bigrams(normalize_answer(left))
    right_grams = _bigrams(normalize_answer(right))
    if not left_grams or not right_grams:
        return 0.0
    return len(left_grams & right_grams) / len(left_grams | right_grams)


def _bigrams(text: str) -> set[str]:
    compact = re.sub(r"\s+", "", text)
    if len(compact) < 2:
        return {compact} if compact else set()
    return {compact[index : index + 2] for index in range(len(compact) - 1)}
