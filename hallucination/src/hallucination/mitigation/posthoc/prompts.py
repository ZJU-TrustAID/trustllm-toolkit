from __future__ import annotations


def baseline_messages(question: str) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": "Answer the user's question directly and concisely. If you are unsure, say what is uncertain.",
        },
        {"role": "user", "content": question.strip()},
    ]


def plan_messages(
    question: str, baseline: str, max_questions: int
) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "You design factual verification questions. Given a question and a draft answer, "
                "write independent, answerable questions that check the factual claims in the draft. "
                "Do not answer them. Return only a numbered list, one question per line. "
                "Write the questions in the same language as the user's question."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Question:\n{question.strip()}\n\nDraft answer:\n{baseline.strip()}\n\n"
                f"Write at most {max_questions} verification questions."
            ),
        },
    ]


def verify_messages(question: str) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "Answer the verification question from your own knowledge. "
                "Be concise and do not repeat the original draft answer. "
                "Answer in the same language as the question. "
                "If the answer is unknown, say 'unknown'."
            ),
        },
        {"role": "user", "content": question.strip()},
    ]


def joint_verify_messages(questions: list[str]) -> list[dict[str, str]]:
    numbered = "\n".join(
        f"{index + 1}. {question}" for index, question in enumerate(questions)
    )
    return [
        {
            "role": "system",
            "content": (
                "Answer each verification question independently and concisely. "
                "Return a numbered list with the same numbering; if unknown, write 'unknown'. "
                "Answer in the same language as the questions."
            ),
        },
        {"role": "user", "content": numbered},
    ]


def revise_messages(
    question: str, baseline: str, answers: list[tuple[str, str]]
) -> list[dict[str, str]]:
    qa_text = "\n".join(
        f"Q: {verification_question}\nA: {answer}"
        for verification_question, answer in answers
    )
    return [
        {
            "role": "system",
            "content": (
                "You revise a draft answer using independent verification answers. "
                "Keep information that is supported, correct what is contradicted, and omit claims that "
                "remain unknown. Do not invent new facts. Return only the final answer, in the same "
                "language as the original question and draft answer."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Original question:\n{question.strip()}\n\n"
                f"Draft answer:\n{baseline.strip()}\n\n"
                f"Verification question/answer pairs:\n{qa_text}\n\n"
                "Write the corrected final answer."
            ),
        },
    ]
