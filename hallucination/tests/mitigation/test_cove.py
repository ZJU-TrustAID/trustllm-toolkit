from __future__ import annotations

from hallucination.mitigation.backends.callable import CallableBackend
from hallucination.mitigation.core.contracts import GenerationRequest
from hallucination.mitigation.methods.cove import CoVeMethod, parse_questions


def test_parse_questions_numbered_list() -> None:
    text = "1. Is the sky blue?\n2. Is grass green?\n3. Not a question"
    assert parse_questions(text, 6) == ["Is the sky blue?", "Is grass green?"]


def test_parse_questions_json_list() -> None:
    text = '["Is A true?", "Is B true?"]'
    assert parse_questions(text, 6) == ["Is A true?", "Is B true?"]


def test_parse_questions_chinese_fullwidth_mark() -> None:
    text = "1. 蓝鲸的食性是什么？\n2. 蓝鲸属于哪类动物？"
    assert parse_questions(text, 6) == ["蓝鲸的食性是什么？", "蓝鲸属于哪类动物？"]


def test_cove_runs_four_stages_with_callable_backend() -> None:
    backend = CallableBackend(
        [
            "Draft answer.",
            "1. Is fact A true?\n2. Is fact B true?",
            "Yes.",
            "No.",
            "Revised answer.",
        ]
    )
    report = CoVeMethod().run(
        backend, GenerationRequest(prompt="Question?", max_new_tokens=32)
    )
    assert report.baseline == "Draft answer."
    assert report.response == "Revised answer."
    assert len(report.trace["questions"]) == 2
    assert report.trace["answers"][0]["answer"] == "Yes."
    assert len(backend.calls) == 5


def test_cove_falls_back_when_plan_has_no_questions() -> None:
    backend = CallableBackend(["Draft.", "No questions here."])
    report = CoVeMethod().run(
        backend, GenerationRequest(prompt="Question?", max_new_tokens=32)
    )
    assert report.response == "Draft."
    assert any("No verification questions" in warning for warning in report.warnings)
