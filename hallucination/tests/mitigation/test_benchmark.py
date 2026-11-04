from __future__ import annotations

import pytest

from hallucination.mitigation.api.report import MitigationReport
from hallucination.mitigation.evaluation import evaluate_records
from hallucination.mitigation.evaluation.metrics import (
    contains_match,
    exact_match,
    normalize_text,
)


class FakeMitigator:
    method = "cove"

    def mitigate(self, prompt: str, context: str | None = None, **kwargs):
        return MitigationReport(
            method=self.method,
            prompt=prompt,
            baseline="wrong",
            response="the right answer",
        )


def test_normalize_and_metrics() -> None:
    assert normalize_text("  The  Answer! ") == "the answer"
    assert exact_match("The answer!", ["the answer"]) == 1.0
    assert contains_match("the right answer", ["right"]) == 1.0


def test_benchmark_aggregates_scores() -> None:
    records = [{"prompt": "Q?", "reference": "right answer"}]
    result = evaluate_records(FakeMitigator(), records, metric="contains")
    assert result.n == 1
    assert result.baseline_score == 0.0
    assert result.mitigated_score == 1.0
    assert result.win == 1
    assert "n" in result.to_markdown()


class ScriptedMitigator:
    """Returns per-call (baseline, mitigated) text pairs to control win/tie/loss."""

    method = "scripted"

    def __init__(self, pairs: list[tuple[str, str]]) -> None:
        self.pairs = list(pairs)
        self.calls = 0

    def mitigate(self, prompt: str, context: str | None = None, **kwargs):
        baseline, response = self.pairs[self.calls % len(self.pairs)]
        self.calls += 1
        return MitigationReport(method=self.method, prompt=prompt, baseline=baseline, response=response)


def test_benchmark_significance_section_reports_exact_binomial() -> None:
    # 9 decisive wins, 1 decisive loss, 30 ties
    pairs = [("wrong", "right")] * 9 + [("right", "wrong")] + [("wrong", "wrong")] * 30
    records = [{"prompt": f"Q{i}?", "reference": "right"} for i in range(len(pairs))]
    result = evaluate_records(ScriptedMitigator(pairs), records, metric="contains")
    stats = result.significance
    assert stats is not None
    assert stats["win"] == 9 and stats["loss"] == 1 and stats["tie"] == 30
    assert stats["p_value"] == pytest.approx(22 / 1024)
    assert stats["verdict"] == "mitigation significantly improves correctness"
    markdown = result.to_markdown()
    assert "95% CI" in markdown and "exact binomial p" in markdown
    assert "significantly improves" in markdown
    assert "significance" in result.to_dict()


def test_benchmark_significance_inconclusive_when_all_ties() -> None:
    pairs = [("wrong", "wrong")] * 5
    records = [{"prompt": f"Q{i}?", "reference": "right"} for i in range(5)]
    result = evaluate_records(ScriptedMitigator(pairs), records, metric="contains")
    assert result.significance["decisive"] == 0
    assert result.significance["p_value"] is None
