from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from hallucination.mitigation.evaluation.metrics import contains_match, exact_match
from hallucination.mitigation.evaluation.stats import paired_sign_stats


@dataclass
class BenchmarkResult:
    method: str
    n: int
    baseline_score: float
    mitigated_score: float
    win: int = 0
    tie: int = 0
    loss: int = 0
    details: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    significance: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "hm/benchmark/v1",
            "method": self.method,
            "n": self.n,
            "baseline_score": self.baseline_score,
            "mitigated_score": self.mitigated_score,
            "delta": self.mitigated_score - self.baseline_score,
            "win": self.win,
            "tie": self.tie,
            "loss": self.loss,
            "significance": self.significance,
            "details": self.details,
            "warnings": self.warnings,
        }

    def to_json(self, path: str | Path | None = None) -> str:
        payload = json.dumps(self.to_dict(), ensure_ascii=False, indent=2)
        if path is not None:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            Path(path).write_text(payload + "\n", encoding="utf-8")
        return payload

    def to_markdown(self) -> str:
        lines = [
            f"# Benchmark: `{self.method}`",
            "",
            "| n | baseline | mitigated | delta | win/tie/loss |",
            "|---|---|---|---|---|",
            (
                f"| {self.n} | {self.baseline_score:.4f} | {self.mitigated_score:.4f} | "
                f"{self.mitigated_score - self.baseline_score:+.4f} | {self.win}/{self.tie}/{self.loss} |"
            ),
        ]
        if self.significance is not None:
            stats = self.significance
            if stats.get("decisive"):
                ci = stats["decisive_win_rate_ci95"]
                row = (
                    f"| {stats['decisive']} | {stats['decisive_win_rate']:.3f} | "
                    f"[{ci[0]:.3f}, {ci[1]:.3f}] | {stats['p_value']:.4f} | {stats['verdict']} |"
                )
            else:
                row = f"| 0 | n/a | n/a | n/a | {stats['verdict']} |"
            lines += [
                "",
                "| decisive | decisive win rate | 95% CI | exact binomial p | verdict |",
                "|---|---|---|---|---|",
                row,
            ]
        if self.warnings:
            lines += ["", "## Warnings", ""] + [
                f"- {warning}" for warning in self.warnings
            ]
        return "\n".join(lines) + "\n"


def _references(record: dict[str, Any]) -> list[str]:
    value = record.get("references", record.get("answers", record.get("reference")))
    if value is None:
        raise ValueError(
            "each record must contain 'reference', 'references', or 'answers'"
        )
    if isinstance(value, str):
        return [value]
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return value
    raise ValueError("reference(s) must be a string or a list of strings")


def evaluate_records(
    mitigator: Any,
    records: Iterable[dict[str, Any]],
    *,
    max_new_tokens: int = 128,
    limit: int | None = None,
    metric: str = "contains",
    baseline: bool = True,
) -> BenchmarkResult:
    if metric not in {"exact", "contains"}:
        raise ValueError("metric must be 'exact' or 'contains'")
    scorer = exact_match if metric == "exact" else contains_match
    rows = list(records)
    if limit is not None:
        rows = rows[:limit]
    if not rows:
        raise ValueError("no benchmark records")
    details: list[dict[str, Any]] = []
    baseline_total = 0.0
    mitigated_total = 0.0
    win = tie = loss = 0
    for index, record in enumerate(rows):
        try:
            prompt = str(record["prompt"])
        except KeyError as exc:
            raise ValueError(f"record {index} missing prompt") from exc
        context = record.get("context")
        references = _references(record)
        report = mitigator.mitigate(
            prompt,
            context=str(context) if context else None,
            max_new_tokens=max_new_tokens,
            baseline=baseline,
        )
        baseline_score = scorer(report.baseline or "", references)
        mitigated_score = scorer(report.response, references)
        baseline_total += baseline_score
        mitigated_total += mitigated_score
        if mitigated_score > baseline_score:
            win += 1
        elif mitigated_score == baseline_score:
            tie += 1
        else:
            loss += 1
        details.append(
            {
                "prompt": prompt,
                "context": context,
                "references": references,
                "baseline": report.baseline,
                "mitigated": report.response,
                "baseline_score": baseline_score,
                "mitigated_score": mitigated_score,
            }
        )
    n = len(rows)
    significance = paired_sign_stats(win, tie, loss)
    warnings = ["Low-exact-match tasks may need an LLM judge or human evaluation."]
    warnings.extend(significance.get("warnings", []))
    return BenchmarkResult(
        method=getattr(mitigator, "method", "unknown"),
        n=n,
        baseline_score=baseline_total / n,
        mitigated_score=mitigated_total / n,
        win=win,
        tie=tie,
        loss=loss,
        details=details,
        warnings=warnings,
        significance=significance,
    )
