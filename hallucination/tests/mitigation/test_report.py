from __future__ import annotations

import json

from hallucination.mitigation.api.report import MitigationReport


def test_report_json_roundtrip(tmp_path) -> None:
    report = MitigationReport(
        method="cove",
        prompt="Q?",
        baseline="A.",
        response="B.",
        trace={"steps": 2},
        warnings=["w"],
        cost={"latency_seconds": 0.1},
        provenance={"provider": "callable"},
    )
    path = tmp_path / "report.json"
    payload = report.to_json(path)
    assert path.exists()
    data = json.loads(payload)
    assert data["schema_version"] == "hm/report/v1"
    assert data["response"] == "B."


def test_report_markdown_contains_baseline_and_response() -> None:
    report = MitigationReport(method="dola", prompt="Q?", baseline="A.", response="B.")
    text = report.to_markdown()
    assert "## Baseline" in text
    assert "## Mitigated" in text
    assert "A." in text and "B." in text
