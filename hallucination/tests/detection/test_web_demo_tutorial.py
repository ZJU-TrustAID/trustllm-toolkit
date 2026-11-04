"""Offline tests for the tutorial endpoints of the web demo.

Every theme×method combination the UI can produce is executed through
``server.handle_method`` — the same handler the browser hits — and the
payloads are checked for completeness, JSON-serializability and determinism.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

WEB_DEMO = Path(__file__).resolve().parents[2] / "web_demo"
if str(WEB_DEMO) not in sys.path:
    sys.path.insert(0, str(WEB_DEMO))

import catalog as tutorial_catalog  # noqa: E402
from server import handle_method  # noqa: E402

DETECTION_KEYS = [m["key"] for m in tutorial_catalog.DETECTION_WHITEBOX + tutorial_catalog.DETECTION_BLACKBOX]
MITIGATION_KEYS = [m["key"] for m in tutorial_catalog.MITIGATION]

ALL_CASES: list[tuple[str, str]] = [
    (key, "hallucinated") for key in DETECTION_KEYS
] + [
    (key, "faithful") for key in DETECTION_KEYS
] + [
    (key, "hallucinated") for key in ("dola", "iti", "caa", "cove")
] + [
    ("cad", "conflict"),
]

_THEME_IDS = {theme["id"] for theme in tutorial_catalog.THEMES}


def test_catalog_lists_groups_methods_and_themes() -> None:
    payload = tutorial_catalog.catalog_payload()
    assert len(payload["detection_whitebox"]) == 5
    assert len(payload["detection_blackbox"]) == 9
    groups = {g["title"]: [m["key"] for m in g["methods"]] for g in payload["detection_blackbox_groups"]}
    assert list(groups) == ["基于外部知识库验证", "基于模型自评估检测", "基于一致性检测"]
    assert groups["基于外部知识库验证"] == ["factscore"]
    assert groups["基于模型自评估检测"] == ["p_true", "cove_detector"]
    assert len(groups["基于一致性检测"]) == 6
    assert [m["key"] for m in payload["mitigation"]] == ["dola", "iti", "cad", "caa", "cove"]
    keys = DETECTION_KEYS + MITIGATION_KEYS
    assert len(keys) == len(set(keys)) == 19
    assert {"llm_judge", "llm_panel"}.isdisjoint(keys)
    assert {"hallucinated", "faithful", "conflict"} == _THEME_IDS
    # Theme availability must only reference real methods.
    for theme in tutorial_catalog.THEMES:
        assert set(theme["available"]["detection"]) <= set(DETECTION_KEYS), theme["id"]
        assert set(theme["available"]["mitigation"]) <= set(MITIGATION_KEYS), theme["id"]


def test_theme_availability_matches_method_scenarios() -> None:
    for method_key in MITIGATION_KEYS:
        method_scenarios = set(tutorial_catalog.METHODS[method_key]["scenarios"])
        offered = {
            theme["id"]
            for theme in tutorial_catalog.THEMES
            if method_key in theme["available"]["mitigation"]
        }
        assert offered == method_scenarios, method_key
    for method_key in DETECTION_KEYS:
        method_scenarios = set(tutorial_catalog.METHODS[method_key]["scenarios"])
        offered = {
            theme["id"]
            for theme in tutorial_catalog.THEMES
            if method_key in theme["available"]["detection"]
        }
        assert offered == method_scenarios, method_key


def test_every_method_scenario_returns_complete_payload() -> None:
    assert len(ALL_CASES) == 33
    for method_key, scenario_id in ALL_CASES:
        payload = handle_method({"method": method_key, "scenario": scenario_id})
        assert payload["method"] == method_key, method_key
        assert payload["scenario"] == scenario_id, method_key
        assert payload["prompt"], method_key
        # JSON-serializable round trip (what the HTTP layer does).
        encoded = json.dumps(payload, ensure_ascii=False)
        assert "NaN" not in encoded and "Infinity" not in encoded, method_key
        # Comparison: detection returns the selected scenario's conclusion
        # card only; mitigation keeps its baseline/mitigated pair.
        cards = payload["compare"]["cards"]
        if method_key in {"dola", "iti", "cad", "caa", "cove"}:
            assert len(cards) == 2, method_key
        else:
            assert len(cards) == 1 and cards[0]["active"], method_key
        assert payload["compare"]["threshold_note"], method_key
        assert payload["viz"], method_key
        assert payload["warnings"], method_key


def test_payload_is_deterministic() -> None:
    for method_key, scenario_id in ALL_CASES:
        first = handle_method({"method": method_key, "scenario": scenario_id})
        second = handle_method({"method": method_key, "scenario": scenario_id})
        assert first == second, method_key


@pytest.mark.parametrize(
    "method_key, scenario_id",
    ALL_CASES,
)
def test_single_case_contract(method_key: str, scenario_id: str) -> None:
    payload = handle_method({"method": method_key, "scenario": scenario_id})
    card = next(card for card in payload["compare"]["cards"] if card["active"])
    if method_key in {"dola", "iti", "cad", "caa", "cove"}:
        assert card["text"], method_key
        assert payload["compare"]["cards"][0]["text"], method_key
    else:
        assert payload["method_scores"], method_key
        assert card.get("formula"), method_key
        first = payload["method_scores"][0]
        assert set(first) >= {"method", "score", "risk", "level", "raw"}, method_key
        assert 0.0 <= first["risk"] <= 1.0, method_key


def test_detection_risk_separates_scenarios() -> None:
    """The hallucinated theme must score riskier than the faithful one."""

    for method_key in DETECTION_KEYS:
        hallucinated = handle_method({"method": method_key, "scenario": "hallucinated"})
        faithful = handle_method({"method": method_key, "scenario": "faithful"})
        assert hallucinated["risk"] > faithful["risk"], method_key


def test_mitigation_compare_flips_baseline() -> None:
    dola = handle_method({"method": "dola", "scenario": "hallucinated"})
    baseline, mitigated = dola["compare"]["cards"]
    assert "海藻" in baseline["text"] and "磷虾" in mitigated["text"]

    cad = handle_method({"method": "cad", "scenario": "conflict"})
    assert "2" in cad["compare"]["cards"][0]["text"]
    assert "3" in cad["compare"]["cards"][1]["text"]



def test_dola_replays_real_math() -> None:
    """The trace must match a hand computation with the toolkit functions."""

    import numpy as np

    from hallucination.mitigation.core.sampling import softmax
    from hallucination.mitigation.methods.dola import (
        apc_mask,
        contrast_logits,
        select_premature_layer,
    )

    payload = handle_method({"method": "dola", "scenario": "hallucinated"})
    step = payload["viz"][1]
    assert step["type"] == "logit-table"

    mature = np.array([0.9, 1.0, 0.2, -1.0, -0.5, -2.0])
    candidates = {
        10: np.array([0.2, 1.4, 0.3, -0.8, -0.4, -1.5]),
        14: np.array([0.6, 1.2, 0.25, -0.9, -0.45, -1.7]),
        18: np.array([0.85, 1.05, 0.22, -0.95, -0.48, -1.9]),
    }
    layer, jsd_scores = select_premature_layer(mature, candidates)
    contrast = contrast_logits(mature, candidates[layer], relative_top=0.1)
    probs = softmax(contrast)
    assert int(np.argmax(probs)) == 0  # 磷虾 wins the contrast.
    assert apc_mask(mature, 0.1)[5] == -np.inf  # "。" pruned by the APC mask.
    table_rows = step["rows"]
    assert table_rows[0][4] == "对比分布选中"
    assert "剪除" in table_rows[5][4]
    assert layer == 10 and jsd_scores[layer] > jsd_scores[14]


def test_cove_runs_real_four_stage_chain() -> None:
    payload = handle_method({"method": "cove", "scenario": "hallucinated"})
    trace = payload["metadata"]["trace"]
    assert trace["baseline"].startswith("蓝鲸主要以海藻为食")
    assert len(trace["questions"]) == 2
    assert [item["question"] for item in trace["answers"]] == trace["questions"]
    assert "磷虾" in payload["compare"]["cards"][1]["text"]
    stages = [stage["stage"] for stage in payload["viz"][0]["stages"]]
    assert stages == ["baseline", "plan", "verify", "revise"]
    for stage in trace["stage_costs"]:
        assert "seconds" not in stage  # Determinism: no wall-clock in the payload.


def test_unknown_method_and_scenario_rejected() -> None:
    with pytest.raises(ValueError, match="未知方法"):
        handle_method({"method": "nope", "scenario": "hallucinated"})
    with pytest.raises(ValueError, match="不支持场景"):
        handle_method({"method": "mtp", "scenario": "conflict"})
    with pytest.raises(ValueError, match="不支持场景"):
        handle_method({"method": "dola", "scenario": "faithful"})
    with pytest.raises(ValueError, match="请指定方法"):
        handle_method({"method": "", "scenario": "hallucinated"})
