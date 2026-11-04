
import pytest

from hallucination.mitigation.evaluation.datasets import (
    _subsample,
    nq_record,
    nq_swap_record,
    truth_record,
)
from hallucination.mitigation.evaluation.stats import (
    holm_adjust,
    paired_sign_stats,
    two_sided_binomial_p,
    wilson_interval,
)


def test_two_sided_binomial_matches_known_values():
    # n=10, 9 successes: two-sided p = 2 * P(X >= 9) = 22/1024
    assert two_sided_binomial_p(9, 10) == pytest.approx(22 / 1024)
    # symmetric case: p = 1.0
    assert two_sided_binomial_p(5, 10) == pytest.approx(1.0)
    # extreme case: all successes
    assert two_sided_binomial_p(10, 10) == pytest.approx(2 / 1024)


def test_wilson_interval_covers_point_estimate():
    low, high = wilson_interval(8, 10)
    assert 0.0 <= low < 0.8 < high <= 1.0
    assert wilson_interval(0, 5)[0] == 0.0
    assert wilson_interval(5, 5)[1] == 1.0


def test_paired_sign_stats_reports_verdict_and_warnings():
    stats = paired_sign_stats(win=9, tie=40, loss=1)
    assert stats["decisive"] == 10
    assert stats["decisive_win_rate"] == pytest.approx(0.9)
    assert stats["p_value"] == pytest.approx(22 / 1024)
    assert stats["verdict"] == "mitigation significantly improves correctness"
    assert stats["decisive_win_rate_ci95"][0] > 0.5


def test_paired_sign_stats_inconclusive_without_decisive_pairs():
    stats = paired_sign_stats(win=0, tie=30, loss=0)
    assert stats["decisive"] == 0
    assert stats["p_value"] is None
    assert stats["verdict"] == "inconclusive: no decisive pairs"


def test_paired_sign_stats_flags_low_power():
    stats = paired_sign_stats(win=4, tie=50, loss=1)
    assert any("underpowered" in warning for warning in stats["warnings"])


def test_holm_adjust_controls_ordering():
    adjusted = holm_adjust([0.01, 0.04, 0.03])
    assert adjusted[0]["p_holm"] == pytest.approx(0.03)
    assert adjusted[1]["p_holm"] == pytest.approx(0.06)
    assert adjusted[2]["p_holm"] == pytest.approx(0.06)
    assert adjusted[0]["significant"] is True
    assert adjusted[1]["significant"] is False


def test_truth_record_collects_correct_answers():
    record = truth_record(
        {"question": "Who is a famous bear?", "best_answer": "Winnie", "correct_answers": ["Winnie", "Paddington"]},
        3,
    )
    assert record == {
        "id": "truthfulqa-3",
        "prompt": "Who is a famous bear?",
        "references": ["Winnie", "Paddington"],
    }


def test_nq_record_maps_answers():
    record = nq_record({"question": "who wrote hamlet", "answer": ["shakespeare", "Shakespeare"]}, 0)
    assert record["references"] == ["shakespeare", "Shakespeare"]


def test_nq_swap_record_prefers_pair_answer():
    record = nq_swap_record(
        {"question": "q", "context": "ctx", "pair_answer": ["swapped"], "answer": ["original"]}, 0
    )
    assert record["context"] == "ctx"
    assert record["references"] == ["swapped"]


def test_subsample_is_deterministic_and_respects_limit():
    rows = [{"i": index} for index in range(100)]
    first = _subsample(rows, 10, seed=7)
    second = _subsample(rows, 10, seed=7)
    assert first == second
    assert len(first) == 10
    assert len({row["i"] for row in first}) == 10
    assert _subsample(rows, None, seed=1) == rows


def test_unknown_dataset_name_rejected():
    from hallucination.mitigation.evaluation.datasets import load_dataset_records

    with pytest.raises(ValueError, match="unknown dataset"):
        load_dataset_records("trivia_qa")


def test_nq_swap_hub_record_maps_substitution_variant():
    from hallucination.mitigation.evaluation.datasets import nq_swap_hub_record

    record = nq_swap_hub_record(
        {
            "question": "q",
            "org_context": "original",
            "org_answer": ["23"],
            "sub_context": "swapped",
            "sub_answer": ["775"],
        },
        0,
    )
    assert record["context"] == "swapped"
    assert record["references"] == ["775"]
