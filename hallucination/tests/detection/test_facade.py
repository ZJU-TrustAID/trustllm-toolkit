import sys

import pytest

from hallucination.detection import DetectionReport, Detector
from hallucination.detection.calibration import CalibrationBundle


def test_offline_detector_runs_end_to_end_and_serializes():
    detector = Detector(preset="offline_toy")
    report = detector.detect("请介绍蓝鲸的食性。")
    assert isinstance(report, DetectionReport)
    assert 0.0 <= report.result.risk <= 1.0
    assert report.result.cost["generation_calls"] == 1 + len(report.samples[0])
    assert {score.method for score in report.result.method_scores} == {
        "response_cosine_consistency",
        "response_semantic_entropy",
    }
    payload = report.to_json()
    assert "results" in payload
    markdown = report.to_markdown()
    assert "response_cosine_consistency" in markdown
    assert "Sentence-level results" in markdown


def test_offline_detector_can_include_exact_match_and_ngram():
    detector = Detector(preset="offline_toy", include_exact_match=True, include_ngram=True)
    report = detector.detect("请介绍蓝鲸的食性。")
    methods = {score.method for score in report.result.method_scores}
    assert "exact_match_consistency" in methods
    assert "selfcheck_ngram" in report.result.metadata["sentence_summaries"]


def test_detector_detect_many_and_dataframe():
    pytest.importorskip("pandas")
    detector = Detector(preset="offline_toy")
    report = detector.detect_many(["请介绍蓝鲸的食性。", "海葵是植物吗？"])
    assert len(report.results) == 2
    dataframe = report.to_dataframe()
    assert len(dataframe) == 2
    assert "risk" in dataframe.columns


def test_detector_rejects_unknown_preset():
    with pytest.raises(ValueError):
        Detector(preset="nope")


def test_response_only_without_samples_raises_actionable_error():
    detector = Detector(preset="offline_toy")
    with pytest.raises(ValueError) as excinfo:
        detector.detect(response="蓝鲸以磷虾为食。")
    message = str(excinfo.value)
    assert "samples" in message
    assert "prompt" in message


def test_response_only_with_empty_samples_raises_the_same_error():
    detector = Detector(preset="offline_toy")
    with pytest.raises(ValueError, match="response-only detection requires samples"):
        detector.detect(response="蓝鲸以磷虾为食。", samples=[])


def test_response_only_accepts_explicit_samples():
    detector = Detector(preset="offline_toy")
    samples = ["蓝鲸以磷虾为食。", "蓝鲸吃磷虾。"]
    report = detector.detect(response="蓝鲸以磷虾为食。", samples=samples)
    assert isinstance(report, DetectionReport)
    assert 0.0 <= report.result.risk <= 1.0
    assert {score.method for score in report.result.method_scores} == {
        "response_cosine_consistency",
        "response_semantic_entropy",
    }
    assert report.samples == [samples]
    assert report.responses == ["蓝鲸以磷虾为食。"]
    assert report.result.cost == {"generation_calls": 0, "num_samples": 2}


def test_detector_caches_the_blackbox_pipeline():
    detector = Detector(preset="offline_toy")
    first = detector.as_pipeline()
    assert detector.as_pipeline() is first
    detector.detect("请介绍蓝鲸的食性。")
    assert detector.as_pipeline() is first


class _StubBackend:
    def generate(self, prompt, **kwargs):
        return "蓝鲸以磷虾为食。"

    def sample(self, prompt, num_samples, **kwargs):
        return ["蓝鲸以磷虾为食。"] * num_samples


def test_fallback_warnings_are_recorded_once(monkeypatch):
    monkeypatch.setitem(sys.modules, "hallucination.detection.backends.nli", None)
    monkeypatch.setitem(sys.modules, "hallucination.detection.backends.embeddings", None)
    detector = Detector(preset="openai_blackbox", backend=_StubBackend())
    assert detector.available_methods() == ["exact_match_consistency", "selfcheck_ngram"]
    baseline = list(detector.warnings)
    assert len(baseline) == 4

    for _ in range(3):
        report = detector.detect("蓝鲸吃什么？", samples=["蓝鲸以磷虾为食。", "蓝鲸吃磷虾。"])

    assert detector.warnings == baseline
    assert len(baseline) == len(set(baseline))
    assert report.metadata["warnings"] == baseline
    reported = {score.method for score in report.result.method_scores}
    reported |= {score.method for unit in report.result.units for score in unit.method_scores}
    assert set(detector.available_methods()) == reported


def test_available_methods_matches_the_scorers_that_run():
    detector = Detector(preset="offline_toy")
    report = detector.detect("请介绍蓝鲸的食性。")
    reported = {score.method for score in report.result.method_scores}
    reported |= {score.method for unit in report.result.units for score in unit.method_scores}
    assert set(detector.available_methods()) == reported


def test_detector_applies_calibration_to_the_matching_method():
    bundle = CalibrationBundle.fit(
        [0.1, 0.2, 0.7, 0.9],
        [0, 0, 1, 1],
        score_name="response_semantic_entropy",
        granularity="response",
        probability_method="platt",
    )
    detector = Detector(preset="offline_toy", calibration=bundle)
    report = detector.detect("请介绍蓝鲸的食性。")
    semantic = next(score for score in report.result.method_scores if score.method == "response_semantic_entropy")
    assert semantic.calibrated_risk is not None
    assert report.result.risk == semantic.calibrated_risk
    assert report.result.calibration["applied_to"] == ["response_semantic_entropy"]
    assert "response_cosine_consistency" in report.result.calibration["uncalibrated_methods"]


def test_detector_rejects_a_calibration_for_an_absent_method():
    bundle = CalibrationBundle.fit(
        [0.1, 0.2, 0.7, 0.9],
        [0, 0, 1, 1],
        score_name="p_true",
        granularity="response",
        probability_method="platt",
    )
    detector = Detector(preset="offline_toy", calibration=bundle)
    with pytest.raises(ValueError, match="does not match any detected method"):
        detector.detect("请介绍蓝鲸的食性。")
