import json

import pytest

from hallucination.detection.calibration import (
    CalibrationBundle,
    CalibrationRegistry,
    CalibrationReport,
    ProbabilityCalibrator,
    ThresholdCalibrator,
    calibration_metrics,
    expected_calibration_error,
    load_calibration,
)
from hallucination.detection.core.results import ScoreResult


def test_expected_calibration_error_known_value():
    scores = [0.9, 0.9, 0.1, 0.1]
    labels = [True, True, False, False]
    assert expected_calibration_error(scores, labels) == pytest.approx(0.1)
    metrics = calibration_metrics(scores, labels)
    assert metrics["brier"] == pytest.approx(0.01)
    assert metrics["count"] == 4


def test_probability_calibrators_fit_and_transform():
    scores = [0.2, 0.3, 0.4, 0.6, 0.7, 0.8]
    labels = [0, 0, 0, 1, 1, 1]
    for method in ("platt", "isotonic", "temperature"):
        calibrator = ProbabilityCalibrator(method).fit(scores, labels, split="validation")
        transformed = calibrator.transform(scores)
        assert len(transformed) == len(scores)
        assert all(0.0 <= value <= 1.0 for value in transformed)
        assert calibrator.split == "validation"


def test_isotonic_is_monotone():
    scores = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    labels = [0, 1, 0, 1, 1, 0]
    calibrator = ProbabilityCalibrator("isotonic").fit(scores, labels)
    transformed = calibrator.transform(scores)
    assert transformed == sorted(transformed)


def test_isotonic_handles_duplicate_scores():
    scores = [0.2, 0.2, 0.4, 0.4, 0.6, 0.8]
    labels = [0, 0, 1, 0, 1, 1]
    calibrator = ProbabilityCalibrator("isotonic").fit(scores, labels)
    assert len(calibrator._params["x"]) == len(calibrator._params["y"])
    transformed = calibrator.transform(scores)
    assert len(transformed) == len(scores)
    assert transformed == sorted(transformed)


def test_calibration_bundle_roundtrip_and_matching_score():
    bundle = CalibrationBundle.fit(
        [0.1, 0.2, 0.7, 0.9],
        [0, 0, 1, 1],
        score_name="response_semantic_entropy",
        granularity="response",
        probability_method="platt",
    )
    score = ScoreResult(
        method="response_semantic_entropy",
        score=0.2,
        risk=0.8,
        level="response",
    )
    assert bundle.apply_to_score(score) is True
    assert score.calibrated_risk is not None
    assert score.calibrated_score == pytest.approx(1.0 - score.calibrated_risk)
    assert score.calibration["score_name"] == "response_semantic_entropy"

    loaded = CalibrationBundle.from_dict(bundle.to_dict())
    assert loaded.calibrate_risk(0.8) == pytest.approx(bundle.calibrate_risk(0.8))


def test_calibration_bundle_does_not_match_another_method():
    bundle = CalibrationBundle.fit(
        [0.1, 0.2, 0.7, 0.9],
        [0, 0, 1, 1],
        score_name="response_semantic_entropy",
        granularity="response",
        probability_method="platt",
    )
    other = ScoreResult(method="response_cosine_consistency", score=0.2, risk=0.8)
    assert bundle.apply_to_score(other) is False
    assert other.calibrated_risk is None


def test_calibration_bundle_loads_legacy_payload_without_type():
    bundle = CalibrationBundle(
        score_name="response_semantic_entropy",
        threshold=ThresholdCalibrator().fit([0.2, 0.8], [0, 1]),
    )
    payload = bundle.to_dict()
    payload.pop("type")
    payload["threshold"].pop("type")
    restored = load_calibration(payload)
    assert isinstance(restored, CalibrationBundle)
    assert restored.threshold is not None
    assert restored.threshold.threshold == pytest.approx(bundle.threshold.threshold)


def test_calibration_registry_prefers_method_specific_bundle():
    generic = CalibrationBundle(score_name="*", threshold=ThresholdCalibrator().fit([0.2, 0.8], [0, 1]))
    specific = CalibrationBundle(
        score_name="response_semantic_entropy",
        granularity="response",
        threshold=ThresholdCalibrator().fit([0.2, 0.8], [0, 1]),
    )
    registry = CalibrationRegistry([generic, specific])
    assert registry.for_score("response_semantic_entropy", "response") is specific
    assert registry.for_score("response_cosine_consistency", "response") is generic


def test_calibrator_rejects_test_split_and_bad_labels():
    with pytest.raises(ValueError, match="test split"):
        ProbabilityCalibrator().fit([0.2, 0.8], [0, 1], split="test")
    with pytest.raises(ValueError, match="binary"):
        ProbabilityCalibrator().fit([0.2, 0.8], [0, 2])


def test_calibrator_save_load(tmp_path):
    scores = [0.2, 0.4, 0.6, 0.8]
    labels = [0, 0, 1, 1]
    calibrator = ProbabilityCalibrator("platt").fit(scores, labels)
    path = tmp_path / "calibrator.json"
    calibrator.save(path)
    loaded = ProbabilityCalibrator.load(path)
    assert loaded.transform(scores) == pytest.approx(calibrator.transform(scores))
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["method"] == "platt"
    assert payload["split"] == "validation"


def test_threshold_calibrator_selects_perfect_threshold():
    risks = [0.1, 0.4, 0.6, 0.9]
    labels = [0, 0, 1, 1]
    calibrator = ThresholdCalibrator().fit(risks, labels, split="validation")
    assert calibrator.threshold in {0.5, 0.6}
    assert calibrator.score == pytest.approx(1.0)
    assert calibrator.predict(risks) == [False, False, True, True]


def test_threshold_calibrator_rejects_test_split():
    with pytest.raises(ValueError, match="test split"):
        ThresholdCalibrator().fit([0.1, 0.9], [0, 1], split="test")


def test_calibration_report_compares_raw_and_calibrated():
    scores = [0.9, 0.8, 0.7, 0.3, 0.2, 0.1]
    labels = [1, 1, 1, 0, 0, 0]
    calibrated = ProbabilityCalibrator("isotonic").fit_transform(scores, labels)
    report = CalibrationReport(scores, labels, calibrated_scores=calibrated, method="isotonic")
    assert report.metrics["ece"] >= 0
    assert report.calibrated_metrics is not None
    assert "Brier" in report.to_markdown()
    assert "reliability" in report.to_dict()


def test_calibration_rejects_non_binary_and_out_of_range():
    with pytest.raises(ValueError):
        calibration_metrics([1.2, 0.2], [0, 1])
    with pytest.raises(ValueError):
        calibration_metrics([0.2, 0.8], [0, 0.5])
