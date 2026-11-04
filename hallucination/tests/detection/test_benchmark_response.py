import pytest

from hallucination.detection import Detector
from hallucination.detection.calibration import CalibrationBundle, ProbabilityCalibrator
from hallucination.detection.datasets.schema import DatasetRecord
from hallucination.detection.evaluation import BenchmarkRunner


def _records():
    return [
        DatasetRecord(
            id="correct",
            response="蓝鲸主要以磷虾为食。",
            samples=["蓝鲸主要吃磷虾。", "蓝鲸以磷虾为主食。", "蓝鲸主要以磷虾为食。"],
            response_label=False,
        ),
        DatasetRecord(
            id="hallucinated",
            response="蓝鲸主要以海藻为食。",
            samples=["蓝鲸主要吃磷虾。", "蓝鲸以磷虾为主食。", "蓝鲸主要以磷虾为食。"],
            response_label=True,
        ),
    ]


def test_response_level_benchmark_reports_calibration_metrics():
    detector = Detector(preset="offline_toy")
    report = BenchmarkRunner(detector.as_pipeline(), granularity="response").evaluate(_records())
    assert "response_cosine_consistency" in report.metrics
    metrics = report.metrics["response_cosine_consistency"]
    assert metrics["count"] == 2.0
    assert "ece" in metrics
    assert "brier" in metrics
    assert report.metadata["granularity"] == "response"


def test_response_level_benchmark_with_prefit_calibrator():
    calibrator = ProbabilityCalibrator("isotonic").fit(
        [0.1, 0.2, 0.7, 0.9], [False, False, True, True], split="validation"
    )
    detector = Detector(preset="offline_toy")
    report = BenchmarkRunner(
        detector.as_pipeline(), granularity="response", calibration=calibrator
    ).evaluate(_records())
    metrics = report.metrics["response_cosine_consistency"]
    assert "calibrated_ece" in metrics
    assert report.metadata["calibration"]["fit_split"] == "validation"


def test_benchmark_rejects_calibrator_fitted_on_test():
    calibrator = ProbabilityCalibrator("isotonic")
    calibrator._fitted = True
    calibrator.split = "test"
    with pytest.raises(ValueError, match="test split"):
        BenchmarkRunner(Detector(preset="offline_toy").as_pipeline(), calibration=calibrator)


def test_benchmark_calibrates_only_the_registered_method():
    bundle = CalibrationBundle.fit(
        [0.1, 0.2, 0.7, 0.9],
        [0, 0, 1, 1],
        score_name="response_cosine_consistency",
        granularity="response",
        probability_method="platt",
    )
    report = BenchmarkRunner(
        Detector(preset="offline_toy").as_pipeline(),
        calibration=bundle,
    ).evaluate(_records(), granularity="response")
    assert "calibrated_ece" in report.metrics["response_cosine_consistency"]
    assert "calibrated_ece" not in report.metrics["response_semantic_entropy"]
    assert report.metadata["calibration"]["type"] == "per_method"
