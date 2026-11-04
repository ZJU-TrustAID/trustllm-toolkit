import math

import pytest

from hallucination.detection.blackbox.exact_match import (
    ExactMatchConsistencyScorer,
)
from hallucination.detection.core.results import (
    DetectionExample,
    UnitResult,
)
from hallucination.detection.pipelines.ensemble import DetectionPipeline
from hallucination.detection.whitebox.token_probability import MTPScorer


def test_pipeline_groups_single_signal_type():
    result = DetectionPipeline([MTPScorer()]).detect(DetectionExample(response="x", token_logprobs=[math.log(0.5)]))
    assert list(result.signal_scores) == ["generation_confidence"]


def test_pipeline_rejects_default_mixed_signal_aggregation():
    pipeline = DetectionPipeline([MTPScorer(), ExactMatchConsistencyScorer()])
    with pytest.raises(ValueError, match="different signal types"):
        pipeline.detect(DetectionExample(response="x", token_logprobs=[math.log(0.5)], samples=["x", "y"]))


def test_unit_result_clamps_score_and_risk():
    result = UnitResult(text="x", index=0, unit_type="sentence", score=2, risk=-1)
    assert result.score == 1.0
    assert result.risk == 0.0
