import math

from hallucination.detection.core.results import DetectionExample
from hallucination.detection.pipelines.ensemble import DetectionPipeline
from hallucination.detection.whitebox.token_probability import LNTPScorer, MTPScorer


def test_detection_pipeline_aggregates_scores():
    pipeline = DetectionPipeline([MTPScorer(), LNTPScorer()], threshold=0.5)
    result = pipeline.detect(DetectionExample(response="x", token_logprobs=[math.log(0.25), math.log(1.0)]))
    assert len(result.method_scores) == 2
    assert result.risk > 0
