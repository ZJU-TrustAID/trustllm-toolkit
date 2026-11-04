import pytest

from hallucination.detection.backends.api import CallableLLMBackend
from hallucination.detection.pipelines.short_answer import (
    BlackBoxSamplingPipeline,
    build_blackbox_short_answer_pipeline,
)


def test_sampling_pipeline_defaults_to_exact_match_consistency():
    pipeline = BlackBoxSamplingPipeline(CallableLLMBackend(lambda prompt, **kwargs: "answer"), num_samples=2)
    result = pipeline.detect_prompt("question")
    assert result.score == 1.0
    assert list(result.signal_scores) == ["self_consistency"]


def test_short_answer_pipeline_requires_at_least_one_scorer():
    with pytest.raises(ValueError, match="at least one"):
        build_blackbox_short_answer_pipeline(include_sample_consistency=False)
