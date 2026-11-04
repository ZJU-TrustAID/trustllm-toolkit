from __future__ import annotations

from hallucination.detection.backends.hf_whitebox import HuggingFaceWhiteBoxBackend
from hallucination.detection.core.results import DetectionResult
from hallucination.detection.pipelines.ensemble import DetectionPipeline
from hallucination.detection.whitebox.entropy import TokenEntropyScorer
from hallucination.detection.whitebox.token_probability import LNTPScorer, MTPScorer
from hallucination.detection.whitebox.tokens import TokenFilter


def build_whitebox_generation_pipeline(
    *,
    include_entropy: bool = True,
    threshold: float = 0.5,
    token_filter: TokenFilter | None = None,
) -> DetectionPipeline:
    selected_filter = token_filter if token_filter is not None else TokenFilter(skip_special=True)
    scorers = [MTPScorer(selected_filter), LNTPScorer(selected_filter)]
    if include_entropy:
        scorers.append(TokenEntropyScorer())
    return DetectionPipeline(scorers, threshold=threshold)


class WhiteBoxGenerationPipeline(DetectionPipeline):
    def __init__(
        self,
        backend: HuggingFaceWhiteBoxBackend,
        *,
        include_entropy: bool = True,
        threshold: float = 0.5,
        token_filter: TokenFilter | None = None,
    ) -> None:
        selected_filter = token_filter if token_filter is not None else TokenFilter(skip_special=True)
        scorers = [MTPScorer(selected_filter), LNTPScorer(selected_filter)]
        if include_entropy:
            scorers.append(TokenEntropyScorer())
        super().__init__(scorers, threshold=threshold)
        self.backend = backend
        self.token_filter = selected_filter

    def detect_prompt(self, prompt: str, **generate_kwargs: object) -> DetectionResult:
        example = self.backend.generate_detection_example(prompt, **generate_kwargs)
        result = self.detect(example)
        result.metadata.update(example.metadata)
        result.metadata["prompt"] = prompt
        result.metadata["response"] = example.response
        return result
