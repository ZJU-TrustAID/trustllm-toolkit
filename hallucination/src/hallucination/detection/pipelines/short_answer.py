from __future__ import annotations

from collections.abc import Sequence

from hallucination.detection.backends.llm import LLMBackend
from hallucination.detection.blackbox.exact_match import (
    ExactMatchConsistencyScorer,
)
from hallucination.detection.blackbox.sentence import BaseSentenceScorer
from hallucination.detection.core.base import BaseScorer
from hallucination.detection.core.results import (
    DetectionExample,
    DetectionResult,
    UnitResult,
)
from hallucination.detection.pipelines.ensemble import DetectionPipeline


def build_blackbox_short_answer_pipeline(
    *,
    include_sample_consistency: bool = True,
    threshold: float = 0.5,
) -> DetectionPipeline:
    """Build a response-level black-box preset.

    Exact-match consistency is the only response-level sampled consistency
    scorer in this helper. Response-level cosine and semantic entropy are
    separate scorers; sentence-level consistency belongs to the SelfCheck
    family in ``BlackBoxSamplingPipeline(sentence_scorers=...)``.
    """

    scorers: list[BaseScorer] = []
    if include_sample_consistency:
        scorers.append(ExactMatchConsistencyScorer())
    if not scorers:
        raise ValueError("Select at least one black-box scorer.")
    return DetectionPipeline(scorers, threshold=threshold)


class BlackBoxSamplingPipeline(DetectionPipeline):
    """Generate one response and one shared sample set for black-box scorers."""

    def __init__(
        self,
        backend: LLMBackend,
        *,
        scorers: Sequence[BaseScorer] | None = None,
        sentence_scorers: Sequence[BaseSentenceScorer] | None = None,
        num_samples: int = 5,
        threshold: float = 0.5,
    ) -> None:
        super().__init__(scorers or [ExactMatchConsistencyScorer()], threshold=threshold)
        self.backend = backend
        self.sentence_scorers = list(sentence_scorers or [])
        self.num_samples = num_samples

    def detect(self, example: DetectionExample | None = None, **kwargs: object) -> DetectionResult:
        if example is None:
            example = DetectionExample(**kwargs)
        result = super().detect(example)
        if not self.sentence_scorers:
            return result

        sentence_signal_types = {scorer.signal_type for scorer in self.sentence_scorers}
        if len(sentence_signal_types) > 1 and not self.allow_mixed_signals:
            names = ", ".join(sorted(sentence_signal_types))
            raise ValueError(
                "Refusing to average sentence scorers with different signal types by default: "
                f"{names}. Aggregate them explicitly after validation."
            )
        existing_signal_types = set(result.signal_scores)
        if existing_signal_types and sentence_signal_types - existing_signal_types and not self.allow_mixed_signals:
            names = ", ".join(sorted(existing_signal_types | sentence_signal_types))
            raise ValueError(
                "Refusing to combine response and sentence scorers with different signal types by default: "
                f"{names}. Aggregate them explicitly after validation."
            )

        sentences = self._source_sentences(example)
        if not sentences:
            return result
        per_scorer = [scorer.score_sentences(example, sentences) for scorer in self.sentence_scorers]
        for scorer, scores in zip(self.sentence_scorers, per_scorer, strict=False):
            if len(scores) != len(sentences):
                raise ValueError(f"{scorer.name} returned {len(scores)} scores for {len(sentences)} source sentences")

        units: list[UnitResult] = []
        for index, sentence in enumerate(sentences):
            scores = [scores_for_scorer[index] for scores_for_scorer in per_scorer]
            score = sum(item.score for item in scores) / len(scores)
            risk = sum(item.risk for item in scores) / len(scores)
            label = "high_risk" if risk >= self.threshold else "low_risk"
            units.append(
                UnitResult(
                    text=sentence,
                    index=index,
                    unit_type="sentence",
                    score=score,
                    risk=risk,
                    label=label,
                    method_scores=scores,
                )
            )
            for item in scores:
                result.signal_scores.setdefault(item.signal_type, []).append(item)

        result.units = units
        summaries: dict[str, dict[str, object]] = {}
        for index, (scorer, score_list) in enumerate(zip(self.sentence_scorers, per_scorer, strict=False)):
            key = scorer.name if scorer.name not in summaries else f"{scorer.name}_{index}"
            risks = [item.risk for item in score_list]
            summaries[key] = {
                "method": scorer.name,
                "mean_risk": sum(risks) / len(risks),
                "max_risk": max(risks),
                "min_risk": min(risks),
                "count": len(risks),
            }
        result.metadata["sentence_summaries"] = summaries
        return result

    def detect_prompt(self, prompt: str, *, response: str | None = None, **generate_kwargs: object) -> DetectionResult:
        if response is None:
            response = self.backend.generate(prompt, **generate_kwargs)
        samples = self.backend.sample(prompt, self.num_samples, **generate_kwargs)
        example = DetectionExample(prompt=prompt, response=response, samples=samples)
        result = self.detect(example)
        result.metadata.update({"prompt": prompt, "response": response, "samples": samples, "num_samples": self.num_samples})
        return result

    @staticmethod
    def _source_sentences(example: DetectionExample) -> list[str]:
        if example.units is not None:
            if example.unit_type not in {None, "sentence"}:
                raise ValueError("BlackBoxSamplingPipeline sentence scorers require sentence units")
            return example.units
        from hallucination.detection.utils.text import split_sentences

        return split_sentences(example.response)
