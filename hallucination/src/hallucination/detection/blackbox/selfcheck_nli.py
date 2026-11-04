from __future__ import annotations

from statistics import mean

from hallucination.detection.backends.nli import NLIBackend
from hallucination.detection.blackbox.sentence import BaseSentenceScorer
from hallucination.detection.core.results import DetectionExample, ScoreResult
from hallucination.detection.core.thresholds import clamp01


class SelfCheckNLIScorer(BaseSentenceScorer):
    """SelfCheck-style sentence risk from NLI contradiction against sampled passages.

    The official SelfCheckGPT implementation feeds ``(sentence, sample)`` to the
    NLI model, i.e. the generated sentence is the premise and the sampled passage
    is the hypothesis. ``premise="sentence"`` reproduces that order;
    ``premise="sample"`` keeps the earlier behaviour of this toolkit.
    """

    name = "selfcheck_nli"

    def __init__(self, nli_backend: NLIBackend, *, premise: str = "sentence", aggregation: str = "mean") -> None:
        if premise not in {"sentence", "sample"}:
            raise ValueError("premise must be 'sentence' or 'sample'")
        if aggregation not in {"mean", "max"}:
            raise ValueError("aggregation must be 'mean' or 'max'")
        self.nli_backend = nli_backend
        self.premise = premise
        self.aggregation = aggregation

    def score_sentences(self, example: DetectionExample, sentences: list[str]) -> list[ScoreResult]:
        if not example.samples:
            raise ValueError("selfcheck_nli requires samples")
        results: list[ScoreResult] = []
        for sentence in sentences:
            comparisons = []
            for sample_index, sample in enumerate(example.samples):
                if self.premise == "sentence":
                    relation = self.nli_backend(sentence, sample)
                else:
                    relation = self.nli_backend(sample, sentence)
                comparisons.append(
                    {
                        "sample_index": sample_index,
                        "premise_role": self.premise,
                        "entailment": relation.entailment,
                        "contradiction": relation.contradiction,
                        "neutral": relation.neutral,
                    }
                )

            contradictions = [float(item["contradiction"]) for item in comparisons]
            risk = clamp01(max(contradictions) if self.aggregation == "max" else mean(contradictions))
            results.append(
                ScoreResult(
                    method=self.name,
                    signal_type=self.signal_type,
                    score=clamp01(1.0 - risk),
                    risk=risk,
                    level="sentence",
                    raw={
                        "source_sentence": sentence,
                        "comparisons": comparisons,
                        "aggregation": self.aggregation,
                        "premise": self.premise,
                        "mean_contradiction": mean(contradictions),
                        "max_contradiction": max(contradictions),
                    },
                    assumptions=[
                        "The sampled passages are treated as model-internal consistency evidence.",
                        f"NLI input order is ({self.premise} -> {'sample' if self.premise == 'sentence' else 'sentence'}).",
                    ],
                    warnings=[
                        "A consistent model can repeat a false statement across samples.",
                        "NLI probabilities are model outputs, not factual truth values.",
                    ],
                    provenance={
                        "nli_backend": getattr(self.nli_backend, "model_name", type(self.nli_backend).__name__),
                        "premise": self.premise,
                        "aggregation": self.aggregation,
                    },
                )
            )
        return results
