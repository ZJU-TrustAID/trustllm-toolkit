from __future__ import annotations

from pathlib import Path
from typing import Any

from hallucination.detection.api.report import DetectionReport
from hallucination.detection.backends.api import OpenAIResponsesBackend
from hallucination.detection.backends.demo import (
    DemoLLMBackend,
    DemoNLIBackend,
    HashingEmbeddingBackend,
)
from hallucination.detection.backends.hf_whitebox import HuggingFaceWhiteBoxBackend
from hallucination.detection.blackbox.exact_match import ExactMatchConsistencyScorer
from hallucination.detection.blackbox.response_cosine import (
    ResponseCosineConsistencyScorer,
)
from hallucination.detection.blackbox.response_semantic_entropy import (
    ResponseSemanticEntropyScorer,
)
from hallucination.detection.blackbox.selfcheck_ngram import SelfCheckNgramScorer
from hallucination.detection.blackbox.selfcheck_nli import SelfCheckNLIScorer
from hallucination.detection.calibration.bundle import (
    CalibrationArtifact,
    CalibrationBundle,
    CalibrationRegistry,
    load_calibration,
)
from hallucination.detection.core.results import DetectionExample, DetectionResult
from hallucination.detection.pipelines.short_answer import BlackBoxSamplingPipeline
from hallucination.detection.pipelines.whitebox_generation import (
    WhiteBoxGenerationPipeline,
)

_PRESETS = {"offline_toy", "openai_blackbox", "openai_whitebox", "hf_whitebox"}

_RESPONSE_ONLY_ERROR = (
    "response-only detection requires samples: the self-consistency scorers compare the response "
    "with sampled answers. Pass samples=[...] (or --samples as a JSON list on the CLI), or pass a "
    "prompt so the configured backend can generate the samples."
)


class Detector:
    """One-click detection facade.

    ``Detector(preset="offline_toy").detect("请介绍蓝鲸的食性。")`` runs entirely
    offline. Other presets require the corresponding optional dependency:

    - ``openai_blackbox``: ``.[openai]``; add ``.[nli,embeddings]`` for
      semantic entropy and cosine consistency.
    - ``openai_whitebox``: ``.[openai]``; MTP, LNTP, and token entropy over
      Chat Completions logprobs (top-k approximation; no hidden states).
    - ``hf_whitebox``: ``.[whitebox]`` and a local model path.

    The black-box pipeline is built once on first use and reused, so
    optional-backend fallback warnings are reported once per detector.
    """

    def __init__(
        self,
        preset: str = "offline_toy",
        *,
        backend: Any | None = None,
        nli: Any | None = None,
        embedding: Any | None = None,
        model: str | None = None,
        num_samples: int = 5,
        threshold: float = 0.5,
        include_sentence: bool = True,
        include_response: bool = True,
        include_exact_match: bool = False,
        include_ngram: bool = False,
        device: str | None = None,
        calibration: CalibrationArtifact | str | Path | dict[str, Any] | None = None,
        warnings: list[str] | None = None,
    ) -> None:
        if preset not in _PRESETS:
            raise ValueError(f"preset must be one of {sorted(_PRESETS)}")
        if num_samples < 1:
            raise ValueError("num_samples must be positive")
        self.preset = preset
        self.backend = backend
        self.nli = nli
        self.embedding = embedding
        self.model = model
        self.num_samples = num_samples
        self.threshold = threshold
        self.include_sentence = include_sentence
        self.include_response = include_response
        self.include_exact_match = include_exact_match
        self.include_ngram = include_ngram
        # None means auto-pick (cuda if available); the literal "auto" would
        # reach torch as an invalid device, so normalize it here as well.
        self.device = None if device in (None, "", "auto") else device
        self.calibration = load_calibration(calibration) if calibration is not None else None
        self.warnings: list[str] = list(dict.fromkeys(warnings or []))
        self._pipeline: Any | None = None
        self._blackbox: BlackBoxSamplingPipeline | None = None
        self._build()

    def _warn(self, message: str) -> None:
        """Record a run-level warning once per detector instance."""

        if message not in self.warnings:
            self.warnings.append(message)

    def _apply_calibration(
        self,
        result: DetectionResult,
        calibration: CalibrationArtifact | str | Path | dict[str, Any] | None = None,
    ) -> None:
        artifact = load_calibration(calibration) if calibration is not None else self.calibration
        if artifact is None:
            return
        if isinstance(artifact, (CalibrationBundle, CalibrationRegistry)):
            matched = artifact.apply_result(result, default_threshold=self.threshold)
        else:  # pragma: no cover - load_calibration already validates this
            raise TypeError(f"unsupported calibration artifact: {type(artifact)!r}")
        if not matched:
            methods = sorted({score.method for score in result.method_scores})
            methods.extend(sorted({score.method for unit in result.units for score in unit.method_scores}))
            raise ValueError(
                "the supplied calibration does not match any detected method; "
                f"available methods: {sorted(set(methods))}. Fit the calibrator on the same method "
                "and granularity, or use score_name='*' only when the score scale is shared."
            )
        result.metadata["calibration_applied"] = sorted(set(matched))

    def _build(self) -> None:
        if self.preset == "offline_toy":
            self.backend = self.backend or DemoLLMBackend()
            self.nli = self.nli or DemoNLIBackend()
            self.embedding = self.embedding or HashingEmbeddingBackend()
            return
        if self.preset == "openai_blackbox":
            if self.backend is None:
                self.backend = OpenAIResponsesBackend(self.model or "gpt-4o-mini")
            if self.nli is None:
                try:
                    from hallucination.detection.backends.nli import (
                        BidirectionalNLIBackend,
                    )

                    self.nli = BidirectionalNLIBackend(device=self.device)
                except ImportError as exc:
                    self._warn(f"NLI backend unavailable; semantic entropy and SelfCheck-NLI are skipped: {exc}")
            if self.embedding is None:
                try:
                    from hallucination.detection.backends.embeddings import (
                        SentenceTransformerEmbeddingBackend,
                    )

                    self.embedding = SentenceTransformerEmbeddingBackend(device=self.device)
                except ImportError as exc:
                    self._warn(f"Embedding backend unavailable; response cosine is skipped: {exc}")
            return
        if self.preset == "hf_whitebox":
            if self.backend is None:
                if not self.model:
                    raise ValueError("hf_whitebox requires model='path-or-model-id'")
                self.backend = HuggingFaceWhiteBoxBackend.from_pretrained(self.model, device=self.device)
            self._pipeline = WhiteBoxGenerationPipeline(self.backend, threshold=self.threshold)
        if self.preset == "openai_whitebox":
            if self.backend is None:
                from hallucination.detection.backends.api import OpenAIChatBackend

                self.backend = OpenAIChatBackend(self.model or "gpt-4o-mini")
            self._pipeline = WhiteBoxGenerationPipeline(self.backend, threshold=self.threshold)

    def detect(
        self,
        prompt: str | None = None,
        *,
        response: str | None = None,
        samples: list[str] | None = None,
        claims: list[str] | None = None,
        units: list[str] | None = None,
        calibration: CalibrationArtifact | str | Path | dict[str, Any] | None = None,
        **generate_kwargs: object,
    ) -> DetectionReport:
        """Score one response.

        Pass ``prompt`` to let the backend generate the response and its samples,
        or pass ``response`` together with ``samples`` to score answers you
        already have. A response without samples cannot be scored by the
        self-consistency scorers and raises :class:`ValueError`.

        ``calibration`` overrides the detector-level calibration for this call.
        """

        if self.preset in {"hf_whitebox", "openai_whitebox"}:
            return self._detect_whitebox(prompt=prompt, response=response, calibration=calibration, **generate_kwargs)

        if prompt is None and response is None:
            raise ValueError("provide prompt and/or response")
        if prompt is None and not samples:
            raise ValueError(_RESPONSE_ONLY_ERROR)
        response_supplied = response is not None
        samples_supplied = samples is not None
        if response is None:
            if self.backend is None:
                raise ValueError("response is required when no backend is configured")
            response = self.backend.generate(prompt or "", **generate_kwargs)
        if samples is None:
            if self.backend is None:
                raise ValueError("samples are required when no backend is configured")
            assert prompt is not None
            samples = self.backend.sample(prompt, self.num_samples, **generate_kwargs)
        generation_calls = (0 if response_supplied else 1) + (0 if samples_supplied else len(samples))

        example = DetectionExample(
            prompt=prompt,
            response=response,
            samples=samples,
            claims=claims,
            units=units,
        )
        result = self._blackbox_pipeline().detect(example)
        self._apply_calibration(result, calibration)
        result.metadata.update(
            {
                "preset": self.preset,
                "prompt": prompt,
                "response": response,
                "samples": samples,
                "num_samples": len(samples),
                "generation_calls": generation_calls,
            }
        )
        result.cost = {"generation_calls": generation_calls, "num_samples": len(samples)}
        result.granularity = "sentence" if result.units else "response"
        return DetectionReport.from_result(
            result,
            prompt=prompt,
            response=response,
            samples=samples,
            metadata={"preset": self.preset, "warnings": list(self.warnings)},
        )

    def detect_many(self, prompts: list[str], **kwargs: object) -> DetectionReport:
        if not prompts:
            raise ValueError("prompts must not be empty")
        reports = [self.detect(prompt, **kwargs) for prompt in prompts]
        return DetectionReport(
            results=[report.result for report in reports],
            prompts=[report.prompts[0] for report in reports],
            responses=[report.responses[0] for report in reports],
            samples=[report.samples[0] for report in reports],
            metadata={"preset": self.preset, "warnings": list(self.warnings)},
        )

    def as_pipeline(self) -> Any:
        if self.preset == "hf_whitebox":
            assert self._pipeline is not None
            return self._pipeline
        return self._blackbox_pipeline()

    def available_methods(self) -> list[str]:
        """Names of the scorers actually wired into this preset.

        The black-box list is read from the built pipeline, so fallbacks added at
        build time (for example SelfCheck n-gram when no NLI backend is
        installed) are reported here too.
        """

        if self.preset == "hf_whitebox":
            return ["mtp", "lntp", "token_entropy"]
        pipeline = self._blackbox_pipeline()
        names = [scorer.name for scorer in pipeline.scorers]
        names.extend(scorer.name for scorer in pipeline.sentence_scorers)
        return list(dict.fromkeys(names))

    def _blackbox_pipeline(self) -> BlackBoxSamplingPipeline:
        if self._blackbox is not None:
            return self._blackbox
        scorers = []
        if self.include_exact_match:
            scorers.append(ExactMatchConsistencyScorer())
        if self.include_response and self.embedding is not None:
            scorers.append(ResponseCosineConsistencyScorer(self.embedding))
        if self.include_response and self.nli is not None:
            scorers.append(ResponseSemanticEntropyScorer(self.nli))
        if not scorers:
            self._warn("No response-level scorer available; falling back to exact-match consistency.")
            scorers.append(ExactMatchConsistencyScorer())
        sentence_scorers = []
        if self.include_ngram:
            sentence_scorers.append(SelfCheckNgramScorer())
        if self.include_sentence and self.nli is not None:
            sentence_scorers.append(SelfCheckNLIScorer(self.nli))
        if not sentence_scorers and self.include_sentence:
            self._warn("No NLI backend available; falling back to SelfCheck n-gram.")
            sentence_scorers.append(SelfCheckNgramScorer())
        self._blackbox = BlackBoxSamplingPipeline(
            self.backend,
            scorers=scorers,
            sentence_scorers=sentence_scorers,
            num_samples=self.num_samples,
            threshold=self.threshold,
        )
        return self._blackbox

    def _detect_whitebox(
        self,
        *,
        prompt: str | None,
        response: str | None,
        calibration: CalibrationArtifact | str | Path | dict[str, Any] | None = None,
        **generate_kwargs: object,
    ) -> DetectionReport:
        if response is not None and prompt is None:
            raise ValueError("hf_whitebox scores generated tokens; provide a prompt instead of a raw response")
        if prompt is None:
            raise ValueError("hf_whitebox requires a prompt")
        assert self.backend is not None and self._pipeline is not None
        result = self._pipeline.detect_prompt(prompt, **generate_kwargs)
        self._apply_calibration(result, calibration)
        result.metadata.update({"preset": self.preset, "generation_calls": 1})
        result.cost = {"generation_calls": 1}
        result.granularity = "token"
        return DetectionReport.from_result(
            result,
            prompt=prompt,
            response=result.metadata.get("response"),
            samples=[],
            metadata={"preset": self.preset, "warnings": list(self.warnings)},
        )


def detect(
    prompt: str | None = None,
    *,
    preset: str = "offline_toy",
    response: str | None = None,
    **kwargs: object,
) -> DetectionReport:
    """Convenience wrapper around :class:`Detector`."""

    return Detector(preset=preset).detect(prompt, response=response, **kwargs)
