from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from types import TracebackType
from typing import Any

from typing_extensions import Self

from hallucination._version import __version__
from hallucination.mitigation.core.contracts import (
    ComparisonReport,
    GenerationRequest,
    MitigationReport,
)
from hallucination.mitigation.core.errors import (
    ConfigError,
)
from hallucination.mitigation.core.model_config import ModelConfig
from hallucination.mitigation.core.presets import load_preset

_METHODS = {"dola", "caa", "iti", "cad", "cove"}


class Mitigator:
    """One-click facade for inference-time hallucination mitigation."""

    def __init__(
        self,
        preset: str | None = None,
        *,
        method: str | None = None,
        model: str | None = None,
        backend: str | None = None,
        device: str | None = None,
        dtype: str | None = None,
        revision: str | None = None,
        local_files_only: bool | None = None,
        trust_remote_code: bool | None = None,
        load_in_4bit: bool | None = None,
        api_base: str | None = None,
        api_key_env: str | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
        steering: str | Path | None = None,
        probe: str | Path | None = None,
        layer: int | None = None,
        multiplier: float | None = None,
        alpha: float | None = None,
        context_template: str | None = None,
        seed: int = 0,
        backend_obj: Any | None = None,
        **method_kwargs: Any,
    ) -> None:
        preset_data: dict[str, Any] = {}
        if preset:
            preset_data = load_preset(preset)
        self.preset = preset
        self.method = (method or preset_data.get("method") or "dola").lower()
        if self.method not in _METHODS:
            raise ConfigError(f"method must be one of {sorted(_METHODS)}")
        self.model = model or preset_data.get("model")
        if not self.model:
            raise ConfigError("model is required (or choose a preset that defines one)")
        self.backend = (backend or preset_data.get("backend") or "hf").lower()
        self.device = device if device is not None else preset_data.get("device")
        self.dtype = dtype if dtype is not None else preset_data.get("dtype", "auto")
        self.revision = (
            revision if revision is not None else preset_data.get("revision")
        )
        self.local_files_only = (
            local_files_only
            if local_files_only is not None
            else bool(preset_data.get("local_files_only", False))
        )
        self.trust_remote_code = (
            trust_remote_code
            if trust_remote_code is not None
            else bool(preset_data.get("trust_remote_code", False))
        )
        self.load_in_4bit = (
            load_in_4bit
            if load_in_4bit is not None
            else bool(preset_data.get("load_in_4bit", False))
        )
        self.api_base = (
            api_base if api_base is not None else preset_data.get("api_base")
        )
        self.api_key_env = (
            api_key_env or preset_data.get("api_key_env") or "OPENAI_API_KEY"
        )
        self.timeout = float(
            timeout if timeout is not None else preset_data.get("timeout", 60.0)
        )
        self.max_retries = int(
            max_retries
            if max_retries is not None
            else preset_data.get("max_retries", 2)
        )
        method_args = dict(preset_data.get("method_args") or {})
        method_args.update(method_kwargs)
        if layer is not None:
            method_args["layer"] = layer
        if multiplier is not None:
            method_args["multiplier"] = multiplier
        if alpha is not None:
            method_args["alpha"] = alpha
        if context_template is not None:
            method_args["context_template"] = context_template
        self.method_args = method_args
        self.seed = int(seed)
        self.steering = steering
        self.probe = probe
        self.backend_obj = backend_obj
        generation = dict(preset_data.get("generation") or {})
        self.generation = {
            "max_new_tokens": int(generation.get("max_new_tokens", 128)),
            "do_sample": bool(generation.get("do_sample", False)),
            "temperature": float(generation.get("temperature", 1.0)),
            "top_p": float(generation.get("top_p", 1.0)),
            "top_k": int(generation.get("top_k", 0)),
            "repetition_penalty": float(generation.get("repetition_penalty", 1.0)),
        }
        self._engine: Any | None = None
        self._llm_backend: Any | None = None

    @property
    def model_config(self) -> ModelConfig:
        return ModelConfig(
            model=str(self.model),
            backend=self.backend,
            revision=self.revision,
            device=self.device,
            dtype=self.dtype or "auto",
            local_files_only=bool(self.local_files_only),
            trust_remote_code=bool(self.trust_remote_code),
            load_in_4bit=bool(self.load_in_4bit),
            api_base=self.api_base,
            api_key_env=self.api_key_env,
            timeout=self.timeout,
            max_retries=self.max_retries,
        )

    def available_methods(self) -> list[str]:
        methods = ["cove"]
        try:
            import importlib.util

            if importlib.util.find_spec("torch") and importlib.util.find_spec(
                "transformers"
            ):
                methods = ["dola", "cad", "caa", "cove"]
                if importlib.util.find_spec("sklearn"):
                    methods.append("iti")
        except (ImportError, ValueError):
            pass
        return methods

    def _ensure_engine(self) -> Any:
        if self._engine is None:
            from hallucination.mitigation.engine.hf import HFEngine

            self._engine = HFEngine.from_pretrained(self.model_config)
        return self._engine

    def _ensure_llm_backend(self) -> Any:
        if self._llm_backend is not None:
            return self._llm_backend
        if self.backend == "hf":
            from hallucination.mitigation.backends.hf import HFBackend

            self._llm_backend = HFBackend(self._ensure_engine(), seed=self.seed)
        elif self.backend in {"openai", "openai-compatible"}:
            from hallucination.mitigation.backends.openai import OpenAIBackend

            self._llm_backend = OpenAIBackend(
                str(self.model),
                api_key=self.model_config.resolve_api_key(),
                base_url=self.api_base,
                timeout=self.timeout,
                max_retries=self.max_retries,
            )
        elif self.backend == "callable":
            if self.backend_obj is None:
                raise ConfigError("backend='callable' requires backend_obj")
            self._llm_backend = self.backend_obj
        else:
            raise ConfigError(f"unsupported backend: {self.backend}")
        return self._llm_backend

    def _request(
        self,
        prompt: str,
        context: str | None,
        *,
        max_new_tokens: int | None,
        do_sample: bool | None,
        temperature: float | None,
        top_p: float | None,
        top_k: int | None,
        repetition_penalty: float | None,
        seed: int | None,
    ) -> GenerationRequest:
        return GenerationRequest(
            prompt=prompt,
            context=context,
            max_new_tokens=int(
                max_new_tokens
                if max_new_tokens is not None
                else self.generation["max_new_tokens"]
            ),
            do_sample=bool(
                do_sample if do_sample is not None else self.generation["do_sample"]
            ),
            temperature=float(
                temperature
                if temperature is not None
                else self.generation["temperature"]
            ),
            top_p=float(top_p if top_p is not None else self.generation["top_p"]),
            top_k=int(top_k if top_k is not None else self.generation["top_k"]),
            repetition_penalty=float(
                repetition_penalty
                if repetition_penalty is not None
                else self.generation["repetition_penalty"]
            ),
            seed=int(seed if seed is not None else self.seed),
        )

    def _common_provenance(
        self, method: str, request: GenerationRequest
    ) -> dict[str, Any]:
        return {
            "package_version": __version__,
            "method": method,
            "preset": self.preset,
            "model_config": self.model_config.fingerprint(),
            "seed": request.seed,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    def mitigate(
        self,
        prompt: str,
        context: str | None = None,
        *,
        max_new_tokens: int | None = None,
        do_sample: bool | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        repetition_penalty: float | None = None,
        seed: int | None = None,
        baseline: bool = True,
    ) -> MitigationReport:
        request = self._request(
            prompt,
            context,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            seed=seed,
        )
        self.model_config.validate_for_method(self.method)
        report = self._run(request, baseline=baseline)
        common = self._common_provenance(self.method, request)
        merged = dict(common)
        merged.update(report.provenance or {})
        report.provenance = merged
        if "Mitigation is not a guarantee" not in " ".join(report.warnings):
            report.warnings.append(
                "Mitigation is not a guarantee of factual correctness."
            )
        return report

    def _run(self, request: GenerationRequest, *, baseline: bool) -> MitigationReport:
        method = self.method
        if method == "dola":
            from hallucination.mitigation.methods.dola import DoLaConfig, DoLaMethod

            config = DoLaConfig(
                candidate_premature_layers=self.method_args.get(
                    "candidate_premature_layers"
                ),
                mature_layer=self.method_args.get("mature_layer"),
                relative_top=float(self.method_args.get("relative_top", 0.1)),
                post_softmax=bool(self.method_args.get("post_softmax", True)),
                official_compat=bool(self.method_args.get("official_compat", False)),
                baseline=baseline,
            )
            return DoLaMethod().run(self._ensure_engine(), request, config)
        if method == "caa":
            from hallucination.mitigation.core.artifacts import SteeringArtifact
            from hallucination.mitigation.methods.caa import CAAConfig, CAAMethod

            if self.steering is None:
                from hallucination.mitigation.core.errors import ArtifactNotFoundError

                raise ArtifactNotFoundError(
                    "CAA requires a steering artifact. Run 'hallucination-mitigation fit-steering' first "
                    "and pass --steering."
                )
            artifact = (
                self.steering
                if isinstance(self.steering, SteeringArtifact)
                else SteeringArtifact.load(self.steering)
            )
            config = CAAConfig(
                layer=int(self.method_args.get("layer", artifact.layer)),
                multiplier=float(
                    self.method_args.get("multiplier", artifact.multiplier)
                ),
                position=str(self.method_args.get("position", "generated")),
                baseline=baseline,
            )
            return CAAMethod().run(self._ensure_engine(), request, artifact, config)
        if method == "cad":
            from hallucination.mitigation.methods.cad import CADConfig, CADMethod

            config = CADConfig(
                alpha=float(self.method_args.get("alpha", 0.5)),
                context_template=str(self.method_args.get("context_template", "plain")),
                filter_top_p=float(self.method_args.get("filter_top_p", 1.0)),
                baseline=baseline,
            )
            return CADMethod().run(self._ensure_engine(), request, config)
        if method == "iti":
            from hallucination.mitigation.core.artifacts import ITIArtifact
            from hallucination.mitigation.methods.iti import ITIConfig, ITIMethod

            if self.probe is None:
                from hallucination.mitigation.core.errors import ArtifactNotFoundError

                raise ArtifactNotFoundError(
                    "ITI requires a probe artifact. Run 'hallucination-mitigation fit-iti' first "
                    "and pass --probe."
                )
            artifact = (
                self.probe
                if isinstance(self.probe, ITIArtifact)
                else ITIArtifact.load(self.probe)
            )
            config = ITIConfig(
                top_k=int(self.method_args.get("top_k", artifact.top_k)),
                alpha=float(self.method_args.get("alpha", artifact.alpha)),
                position=str(self.method_args.get("position", "generated")),
                baseline=baseline,
                bake=bool(self.method_args.get("bake", False)),
            )
            return ITIMethod().run(self._ensure_engine(), request, artifact, config)
        if method == "cove":
            from hallucination.mitigation.methods.cove import CoVeConfig, CoVeMethod

            config = CoVeConfig(
                max_questions=int(self.method_args.get("max_questions", 6)),
                mode=str(self.method_args.get("mode", "factored")),
                temperature=float(self.method_args.get("temperature", 0.0)),
                max_answer_tokens=int(self.method_args.get("max_answer_tokens", 128)),
                baseline=baseline,
            )
            return CoVeMethod().run(self._ensure_llm_backend(), request, config)
        raise ConfigError(f"unsupported method: {method}")

    def compare(
        self,
        prompt: str,
        context: str | None = None,
        *,
        methods: list[str] | None = None,
        **kwargs: Any,
    ) -> ComparisonReport:
        selected = methods or ["dola", "cad", "cove"]
        reports = []
        for method in selected:
            if method not in _METHODS:
                raise ConfigError(f"unknown method: {method}")
            previous = self.method
            self.method = method
            try:
                reports.append(self.mitigate(prompt, context=context, **kwargs))
            finally:
                self.method = previous
        return ComparisonReport(prompt=prompt, context=context, reports=reports)

    def close(self) -> None:
        self._llm_backend = None
        self._engine = None
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()
