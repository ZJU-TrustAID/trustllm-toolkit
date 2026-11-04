"""Real-model backend for the Signal Desk workbench.

Loads one causal LM and serves the hub's interactive flow:
generate → choose a detection/mitigation method → run with parameters.
Detection scorers, mitigation methods and the auxiliary NLI / embedding
backends all come from the hallucination package itself, so the numbers shown
in the browser are exactly what the toolkit computes.

The auxiliary models (NLI, sentence embeddings) are configurable per instance
via :meth:`RealModelManager.configure_aux` — local path or HuggingFace repo id.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import threading
import time
from typing import Any

from hallucination.detection.backends.embeddings import (
    SentenceTransformerEmbeddingBackend,
)
from hallucination.detection.backends.hf_whitebox import (
    HuggingFaceWhiteBoxBackend,
)
from hallucination.detection.backends.nli import BidirectionalNLIBackend
from hallucination.detection.blackbox.exact_match import ExactMatchConsistencyScorer
from hallucination.detection.blackbox.response_cosine import (
    ResponseCosineConsistencyScorer,
)
from hallucination.detection.blackbox.response_semantic_entropy import (
    ResponseSemanticEntropyScorer,
)
from hallucination.detection.blackbox.selfcheck_bertscore import (
    SelfCheckBERTScoreScorer,
)
from hallucination.detection.blackbox.selfcheck_ngram import SelfCheckNgramScorer
from hallucination.detection.blackbox.selfcheck_nli import SelfCheckNLIScorer
from hallucination.detection.core.results import DetectionExample
from hallucination.detection.utils.text import split_sentences
from hallucination.detection.whitebox.entropy import TokenEntropyScorer
from hallucination.detection.whitebox.p_true import PTrueScorer
from hallucination.detection.whitebox.token_probability import (
    LNTPScorer,
    MTPScorer,
)
from hallucination.mitigation.engine.hf import HFEngine

DEFAULT_NLI_MODEL = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

WHITEBOX_METHODS = ("mtp", "lntp", "token_entropy", "eigenscore", "effective_rank")
BLACKBOX_METHODS = (
    "exact_match", "selfcheck_ngram", "response_cosine",
    "semantic_entropy", "selfcheck_nli", "bertscore", "p_true",
)
MITIGATION_METHODS = ("dola", "cad", "iti", "cove")


class RealBackendError(RuntimeError):
    pass


@dataclass
class AuxModels:
    """Lazily built NLI / embedding backends with user-chosen sources."""

    nli_source: str = DEFAULT_NLI_MODEL          # local path or HF repo id
    nli_device: str | None = None
    embedding_source: str = DEFAULT_EMBEDDING_MODEL
    embedding_device: str | None = None
    _nli: Any = field(default=None, init=False, repr=False)
    _embedding: Any = field(default=None, init=False, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    def set_sources(self, nli_source: str | None = None, embedding_source: str | None = None,
                    nli_device: str | None = None, embedding_device: str | None = None) -> None:
        with self._lock:
            if nli_source and nli_source != self.nli_source:
                self.nli_source = nli_source
                self._nli = None
            if embedding_source and embedding_source != self.embedding_source:
                self.embedding_source = embedding_source
                self._embedding = None
            if nli_device is not None:
                self.nli_device = nli_device or None
            if embedding_device is not None:
                self.embedding_device = embedding_device or None

    def nli(self) -> BidirectionalNLIBackend:
        with self._lock:
            if self._nli is None:
                self._nli = BidirectionalNLIBackend(model_name=self.nli_source, device=self.nli_device)
            return self._nli

    def embedding(self) -> SentenceTransformerEmbeddingBackend:
        with self._lock:
            if self._embedding is None:
                self._embedding = SentenceTransformerEmbeddingBackend(
                    model_name=self.embedding_source, device=self.embedding_device
                )
            return self._embedding


class RealModelManager:
    """One loaded causal LM shared by detection, mitigation and generation."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._backend: HuggingFaceWhiteBoxBackend | None = None
        self._engine: Any = None
        self._config: dict[str, Any] = {}
        self.loaded_at: float | None = None
        self.aux = AuxModels()
        # 表征分析（EigenScore/有效秩）抽样缓存：抽样一次，检测复用
        self._hidden_cache: dict[str, dict[str, Any]] = {}
        # hub 渲染配置里的默认模型（首次真实调用时懒加载）
        self._pending_model: str | None = None
        self._pending_device: str | None = None
        self._pending_dtype: str = "auto"

    def sample_hidden(self, prompt: str, n_samples: int, hidden_state_layer: int,
                      max_new_tokens: int = 32) -> dict[str, Any]:
        """独立生成 n 条回答：返回文本列表并缓存隐状态矩阵（供谱分析复用）。"""
        import hashlib

        backend = self._require()
        key = hashlib.sha1(
            f"{self._config.get('model')}|{prompt}|{n_samples}|{hidden_state_layer}|{max_new_tokens}".encode()
        ).hexdigest()[:12]
        cached = self._hidden_cache.get(key)
        if cached:
            return {"sample_key": key, "texts": cached["texts"], "n_samples": len(cached["texts"])}
        texts, vectors = [], []
        for _ in range(max(2, n_samples)):
            generation = backend.generate_with_scores(
                prompt, max_new_tokens=max_new_tokens, do_sample=True, temperature=1.0, top_p=0.95
            )
            texts.append(generation.response)
            vectors.append(self._pooled_generated_hidden(backend, prompt, generation, hidden_state_layer))
        import numpy as np

        self._hidden_cache = {key: {"texts": texts, "matrix": np.vstack(vectors)}}
        return {"sample_key": key, "texts": texts, "n_samples": len(texts)}

    def hidden_matrix(self, sample_key: str | None) -> Any | None:
        return self._hidden_cache.get(sample_key or "", {}).get("matrix")

    # -- lifecycle -----------------------------------------------------------
    def status(self) -> dict[str, Any]:
        backend = self._backend
        return {
            "loaded": backend is not None,
            "model": self._config.get("model") or self._pending_model,
            "device": self._config.get("device") or self._pending_device,
            "dtype": self._config.get("dtype", "auto"),
            "loaded_at": self.loaded_at,
            "aux": {
                "nli": self.aux.nli_source,
                "embedding": self.aux.embedding_source,
            },
        }

    @property
    def pending_model(self) -> str | None:
        return self._pending_model

    def load(self, model: str, device: str | None = None, dtype: str = "auto", *, force: bool = False) -> dict[str, Any]:
        with self._lock:
            if self._backend is not None and not force and self._config.get("model") == model:
                return self.status()
            load_kwargs: dict[str, Any] = {"device": device}
            if dtype != "auto":
                load_kwargs["torch_dtype"] = dtype
            backend = HuggingFaceWhiteBoxBackend.from_pretrained(model, **load_kwargs)
            self._backend = backend
            self._engine = None
            self._config = {"model": model, "device": backend.device, "dtype": dtype}
            self.loaded_at = time.time()
            return self.status()

    def _require(self) -> HuggingFaceWhiteBoxBackend:
        if self._backend is None and self._pending_model:
            print(f"[web-demo] 懒加载模型: {self._pending_model} @ {self._pending_device or 'auto'}")
            self.load(self._pending_model, device=self._pending_device, dtype=self._pending_dtype)
        if self._backend is None:
            raise RealBackendError("模型尚未加载，请先调用 /api/real/load 或点击「加载模型」")
        return self._backend

    def _engine_for_mitigation(self) -> Any:
        """Reuse the already-loaded weights for mitigation (no second copy)."""
        if self._engine is None:
            backend = self._require()
            self._engine = HFEngine(backend.model, backend.tokenizer, device=backend.device)
        return self._engine

    def _llm_backend(self) -> Any:
        """A CoVe-compatible backend over the loaded engine."""
        from hallucination.mitigation.backends.hf import HFBackend

        return HFBackend(self._engine_for_mitigation())

    # -- generation ------------------------------------------------------------
    def generate(self, prompt: str, *, max_new_tokens: int = 128, do_sample: bool = False,
                 temperature: float = 1.0, top_p: float = 1.0) -> dict[str, Any]:
        backend = self._require()
        started = time.time()
        generation = backend.generate_with_scores(
            prompt, max_new_tokens=max_new_tokens, do_sample=do_sample,
            temperature=temperature, top_p=top_p, top_k=5,
        )
        return {
            "response": generation.response,
            "num_tokens": len(generation.token_ids),
            "latency_seconds": round(time.time() - started, 2),
            "token_logprobs": generation.token_logprobs,
            "tokens": self._display_tokens(backend, generation.tokens),
            "metadata": generation.metadata,
        }

    def generate_stream(self, prompt: str, *, max_new_tokens: int = 128, do_sample: bool = False,
                        temperature: float = 1.0, top_p: float = 1.0):
        """逐 token 产出的生成器：{"token": 片段} … 最后 {"done": true, …完整统计}。"""
        import threading

        from transformers import TextIteratorStreamer

        backend = self._require()
        started = time.time()
        encoded = backend.tokenizer(prompt, return_tensors="pt")
        encoded = backend._move_batch_to_device(encoded, backend.device)
        prompt_length = int(encoded["input_ids"].shape[-1])
        streamer = TextIteratorStreamer(backend.tokenizer, skip_prompt=True, skip_special_tokens=True)
        gen_kwargs = {
            **encoded,
            "max_new_tokens": int(max_new_tokens),
            "streamer": streamer,
            "return_dict_in_generate": True,
            "output_scores": True,
            "do_sample": do_sample,
            "temperature": float(temperature),
            "top_p": float(top_p),
        }
        holder: dict[str, Any] = {}

        def _run() -> None:
            try:
                holder["output"] = backend.model.generate(**gen_kwargs)
            except Exception as exc:  # surface generation errors on the main loop
                holder["error"] = exc

        worker = threading.Thread(target=_run)
        worker.start()
        for piece in streamer:
            if piece:
                yield {"token": piece}
        worker.join()
        if "error" in holder:
            raise holder["error"]
        output = holder["output"]

        sequences = output.sequences
        generated_ids = [int(token_id) for token_id in sequences[0, prompt_length:].detach().cpu().tolist()]
        response = backend.tokenizer.decode(generated_ids, skip_special_tokens=True)
        tokens = backend.tokenizer.convert_ids_to_tokens(generated_ids, skip_special_tokens=False)
        token_logprobs = []
        for token_id, logits in zip(generated_ids, output.scores, strict=False):
            import torch

            log_probs = torch.log_softmax(logits[0], dim=-1)
            token_logprobs.append(float(log_probs[token_id].detach().cpu().item()))
        yield {
            "done": True,
            "response": response,
            "num_tokens": len(generated_ids),
            "latency_seconds": round(time.time() - started, 2),
            "tokens": self._display_tokens(backend, tokens),
            "token_logprobs": token_logprobs,
        }

    def sample_texts(self, prompt: str, n_samples: int, *, max_new_tokens: int = 64,
                     temperature: float = 1.0, top_p: float = 0.95) -> list[str]:
        backend = self._require()
        texts = []
        for _ in range(max(1, n_samples)):
            generation = backend.generate_with_scores(
                prompt, max_new_tokens=max_new_tokens, do_sample=True,
                temperature=temperature, top_p=top_p,
            )
            texts.append(generation.response)
        return texts

    # -- detection: white-box ----------------------------------------------------
    def detect_whitebox(self, prompt: str, method: str, *, max_new_tokens: int = 128,
                        n_samples: int = 5, hidden_state_layer: int = -1,
                        sample_key: str | None = None, **generate_kwargs: Any) -> dict[str, Any]:
        backend = self._require()
        if method not in WHITEBOX_METHODS:
            raise RealBackendError(f"未知白盒方法: {method}")

        if method in {"eigenscore", "effective_rank"}:
            import numpy as np

            started = time.time()
            matrix = self.hidden_matrix(sample_key)
            if matrix is None:
                vectors = []
                for _ in range(max(2, n_samples)):
                    generation = backend.generate_with_scores(prompt, max_new_tokens=max_new_tokens, **generate_kwargs)
                    vectors.append(self._pooled_generated_hidden(backend, prompt, generation, hidden_state_layer))
                matrix = np.vstack(vectors)

            # 工具包的 EigenScore risk=sigmoid(raw) 未校准、EffectiveRank 的
            # "坍缩即风险"方向是针对单回答 token 矩阵的；对多样本一致性场景
            # （INSIDE 语义：样本表示越聚集 = 模型越确定 = 幻觉风险越低），
            # 这里在适配层按文献方向重算最终值，并暴露原始谱。
            centered = matrix - matrix.mean(axis=0, keepdims=True)
            gram = centered @ centered.T / (matrix.shape[0] - 1)
            eigenvalues = np.clip(np.linalg.eigvalsh(gram), 0.0, None)
            spectrum = [round(float(value), 2) for value in eigenvalues]
            singular_values = np.linalg.svd(matrix, compute_uv=False)
            total = float(singular_values.sum())
            probs = singular_values / total if total > 0 else np.ones_like(singular_values) / len(singular_values)
            spectral_entropy = float(-np.sum(probs * np.log(probs + 1e-12)))
            effective_rank = float(np.exp(spectral_entropy))
            max_rank = max(1, min(matrix.shape))
            # 多样本一致性语义：秩越低 = 表示越聚集 = 越确定；秩越高 = 越分散 = 越不确定
            diversity = max(0.0, min(1.0, (effective_rank - 1.0) / (max_rank - 1.0))) if max_rank > 1 else 0.0
            risk = round(diversity, 4)
            score = round(1.0 - risk, 4)
            return {
                "final": {
                    "method": method,
                    "signal_type": "generation_confidence",
                    "score": score,
                    "risk": risk,
                    "is_hallucinated": risk >= 0.5,
                },
                "intermediates": {
                    "n_samples": int(matrix.shape[0]),
                    "hidden_dim": int(matrix.shape[1]),
                    "hidden_state_layer": hidden_state_layer,
                    "effective_rank": round(effective_rank, 3),
                    "spectral_entropy": round(spectral_entropy, 3),
                    "eigenvalues_gram": spectrum,
                    "note": (
                        "方法语义（INSIDE）：n 条独立回答各取生成 token 的平均隐状态向量，"
                        "谱分析衡量样本间一致性——表示越聚集（有效秩低）= 模型越确定 = 风险越低；"
                        "表示越分散（有效秩高）= 模型在猜 = 风险越高。绝对阈值需在验证集上拟合。"
                    ),
                },
                "per_sample": [],
                "latency_seconds": round(time.time() - started, 2),
            }

        started = time.time()
        generation = backend.generate_with_scores(prompt, max_new_tokens=max_new_tokens, **generate_kwargs)
        example = generation.to_detection_example()
        scorer = {"mtp": MTPScorer, "lntp": LNTPScorer, "token_entropy": TokenEntropyScorer}[method]()
        result = scorer.score(example)
        intermediates: dict[str, Any] = {
            "tokens": self._display_tokens(backend, generation.tokens),
            "token_logprobs": generation.token_logprobs,
            "top_logprobs": generation.top_logprob_entries,
        }
        if method == "token_entropy":
            per_token = [-float(lp) for lp in generation.token_logprobs]
            intermediates["per_token_entropy"] = per_token
            intermediates["mean_entropy"] = round(sum(per_token) / max(1, len(per_token)), 4)
            intermediates["note"] = "entropy = -log p(token)，均值即这段生成的不确定性"
        return self._detection_payload(result, started, intermediates=intermediates)

    # -- detection: black-box ------------------------------------------------------
    def detect_blackbox(self, prompt: str, method: str, *, response: str | None = None,
                        samples: list[str] | None = None, num_samples: int = 5,
                        n: int = 1, max_sample_tokens: int = 64, **sample_kwargs: Any) -> dict[str, Any]:
        backend = self._require()
        if method not in BLACKBOX_METHODS:
            raise RealBackendError(f"未知黑盒方法: {method}")
        started = time.time()
        if not response:
            generation = backend.generate_with_scores(prompt, max_new_tokens=max_sample_tokens)
            response = generation.response
        if not samples:
            samples = self.sample_texts(prompt, num_samples, max_new_tokens=max_sample_tokens, **sample_kwargs)

        example = DetectionExample(response=response, prompt=prompt, samples=samples)
        per_sample: list[dict[str, Any]] = {}

        if method == "exact_match":
            result = ExactMatchConsistencyScorer().score(example)
            intermediates: dict[str, Any] = {"samples": samples}
        elif method == "selfcheck_ngram":
            scorer = SelfCheckNgramScorer(n=n)
            sentences = split_sentences(response)
            results = scorer.score_sentences(example, sentences)
            intermediates = self._sentence_intermediates(sentences, results, samples)
            result = self._aggregate_sentence_results(results)
        elif method == "response_cosine":
            result = ResponseCosineConsistencyScorer(self.aux.embedding()).score(example)
            intermediates = {"samples": samples}
        elif method == "semantic_entropy":
            result = ResponseSemanticEntropyScorer(self.aux.nli()).score(example)
            intermediates = {"samples": samples}
        elif method == "selfcheck_nli":
            sentences = split_sentences(response)
            results = SelfCheckNLIScorer(self.aux.nli()).score_sentences(example, sentences)
            intermediates = self._sentence_intermediates(sentences, results, samples)
            result = self._aggregate_sentence_results(results)
        elif method == "bertscore":
            from hallucination.detection.backends.embeddings import BERTScoreSimilarityBackend

            sentences = split_sentences(response)
            scorer = SelfCheckBERTScoreScorer(BERTScoreSimilarityBackend())
            results = scorer.score_sentences(example, sentences)
            intermediates = self._sentence_intermediates(sentences, results, samples)
            result = self._aggregate_sentence_results(results)
        elif method == "p_true":
            backend_with_generate = _GenerateAdapter(self)
            result = PTrueScorer(backend_with_generate).score(example)
            intermediates = {"samples": samples, "note": "P(True) 另需少量采样做头脑风暴，已自动完成"}
        else:  # pragma: no cover - guarded above
            raise RealBackendError(f"未知黑盒方法: {method}")

        return self._detection_payload(result, started, intermediates=intermediates, per_sample=per_sample)

    @staticmethod
    def _display_tokens(backend: HuggingFaceWhiteBoxBackend, tokens: list[str]) -> list[str]:
        """Byte-level BPE 片段 → 可读文本：增量解码后做差，保持与 token 一一对应。"""
        try:
            pieces: list[str] = []
            for index in range(len(tokens)):
                previous = backend.tokenizer.convert_tokens_to_string(tokens[:index]) or ""
                current = backend.tokenizer.convert_tokens_to_string(tokens[: index + 1]) or ""
                piece = current[len(previous):]
                pieces.append(piece if piece else tokens[index])
            return pieces
        except Exception:
            return tokens

    @staticmethod
    def _sentence_intermediates(sentences: list[str], results: list[Any], samples: list[str]) -> dict[str, Any]:
        return {
            "sentences": sentences,
            "sentence_risk": [round(float(item.risk), 4) for item in results],
            "samples": samples,
        }

    @staticmethod
    def _pooled_generated_hidden(backend: HuggingFaceWhiteBoxBackend, prompt: str,
                                 generation: Any, layer: int) -> list[float]:
        """Mean hidden state over the generated tokens, cast to float32.

        Re-implemented here because the toolkit's own extraction cannot convert
        bfloat16 tensors to numpy.
        """
        import torch

        encoded = backend.tokenizer(prompt, return_tensors="pt")
        encoded = backend._move_batch_to_device(encoded, backend.device)
        prompt_length = int(encoded["input_ids"].shape[-1])
        generated_ids = backend.tokenizer(generation.response, add_special_tokens=False, return_tensors="pt")["input_ids"]
        input_ids = torch.cat([encoded["input_ids"], generated_ids.to(encoded["input_ids"].device)], dim=1)
        attention_mask = torch.ones_like(input_ids)
        with torch.no_grad():
            output = backend.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        hidden = output.hidden_states[layer][0]           # [seq_len, dim]
        generated_part = hidden[prompt_length:]
        if generated_part.shape[0] == 0:
            generated_part = hidden
        vector = generated_part.mean(dim=0)
        return vector.detach().to(torch.float32).cpu().tolist()

    @staticmethod
    def _aggregate_sentence_results(results: list[Any]) -> Any:
        class _Aggregated:
            def __init__(self, items: list[Any]) -> None:
                self.score = sum(float(item.score) for item in items) / max(1, len(items))
                self.risk = sum(float(item.risk) for item in items) / max(1, len(items))
                self.method = items[0].method if items else "sentence"
                self.signal_type = items[0].signal_type if items else "self_consistency"
                self.level = "response"
        return _Aggregated(results)

    def _detection_payload(self, result: Any, started: float, *, intermediates: dict[str, Any],
                           per_sample: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        return {
            "final": {
                "method": getattr(result, "method", "unknown"),
                "signal_type": str(getattr(result, "signal_type", "")),
                "score": round(float(result.score), 4),
                "risk": round(float(result.risk), 4),
                "is_hallucinated": bool(result.risk >= 0.5),
            },
            "intermediates": intermediates,
            "per_sample": per_sample or [],
            "latency_seconds": round(time.time() - started, 2),
        }

    # -- mitigation --------------------------------------------------------------
    def mitigate(self, method: str, prompt: str, *, context: str | None = None,
                 max_new_tokens: int = 128, probe: str | None = None,
                 method_args: dict[str, Any] | None = None,
                 **generate_kwargs: Any) -> dict[str, Any]:
        from hallucination.mitigation import Mitigator

        if method not in MITIGATION_METHODS:
            raise RealBackendError(f"未知缓解方法: {method}（CAA 已忽略）")
        backend = self._require()
        started = time.time()
        mitigator = Mitigator(
            method=method,
            model=str(self._config.get("model") or "loaded"),
            backend="hf",
            device=backend.device,
            seed=0,
            probe=probe or None,
            **(method_args or {}),
        )
        mitigator._engine = self._engine_for_mitigation()  # reuse loaded weights
        report = mitigator.mitigate(
            prompt, context=context or None, max_new_tokens=max_new_tokens,
            **generate_kwargs,
        )
        return {
            "method": report.method,
            "baseline": report.baseline,
            "mitigated": report.response,
            "trace": report.trace,
            "warnings": report.warnings,
            "latency_seconds": round(time.time() - started, 2),
        }


class _GenerateAdapter:
    """Expose manager.generate under the .generate() name PTrueScorer expects."""

    def __init__(self, manager: RealModelManager) -> None:
        self._manager = manager

    def generate(self, prompt: str, **kwargs: Any) -> str:
        return self._manager.generate(prompt, max_new_tokens=int(kwargs.get("max_new_tokens", 96))).get("response", "")


def build_manager() -> RealModelManager:
    return RealModelManager()


REAL = RealModelManager()
