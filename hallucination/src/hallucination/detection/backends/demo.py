from __future__ import annotations

import hashlib
import math

from hallucination.detection.backends.nli import NLIProbabilities
from hallucination.detection.utils.text import normalize_answer


class DemoLLMBackend:
    """Deterministic offline generator used by the ``offline_toy`` preset.

    The outputs mirror the prompts used in the textbook examples. They are fixed
    teaching fixtures, not a model.
    """

    _RESPONSES = {
        "蓝鲸": (
            "蓝鲸主要以磷虾和小型鱼类为食，是海洋中的哺乳动物。",
            [
                "蓝鲸主要以磷虾为食。",
                "蓝鲸以磷虾等小型甲壳动物为主食。",
                "蓝鲸主要捕食磷虾，也会吃小鱼。",
                "蓝鲸是海洋哺乳动物，以磷虾为主食。",
            ],
        ),
        "海葵": (
            "海葵是动物，不是植物，属于刺胞动物。",
            [
                "海葵属于动物，不是植物。",
                "海葵是刺胞动物。",
                "海葵是肉食性海洋动物。",
                "海葵不是植物，而是动物。",
            ],
        ),
    }

    def __init__(self, responses: dict[str, tuple[str, list[str]]] | None = None) -> None:
        self.responses = responses or self._RESPONSES

    def generate(self, prompt: str, **kwargs: object) -> str:
        return self._select(prompt)[0]

    def sample(self, prompt: str, n: int, **kwargs: object) -> list[str]:
        if n < 1:
            raise ValueError("n must be positive")
        response, samples = self._select(prompt)
        pool = [sample for sample in samples if sample != response]
        return [pool[index % len(pool)] for index in range(n)]

    def _select(self, prompt: str) -> tuple[str, list[str]]:
        for keyword, value in self.responses.items():
            if keyword in prompt:
                return value
        return (
            "这是一个离线教学回答。",
            [
                "这是一个离线教学回答。",
                "这是一个用于演示的离线回答。",
                "离线演示回答。",
            ],
        )


class DemoNLIBackend:
    """Small deterministic NLI fixture for offline response-level clustering."""
    _TOPICS = {
        "蓝鲸": {"蓝鲸", "磷虾", "小鱼", "哺乳动物", "海洋"},
        "海葵": {"海葵", "刺胞动物", "动物", "植物", "海洋"},
    }

    def __init__(self, model_name: str = "offline_demo_nli") -> None:
        self.model_name = model_name

    def __call__(self, premise: str, hypothesis: str) -> NLIProbabilities:
        left = normalize_answer(premise)
        right = normalize_answer(hypothesis)
        if left == right:
            return NLIProbabilities(entailment=0.98, contradiction=0.01, neutral=0.01)
        if self._contradiction(left, right):
            return NLIProbabilities(entailment=0.01, contradiction=0.95, neutral=0.04)
        overlap = self._overlap(left, right)
        if overlap >= 1:
            return NLIProbabilities(entailment=0.85, contradiction=0.03, neutral=0.12)
        return NLIProbabilities(entailment=0.05, contradiction=0.02, neutral=0.93)

    @staticmethod
    def _overlap(left: str, right: str) -> int:
        left_terms = {term for term in ("磷虾", "海藻", "植物", "动物", "刺胞", "哺乳", "海洋") if term in left}
        right_terms = {term for term in ("磷虾", "海藻", "植物", "动物", "刺胞", "哺乳", "海洋") if term in right}
        return len(left_terms & right_terms)

    @staticmethod
    def _contradiction(left: str, right: str) -> bool:
        if ("海藻" in left and "磷虾" in right) or ("磷虾" in left and "海藻" in right):
            return True

        def plant_claim(text: str) -> bool:
            return "是植物" in text and "不是植物" not in text

        return plant_claim(left) != plant_claim(right)


class DemoPTrueBackend:
    """Deterministic logprob backend for the P(True) offline demo."""

    model_name = "offline_demo_p_true"

    def generate(self, prompt: str, **kwargs: object) -> str:
        value = self._p_true(prompt)
        return "A" if value >= 0.5 else "B"

    def next_token_logprobs(self, prompt: str, candidates: list[str]) -> dict[str, float]:
        probability = self._p_true(prompt)
        true_tokens = {" A", "A", " True", "True"}
        return {
            candidate: math.log(probability if candidate in true_tokens else 1.0 - probability)
            for candidate in candidates
        }

    @staticmethod
    def _p_true(prompt: str) -> float:
        possible = prompt.split("Possible answer:", 1)[-1].split("Is the possible answer:", 1)[0]
        if "海藻" in possible and "蓝鲸" in prompt:
            return 0.2
        if "植物" in possible and "不是植物" not in possible and "海葵" in prompt:
            return 0.2
        return 0.9


class HashingEmbeddingBackend:
    """Deterministic character-n-gram hashing embedding for offline demos."""

    def __init__(self, dimensions: int = 64, ngram: int = 2, model_name: str = "offline_hashing_embedding") -> None:
        if dimensions < 4:
            raise ValueError("dimensions must be at least 4")
        if ngram < 1:
            raise ValueError("ngram must be at least 1")
        self.dimensions = dimensions
        self.ngram = ngram
        self.model_name = model_name

    def __call__(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def _embed(self, text: str) -> list[float]:
        normalized = normalize_answer(text)
        if len(normalized) < self.ngram:
            grams = [normalized] if normalized else [""]
        else:
            grams = [normalized[index : index + self.ngram] for index in range(len(normalized) - self.ngram + 1)]
        vector = [0.0] * self.dimensions
        for gram in grams:
            digest = hashlib.sha256(gram.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "little") % self.dimensions
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[index] += sign
        norm = math.sqrt(sum(value * value for value in vector))
        return [value / norm for value in vector] if norm else vector


def build_demo_backends() -> tuple[DemoLLMBackend, DemoNLIBackend, HashingEmbeddingBackend]:
    return DemoLLMBackend(), DemoNLIBackend(), HashingEmbeddingBackend()
