from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from hallucination.detection.utils.text import normalize_answer


@dataclass(frozen=True)
class NLIProbabilities:
    """Probabilities for `premise -> hypothesis` natural-language inference."""

    entailment: float
    contradiction: float
    neutral: float = 0.0

    def __post_init__(self) -> None:
        for value in (self.entailment, self.contradiction, self.neutral):
            if not 0.0 <= value <= 1.0:
                raise ValueError("NLI probabilities must be in [0, 1]")


class NLIBackend(Protocol):
    def __call__(self, premise: str, hypothesis: str) -> NLIProbabilities:
        ...


class BidirectionalNLIBackend:
    """Transformers NLI backend with explicit bidirectional-equivalence support."""

    def __init__(self, model_name: str = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli", *, device: str | None = None) -> None:
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as exc:
            raise ImportError("Install the nli extra first: python3 -m pip install -e '.[nli]'") from exc
        self.model_name = model_name
        self._torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        self._labels = {str(label).lower(): int(index) for index, label in self.model.config.id2label.items()}
        if not any("entail" in label for label in self._labels) or not any("contrad" in label for label in self._labels):
            raise ValueError(
                f"NLI model '{model_name}' must expose entailment and contradiction labels; got {sorted(self._labels)}"
            )

    def __call__(self, premise: str, hypothesis: str) -> NLIProbabilities:
        inputs = self.tokenizer(premise, hypothesis, truncation=True, return_tensors="pt").to(self.device)
        with self._torch.no_grad():
            values = self._torch.softmax(self.model(**inputs).logits[0], dim=-1).detach().cpu().tolist()
        return NLIProbabilities(
            entailment=self._label_probability(values, "entail"),
            contradiction=self._label_probability(values, "contrad"),
            neutral=self._label_probability(values, "neutral"),
        )

    def equivalent(self, left: str, right: str, *, threshold: float = 0.5) -> bool:
        return self(left, right).entailment >= threshold and self(right, left).entailment >= threshold

    def _label_probability(self, values: list[float], prefix: str) -> float:
        for label, index in self._labels.items():
            if prefix in label:
                return float(values[index])
        return 0.0


class LexicalEquivalenceBackend:
    """Placeholder equivalence backend; replace with NLI or LLM-as-NLI in production."""

    def __call__(self, left: str, right: str) -> bool:
        return normalize_answer(left) == normalize_answer(right)
