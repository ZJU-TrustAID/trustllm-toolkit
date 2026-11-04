from __future__ import annotations

from collections import Counter


class KeywordEmbeddingBackend:
    """Tiny deterministic embedding backend for tests and demos."""

    def __call__(self, texts: list[str]) -> list[list[float]]:
        vocab = sorted({token for text in texts for token in text.lower().split()})
        vectors: list[list[float]] = []
        for text in texts:
            counts = Counter(text.lower().split())
            vectors.append([float(counts[token]) for token in vocab])
        return vectors


class SentenceTransformerEmbeddingBackend:
    """Production embedding adapter loaded only when explicitly constructed."""

    def __init__(self, model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", *, device: str | None = None) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError("Install the embeddings extra first: python3 -m pip install -e '.[embeddings]'") from exc
        self.model_name = model_name
        self.model = SentenceTransformer(model_name, device=device)

    def __call__(self, texts: list[str]) -> list[list[float]]:
        vectors = self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        return vectors.tolist()


class BERTScoreSimilarityBackend:
    """Return BERTScore F1 values for aligned candidate/reference text pairs."""

    def __init__(self, *, lang: str = "en", model_type: str | None = None, rescale_with_baseline: bool = True) -> None:
        try:
            import bert_score
        except ImportError as exc:
            raise ImportError("Install the selfcheck-bertscore extra first: python3 -m pip install -e '.[selfcheck-bertscore]'") from exc
        self._bert_score = bert_score
        self.lang = lang
        self.model_type = model_type
        self.rescale_with_baseline = rescale_with_baseline

    def __call__(self, candidates: list[str], references: list[str]) -> list[float]:
        if len(candidates) != len(references):
            raise ValueError("BERTScore candidates and references must have the same length")
        _, _, f1 = self._bert_score.score(
            candidates,
            references,
            lang=self.lang,
            model_type=self.model_type,
            rescale_with_baseline=self.rescale_with_baseline,
            verbose=False,
        )
        return [float(value) for value in f1.tolist()]
