from types import SimpleNamespace

import pytest

from hallucination.detection.backends.api import (
    OpenAIChatBackend,
)
from hallucination.detection.core.results import DetectionExample
from hallucination.detection.pipelines.whitebox_generation import (
    WhiteBoxGenerationPipeline,
)
from hallucination.detection.whitebox.p_true import PTrueScorer


class _FakeChatClient:
    """Offline stand-in for openai.OpenAI with scripted chat.completions payloads."""

    def __init__(self, *, text="蓝鲸主要以磷虾为食。", logprob_entries=None) -> None:
        self.text = text
        self.logprob_entries = logprob_entries
        self.calls: list[dict[str, object]] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, *, model: str, messages: list[dict[str, str]], **kwargs: object) -> object:
        self.calls.append({"model": model, "messages": messages, **kwargs})
        if self.logprob_entries is None:
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=self.text), logprobs=None)]
            )
        entry_objects = [
            SimpleNamespace(
                token=entry["token"],
                logprob=entry["logprob"],
                top_logprobs=[SimpleNamespace(token=t, logprob=v) for t, v in entry["top"]],
            )
            for entry in self.logprob_entries
        ]
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=self.text),
                    logprobs=SimpleNamespace(content=entry_objects),
                )
            ]
        )


def _token_entries() -> list[dict[str, object]]:
    return [
        {"token": "蓝鲸", "logprob": -0.2, "top": [("蓝鲸", -0.2), ("海豚", -1.7)]},
        {"token": "磷虾", "logprob": -0.6, "top": [("磷虾", -0.6), ("海藻", -1.1)]},
    ]


def test_openai_chat_backend_generate_and_sample():
    client = _FakeChatClient()
    backend = OpenAIChatBackend("test-model", client=client)
    assert backend.generate("请介绍蓝鲸的食性。") == "蓝鲸主要以磷虾为食。"
    assert len(backend.sample("请介绍蓝鲸的食性。", 2)) == 2
    assert client.calls[0]["model"] == "test-model"


def test_openai_chat_backend_generate_with_scores_feeds_whitebox_pipeline():
    client = _FakeChatClient(logprob_entries=_token_entries())
    backend = OpenAIChatBackend("test-model", client=client)
    generation = backend.generate_with_scores("请介绍蓝鲸的食性。", max_new_tokens=8, top_k=2)
    assert generation.tokens == ["蓝鲸", "磷虾"]
    assert generation.token_logprobs == pytest.approx([-0.2, -0.6])
    assert generation.top_logprobs[0]["海豚"] == pytest.approx(-1.7)
    assert client.calls[0]["logprobs"] is True
    assert client.calls[0]["top_logprobs"] == 2

    pipeline = WhiteBoxGenerationPipeline(backend)
    result = pipeline.detect_prompt("请介绍蓝鲸的食性。", max_new_tokens=8, top_k=2)
    methods = {item.method for item in result.method_scores}
    assert {"mtp", "lntp"} <= methods


def test_openai_chat_backend_rejects_missing_logprobs():
    backend = OpenAIChatBackend("test-model", client=_FakeChatClient())
    with pytest.raises(ValueError, match="did not return token logprobs"):
        backend.generate_with_scores("请介绍蓝鲸的食性。")


def test_openai_chat_backend_next_token_logprobs_returns_window_only():
    entries = [{"token": " A", "logprob": -0.3, "top": [(" A", -0.3), (" B", -1.4), (" C", -9.0)]}]
    backend = OpenAIChatBackend("test-model", client=_FakeChatClient(logprob_entries=entries))
    values = backend.next_token_logprobs("prompt", [" A", " B", " absent"])
    assert values == {" A": pytest.approx(-0.3), " B": pytest.approx(-1.4)}
    assert " absent" not in values


def test_p_true_falls_back_to_verbalized_when_ab_tokens_missing():
    entries = [{"token": " 蓝鲸", "logprob": -0.1, "top": [(" 蓝鲸", -0.1)]}]
    client = _FakeChatClient(text="A", logprob_entries=entries)
    backend = OpenAIChatBackend("test-model", client=client)
    scorer = PTrueScorer(backend)
    result = scorer.score(
        DetectionExample(prompt="蓝鲸吃什么？", response="蓝鲸以磷虾为食。", samples=["蓝鲸吃磷虾。"])
    )
    assert result.raw["method"] == "verbalized"
    assert "fallback_reason" in result.raw
    assert any("verbalized fallback was used" in warning for warning in result.warnings)


def test_openai_whitebox_preset_uses_chat_backend():
    from hallucination.detection import Detector

    client = _FakeChatClient(logprob_entries=_token_entries())
    backend = OpenAIChatBackend("test-model", client=client)
    detector = Detector(preset="openai_whitebox", backend=backend)
    report = detector.detect("请介绍蓝鲸的食性。", max_new_tokens=8, top_k=2)
    methods = {item.method for item in report.result.method_scores}
    assert {"mtp", "lntp"} <= methods
