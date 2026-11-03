import json
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from guardrail_toolkit.model import DEFAULT_MODEL, XGuard, parse_output

LABELS = {"sec": "Safe-Safe", "dw": "Crimes and Illegal Activities-Dangerous Weapons"}


@pytest.mark.parametrize("raw,verdict,label", [
    ("sec", "safe", "sec"),
    ("dw\n<explanation>Risk</explanation>", "unsafe", "dw"),
    ("unexpected", "unknown", "unexpected"),
    ("", "unknown", None),
    ("The result is sec", "unknown", "The"),
    ("<explanation>safe</explanation>", "unknown", None),
    ("sec_other", "unknown", None),
])
def test_decision_uses_only_valid_native_label(raw, verdict, label):
    parsed = parse_output(raw, LABELS)
    assert parsed["verdict"] == verdict
    assert parsed["label"] == label


def test_partial_explanation_is_preserved_and_marked_incomplete():
    parsed = parse_output("dw\n<explanation>The request", LABELS)
    assert parsed["explanation"] == "The request"
    assert not parsed["explanation_complete"]


def test_explanation_cannot_override_native_label():
    parsed = parse_output("dw\n<explanation>Contains the word sec.</explanation>", LABELS)
    assert parsed["verdict"] == "unsafe"
    assert parsed["explanation_complete"]


@pytest.fixture
def guard(tmp_path):
    (tmp_path / "tokenizer_config.json").write_text(json.dumps({"id2risk": LABELS}))
    return XGuard(str(tmp_path))


def test_metadata_does_not_load_weights(guard):
    assert guard.metadata()["state"] == "unloaded"
    assert len(guard.metadata()["labels"]) == 2
    assert guard._model is None


def test_hub_model_downloads_labels_then_loads_weights_lazily(tmp_path, monkeypatch):
    config = tmp_path / "tokenizer_config.json"
    config.write_text(json.dumps({"id2risk": LABELS}))
    download = Mock(return_value=str(config))
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(hf_hub_download=download))
    tokenizer = Mock()
    tokenizer.encode.side_effect = [[1], [2]]
    load_tokenizer = Mock(return_value=tokenizer)
    model = Mock()
    load_model = Mock(return_value=model)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(float32="float32"))
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        AutoTokenizer=SimpleNamespace(from_pretrained=load_tokenizer),
        AutoModelForCausalLM=SimpleNamespace(from_pretrained=load_model),
    ))

    guard = XGuard()
    assert guard.model_path == DEFAULT_MODEL
    assert guard.metadata()["state"] == "unloaded"
    download.assert_called_once_with(DEFAULT_MODEL, "tokenizer_config.json")
    load_model.assert_not_called()
    guard._load()
    load_tokenizer.assert_called_once_with(DEFAULT_MODEL)
    load_model.assert_called_once_with(
        DEFAULT_MODEL, torch_dtype="auto", device_map="auto", attn_implementation="sdpa",
    )
    assert guard.state == "ready"
    guard._load()
    assert load_model.call_count == 1


def test_hub_download_error_is_reported(monkeypatch):
    download = Mock(side_effect=OSError("Hub unavailable"))
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(hf_hub_download=download))
    with pytest.raises(OSError, match="Hub unavailable"):
        XGuard()


def test_explicit_missing_local_model_does_not_contact_hub(tmp_path, monkeypatch):
    download = Mock()
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(hf_hub_download=download))
    monkeypatch.chdir(tmp_path)
    with pytest.raises(FileNotFoundError):
        XGuard("./models/missing")
    download.assert_not_called()


def test_existing_relative_local_model_does_not_contact_hub(tmp_path, monkeypatch):
    model_dir = tmp_path / "models" / "guard"
    model_dir.mkdir(parents=True)
    (model_dir / "tokenizer_config.json").write_text(json.dumps({"id2risk": LABELS}))
    download = Mock()
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(hf_hub_download=download))
    monkeypatch.chdir(tmp_path)
    guard = XGuard("models/guard")
    assert guard.model_path == str(model_dir)
    assert guard.labels == LABELS
    download.assert_not_called()


def test_concurrent_request_is_rejected_before_loading(guard):
    guard._lock.acquire()
    try:
        with pytest.raises(BlockingIOError):
            guard.check_input("hello")
        assert guard._model is None
    finally:
        guard._lock.release()


def test_failed_load_releases_inference_lock(guard, monkeypatch):
    def fail():
        raise RuntimeError("unavailable")
    monkeypatch.setattr(guard, "_load", fail)
    with pytest.raises(RuntimeError):
        guard.check_input("hello")
    assert guard._lock.acquire(blocking=False)
    guard._lock.release()


@pytest.mark.parametrize("text,response", [("", "answer"), ("hello", ""), ("hello", None)])
def test_invalid_output_request_is_rejected_before_loading(guard, text, response):
    with pytest.raises(ValueError):
        guard.check_output(text, response)
    assert guard._model is None
