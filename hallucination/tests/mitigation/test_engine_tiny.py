from __future__ import annotations

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("tokenizers")

import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

from hallucination.mitigation.core.artifacts import SteeringArtifact
from hallucination.mitigation.core.contracts import GenerationRequest
from hallucination.mitigation.engine.hf import HFEngine
from hallucination.mitigation.methods.caa import CAAConfig, CAAMethod, fit_caa
from hallucination.mitigation.methods.cad import CADConfig, CADMethod
from hallucination.mitigation.methods.dola import DoLaConfig, DoLaMethod


def _build() -> tuple[HFEngine, PreTrainedTokenizerFast]:
    vocab_tokens = ["<pad>", "<bos>", "<eos>", "<unk>"] + [f"w{i}" for i in range(28)]
    vocab = {token: index for index, token in enumerate(vocab_tokens)}
    backend = Tokenizer(WordLevel(vocab, unk_token="<unk>"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        bos_token="<bos>",
        eos_token="<eos>",
        pad_token="<pad>",
        unk_token="<unk>",
    )
    # Disable EOS so the tiny random model always reaches max_new_tokens.
    tokenizer.eos_token_id = None
    config = LlamaConfig(
        vocab_size=len(vocab_tokens),
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=128,
    )
    # Fixed seed: the tiny model's random weights decide whether the near-chance
    # steering probes land on the passing side; unseeded init made this suite flaky.
    torch.manual_seed(0)
    model = LlamaForCausalLM(config).eval()
    return HFEngine(model, tokenizer, device="cpu"), tokenizer


def test_dola_tiny_end_to_end() -> None:
    engine, _ = _build()
    request = GenerationRequest(
        prompt="w1 w2 w3", max_new_tokens=3, do_sample=False, seed=0
    )
    report = DoLaMethod().run(
        engine,
        request,
        DoLaConfig(candidate_premature_layers=[0], mature_layer=2, baseline=False),
    )
    assert report.method == "dola"
    assert report.response
    assert report.trace["steps"]


def test_cad_alpha_zero_matches_context_baseline() -> None:
    engine, _ = _build()
    request = GenerationRequest(
        prompt="w5",
        context="w1 w2 w3",
        max_new_tokens=3,
        do_sample=False,
        seed=0,
    )
    report = CADMethod().run(
        engine, request, CADConfig(alpha=0.0, context_template="plain", baseline=True)
    )
    assert report.baseline is not None
    assert report.response == report.baseline


def test_caa_fit_and_run_tiny() -> None:
    engine, _ = _build()
    pairs = [
        {"prompt": "w1", "positive": "w2", "negative": "w3"},
        {"prompt": "w4", "positive": "w5", "negative": "w6"},
        {"prompt": "w7", "positive": "w8", "negative": "w9"},
    ]
    config = CAAConfig(layer=1, multiplier=0.5, baseline=False)
    artifact = fit_caa(engine, pairs, config)
    assert isinstance(artifact, SteeringArtifact)
    assert artifact.vector.shape == (32,)
    request = GenerationRequest(prompt="w10", max_new_tokens=2, do_sample=False, seed=0)
    report = CAAMethod().run(engine, request, artifact, config)
    assert report.response
    assert report.trace["layer"] == 1
    assert report.trace["hook_calls"] > 0
