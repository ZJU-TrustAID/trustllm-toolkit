from __future__ import annotations

from types import SimpleNamespace

import torch
from transformers import GPT2Config

from src.watermark import GreenlistWatermark, TransformersWatermark, load_watermark


class DummyTokenizer:
    eos_token = '<eos>'
    eos_token_id = 0
    pad_token = '<eos>'
    pad_token_id = 0

    def __len__(self) -> int:
        return 128

    def __call__(self, text, add_special_tokens=False, return_tensors=None):
        del add_special_tokens
        token_ids = [((ord(char) % 127) + 1) for char in str(text)]
        if return_tensors == 'pt':
            return {'input_ids': torch.tensor([token_ids], dtype=torch.long)}
        return {'input_ids': token_ids}

    def decode(self, token_ids, skip_special_tokens=True) -> str:
        del skip_special_tokens
        return ''.join(chr((int(token_id) % 26) + 97) for token_id in token_ids)


def dummy_model(vocab_size: int = 128):
    return SimpleNamespace(
        config=GPT2Config(
            vocab_size=vocab_size,
            bos_token_id=0,
            eos_token_id=0,
            pad_token_id=0,
        )
    )


def make_watermark(algorithm_name: str, *, max_new_tokens: int = 8):
    return load_watermark(
        algorithm_name=algorithm_name,
        algorithm_config_path=None,
        model=dummy_model(),
        tokenizer=DummyTokenizer(),
        device='cpu',
        max_new_tokens=max_new_tokens,
        min_new_tokens=0,
    )


def test_kgw_uses_transformers_watermark_backend_and_vllm_shape():
    watermark = make_watermark('KGW')

    assert isinstance(watermark, TransformersWatermark)
    assert watermark.watermarking_config.greenlist_ratio == 0.5

    scores = torch.zeros(128)
    processed = watermark([10, 11, 12], [13, 14], scores)

    assert processed.shape == scores.shape
    assert torch.count_nonzero(processed > scores).item() > 0


def test_synthid_uses_transformers_backend_and_statistical_detector():
    watermark = make_watermark('SynthID')

    assert isinstance(watermark, TransformersWatermark)
    assert watermark.watermarking_config.ngram_len == 5

    processed = watermark([1, 2, 3, 4], [5, 6], torch.zeros(128))
    assert processed.shape == (128,)
    assert torch.isfinite(processed).all()

    result = watermark.detect_watermark(
        'this is a deliberately long enough synthetic detector smoke test'
    )
    assert result['detector'] == 'marksec_synthid_statistical'
    assert result['num_tokens_scored'] > 0
    assert isinstance(result['is_watermarked'], bool)


def test_non_transformers_compatibility_backend_remains_available():
    watermark = make_watermark('Unigram')

    assert isinstance(watermark, GreenlistWatermark)

    input_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    scores = torch.zeros(1, 128)
    processed = watermark(input_ids, scores.clone())

    assert processed.shape == scores.shape
    assert torch.count_nonzero(processed > scores).item() > 0


def test_greenlist_detection_is_deterministic_for_same_text():
    watermark = make_watermark('Unigram')

    first = watermark.detect_watermark('the same text should produce the same score')
    second = watermark.detect_watermark('the same text should produce the same score')

    assert first['score'] == second['score']
    assert first['num_green_tokens'] == second['num_green_tokens']
