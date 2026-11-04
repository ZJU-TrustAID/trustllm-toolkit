from dataclasses import replace

import pytest
import torch
from fastapi.testclient import TestClient
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

from src.api.app import create_app
from src.api.engine import Settings, ToolkitEngine


@pytest.fixture
def engine():
    # A real, randomly initialized model keeps unit tests offline. A separate
    # pretrained-model smoke run validates useful inference before handoff.
    torch.set_num_threads(1)
    vocabulary = ['[UNK]', '[EOS]', 'the', 'a', 'proof', 'is', 'private', 'text',
                  'rewrite', 'meaning', 'one', 'two', 'three', 'test', '.', ':']
    tokenizer = Tokenizer(WordLevel({word: i for i, word in enumerate(vocabulary)}, unk_token='[UNK]'))
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token='[UNK]',
                                       eos_token='[EOS]', pad_token='[EOS]')
    torch.manual_seed(7)
    model = GPT2LMHeadModel(GPT2Config(
        vocab_size=len(vocabulary), n_layer=1, n_head=2, n_embd=16,
        n_positions=256, bos_token_id=1, eos_token_id=1, pad_token_id=1,
    )).eval()
    return ToolkitEngine(Settings(model_id='offline-unit-test-model', device='cpu'),
                         model=model, tokenizer=tokenizer)


@pytest.mark.parametrize('watermark', ['KGW', 'Unigram'])
def test_run_uses_model_and_detector_with_shared_frontend_contract(engine, watermark):
    client = TestClient(create_app(engine))
    request = {'prompt': 'the proof is private', 'watermark': watermark, 'max_new_tokens': 16}
    response = client.post('/v1/run', json=request)
    assert response.status_code == 200, response.text
    result = response.json()
    assert result['mode'] == 'live'
    assert result['prompt'] == request['prompt']
    assert result['provenance']['backend'] == 'transformers'
    assert len(result['variants']) == 2
    for variant in result['variants']:
        assert variant['text']
        assert ''.join(t['text'] for t in variant['tokens']) == variant['text']
        detected = client.post('/v1/detect', json={'text': variant['text'], 'watermark': watermark})
        assert detected.status_code == 200
        assert detected.json()['variant']['detection'] == variant['detection']
    repeated = client.post('/v1/run', json=request).json()
    assert [v['text'] for v in repeated['variants']] == [v['text'] for v in result['variants']]


def test_model_failure_returns_503_and_does_not_return_presets():
    engine = ToolkitEngine(Settings(model_id='missing-local-model-for-test', local_files_only=True))
    client = TestClient(create_app(engine))
    assert client.get('/health').json()['model_loaded'] is False
    result = client.post('/v1/run', json={'prompt': 'test'})
    assert result.status_code == 503
    assert 'variants' not in result.json()
    assert engine.lock.acquire(blocking=False)
    engine.lock.release()


def test_auth_validation_and_busy_responses(engine):
    engine.settings = replace(engine.settings, api_token='local-test-token')
    client = TestClient(create_app(engine))
    assert client.get('/v1/capabilities').status_code == 401
    headers = {'Authorization': 'Bearer local-test-token'}
    assert client.get('/v1/capabilities', headers=headers).json()['attacks'] == ['LLMP']
    for body in [{'prompt': ' '}, {'prompt': 'test', 'attack': 'RW'},
                 {'prompt': 'test', 'max_new_tokens': 10000},
                 {'prompt': 'test', 'model': 'untrusted-request-model'}]:
        assert client.post('/v1/run', json=body, headers=headers).status_code == 422
    with engine.lock:
        response = client.post('/v1/run', json={'prompt': 'test'}, headers=headers)
        assert response.status_code == 429
        assert response.headers['retry-after'] == '5'


def test_short_detection_and_context_limit(engine):
    client = TestClient(create_app(engine))
    result = client.post('/v1/detect', json={'text': 'proof', 'watermark': 'KGW'})
    assert result.status_code == 200
    assert result.json()['variant']['detection']['z_score'] is None
    assert client.post('/v1/run', json={'prompt': 'proof ' * 250}).status_code == 422
