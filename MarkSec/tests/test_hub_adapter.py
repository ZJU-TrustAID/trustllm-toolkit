"""Hub adapter contract tests (offline): tiny random model, no network, no GPU."""

import json

import pytest
import yaml
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

from src.api.engine import Settings, ToolkitEngine
from src import hub_adapter


@pytest.fixture
def engine():
    # Same offline fixture as test_toolkit_api: a randomly initialized model
    # keeps the adapter tests hermetic.
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


def write_config(tmp_path, *, chapter_params=None, model_entry=None):
    config = {
        'version': 1,
        'chapter': 7,
        'toolkit': 'marksec',
        'common': {'device': 'cpu', 'dtype': 'float32', 'seed': 42},
        'models': {'default': model_entry or {'source': 'local_path', 'path': 'offline-unit-test-model'}},
        'chapter_params': chapter_params if chapter_params is not None else {
            'prompt': 'the proof is private',
            'watermark': 'KGW',
            'attack': 'LLMP',
            'max_new_tokens': 16,
            'seed': 42,
        },
    }
    path = tmp_path / 'config.yaml'
    path.write_text(yaml.safe_dump(config, allow_unicode=True), encoding='utf-8')
    return path


def read_result(output_dir):
    return json.loads((output_dir / 'result.json').read_text(encoding='utf-8'))


def test_run_succeeds_and_writes_full_contract(tmp_path, monkeypatch, engine):
    monkeypatch.setattr(hub_adapter, 'load_engine', lambda settings: engine)
    output_dir = tmp_path / 'session'
    config_path = write_config(tmp_path)

    exit_code = hub_adapter.main(['--config', str(config_path), '--output-dir', str(output_dir)])

    assert exit_code == 0
    result = read_result(output_dir)
    assert result['status'] == 'succeeded'
    assert 'LLMP' in result['summary'] or '改写' in result['summary']
    tables = [b for b in result['blocks'] if b['type'] == 'table']
    assert len(tables) == 1
    assert tables[0]['header'][0] == '变体'
    assert len(tables[0]['rows']) == 2
    assert {row[0].split('（')[0] for row in tables[0]['rows']} == {'Watermarked', 'Attacked'}
    assert {item['name'] for item in result['artifacts']} == {'report.md', 'results.json'}
    assert result['data']['prompt'] == 'the proof is private'
    assert len(result['data']['variants']) == 2

    report = (output_dir / 'report.md').read_text(encoding='utf-8')
    assert '# 文本水印实验报告' in report
    assert '水印检测对比' in report
    scenario = json.loads((output_dir / 'results.json').read_text(encoding='utf-8'))
    assert scenario['watermark']['type'] == 'KGW'


def test_missing_prompt_fails_with_exit_code_2(tmp_path, monkeypatch, engine):
    monkeypatch.setattr(hub_adapter, 'load_engine', lambda settings: engine)
    output_dir = tmp_path / 'session'
    config_path = write_config(tmp_path, chapter_params={'watermark': 'KGW'})

    exit_code = hub_adapter.main(['--config', str(config_path), '--output-dir', str(output_dir)])

    assert exit_code == 2
    result = read_result(output_dir)
    assert result['status'] == 'failed'
    assert 'prompt' in result['summary']
    assert result['artifacts'] == []


def test_api_model_source_is_rejected(tmp_path, monkeypatch, engine):
    monkeypatch.setattr(hub_adapter, 'load_engine', lambda settings: engine)
    output_dir = tmp_path / 'session'
    entry = {'source': 'api', 'api': {'base_url': 'https://example.test/v1', 'model': 'x'}}
    config_path = write_config(tmp_path, model_entry=entry)

    exit_code = hub_adapter.main(['--config', str(config_path), '--output-dir', str(output_dir)])

    assert exit_code == 2
    assert read_result(output_dir)['status'] == 'failed'


@pytest.mark.parametrize('params', [
    {'prompt': 'test', 'watermark': 'KGWX'},
    {'prompt': 'test', 'attack': 'RW'},
    {'prompt': 'test', 'max_new_tokens': 10000},
])
def test_invalid_chapter_params_fail(tmp_path, monkeypatch, engine, params):
    monkeypatch.setattr(hub_adapter, 'load_engine', lambda settings: engine)
    output_dir = tmp_path / 'session'
    config_path = write_config(tmp_path, chapter_params=params)

    exit_code = hub_adapter.main(['--config', str(config_path), '--output-dir', str(output_dir)])

    assert exit_code == 2
    assert read_result(output_dir)['status'] == 'failed'
