from __future__ import annotations

import os
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from threading import Lock

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.attack.llmp.llmp import LLMParaphraseAttack, LLMParaphraseConfig
from src.watermark import load_watermark


@dataclass(frozen=True)
class Settings:
    model_id: str = 'HuggingFaceTB/SmolLM2-135M-Instruct'
    model_revision: str = 'main'
    device: str = 'auto'
    dtype: str = 'auto'
    local_files_only: bool = False
    api_token: str = ''

    @classmethod
    def from_env(cls):
        return cls(
            model_id=os.getenv('MARKSEC_MODEL_ID', cls.model_id),
            model_revision=os.getenv('MARKSEC_MODEL_REVISION', 'main'),
            device=os.getenv('MARKSEC_DEVICE', 'auto'),
            dtype=os.getenv('MARKSEC_DTYPE', 'auto'),
            local_files_only=os.getenv('MARKSEC_LOCAL_FILES_ONLY', 'false').lower() == 'true',
            api_token=os.getenv('MARKSEC_API_TOKEN', ''),
        )


class ToolkitEngine:
    """One resident model; each call runs real watermarking, LLMP and detection.

    The interactive LLMP route reuses the generation model to fit small hosts.
    Benchmark threat-model selection remains in the original experiment runner.
    """

    def __init__(self, settings: Settings, *, model=None, tokenizer=None):
        self.settings = settings
        self.model = model
        self.tokenizer = tokenizer
        self.lock = Lock()
        self.device = settings.device

    def _load(self):
        if self.model is not None and self.tokenizer is not None:
            self.device = str(self.model.device)
            return
        device = self.settings.device
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if device.startswith('cuda') and not torch.cuda.is_available():
            raise RuntimeError('The configured CUDA device is unavailable.')
        dtype = (self.settings.dtype or 'auto').lower()
        if dtype in ('auto', ''):
            torch_dtype = torch.float16 if device.startswith('cuda') else torch.float32
        elif dtype in ('float16', 'bfloat16', 'float32'):
            # float16 compute is unsupported on CPU; keep float32 there.
            torch_dtype = getattr(torch, dtype) if device.startswith('cuda') else torch.float32
        else:
            raise RuntimeError(f"Unsupported dtype: {self.settings.dtype!r} (use auto/float16/bfloat16/float32).")
        options = {'local_files_only': self.settings.local_files_only, 'trust_remote_code': False,
                   'revision': self.settings.model_revision}
        tokenizer = AutoTokenizer.from_pretrained(self.settings.model_id, **options)
        # Only safetensors weights are accepted, including on the pinned Torch runtime.
        model = AutoModelForCausalLM.from_pretrained(
            self.settings.model_id,
            use_safetensors=True,
            torch_dtype=torch_dtype,
            **options,
        ).to(device).eval()
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        self.model, self.tokenizer, self.device = model, tokenizer, device

    def capabilities(self):
        return {
            'mode': 'live',
            'model': self.settings.model_id,
            'model_revision': self.settings.model_revision,
            'model_loaded': self.model is not None,
            'watermarks': ['KGW', 'Unigram'],
            'attacks': ['LLMP'],
            'max_new_tokens': 256,
            'description': 'Real generation, MarkSec LLMP rewriting, and watermark detection.',
        }

    def _watermark(self, name: str, max_new_tokens: int):
        return load_watermark(
            algorithm_name=name, algorithm_config_path=None,
            model=self.model, tokenizer=self.tokenizer, device=self.device,
            max_new_tokens=max_new_tokens, min_new_tokens=min(16, max_new_tokens),
        )

    def _format_prompt(self, prompt: str):
        if getattr(self.tokenizer, 'chat_template', None):
            return self.tokenizer.apply_chat_template(
                [{'role': 'user', 'content': prompt}],
                tokenize=False, add_generation_prompt=True,
            )
        return prompt

    def _check_context(self, text: str, new_tokens: int):
        limit = int(getattr(self.model.config, 'max_position_embeddings', 2048))
        count = len(self.tokenizer(text)['input_ids'])
        if count + new_tokens > limit:
            raise ValueError(f'Input and output exceed the model context limit ({limit} tokens).')

    def _variant(self, text: str, watermark, *, attacked: bool):
        detection = watermark.detect_watermark(text)
        ids = self.tokenizer(text, add_special_tokens=False)['input_ids']
        processor = watermark.logits_processor
        width = int(watermark.config.config_dict.get('prefix_length', 1))
        tokens = []
        # Preserve tokenizer offsets and whitespace rather than inserting spaces
        # between subword tokens in the browser.
        encoded = self.tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        cursor = 0
        for position, (token_id, (_start, end)) in enumerate(zip(ids, encoded['offset_mapping'])):
            label = 'neutral'
            if position >= width:
                if hasattr(processor, 'token_is_green'):
                    green = processor.token_is_green(ids[:position], token_id)
                else:
                    context = torch.tensor(ids[:position], dtype=torch.long, device=self.device)
                    green_ids = processor._get_greenlist_ids(context)
                    green = bool((green_ids == token_id).any().item())
                label = 'green' if green else 'red'
            end = max(cursor, end)
            tokens.append({'text': text[cursor:end], 'label': label, 'token_id': token_id, 'position': position})
            cursor = end
        if cursor < len(text):
            tokens.append({'text': text[cursor:], 'label': 'neutral', 'position': len(ids)})
        return {
            'label': 'Attacked' if attacked else 'Watermarked',
            'description': 'MarkSec LLMP model rewrite.' if attacked else 'Fresh model generation with watermark logits processing.',
            'text': text, 'tokens': tokens, 'token_spacing': 'verbatim',
            'detection': {
                'z_score': detection.get('score'),
                'green_fraction': detection.get('green_fraction', 0.0),
                'num_scored_tokens': detection['num_tokens_scored'],
                'is_watermarked': detection['is_watermarked'],
                'threshold': detection.get('threshold', 4.0),
                'reason': detection.get('reason'),
            },
            **({'attack_method': 'LLMP'} if attacked else {}),
        }

    def run(self, *, prompt: str, watermark: str, max_new_tokens: int, seed: int):
        self._load()
        start = time.monotonic()
        formatted = self._format_prompt(prompt)
        self._check_context(formatted, max_new_tokens)
        wm = self._watermark(watermark, max_new_tokens)
        config_path = Path(__file__).resolve().parents[2] / 'configs/attack/LLMP.json'
        attack_config = LLMParaphraseConfig(
            str(config_path),
            generation_config={
                'max_new_tokens': max_new_tokens, 'min_new_tokens': min(16, max_new_tokens),
                'do_sample': True, 'temperature': 0.7, 'top_p': 0.9,
            },
        )
        attack = LLMParaphraseAttack(
            config=attack_config, base_model=self.model, tokenizer=self.tokenizer,
        )
        # The HTTP layer serializes access; fork_rng restores process RNG state.
        devices = [torch.device(self.device).index or 0] if self.device.startswith('cuda') else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(seed)
            generated = wm.generate_watermarked_text(formatted)[len(formatted):].strip()
            if not generated:
                raise RuntimeError('The model returned an empty completion.')
            rewrite_prompt = attack_config.prompt_template.format(text=generated, prompt=prompt)
            self._check_context(self._format_prompt(rewrite_prompt), max_new_tokens)
            rewritten = attack.edit(generated, prompt=prompt)
        return {
            'id': str(uuid.uuid4()), 'mode': 'live',
            'title': f'{watermark} watermark vs LLMP',
            'description': 'Computed by MarkSec for this request.',
            'prompt': prompt, 'model': self.settings.model_id,
            'watermark': {'type': watermark, 'gamma': 0.5, 'delta': 2.0,
                          'prefix_length': 1 if watermark == 'KGW' else 0},
            'variants': [self._variant(generated, wm, attacked=False),
                         self._variant(rewritten, wm, attacked=True)],
            'provenance': {
                'backend': 'transformers', 'device': self.device, 'seed': seed,
                'model_revision': getattr(self.model.config, '_commit_hash', None) or self.settings.model_revision,
                'attack': 'MarkSec LLMP (shared generation model)',
                'elapsed_seconds': round(time.monotonic() - start, 3),
            },
        }

    def detect(self, *, text: str, watermark: str):
        self._load()
        self._check_context(text, 0)
        return {'mode': 'live', 'model': self.settings.model_id,
                'watermark': watermark,
                'variant': self._variant(text, self._watermark(watermark, 16), attacked=False)}
