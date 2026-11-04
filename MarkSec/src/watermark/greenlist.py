from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any

import torch
from transformers import LogitsProcessor

from src.util.text import slice_completion


@dataclass
class WatermarkConfig:
    config_dict: dict[str, Any]

    def __getattr__(self, name: str) -> Any:
        try:
            return self.config_dict[name]
        except KeyError as exc:
            raise AttributeError(name) from exc


@dataclass
class WatermarkRuntimeConfig:
    model: Any
    tokenizer: Any
    device: str
    max_new_tokens: int
    min_new_tokens: int = 0
    do_sample: bool = True
    no_repeat_ngram_size: int = 4


class GreenlistLogitsProcessor(LogitsProcessor):
    def __init__(
        self,
        *,
        vocab_size: int,
        gamma: float,
        delta: float,
        hash_key: int,
        prefix_length: int,
        device: str | torch.device,
    ) -> None:
        self.vocab_size = int(vocab_size)
        self.gamma = min(max(float(gamma), 0.0), 1.0)
        self.delta = float(delta)
        self.hash_key = int(hash_key)
        self.prefix_length = max(int(prefix_length), 0)
        self.device = torch.device(device)
        self.greenlist_size = max(1, min(self.vocab_size, int(round(self.gamma * self.vocab_size))))

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if self.delta == 0.0 or self.greenlist_size <= 0:
            return scores

        batch_size = int(input_ids.shape[0])
        for row_idx in range(batch_size):
            green_ids = self._greenlist_ids(input_ids[row_idx].tolist(), scores.device)
            scores[row_idx, green_ids] = scores[row_idx, green_ids] + self.delta
        return scores

    def _context_seed(self, token_ids: list[int]) -> int:
        if self.prefix_length <= 0:
            context = []
        else:
            context = token_ids[-self.prefix_length :]
        payload = ','.join(str(int(token_id)) for token_id in context)
        raw = f'{self.hash_key}:{payload}'.encode('utf-8')
        digest = hashlib.blake2b(raw, digest_size=8).digest()
        return int.from_bytes(digest, 'big') % (2**31 - 1)

    def _greenlist_ids(self, token_ids: list[int], device: torch.device) -> torch.LongTensor:
        generator = torch.Generator(device='cpu')
        generator.manual_seed(self._context_seed(token_ids))
        permutation = torch.randperm(self.vocab_size, generator=generator)
        return permutation[: self.greenlist_size].to(device)

    def token_is_green(self, token_ids_before: list[int], token_id: int) -> bool:
        green_ids = self._greenlist_ids(token_ids_before, torch.device('cpu'))
        return bool((green_ids == int(token_id)).any().item())


class GreenlistWatermark:
    def __init__(
        self,
        *,
        config_dict: dict[str, Any],
        runtime_config: WatermarkRuntimeConfig,
    ) -> None:
        self.config = WatermarkConfig(dict(config_dict))
        self.runtime_config = runtime_config
        self.model = runtime_config.model
        self.tokenizer = runtime_config.tokenizer
        self.device = runtime_config.device

        vocab_size = int(getattr(self.model.config, 'vocab_size', 0) or len(self.tokenizer))
        gamma = float(self.config.config_dict.get('gamma', 0.5))
        delta = float(self.config.config_dict.get('delta', 2.0))
        hash_key = int(self.config.config_dict.get('hash_key', 15485863))
        prefix_length = int(self.config.config_dict.get('prefix_length', 1))
        self.z_threshold = float(self.config.config_dict.get('z_threshold', 4.0))

        self.logits_processor = GreenlistLogitsProcessor(
            vocab_size=vocab_size,
            gamma=gamma,
            delta=delta,
            hash_key=hash_key,
            prefix_length=prefix_length,
            device=runtime_config.device,
        )

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        return self.logits_processor(input_ids, scores)

    @property
    def gamma(self) -> float:
        return self.logits_processor.gamma

    @property
    def prefix_length(self) -> int:
        return self.logits_processor.prefix_length

    def _generate(self, prompt: str, *, logits_processors: list[LogitsProcessor]) -> str:
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        inputs = self.tokenizer(prompt, return_tensors='pt').to(self.device)
        gen_kwargs = {
            'max_new_tokens': int(self.runtime_config.max_new_tokens),
            'min_new_tokens': int(self.runtime_config.min_new_tokens),
            'do_sample': bool(self.runtime_config.do_sample),
            'no_repeat_ngram_size': int(self.runtime_config.no_repeat_ngram_size),
            'pad_token_id': self.tokenizer.pad_token_id,
        }
        if logits_processors:
            gen_kwargs['logits_processor'] = logits_processors

        with torch.no_grad():
            output_ids = self.model.generate(**inputs, **gen_kwargs)

        completion = slice_completion(
            output_ids,
            int(inputs['input_ids'].shape[-1]),
            self.tokenizer,
            skip_special_tokens=True,
        )
        return prompt + completion

    def generate_watermarked_text(self, prompt: str, **_: Any) -> str:
        return self._generate(prompt, logits_processors=[self.logits_processor])

    def generate_unwatermarked_text(self, prompt: str, **_: Any) -> str:
        return self._generate(prompt, logits_processors=[])

    def detect_watermark(self, text: str, **_: Any) -> dict[str, Any]:
        token_ids = self.tokenizer(text, add_special_tokens=False)['input_ids']
        if token_ids and isinstance(token_ids[0], list):
            token_ids = token_ids[0]
        token_ids = [int(token_id) for token_id in token_ids]

        min_context = self.prefix_length
        if len(token_ids) <= min_context:
            return {
                'is_watermarked': False,
                'score': None,
                'num_tokens_scored': 0,
                'num_green_tokens': 0,
                'green_fraction': 0.0,
                'reason': 'not enough tokens for detection',
            }

        green_count = 0
        total = 0
        for idx in range(min_context, len(token_ids)):
            if self.logits_processor.token_is_green(token_ids[:idx], token_ids[idx]):
                green_count += 1
            total += 1

        expected = self.gamma
        variance = total * expected * (1.0 - expected)
        z_score = 0.0 if variance <= 0.0 else (green_count - total * expected) / math.sqrt(variance)
        return {
            'is_watermarked': bool(z_score > self.z_threshold),
            'score': float(z_score),
            'num_tokens_scored': int(total),
            'num_green_tokens': int(green_count),
            'green_fraction': float(green_count / total) if total else 0.0,
            'threshold': self.z_threshold,
        }
