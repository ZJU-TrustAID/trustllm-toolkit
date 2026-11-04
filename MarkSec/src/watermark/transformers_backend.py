from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
from transformers import (
    SynthIDTextWatermarkingConfig,
    WatermarkDetector,
    WatermarkingConfig,
)

from src.util.text import slice_completion

from .greenlist import WatermarkConfig, WatermarkRuntimeConfig


@dataclass
class DetectionStats:
    num_tokens_scored: int
    positive_tokens: int
    positive_fraction: float
    score: float
    threshold: float
    is_watermarked: bool


class VLLMLogitsProcessorAdapter:
    def __init__(self, watermarking_config, vocab_size: int):
        self.watermarking_config = watermarking_config
        self.vocab_size = int(vocab_size)
        self._processors: dict[str, Any] = {}

    def __call__(self, prompt_token_ids, past_token_ids=None, scores=None):
        if scores is None:
            scores = past_token_ids
            token_ids = prompt_token_ids
        else:
            token_ids = [*list(prompt_token_ids), *list(past_token_ids)]

        ids = self._as_batched_ids(token_ids, scores.device)
        logits = self._as_batched_scores(scores)
        if isinstance(self.watermarking_config, SynthIDTextWatermarkingConfig):
            processed = self._process_synthid_stateless(ids, logits)
        else:
            processed = self._processor_for(scores.device)(ids, logits)
        if scores.dim() == 1:
            return processed[0]
        return processed

    def _processor_for(self, device: torch.device):
        key = str(device)
        processor = self._processors.get(key)
        if processor is None:
            processor = self.watermarking_config.construct_processor(self.vocab_size, device)
            self._processors[key] = processor
        return processor

    def _process_synthid_stateless(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor,
    ) -> torch.FloatTensor:
        processor = self._processor_for(scores.device)
        batch_size, vocab_size = scores.shape
        context_width = int(self.watermarking_config.ngram_len) - 1
        if input_ids.shape[-1] >= context_width:
            context = input_ids[:, -context_width:]
        else:
            padding = torch.zeros(
                (batch_size, context_width - input_ids.shape[-1]),
                dtype=torch.long,
                device=scores.device,
            )
            context = torch.cat((padding, input_ids), dim=1)

        all_indices = torch.arange(vocab_size, device=scores.device).unsqueeze(0)
        all_indices = all_indices.expand(batch_size, -1)
        ngram_keys, context_hash = processor._compute_keys(context, all_indices)
        g_values = processor.sample_g_values(ngram_keys)

        updated_scores = processor.update_scores(scores, g_values)
        if input_ids.shape[-1] >= context_width * 2 and context_width > 0:
            previous_contexts = input_ids[:, :-1].unfold(dimension=1, size=context_width, step=1)
            previous_hashes = torch.vmap(processor.accumulate_hash, in_dims=(None, 1), out_dims=1)(
                torch.ones(batch_size, device=scores.device, dtype=torch.long),
                previous_contexts,
            )
            is_repeated = (previous_hashes[:, :-1] == context_hash[:, None]).any(
                dim=1,
                keepdim=True,
            )
            updated_scores = torch.where(is_repeated, scores, updated_scores)
        return updated_scores

    @staticmethod
    def _as_batched_ids(input_ids, device: torch.device) -> torch.LongTensor:
        if isinstance(input_ids, torch.Tensor):
            ids = input_ids.to(device=device, dtype=torch.long)
        else:
            ids = torch.tensor(list(input_ids), dtype=torch.long, device=device)
        if ids.dim() == 1:
            ids = ids.unsqueeze(0)
        return ids

    @staticmethod
    def _as_batched_scores(scores: torch.FloatTensor) -> torch.FloatTensor:
        if scores.dim() == 1:
            return scores.unsqueeze(0)
        return scores


class TransformersWatermark:
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

        algorithm_name = str(self.config.config_dict.get('algorithm_name') or '').upper()
        if algorithm_name == 'SYNTHID':
            self.watermarking_config = self._build_synthid_config(self.config.config_dict)
            self._detector = None
        else:
            self.watermarking_config = self._build_kgw_config(self.config.config_dict)
            self._detector = WatermarkDetector(
                model_config=self.model.config,
                device=self.device,
                watermarking_config=self.watermarking_config,
                ignore_repeated_ngrams=bool(
                    self.config.config_dict.get('ignore_repeated_ngrams', False)
                ),
            )

        self.logits_processor = self.watermarking_config.construct_processor(
            self._vocab_size(),
            self.device,
        )
        self.vllm_logits_processor = VLLMLogitsProcessorAdapter(
            self.watermarking_config,
            self._vocab_size(),
        )

    @staticmethod
    def _build_kgw_config(config_dict: dict[str, Any]) -> WatermarkingConfig:
        greenlist_ratio = config_dict.get('greenlist_ratio', config_dict.get('gamma', 0.25))
        bias = config_dict.get('bias', config_dict.get('delta', 2.0))
        hashing_key = config_dict.get('hashing_key', config_dict.get('hash_key', 15485863))
        seeding_scheme = config_dict.get('seeding_scheme', 'lefthash')
        context_width = config_dict.get('context_width', config_dict.get('prefix_length', 1))
        return WatermarkingConfig(
            greenlist_ratio=float(greenlist_ratio),
            bias=float(bias),
            hashing_key=int(hashing_key),
            seeding_scheme=str(seeding_scheme),
            context_width=max(1, int(context_width)),
        )

    @staticmethod
    def _build_synthid_config(config_dict: dict[str, Any]) -> SynthIDTextWatermarkingConfig:
        keys = config_dict.get('keys') or [654, 400, 836, 123, 340, 443, 597, 160, 57]
        return SynthIDTextWatermarkingConfig(
            ngram_len=max(2, int(config_dict.get('ngram_len', 5))),
            keys=[int(key) for key in keys],
            context_history_size=int(config_dict.get('context_history_size', 1024)),
            sampling_table_seed=int(config_dict.get('sampling_table_seed', 0)),
            sampling_table_size=int(config_dict.get('sampling_table_size', 2**16)),
            skip_first_ngram_calls=bool(config_dict.get('skip_first_ngram_calls', False)),
            debug_mode=bool(config_dict.get('debug_mode', False)),
        )

    def _vocab_size(self) -> int:
        return int(getattr(self.model.config, 'vocab_size', 0) or len(self.tokenizer))

    def __call__(self, prompt_token_ids, past_token_ids=None, scores=None):
        return self.vllm_logits_processor(prompt_token_ids, past_token_ids, scores)

    def _generate(self, prompt: str, *, watermarked: bool) -> str:
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
        if watermarked:
            gen_kwargs['watermarking_config'] = self.watermarking_config

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
        return self._generate(prompt, watermarked=True)

    def generate_unwatermarked_text(self, prompt: str, **_: Any) -> str:
        return self._generate(prompt, watermarked=False)

    def detect_watermark(self, text: str, **_: Any) -> dict[str, Any]:
        algorithm_name = str(self.config.config_dict.get('algorithm_name') or '').upper()
        if algorithm_name == 'SYNTHID':
            return self._detect_synthid(text)
        return self._detect_kgw(text)

    def _tokenize_for_detection(self, text: str) -> torch.LongTensor:
        tokenized = self.tokenizer(text, add_special_tokens=False, return_tensors='pt')
        return tokenized['input_ids'].to(self.device)

    @staticmethod
    def _first_scalar(value: Any) -> float:
        if hasattr(value, 'tolist'):
            value = value.tolist()
        if isinstance(value, list):
            if not value:
                return 0.0
            return TransformersWatermark._first_scalar(value[0])
        return float(value)

    def _detect_kgw(self, text: str) -> dict[str, Any]:
        input_ids = self._tokenize_for_detection(text)
        z_threshold = float(self.config.config_dict.get('z_threshold', 3.0))
        try:
            detection = self._detector(input_ids, z_threshold=z_threshold, return_dict=True)
        except ValueError as exc:
            return {
                'is_watermarked': False,
                'score': None,
                'num_tokens_scored': 0,
                'num_green_tokens': 0,
                'green_fraction': 0.0,
                'reason': str(exc),
            }

        prediction = bool(self._first_scalar(detection.prediction))
        return {
            'is_watermarked': prediction,
            'score': self._first_scalar(detection.z_score),
            'num_tokens_scored': int(self._first_scalar(detection.num_tokens_scored)),
            'num_green_tokens': int(self._first_scalar(detection.num_green_tokens)),
            'green_fraction': self._first_scalar(detection.green_fraction),
            'p_value': self._first_scalar(detection.p_value),
            'confidence': self._first_scalar(detection.confidence),
            'threshold': z_threshold,
        }

    def _detect_synthid(self, text: str) -> dict[str, Any]:
        input_ids = self._tokenize_for_detection(text)
        stats = self._score_synthid(input_ids)
        if stats.num_tokens_scored <= 0:
            return {
                'is_watermarked': False,
                'score': None,
                'num_tokens_scored': 0,
                'num_positive_tokens': 0,
                'positive_fraction': 0.0,
                'reason': 'not enough tokens for SynthID detection',
            }
        return {
            'is_watermarked': stats.is_watermarked,
            'score': stats.score,
            'num_tokens_scored': stats.num_tokens_scored,
            'num_positive_tokens': stats.positive_tokens,
            'positive_fraction': stats.positive_fraction,
            'threshold': stats.threshold,
            'detector': 'marksec_synthid_statistical',
        }

    def _score_synthid(self, input_ids: torch.LongTensor) -> DetectionStats:
        processor = self.watermarking_config.construct_processor(
            self._vocab_size(),
            self.device,
        )
        ngram_len = int(self.watermarking_config.ngram_len)
        if input_ids.shape[-1] < ngram_len:
            return DetectionStats(0, 0, 0.0, 0.0, 0.0, False)

        g_values = processor.compute_g_values(input_ids)
        context_mask = processor.compute_context_repetition_mask(input_ids)
        eos_token_id = self.tokenizer.eos_token_id
        if eos_token_id is not None:
            eos_mask = processor.compute_eos_token_mask(input_ids, eos_token_id=eos_token_id)
            context_mask = context_mask * eos_mask[:, ngram_len - 1 :]

        active = context_mask.bool()
        if not active.any():
            return DetectionStats(0, 0, 0.0, 0.0, 0.0, False)

        token_positive = (g_values.float().mean(dim=-1) > 0.5) & active
        total = int(active.sum().item())
        positive = int(token_positive.sum().item())
        fraction = positive / total if total else 0.0
        variance = total * 0.25
        z_score = 0.0 if variance <= 0 else (positive - 0.5 * total) / math.sqrt(variance)
        threshold = float(self.config.config_dict.get('z_threshold', 3.0))
        return DetectionStats(
            num_tokens_scored=total,
            positive_tokens=positive,
            positive_fraction=fraction,
            score=float(z_score),
            threshold=threshold,
            is_watermarked=bool(z_score > threshold),
        )
