from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Any

import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    default_data_collator,
)

logger = logging.getLogger('MarkSec')


@dataclass
class B4DistillConfig:
    """Runtime configuration for distilling B4 proxy watermark model."""

    train_file: str
    proxy_model_name: str = 'facebook/opt-125m'
    output_dir: str = 'output/models/b4_proxy_opt125m'
    text_key: str = 'watermarked_text'
    block_size: int = 256
    max_samples: int | None = None
    epochs: float = 3.0
    batch_size: int = 32
    lr: float = 2e-5
    save_steps: int = 500
    save_total_limit: int = 2
    logging_steps: int = 50
    warmup_ratio: float = 0.0
    weight_decay: float = 0.0
    gradient_accumulation_steps: int = 1
    seed: int = 42
    trust_remote_code: bool = True
    local_files_only: bool = False
    fp16: bool | None = None
    bf16: bool | None = None
    tf32: bool | None = None
    optim: str = 'adamw_torch_fused'
    dataloader_num_workers: int = 4
    dataloader_pin_memory: bool = True
    pack_texts: bool = False
    attn_implementation: str | None = 'auto'
    torch_compile: bool = False
    gradient_checkpointing: bool = False

    def normalize(self) -> None:
        self.train_file = os.path.abspath(self.train_file)
        self.output_dir = os.path.abspath(self.output_dir)
        if self.max_samples is not None and self.max_samples <= 0:
            self.max_samples = None
        if self.block_size <= 0:
            self.block_size = 256
        if self.batch_size <= 0:
            self.batch_size = 1
        if self.epochs <= 0:
            self.epochs = 1.0
        if self.lr <= 0:
            self.lr = 2e-5
        if self.gradient_accumulation_steps <= 0:
            self.gradient_accumulation_steps = 1
        if self.dataloader_num_workers < 0:
            self.dataloader_num_workers = 0
        if self.attn_implementation is not None:
            value = str(self.attn_implementation).strip().lower()
            self.attn_implementation = value or None


def _supports_bf16() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        major, _minor = torch.cuda.get_device_capability()
    except Exception:
        return False
    return major >= 8


class B4ProxyDistiller:
    """
    Reusable distillation helper for B4 proxy model.

    Typical usage:
      distiller = B4ProxyDistiller(B4DistillConfig(train_file=..., ...))
      stats = distiller.run()
      model, tok = distiller.load_trained_proxy()
    """

    def __init__(self, config: B4DistillConfig):
        self.cfg = config
        self.cfg.normalize()
        self._resolved_attention_impl: str | None = None

    @staticmethod
    def load_corpus_texts(
        file_path: str,
        text_key: str = 'watermarked_text',
        max_samples: int | None = None,
    ) -> list[str]:
        texts: list[str] = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    logger.warning(
                        '[B4Distiller] skip invalid json line=%d in %s', line_idx, file_path
                    )
                    continue

                value = item.get(text_key, '')
                if isinstance(value, str):
                    value = value.strip()
                if value:
                    texts.append(value)

                if max_samples is not None and len(texts) >= max_samples:
                    break
        return texts

    def _resolve_precision_mode(self) -> tuple[torch.dtype | None, bool, bool]:
        use_bf16 = bool(self.cfg.bf16) if self.cfg.bf16 is not None else _supports_bf16()
        use_fp16 = bool(self.cfg.fp16) if self.cfg.fp16 is not None else False

        if use_bf16:
            use_fp16 = False
        elif not use_fp16 and torch.cuda.is_available():
            use_fp16 = True

        dtype: torch.dtype | None = None
        if torch.cuda.is_available():
            if use_bf16:
                dtype = torch.bfloat16
            elif use_fp16:
                dtype = torch.float16
        return dtype, use_fp16, use_bf16

    def _enable_cuda_speedups(self) -> None:
        if not torch.cuda.is_available():
            return
        use_tf32 = True if self.cfg.tf32 is None else bool(self.cfg.tf32)
        if use_tf32:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            if hasattr(torch, 'set_float32_matmul_precision'):
                torch.set_float32_matmul_precision('high')

    def _load_tokenizer_and_model(self):
        logger.info('[B4Distiller] loading base proxy model: %s', self.cfg.proxy_model_name)
        self._enable_cuda_speedups()
        torch_dtype, _use_fp16, use_bf16 = self._resolve_precision_mode()
        tokenizer = AutoTokenizer.from_pretrained(
            self.cfg.proxy_model_name,
            trust_remote_code=self.cfg.trust_remote_code,
            local_files_only=self.cfg.local_files_only,
        )

        model_kwargs: dict[str, Any] = {
            'trust_remote_code': self.cfg.trust_remote_code,
            'local_files_only': self.cfg.local_files_only,
            'low_cpu_mem_usage': True,
        }
        if torch_dtype is not None:
            model_kwargs['torch_dtype'] = torch_dtype

        requested_attn = self.cfg.attn_implementation
        attn_candidates: list[str | None]
        if requested_attn == 'auto':
            attn_candidates = ['flash_attention_2', 'sdpa', None]
        elif requested_attn:
            attn_candidates = [requested_attn]
        else:
            attn_candidates = [None]

        last_error: Exception | None = None
        model = None
        for attn_impl in attn_candidates:
            load_kwargs = dict(model_kwargs)
            if attn_impl is not None:
                load_kwargs['attn_implementation'] = attn_impl
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    self.cfg.proxy_model_name,
                    **load_kwargs,
                )
                self._resolved_attention_impl = attn_impl
                break
            except Exception as exc:
                last_error = exc
                logger.warning(
                    '[B4Distiller] failed to load with attn_implementation=%s: %s',
                    attn_impl,
                    exc,
                )

        if model is None:
            if last_error is not None:
                raise last_error
            raise RuntimeError('Failed to load proxy model for distillation.')

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        if hasattr(model.config, 'use_cache'):
            model.config.use_cache = False
        if self.cfg.gradient_checkpointing and hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable()
        precision_name = 'bf16' if use_bf16 else 'fp16' if torch_dtype == torch.float16 else 'fp32'
        logger.info(
            '[B4Distiller] precision=%s attn_implementation=%s',
            precision_name,
            self._resolved_attention_impl or 'default',
        )
        return model, tokenizer

    def _build_train_dataset(self, tokenizer) -> Dataset:
        texts = self.load_corpus_texts(
            self.cfg.train_file,
            text_key=self.cfg.text_key,
            max_samples=self.cfg.max_samples,
        )
        if not texts:
            raise ValueError(
                f'No training samples found in {self.cfg.train_file} with text_key="{self.cfg.text_key}".'
            )

        logger.info('[B4Distiller] loaded %d samples from %s', len(texts), self.cfg.train_file)
        raw_dataset = Dataset.from_list([{'text': t} for t in texts])

        if self.cfg.pack_texts:
            eos_token = tokenizer.eos_token or ''

            def tokenize_function(examples: dict[str, Any]) -> dict[str, Any]:
                merged = [
                    f'{text} {eos_token}'.strip() if eos_token else text
                    for text in examples['text']
                ]
                return tokenizer(merged, add_special_tokens=False)

            tokenized = raw_dataset.map(
                tokenize_function,
                batched=True,
                remove_columns=['text'],
                desc='Tokenizing distillation corpus',
            )

            block_size = int(self.cfg.block_size)

            def group_texts(examples: dict[str, list[list[int]]]) -> dict[str, list[list[int]]]:
                concatenated = {
                    key: [token for sample in examples[key] for token in sample]
                    for key in examples.keys()
                }
                total_length = len(concatenated['input_ids'])
                total_length = (total_length // block_size) * block_size
                result = {
                    key: [
                        value[idx : idx + block_size] for idx in range(0, total_length, block_size)
                    ]
                    for key, value in concatenated.items()
                }
                result['labels'] = [ids[:] for ids in result['input_ids']]
                return result

            return tokenized.map(
                group_texts,
                batched=True,
                desc='Packing distillation corpus',
            )

        def tokenize_function(examples: dict[str, Any]) -> dict[str, Any]:
            tokenized = tokenizer(
                examples['text'],
                truncation=True,
                max_length=self.cfg.block_size,
                padding='max_length',
            )
            tokenized['labels'] = [ids[:] for ids in tokenized['input_ids']]
            return tokenized

        return raw_dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=['text'],
            desc='Tokenizing distillation corpus',
        )

    def _build_training_args(self) -> TrainingArguments:
        _dtype, use_fp16, use_bf16 = self._resolve_precision_mode()
        candidate_kwargs = {
            'output_dir': self.cfg.output_dir,
            'overwrite_output_dir': True,
            'num_train_epochs': self.cfg.epochs,
            'per_device_train_batch_size': self.cfg.batch_size,
            'learning_rate': self.cfg.lr,
            'save_steps': self.cfg.save_steps,
            'save_total_limit': self.cfg.save_total_limit,
            'logging_steps': self.cfg.logging_steps,
            'gradient_accumulation_steps': self.cfg.gradient_accumulation_steps,
            'warmup_ratio': self.cfg.warmup_ratio,
            'weight_decay': self.cfg.weight_decay,
            'fp16': use_fp16,
            'bf16': use_bf16,
            'seed': self.cfg.seed,
            'report_to': 'none',
            'dataloader_num_workers': self.cfg.dataloader_num_workers,
            'dataloader_pin_memory': self.cfg.dataloader_pin_memory,
            'optim': self.cfg.optim,
            'tf32': True if self.cfg.tf32 is None else bool(self.cfg.tf32),
            'torch_compile': self.cfg.torch_compile,
        }
        supported_fields = getattr(TrainingArguments, '__dataclass_fields__', {})
        filtered_kwargs = {
            key: value for key, value in candidate_kwargs.items() if key in supported_fields
        }
        return TrainingArguments(**filtered_kwargs)

    def run(self) -> dict[str, Any]:
        if not os.path.exists(self.cfg.train_file):
            raise FileNotFoundError(f'B4 distillation corpus not found: {self.cfg.train_file}')
        os.makedirs(self.cfg.output_dir, exist_ok=True)

        model, tokenizer = self._load_tokenizer_and_model()
        train_dataset = self._build_train_dataset(tokenizer)
        train_args = self._build_training_args()

        trainer = Trainer(
            model=model,
            args=train_args,
            train_dataset=train_dataset,
            data_collator=default_data_collator,
        )

        logger.info('[B4Distiller] starting training (samples=%d)', len(train_dataset))
        train_result = trainer.train()
        trainer.save_model(self.cfg.output_dir)
        tokenizer.save_pretrained(self.cfg.output_dir)
        logger.info('[B4Distiller] saved distilled proxy to %s', self.cfg.output_dir)

        return {
            'output_dir': self.cfg.output_dir,
            'train_file': self.cfg.train_file,
            'proxy_model_name': self.cfg.proxy_model_name,
            'num_train_samples': int(len(train_dataset)),
            'train_loss': float(getattr(train_result, 'training_loss', 0.0)),
            'global_step': int(getattr(train_result, 'global_step', 0)),
            'pack_texts': bool(self.cfg.pack_texts),
            'attn_implementation': self._resolved_attention_impl,
            'precision': 'bf16' if train_args.bf16 else 'fp16' if train_args.fp16 else 'fp32',
        }

    def load_trained_proxy(self):
        if not os.path.exists(self.cfg.output_dir):
            raise FileNotFoundError(
                f'Distilled proxy output_dir does not exist: {self.cfg.output_dir}'
            )
        tokenizer = AutoTokenizer.from_pretrained(
            self.cfg.output_dir,
            trust_remote_code=self.cfg.trust_remote_code,
            local_files_only=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            self.cfg.output_dir,
            trust_remote_code=self.cfg.trust_remote_code,
            local_files_only=True,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return model, tokenizer
