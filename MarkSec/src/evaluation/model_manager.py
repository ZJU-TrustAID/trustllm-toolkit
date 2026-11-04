import logging
import os
import gc
import torch
from typing import Any
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)
from src.util.env import resolve_model_path
from src.config import ExperimentConfig

logger = logging.getLogger('MarkSec')


class ModelManager:
    """
    Handles loading of models and tokenizers.
    """

    def __init__(self, device: str, config: ExperimentConfig):
        self.device = device
        self.config = config
        self.loaded_models = {}

    def clear_cache(self):
        """Clears the model cache and releases memory."""
        self.loaded_models.clear()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info('ModelManager cache cleared.')

    @staticmethod
    def _normalize_torch_dtype(dtype: Any) -> Any:
        if dtype is None:
            return torch.bfloat16
        if isinstance(dtype, torch.dtype):
            return dtype

        normalized = str(dtype).strip().lower()
        dtype_map = {
            'auto': 'auto',
            'float32': torch.float32,
            'fp32': torch.float32,
            'float16': torch.float16,
            'fp16': torch.float16,
            'half': torch.float16,
            'bfloat16': torch.bfloat16,
            'bf16': torch.bfloat16,
        }
        if normalized not in dtype_map:
            raise ValueError(
                f"Unsupported torch_dtype '{dtype}'. "
                'Use one of: auto, float32, fp32, float16, fp16, half, bfloat16, bf16.'
            )
        return dtype_map[normalized]

    def load_model_and_tokenizer(
        self,
        model_name: str,
        tokenizer_name: str | None = None,
        *,
        allow_model_parallel: bool = False,
        torch_dtype: Any = None,
    ):
        """Loads a model and tokenizer. Supports separate tokenizer path and optional model parallelism."""
        local_files = self.config.runner_settings.models_local_files_only

        # Resolve local paths when possible.
        load_path = resolve_model_path(model_name, models_root=self.config.models_root)
        tokenizer_source = tokenizer_name or model_name
        tokenizer_load_path = resolve_model_path(
            tokenizer_source, models_root=self.config.models_root
        )
        resolved_torch_dtype = self._normalize_torch_dtype(torch_dtype)
        cache_key = (
            load_path,
            tokenizer_load_path,
            bool(allow_model_parallel),
            str(resolved_torch_dtype),
        )

        if cache_key in self.loaded_models:
            logger.info(f'  -> Using cached model/tokenizer for: {load_path}')
            return self.loaded_models[cache_key]

        if load_path != model_name:
            logger.info(f'     Resolved model path: {load_path}')
        else:
            logger.info(f'     Using original model name/path: {load_path}')
        if tokenizer_load_path != load_path:
            logger.info(f'     Resolved tokenizer path: {tokenizer_load_path}')

        if local_files and not os.path.isdir(load_path) and not os.path.isfile(load_path):
            raise FileNotFoundError(
                f"Model path '{load_path}' not found locally, and 'models_local_files_only' is set to True. "
            )
        if (
            local_files
            and not os.path.isdir(tokenizer_load_path)
            and not os.path.isfile(tokenizer_load_path)
        ):
            raise FileNotFoundError(
                f"Tokenizer path '{tokenizer_load_path}' not found locally, and 'models_local_files_only' is set to True. "
            )

        logger.info(f'  -> Loading model: {load_path} (Device: {self.device})')

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_load_path, local_files_only=local_files)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
            logger.info(
                f"Tokenizer for '{tokenizer_source}' has no pad_token; set to eos_token: '{tokenizer.eos_token}'."
            )

        config = AutoConfig.from_pretrained(load_path, local_files_only=local_files)

        model_class = None
        if hasattr(config, 'architectures') and config.architectures:
            for arch in config.architectures:
                if 'ForSequenceClassification' in arch:
                    logger.info(
                        f"     '{model_name}' detected as a Sequence Classification model via architecture '{arch}'."
                    )
                    model_class = AutoModelForSequenceClassification
                    break
        if model_class is None:
            if config.is_encoder_decoder:
                logger.info(f"     '{model_name}' detected as a Sequence-to-Sequence model.")
                model_class = AutoModelForSeq2SeqLM
            else:
                logger.info(f"     '{model_name}' detected as a Causal LM (Decoder-Only) model.")
                model_class = AutoModelForCausalLM

        use_flash = self.config.runner_settings.use_flash_attention
        base_load_kwargs = {
            'config': config,
            'torch_dtype': resolved_torch_dtype,
            'local_files_only': local_files,
            'trust_remote_code': True,
        }
        if use_flash:
            attn_candidates = ['flash_attention_2', 'sdpa', None]
        else:
            attn_candidates = [None]

        last_error = None
        model = None
        resolved_attn_impl = None
        for attn_impl in attn_candidates:
            load_kwargs = dict(base_load_kwargs)
            if attn_impl is not None:
                load_kwargs['attn_implementation'] = attn_impl
            try:
                should_parallel = (
                    allow_model_parallel
                    and torch.cuda.is_available()
                    and torch.cuda.device_count() > 1
                    and str(self.device).startswith('cuda')
                )
                if should_parallel:
                    model = model_class.from_pretrained(
                        load_path,
                        device_map='auto',
                        **load_kwargs,
                    )
                else:
                    model = model_class.from_pretrained(load_path, **load_kwargs).to(self.device)
                resolved_attn_impl = attn_impl
                break
            except Exception as exc:
                last_error = exc
                logger.warning(
                    "     Failed to load '%s' with attn_implementation=%s: %s",
                    model_name,
                    attn_impl or 'default',
                    exc,
                )

        if model is None:
            if last_error is not None:
                raise last_error
            raise RuntimeError(f"Failed to load model '{model_name}'.")

        if use_flash:
            logger.info(
                '     Resolved attention implementation: %s',
                resolved_attn_impl or 'default',
            )

        if model.config.is_decoder and not model.config.is_encoder_decoder:
            tokenizer.padding_side = 'left'
            logger.info("     Set tokenizer.padding_side to 'left' for Decoder-only model.")

        model.config.pad_token_id = model.config.eos_token_id
        model.eval()

        self.loaded_models[cache_key] = (model, tokenizer)
        return model, tokenizer
