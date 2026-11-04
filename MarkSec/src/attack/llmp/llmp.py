from __future__ import annotations

from typing import Any

import torch

from src.attack.base import BaseAttack, BaseConfig
from src.util.text import slice_completion, strip_prompt


def _to_device(batch: dict[str, Any], device: str) -> dict[str, Any]:
    """Move tensor fields in a tokenized batch to the target device."""
    moved: dict[str, Any] = {}
    for key, value in batch.items():
        moved[key] = value.to(device) if torch.is_tensor(value) else value
    return moved


class LLMParaphraseConfig(BaseConfig):
    def __init__(self, config_path: str = None, **kwargs) -> None:
        self._algorithm_name = 'LLMP'
        super().__init__(config_path, **kwargs)

    @property
    def algorithm_name(self) -> str:
        return self._algorithm_name

    def initialize_parameters(self) -> None:
        self.model_requirements = dict(self.cfg_dict.get('model_requirements', {}))
        attack_model_override = self.cfg_dict.get('attack_model')
        if attack_model_override:
            self.model_requirements['attack'] = str(attack_model_override).strip()
        # Default to threat-model based surrogate selection.
        self.model_requirements.setdefault('attack', 'auto')

        self.generation_config: dict[str, Any] = dict(self.cfg_dict.get('generation_config', {}))
        self.generation_config.setdefault('max_new_tokens', 256)
        self.generation_config.setdefault('do_sample', True)
        self.generation_config.setdefault('temperature', 0.7)
        self.generation_config.setdefault('top_p', 0.9)

        op = self.cfg_dict.get('operational_parameters', {})
        preferred_device = str(op.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
        if preferred_device.startswith('cuda') and not torch.cuda.is_available():
            preferred_device = 'cpu'
        self.device: str = preferred_device

        self.prompt_template: str = self.cfg_dict.get(
            'prompt_template',
            (
                'Paraphrase the following text to remove any potential watermarks while '
                'preserving meaning and fluency. Do not add or omit information.\n'
                'Text:\n{text}'
            ),
        )


class LLMParaphraseAttack(BaseAttack):
    def __init__(
        self,
        config: LLMParaphraseConfig,
        attack_model=None,
        attack_tokenizer=None,
        base_model=None,
        tokenizer=None,
        **kwargs,
    ):
        super().__init__(watermark_model=base_model, watermark_tokenizer=tokenizer)
        self.cfg = config
        self.model = attack_model if attack_model is not None else base_model
        self.tokenizer = attack_tokenizer if attack_tokenizer is not None else tokenizer
        if self.model is None or self.tokenizer is None:
            raise ValueError('LLMP attack requires both model and tokenizer.')

        model_device = getattr(self.model, 'device', None)
        self.device = str(model_device) if model_device is not None else self.cfg.device
        if self.device.startswith('cuda') and not torch.cuda.is_available():
            self.device = 'cpu'

        if hasattr(self.model, 'to'):
            self.model = self.model.to(self.device)
        if hasattr(self.model, 'eval'):
            self.model.eval()

    def prepare(self) -> bool:
        return True

    def _tokenize_prompt(self, prompt_text: str) -> tuple[dict[str, Any], int]:
        if hasattr(self.tokenizer, 'apply_chat_template'):
            try:
                messages = [{'role': 'user', 'content': prompt_text}]
                chat_ids = self.tokenizer.apply_chat_template(
                    messages,
                    return_tensors='pt',
                    add_generation_prompt=True,
                )
                if torch.is_tensor(chat_ids):
                    return {'input_ids': chat_ids}, int(chat_ids.shape[-1])
            except Exception:
                pass

        encoded = self.tokenizer(prompt_text, return_tensors='pt')
        if not isinstance(encoded, dict):
            encoded = dict(encoded)

        input_ids = encoded.get('input_ids')
        input_len = int(input_ids.shape[-1]) if torch.is_tensor(input_ids) else 0
        return encoded, input_len

    def _generate(self, prompt_text: str) -> str:
        inputs, input_len = self._tokenize_prompt(prompt_text)
        inputs = _to_device(inputs, self.device)

        gen_kwargs = dict(self.cfg.generation_config)
        if 'pad_token_id' not in gen_kwargs:
            pad_token_id = getattr(self.tokenizer, 'pad_token_id', None)
            if pad_token_id is not None:
                gen_kwargs['pad_token_id'] = pad_token_id
        if 'eos_token_id' not in gen_kwargs:
            eos_token_id = getattr(self.tokenizer, 'eos_token_id', None)
            if eos_token_id is not None:
                gen_kwargs['eos_token_id'] = eos_token_id

        with torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_kwargs)

        if input_len > 0:
            completion = slice_completion(outputs, input_len, self.tokenizer)
            if completion:
                return completion

        if torch.is_tensor(outputs):
            decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        else:
            decoded = self.tokenizer.decode(outputs, skip_special_tokens=True)
        return strip_prompt(decoded, prompt_text).strip() or decoded.strip()

    def edit(self, text: str, prompt: str = '', **kwargs) -> str:
        source_text = (text or '').strip()
        if not source_text:
            source_text = str(kwargs.get('reference') or prompt or '').strip()
        if not source_text:
            return ''

        prompt_values = {'text': source_text, 'prompt': prompt or ''}
        try:
            prompt_text = self.cfg.prompt_template.format(**prompt_values)
        except KeyError as exc:
            missing_key = str(exc).strip("'")
            raise ValueError(
                f"LLMP prompt_template is missing key '{missing_key}'. "
                'Supported keys are {text} and {prompt}.'
            ) from exc

        rewritten = self._generate(prompt_text).strip()
        return rewritten if rewritten else source_text
