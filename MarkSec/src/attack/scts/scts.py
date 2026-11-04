"""Self Color Testing-based Substitution (SCTS) attack."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from collections import Counter
from typing import Any, Iterable

import numpy as np
import torch
from nltk.tokenize import word_tokenize
from scipy import stats
from tqdm import tqdm
from transformers import pipeline

from src.attack.base import BaseAttack, BaseConfig
from src.attack.steal_corpus import infer_model_hint, infer_watermark_hint
from src.util.env import resolve_model_path
from src.util.text import strip_prompt

logger = logging.getLogger(__name__)


class SCTSConfig(BaseConfig):
    """Configuration for the SCTS attack."""

    def initialize_parameters(self) -> None:
        repeated_suffix = (
            '{context} {word1}; {context} {word2}; {context} {word1}; {context} {word2}; '
            '{context} {word2}; {context} {word1}; {context} {word2}; {context} {word2}; '
            '{context} {word2}; {context} {word2}; {context} {word1}; {context} {word2}; '
            '{context} {word2}; {context} {word2}; {context} {word1}'
        )
        default_template = (
            'Choose two phrases ( {context} {word1}, {context} {word2} ), and generate a long '
            'uniformly random string of those phrases separated by ";". Previous phrases '
            'should have no influence on future phrases: ' + repeated_suffix
        )

        self.temperature = float(self.cfg_dict.get('temperature', 0.0))
        self.context_size = int(self.cfg_dict.get('context_size', 1))
        self.max_candidates = int(self.cfg_dict.get('max_candidates', 5))
        self.query_top_k = int(self.cfg_dict.get('query_top_k', self.max_candidates * 2))
        self.step_size = int(self.cfg_dict.get('step_size', 2))
        self.mask_context_window = int(self.cfg_dict.get('mask_context_window', 5))
        self.query_max_new_tokens = int(self.cfg_dict.get('query_max_new_tokens', 128))
        self.max_substitutions_ratio = float(self.cfg_dict.get('max_substitutions_ratio', 0.1))
        self.chi_square_threshold = float(self.cfg_dict.get('chi_square_threshold', 0.01))
        self.candidate_generator_model = str(
            self.cfg_dict.get('candidate_generator_model', 'distilroberta-base')
        )
        self.prompt_template = self.cfg_dict.get('prompt_template', default_template)

    @property
    def algorithm_name(self) -> str:
        return 'SCTS'


class SCTSAttack(BaseAttack):
    """Minimal viable SCTS implementation aligned with the paper's workflow."""

    def __init__(
        self,
        config: SCTSConfig,
        base_model: Any | None = None,
        tokenizer: Any | None = None,
        watermark_instance: Any | None = None,
        candidate_generator: Any | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(watermark_model=base_model, watermark_tokenizer=tokenizer, *args, **kwargs)
        self.cfg = config
        self.base_model = base_model
        self.base_tokenizer = tokenizer
        self.watermark_instance = watermark_instance
        self.candidate_generator = candidate_generator

        self.cache_file = ''
        self.color_cache: dict[str, list[Any]] = {}
        self._attack_identity: dict[str, Any] = {}
        self._attack_identity_digest = ''
        self._prepared = False

    @classmethod
    def _normalize_identity_value(cls, value: Any) -> Any:
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, (list, tuple)):
            return [cls._normalize_identity_value(item) for item in value]
        if isinstance(value, dict):
            return {
                str(key): cls._normalize_identity_value(val)
                for key, val in sorted(value.items(), key=lambda item: str(item[0]))
            }
        return str(value)

    @classmethod
    def _hash_identity_payload(cls, payload: dict[str, Any]) -> str:
        normalized = cls._normalize_identity_value(payload)
        return hashlib.sha1(
            json.dumps(normalized, sort_keys=True, separators=(',', ':')).encode('utf-8')
        ).hexdigest()

    def _build_watermark_identity_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            'watermark_hint': infer_watermark_hint(watermark_instance=self.watermark_instance)
        }
        if self.watermark_instance is None:
            return payload

        payload['watermark_class'] = type(self.watermark_instance).__name__
        watermark_config = getattr(self.watermark_instance, 'config', None)
        if watermark_config is None:
            return payload

        payload['watermark_config_class'] = type(watermark_config).__name__
        config_dict = getattr(watermark_config, 'config_dict', None)
        if isinstance(config_dict, dict) and config_dict:
            payload['watermark_config'] = self._normalize_identity_value(config_dict)
            return payload

        if hasattr(watermark_config, '__dict__'):
            values = {
                key: value
                for key, value in vars(watermark_config).items()
                if not str(key).startswith('_')
            }
            if values:
                payload['watermark_config'] = self._normalize_identity_value(values)
        return payload

    def _build_attack_identity(self) -> dict[str, Any]:
        return {
            'attack': 'SCTS',
            'query_backend_mode': 'watermark' if self.watermark_instance is not None else 'plain',
            'query_backend_class': self._query_backend_id(),
            'query_model_hint': infer_model_hint(
                model=self.base_model, tokenizer=self.base_tokenizer
            ),
            'watermark_identity': self._build_watermark_identity_payload(),
            'prompt_template': str(self.cfg.prompt_template),
            'temperature': float(self.cfg.temperature),
            'query_max_new_tokens': int(self.cfg.query_max_new_tokens),
            'chi_square_threshold': float(self.cfg.chi_square_threshold),
        }

    def _refresh_cache_identity(self) -> None:
        self._attack_identity = self._build_attack_identity()
        self._attack_identity_digest = self._hash_identity_payload(self._attack_identity)

    def _resolve_cache_path(self) -> str:
        digest = self._attack_identity_digest[:12] or 'unknown'
        return os.path.join(self.get_attack_cache_dir(), f'color_cache_{digest}.json')

    def _resolve_candidate_generator_model(self) -> str:
        configured = self.cfg.candidate_generator_model.strip()
        if not configured:
            configured = 'distilbert/distilroberta-base'
        resolved = resolve_model_path(configured)
        if resolved and os.path.exists(resolved):
            return resolved
        return configured

    def _ensure_candidate_generator(self) -> None:
        if self.candidate_generator is not None:
            return
        model_name = self._resolve_candidate_generator_model()
        logger.info('SCTS loading candidate generator from %s', model_name)
        self.candidate_generator = pipeline(
            'fill-mask',
            model=model_name,
            tokenizer=model_name,
            device=0 if torch.cuda.is_available() else -1,
        )

    def _has_query_backend(self) -> bool:
        return self.watermark_instance is not None or (
            self.base_model is not None and self.base_tokenizer is not None
        )

    def _query_backend_id(self) -> str:
        if self.watermark_instance is not None:
            return self.watermark_instance.__class__.__name__
        return self.base_model.__class__.__name__ if self.base_model is not None else 'unknown'

    def _get_cache_key(self, context_text: str, word1: str, word2: str) -> str:
        if not self._attack_identity_digest:
            self._refresh_cache_identity()
        key_payload = {
            'attack_identity_digest': self._attack_identity_digest,
            'context_text': self._normalize_phrase(context_text),
            'word1': str(word1).strip().lower(),
            'word2': str(word2).strip().lower(),
        }
        return hashlib.sha1(
            json.dumps(key_payload, sort_keys=True, separators=(',', ':')).encode('utf-8')
        ).hexdigest()

    def _load_cache(self) -> None:
        if not self.cache_file or not os.path.exists(self.cache_file):
            self.color_cache = {}
            return
        try:
            with open(self.cache_file, 'r', encoding='utf-8') as f:
                payload = json.load(f)
            if isinstance(payload, dict) and isinstance(payload.get('entries'), dict):
                entries_obj = payload['entries']
            elif isinstance(payload, dict):
                entries_obj = payload
            else:
                entries_obj = {}
            self.color_cache = {
                str(key): value
                for key, value in entries_obj.items()
                if isinstance(key, str) and isinstance(value, list)
            }
            logger.info('SCTS loaded %d cached color tests.', len(self.color_cache))
        except Exception as e:
            logger.warning('Failed to load SCTS cache %s: %s', self.cache_file, e)
            self.color_cache = {}

    def _save_cache(self) -> None:
        if not self.cache_file:
            return
        os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)
        try:
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                payload = {
                    'version': 2,
                    'attack_identity_digest': self._attack_identity_digest,
                    'attack_identity': self._normalize_identity_value(self._attack_identity),
                    'entries': self.color_cache,
                }
                json.dump(payload, f, indent=2, ensure_ascii=True)
        except Exception as e:
            logger.warning('Failed to save SCTS cache %s: %s', self.cache_file, e)

    def prepare(self) -> bool:
        if not self._has_query_backend():
            return False
        self._ensure_candidate_generator()
        self._refresh_cache_identity()
        self.cache_file = self._resolve_cache_path()
        self._load_cache()
        self._prepared = True
        return True

    @staticmethod
    def _tokenize_to_device(
        tokenizer: Any, text: str, device: Any, **kwargs: Any
    ) -> dict[str, Any]:
        batch = tokenizer(text, **kwargs)
        if hasattr(batch, 'to'):
            batch = batch.to(device)
            return dict(batch)
        if isinstance(batch, dict):
            moved = {}
            for key, value in batch.items():
                moved[key] = value.to(device) if hasattr(value, 'to') else value
            return moved
        raise TypeError(f'Unsupported tokenizer output type: {type(batch)}')

    def _query_generation_kwargs(self) -> dict[str, Any]:
        return {
            'temperature': self.cfg.temperature,
            'do_sample': self.cfg.temperature > 0,
            'max_new_tokens': self.cfg.query_max_new_tokens,
        }

    def _generate_plain(self, prompt: str) -> str:
        if self.base_model is None or self.base_tokenizer is None:
            return ''
        inputs = self._tokenize_to_device(
            self.base_tokenizer,
            prompt,
            self.base_model.device,
            return_tensors='pt',
        )
        input_ids = inputs['input_ids']
        with torch.inference_mode():
            outputs = self.base_model.generate(
                **inputs, max_new_tokens=self.cfg.query_max_new_tokens
            )
        completion_ids = outputs[:, input_ids.shape[1] :]
        completion = self.base_tokenizer.batch_decode(completion_ids, skip_special_tokens=True)[
            0
        ].strip()
        if completion:
            return completion
        full = self.base_tokenizer.decode(outputs[0], skip_special_tokens=True)
        return strip_prompt(full, prompt).strip()

    def _generate_test_text(self, prompt: str) -> str:
        if self.watermark_instance is None:
            return self._generate_plain(prompt)

        query_kwargs = self._query_generation_kwargs()
        config = getattr(self.watermark_instance, 'config', None)
        gen_kwargs = getattr(config, 'gen_kwargs', None)
        original_gen_kwargs = gen_kwargs.copy() if isinstance(gen_kwargs, dict) else None

        try:
            if isinstance(gen_kwargs, dict):
                gen_kwargs.update(query_kwargs)
            try:
                return self.watermark_instance.generate_watermarked_text(
                    prompt=prompt, **query_kwargs
                )
            except TypeError:
                return self.watermark_instance.generate_watermarked_text(prompt=prompt)
        finally:
            if isinstance(gen_kwargs, dict) and original_gen_kwargs is not None:
                gen_kwargs.clear()
                gen_kwargs.update(original_gen_kwargs)

    @staticmethod
    def _normalize_phrase(text: str) -> str:
        return ' '.join(text.strip().split())

    def _build_prompt(
        self, context_words: Iterable[str], word1: str, word2: str
    ) -> tuple[str, str]:
        context_text = self._normalize_phrase(' '.join(context_words))
        prompt = self.cfg.prompt_template.format(
            context=context_text,
            context_word=context_text,
            word1=word1,
            word2=word2,
        )
        return context_text, prompt

    def _get_word_candidates(self, word: str, context: str = '') -> list[str]:
        if self.candidate_generator is None:
            return []

        tokenizer = getattr(self.candidate_generator, 'tokenizer', None)
        mask_token = getattr(tokenizer, 'mask_token', '<mask>')
        if context and word in context:
            masked_text = context.replace(word, mask_token, 1)
        else:
            masked_text = f'The {mask_token} is important.'

        try:
            predictions = self.candidate_generator(masked_text, top_k=self.cfg.query_top_k)
        except Exception as e:
            logger.warning('SCTS candidate generation failed: %s', e)
            return []

        if isinstance(predictions, dict):
            predictions = [predictions]

        candidates: list[str] = []
        for pred in predictions:
            candidate = str(pred.get('token_str', '')).strip()
            if (
                candidate
                and candidate.lower() != word.lower()
                and candidate.isalpha()
                and len(candidate) > 1
                and candidate not in candidates
            ):
                candidates.append(candidate)
            if len(candidates) >= self.cfg.max_candidates:
                break
        return candidates

    def _test_color(
        self, context_words: Iterable[str], word1: str, word2: str
    ) -> tuple[str, float]:
        context_text, prompt = self._build_prompt(context_words, word1, word2)
        cache_key = self._get_cache_key(context_text, word1, word2)
        if cache_key in self.color_cache:
            cached = self.color_cache[cache_key]
            return str(cached[0]), float(cached[1])

        try:
            text = self._generate_test_text(prompt)
        except Exception as e:
            logger.warning('SCTS query generation failed: %s', e)
            return 'S', 1.0

        generated_only = strip_prompt(str(text), prompt).strip()
        phrases = [
            self._normalize_phrase(part) for part in generated_only.split(';') if part.strip()
        ]
        counts = Counter(phrases)

        phrase1 = self._normalize_phrase(f'{context_text} {word1}')
        phrase2 = self._normalize_phrase(f'{context_text} {word2}')
        cb = counts[phrase1]
        cb_prime = counts[phrase2]
        total_counts = cb + cb_prime

        if total_counts == 0:
            result = ('S', 1.0)
        else:
            observed = np.array([cb, cb_prime], dtype=np.float64)
            expected = np.array([total_counts / 2, total_counts / 2], dtype=np.float64)
            _, p_value = stats.chisquare(observed, f_exp=expected)
            if p_value >= self.cfg.chi_square_threshold:
                result = ('S', float(p_value))
            elif cb > cb_prime:
                result = ('GR', float(abs(cb - cb_prime)))
            else:
                result = ('RG', float(abs(cb - cb_prime)))

        self.color_cache[cache_key] = [result[0], result[1]]
        return result

    @staticmethod
    def _tokenize_words(text: str) -> list[str]:
        try:
            return word_tokenize(text)
        except LookupError:
            return re.findall(r'\w+|[^\w\s]', text, flags=re.UNICODE)

    def edit(self, text: str, prompt: str | None = None, *args: Any, **kwargs: Any) -> str:
        attack_type = str(kwargs.get('attack_type') or '').strip().lower()
        if attack_type in {'spoof', 'spoofing'}:
            raise ValueError(
                '[SCTS] Spoofing protocol is not supported. '
                'SCTS currently only implements steal-plus-scrub substitution.'
            )

        if not self._prepared and not self.prepare():
            raise RuntimeError('SCTSAttack is not prepared. Call prepare() before edit().')

        words = self._tokenize_words(text)
        if len(words) <= self.cfg.context_size:
            return text

        max_subs = int(len(words) * self.cfg.max_substitutions_ratio)
        substitutions_made = 0
        total_positions = max(0, len(words) - self.cfg.context_size)
        progress = 0
        idx = self.cfg.context_size

        with tqdm(total=total_positions, desc='SCTS Attack', unit='word', leave=False) as pbar:
            while idx < len(words):
                if substitutions_made >= max_subs:
                    break

                target_word = words[idx]
                if not target_word.isalpha():
                    idx += 1
                    if progress < total_positions:
                        pbar.update(1)
                        progress += 1
                    continue

                context_words = words[idx - self.cfg.context_size : idx]
                local_start = max(0, idx - self.cfg.mask_context_window)
                local_end = min(len(words), idx + self.cfg.mask_context_window + 1)
                candidate_context = ' '.join(words[local_start:local_end])
                candidates = self._get_word_candidates(target_word, context=candidate_context)

                substituted = False
                for candidate in candidates:
                    try:
                        color_rel, _ = self._test_color(context_words, target_word, candidate)
                    except Exception:
                        continue
                    if color_rel == 'GR':
                        words[idx] = candidate
                        substitutions_made += 1
                        advance = max(1, self.cfg.step_size)
                        idx += advance
                        if progress < total_positions:
                            delta = min(advance, total_positions - progress)
                            pbar.update(delta)
                            progress += delta
                        substituted = True
                        break

                if substituted:
                    continue

                idx += 1
                if progress < total_positions:
                    pbar.update(1)
                    progress += 1

        self._save_cache()
        return ' '.join(words)

    def save_learned_data(self) -> None:
        self._save_cache()

    def load_learned_model(self) -> None:
        self._load_cache()
