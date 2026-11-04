from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    LogitsProcessor,
    LogitsProcessorList,
)

from src.attack.base import BaseAttack, BaseConfig
from src.attack.b4.solver import optimize_distribution
from src.attack.steal_corpus import (
    discover_joint_corpus_path,
    infer_dataset_hint,
    infer_model_hint,
    infer_watermark_hint,
    requires_exact_watermark_match,
)
from src.util.env import resolve_model_path
from src.util.text import strip_prompt

logger = logging.getLogger('MarkSec')


def _to_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {'1', 'true', 'yes', 'y', 'on'}


class B4ContraProcessor(LogitsProcessor):
    """
    Contra-style logits processor for ablation:
    - beam-aware top-k masking
    - indices_to_keep / indices_to_remove logic
    - optional closed-form optimization with epsilon constraint
    """

    def __init__(
        self,
        proxy_model,
        ref_model,
        proxy_device: str,
        ref_device: str,
        lambda_solver: str,
        epsilon: float,
        fixed_coef: float,
        num_beams: int,
        topk: int,
        use_aea: bool,
        aea_mu: float,
    ) -> None:
        self.proxy_model = proxy_model
        self.ref_model = ref_model
        self.proxy_device = proxy_device
        self.ref_device = ref_device

        self.lambda_solver = lambda_solver
        self.epsilon = float(epsilon)
        self.fixed_coef = float(max(fixed_coef, 0.0))
        self.num_beams = int(max(num_beams, 1))
        self.topk = int(max(topk, 1))
        self.use_aea = bool(use_aea and ref_model is not None)
        self.aea_mu = float(aea_mu)

        self.step_count = 0
        self.keep_ratio_acc = 0.0
        self.remove_ratio_acc = 0.0
        self.alpha_acc = 0.0
        self.lambda_acc = 0.0

    @staticmethod
    def _alpha_to_lambda(alpha: float) -> float:
        if alpha <= 0:
            return 0.0
        lam = alpha / (1.0 + alpha)
        if lam >= 1.0:
            return 0.95
        return max(lam, 0.0)

    @staticmethod
    def _match_vocab_dim(tensor: torch.Tensor, target_vocab: int) -> torch.Tensor:
        current = tensor.size(-1)
        if current == target_vocab:
            return tensor
        if current > target_vocab:
            return tensor[..., :target_vocab]
        pad = torch.full(
            (*tensor.shape[:-1], target_vocab - current),
            float('-inf'),
            device=tensor.device,
            dtype=tensor.dtype,
        )
        return torch.cat([tensor, pad], dim=-1)

    def _compute_keep_remove_masks(
        self,
        scores: torch.Tensor,
        proxy_scores: torch.Tensor,
        ref_scores: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        vocab_size = max(int(scores.shape[-1]), 1)
        uniform_prob = 1.0 / vocab_size

        if self.use_aea and ref_scores is not None:
            diff = torch.abs(torch.exp(proxy_scores) - torch.exp(ref_scores))
            indices_to_keep = diff < uniform_prob
            if self.aea_mu > 0:
                indices_to_keep |= diff < self.aea_mu
        else:
            indices_to_keep = torch.zeros_like(scores, dtype=torch.bool)

        topk_total = min(vocab_size, self.topk * self.num_beams)
        topk_vals = torch.topk(scores, topk_total, dim=-1).values

        remove_rank = max(1, (self.topk - 2) * self.num_beams)
        remove_rank = min(remove_rank, topk_total)
        keep_rank = 1  # last value among top-k set

        remove_threshold = topk_vals[..., -remove_rank, None]
        keep_threshold = topk_vals[..., -keep_rank, None]

        indices_to_remove = scores < remove_threshold
        indices_to_keep |= scores < keep_threshold
        return indices_to_keep, indices_to_remove

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        with torch.inference_mode():
            proxy_logits = self.proxy_model(input_ids.to(self.proxy_device)).logits[:, -1, :]
            proxy_logits = self._match_vocab_dim(proxy_logits, scores.size(-1))
            proxy_scores = F.log_softmax(proxy_logits, dim=-1).to(scores.device)

            ref_scores = None
            if self.use_aea and self.ref_model is not None:
                ref_logits = self.ref_model(input_ids.to(self.ref_device)).logits[:, -1, :]
                ref_logits = self._match_vocab_dim(ref_logits, scores.size(-1))
                ref_scores = F.log_softmax(ref_logits, dim=-1).to(scores.device)

            indices_to_keep, indices_to_remove = self._compute_keep_remove_masks(
                scores=scores,
                proxy_scores=proxy_scores,
                ref_scores=ref_scores,
            )
            backup_scores = scores.clone()

            if self.lambda_solver == 'newton':
                p_scores = F.log_softmax(scores, dim=-1)
                q_scores = F.log_softmax(proxy_scores, dim=-1)
                valid_batch_mask = ~(indices_to_keep.all(dim=-1))

                if valid_batch_mask.any():
                    alpha, h_scores = optimize_distribution(
                        p_scores[valid_batch_mask],
                        q_scores[valid_batch_mask],
                        self.epsilon,
                    )
                    new_scores = torch.full_like(scores, float('-inf'))
                    new_scores[valid_batch_mask] = h_scores
                    alpha_value = (
                        float(alpha.mean().item())
                        if torch.is_tensor(alpha) and alpha.numel() > 0
                        else 0.0
                    )
                else:
                    new_scores = backup_scores.clone()
                    alpha_value = 0.0
            else:
                local_proxy = proxy_scores.clone()
                local_proxy[local_proxy == float('-inf')] = float('inf')
                new_scores = (1.0 + self.fixed_coef) * scores - self.fixed_coef * local_proxy
                alpha_value = float(self.fixed_coef)

            new_scores = torch.where(indices_to_keep, backup_scores, new_scores)
            new_scores = new_scores.masked_fill(indices_to_remove, float('-inf'))
            all_inf_mask = torch.isneginf(new_scores).all(dim=-1, keepdim=True)
            if all_inf_mask.any():
                new_scores = torch.where(all_inf_mask, backup_scores, new_scores)
            new_scores = F.log_softmax(new_scores, dim=-1)

            self.step_count += 1
            self.keep_ratio_acc += float(indices_to_keep.float().mean().item())
            self.remove_ratio_acc += float(indices_to_remove.float().mean().item())
            self.alpha_acc += float(alpha_value)
            self.lambda_acc += float(self._alpha_to_lambda(alpha_value))
            return new_scores

    def summary(self) -> dict[str, float]:
        if self.step_count <= 0:
            lam = self._alpha_to_lambda(self.fixed_coef)
            return {
                'contra_steps': 0.0,
                'contra_keep_ratio': 0.0,
                'contra_remove_ratio': 0.0,
                'alpha': float(self.fixed_coef),
                'lambda': float(lam),
            }
        denom = float(self.step_count)
        return {
            'contra_steps': float(self.step_count),
            'contra_keep_ratio': float(self.keep_ratio_acc / denom),
            'contra_remove_ratio': float(self.remove_ratio_acc / denom),
            'alpha': float(self.alpha_acc / denom),
            'lambda': float(self.lambda_acc / denom),
        }


class B4Config(BaseConfig):
    """Configuration for B4 attack with backward-compatible parsing."""

    def initialize_parameters(self) -> None:
        self.model_requirements = dict(self.cfg_dict.get('model_requirements', {}))

        # attack mode
        mode = str(self.cfg_dict.get('mode', 'scrub')).strip().lower()
        self.mode = mode if mode in {'scrub', 'spoof'} else 'scrub'

        # stealing stage (query + proxy prep)
        stealing_cfg = dict(self.cfg_dict.get('stealing', {}))
        legacy_steal_samples = self.cfg_dict.get('stealing_samples', 0)
        self.stealing_enabled = _to_bool(stealing_cfg.get('enabled', False))
        self.stealing_num_queries = int(stealing_cfg.get('num_queries', legacy_steal_samples or 0))
        self.stealing_batch_size = int(stealing_cfg.get('batch_size', 8))
        self.prompt_source = str(stealing_cfg.get('prompt_source', 'dataset'))
        self.save_corpus_path = str(stealing_cfg.get('save_corpus_path', '')).strip()
        self.load_proxy_path = str(stealing_cfg.get('load_proxy_path', '')).strip()
        self.use_prebuilt_corpus = _to_bool(stealing_cfg.get('use_prebuilt_corpus', True))
        self.auto_discover_corpus = _to_bool(stealing_cfg.get('auto_discover_corpus', True))
        self.corpus_dir = str(stealing_cfg.get('corpus_dir', 'output/corpus')).strip()
        self.corpus_path = str(stealing_cfg.get('corpus_path', '')).strip()
        self.corpus_base_path = str(stealing_cfg.get('corpus_base_path', '')).strip()
        self.corpus_wm_path = str(stealing_cfg.get('corpus_wm_path', '')).strip()
        self.corpus_prompt_field = str(stealing_cfg.get('corpus_prompt_field', 'prompt'))
        self.corpus_base_field = str(stealing_cfg.get('corpus_base_field', 'base_completion'))
        self.corpus_wm_field = str(stealing_cfg.get('corpus_wm_field', 'wm_completion'))
        self.save_prepare_artifact = _to_bool(stealing_cfg.get('save_prepare_artifact', True))
        self.prepare_artifact_path = str(stealing_cfg.get('prepare_artifact_path', '')).strip()

        # optimization (proxy fusion)
        optimization_cfg = dict(self.cfg_dict.get('optimization', {}))
        legacy_coef = float(self.cfg_dict.get('coef', 1.0))
        default_lambda = legacy_coef / (1.0 + legacy_coef) if legacy_coef > 0 else 0.0
        self.epsilon = float(
            optimization_cfg.get(
                'epsilon',
                self.cfg_dict.get('kl_threshold', 0.0),
            )
        )
        self.lambda_solver = str(optimization_cfg.get('lambda_solver', 'fixed')).strip().lower()
        self.lambda_init = float(optimization_cfg.get('lambda_init', default_lambda))
        self.lambda_value = float(optimization_cfg.get('lambda_value', default_lambda))

        # AEA
        aea_cfg = dict(self.cfg_dict.get('aea', {}))
        self.aea_enabled = _to_bool(aea_cfg.get('enabled', False))
        self.aea_mu = float(aea_cfg.get('mu', 1e-5))

        # decoding
        generation_cfg = dict(self.cfg_dict.get('generation', {}))
        self.max_new_tokens = int(
            generation_cfg.get('max_new_tokens', self.cfg_dict.get('max_new_tokens', 256))
        )
        self.do_sample = _to_bool(generation_cfg.get('do_sample', False))
        self.temperature = float(generation_cfg.get('temperature', 0.8))
        self.top_p = float(generation_cfg.get('top_p', 0.9))
        self.num_beams = int(generation_cfg.get('num_beams', self.cfg_dict.get('num_beams', 1)))
        self.min_new_tokens = int(generation_cfg.get('min_new_tokens', 0))
        decoding_impl = str(generation_cfg.get('decoding_impl', 'step')).strip().lower()
        self.decoding_impl = decoding_impl if decoding_impl in {'step', 'contra_beam'} else 'step'
        self.contra_topk = int(generation_cfg.get('contra_topk', 10))
        if self.contra_topk < 1:
            self.contra_topk = 1

        # prompt templates
        legacy_prompt = str(
            self.cfg_dict.get(
                'prompt_template',
                'Generate text that preserves meaning and emulates desired style.\nText:\n{text}',
            )
        )
        self.scrub_prompt_template = str(
            self.cfg_dict.get(
                'scrub_prompt_template',
                (
                    'Paraphrase the following text while preserving semantics and fluency. '
                    'Avoid copying n-grams when possible.\nText:\n{text}'
                ),
            )
        )
        self.spoof_prompt_template = str(
            self.cfg_dict.get(
                'spoof_prompt_template',
                (
                    'Write a natural response for the following prompt in your own wording. '
                    'Keep style coherent and concise.\nPrompt:\n{prompt}'
                ),
            )
        )
        # Backward compatibility: if only prompt_template is provided, reuse it for spoof.
        if 'spoof_prompt_template' not in self.cfg_dict:
            self.spoof_prompt_template = legacy_prompt

    @property
    def algorithm_name(self) -> str:
        return 'B4'


class B4Attack(BaseAttack):
    """
    B4 implementation for MarkSec:
    - prepare(): optional stealing/query collection + proxy readiness checks
    - edit(): fused decoding with fidelity model + proxy watermark model
    """

    def __init__(
        self,
        config: B4Config,
        base_model=None,
        tokenizer=None,
        dataset=None,
        watermark_instance=None,
        attack_model=None,
        attack_tokenizer=None,
        proxy_model=None,
        proxy_tokenizer=None,
        ref_model=None,
        ref_tokenizer=None,
        watermark_name: str = '',
        **kwargs,
    ):
        super().__init__(watermark_model=base_model, watermark_tokenizer=tokenizer)
        self.cfg = config

        # fixed role naming: attack / proxy / ref
        self.attack_model = attack_model if attack_model is not None else base_model
        self.attack_tokenizer = attack_tokenizer if attack_tokenizer is not None else tokenizer

        self.proxy_model = proxy_model
        self.proxy_tokenizer = proxy_tokenizer

        self.ref_model = ref_model
        self.ref_tokenizer = ref_tokenizer if ref_tokenizer is not None else self.proxy_tokenizer

        self.dataset = dataset
        self.watermark_instance = watermark_instance
        self.watermark_name = watermark_name

        if self.attack_model is None or self.attack_tokenizer is None:
            raise ValueError('B4 requires at least attack model/tokenizer or base model/tokenizer.')

        self.attack_device = self._infer_model_device(self.attack_model, fallback='cpu')
        self.proxy_device = self._infer_model_device(self.proxy_model, fallback=self.attack_device)
        self.ref_device = self._infer_model_device(self.ref_model, fallback=self.proxy_device)

        self._prepared = False
        self._prepare_meta: dict[str, Any] = {}
        self._last_attack_meta: dict[str, Any] = {}
        self._explicit_corpus_path = bool(str(self.cfg.save_corpus_path or '').strip())
        self._explicit_prepare_artifact_path = bool(
            str(self.cfg.prepare_artifact_path or '').strip()
        )
        self._auto_proxy_path_discovered = False
        self._artifact_identity: dict[str, Any] = {}
        self._artifact_identity_digest = ''
        self._attack_cache_dir = self.get_attack_cache_dir()
        self._prepare_artifact_path = ''
        self._refresh_prepare_paths()

    @staticmethod
    def _infer_model_device(model: Any, fallback: str = 'cpu') -> str:
        if model is None:
            return fallback
        model_device = getattr(model, 'device', None)
        if model_device is not None:
            return str(model_device)
        try:
            first_param = next(model.parameters())
            return str(first_param.device)
        except Exception:
            return fallback

    def _resolve_corpus_path(self) -> str:
        if self.cfg.save_corpus_path:
            return os.path.abspath(self.cfg.save_corpus_path)
        return os.path.join(self._attack_cache_dir, 'stealing_corpus.jsonl')

    def _resolve_prepare_artifact_path(self) -> str:
        if self.cfg.prepare_artifact_path:
            return os.path.abspath(self.cfg.prepare_artifact_path)
        return os.path.join(self._attack_cache_dir, 'prepare_artifact.json')

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
        return hashlib.sha1(
            json.dumps(
                cls._normalize_identity_value(payload),
                sort_keys=True,
                separators=(',', ':'),
            ).encode('utf-8')
        ).hexdigest()

    @staticmethod
    def _sanitize_artifact_component(value: str) -> str:
        collapsed = ''.join(ch if ch.isalnum() or ch in '._-' else '_' for ch in str(value).strip())
        return collapsed[:80].strip('._-') or 'unknown'

    @staticmethod
    def _resolve_identity_path(path: str | None) -> str:
        if not path:
            return ''
        try:
            return str(Path(path).expanduser().resolve())
        except OSError:
            return os.path.abspath(str(path))

    @staticmethod
    def _fingerprint_path(path: str) -> str:
        if not path:
            return ''
        try:
            resolved = Path(path).resolve()
            stat = resolved.stat()
        except OSError:
            return ''
        material = f'{resolved}:{stat.st_size}:{stat.st_mtime_ns}'
        return hashlib.sha256(material.encode('utf-8')).hexdigest()[:12]

    @staticmethod
    def _normalize_match_text(value: str) -> str:
        return ''.join(ch for ch in str(value).lower() if ch.isalnum())

    @classmethod
    def _build_model_aliases(cls, value: str) -> set[str]:
        raw = str(value or '').strip()
        if not raw:
            return set()
        candidates = {raw, os.path.basename(raw.rstrip('/'))}
        aliases: set[str] = set()
        for candidate in candidates:
            normalized = cls._normalize_match_text(candidate)
            if normalized:
                aliases.add(normalized)
                for suffix in ('instruct', 'chat'):
                    if normalized.endswith(suffix) and len(normalized) > len(suffix):
                        aliases.add(normalized[: -len(suffix)])
                    stripped = normalized.replace(suffix, '')
                    if stripped:
                        aliases.add(stripped)
        return {alias for alias in aliases if alias}

    @staticmethod
    def _hint_segments(value: str) -> list[str]:
        return [segment for segment in re.split(r'[^a-z0-9.]+', str(value).lower()) if segment]

    def _candidate_proxy_model_aliases(self) -> set[str]:
        aliases: set[str] = set()
        for hint in (
            infer_model_hint(model=self.attack_model, tokenizer=self.attack_tokenizer),
            infer_model_hint(model=self.watermark_model, tokenizer=self.watermark_tokenizer),
        ):
            aliases.update(self._build_model_aliases(hint))
        return aliases

    @staticmethod
    def _is_complete_proxy_dir(path: Path) -> bool:
        if not path.is_dir():
            return False
        has_config = (path / 'config.json').exists()
        has_tokenizer = (path / 'tokenizer.json').exists() or (
            path / 'tokenizer_config.json'
        ).exists()
        has_weights = (path / 'model.safetensors').exists() or (path / 'pytorch_model.bin').exists()
        return has_config and has_tokenizer and has_weights

    def _build_watermark_identity_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            'watermark_hint': infer_watermark_hint(
                watermark_name=self.watermark_name,
                watermark_instance=self.watermark_instance,
            )
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

    def _build_artifact_identity(self) -> dict[str, Any]:
        corpus_path = self._resolve_identity_path(self.cfg.corpus_path)
        corpus_base_path = self._resolve_identity_path(self.cfg.corpus_base_path)
        corpus_wm_path = self._resolve_identity_path(self.cfg.corpus_wm_path)
        save_corpus_path = self._resolve_identity_path(self.cfg.save_corpus_path)
        return {
            'attack': 'B4',
            'mode': str(self.cfg.mode),
            'attack_model_hint': infer_model_hint(
                model=self.attack_model,
                tokenizer=self.attack_tokenizer,
            ),
            'watermark_name': str(self.watermark_name or ''),
            'watermark_identity': self._build_watermark_identity_payload(),
            'stealing_enabled': bool(self.cfg.stealing_enabled),
            'stealing_num_queries': int(self.cfg.stealing_num_queries),
            'use_prebuilt_corpus': bool(self.cfg.use_prebuilt_corpus),
            'corpus_path': corpus_path,
            'corpus_base_path': corpus_base_path,
            'corpus_wm_path': corpus_wm_path,
            'save_corpus_path': save_corpus_path,
            'load_proxy_path': self._resolve_identity_path(self.cfg.load_proxy_path),
            'corpus_path_fingerprint': self._fingerprint_path(corpus_path),
            'corpus_base_path_fingerprint': self._fingerprint_path(corpus_base_path),
            'corpus_wm_path_fingerprint': self._fingerprint_path(corpus_wm_path),
        }

    def _build_default_cache_dir(self) -> str:
        model_tag = self._sanitize_artifact_component(
            infer_model_hint(model=self.attack_model, tokenizer=self.attack_tokenizer)
            or 'unknown_model'
        )
        watermark_tag = self._sanitize_artifact_component(
            infer_watermark_hint(
                watermark_name=self.watermark_name,
                watermark_instance=self.watermark_instance,
            )
            or 'unknown_watermark'
        )
        mode_tag = self._sanitize_artifact_component(self.cfg.mode)
        digest_tag = self._artifact_identity_digest[:12] or 'unknown'
        return os.path.join(
            self.get_attack_cache_dir(), f'{model_tag}_{watermark_tag}_{mode_tag}_{digest_tag}'
        )

    def _refresh_prepare_paths(self) -> None:
        self._artifact_identity = self._build_artifact_identity()
        self._artifact_identity_digest = self._hash_identity_payload(self._artifact_identity)
        self._attack_cache_dir = self._build_default_cache_dir()
        self._prepare_artifact_path = self._resolve_prepare_artifact_path()

    def _resolve_proxy_path(self) -> str:
        if not self.cfg.load_proxy_path:
            return ''
        resolved = resolve_model_path(self.cfg.load_proxy_path)
        return resolved if resolved else self.cfg.load_proxy_path

    def _discover_local_proxy_path(self) -> str:
        models_root = Path(self.output_dir) / 'models'
        if not models_root.exists():
            return ''

        dataset_hint = infer_dataset_hint(self.dataset)
        dataset_norm = self._normalize_match_text(dataset_hint)
        watermark_hint = infer_watermark_hint(
            watermark_name=self.watermark_name,
            watermark_instance=self.watermark_instance,
        )
        watermark_segments = self._hint_segments(watermark_hint)
        model_aliases = self._candidate_proxy_model_aliases()
        require_exact_watermark = requires_exact_watermark_match(watermark_hint)

        # Auto-discovery is only safe when the proxy name encodes the full intended
        # model family, dataset, and watermark setting. If any of these hints are
        # unavailable, force an explicit proxy path instead of guessing.
        if not dataset_norm or not watermark_segments or not model_aliases:
            return ''

        exact_matches: list[tuple[int, float, Path]] = []
        for path in sorted(models_root.iterdir()):
            if not self._is_complete_proxy_dir(path):
                continue
            name = path.name.lower()
            if not name.startswith('b4_proxy'):
                continue

            name_norm = self._normalize_match_text(name)
            if require_exact_watermark and not all(
                segment in name for segment in watermark_segments
            ):
                continue
            if dataset_norm not in name_norm:
                continue
            if not any(alias in name_norm for alias in model_aliases):
                continue

            score = 0
            if all(segment in name for segment in watermark_segments):
                score += 100
            if dataset_norm in name_norm:
                score += 40
            if any(alias in name_norm for alias in model_aliases):
                score += 80
            if '50k' in name:
                score += 5
            exact_matches.append((score, path.stat().st_mtime, path))

        if not exact_matches:
            return ''

        exact_matches.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return str(exact_matches[0][2].resolve())

    def _ensure_proxy_model_from_path(self) -> None:
        proxy_path = self._resolve_proxy_path()
        if not proxy_path or self.proxy_model is not None:
            return

        if not os.path.exists(proxy_path):
            raise FileNotFoundError(f'B4 proxy path does not exist: {proxy_path}')

        logger.info('[B4] Loading proxy model from path: %s', proxy_path)
        self.proxy_tokenizer = AutoTokenizer.from_pretrained(
            proxy_path, local_files_only=True, trust_remote_code=True
        )
        self.proxy_model = AutoModelForCausalLM.from_pretrained(
            proxy_path, local_files_only=True, trust_remote_code=True
        ).to(self.attack_device)
        self.proxy_model.eval()
        self.proxy_device = self.attack_device
        self._prepare_meta['proxy_source'] = (
            'auto_discovered_path' if self._auto_proxy_path_discovered else 'path'
        )
        self._prepare_meta['proxy_path'] = proxy_path

    def _raise_missing_proxy_error(self, stage: str) -> None:
        model_hint = infer_model_hint(model=self.attack_model, tokenizer=self.attack_tokenizer)
        dataset_hint = infer_dataset_hint(self.dataset)
        watermark_hint = infer_watermark_hint(
            watermark_name=self.watermark_name,
            watermark_instance=self.watermark_instance,
        )
        proxy_path = self._resolve_proxy_path()
        if proxy_path:
            proxy_detail = f'Configured proxy path was unavailable or failed to load: {proxy_path}.'
        else:
            proxy_detail = (
                'No explicit proxy path was configured and no auto-discovered distilled proxy '
                'checkpoint with an exact model-family, dataset, and watermark-setting match '
                'was found.'
            )
        raise RuntimeError(
            '[B4] B4 requires a usable distilled proxy model for proxy-based decoding. '
            f'Cannot {stage} without a proxy. {proxy_detail} '
            f'model={model_hint or "unknown"} '
            f'dataset={dataset_hint or "unknown"} '
            f'watermark={watermark_hint or "unknown"}.'
        )

    def _require_proxy_ready(self, stage: str) -> None:
        if self.proxy_model is None:
            self._raise_missing_proxy_error(stage)

    def _collect_prompt(self, idx: int) -> str:
        if self.dataset is None:
            return f'Write a coherent paragraph about topic #{idx}.'

        if hasattr(self.dataset, 'get_prompt'):
            try:
                return str(self.dataset.get_prompt(idx))
            except Exception:
                pass

        try:
            sample = self.dataset[idx]
            if hasattr(sample, 'prompt'):
                return str(sample.prompt)
            if isinstance(sample, dict):
                return str(sample.get('prompt') or sample.get('text') or '')
            return str(sample)
        except Exception:
            return f'Write a coherent paragraph about topic #{idx}.'

    def _collect_stealing_corpus(self) -> int:
        if not self.cfg.stealing_enabled or self.cfg.stealing_num_queries <= 0:
            return 0
        if self.watermark_instance is None:
            logger.warning(
                '[B4] stealing.enabled=true but watermark_instance is missing. Skip corpus collection.'
            )
            return 0

        corpus_path = self._resolve_corpus_path()
        os.makedirs(os.path.dirname(corpus_path), exist_ok=True)
        written = 0

        logger.info(
            '[B4] Collecting stealing corpus to %s (%d queries).',
            corpus_path,
            self.cfg.stealing_num_queries,
        )
        with open(corpus_path, 'a', encoding='utf-8') as f:
            for idx in range(self.cfg.stealing_num_queries):
                prompt = self._collect_prompt(idx)
                try:
                    wm_full_text = self.watermark_instance.generate_watermarked_text(prompt=prompt)
                    wm_text = (
                        strip_prompt(wm_full_text, prompt).strip()
                        if isinstance(wm_full_text, str)
                        else ''
                    )
                except Exception as e:
                    logger.warning('[B4] failed to query watermark instance at idx=%d: %s', idx, e)
                    continue

                payload = {
                    'idx': idx,
                    'prompt': prompt,
                    'watermarked_text': wm_text,
                }
                f.write(json.dumps(payload, ensure_ascii=False) + '\n')
                written += 1

        self._prepare_meta['stealing_corpus_path'] = corpus_path
        self._prepare_meta['stealing_queries_collected'] = written
        self._prepare_meta['stealing_source'] = 'online_query'
        return written

    @staticmethod
    def _first_nonempty_string(record: dict[str, Any], keys: list[str]) -> str:
        for key in keys:
            value = record.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return ''

    def _resolve_prebuilt_corpus_path(self) -> str:
        if self.cfg.corpus_path:
            self.cfg.corpus_path = os.path.abspath(self.cfg.corpus_path)
            return self.cfg.corpus_path
        if not self.cfg.auto_discover_corpus:
            return ''

        watermark_hint = infer_watermark_hint(
            watermark_name=self.watermark_name,
            watermark_instance=self.watermark_instance,
        )
        model_hint = infer_model_hint(model=self.attack_model, tokenizer=self.attack_tokenizer)
        dataset_hint = infer_dataset_hint(self.dataset)

        discovered = discover_joint_corpus_path(
            corpus_dir=self.cfg.corpus_dir,
            model_hint=model_hint,
            dataset_hint=dataset_hint,
            watermark_hint=watermark_hint,
        )
        if discovered:
            self.cfg.corpus_path = os.path.abspath(discovered)
            logger.info('[B4] Auto-selected prebuilt corpus: %s', discovered)
            return self.cfg.corpus_path
        if watermark_hint and requires_exact_watermark_match(watermark_hint):
            raise FileNotFoundError(
                '[B4] No exact prebuilt corpus match found for '
                f'model={model_hint or "unknown"} dataset={dataset_hint or "unknown"} '
                f'watermark={watermark_hint}. Refusing to reuse a different setting.'
            )
        return ''

    def _materialize_prebuilt_corpus(self) -> int:
        if not self.cfg.use_prebuilt_corpus:
            return 0

        source_path = self._resolve_prebuilt_corpus_path()
        if not source_path:
            return 0
        if not os.path.exists(source_path):
            raise FileNotFoundError(f'B4 prebuilt corpus not found: {source_path}')

        output_path = self._resolve_corpus_path()
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        max_rows = self.cfg.stealing_num_queries if self.cfg.stealing_num_queries > 0 else None
        written = 0
        with (
            open(source_path, 'r', encoding='utf-8') as src,
            open(output_path, 'w', encoding='utf-8') as dst,
        ):
            for line_idx, line in enumerate(src):
                if max_rows is not None and written >= max_rows:
                    break
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except Exception:
                    continue

                prompt = self._first_nonempty_string(
                    record,
                    [self.cfg.corpus_prompt_field, 'prompt', 'source_prompt', 'input'],
                )
                base_completion = self._first_nonempty_string(
                    record,
                    [
                        self.cfg.corpus_base_field,
                        'base_completion',
                        'clean_text',
                        'natural_text',
                        'base_full_text',
                    ],
                )
                wm_completion = self._first_nonempty_string(
                    record,
                    [
                        self.cfg.corpus_wm_field,
                        'wm_completion',
                        'watermarked_text',
                        'generated_only',
                        'wm_full_text',
                    ],
                )
                if not wm_completion:
                    continue

                payload = {
                    'idx': int(record.get('idx', record.get('id', line_idx))),
                    'prompt': prompt,
                    'watermarked_text': wm_completion,
                    'base_completion': base_completion,
                    'wm_completion': wm_completion,
                    'source_corpus': source_path,
                }
                dst.write(json.dumps(payload, ensure_ascii=False) + '\n')
                written += 1

        self._prepare_meta['stealing_corpus_path'] = output_path
        self._prepare_meta['stealing_queries_collected'] = written
        self._prepare_meta['stealing_source'] = 'prebuilt_corpus'
        self._prepare_meta['source_corpus_path'] = source_path
        return written

    def _save_prepare_artifact(self) -> None:
        if not self.cfg.save_prepare_artifact:
            return
        payload = {
            'attack': 'B4',
            'meta_version': 'b4_prepare_meta_v2',
            'mode': self.cfg.mode,
            'stealing_source': self._prepare_meta.get('stealing_source', ''),
            'stealing_corpus_path': self._prepare_meta.get('stealing_corpus_path', ''),
            'source_corpus_path': self._prepare_meta.get('source_corpus_path', ''),
            'stealing_queries_collected': int(
                self._prepare_meta.get('stealing_queries_collected', 0)
            ),
            'proxy_source': self._prepare_meta.get('proxy_source', ''),
            'proxy_path': self._prepare_meta.get('proxy_path', ''),
            'identity_digest': self._artifact_identity_digest,
            'artifact_identity': self._normalize_identity_value(self._artifact_identity),
        }
        out_path = Path(self._prepare_artifact_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        logger.info('[B4] Prepare artifact saved to %s', out_path)

    def _validate_tokenizer_alignment(self) -> None:
        if self.proxy_tokenizer is None:
            self.proxy_tokenizer = self.attack_tokenizer

        attack_vocab = getattr(self.attack_tokenizer, 'vocab_size', None)
        proxy_vocab = getattr(self.proxy_tokenizer, 'vocab_size', None)
        if (
            isinstance(attack_vocab, int)
            and isinstance(proxy_vocab, int)
            and attack_vocab != proxy_vocab
        ):
            raise RuntimeError(
                f'B4 requires aligned token spaces. attack vocab={attack_vocab}, proxy vocab={proxy_vocab}.'
            )

    def _resolve_lambda(self) -> float:
        # `newton` is solved step-wise in `_fused_decode`; keep a bounded fallback here.
        if self.cfg.lambda_solver == 'fixed':
            lam = self.cfg.lambda_value
        else:
            lam = self.cfg.lambda_init
        lam = float(lam)
        if lam < 0:
            lam = 0.0
        if lam >= 1.0:
            lam = 0.95
        return lam

    @staticmethod
    def _alpha_to_lambda(alpha: float) -> float:
        # In B4 closed-form, alpha = lambda / (1 - lambda).
        alpha = float(alpha)
        if alpha <= 0:
            return 0.0
        lam = alpha / (1.0 + alpha)
        if lam >= 1.0:
            return 0.95
        return max(lam, 0.0)

    def _solve_closed_form_scores(
        self,
        log_pf: torch.Tensor,
        log_pw: torch.Tensor,
        keep_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, float]:
        # Inputs are 1D log-probs over vocab.
        p_scores = log_pf.unsqueeze(0)
        q_scores = log_pw.unsqueeze(0)

        if keep_mask is None:
            keep_mask = torch.zeros_like(log_pf, dtype=torch.bool)
        keep_mask_b = keep_mask.unsqueeze(0)

        # If all tokens are excluded by AEA keep-mask, fall back to fidelity only.
        if bool(keep_mask_b.all(dim=-1).item()):
            return log_pf, 0.0

        alpha, h_scores = optimize_distribution(p_scores, q_scores, self.cfg.epsilon)
        if torch.is_tensor(alpha):
            alpha_value = float(alpha.mean().item()) if alpha.numel() > 0 else 0.0
        else:
            alpha_value = float(alpha)

        merged_scores = torch.where(keep_mask_b, p_scores, h_scores)
        # Keep normalized log-prob for consistent downstream sampling behavior.
        merged_scores = F.log_softmax(merged_scores, dim=-1)
        return merged_scores[0], alpha_value

    def _encode_prompt_inputs(self, prompt_text: str) -> tuple[torch.Tensor, torch.Tensor]:
        if hasattr(self.attack_tokenizer, 'apply_chat_template'):
            try:
                messages = [{'role': 'user', 'content': prompt_text}]
                prompt_ids = self.attack_tokenizer.apply_chat_template(
                    messages,
                    return_tensors='pt',
                    add_generation_prompt=True,
                )
                if torch.is_tensor(prompt_ids):
                    prompt_ids = prompt_ids.to(self.attack_device)
                    return prompt_ids, torch.ones_like(prompt_ids, device=prompt_ids.device)
            except Exception:
                pass

        encoded = self.attack_tokenizer(prompt_text, return_tensors='pt')
        if isinstance(encoded, dict):
            prompt_ids = encoded.get('input_ids')
            attention_mask = encoded.get('attention_mask')
        else:
            prompt_ids = encoded.input_ids
            attention_mask = getattr(encoded, 'attention_mask', None)
        if not torch.is_tensor(prompt_ids):
            raise TypeError('B4 tokenizer did not return tensor input_ids.')
        prompt_ids = prompt_ids.to(self.attack_device)
        if torch.is_tensor(attention_mask):
            attention_mask = attention_mask.to(self.attack_device)
        else:
            attention_mask = torch.ones_like(prompt_ids, device=prompt_ids.device)
        return prompt_ids, attention_mask

    def _encode_prompt(self, prompt_text: str) -> torch.Tensor:
        prompt_ids, _ = self._encode_prompt_inputs(prompt_text)
        return prompt_ids

    def _top_p_filter(self, probs: torch.Tensor, top_p: float) -> torch.Tensor:
        if top_p >= 1.0:
            return probs

        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumulative = torch.cumsum(sorted_probs, dim=-1)
        mask = cumulative > top_p
        mask[..., 1:] = mask[..., :-1].clone()
        mask[..., 0] = False
        sorted_probs = sorted_probs.masked_fill(mask, 0.0)
        filtered = torch.zeros_like(probs)
        filtered.scatter_(0, sorted_indices, sorted_probs)
        return filtered

    def _sample_next_token(self, combined_scores: torch.Tensor, step: int) -> int:
        eos_token_id = getattr(self.attack_tokenizer, 'eos_token_id', None)
        scores = combined_scores.clone()

        if eos_token_id is not None and step < self.cfg.min_new_tokens:
            scores[eos_token_id] = -float('inf')

        if not self.cfg.do_sample:
            return int(torch.argmax(scores).item())

        sampling_temp = max(float(self.cfg.temperature), 1e-5)
        probs = F.softmax(scores / sampling_temp, dim=-1)
        probs = self._top_p_filter(probs, self.cfg.top_p)
        denom = probs.sum()
        if not torch.isfinite(denom) or float(denom.item()) <= 0:
            return int(torch.argmax(scores).item())
        probs = probs / denom

        token = torch.multinomial(probs, 1)
        return int(token.item())

    def _fused_decode_step(self, prompt_text: str, attack_mode: str) -> tuple[str, dict[str, Any]]:
        prompt_ids = self._encode_prompt(prompt_text)
        generated_ids: list[int] = []

        fixed_lam = self._resolve_lambda()
        denom = max(1.0 - fixed_lam, 1e-6)
        fixed_fidelity_weight = 1.0 / denom
        fixed_anti_wm_weight = fixed_lam / denom

        uncertain_ratio_acc = 0.0
        uncertain_steps = 0
        alpha_steps: list[float] = []
        lambda_steps: list[float] = []
        t0 = time.time()

        with torch.inference_mode():
            for step in range(self.cfg.max_new_tokens):
                if generated_ids:
                    add = torch.tensor([generated_ids], dtype=torch.long, device=self.attack_device)
                    attack_ids = torch.cat([prompt_ids, add], dim=1)
                else:
                    attack_ids = prompt_ids

                attack_logits = self.attack_model(attack_ids).logits[0, -1, :]
                log_pf = F.log_softmax(attack_logits, dim=-1)

                proxy_ids = attack_ids.to(self.proxy_device)
                proxy_logits = self.proxy_model(proxy_ids).logits[0, -1, :]
                log_pw = F.log_softmax(proxy_logits, dim=-1).to(log_pf.device)

                uncertain = None
                if self.cfg.aea_enabled and self.ref_model is not None:
                    ref_ids = attack_ids.to(self.ref_device)
                    ref_logits = self.ref_model(ref_ids).logits[0, -1, :]
                    log_pref = F.log_softmax(ref_logits, dim=-1).to(log_pf.device)
                    uncertain = (torch.exp(log_pw) - torch.exp(log_pref)).abs() < self.cfg.aea_mu
                    uncertain_ratio_acc += float(uncertain.float().mean().item())
                    uncertain_steps += 1

                if self.cfg.lambda_solver == 'newton':
                    combined, alpha_t = self._solve_closed_form_scores(
                        log_pf=log_pf,
                        log_pw=log_pw,
                        keep_mask=uncertain,
                    )
                    lam_t = self._alpha_to_lambda(alpha_t)
                    alpha_steps.append(alpha_t)
                    lambda_steps.append(lam_t)
                else:
                    combined = fixed_fidelity_weight * log_pf - fixed_anti_wm_weight * log_pw
                    if uncertain is not None:
                        combined = torch.where(uncertain, log_pf, combined)

                next_token_id = self._sample_next_token(combined, step=step)
                generated_ids.append(next_token_id)

                eos_token_id = getattr(self.attack_tokenizer, 'eos_token_id', None)
                if eos_token_id is not None and next_token_id == eos_token_id:
                    break

        if not generated_ids:
            decoded = ''
        else:
            decoded = self.attack_tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

        if lambda_steps:
            effective_lambda = float(sum(lambda_steps) / len(lambda_steps))
            effective_alpha = float(sum(alpha_steps) / len(alpha_steps))
            effective_denom = max(1.0 - effective_lambda, 1e-6)
            fidelity_weight = float(1.0 / effective_denom)
            anti_wm_weight = float(effective_lambda / effective_denom)
        else:
            effective_lambda = float(fixed_lam)
            effective_alpha = float(fixed_anti_wm_weight)
            fidelity_weight = float(fixed_fidelity_weight)
            anti_wm_weight = float(fixed_anti_wm_weight)

        meta = {
            'mode': attack_mode,
            'lambda': effective_lambda,
            'alpha': effective_alpha,
            'lambda_solver': self.cfg.lambda_solver,
            'decode_impl': 'step',
            'fidelity_weight': fidelity_weight,
            'anti_wm_weight': anti_wm_weight,
            'aea_enabled': bool(self.cfg.aea_enabled and self.ref_model is not None),
            'uncertain_token_ratio': float(uncertain_ratio_acc / uncertain_steps)
            if uncertain_steps
            else 0.0,
            'decode_steps': len(generated_ids),
            'decode_strategy': 'sample' if self.cfg.do_sample else 'greedy',
            'runtime_cost': float(time.time() - t0),
            'query_cost': float(self._prepare_meta.get('stealing_queries_collected', 0)),
        }
        return decoded, meta

    def _fused_decode_contra_beam(
        self, prompt_text: str, attack_mode: str
    ) -> tuple[str, dict[str, Any]]:
        prompt_ids, attention_mask = self._encode_prompt_inputs(prompt_text)
        prompt_len = int(prompt_ids.shape[1])

        beam_count = self.cfg.num_beams if self.cfg.num_beams > 1 else 2
        if beam_count != self.cfg.num_beams:
            logger.warning('[B4] contra_beam requires num_beams>1, overriding to 2.')

        fixed_lam = self._resolve_lambda()
        fixed_denom = max(1.0 - fixed_lam, 1e-6)
        fixed_coef = float(fixed_lam / fixed_denom)

        processor = B4ContraProcessor(
            proxy_model=self.proxy_model,
            ref_model=self.ref_model,
            proxy_device=self.proxy_device,
            ref_device=self.ref_device,
            lambda_solver=self.cfg.lambda_solver,
            epsilon=self.cfg.epsilon,
            fixed_coef=fixed_coef,
            num_beams=beam_count,
            topk=self.cfg.contra_topk,
            use_aea=self.cfg.aea_enabled,
            aea_mu=self.cfg.aea_mu,
        )

        eos_token_id = getattr(self.attack_tokenizer, 'eos_token_id', None)
        pad_token_id = getattr(self.attack_tokenizer, 'pad_token_id', None)
        if pad_token_id is None:
            pad_token_id = eos_token_id

        generate_kwargs: dict[str, Any] = {
            'max_new_tokens': self.cfg.max_new_tokens,
            'num_beams': beam_count,
            'do_sample': self.cfg.do_sample,
            'logits_processor': LogitsProcessorList([processor]),
            'pad_token_id': pad_token_id,
            'eos_token_id': eos_token_id,
        }
        if self.cfg.min_new_tokens > 0:
            generate_kwargs['min_new_tokens'] = self.cfg.min_new_tokens
        if self.cfg.do_sample:
            generate_kwargs['temperature'] = self.cfg.temperature
            generate_kwargs['top_p'] = self.cfg.top_p

        t0 = time.time()
        with torch.inference_mode():
            outputs = self.attack_model.generate(
                input_ids=prompt_ids,
                attention_mask=attention_mask,
                **generate_kwargs,
            )

        output_ids = outputs[0]
        completion_ids = output_ids[prompt_len:]
        decoded = self.attack_tokenizer.decode(completion_ids, skip_special_tokens=True).strip()

        contra_meta = processor.summary()
        effective_lambda = float(contra_meta.get('lambda', fixed_lam))
        effective_alpha = float(contra_meta.get('alpha', fixed_coef))
        effective_denom = max(1.0 - effective_lambda, 1e-6)
        fidelity_weight = float(1.0 / effective_denom)
        anti_wm_weight = float(effective_lambda / effective_denom)

        meta = {
            'mode': attack_mode,
            'lambda': effective_lambda,
            'alpha': effective_alpha,
            'lambda_solver': self.cfg.lambda_solver,
            'decode_impl': 'contra_beam',
            'fidelity_weight': fidelity_weight,
            'anti_wm_weight': anti_wm_weight,
            'aea_enabled': bool(self.cfg.aea_enabled and self.ref_model is not None),
            'uncertain_token_ratio': float(contra_meta.get('contra_keep_ratio', 0.0)),
            'decode_steps': int(contra_meta.get('contra_steps', 0.0)),
            'decode_strategy': 'beam_sample' if self.cfg.do_sample else 'beam',
            'contra_topk': int(self.cfg.contra_topk),
            'contra_keep_ratio': float(contra_meta.get('contra_keep_ratio', 0.0)),
            'contra_remove_ratio': float(contra_meta.get('contra_remove_ratio', 0.0)),
            'runtime_cost': float(time.time() - t0),
            'query_cost': float(self._prepare_meta.get('stealing_queries_collected', 0)),
        }
        return decoded, meta

    def _fused_decode(self, prompt_text: str, attack_mode: str) -> tuple[str, dict[str, Any]]:
        if self.cfg.decoding_impl == 'contra_beam':
            return self._fused_decode_contra_beam(prompt_text, attack_mode)
        return self._fused_decode_step(prompt_text, attack_mode)

    def prepare(self) -> bool:
        if self.cfg.use_prebuilt_corpus:
            self._resolve_prebuilt_corpus_path()
        if not self.cfg.load_proxy_path and self.proxy_model is None:
            discovered_proxy_path = self._discover_local_proxy_path()
            if discovered_proxy_path:
                self.cfg.load_proxy_path = discovered_proxy_path
                self._auto_proxy_path_discovered = True
                logger.info('[B4] Auto-selected distilled proxy model: %s', discovered_proxy_path)
        self._refresh_prepare_paths()

        self._ensure_proxy_model_from_path()
        self._require_proxy_ready(stage='prepare')
        written_from_prebuilt = self._materialize_prebuilt_corpus()
        if written_from_prebuilt <= 0:
            self._collect_stealing_corpus()

        if self.proxy_tokenizer is None:
            self.proxy_tokenizer = self.attack_tokenizer

        if hasattr(self.proxy_model, 'eval'):
            self.proxy_model.eval()
        if self.ref_model is not None and hasattr(self.ref_model, 'eval'):
            self.ref_model.eval()
        if hasattr(self.attack_model, 'eval'):
            self.attack_model.eval()

        self.proxy_device = self._infer_model_device(self.proxy_model, fallback=self.attack_device)
        self.ref_device = self._infer_model_device(self.ref_model, fallback=self.proxy_device)

        self._validate_tokenizer_alignment()
        self._save_prepare_artifact()
        self._prepared = True
        return True

    def edit(self, text: str, prompt: str = '', **kwargs) -> str:
        if not self._prepared:
            raise RuntimeError('B4Attack is not prepared. Call prepare() before edit().')
        self._require_proxy_ready(stage='edit')

        mode = self.cfg.mode
        attack_type = str(kwargs.get('attack_type', '')).strip().lower()
        if attack_type in {'scrub', 'scrubbing'}:
            mode = 'scrub'
        elif attack_type in {'spoof', 'spoofing'}:
            mode = 'spoof'

        if mode == 'spoof':
            source_prompt = str(prompt or kwargs.get('reference') or text).strip()
            if not source_prompt:
                return ''
            prompt_text = self.cfg.spoof_prompt_template.format(
                prompt=source_prompt, text=source_prompt
            )
            decoded, meta = self._fused_decode(prompt_text, attack_mode='spoof')
            self._last_attack_meta = {**self._prepare_meta, **meta}
            return decoded if decoded else source_prompt

        source_text = str(text or '').strip()
        if not source_text:
            source_text = str(kwargs.get('reference') or prompt).strip()
        if not source_text:
            return ''

        prompt_text = self.cfg.scrub_prompt_template.format(text=source_text, prompt=prompt or '')
        decoded, meta = self._fused_decode(prompt_text, attack_mode='scrub')
        self._last_attack_meta = {**self._prepare_meta, **meta}
        return decoded if decoded else source_text

    def pop_last_attack_meta(self) -> dict[str, Any]:
        meta = dict(self._last_attack_meta)
        self._last_attack_meta = {}
        return meta
