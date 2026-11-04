import hashlib
import json
import logging
import os
import random
import time
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F
from src.attack.base import BaseAttack, BaseConfig
from src.util.env import extract_full_model_id
from tqdm import tqdm

logger = logging.getLogger('MarkSec')


class DemarkConfig(BaseConfig):
    def __init__(self, config_path: str = None, **kwargs) -> None:
        self._algorithm_name = 'DEMARK'
        super().__init__(config_path, **kwargs)

    @property
    def algorithm_name(self) -> str:
        return self._algorithm_name

    def initialize_parameters(self) -> None:
        self.model_requirements = self.cfg_dict.get('model_requirements', {})
        self.generation_config: Dict = self.cfg_dict.get('generation_config', {})
        self.generation_config.setdefault('max_new_tokens', 100)
        self.generation_config.setdefault('do_sample', True)
        op = self.cfg_dict.get('operational_parameters', {})
        self.device = op.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.wm_type = self.cfg_dict.get('wm_type', 'dipmark')
        self.ctx_width = int(self.cfg_dict.get('ctx_width', 3))
        self.total_word_num = int(self.cfg_dict.get('total_word_num', 20))
        self.sample_strategy = self.cfg_dict.get('sample_strategy', 'top_k')
        self.probe_batch_size = int(self.cfg_dict.get('probe_batch_size', 32))
        self.delta_predicted = float(self.cfg_dict.get('delta_predicted', 2.0))
        self.delta_reduce_multiplier = float(self.cfg_dict.get('delta_reduce_multiplier', 1.0))
        self.delta_spoof = float(self.cfg_dict.get('delta_spoof', self.delta_predicted))
        self.delta_actual = float(self.cfg_dict.get('delta_actual', 2.0))
        self.dipmark_alpha = float(self.cfg_dict.get('dipmark_alpha', 0.3))
        self.mode = str(self.cfg_dict.get('mode', '') or '').strip().lower()
        self.spoof_prompt_template = str(
            self.cfg_dict.get(
                'spoof_prompt_template',
                (
                    'Write a natural response for the following prompt in your own wording. '
                    'Keep the answer fluent, coherent, and concise.\nPrompt:\n{prompt}'
                ),
            )
        )
        self.clear_cuda_cache_after_probe = bool(
            self.cfg_dict.get('clear_cuda_cache_after_probe', False)
        )
        probe_cache = self.cfg_dict.get('probe_cache', {})
        self.probe_cache_enabled = bool(probe_cache.get('enabled', True))
        self.probe_cache_persist = bool(probe_cache.get('persist', True))
        self.probe_cache_path = str(probe_cache.get('path', '') or '')
        self.probe_cache_max_entries = int(probe_cache.get('max_entries', 50000))
        self.probe_cache_save_interval = int(probe_cache.get('save_interval', 200))
        estimation = self.cfg_dict.get('parameter_estimation', {})
        self.parameter_estimation_enabled = bool(estimation.get('enabled', False))
        self.estimate_ctx_width = bool(
            estimation.get('estimate_ctx_width', self.parameter_estimation_enabled)
        )
        self.estimate_delta = bool(
            estimation.get('estimate_delta', self.parameter_estimation_enabled)
        )
        self.estimation_max_ctx_width = int(estimation.get('max_ctx_width', max(self.ctx_width, 5)))
        self.estimation_random_word_num = int(estimation.get('random_word_num', 50))
        self.estimation_repeat_time = int(estimation.get('repeat_time', 5))
        self.estimation_run_num = int(estimation.get('run_num', 3))
        self.estimation_alpha1 = float(estimation.get('alpha1', 0.2))
        self.estimation_alpha2 = float(estimation.get('alpha2', 10.0))
        self.estimation_gamma = float(estimation.get('gamma', 0.1))
        self.estimation_consistency_threshold = float(estimation.get('consistency_threshold', 0.95))
        self.estimation_blackbox = bool(estimation.get('blackbox', False))
        self.estimation_max_vocab_scan = int(estimation.get('max_vocab_scan', 8192))
        self.estimation_random_seed = int(estimation.get('random_seed', 42))


class DemarkAttack(BaseAttack):
    def __init__(
        self,
        config: DemarkConfig,
        base_model=None,
        tokenizer=None,
        watermark_instance=None,
        **kwargs,
    ):
        super().__init__(watermark_model=base_model, watermark_tokenizer=tokenizer, **kwargs)
        self.cfg = config
        self.model = base_model
        self.tokenizer = tokenizer
        self.device = getattr(self.model, 'device', self.cfg.device)
        self.model_id = extract_full_model_id(getattr(self.tokenizer, 'name_or_path', ''))

        if watermark_instance is None:
            raise ValueError('DemarkAttack requires a watermark_instance to act as the Oracle.')
        self.oracle = watermark_instance
        self._oracle_identity = self._build_oracle_identity_payload()
        self._oracle_identity_digest = self._hash_identity_payload(self._oracle_identity)
        self._probe_cache_path = self._resolve_probe_cache_path()
        self._maybe_migrate_legacy_probe_cache()
        self._probe_cache: dict[str, list[int]] = {}
        self._probe_cache_hits = 0
        self._probe_cache_misses = 0
        self._probe_requests_total = 0
        self._probe_live_queries_total = 0
        self._probe_cache_dirty = 0
        self._prepare_meta: dict[str, Any] = {}
        self._last_attack_meta: dict[str, Any] = {}
        self._load_probe_cache()

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

    def _build_oracle_identity_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {'oracle_class': type(self.oracle).__name__}
        oracle_config = getattr(self.oracle, 'config', None)
        if oracle_config is None:
            return payload

        payload['oracle_config_class'] = type(oracle_config).__name__
        config_dict = getattr(oracle_config, 'config_dict', None)
        if isinstance(config_dict, dict) and config_dict:
            payload['oracle_config'] = self._normalize_identity_value(config_dict)
            return payload

        identity_fields = (
            'algorithm_name',
            'gamma',
            'delta',
            'hash_key',
            'prefix_length',
            'f_scheme',
            'window_scheme',
            'z_threshold',
            'seeding_scheme',
        )
        for field_name in identity_fields:
            if not hasattr(oracle_config, field_name):
                continue
            value = getattr(oracle_config, field_name)
            if callable(value):
                continue
            payload[field_name] = self._normalize_identity_value(value)
        return payload

    def _resolve_probe_cache_path(self) -> str:
        if self.cfg.probe_cache_path:
            return os.path.abspath(self.cfg.probe_cache_path)
        model_tag = (self.model_id or 'unknown_model').replace('/', '_')
        oracle_tag = self._oracle_identity_digest[:12]
        filename = f'probe_cache_{self.cfg.wm_type.lower()}_{model_tag}_{oracle_tag}.json'
        return os.path.join(self.get_attack_cache_dir(), filename)

    def _resolve_legacy_probe_cache_path(self) -> str:
        model_tag = (self.model_id or 'unknown_model').replace('/', '_')
        filename = f'probe_cache_{self.cfg.wm_type.lower()}_{model_tag}.json'
        return os.path.join(self.get_attack_cache_dir(), filename)

    def _maybe_migrate_legacy_probe_cache(self) -> None:
        if self.cfg.probe_cache_path:
            return
        if os.path.exists(self._probe_cache_path):
            return

        legacy_path = self._resolve_legacy_probe_cache_path()
        if legacy_path == self._probe_cache_path or not os.path.exists(legacy_path):
            return

        os.makedirs(os.path.dirname(self._probe_cache_path), exist_ok=True)
        try:
            os.replace(legacy_path, self._probe_cache_path)
            logger.info(
                'DEMARK migrated legacy probe cache %s -> %s',
                legacy_path,
                self._probe_cache_path,
            )
        except Exception as exc:
            logger.warning(
                'Failed to migrate legacy DEMARK probe cache %s -> %s: %s',
                legacy_path,
                self._probe_cache_path,
                exc,
            )

    def _load_probe_cache(self) -> None:
        if not (self.cfg.probe_cache_enabled and self.cfg.probe_cache_persist):
            return
        if not os.path.exists(self._probe_cache_path):
            return
        try:
            with open(self._probe_cache_path, 'r', encoding='utf-8') as f:
                payload = json.load(f)
            if isinstance(payload, dict) and isinstance(payload.get('entries'), dict):
                entries_obj = payload['entries']
            elif isinstance(payload, dict):
                # Backward compatibility with flat dict payloads.
                entries_obj = payload
            else:
                entries_obj = {}
            loaded: dict[str, list[int]] = {}
            for key, value in entries_obj.items():
                if not isinstance(key, str) or not isinstance(value, list):
                    continue
                try:
                    loaded[key] = [int(token_id) for token_id in value]
                except (TypeError, ValueError):
                    continue
            self._probe_cache = loaded
            logger.info(
                'DEMARK probe cache loaded: %d entries from %s',
                len(self._probe_cache),
                self._probe_cache_path,
            )
        except Exception as exc:
            logger.warning('Failed to load DEMARK probe cache %s: %s', self._probe_cache_path, exc)
            self._probe_cache = {}

    def _save_probe_cache(self, force: bool = False) -> None:
        if not (self.cfg.probe_cache_enabled and self.cfg.probe_cache_persist):
            return
        if not force and self._probe_cache_dirty <= 0:
            return
        os.makedirs(os.path.dirname(self._probe_cache_path), exist_ok=True)
        payload = {
            'version': 2,
            'entry_count': len(self._probe_cache),
            'oracle_identity_digest': self._oracle_identity_digest,
            'oracle_identity': self._oracle_identity,
            'entries': self._probe_cache,
        }
        try:
            with open(self._probe_cache_path, 'w', encoding='utf-8') as f:
                json.dump(payload, f, ensure_ascii=True)
            self._probe_cache_dirty = 0
        except Exception as exc:
            logger.warning('Failed to save DEMARK probe cache %s: %s', self._probe_cache_path, exc)

    def _build_probe_cache_key(
        self,
        prompt_ids: torch.Tensor,
        possible_token_ids: torch.Tensor | list[int] | None,
    ) -> str:
        context_ids = (
            prompt_ids[0, -self.cfg.ctx_width :].detach().cpu().tolist()
            if self.cfg.ctx_width > 0
            else []
        )
        if possible_token_ids is None:
            token_ids: list[int] = []
        elif isinstance(possible_token_ids, torch.Tensor):
            token_ids = [int(token_id) for token_id in possible_token_ids.detach().cpu().tolist()]
        else:
            token_ids = [int(token_id) for token_id in possible_token_ids]
        token_ids = sorted(set(token_ids))
        key_payload = {
            'wm_type': self.cfg.wm_type.lower(),
            'model_id': self.model_id,
            'oracle_identity_digest': self._oracle_identity_digest,
            'ctx_width': int(self.cfg.ctx_width),
            'sample_strategy': self.cfg.sample_strategy,
            'total_word_num': int(self.cfg.total_word_num),
            'context_ids': context_ids,
            'token_ids': token_ids,
        }
        digest = hashlib.sha1(
            json.dumps(key_payload, sort_keys=True, separators=(',', ':')).encode('utf-8')
        ).hexdigest()
        return digest

    def _cache_probe_result(self, key: str, detected_green_token_ids: list[int]) -> None:
        if key in self._probe_cache:
            return
        if self.cfg.probe_cache_max_entries > 0:
            while len(self._probe_cache) >= self.cfg.probe_cache_max_entries:
                oldest_key = next(iter(self._probe_cache))
                self._probe_cache.pop(oldest_key, None)
        self._probe_cache[key] = [int(token_id) for token_id in detected_green_token_ids]
        self._probe_cache_dirty += 1
        if self._probe_cache_dirty >= max(1, self.cfg.probe_cache_save_interval):
            self._save_probe_cache()

    def _detect_red_green_list_kgw_cached(
        self,
        prompt_ids: torch.Tensor,
        wm_logits: torch.Tensor,
        possible_token_ids: torch.Tensor | list[int] | None,
        vocab_size: int,
    ) -> list[int]:
        self._probe_requests_total += 1
        if not self.cfg.probe_cache_enabled:
            self._probe_live_queries_total += 1
            detected, _, _ = detect_red_green_list_KGW(
                self.model_id,
                prompt_ids,
                self.cfg.ctx_width,
                total_word_num=self.cfg.total_word_num,
                model=self.model,
                tokenizer=self.tokenizer,
                oracle=self.oracle,
                delta_predicted=self.cfg.delta_predicted,
                sample_strategy=self.cfg.sample_strategy,
                print_all=False,
                predefined_possible_token_ids=possible_token_ids,
                probe_batch_size=self.cfg.probe_batch_size,
                precomputed_wm_logits=wm_logits,
                clear_cuda_cache_after_probe=self.cfg.clear_cuda_cache_after_probe,
                vocab_size=vocab_size,
            )
            return [int(token_id) for token_id in detected]

        cache_key = self._build_probe_cache_key(prompt_ids, possible_token_ids)
        cached = self._probe_cache.get(cache_key)
        if cached is not None:
            self._probe_cache_hits += 1
            logger.debug(
                'DEMARK probe cache hit: path=%s key=%s hits=%d',
                self._probe_cache_path,
                cache_key[:12],
                self._probe_cache_hits,
            )
            return [int(token_id) for token_id in cached]

        self._probe_cache_misses += 1
        self._probe_live_queries_total += 1
        detected, _, _ = detect_red_green_list_KGW(
            self.model_id,
            prompt_ids,
            self.cfg.ctx_width,
            total_word_num=self.cfg.total_word_num,
            model=self.model,
            tokenizer=self.tokenizer,
            oracle=self.oracle,
            delta_predicted=self.cfg.delta_predicted,
            sample_strategy=self.cfg.sample_strategy,
            print_all=False,
            predefined_possible_token_ids=possible_token_ids,
            probe_batch_size=self.cfg.probe_batch_size,
            precomputed_wm_logits=wm_logits,
            clear_cuda_cache_after_probe=self.cfg.clear_cuda_cache_after_probe,
            vocab_size=vocab_size,
        )
        detected_green_token_ids = [int(token_id) for token_id in detected]
        self._cache_probe_result(cache_key, detected_green_token_ids)
        return detected_green_token_ids

    def _build_prepare_meta(self, prepare_runtime_cost: float) -> dict[str, Any]:
        artifact_path = self._probe_cache_path if self.cfg.probe_cache_persist else ''
        return {
            'attack': 'DEMARK',
            'wm_type': str(self.cfg.wm_type).lower(),
            'prepare_runtime_cost': float(prepare_runtime_cost),
            'parameter_estimation_enabled': bool(self.cfg.parameter_estimation_enabled),
            'parameter_estimation_ctx_width_enabled': bool(self.cfg.estimate_ctx_width),
            'parameter_estimation_delta_enabled': bool(self.cfg.estimate_delta),
            'probe_cache_enabled': bool(self.cfg.probe_cache_enabled),
            'probe_cache_persist': bool(self.cfg.probe_cache_persist),
            'probe_cache_path': artifact_path,
            'artifact_type': 'probe_cache' if artifact_path else '',
            'artifact_path': artifact_path,
            'probe_cache_entry_count_prepare': int(len(self._probe_cache)),
            'probe_batch_size': int(self.cfg.probe_batch_size),
            'probe_strategy': str(self.cfg.sample_strategy),
            'ctx_width': int(self.cfg.ctx_width),
            'delta_predicted': float(self.cfg.delta_predicted),
            'delta_spoof': float(self.cfg.delta_spoof),
        }

    def _capture_edit_attack_meta(
        self,
        *,
        runtime_cost: float,
        attack_mode: str,
        probe_requests_before: int,
        probe_live_queries_before: int,
        probe_cache_hits_before: int,
        probe_cache_misses_before: int,
        probe_cache_entries_before: int,
        generated_token_count: int | None,
    ) -> None:
        probe_requests = max(0, self._probe_requests_total - probe_requests_before)
        probe_queries = max(0, self._probe_live_queries_total - probe_live_queries_before)
        probe_cache_hits = max(0, self._probe_cache_hits - probe_cache_hits_before)
        probe_cache_misses = max(0, self._probe_cache_misses - probe_cache_misses_before)
        probe_cache_entries_after = len(self._probe_cache)
        meta = {
            **self._prepare_meta,
            'attack': 'DEMARK',
            'mode': str(attack_mode),
            'runtime_cost': float(runtime_cost),
            'query_cost': float(probe_queries),
            'probe_requests': int(probe_requests),
            'probe_queries': int(probe_queries),
            'probe_cache_hits': int(probe_cache_hits),
            'probe_cache_misses': int(probe_cache_misses),
            'probe_cache_used': bool(probe_cache_hits > 0),
            'probe_cache_hit_rate': float(probe_cache_hits / probe_requests)
            if probe_requests > 0
            else 0.0,
            'probe_cache_entry_count_before': int(probe_cache_entries_before),
            'probe_cache_entry_count_after': int(probe_cache_entries_after),
            'probe_cache_entries_added': int(
                max(0, probe_cache_entries_after - probe_cache_entries_before)
            ),
            'generated_token_count': None
            if generated_token_count is None
            else int(generated_token_count),
        }
        self._last_attack_meta = meta

    def _resolve_attack_mode(self, *, text: str, attack_type: str) -> str:
        normalized_type = str(attack_type or '').strip().lower()
        if normalized_type in {'spoof', 'spoofing'}:
            return 'spoof'
        if normalized_type in {'scrub', 'scrubbing', 'removal'}:
            return 'scrub'

        configured_mode = str(getattr(self.cfg, 'mode', '') or '').strip().lower()
        if configured_mode in {'spoof', 'scrub'}:
            return configured_mode
        return 'scrub' if str(text or '').strip() else 'spoof'

    def _build_generation_prompt(
        self,
        *,
        attack_mode: str,
        text: str,
        prompt: str,
        reference: str,
    ) -> str:
        if attack_mode == 'spoof':
            source_prompt = str(prompt or reference or text).strip()
            if not source_prompt:
                return ''
            reference_text = str(reference or source_prompt)
            return self.cfg.spoof_prompt_template.format(
                prompt=source_prompt,
                text=source_prompt,
                reference=reference_text,
            )
        return str(prompt or text or reference).strip()

    def _apply_detected_token_bias(
        self,
        wm_logits: torch.Tensor,
        detected_green_token_ids: list[int],
        *,
        attack_mode: str,
        delta: float,
    ) -> None:
        if not detected_green_token_ids:
            return
        bias = float(delta)
        if attack_mode != 'spoof':
            bias = -bias * float(self.cfg.delta_reduce_multiplier)
        wm_logits[detected_green_token_ids] += bias

    def prepare(self) -> bool:
        prepare_t0 = time.time()
        if self.model is not None and hasattr(self.model, 'eval'):
            self.model.eval()
        if not self.cfg.parameter_estimation_enabled:
            self._prepare_meta = self._build_prepare_meta(time.time() - prepare_t0)
            return True

        if self.cfg.wm_type.lower() != 'kgw':
            logger.warning(
                'DEMARK parameter estimation currently supports KGW only; '
                "skipping for wm_type='%s'.",
                self.cfg.wm_type,
            )
            self._prepare_meta = self._build_prepare_meta(time.time() - prepare_t0)
            return True

        estimated_ctx_width = None
        same_rate_dict: dict[int, float] = {}
        if self.cfg.estimate_ctx_width:
            try:
                estimated_ctx_width, same_rate_dict = estimate_ctx_width_KGW(
                    model=self.model,
                    model_id=self.model_id,
                    tokenizer=self.tokenizer,
                    oracle=self.oracle,
                    max_ctx_width=self.cfg.estimation_max_ctx_width,
                    random_word_num=self.cfg.estimation_random_word_num,
                    alpha1=self.cfg.estimation_alpha1,
                    alpha2=self.cfg.estimation_alpha2,
                    consistency_threshold=self.cfg.estimation_consistency_threshold,
                    blackbox=self.cfg.estimation_blackbox,
                    probe_batch_size=self.cfg.probe_batch_size,
                    random_seed=self.cfg.estimation_random_seed,
                    max_vocab_scan=self.cfg.estimation_max_vocab_scan,
                    clear_cuda_cache_after_probe=self.cfg.clear_cuda_cache_after_probe,
                )
                logger.info(
                    'DEMARK estimated ctx_width=%s from same_rate=%s',
                    estimated_ctx_width,
                    same_rate_dict,
                )
                self.cfg.ctx_width = estimated_ctx_width
            except Exception as exc:
                logger.warning(
                    'DEMARK ctx_width estimation failed; keeping configured ctx_width=%s. Error: %s',
                    self.cfg.ctx_width,
                    exc,
                )

        estimated_delta = None
        if self.cfg.estimate_delta:
            try:
                estimated_delta = estimate_delta_KGW(
                    model=self.model,
                    model_id=self.model_id,
                    tokenizer=self.tokenizer,
                    oracle=self.oracle,
                    ctx_width=self.cfg.ctx_width,
                    random_word_num=self.cfg.estimation_random_word_num,
                    repeat_time=self.cfg.estimation_repeat_time,
                    run_num=self.cfg.estimation_run_num,
                    alpha1=self.cfg.estimation_alpha1,
                    alpha2=self.cfg.estimation_alpha2,
                    gamma=self.cfg.estimation_gamma,
                    blackbox=self.cfg.estimation_blackbox,
                    probe_batch_size=self.cfg.probe_batch_size,
                    random_seed=self.cfg.estimation_random_seed,
                    max_vocab_scan=self.cfg.estimation_max_vocab_scan,
                    clear_cuda_cache_after_probe=self.cfg.clear_cuda_cache_after_probe,
                )
                if estimated_delta is not None:
                    logger.info('DEMARK estimated delta_predicted=%.4f', estimated_delta)
                    self.cfg.delta_predicted = estimated_delta
                else:
                    logger.warning(
                        'DEMARK delta estimation produced no high-confidence token pairs; '
                        'keeping configured delta_predicted=%s.',
                        self.cfg.delta_predicted,
                    )
            except Exception as exc:
                logger.warning(
                    'DEMARK delta estimation failed; keeping configured delta_predicted=%s. '
                    'Error: %s',
                    self.cfg.delta_predicted,
                    exc,
                )
        self._prepare_meta = self._build_prepare_meta(time.time() - prepare_t0)
        if estimated_ctx_width is not None:
            self._prepare_meta['estimated_ctx_width'] = int(estimated_ctx_width)
        if same_rate_dict:
            self._prepare_meta['estimated_ctx_width_same_rates'] = {
                str(width): float(rate) for width, rate in same_rate_dict.items()
            }
        if estimated_delta is not None:
            self._prepare_meta['estimated_delta_predicted'] = float(estimated_delta)
        return True

    def edit(self, text: str, prompt: str, **kwargs) -> str:
        attack_t0 = time.time()
        probe_requests_before = self._probe_requests_total
        probe_live_queries_before = self._probe_live_queries_total
        probe_cache_hits_before = self._probe_cache_hits
        probe_cache_misses_before = self._probe_cache_misses
        probe_cache_entries_before = len(self._probe_cache)
        attack_mode = self._resolve_attack_mode(
            text=text,
            attack_type=str(kwargs.get('attack_type') or ''),
        )
        prompt_text = self._build_generation_prompt(
            attack_mode=attack_mode,
            text=text,
            prompt=prompt,
            reference=str(kwargs.get('reference') or ''),
        )
        if not prompt_text:
            self._capture_edit_attack_meta(
                runtime_cost=time.time() - attack_t0,
                attack_mode=attack_mode,
                probe_requests_before=probe_requests_before,
                probe_live_queries_before=probe_live_queries_before,
                probe_cache_hits_before=probe_cache_hits_before,
                probe_cache_misses_before=probe_cache_misses_before,
                probe_cache_entries_before=probe_cache_entries_before,
                generated_token_count=0,
            )
            return ''
        inputs = self.tokenizer(prompt_text, return_tensors='pt').to(self.device)
        prompt_ids = inputs['input_ids']
        # Determine watermark type from config or oracle
        # For now, rely on config as before, but we could inspect self.oracle
        if self.cfg.wm_type.lower() == 'kgw':
            attacked_text = self._generate_kgw(prompt_text, prompt_ids, attack_mode=attack_mode)
        else:
            attacked_text = self._generate_dipmark(prompt_text, prompt_ids, attack_mode=attack_mode)
        if self._probe_cache_dirty > 0:
            self._save_probe_cache()
        generated_token_count = None
        if self.tokenizer is not None and hasattr(self.tokenizer, '__call__'):
            try:
                generated_token_count = len(
                    _extract_input_ids(
                        self.tokenizer(attacked_text, add_special_tokens=False, return_tensors='pt')
                    )
                )
            except Exception:
                generated_token_count = None
        self._capture_edit_attack_meta(
            runtime_cost=time.time() - attack_t0,
            attack_mode=attack_mode,
            probe_requests_before=probe_requests_before,
            probe_live_queries_before=probe_live_queries_before,
            probe_cache_hits_before=probe_cache_hits_before,
            probe_cache_misses_before=probe_cache_misses_before,
            probe_cache_entries_before=probe_cache_entries_before,
            generated_token_count=generated_token_count,
        )
        if attacked_text:
            return attacked_text
        if attack_mode == 'scrub':
            return str(text or '')
        return ''

    def _generate_kgw(
        self,
        raw_prompt: str,
        prompt_ids: torch.Tensor,
        attack_mode: str = 'scrub',
    ) -> str:
        generated_token_ids: List[int] = []
        for _ in tqdm(range(self.cfg.generation_config['max_new_tokens']), desc='KGW Generation'):
            new_prompt_ids = torch.cat(
                [
                    prompt_ids,
                    torch.tensor(generated_token_ids, dtype=torch.long, device=self.device).view(
                        1, -1
                    ),
                ],
                dim=1,
            )
            with torch.inference_mode():
                # 1. Base Model Logits
                forward_res = self.model(new_prompt_ids)
                last_token_logits = forward_res.logits[0, -1, :]  # [vocab_size]

                # 2. Ask Oracle for Watermarked Logits
                # logits_processor expects [batch, vocab]
                wm_logits = self.oracle.logits_processor(
                    new_prompt_ids, last_token_logits.clone().unsqueeze(0)
                ).squeeze(0)
                possible_token_ids = None
                if self.cfg.sample_strategy == 'top_k':
                    possible_token_ids = torch.topk(wm_logits, k=self.cfg.total_word_num).indices
            wm_logits = wm_logits.clone()

            # Capture what would be the most likely token before modification
            orig_argmax_id = torch.argmax(wm_logits).item()

            # 3. Probe & Steal (Detect Green List)
            bias_delta = (
                self.cfg.delta_spoof if attack_mode == 'spoof' else self.cfg.delta_predicted
            )
            if bias_delta != 0:
                new_green_list = self._detect_red_green_list_kgw_cached(
                    prompt_ids=new_prompt_ids,
                    wm_logits=wm_logits,
                    possible_token_ids=possible_token_ids,
                    vocab_size=last_token_logits.shape[-1],
                )
                # 4. Reweight predicted green tokens for scrub or spoof mode.
                self._apply_detected_token_bias(
                    wm_logits,
                    new_green_list,
                    attack_mode=attack_mode,
                    delta=bias_delta,
                )

            # 5. Sample
            if self.cfg.sample_strategy == 'top_k':
                to_sample_tokens = torch.topk(wm_logits, k=self.cfg.total_word_num).indices
            else:
                raise ValueError('Unknown sample strategy')

            mask = torch.zeros_like(wm_logits, device=self.device)
            mask[to_sample_tokens] = 1
            new_probs = F.softmax(wm_logits, dim=-1) * mask
            new_probs = new_probs / new_probs.sum()

            new_token = torch.multinomial(new_probs, 1)
            tid = int(new_token)

            # Debug Log: Check if we replaced a meaningful token with an empty/special one
            orig_text_decoded = self.tokenizer.decode([orig_argmax_id], skip_special_tokens=True)
            new_text_decoded = self.tokenizer.decode([tid], skip_special_tokens=True)
            if orig_text_decoded.strip() and not new_text_decoded.strip():
                logger.debug(
                    f"[KGW] Replaced normal token '{orig_text_decoded}' (id={orig_argmax_id}) with empty/special token (id={tid})"
                )

            generated_token_ids.append(tid)
            if tid == getattr(self.tokenizer, 'eos_token_id', -1):
                break

        gen_ids = torch.tensor(generated_token_ids, dtype=torch.long, device=self.device)
        decoded = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
        # Strip leading ellipses often generated by chat models when continuing text ending in ...
        if decoded.strip().startswith('...'):
            decoded = decoded.strip()[3:].lstrip()
        if self._probe_cache_dirty > 0:
            self._save_probe_cache()
        return decoded

    def _generate_dipmark(
        self,
        raw_prompt: str,
        prompt_ids: torch.Tensor,
        attack_mode: str = 'scrub',
    ) -> str:
        generated_token_ids: List[int] = []
        for _ in range(self.cfg.generation_config['max_new_tokens']):
            new_prompt_ids = torch.cat(
                [
                    prompt_ids,
                    torch.tensor(generated_token_ids, dtype=torch.long, device=self.device).view(
                        1, -1
                    ),
                ],
                dim=1,
            )

            # 1. Base Model Logits
            forward_res = self.model(new_prompt_ids)
            last_token_logits = forward_res.logits[0, -1, :]

            # 2. Ask Oracle for Watermarked Logits
            wm_logits = self.oracle.logits_processor(
                new_prompt_ids, last_token_logits.clone().unsqueeze(0)
            ).squeeze(0)

            # Capture what would be the most likely token before modification
            orig_argmax_id = torch.argmax(wm_logits).item()

            if hasattr(self.tokenizer, 'eos_token_id'):
                wm_logits[self.tokenizer.eos_token_id] = -1e5

            # 3. Probe & Steal
            bias_delta = (
                self.cfg.delta_spoof if attack_mode == 'spoof' else self.cfg.delta_predicted
            )
            if bias_delta != 0:
                self._probe_requests_total += 1
                self._probe_live_queries_total += 1
                new_green_list, _, _ = detect_red_green_list_dipmark(
                    self.model_id,
                    new_prompt_ids,
                    self.cfg.ctx_width,
                    total_word_num=self.cfg.total_word_num,
                    model=self.model,
                    tokenizer=self.tokenizer,
                    oracle=self.oracle,
                    delta_predicted=self.cfg.delta_predicted,
                    sample_strategy=self.cfg.sample_strategy,
                    print_all=False,
                    vocab_size=last_token_logits.shape[-1],
                    dipmark_alpha=self.cfg.dipmark_alpha,
                )
                # 4. Reweight predicted green tokens for scrub or spoof mode.
                self._apply_detected_token_bias(
                    wm_logits,
                    new_green_list,
                    attack_mode=attack_mode,
                    delta=bias_delta,
                )

            # 5. Sample
            if self.cfg.sample_strategy == 'top_k':
                to_sample_tokens = torch.topk(wm_logits, k=self.cfg.total_word_num).indices
            else:
                raise ValueError('Unknown sample strategy')

            mask = torch.zeros_like(wm_logits, device=self.device)
            mask[to_sample_tokens] = 1
            new_probs = F.softmax(wm_logits, dim=-1) * mask
            new_probs = new_probs / new_probs.sum()

            new_token = torch.multinomial(new_probs, 1)
            tid = int(new_token)

            # Debug Log: Check if we replaced a meaningful token with an empty/special one
            orig_text_decoded = self.tokenizer.decode([orig_argmax_id], skip_special_tokens=True)
            new_text_decoded = self.tokenizer.decode([tid], skip_special_tokens=True)
            if orig_text_decoded.strip() and not new_text_decoded.strip():
                logger.debug(
                    f"[DipMark] Replaced normal token '{orig_text_decoded}' (id={orig_argmax_id}) with empty/special token (id={tid})"
                )

            generated_token_ids.append(tid)
            if tid == getattr(self.tokenizer, 'eos_token_id', -1):
                break

        gen_ids = torch.tensor(generated_token_ids, dtype=torch.long, device=self.device)
        decoded = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
        # Strip leading ellipses often generated by chat models when continuing text ending in ...
        if decoded.strip().startswith('...'):
            decoded = decoded.strip()[3:].lstrip()
        return decoded

    def pop_last_attack_meta(self) -> dict[str, Any]:
        meta = dict(self._last_attack_meta)
        self._last_attack_meta = {}
        return meta


# Global cache to avoid repeated tokenization
_TEMPLATE_CACHE = {}


def get_cached_templates(tokenizer, model_id):
    cache_key = model_id
    if cache_key in _TEMPLATE_CACHE:
        return _TEMPLATE_CACHE[cache_key]

    # Template definitions
    model_id_lower = model_id.lower()
    if 'llama-3' in model_id_lower:
        t1 = '<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nI need you to randomly choose a phrase without exact meaning. Randomly start your answer with: "'
        t2 = '" or "'
        t3 = '".<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n'
    elif 'mistral' in model_id_lower:
        t1 = '<s>[INST] I need you to randomly choose a phrase without exact meaning. Randomly start your answer with: "'
        t2 = '" or "'
        t3 = '".[/INST] '
    elif 'qwen' in model_id_lower:
        t1 = '<|im_start|>user\nI need you to randomly choose a phrase without exact meaning. Randomly start your answer with: "'
        t2 = '" or "'
        t3 = '".<|im_end|>\n<|im_start|>assistant\n'
    else:
        t1 = '<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nI need you to randomly choose a phrase without exact meaning. Randomly start your answer with: "'
        t2 = '" or "'
        t3 = '".<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n'

    t1_ids = tokenizer(t1, add_special_tokens=False, return_tensors='pt').input_ids[0]
    t2_ids = tokenizer(t2, add_special_tokens=False, return_tensors='pt').input_ids[0]
    t3_ids = tokenizer(t3, add_special_tokens=False, return_tensors='pt').input_ids[0]

    _TEMPLATE_CACHE[cache_key] = (t1_ids, t2_ids, t3_ids)
    return t1_ids, t2_ids, t3_ids


def construct_input_ids_manual(tokenizer, ctx_ids_cpu, target_id_pair, model_id):
    """
    Manually construct A/B test input IDs.
    ctx_ids_cpu must be a CPU tensor.
    target_id_pair: [id_A, id_B]
    """
    # Get cached template IDs
    t1_ids, t2_ids, t3_ids = get_cached_templates(tokenizer, model_id)

    id_a = torch.tensor([target_id_pair[0]], dtype=torch.long)
    id_b = torch.tensor([target_id_pair[1]], dtype=torch.long)

    # Build Prompt A: "... choose A or B ..."
    inputs1 = torch.cat(
        [t1_ids, ctx_ids_cpu, id_a, t2_ids, ctx_ids_cpu, id_b, t3_ids, ctx_ids_cpu], dim=0
    )

    # Build Prompt B: "... choose B or A ..."
    inputs2 = torch.cat(
        [t1_ids, ctx_ids_cpu, id_b, t2_ids, ctx_ids_cpu, id_a, t3_ids, ctx_ids_cpu], dim=0
    )

    # Return shape: [2, seq_len]
    return torch.stack([inputs1, inputs2], dim=0)


def generate_random_choose_probs_KGW(
    model,
    model_id: str,
    tokenizer,
    oracle,
    ctx_ids: torch.Tensor,
    token_lists: List[int],
    batch_size: int = 32,
    blackbox: bool = False,
    sample_num: int = 100,
    clear_cuda_cache_after_probe: bool = False,
):
    device = getattr(model, 'device', ctx_ids.device)

    # Result matrix (e.g., 20x20 is tiny and safe on GPU)
    probs_res_matrix = torch.zeros((len(token_lists), len(token_lists)), device=device)

    # 1. Preprocess: build all candidate pairs (idx1, idx2)
    # This is a Python list and does not consume VRAM.
    all_pairs_indices = []
    for idx1 in range(len(token_lists)):
        for idx2 in range(idx1 + 1, len(token_lists)):
            all_pairs_indices.append((idx1, idx2))

    # 2. Prepare CPU-side data
    ctx_ids_cpu = ctx_ids.cpu()
    # Pre-convert token_lists to tensor on GPU for indexing
    token_lists_tensor = torch.tensor(token_lists, device=device).long()

    # 3. Manual batch loop
    total_pairs = len(all_pairs_indices)

    # Explicitly disable gradients to avoid graph construction
    with torch.inference_mode():
        for i in range(0, total_pairs, batch_size):
            # === Step A: Prepare current batch data ===
            batch_indices = all_pairs_indices[i : i + batch_size]
            current_bsz = len(batch_indices)

            # Build tensors on CPU on demand
            batch_inputs_list = []
            batch_idx1 = []  # token index 1 for current batch
            batch_idx2 = []  # token index 2 for current batch

            for p_idx1, p_idx2 in batch_indices:
                # Get actual token IDs
                real_pair = [token_lists[p_idx1], token_lists[p_idx2]]
                # Call manual constructor above
                input_tensor = construct_input_ids_manual(
                    tokenizer, ctx_ids_cpu, real_pair, model_id
                )
                batch_inputs_list.append(input_tensor)
                batch_idx1.append(p_idx1)
                batch_idx2.append(p_idx2)

            # Stack to batch tensor: [bsz, 2, seq_len]
            # Still on CPU at this point.
            inputs = torch.stack(batch_inputs_list, dim=0)

            # === Step B: Move to GPU and run inference ===
            # Data enters VRAM only here.
            inputs = inputs.to(device)

            # Flatten to [bsz*2, seq_len]
            inputs_flat = inputs.view(current_bsz * 2, -1)

            # Model inference
            forward_res = model(inputs_flat)
            logits = forward_res.logits[:, -1, :]

            # Call Oracle (compatibility-aware)
            if hasattr(oracle, 'logits_processor'):
                logits = oracle.logits_processor(inputs_flat, logits)
            else:
                logits = oracle(inputs_flat, logits)

            probs = F.softmax(logits, dim=-1)

            # === Step C: Compute results and fill matrix ===
            # Convert lists to tensor indices
            b_idx1 = torch.tensor(batch_idx1, device=device)
            b_idx2 = torch.tensor(batch_idx2, device=device)

            if not blackbox:
                probs = probs.reshape(current_bsz, 2, -1).sum(dim=1)

                # Gather target-token probabilities
                p1 = probs[range(current_bsz), token_lists_tensor[b_idx1]]
                p2 = probs[range(current_bsz), token_lists_tensor[b_idx2]]

                s1 = torch.log(p1 / p2)
                s1 = s1.to(probs_res_matrix.dtype)

                # Fill matrix entries
                probs_res_matrix[b_idx1, b_idx2] = s1
                probs_res_matrix[b_idx2, b_idx1] = -s1

            else:
                # Black-box logic (simplified, preserving original intent)
                probs = probs.reshape(current_bsz, 2, -1)
                p_1 = probs[range(current_bsz), :, token_lists_tensor[b_idx1]]
                p_2 = probs[range(current_bsz), :, token_lists_tensor[b_idx2]]
                normed_prob = p_1 / (p_1 + p_2)

                random_v = torch.rand((sample_num, current_bsz, 2), device=device)
                res = (random_v < normed_prob.view(1, current_bsz, 2)).sum(dim=0).sum(dim=-1)

                # ... (existing clipping logic) ...
                res = torch.clamp(res, min=1.0, max=2 * sample_num - 1.0)

                s1 = torch.log(res / (2 * sample_num - res))
                s1 = s1.to(probs_res_matrix.dtype)

                probs_res_matrix[b_idx1, b_idx2] = s1
                probs_res_matrix[b_idx2, b_idx1] = -s1

            del inputs, inputs_flat, forward_res, logits, probs, batch_inputs_list

    if clear_cuda_cache_after_probe and torch.cuda.is_available():
        torch.cuda.empty_cache()

    return probs_res_matrix


def _resolve_vocab_size(tokenizer, model=None) -> int:
    vocab_size = int(getattr(tokenizer, 'vocab_size', 0) or 0)
    if vocab_size <= 0 and hasattr(tokenizer, '__len__'):
        try:
            vocab_size = int(len(tokenizer))
        except TypeError:
            vocab_size = 0
    if vocab_size <= 0 and model is not None:
        vocab_size = int(getattr(getattr(model, 'config', None), 'vocab_size', 0) or 0)
    if vocab_size <= 0:
        raise ValueError('Unable to resolve tokenizer vocab size for DEMARK parameter estimation.')
    return vocab_size


def _extract_input_ids(tokenizer_output) -> List[int]:
    if hasattr(tokenizer_output, 'input_ids'):
        input_ids = tokenizer_output.input_ids
    elif isinstance(tokenizer_output, dict):
        input_ids = tokenizer_output['input_ids']
    else:
        raise TypeError(f'Unsupported tokenizer output type: {type(tokenizer_output)!r}')

    if isinstance(input_ids, torch.Tensor):
        input_ids = input_ids.tolist()
    if input_ids and isinstance(input_ids[0], list):
        input_ids = input_ids[0]
    return [int(token_id) for token_id in input_ids]


def _normalize_candidate_token_text(text: str) -> str:
    cleaned = text.replace('Ġ', ' ').replace('▁', ' ').replace('Ċ', ' ').replace('Ŀ', '').strip()
    if len(cleaned) < 3 or not cleaned.isalpha():
        return ''
    return cleaned.lower()


def sample_target_token_ids(
    tokenizer,
    random_word_num: int,
    *,
    random_seed: int = 42,
    max_vocab_scan: int = 8192,
) -> List[int]:
    rng = random.Random(random_seed)
    candidate_token_ids: List[int] = []
    seen_token_ids: set[int] = set()

    try:
        from nltk import FreqDist
        from nltk.corpus import brown

        frequency_list = FreqDist(word.lower() for word in brown.words())
        for word, _ in frequency_list.most_common()[200:4000]:
            if len(word) < 3:
                continue
            token_ids = _extract_input_ids(tokenizer(f' {word}', add_special_tokens=False))
            if len(token_ids) != 1:
                continue
            token_id = int(token_ids[0])
            if token_id in seen_token_ids:
                continue
            seen_token_ids.add(token_id)
            candidate_token_ids.append(token_id)
        if len(candidate_token_ids) >= random_word_num:
            return rng.sample(candidate_token_ids, k=random_word_num)
        logger.debug(
            'Brown corpus candidate pool too small for DEMARK estimation (%s < %s); '
            'falling back to tokenizer vocab scan.',
            len(candidate_token_ids),
            random_word_num,
        )
    except Exception as exc:
        logger.debug('Falling back from Brown corpus token sampling: %s', exc)

    vocab_size = _resolve_vocab_size(tokenizer)
    scan_limit = vocab_size if max_vocab_scan <= 0 else min(max_vocab_scan, vocab_size)
    seen_normalized: set[str] = set()

    for token_id in range(scan_limit):
        decoded = tokenizer.decode([token_id], skip_special_tokens=True)
        normalized = _normalize_candidate_token_text(decoded)
        if not normalized or normalized in seen_normalized:
            continue
        seen_normalized.add(normalized)
        candidate_token_ids.append(int(token_id))
        if len(candidate_token_ids) >= max(random_word_num * 4, random_word_num):
            break

    deduped_candidates = list(dict.fromkeys(candidate_token_ids))
    if len(deduped_candidates) < random_word_num:
        raise ValueError(
            'Unable to sample enough single-token lexical items for DEMARK estimation '
            f'({len(deduped_candidates)} < {random_word_num}).'
        )
    return rng.sample(deduped_candidates, k=random_word_num)


def _threshold_token_scores(
    res_matrix: torch.Tensor,
    *,
    alpha1: float,
    alpha2: float,
) -> torch.Tensor:
    green_score = ((alpha1 < res_matrix) & (res_matrix < alpha2)).sum(dim=-1)
    red_score = ((-alpha2 < res_matrix) & (res_matrix < -alpha1)).sum(dim=-1)
    return green_score - red_score


def _estimate_ctx_width_from_same_rates(
    same_rate_dict: Dict[int, float], *, threshold: float, max_ctx_width: int
) -> int:
    if not same_rate_dict:
        return max(1, max_ctx_width)
    for window_width in sorted(same_rate_dict.keys(), reverse=True):
        if same_rate_dict[window_width] < threshold:
            return max(1, min(max_ctx_width, window_width + 1))
    return 1


def estimate_ctx_width_KGW(
    *,
    model,
    model_id: str,
    tokenizer,
    oracle,
    max_ctx_width: int,
    random_word_num: int,
    alpha1: float,
    alpha2: float,
    consistency_threshold: float,
    blackbox: bool = False,
    probe_batch_size: int = 32,
    random_seed: int = 42,
    max_vocab_scan: int = 8192,
    clear_cuda_cache_after_probe: bool = False,
) -> tuple[int, Dict[int, float]]:
    if max_ctx_width < 1:
        raise ValueError('max_ctx_width must be >= 1')

    device = getattr(model, 'device', torch.device('cpu'))
    vocab_size = _resolve_vocab_size(tokenizer, model)
    target_token_ids = sample_target_token_ids(
        tokenizer,
        random_word_num,
        random_seed=random_seed,
        max_vocab_scan=max_vocab_scan,
    )
    cpu_rng = torch.Generator()
    cpu_rng.manual_seed(random_seed)
    context_token_ids = torch.randint(
        low=1,
        high=vocab_size,
        size=(max_ctx_width,),
        generator=cpu_rng,
        dtype=torch.long,
    ).to(device)

    same_rate_dict: Dict[int, float] = {}
    last_final_score: Optional[torch.Tensor] = None

    for window_width in range(max_ctx_width, 0, -1):
        input_ctx_ids = context_token_ids[-window_width:]
        res_matrix = generate_random_choose_probs_KGW(
            model,
            model_id,
            tokenizer,
            oracle,
            input_ctx_ids,
            target_token_ids,
            batch_size=probe_batch_size,
            blackbox=blackbox,
            sample_num=10,
            clear_cuda_cache_after_probe=clear_cuda_cache_after_probe,
        )
        final_score = _threshold_token_scores(res_matrix, alpha1=alpha1, alpha2=alpha2)

        if last_final_score is not None:
            same_rate = ((final_score * last_final_score) > 0).float().mean().item()
            same_rate_dict[window_width] = float(same_rate)
        last_final_score = final_score

    estimated_ctx_width = _estimate_ctx_width_from_same_rates(
        same_rate_dict,
        threshold=consistency_threshold,
        max_ctx_width=max_ctx_width,
    )
    return estimated_ctx_width, same_rate_dict


def _estimate_delta_once_KGW(
    *,
    model,
    model_id: str,
    tokenizer,
    oracle,
    ctx_width: int,
    random_word_num: int,
    repeat_time: int,
    alpha1: float,
    alpha2: float,
    gamma: float,
    blackbox: bool = False,
    probe_batch_size: int = 32,
    random_seed: int = 42,
    max_vocab_scan: int = 8192,
    clear_cuda_cache_after_probe: bool = False,
) -> Optional[float]:
    if ctx_width < 1:
        raise ValueError('ctx_width must be >= 1 for DEMARK delta estimation')

    device = getattr(model, 'device', torch.device('cpu'))
    vocab_size = _resolve_vocab_size(tokenizer, model)
    cpu_rng = torch.Generator()
    cpu_rng.manual_seed(random_seed)

    total_delta = 0.0
    total_cnt = 0
    threshold = gamma * random_word_num

    for repeat_idx in range(repeat_time):
        context_token_ids = torch.randint(
            low=1,
            high=vocab_size,
            size=(ctx_width,),
            generator=cpu_rng,
            dtype=torch.long,
        ).to(device)
        target_token_ids = sample_target_token_ids(
            tokenizer,
            random_word_num,
            random_seed=random_seed + repeat_idx + 1,
            max_vocab_scan=max_vocab_scan,
        )
        res_matrix = generate_random_choose_probs_KGW(
            model,
            model_id,
            tokenizer,
            oracle,
            context_token_ids,
            target_token_ids,
            batch_size=probe_batch_size,
            blackbox=blackbox,
            sample_num=10,
            clear_cuda_cache_after_probe=clear_cuda_cache_after_probe,
        )
        final_score = _threshold_token_scores(res_matrix, alpha1=alpha1, alpha2=alpha2)
        green_indices = torch.where(final_score > threshold)[0].tolist()
        red_indices = torch.where(final_score < -threshold)[0].tolist()

        for green_idx in green_indices:
            for red_idx in red_indices:
                total_cnt += 1
                total_delta += float(res_matrix[green_idx][red_idx].item())

    if total_cnt == 0:
        return None
    return total_delta / total_cnt


def estimate_delta_KGW(
    *,
    model,
    model_id: str,
    tokenizer,
    oracle,
    ctx_width: int,
    random_word_num: int,
    repeat_time: int,
    run_num: int,
    alpha1: float,
    alpha2: float,
    gamma: float,
    blackbox: bool = False,
    probe_batch_size: int = 32,
    random_seed: int = 42,
    max_vocab_scan: int = 8192,
    clear_cuda_cache_after_probe: bool = False,
) -> Optional[float]:
    if run_num < 1:
        raise ValueError('run_num must be >= 1 for DEMARK delta estimation')

    estimates: List[float] = []
    for run_idx in range(run_num):
        estimated_delta = _estimate_delta_once_KGW(
            model=model,
            model_id=model_id,
            tokenizer=tokenizer,
            oracle=oracle,
            ctx_width=ctx_width,
            random_word_num=random_word_num,
            repeat_time=repeat_time,
            alpha1=alpha1,
            alpha2=alpha2,
            gamma=gamma,
            blackbox=blackbox,
            probe_batch_size=probe_batch_size,
            random_seed=random_seed + run_idx * 997,
            max_vocab_scan=max_vocab_scan,
            clear_cuda_cache_after_probe=clear_cuda_cache_after_probe,
        )
        if estimated_delta is not None:
            estimates.append(float(estimated_delta))

    if not estimates:
        return None
    return sum(estimates) / len(estimates)


def detect_red_green_list_KGW(
    model_id: str,
    prompt_ids: torch.Tensor,
    ctx_width: int,
    total_word_num: int,
    model,
    tokenizer,
    oracle,
    delta_predicted: float = 2.0,
    sample_strategy: str = 'top_k',
    print_all: bool = False,
    predefined_possible_token_ids: torch.Tensor = None,
    precomputed_wm_logits: Optional[torch.Tensor] = None,
    blackbox: bool = False,
    probe_batch_size: int = 32,
    clear_cuda_cache_after_probe: bool = False,
    vocab_size: int = -1,
):
    ctx_ids = prompt_ids[0, -ctx_width:]
    if precomputed_wm_logits is None:
        with torch.inference_mode():
            forward_res = model(prompt_ids)
            last_token_logits = forward_res.logits[0, -1, :]

            # Use Oracle
            wm_logits = oracle.logits_processor(
                prompt_ids, last_token_logits.clone().unsqueeze(0)
            ).squeeze(0)
    else:
        wm_logits = precomputed_wm_logits

    if sample_strategy == 'top_k':
        if predefined_possible_token_ids is None:
            possible_token_ids = torch.topk(wm_logits, k=total_word_num).indices
        else:
            possible_token_ids = predefined_possible_token_ids
        token_lists = list(set(possible_token_ids.cpu().tolist()))
    elif sample_strategy == 'predefined_token_list':
        possible_token_ids = predefined_possible_token_ids
        token_lists = list(set(possible_token_ids.cpu().tolist()))
    else:
        raise NotImplementedError
    res_matrix_raw = generate_random_choose_probs_KGW(
        model,
        model_id,
        tokenizer,
        oracle,
        ctx_ids,
        token_lists,
        batch_size=probe_batch_size,
        blackbox=blackbox,
        sample_num=10,
        clear_cuda_cache_after_probe=clear_cuda_cache_after_probe,
    )
    res_matrix = torch.zeros_like(res_matrix_raw)
    for idx1 in range(len(token_lists)):
        for idx2 in range(idx1 + 1, len(token_lists)):
            s1 = res_matrix_raw[idx1][idx2]
            s1_abs = torch.abs(s1)
            pairwise_score = torch.sign(s1) * torch.minimum(
                s1_abs / delta_predicted, delta_predicted / s1_abs
            )
            res_matrix[idx1][idx2] = pairwise_score
            res_matrix[idx2][idx1] = -pairwise_score
    res_matrix_sum = res_matrix.sum(dim=-1)
    token_scores = torch.zeros((len(token_lists),), device=prompt_ids.device)
    for idx1 in range(len(token_lists)):
        for idx2 in range(len(token_lists)):
            token_scores[idx1] += torch.abs(res_matrix_sum[idx2]) * res_matrix[idx1][idx2]
    detected_green_token_ids = []
    for idx in range(len(token_lists)):
        if (token_lists[idx] in possible_token_ids) and token_scores[idx] > 0:
            detected_green_token_ids.append(token_lists[idx])
    return (
        detected_green_token_ids,
        torch.zeros((4,), device=prompt_ids.device),
        torch.zeros((4,), device=prompt_ids.device),
    )


def detect_red_green_list_dipmark(*args, **kwargs):
    raise NotImplementedError('DEMARK DipMark probing is not implemented in this port.')
