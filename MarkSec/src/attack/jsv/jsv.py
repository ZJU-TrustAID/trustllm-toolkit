# ================================================
# jsv.py
# Watermark stealing attack adapted to this project's
# experiment runner interfaces.
# ================================================
from __future__ import annotations

import hashlib
import json
import logging
import os
import pickle
import re
import time
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm
from transformers import LogitsProcessorList

from src.attack.base import BaseAttack, BaseConfig
from src.attack.steal_corpus import (
    discover_joint_corpus_path,
    infer_dataset_hint,
    infer_model_hint,
    infer_watermark_hint,
    requires_exact_watermark_match,
)
from src.util.text import strip_prompt

from .count_store import CountStore
from .processors import SpoofedProcessor

logger = logging.getLogger(__name__)


class JSVConfig(BaseConfig):
    def __init__(self, config_path: str | None = None, **kwargs) -> None:
        self._algorithm_name = 'JSV'
        super().__init__(config_path, **kwargs)

    @property
    def algorithm_name(self) -> str:
        return self._algorithm_name

    def initialize_parameters(self) -> None:
        self.model_requirements = self.cfg_dict.get('model_requirements', {})

        stealing_params = self.cfg_dict.get('stealing_parameters', {})
        self.num_queries: int = int(stealing_params.get('num_queries', 30000))
        self.batch_size: int = int(stealing_params.get('batch_size', 8))
        self.learning_mode: str = str(stealing_params.get('learning_mode', 'slow')).lower()
        self.prevctx_width: int = int(stealing_params.get('prevctx_width', 3))
        raw_use_prebuilt = stealing_params.get('use_prebuilt_corpus', True)
        self.use_prebuilt_corpus: bool = (
            raw_use_prebuilt
            if isinstance(raw_use_prebuilt, bool)
            else str(raw_use_prebuilt).strip().lower() in {'1', 'true', 'yes', 'y', 'on'}
        )
        self.corpus_path: str = str(stealing_params.get('corpus_path') or '')
        self.corpus_base_path: str = str(stealing_params.get('corpus_base_path') or '')
        self.corpus_wm_path: str = str(stealing_params.get('corpus_wm_path') or '')
        self.auto_discover_corpus: bool = bool(stealing_params.get('auto_discover_corpus', True))
        self.corpus_dir: str = str(
            stealing_params.get('corpus_dir', 'output/corpus') or 'output/corpus'
        )
        self.corpus_prompt_field: str = str(stealing_params.get('corpus_prompt_field', 'prompt'))
        self.corpus_base_field: str = str(
            stealing_params.get('corpus_base_field', 'base_completion')
        )
        self.corpus_wm_field: str = str(stealing_params.get('corpus_wm_field', 'wm_completion'))
        self.reuse_stolen_artifact: bool = bool(stealing_params.get('reuse_stolen_artifact', True))
        self.save_stolen_artifact: bool = bool(stealing_params.get('save_stolen_artifact', True))
        self.stolen_artifact_path: str = str(stealing_params.get('stolen_artifact_path', '') or '')
        self.prepare_artifact_path: str = str(
            stealing_params.get('prepare_artifact_path', '') or ''
        )

        downstream_params = self.cfg_dict.get('downstream_parameters', {})
        self.delta_spoof: float = float(downstream_params.get('delta_spoof', 7.5))
        self.delta_scrub: float = float(downstream_params.get('delta_scrub', -7.5))

        score_weights = downstream_params.get('score_weights', {})
        self.w_full_context: float = float(score_weights.get('full_context', 1.0))
        self.w_partial_context: float = float(score_weights.get('partial_context', 0.5))
        self.w_no_context: float = float(score_weights.get('no_context', 0.25))

        threshold_params = downstream_params.get('thresholds', {})
        self.min_wm_count_nonempty: int = int(threshold_params.get('min_wm_count_nonempty', 3))
        self.min_wm_mass_empty: float = float(threshold_params.get('min_wm_mass_empty', 7e-5))

        processor_cfg = self.cfg_dict.get('processor_config', {})
        self.clip_at: float = float(
            processor_cfg.get('clip_at', downstream_params.get('clip_at', 2.0))
        )
        self.w_abcd: float = float(processor_cfg.get('w_abcd', self.w_full_context))
        self.w_partials: float = float(processor_cfg.get('w_partials', self.w_partial_context))
        self.w_empty: float = float(processor_cfg.get('w_empty', self.w_no_context))
        self.min_wm_count_nonempty = int(
            processor_cfg.get('min_wm_count_nonempty', self.min_wm_count_nonempty)
        )
        self.min_wm_mass_empty = float(
            processor_cfg.get('min_wm_mass_empty', self.min_wm_mass_empty)
        )

        self.generation_config: dict[str, Any] = self.cfg_dict.get('generation_config', {})
        self.generation_config.setdefault('do_sample', True)
        self.generation_config.setdefault('top_p', 0.9)
        self.generation_config.setdefault('temperature', 0.8)
        self.generation_config.setdefault('max_new_tokens', 200)

        self.scrub_prompt_template: str = self.cfg_dict.get(
            'scrub_prompt_template',
            (
                'Paraphrase the following text while preserving meaning and fluency. '
                'Do not add or remove facts.\nText:\n{text}'
            ),
        )
        self.spoof_prompt_template: str = self.cfg_dict.get(
            'spoof_prompt_template',
            (
                'Write a natural response for the following prompt in your own wording. '
                'Keep the answer fluent, coherent, and concise.\nPrompt:\n{prompt}'
            ),
        )


class JSVAttack(BaseAttack):
    def __init__(
        self,
        config: JSVConfig,
        attacker_model: Any | None = None,
        attacker_tokenizer: Any | None = None,
        base_model: Any | None = None,
        tokenizer: Any | None = None,
        dataset: Any | None = None,
        watermark_instance: Any | None = None,
        watermark_name: str = '',
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.cfg = config

        self.base_model = base_model
        self.base_tokenizer = tokenizer
        self.attacker_model = attacker_model if attacker_model is not None else base_model
        self.attacker_tokenizer = (
            attacker_tokenizer if attacker_tokenizer is not None else tokenizer
        )
        self.dataset = dataset
        self.watermark_instance = watermark_instance
        self.watermark_name = watermark_name

        if self.attacker_model is None or self.attacker_tokenizer is None:
            raise ValueError(
                'JSVAttack requires attacker_model/attacker_tokenizer or base_model/tokenizer.'
            )

        if hasattr(self.attacker_model, 'eval'):
            self.attacker_model.eval()

        self.counts_base = CountStore(self.cfg.prevctx_width)
        self.counts_wm = CountStore(self.cfg.prevctx_width)
        self._learning_data_ready = False
        self._scoring_function_ready = False
        self._stolen_artifact_path = self._resolve_stolen_artifact_path()
        self._prepare_artifact_path = self._resolve_prepare_artifact_path()
        self._prepare_meta: dict[str, Any] = {}
        self._last_attack_meta: dict[str, Any] = {}

        logger.info('%s attack initialized.', self.cfg.algorithm_name)

    def prepare(self) -> bool:
        prepare_t0 = time.time()
        logger.info('Preparing %s from prebuilt stealing corpus.', self.cfg.algorithm_name)
        if not self.cfg.use_prebuilt_corpus:
            raise ValueError(
                'JSV only supports prebuilt corpus mode. '
                'Set `stealing_parameters.use_prebuilt_corpus=true` and provide '
                '`corpus_path` or both `corpus_base_path` and `corpus_wm_path`.'
            )

        self._ensure_prebuilt_corpus_paths()
        self._refresh_artifact_paths()

        if self.cfg.reuse_stolen_artifact and self._load_stolen_artifact():
            self._learning_data_ready = True
            self._scoring_function_ready = True
            self._prepare_meta['prepare_runtime_cost'] = float(time.time() - prepare_t0)
            self._prepare_meta['loaded_from_artifact'] = True
            logger.info('JSV ready from cached stolen artifact: %s', self._stolen_artifact_path)
            return True

        base_entries, wm_entries = self._load_corpus_entries()
        if not base_entries:
            raise ValueError('Prebuilt corpus mode enabled, but no usable base entries were found.')
        if not wm_entries:
            raise ValueError(
                'Prebuilt corpus mode enabled, but no usable watermarked entries were found.'
            )

        self._learning_data_ready = True
        self._learn_from_entries(base_entries, base=True, source_label='corpus')
        self._learn_from_entries(wm_entries, base=False, source_label='corpus')

        identity = self._build_artifact_identity()
        self._prepare_meta = self._build_prepare_meta(
            identity=identity,
            base_entries=len(base_entries),
            wm_entries=len(wm_entries),
            prepare_runtime_cost=float(time.time() - prepare_t0),
            loaded_from_artifact=False,
        )

        if self.cfg.save_stolen_artifact:
            self._save_stolen_artifact(len(base_entries), len(wm_entries))

        self._scoring_function_ready = True
        logger.info(
            'JSV ready from prebuilt corpus. Base entries=%d, WM entries=%d.',
            self.counts_base.nb_keys(False),
            self.counts_wm.nb_keys(False),
        )
        return True

    def edit(self, text: str, prompt: str = '', **kwargs: Any) -> str:
        if not self._scoring_function_ready:
            raise RuntimeError('Attack is not prepared. Call prepare() before edit().')
        attack_t0 = time.time()

        attack_type = str(kwargs.get('attack_type') or '').strip().lower()
        if not attack_type:
            attack_type = 'scrubbing' if text and text.strip() else 'spoofing'

        if attack_type in {'spoof', 'spoofing'}:
            source_prompt = str(prompt or kwargs.get('reference') or text).strip()
            if not source_prompt:
                return ''
            generation_prompt = self.cfg.spoof_prompt_template.format(
                prompt=source_prompt,
                text=source_prompt,
                reference=str(kwargs.get('reference') or source_prompt),
            )
            delta = self.cfg.delta_spoof
            attack_mode = 'spoof'
        else:
            source_text = str(text or kwargs.get('reference') or prompt).strip()
            if not source_text:
                self._record_last_attack_meta(
                    attack_mode='scrub',
                    runtime_cost=float(time.time() - attack_t0),
                    delta=self.cfg.delta_scrub,
                    generated_text='',
                )
                return ''
            generation_prompt = self.cfg.scrub_prompt_template.format(
                text=source_text,
                prompt=str(prompt or ''),
            )
            delta = self.cfg.delta_scrub
            attack_mode = 'scrub'

        attacked_text = self._generate_with_stolen_scoring(
            generation_prompt=generation_prompt, delta=delta
        )
        self._record_last_attack_meta(
            attack_mode=attack_mode,
            runtime_cost=float(time.time() - attack_t0),
            delta=delta,
            generated_text=str(attacked_text or ''),
        )
        if not attacked_text:
            return text
        return attacked_text

    def _effective_num_queries(self) -> int:
        return self.cfg.num_queries

    def _resolve_stolen_artifact_path(self) -> str:
        if self.cfg.stolen_artifact_path:
            return os.path.abspath(self.cfg.stolen_artifact_path)

        identity = self._build_artifact_identity()
        model_tag = self._sanitize_artifact_component(identity['model_hint'] or 'unknown_model')
        dataset_tag = self._sanitize_artifact_component(
            identity['dataset_hint'] or 'unknown_dataset'
        )
        watermark_tag = self._sanitize_artifact_component(
            identity['watermark_hint'] or 'unknown_watermark'
        )
        corpus_tag = self._sanitize_artifact_component(
            self._build_corpus_tag(identity) or 'unknown_corpus'
        )
        out_path = Path('output/models/jsv_result') / (
            f'stolen_counts_{dataset_tag}_{watermark_tag}_{model_tag}_'
            f'w{self.cfg.prevctx_width}_{corpus_tag}.pkl'
        )
        return str(out_path.resolve())

    def _resolve_prepare_artifact_path(self) -> str:
        if self.cfg.prepare_artifact_path:
            return os.path.abspath(self.cfg.prepare_artifact_path)
        return str(Path(self._stolen_artifact_path).with_suffix('.meta.json'))

    def _ensure_prebuilt_corpus_paths(self) -> None:
        if self.cfg.corpus_path:
            self.cfg.corpus_path = os.path.abspath(self.cfg.corpus_path)
            return
        if self.cfg.corpus_base_path and self.cfg.corpus_wm_path:
            self.cfg.corpus_base_path = os.path.abspath(self.cfg.corpus_base_path)
            self.cfg.corpus_wm_path = os.path.abspath(self.cfg.corpus_wm_path)
            return
        if not self.cfg.auto_discover_corpus:
            return

        watermark_hint = infer_watermark_hint(
            watermark_name=self.watermark_name,
            watermark_instance=self.watermark_instance,
        )
        model_hint = infer_model_hint(model=self.attacker_model, tokenizer=self.attacker_tokenizer)
        dataset_hint = infer_dataset_hint(self.dataset)

        discovered = discover_joint_corpus_path(
            corpus_dir=self.cfg.corpus_dir,
            model_hint=model_hint,
            dataset_hint=dataset_hint,
            watermark_hint=watermark_hint,
        )
        if discovered:
            self.cfg.corpus_path = discovered
            logger.info('JSV auto-selected prebuilt corpus: %s', self.cfg.corpus_path)
            return
        if watermark_hint and requires_exact_watermark_match(watermark_hint):
            raise FileNotFoundError(
                '[JSV] No exact prebuilt corpus match found for '
                f'model={model_hint or "unknown"} dataset={dataset_hint or "unknown"} '
                f'watermark={watermark_hint}. Refusing to reuse a different setting.'
            )

    def _refresh_artifact_paths(self) -> None:
        self._stolen_artifact_path = self._resolve_stolen_artifact_path()
        self._prepare_artifact_path = self._resolve_prepare_artifact_path()

    def _build_artifact_identity(self) -> dict[str, str]:
        corpus_path = self._resolve_path(self.cfg.corpus_path)
        corpus_base_path = self._resolve_path(self.cfg.corpus_base_path)
        corpus_wm_path = self._resolve_path(self.cfg.corpus_wm_path)
        return {
            'dataset_hint': infer_dataset_hint(self.dataset),
            'watermark_hint': infer_watermark_hint(
                watermark_name=self.watermark_name,
                watermark_instance=self.watermark_instance,
            ),
            'model_hint': infer_model_hint(
                model=self.attacker_model,
                tokenizer=self.attacker_tokenizer,
            ),
            'corpus_path': corpus_path,
            'corpus_base_path': corpus_base_path,
            'corpus_wm_path': corpus_wm_path,
            'corpus_path_fingerprint': self._fingerprint_path(corpus_path),
            'corpus_base_path_fingerprint': self._fingerprint_path(corpus_base_path),
            'corpus_wm_path_fingerprint': self._fingerprint_path(corpus_wm_path),
        }

    @staticmethod
    def _sanitize_artifact_component(value: str) -> str:
        collapsed = re.sub(r'[^A-Za-z0-9._-]+', '_', str(value).strip())
        return collapsed[:80].strip('._-') or 'unknown'

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

    def _build_corpus_tag(self, identity: dict[str, str]) -> str:
        corpus_names = [
            Path(path).stem
            for path in (
                identity.get('corpus_path', ''),
                identity.get('corpus_base_path', ''),
                identity.get('corpus_wm_path', ''),
            )
            if path
        ]
        corpus_names = [self._sanitize_artifact_component(name) for name in corpus_names if name]
        prefix = '__'.join(corpus_names[:2]) if corpus_names else 'nocorpus'
        fingerprint_material = '|'.join(
            value
            for value in (
                identity.get('corpus_path', ''),
                identity.get('corpus_base_path', ''),
                identity.get('corpus_wm_path', ''),
                identity.get('corpus_path_fingerprint', ''),
                identity.get('corpus_base_path_fingerprint', ''),
                identity.get('corpus_wm_path_fingerprint', ''),
            )
            if value
        )
        if not fingerprint_material:
            return prefix
        digest = hashlib.sha256(fingerprint_material.encode('utf-8')).hexdigest()[:10]
        return f'{prefix}_{digest}'

    def _build_prepare_meta(
        self,
        *,
        identity: dict[str, str],
        base_entries: int | None,
        wm_entries: int | None,
        prepare_runtime_cost: float,
        loaded_from_artifact: bool,
    ) -> dict[str, Any]:
        base_entries_count = None if base_entries is None else int(base_entries)
        wm_entries_count = None if wm_entries is None else int(wm_entries)
        if base_entries_count is not None or wm_entries_count is not None:
            stealing_queries = max(base_entries_count or 0, wm_entries_count or 0)
        else:
            stealing_queries = 0
        return {
            'attack': 'JSV',
            'artifact_type': 'stolen_counts',
            'artifact_path': str(self._stolen_artifact_path),
            'prepare_artifact_path': str(self._stolen_artifact_path),
            'prepare_meta_path': str(self._prepare_artifact_path),
            'meta_version': 'jsv_prepare_meta_v2',
            'dataset_hint': identity.get('dataset_hint', ''),
            'watermark_hint': identity.get('watermark_hint', ''),
            'model_hint': identity.get('model_hint', ''),
            'corpus_path': identity.get('corpus_path', ''),
            'corpus_base_path': identity.get('corpus_base_path', ''),
            'corpus_wm_path': identity.get('corpus_wm_path', ''),
            'corpus_path_fingerprint': identity.get('corpus_path_fingerprint', ''),
            'corpus_base_path_fingerprint': identity.get('corpus_base_path_fingerprint', ''),
            'corpus_wm_path_fingerprint': identity.get('corpus_wm_path_fingerprint', ''),
            'prevctx_width': int(self.cfg.prevctx_width),
            'base_entries': base_entries_count,
            'wm_entries': wm_entries_count,
            'num_queries_target': int(self.cfg.num_queries),
            'stealing_queries_collected': int(stealing_queries),
            'attack_objective': 'shared_scrub_spoof_prepare',
            'prepare_runtime_cost': float(prepare_runtime_cost),
            'loaded_from_artifact': bool(loaded_from_artifact),
        }

    def _record_last_attack_meta(
        self,
        *,
        attack_mode: str,
        runtime_cost: float,
        delta: float,
        generated_text: str,
    ) -> None:
        generated_token_count = None
        if generated_text and self.attacker_tokenizer is not None:
            try:
                tokenized = self.attacker_tokenizer(
                    generated_text,
                    add_special_tokens=False,
                    return_tensors='pt',
                )
                token_ids = tokenized.get('input_ids')
                if hasattr(token_ids, 'shape'):
                    generated_token_count = int(token_ids.shape[-1])
                elif isinstance(token_ids, list):
                    first_item = (
                        token_ids[0] if token_ids and isinstance(token_ids[0], list) else token_ids
                    )
                    generated_token_count = int(len(first_item))
            except Exception:
                generated_token_count = None

        query_cost = self._prepare_meta.get('stealing_queries_collected')
        if query_cost is None:
            query_cost = self._prepare_meta.get('num_queries_target')

        self._last_attack_meta = {
            **self._prepare_meta,
            'attack': 'JSV',
            'mode': str(attack_mode),
            'runtime_cost': float(runtime_cost),
            'query_cost': float(query_cost) if query_cost is not None else 0.0,
            'delta': float(delta),
            'generated_token_count': generated_token_count,
        }

    def _save_stolen_artifact(self, base_entries: int, wm_entries: int) -> None:
        payload = {
            'version': 'jsv_stolen_counts_v2',
            'prevctx_width': int(self.cfg.prevctx_width),
            'counts_base_ordered': self.counts_base.counts_ordered,
            'counts_base_unordered': self.counts_base.counts_unordered,
            'counts_wm_ordered': self.counts_wm.counts_ordered,
            'counts_wm_unordered': self.counts_wm.counts_unordered,
        }
        artifact_path = Path(self._stolen_artifact_path)
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        with open(artifact_path, 'wb') as f:
            pickle.dump(payload, f)

        meta = {
            **self._prepare_meta,
            'artifact_path': str(artifact_path),
            'prepare_artifact_path': str(artifact_path),
            'prepare_meta_path': str(self._prepare_artifact_path),
        }
        self._prepare_meta = dict(meta)
        meta_path = Path(self._prepare_artifact_path)
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        logger.info('JSV saved stolen artifact to %s', artifact_path)

    def _load_stolen_artifact(self) -> bool:
        artifact_path = Path(self._stolen_artifact_path)
        if not artifact_path.exists():
            return False

        loaded_meta: dict[str, Any] | None = None
        meta_path = Path(self._prepare_artifact_path)
        if meta_path.exists():
            try:
                with open(meta_path, 'r', encoding='utf-8') as f:
                    meta = json.load(f)
                identity = self._build_artifact_identity()
                if not self._artifact_meta_matches(meta, identity):
                    return False
                loaded_meta = meta if isinstance(meta, dict) else None
            except Exception as exc:
                logger.warning('JSV failed to read artifact metadata %s: %s', meta_path, exc)
                return False

        try:
            with open(artifact_path, 'rb') as f:
                payload = pickle.load(f)
        except Exception as exc:
            logger.warning('JSV failed to load cached stolen artifact %s: %s', artifact_path, exc)
            return False

        if int(payload.get('prevctx_width', -1)) != int(self.cfg.prevctx_width):
            logger.warning(
                'JSV cached artifact prevctx_width mismatch (%s != %s); rebuilding.',
                payload.get('prevctx_width'),
                self.cfg.prevctx_width,
            )
            return False

        self.counts_base.counts_ordered = dict(payload.get('counts_base_ordered', {}))
        self.counts_base.counts_unordered = dict(payload.get('counts_base_unordered', {}))
        self.counts_wm.counts_ordered = dict(payload.get('counts_wm_ordered', {}))
        self.counts_wm.counts_unordered = dict(payload.get('counts_wm_unordered', {}))
        if self.counts_base.nb_keys(False) <= 0 or self.counts_wm.nb_keys(False) <= 0:
            logger.info('JSV cached artifact is empty; rebuilding from corpus.')
            return False
        if loaded_meta is not None:
            self._prepare_meta = dict(loaded_meta)
        else:
            identity = self._build_artifact_identity()
            self._prepare_meta = self._build_prepare_meta(
                identity=identity,
                base_entries=None,
                wm_entries=None,
                prepare_runtime_cost=0.0,
                loaded_from_artifact=True,
            )
        return True

    def _artifact_meta_matches(self, meta: dict[str, Any], identity: dict[str, str]) -> bool:
        checks = [
            ('dataset_hint', 'dataset'),
            ('watermark_hint', 'watermark'),
            ('model_hint', 'attacker model'),
            ('corpus_path', 'joint corpus path'),
            ('corpus_base_path', 'base corpus path'),
            ('corpus_wm_path', 'watermarked corpus path'),
            ('corpus_path_fingerprint', 'joint corpus fingerprint'),
            ('corpus_base_path_fingerprint', 'base corpus fingerprint'),
            ('corpus_wm_path_fingerprint', 'watermarked corpus fingerprint'),
        ]
        for key, label in checks:
            current_value = str(identity.get(key, '') or '')
            if not current_value:
                continue
            cached_value = str(meta.get(key, '') or '')
            if not cached_value:
                logger.info(
                    'JSV cached artifact missing %s identity metadata; rebuilding. key=%s',
                    label,
                    key,
                )
                return False
            normalized_current = current_value
            normalized_cached = cached_value
            if key.endswith('_path'):
                normalized_current = os.path.abspath(current_value)
                normalized_cached = os.path.abspath(cached_value)
            if normalized_current != normalized_cached:
                logger.info(
                    'JSV cached artifact %s mismatch; rebuilding. cached=%s current=%s',
                    label,
                    cached_value,
                    current_value,
                )
                return False
        return True

    def _load_corpus_entries(self) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        max_entries = self._effective_num_queries()
        corpus_path = self._resolve_path(self.cfg.corpus_path)
        corpus_base_path = self._resolve_path(self.cfg.corpus_base_path)
        corpus_wm_path = self._resolve_path(self.cfg.corpus_wm_path)

        if corpus_path:
            if not os.path.exists(corpus_path):
                raise FileNotFoundError(f'Configured corpus_path not found: {corpus_path}')
            logger.info('Loading joint prebuilt corpus from: %s', corpus_path)
            return self._read_joint_corpus(corpus_path, max_entries)

        if corpus_base_path and corpus_wm_path:
            if not os.path.exists(corpus_base_path):
                raise FileNotFoundError(
                    f'Configured corpus_base_path not found: {corpus_base_path}'
                )
            if not os.path.exists(corpus_wm_path):
                raise FileNotFoundError(f'Configured corpus_wm_path not found: {corpus_wm_path}')
            logger.info(
                'Loading split prebuilt corpora from base=%s wm=%s',
                corpus_base_path,
                corpus_wm_path,
            )
            base_entries = self._read_single_corpus(
                corpus_base_path, max_entries=max_entries, for_wm=False
            )
            wm_entries = self._read_single_corpus(
                corpus_wm_path, max_entries=max_entries, for_wm=True
            )
            return base_entries, wm_entries

        raise ValueError(
            'Prebuilt corpus mode requires `stealing_parameters.corpus_path` or both '
            '`stealing_parameters.corpus_base_path` and `stealing_parameters.corpus_wm_path`.'
        )

    @staticmethod
    def _resolve_path(path: str) -> str:
        if not path:
            return ''
        if os.path.isabs(path):
            return path
        return os.path.abspath(path)

    def _read_joint_corpus(
        self, path: str, max_entries: int
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        base_entries: list[dict[str, Any]] = []
        wm_entries: list[dict[str, Any]] = []

        with open(path, 'r', encoding='utf-8') as f:
            for line_idx, line in enumerate(f):
                if len(base_entries) >= max_entries and len(wm_entries) >= max_entries:
                    break
                try:
                    record = json.loads(line)
                except Exception:
                    continue

                prompt = self._extract_prompt_from_corpus_record(record)
                item_idx = int(record.get('idx', record.get('id', line_idx)))

                base_text = self._extract_base_completion(record, prompt)
                if base_text and len(base_entries) < max_entries:
                    base_entries.append({'idx': item_idx, 'prompt': prompt, 'textwm': base_text})

                wm_text = self._extract_wm_completion(record, prompt)
                if wm_text and len(wm_entries) < max_entries:
                    wm_entries.append({'idx': item_idx, 'prompt': prompt, 'textwm': wm_text})

        logger.info(
            'Loaded %d base entries and %d wm entries from joint corpus.',
            len(base_entries),
            len(wm_entries),
        )
        return base_entries, wm_entries

    def _read_single_corpus(
        self, path: str, max_entries: int, for_wm: bool
    ) -> list[dict[str, Any]]:
        entries: list[dict[str, Any]] = []
        with open(path, 'r', encoding='utf-8') as f:
            for line_idx, line in enumerate(f):
                if len(entries) >= max_entries:
                    break
                try:
                    record = json.loads(line)
                except Exception:
                    continue

                prompt = self._extract_prompt_from_corpus_record(record)
                item_idx = int(record.get('idx', record.get('id', line_idx)))
                text = (
                    self._extract_wm_completion(record, prompt)
                    if for_wm
                    else self._extract_base_completion(record, prompt)
                )
                if not text:
                    continue
                entries.append({'idx': item_idx, 'prompt': prompt, 'textwm': text})

        logger.info(
            'Loaded %d %s entries from %s.',
            len(entries),
            'wm' if for_wm else 'base',
            path,
        )
        return entries

    @staticmethod
    def _first_nonempty_string(record: dict[str, Any], keys: list[str]) -> str:
        for key in keys:
            value = record.get(key)
            if isinstance(value, str) and value.strip():
                return value
        return ''

    def _extract_prompt_from_corpus_record(self, record: dict[str, Any]) -> str:
        return self._first_nonempty_string(
            record,
            [
                self.cfg.corpus_prompt_field,
                'prompt',
                'source_prompt',
                'input',
            ],
        )

    def _extract_base_completion(self, record: dict[str, Any], prompt: str) -> str:
        raw_text = self._first_nonempty_string(
            record,
            [
                self.cfg.corpus_base_field,
                'base_completion',
                'clean_completion',
                'generated_only_clean',
                'base_text',
                'clean_text',
            ],
        )
        return self._normalize_corpus_completion(prompt, raw_text)

    def _extract_wm_completion(self, record: dict[str, Any], prompt: str) -> str:
        raw_text = self._first_nonempty_string(
            record,
            [
                self.cfg.corpus_wm_field,
                'wm_completion',
                'watermarked_completion',
                'generated_only_wm',
                'generated_only',
                'wm_text',
                'watermarked_text',
            ],
        )
        return self._normalize_corpus_completion(prompt, raw_text)

    @staticmethod
    def _normalize_corpus_completion(prompt: str, text: str) -> str:
        normalized = str(text).strip()
        if not normalized:
            return ''
        if prompt:
            return strip_prompt(normalized, prompt).strip()
        return normalized

    def _learn_from_entries(
        self, entries: list[dict[str, Any]], base: bool, source_label: str
    ) -> None:
        logger.info(
            'Learning JSV counts from %d %s entries (base=%s).',
            len(entries),
            source_label,
            base,
        )
        dest_counts = self.counts_base if base else self.counts_wm
        if base:
            self._learn_fast([str(e.get('textwm', '')) for e in entries], dest_counts)
        elif self.cfg.learning_mode == 'slow':
            self._learn_slow(entries, dest_counts)
        else:
            self._learn_fast([str(e.get('textwm', '')) for e in entries], dest_counts)

    def _learn_fast(self, completions: list[str], dest_counts: CountStore) -> None:
        valid_texts = [text for text in completions if text]
        if not valid_texts:
            return

        try:
            tokenized = self.attacker_tokenizer(
                valid_texts,
                add_special_tokens=False,
                truncation=True,
                max_length=1024,
            )['input_ids']
        except Exception:
            tokenized = [
                self.attacker_tokenizer(
                    text,
                    add_special_tokens=False,
                    truncation=True,
                    max_length=1024,
                )['input_ids'][0]
                for text in valid_texts
            ]

        for seq in tqdm(tokenized, desc='JSV learn fast', leave=False):
            if len(seq) <= self.cfg.prevctx_width:
                continue
            for i in range(self.cfg.prevctx_width, len(seq)):
                ctx = tuple(seq[i - self.cfg.prevctx_width : i])
                dest_counts.add(ctx, int(seq[i]), 1)

    def _learn_slow(self, queries: list[dict[str, Any]], dest_counts: CountStore) -> None:
        for query in tqdm(queries, desc='JSV learn slow', leave=False):
            prompt = str(query.get('prompt', ''))
            text_wm = str(query.get('textwm', ''))
            if not text_wm:
                continue

            full_text = prompt + text_wm
            try:
                model_inputs = self._tokenize_to_device(
                    self.attacker_tokenizer,
                    full_text,
                    self.attacker_model.device,
                    return_tensors='pt',
                    add_special_tokens=False,
                    truncation=True,
                    max_length=1024,
                )
                full_ids = model_inputs['input_ids'][0]
                prompt_ids = self.attacker_tokenizer(prompt, add_special_tokens=False)['input_ids']
                prompt_len = min(len(prompt_ids), max(0, full_ids.shape[0] - 1))
                response_ids = full_ids[prompt_len:].tolist()
                if len(response_ids) <= self.cfg.prevctx_width:
                    continue

                with torch.inference_mode():
                    outputs = self.attacker_model(**model_inputs)
                logits = outputs.logits[0]
            except Exception as e:
                logger.warning('Skipping a JSV slow-learning sample due to model error: %s', e)
                continue

            for i in range(self.cfg.prevctx_width, len(response_ids)):
                global_idx = prompt_len + i
                if global_idx - 1 >= logits.shape[0]:
                    break

                predicted_token_id = int(torch.argmax(logits[global_idx - 1]).item())
                actual_token_id = int(response_ids[i])
                if predicted_token_id == actual_token_id:
                    continue

                ctx = tuple(response_ids[i - self.cfg.prevctx_width : i])
                dest_counts.add(ctx, actual_token_id, 1)

    def _generate_plain(self, model: Any, tokenizer: Any, prompt: str) -> str:
        inputs = self._tokenize_to_device(
            tokenizer,
            prompt,
            model.device,
            return_tensors='pt',
        )
        input_ids = inputs['input_ids']
        with torch.inference_mode():
            outputs = model.generate(**inputs, **self.cfg.generation_config)
        completion_ids = outputs[:, input_ids.shape[1] :]
        completion = tokenizer.batch_decode(completion_ids, skip_special_tokens=True)[0].strip()
        if completion:
            return completion
        full = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return strip_prompt(full, prompt).strip()

    def _generate_with_stolen_scoring(self, generation_prompt: str, delta: float) -> str:
        processor = SpoofedProcessor(
            counts_base=self.counts_base,
            counts_wm=self.counts_wm,
            counts_wm_ft=None,
            prevctx_width=self.cfg.prevctx_width,
            vocab_size=int(getattr(self.attacker_tokenizer, 'vocab_size', 32000)),
            tokenizer=self.attacker_tokenizer,
            cfg=self.cfg,
        )
        processor.set_delta(delta)
        processor_list = LogitsProcessorList([processor])

        inputs = self._tokenize_to_device(
            self.attacker_tokenizer,
            generation_prompt,
            self.attacker_model.device,
            return_tensors='pt',
        )
        input_ids = inputs['input_ids']

        with torch.inference_mode():
            outputs = self.attacker_model.generate(
                **inputs,
                logits_processor=processor_list,
                **self.cfg.generation_config,
            )

        completion_ids = outputs[:, input_ids.shape[1] :]
        attacked_text = self.attacker_tokenizer.batch_decode(
            completion_ids, skip_special_tokens=True
        )[0].strip()
        if attacked_text:
            return attacked_text
        full_text = self.attacker_tokenizer.decode(outputs[0], skip_special_tokens=True)
        return strip_prompt(full_text, generation_prompt).strip()

    def pop_last_attack_meta(self) -> dict[str, Any]:
        meta = dict(self._last_attack_meta)
        self._last_attack_meta = {}
        return meta

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
