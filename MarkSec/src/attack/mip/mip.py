"""
Implementation of MIP (Mixed Integer Programming) Attack.
This attack consists of two phases:
1. Stealing Phase (prepare): Reuse a stolen greenlist if available, otherwise invert
   the greenlist from a prebuilt corpus or on-the-fly watermark queries.
2. Removal Phase (edit): Replace identified green tokens with synonyms.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any

import pyomo.environ as pyomo
import torch
import torch.nn.functional as F
from tqdm import tqdm

from src.attack.mip.GensimModel import GensimModel
from src.attack.mip.greenlist_inversion_plus import GreenlistInversorPlus
from src.attack.steal_corpus import (
    discover_joint_corpus_path,
    infer_dataset_hint,
    infer_model_hint,
    infer_watermark_hint,
    requires_exact_watermark_match,
)
from src.util.text import strip_prompt

from ..base import BaseAttack, BaseConfig

logger = logging.getLogger('MarkSec')


class MIPConfig(BaseConfig):
    """Configuration for MIP attack."""

    def initialize_parameters(self) -> None:
        """Initialize MIP-specific parameters."""
        self.model_requirements = dict(self.cfg_dict.get('model_requirements', {}))

        # MIP solver
        self.solver_type = self.cfg_dict.get('solver_type', 'gurobi')
        self.time_limit = self.cfg_dict.get('time_limit', 3600)  # Default 1 hour
        self.mip_gap = self.cfg_dict.get('mip_gap', 0.05)  # Default 5% gap
        self.attack_mode = self.cfg_dict.get('attack_mode', 'AS1')  # AS1 or AS2

        self.stealing_samples = int(self.cfg_dict.get('stealing_samples', 2000))

        #! This is a whitebox parameter
        self.gamma = float(self.cfg_dict.get('gamma', 0.5))
        self.z_threshold = float(self.cfg_dict.get('z_threshold', 4.0))
        self.expect_green_size = float(self.cfg_dict.get('expect_green_size', self.gamma))

        # Attack parameters
        self.max_edit_ratio = float(self.cfg_dict.get('max_edit_ratio', 0.3))
        self.sentence_up_num = float(self.cfg_dict.get('sentence_up_num', 1.0))
        self.sentence_down_num = float(self.cfg_dict.get('sentence_down_num', 0.9))
        self.min_completion_chars = int(self.cfg_dict.get('min_completion_chars', 50))
        self.max_generation_attempts_multiplier = int(
            self.cfg_dict.get('max_generation_attempts_multiplier', 5)
        )
        self.min_green_num = int(self.cfg_dict.get('min_green_num', 2000))
        self.removal_key_strategy = (
            str(self.cfg_dict.get('removal_key_strategy', 'max_green')).strip() or 'max_green'
        )
        self.removal_backend = (
            str(self.cfg_dict.get('removal_backend', 'greedy')).strip().lower() or 'greedy'
        )
        gumbel_cfg = dict(self.cfg_dict.get('gumbel_removal', {}))
        self.gumbel_removal = {
            'steps': int(gumbel_cfg.get('steps', 50)),
            'temperature': float(gumbel_cfg.get('temperature', 1.0)),
            'temperature_min': float(gumbel_cfg.get('temperature_min', 0.1)),
            'temperature_decay': float(gumbel_cfg.get('temperature_decay', 0.95)),
            'candidates_per_token': int(gumbel_cfg.get('candidates_per_token', 8)),
            'max_positions': int(gumbel_cfg.get('max_positions', 32)),
            'lr': float(gumbel_cfg.get('lr', 0.1)),
            'watermark_weight': float(gumbel_cfg.get('watermark_weight', 1.0)),
            'fluency_weight': float(gumbel_cfg.get('fluency_weight', 1.0)),
            'edit_rate_weight': float(gumbel_cfg.get('edit_rate_weight', 0.0)),
            'allow_noop_candidate': bool(gumbel_cfg.get('allow_noop_candidate', True)),
        }

        self.output_dir = self.cfg_dict.get('output_dir', 'output/models/mip_result')
        self.stolen_greenlist_path = self.cfg_dict.get('stolen_greenlist_path')

        raw_key_list = self.cfg_dict.get('key_list', self.cfg_dict.get('key_ids', []))
        if isinstance(raw_key_list, str):
            parsed_key_list = [
                segment.strip() for segment in raw_key_list.split(',') if segment.strip()
            ]
        else:
            parsed_key_list = list(raw_key_list or [])
        if parsed_key_list:
            self.key_list = [int(key_id) for key_id in parsed_key_list]
        else:
            key_count = int(self.cfg_dict.get('key_count', self.cfg_dict.get('key_num', 1)))
            if key_count < 1:
                raise ValueError('[MIP] key_count must be at least 1.')
            self.key_list = list(range(key_count))
        self.key_count = len(self.key_list)
        self.multi_key_iterations = int(self.cfg_dict.get('multi_key_iterations', 5))
        self.multi_key_init = str(self.cfg_dict.get('multi_key_init', 'random')).strip() or 'random'
        self.multi_key_reassign = (
            str(self.cfg_dict.get('multi_key_reassign', 'max_green')).strip() or 'max_green'
        )
        self.multi_key_mu = int(self.cfg_dict.get('multi_key_mu', self.min_green_num))
        self.use_token_weights_in_multikey = bool(
            self.cfg_dict.get('use_token_weights_in_multikey', False)
        )

        stealing_params = dict(self.cfg_dict.get('stealing_parameters', {}))
        self.stealing_parameters = stealing_params
        self.use_prebuilt_corpus = bool(stealing_params.get('use_prebuilt_corpus', False))
        self.corpus_path = stealing_params.get('corpus_path')
        self.corpus_base_path = stealing_params.get('corpus_base_path')
        self.corpus_wm_path = stealing_params.get('corpus_wm_path')
        self.auto_discover_corpus = bool(stealing_params.get('auto_discover_corpus', True))
        self.corpus_dir = str(stealing_params.get('corpus_dir', 'output/corpus') or 'output/corpus')
        self.corpus_prompt_field = str(stealing_params.get('corpus_prompt_field', 'prompt'))
        self.corpus_base_field = str(stealing_params.get('corpus_base_field', 'base_completion'))
        self.corpus_wm_field = str(stealing_params.get('corpus_wm_field', 'wm_completion'))
        self.save_prepare_artifact = bool(stealing_params.get('save_prepare_artifact', True))
        self.prepare_artifact_path = str(stealing_params.get('prepare_artifact_path', '') or '')

    @property
    def algorithm_name(self) -> str:
        return 'MIP'


class WatermarkRemover:
    def __init__(
        self,
        tokenizer,
        greenlist_path: str,
        device: str = 'cuda',
        removal_key_strategy: str = 'max_green',
        language_model: Any | None = None,
        removal_backend: str = 'greedy',
        gumbel_config: dict[str, Any] | None = None,
    ):
        self.device = device
        self.tokenizer = tokenizer
        self.removal_key_strategy = removal_key_strategy
        self.removal_backend = removal_backend
        self.language_model = language_model
        self.gumbel_config = dict(gumbel_config or {})
        self.last_attack_meta: dict[str, Any] = {}

        logger.info('[Remover] Initializing GensimModel...')
        self.gensim = GensimModel()

        self.green_tokens_by_key: dict[str, set[int]] = {}
        self.key_order: list[str] = []

        if os.path.exists(greenlist_path):
            with open(greenlist_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            keyed_data = self._normalize_greenlist_payload(data)
            self.key_order = self._sorted_keys(keyed_data)
            self.green_tokens_by_key = {
                key_id: self._extract_green_tokens(keyed_data[key_id]) for key_id in self.key_order
            }
            logger.info(
                '[Remover] Loaded %s key(s) from greenlist: %s',
                len(self.key_order),
                ', '.join(self.key_order),
            )
        else:
            logger.error(f'[Remover] Greenlist file not found at {greenlist_path}')
            raise FileNotFoundError(f'Greenlist file not found: {greenlist_path}')

    def _normalize_greenlist_payload(self, data: Any) -> dict[str, Any]:
        if isinstance(data, list):
            return {'0': data}
        if isinstance(data, dict):
            if data and all(isinstance(value, dict) for value in data.values()):
                return {str(key_id): value for key_id, value in data.items()}
            if (
                len(data) == 1
                and '0' in data
                and isinstance(data['0'], (dict, list))
                and not (
                    isinstance(data['0'], list)
                    and len(data['0']) == 2
                    and not isinstance(data['0'][0], (dict, list))
                )
            ):
                return {'0': data['0']}
            return {'0': data}
        raise TypeError('[Remover] Unsupported greenlist payload format.')

    def _extract_green_tokens(self, key_data: Any) -> set[int]:
        green_tokens: set[int] = set()
        if isinstance(key_data, list):
            return {int(token_id) for token_id in key_data}
        if not isinstance(key_data, dict):
            return green_tokens

        for token_id, info in key_data.items():
            is_green = info[1] if isinstance(info, list) and len(info) > 1 else info
            if int(is_green) == 1:
                green_tokens.add(int(token_id))
        return green_tokens

    def _sorted_keys(self, keyed_data: dict[str, Any]) -> list[str]:
        return sorted(
            (str(key_id) for key_id in keyed_data),
            key=lambda key_id: (0, int(key_id)) if str(key_id).isdigit() else (1, str(key_id)),
        )

    def _count_green_hits_by_key(self, token_ids: list[int]) -> dict[str, int]:
        return {
            key_id: sum(1 for token_id in token_ids if token_id in self.green_tokens_by_key[key_id])
            for key_id in self.key_order
        }

    def _select_sentence_key(self, token_ids: list[int]) -> tuple[str | None, dict[str, int]]:
        green_hits_by_key = self._count_green_hits_by_key(token_ids)
        if not green_hits_by_key:
            return None, {}
        if self.removal_key_strategy != 'max_green':
            raise ValueError(
                f'[Remover] Unsupported removal_key_strategy: {self.removal_key_strategy}'
            )

        selected_key = max(
            self.key_order,
            key=lambda key_id: (green_hits_by_key.get(key_id, 0), -self.key_order.index(key_id)),
        )
        return selected_key, green_hits_by_key

    def is_green(self, token_id: int, key_id: str | None = None) -> bool:
        if key_id is None:
            return any(
                token_id in green_tokens for green_tokens in self.green_tokens_by_key.values()
            )
        return token_id in self.green_tokens_by_key.get(str(key_id), set())

    def _empty_attack_meta(self) -> dict[str, Any]:
        return {
            'key_count': len(self.key_order),
            'selected_key': None,
            'removal_key_strategy': self.removal_key_strategy,
            'removal_backend': self.removal_backend,
            'green_hits_by_key': {},
            'green_hits_after_by_key': {},
            'green_hits_before': 0,
            'green_hits_after': 0,
        }

    def _resolve_language_model(self):
        if self.language_model is None:
            return None, None, None
        if hasattr(self.language_model, 'model') and hasattr(
            self.language_model.model, 'embed_tokens'
        ):
            embed_tokens = self.language_model.model.embed_tokens
        elif (
            hasattr(self.language_model, 'model')
            and hasattr(self.language_model.model, 'decoder')
            and hasattr(self.language_model.model.decoder, 'embed_tokens')
        ):
            embed_tokens = self.language_model.model.decoder.embed_tokens
        else:
            raise ValueError('[Remover] Unsupported language model embedding layout for gumbel.')
        model_device = getattr(self.language_model, 'device', None)
        if model_device is None:
            model_device = next(embed_tokens.parameters()).device
        return self.language_model, embed_tokens, model_device

    def _candidate_token_ids(
        self,
        token_id: int,
        key_id: str,
        *,
        candidates_per_token: int,
        allow_noop_candidate: bool,
    ) -> list[int]:
        candidate_ids: list[int] = []
        if allow_noop_candidate:
            candidate_ids.append(int(token_id))

        original_token_str = self.tokenizer.decode([token_id])
        original_word = original_token_str.strip()
        has_leading_space = original_token_str.startswith(' ')
        if len(original_word) >= 2:
            synonyms = self.gensim.find_simi_words(original_word, simi_num=candidates_per_token * 2)
            for syn in synonyms:
                candidate_str = f' {syn}' if has_leading_space else syn
                try:
                    syn_ids = self.tokenizer.encode(candidate_str, add_special_tokens=False)
                except Exception:
                    continue
                if len(syn_ids) != 1:
                    continue
                syn_id = int(syn_ids[0])
                if syn_id in candidate_ids:
                    continue
                if self.is_green(syn_id, key_id):
                    continue
                candidate_ids.append(syn_id)
                if len(candidate_ids) >= candidates_per_token + int(allow_noop_candidate):
                    break
        return candidate_ids

    def _build_gumbel_plan(
        self,
        input_ids: list[int],
        selected_key: str | None,
        *,
        max_changes: float,
    ) -> list[dict[str, Any]]:
        if selected_key is None:
            return []
        max_change_count = max(1, int(len(input_ids) * max_changes))
        max_positions = min(self.gumbel_config.get('max_positions', 32), max_change_count)
        candidates_per_token = max(1, self.gumbel_config.get('candidates_per_token', 8))
        allow_noop_candidate = bool(self.gumbel_config.get('allow_noop_candidate', True))

        editable_plan: list[dict[str, Any]] = []
        for position, token_id in enumerate(input_ids):
            if len(editable_plan) >= max_positions:
                break
            if not self.is_green(token_id, selected_key):
                continue
            candidate_ids = self._candidate_token_ids(
                token_id,
                selected_key,
                candidates_per_token=candidates_per_token,
                allow_noop_candidate=allow_noop_candidate,
            )
            if len(candidate_ids) <= int(allow_noop_candidate):
                continue
            green_mask = [
                1.0 if self.is_green(candidate_id, selected_key) else 0.0
                for candidate_id in candidate_ids
            ]
            edit_mask = [0.0 if candidate_id == token_id else 1.0 for candidate_id in candidate_ids]
            editable_plan.append(
                {
                    'position': position,
                    'original_token_id': int(token_id),
                    'candidate_ids': candidate_ids,
                    'green_mask': green_mask,
                    'edit_mask': edit_mask,
                }
            )
        return editable_plan

    def attack_gumbel(self, text: str, max_changes: float = 0.2) -> tuple[str, int]:
        input_ids = self.tokenizer.encode(text, add_special_tokens=False)
        if not input_ids:
            self.last_attack_meta = self._empty_attack_meta()
            return text, 0

        selected_key, green_hits_by_key = self._select_sentence_key(input_ids)
        editable_plan = self._build_gumbel_plan(input_ids, selected_key, max_changes=max_changes)
        if not editable_plan:
            green_hits_after_by_key = self._count_green_hits_by_key(input_ids)
            self.last_attack_meta = {
                'key_count': len(self.key_order),
                'selected_key': selected_key,
                'removal_key_strategy': self.removal_key_strategy,
                'removal_backend': 'gumbel',
                'green_hits_by_key': green_hits_by_key,
                'green_hits_after_by_key': green_hits_after_by_key,
                'green_hits_before': green_hits_by_key.get(selected_key, 0) if selected_key else 0,
                'green_hits_after': green_hits_after_by_key.get(selected_key, 0)
                if selected_key
                else 0,
                'num_optimized_positions': 0,
            }
            return text, 0

        model, embed_tokens, model_device = self._resolve_language_model()
        if model is None:
            logger.warning(
                '[Remover] No language model available for gumbel backend; falling back to greedy.'
            )
            return self.attack_greedy(text, max_changes=max_changes)
        model.eval()

        target_ids = torch.tensor([input_ids], dtype=torch.long, device=model_device)
        with torch.no_grad():
            raw_embedding = embed_tokens(target_ids).detach()

        parameters: list[torch.nn.Parameter] = []
        for plan in editable_plan:
            init_logits = torch.zeros(len(plan['candidate_ids']), device=model_device)
            if plan['candidate_ids'][0] == plan['original_token_id']:
                init_logits[0] = 1.0
            parameters.append(torch.nn.Parameter(init_logits))

        optimizer = torch.optim.Adam(parameters, lr=float(self.gumbel_config.get('lr', 0.1)))
        temperature = float(self.gumbel_config.get('temperature', 1.0))
        temperature_min = float(self.gumbel_config.get('temperature_min', 0.1))
        temperature_decay = float(self.gumbel_config.get('temperature_decay', 0.95))
        watermark_weight = float(self.gumbel_config.get('watermark_weight', 1.0))
        fluency_weight = float(self.gumbel_config.get('fluency_weight', 1.0))
        edit_rate_weight = float(self.gumbel_config.get('edit_rate_weight', 0.0))
        steps = max(1, int(self.gumbel_config.get('steps', 50)))
        final_loss_value = None

        for _ in range(steps):
            optimizer.zero_grad()
            new_embedding = raw_embedding.clone()
            watermark_loss = torch.tensor(0.0, device=model_device)
            edit_mass = torch.tensor(0.0, device=model_device)

            for plan, logits in zip(editable_plan, parameters):
                candidate_ids = torch.tensor(
                    plan['candidate_ids'], dtype=torch.long, device=model_device
                )
                candidate_embs = embed_tokens(candidate_ids)
                probs = F.gumbel_softmax(logits, tau=temperature, hard=False)
                replacement_emb = torch.sum(probs.unsqueeze(-1) * candidate_embs, dim=0)
                new_embedding[0, plan['position'], :] = replacement_emb

                green_mask = torch.tensor(
                    plan['green_mask'], dtype=torch.float32, device=model_device
                )
                edit_mask = torch.tensor(
                    plan['edit_mask'], dtype=torch.float32, device=model_device
                )
                watermark_loss = watermark_loss + torch.sum(probs * green_mask)
                edit_mass = edit_mass + torch.sum(probs * edit_mask)

            outputs = model(inputs_embeds=new_embedding, labels=target_ids)
            fluency_loss = outputs.loss
            total_loss = (
                fluency_weight * fluency_loss
                + watermark_weight * watermark_loss
                + edit_rate_weight * edit_mass
            )
            total_loss.backward()
            optimizer.step()
            final_loss_value = float(total_loss.detach().cpu())
            temperature = max(temperature * temperature_decay, temperature_min)

        new_ids = list(input_ids)
        changes = 0
        for plan, logits in zip(editable_plan, parameters):
            best_idx = int(torch.argmax(logits).item())
            selected_token_id = int(plan['candidate_ids'][best_idx])
            new_ids[plan['position']] = selected_token_id
            if selected_token_id != plan['original_token_id']:
                changes += 1

        attacked_text = self.tokenizer.decode(new_ids)
        green_hits_after_by_key = self._count_green_hits_by_key(new_ids)
        self.last_attack_meta = {
            'key_count': len(self.key_order),
            'selected_key': selected_key,
            'removal_key_strategy': self.removal_key_strategy,
            'removal_backend': 'gumbel',
            'green_hits_by_key': green_hits_by_key,
            'green_hits_after_by_key': green_hits_after_by_key,
            'green_hits_before': green_hits_by_key.get(selected_key, 0) if selected_key else 0,
            'green_hits_after': green_hits_after_by_key.get(selected_key, 0) if selected_key else 0,
            'num_optimized_positions': len(editable_plan),
            'optimization_steps': steps,
            'final_loss': final_loss_value,
        }
        return attacked_text, changes

    def attack_greedy(self, text: str, max_changes: float = 0.2) -> tuple:
        input_ids = self.tokenizer.encode(text, add_special_tokens=False)
        if not input_ids:
            self.last_attack_meta = self._empty_attack_meta()
            return text, 0
        new_ids = input_ids.copy()
        selected_key, green_hits_by_key = self._select_sentence_key(input_ids)

        changes = 0
        max_change_count = int(len(input_ids) * max_changes)

        for i, token_id in enumerate(input_ids):
            if changes >= max_change_count:
                break

            if selected_key is not None and self.is_green(token_id, selected_key):
                original_token_str = self.tokenizer.decode([token_id])
                original_word = original_token_str.strip()

                has_leading_space = original_token_str.startswith(' ')

                if len(original_word) < 2:
                    continue

                synonyms = self.gensim.find_simi_words(original_word, simi_num=10)

                for syn in synonyms:
                    candidate_str = syn
                    if has_leading_space:
                        candidate_str = ' ' + syn

                    syn_ids = self.tokenizer.encode(candidate_str, add_special_tokens=False)

                    if len(syn_ids) == 1:
                        syn_id = syn_ids[0]

                        if not self.is_green(syn_id, selected_key):
                            new_ids[i] = syn_id
                            changes += 1
                            break

        attacked_text = self.tokenizer.decode(new_ids)
        green_hits_after_by_key = self._count_green_hits_by_key(new_ids)
        self.last_attack_meta = {
            'key_count': len(self.key_order),
            'selected_key': selected_key,
            'removal_key_strategy': self.removal_key_strategy,
            'removal_backend': 'greedy',
            'green_hits_by_key': green_hits_by_key,
            'green_hits_after_by_key': green_hits_after_by_key,
            'green_hits_before': green_hits_by_key.get(selected_key, 0) if selected_key else 0,
            'green_hits_after': green_hits_after_by_key.get(selected_key, 0) if selected_key else 0,
        }
        return attacked_text, changes


class MIPAttack(BaseAttack):
    """
    MIP Attack Integration for MarkSec Framework.
    Lifecycle:
    1. __init__: Load config.
    2. prepare(): Check greenlist -> Generate Data -> Solve MIP -> Save Greenlist.
    3. edit(): Load Remover -> Execute Greedy Attack.
    """

    def __init__(
        self,
        config: MIPConfig,
        watermark_instance=None,
        tokenizer=None,
        base_model=None,
        dataset=None,
        watermark_name: str = '',
        **kwargs,
    ):
        super().__init__(watermark_model=base_model, watermark_tokenizer=tokenizer, **kwargs)
        self.config = config
        self.watermark_instance = watermark_instance
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.base_model = base_model
        self._last_attack_meta: dict[str, Any] = {}
        self.watermark_name = watermark_name
        self._explicit_greenlist_path = bool(str(self.config.stolen_greenlist_path or '').strip())
        self._explicit_prepare_artifact_path = bool(
            str(self.config.prepare_artifact_path or '').strip()
        )
        self._artifact_identity: dict[str, Any] = {}
        self._artifact_identity_digest = ''

        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)

        self.greenlist_path = ''
        self.prepare_artifact_path = ''
        self._refresh_artifact_paths()
        self.remover = None

    def prepare(self) -> bool:
        """
        Stealing Phase: Train the proxy model (solve MIP) if needed.
        """
        if self.config.use_prebuilt_corpus:
            self._ensure_prebuilt_corpus_path()
        self._refresh_artifact_paths()

        if os.path.exists(self.greenlist_path):
            logger.info(
                f'[MIP] Found existing greenlist at {self.greenlist_path}. Skipping stealing.'
            )
            self._save_prepare_artifact(
                source='reuse_existing',
                pair_count=0,
                corpus_path=self.config.corpus_path,
            )
            self._init_remover()
            return True

        if self.config.use_prebuilt_corpus:
            logger.info('[MIP] Loading prebuilt stealing corpus.')
            wm_token_ids, nl_token_ids = self._load_prebuilt_corpus()
            stealing_source = 'prebuilt_corpus'
        else:
            logger.info(
                f'[MIP] No greenlist found at {self.greenlist_path}. Starting Stealing Phase '
                f'({self.config.stealing_samples} samples)...'
            )
            wm_token_ids, nl_token_ids = self._generate_stealing_corpus()
            stealing_source = 'online_query'

        if not wm_token_ids or not nl_token_ids:
            raise ValueError('[MIP] No valid stealing pairs available for solver.')

        logger.info(f'[MIP] Generated {len(wm_token_ids)} valid pairs. Starting Solver...')

        self._run_solver(wm_token_ids, nl_token_ids)
        self._save_prepare_artifact(
            source=stealing_source,
            pair_count=len(wm_token_ids),
            corpus_path=self.config.corpus_path,
        )

        self._init_remover()
        return True

    def pop_last_attack_meta(self) -> dict[str, Any]:
        meta = dict(self._last_attack_meta)
        self._last_attack_meta.clear()
        return meta

    def _infer_model_short_name(self) -> str:
        try:
            if self.base_model is not None and hasattr(self.base_model, 'name_or_path'):
                model_name_str = self.base_model.name_or_path
            elif (
                self.base_model is not None
                and hasattr(self.base_model, 'config')
                and hasattr(self.base_model.config, '_name_or_path')
            ):
                model_name_str = self.base_model.config._name_or_path
            elif self.tokenizer is not None and hasattr(self.tokenizer, 'name_or_path'):
                model_name_str = self.tokenizer.name_or_path
            else:
                model_name_str = 'unknown_model'
            return str(model_name_str).split('/')[-1]
        except Exception as e:
            logger.warning(f"Failed to infer model name: {e}. Using 'default'.")
            return 'default'

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

        config_values = {}
        if hasattr(watermark_config, '__dict__'):
            config_values = {
                key: value
                for key, value in vars(watermark_config).items()
                if not str(key).startswith('_')
            }
        if config_values:
            payload['watermark_config'] = self._normalize_identity_value(config_values)
        return payload

    def _build_artifact_identity(self) -> dict[str, Any]:
        corpus_path = self._resolve_identity_path(self.config.corpus_path)
        corpus_base_path = self._resolve_identity_path(self.config.corpus_base_path)
        corpus_wm_path = self._resolve_identity_path(self.config.corpus_wm_path)
        return {
            'attack': 'MIP',
            'model_hint': infer_model_hint(model=self.base_model, tokenizer=self.tokenizer),
            'model_short_name': self._infer_model_short_name(),
            'watermark_name': str(self.watermark_name or ''),
            'watermark_identity': self._build_watermark_identity_payload(),
            'gamma': float(self.config.gamma),
            'key_count': int(self.config.key_count),
            'key_list': [int(key_id) for key_id in self.config.key_list],
            'attack_mode': str(self.config.attack_mode),
            'use_prebuilt_corpus': bool(self.config.use_prebuilt_corpus),
            'corpus_path': corpus_path,
            'corpus_base_path': corpus_base_path,
            'corpus_wm_path': corpus_wm_path,
            'corpus_path_fingerprint': self._fingerprint_path(corpus_path),
            'corpus_base_path_fingerprint': self._fingerprint_path(corpus_base_path),
            'corpus_wm_path_fingerprint': self._fingerprint_path(corpus_wm_path),
        }

    def _build_default_artifact_stem(self) -> str:
        wm_tag = self._sanitize_artifact_component(
            infer_watermark_hint(
                watermark_name=self.watermark_name,
                watermark_instance=self.watermark_instance,
            )
            or 'unknown_watermark'
        )
        model_tag = self._sanitize_artifact_component(self._infer_model_short_name())
        mode_tag = self._sanitize_artifact_component(self.config.attack_mode)
        digest_tag = self._artifact_identity_digest[:12] or 'unknown'
        key_suffix = f'_k{self.config.key_count}' if self.config.key_count > 1 else ''
        return f'{model_tag}_{wm_tag}_{mode_tag}_g{self.config.gamma}{key_suffix}_{digest_tag}'

    def _refresh_artifact_paths(self) -> None:
        self._artifact_identity = self._build_artifact_identity()
        self._artifact_identity_digest = self._hash_identity_payload(self._artifact_identity)
        default_stem = self._build_default_artifact_stem()

        if self._explicit_greenlist_path:
            self.greenlist_path = str(Path(self.config.stolen_greenlist_path))
        else:
            self.greenlist_path = str(
                Path(self.config.output_dir) / f'stolen_greenlist_{default_stem}.json'
            )

        if self._explicit_prepare_artifact_path:
            self.prepare_artifact_path = str(Path(self.config.prepare_artifact_path))
        else:
            self.prepare_artifact_path = str(
                Path(self.config.output_dir) / f'prepare_artifact_{default_stem}.json'
            )

    def _get_dataset_prompt(self, index: int) -> str:
        if self.dataset is None:
            raise ValueError('[MIP] Dataset is required to generate stealing corpus.')
        if hasattr(self.dataset, 'get_prompt'):
            return str(self.dataset.get_prompt(index))
        sample = self.dataset[index]
        return str(getattr(sample, 'prompt', ''))

    def _extract_completion(self, full_text: str, prompt: str) -> str:
        stripped = strip_prompt(full_text, prompt).strip()
        return stripped or str(full_text).strip()

    def _extract_record_text(
        self,
        record: dict[str, Any],
        explicit_field: str,
        fallback_fields: list[str],
    ) -> str:
        prompt = str(record.get(self.config.corpus_prompt_field, '') or '')
        for field_name in [explicit_field, *fallback_fields]:
            value = record.get(field_name)
            if not isinstance(value, str):
                continue
            candidate = value.strip()
            if not candidate:
                continue
            if prompt:
                completion = self._extract_completion(candidate, prompt)
                if completion:
                    return completion
            return candidate
        return ''

    def _tokenize_pair(self, wm_text: str, nl_text: str) -> tuple[list[int], list[int]] | None:
        if self.tokenizer is None or not hasattr(self.tokenizer, 'encode'):
            raise ValueError('[MIP] Tokenizer with encode() is required for MIP.')

        if (
            len(wm_text.strip()) < self.config.min_completion_chars
            or len(nl_text.strip()) < self.config.min_completion_chars
        ):
            return None

        wm_ids = self.tokenizer.encode(wm_text, add_special_tokens=False)
        nl_ids = self.tokenizer.encode(nl_text, add_special_tokens=False)
        if not wm_ids or not nl_ids:
            return None
        return wm_ids, nl_ids

    def _ensure_prebuilt_corpus_path(self) -> None:
        if self.config.corpus_path:
            self.config.corpus_path = os.path.abspath(str(self.config.corpus_path))
            return
        if self.config.corpus_base_path and self.config.corpus_wm_path:
            self.config.corpus_base_path = os.path.abspath(str(self.config.corpus_base_path))
            self.config.corpus_wm_path = os.path.abspath(str(self.config.corpus_wm_path))
            return
        if not self.config.auto_discover_corpus:
            return

        watermark_hint = infer_watermark_hint(
            watermark_name=self.watermark_name,
            watermark_instance=self.watermark_instance,
        )
        model_hint = infer_model_hint(model=self.base_model, tokenizer=self.tokenizer)
        dataset_hint = infer_dataset_hint(self.dataset)

        discovered = discover_joint_corpus_path(
            corpus_dir=self.config.corpus_dir,
            model_hint=model_hint,
            dataset_hint=dataset_hint,
            watermark_hint=watermark_hint,
        )
        if discovered:
            self.config.corpus_path = discovered
            logger.info('[MIP] Auto-selected prebuilt corpus: %s', self.config.corpus_path)
            return
        if watermark_hint and requires_exact_watermark_match(watermark_hint):
            raise FileNotFoundError(
                '[MIP] No exact prebuilt corpus match found for '
                f'model={model_hint or "unknown"} dataset={dataset_hint or "unknown"} '
                f'watermark={watermark_hint}. Refusing to reuse a different setting.'
            )

    def _load_prebuilt_corpus(self) -> tuple[list[list[int]], list[list[int]]]:
        wm_token_ids: list[list[int]] = []
        nl_token_ids: list[list[int]] = []

        if self.config.corpus_path:
            with open(self.config.corpus_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if len(wm_token_ids) >= self.config.stealing_samples:
                        break
                    if not line.strip():
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    wm_text = self._extract_record_text(
                        record,
                        self.config.corpus_wm_field,
                        ['generated_only', 'wm_completion', 'watermarked_text', 'wm_full_text'],
                    )
                    nl_text = self._extract_record_text(
                        record,
                        self.config.corpus_base_field,
                        ['base_completion', 'clean_text', 'natural_text', 'base_full_text'],
                    )
                    pair = self._tokenize_pair(wm_text, nl_text)
                    if pair is None:
                        continue
                    wm_ids, nl_ids = pair
                    wm_token_ids.append(wm_ids)
                    nl_token_ids.append(nl_ids)
        elif self.config.corpus_base_path and self.config.corpus_wm_path:
            with open(self.config.corpus_wm_path, 'r', encoding='utf-8') as wm_file:
                wm_lines = [line for line in wm_file if line.strip()]
            with open(self.config.corpus_base_path, 'r', encoding='utf-8') as nl_file:
                nl_lines = [line for line in nl_file if line.strip()]

            for wm_line, nl_line in zip(wm_lines, nl_lines):
                if len(wm_token_ids) >= self.config.stealing_samples:
                    break
                try:
                    wm_record = json.loads(wm_line)
                    nl_record = json.loads(nl_line)
                except json.JSONDecodeError:
                    continue

                wm_text = self._extract_record_text(
                    wm_record,
                    self.config.corpus_wm_field,
                    ['generated_only', 'wm_completion', 'watermarked_text', 'wm_full_text'],
                )
                nl_text = self._extract_record_text(
                    nl_record,
                    self.config.corpus_base_field,
                    ['base_completion', 'clean_text', 'natural_text', 'base_full_text'],
                )
                pair = self._tokenize_pair(wm_text, nl_text)
                if pair is None:
                    continue
                wm_ids, nl_ids = pair
                wm_token_ids.append(wm_ids)
                nl_token_ids.append(nl_ids)
        else:
            raise ValueError(
                '[MIP] Prebuilt corpus mode requires corpus_path or both corpus_base_path and corpus_wm_path.'
            )

        return wm_token_ids, nl_token_ids

    def _generate_stealing_corpus(self) -> tuple[list[list[int]], list[list[int]]]:
        if self.dataset is None or self.watermark_instance is None:
            raise ValueError(
                '[MIP] Dataset and watermark_instance are required when no prebuilt corpus is configured.'
            )

        wm_token_ids: list[list[int]] = []
        nl_token_ids: list[list[int]] = []
        valid_count = 0

        max_attempts = min(
            len(self.dataset),
            self.config.stealing_samples * self.config.max_generation_attempts_multiplier,
        )

        for i in tqdm(range(max_attempts), desc='MIP Data Gen'):
            if valid_count >= self.config.stealing_samples:
                break

            try:
                prompt = self._get_dataset_prompt(i)
                wm_text = self.watermark_instance.generate_watermarked_text(prompt=prompt)
                nl_text = self.watermark_instance.generate_unwatermarked_text(prompt=prompt)

                wm_gen = self._extract_completion(wm_text, prompt)
                nl_gen = self._extract_completion(nl_text, prompt)
                pair = self._tokenize_pair(wm_gen, nl_gen)
                if pair is None:
                    continue

                wm_ids, nl_ids = pair
                wm_token_ids.append(wm_ids)
                nl_token_ids.append(nl_ids)
                valid_count += 1
            except Exception:
                continue

        return wm_token_ids, nl_token_ids

    def _run_solver(self, wm_token_ids, nl_token_ids):
        """Setup and run GreenlistInversor."""

        min_len = min(len(wm_token_ids), len(nl_token_ids))
        wm_token_ids = wm_token_ids[:min_len]
        nl_token_ids = nl_token_ids[:min_len]

        vocab_size = getattr(getattr(self.base_model, 'config', None), 'vocab_size', None)
        if vocab_size is None:
            vocab_size = getattr(self.tokenizer, 'vocab_size', None)
        if vocab_size is None:
            raise ValueError('[MIP] Unable to infer vocab size for solver.')
        logger.info(f'[MIP] Using vocab size: {vocab_size}')

        attacker = GreenlistInversorPlus(tokenizer=self.tokenizer, vocab_size=vocab_size)
        attacker.gamma = self.config.gamma
        attacker.z_threshold = self.config.z_threshold
        attacker.token_index = list(range(vocab_size))
        attacker.token_dict = {token_id: 1 for token_id in attacker.token_index}

        attacker.dataset = wm_token_ids + nl_token_ids
        attacker.wm_size = min_len
        attacker.nl_size = min_len
        attacker.dataset_size = len(attacker.dataset)
        attacker.wm_sentence_index = list(range(min_len))
        attacker.nl_sentence_index = list(range(min_len, min_len * 2))
        attacker.sentence_len_list = [len(s) for s in attacker.dataset]

        attacker.set_solver(
            TimeLimit=self.config.time_limit,
            solver_type=self.config.solver_type,
            MIPGap=self.config.mip_gap,
        )

        is_as1 = str(self.config.attack_mode).upper() == 'AS1'
        min_green_num = min(
            self.config.multi_key_mu if self.config.key_count > 1 else self.config.min_green_num,
            max(
                1,
                int(vocab_size * max(self.config.gamma, 1e-6) * self.config.expect_green_size),
            ),
        )

        if self.config.key_count <= 1:
            logger.info('[MIP] Building Optimization Model...')
            attacker.set_model(
                sentence_up_num=self.config.sentence_up_num,
                sentence_down_num=self.config.sentence_down_num,
                gamma_flag=is_as1,
                oracle_flag=False,
                expect_green_size=self.config.expect_green_size,
                key_list=self.config.key_list,
                min_green_num=min_green_num,
                use_token_weights=True,
            )

            logger.info('[MIP] Solving Stage 1...')
            attacker.solve_wm_nl_sum_green(
                TimeLimit=self.config.time_limit,
                MIPGap=self.config.mip_gap,
                lock_sentence=False,
            )

            logger.info('[MIP] Solving Stage 2...')
            attacker.solve_green_num(
                max_min=False, TimeLimit=self.config.time_limit, MIPGap=self.config.mip_gap
            )
            self._save_stolen_result(attacker)
            return

        sentence_assignment = self._build_initial_wm_sentence_assignment(attacker)
        max_iterations = max(1, self.config.multi_key_iterations)
        final_assignment = sentence_assignment

        for iteration_idx in range(max_iterations):
            logger.info(
                '[MIP] Building multi-key optimization model (iteration %s/%s)...',
                iteration_idx + 1,
                max_iterations,
            )
            attacker.set_model(
                sentence_up_num=self.config.sentence_up_num,
                sentence_down_num=self.config.sentence_down_num,
                gamma_flag=is_as1,
                oracle_flag=False,
                expect_green_size=self.config.expect_green_size,
                key_list=self.config.key_list,
                min_green_num=min_green_num,
                fixed_wm_sentence_key=sentence_assignment,
                use_token_weights=self.config.use_token_weights_in_multikey,
            )

            logger.info('[MIP] Solving multi-key Stage 1 (iteration %s)...', iteration_idx + 1)
            attacker.solve_wm_nl_sum_green(
                TimeLimit=self.config.time_limit,
                MIPGap=self.config.mip_gap,
                lock_sentence=True,
            )

            logger.info('[MIP] Solving multi-key Stage 2 (iteration %s)...', iteration_idx + 1)
            attacker.solve_green_num(
                max_min=False, TimeLimit=self.config.time_limit, MIPGap=self.config.mip_gap
            )

            token_color = self._extract_token_color(attacker)
            new_assignment = self._reassign_wm_sentence_keys(attacker, token_color)
            changed_count = self._count_assignment_changes(sentence_assignment, new_assignment)
            logger.info(
                '[MIP] Multi-key sentence reassignment changed %s/%s watermark sentences.',
                changed_count,
                len(attacker.wm_sentence_index),
            )
            final_assignment = new_assignment
            if changed_count == 0:
                logger.info(
                    '[MIP] Multi-key assignment converged after %s iteration(s).', iteration_idx + 1
                )
                break
            sentence_assignment = new_assignment

        attacker.multi_key_sentence_assignment = dict(final_assignment)
        self._save_stolen_result(attacker)

    def _build_initial_wm_sentence_assignment(
        self, attacker: GreenlistInversorPlus
    ) -> dict[tuple[int, int], int]:
        assignment = {
            (sentence_id, key_id): 0
            for sentence_id in attacker.wm_sentence_index
            for key_id in self.config.key_list
        }
        key_count = len(self.config.key_list)

        init_mode = str(self.config.multi_key_init).strip().lower()
        if init_mode not in {'random', 'balanced'}:
            logger.warning(
                "[MIP] Unknown multi_key_init='%s'; falling back to 'random'.",
                self.config.multi_key_init,
            )
            init_mode = 'random'

        for offset, sentence_id in enumerate(attacker.wm_sentence_index):
            if init_mode == 'balanced':
                key_id = self.config.key_list[offset % key_count]
            else:
                key_id = self.config.key_list[(sentence_id + offset) % key_count]
            assignment[(sentence_id, key_id)] = 1
        return assignment

    def _extract_token_color(self, attacker: GreenlistInversorPlus) -> dict[str, dict[int, int]]:
        token_color: dict[str, dict[int, int]] = {}
        for key_id in getattr(attacker, 'key_index', []):
            key_str = str(key_id)
            token_color[key_str] = {}
            for token_id in attacker.token_index:
                try:
                    val = pyomo.value(attacker.model.key_token_color[key_id, token_id])
                    is_green = 1 if val >= 0.5 else 0
                except Exception:
                    is_green = 0
                token_color[key_str][int(token_id)] = is_green
        return token_color

    def _reassign_wm_sentence_keys(
        self,
        attacker: GreenlistInversorPlus,
        token_color: dict[str, dict[int, int]],
    ) -> dict[tuple[int, int], int]:
        strategy = str(self.config.multi_key_reassign).strip().lower()
        if strategy != 'max_green':
            raise ValueError(f'[MIP] Unsupported multi_key_reassign strategy: {strategy}')

        assignment = {
            (sentence_id, key_id): 0
            for sentence_id in attacker.wm_sentence_index
            for key_id in self.config.key_list
        }
        key_order = [str(key_id) for key_id in self.config.key_list]
        for sentence_id in attacker.wm_sentence_index:
            token_ids = attacker.dataset[sentence_id]
            key_hits = {
                key_id: sum(
                    token_color.get(key_id, {}).get(int(token_id), 0)
                    for token_id in token_ids
                    if attacker.token_dict.get(token_id, 0) == 1
                )
                for key_id in key_order
            }
            selected_key = max(
                key_order,
                key=lambda key_id: (key_hits.get(key_id, 0), -key_order.index(key_id)),
            )
            assignment[(sentence_id, int(selected_key))] = 1
        return assignment

    @staticmethod
    def _count_assignment_changes(
        previous: dict[tuple[int, int], int], current: dict[tuple[int, int], int]
    ) -> int:
        previous_selected = {
            sentence_id: key_id
            for (sentence_id, key_id), flag in previous.items()
            if int(flag) == 1
        }
        current_selected = {
            sentence_id: key_id for (sentence_id, key_id), flag in current.items() if int(flag) == 1
        }
        sentence_ids = set(previous_selected) | set(current_selected)
        return sum(previous_selected.get(sid) != current_selected.get(sid) for sid in sentence_ids)

    def _save_stolen_result(self, attacker):
        token_color: dict[str, dict[str, list[Any]]] = {}
        green_counts: dict[str, int] = {}
        for key_id in getattr(attacker, 'key_index', [0]):
            key_str = str(key_id)
            token_color[key_str] = {}
            green_counts[key_str] = 0
            for token_id in attacker.token_index:
                try:
                    val = pyomo.value(attacker.model.key_token_color[key_id, token_id])
                    is_green = 1 if val >= 0.5 else 0
                except Exception:
                    is_green = 0
                token_color[key_str][str(token_id)] = ['', is_green]
                if is_green:
                    green_counts[key_str] += 1

        with open(self.greenlist_path, 'w', encoding='utf-8') as f:
            json.dump(token_color, f, indent=4)
        logger.info(
            '[MIP] Stolen green tokens per key %s. Saved to %s',
            green_counts,
            self.greenlist_path,
        )

    def _save_prepare_artifact(self, source: str, pair_count: int, corpus_path: str | None) -> None:
        if not self.config.save_prepare_artifact:
            return
        payload = {
            'attack': 'MIP',
            'meta_version': 'mip_prepare_meta_v2',
            'source': source,
            'pair_count': int(pair_count),
            'stealing_samples_target': int(self.config.stealing_samples),
            'corpus_path': str(corpus_path or ''),
            'greenlist_path': str(self.greenlist_path),
            'attack_mode': str(self.config.attack_mode),
            'gamma': float(self.config.gamma),
            'key_count': int(self.config.key_count),
            'key_list': [int(key_id) for key_id in self.config.key_list],
            'identity_digest': self._artifact_identity_digest,
            'artifact_identity': self._normalize_identity_value(self._artifact_identity),
        }
        out_path = Path(self.prepare_artifact_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        logger.info('[MIP] Prepare artifact saved to %s', out_path)

    def _init_remover(self):
        logger.info('[MIP] Initializing Watermark Remover...')
        self.remover = WatermarkRemover(
            tokenizer=self.tokenizer,
            greenlist_path=self.greenlist_path,
            device=str(getattr(self.base_model, 'device', 'cpu')),
            removal_key_strategy=self.config.removal_key_strategy,
            language_model=self.base_model,
            removal_backend=self.config.removal_backend,
            gumbel_config=self.config.gumbel_removal,
        )

    def edit(self, text: str, **kwargs) -> str:
        """
        Removal Phase: Edit the text using the stolen greenlist.
        """
        attack_type = str(kwargs.get('attack_type') or '').strip().lower()
        if attack_type in {'spoof', 'spoofing'}:
            raise ValueError(
                '[MIP] Spoofing protocol is not supported. '
                'MIP currently only implements steal-plus-scrub removal.'
            )

        if self.remover is None:
            if os.path.exists(self.greenlist_path):
                self._init_remover()
            else:
                logger.error('[MIP] Error: Greenlist not found.')
                return text

        backend = str(self.config.removal_backend).strip().lower()
        if backend == 'greedy':
            attacked_text, changes = self.remover.attack_greedy(
                text, max_changes=self.config.max_edit_ratio
            )
        elif backend == 'gumbel':
            attacked_text, changes = self.remover.attack_gumbel(
                text, max_changes=self.config.max_edit_ratio
            )
        else:
            raise ValueError(f'[MIP] Unsupported removal_backend: {backend}')
        input_len = 0
        if self.tokenizer is not None and hasattr(self.tokenizer, 'encode'):
            input_len = len(self.tokenizer.encode(text, add_special_tokens=False))
        self._last_attack_meta = {
            'attack': 'MIP',
            'greenlist_path': self.greenlist_path,
            'num_changes': changes,
            'change_rate': (changes / input_len) if input_len > 0 else 0.0,
        }
        if getattr(self.remover, 'last_attack_meta', None):
            self._last_attack_meta.update(self.remover.last_attack_meta)
        return attacked_text
