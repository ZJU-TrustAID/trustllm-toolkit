# Copyright 2024 THU-BPM MarkLLM.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ================================================
# RandomWalk.py
# ================================================
import re
import copy
import torch
import logging
import numpy as np
from tqdm import tqdm
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk import pos_tag
from typing import Dict, Any

from transformers import T5ForConditionalGeneration, AutoTokenizer

from src.attack.base import BaseAttack, BaseConfig
from src.util.eval import QualityOracle
from src.util.eval import GrammarCorrector

logger = logging.getLogger('MarkSec')


class RandomWalkConfig(BaseConfig):
    def __init__(self, config_path: str = None, **kwargs) -> None:
        self._algorithm_name = 'random_walk'
        super().__init__(config_path, **kwargs)

    @property
    def algorithm_name(self) -> str:
        return self._algorithm_name

    def initialize_parameters(self) -> None:
        """Load parameters from self.cfg_dict."""

        self.model_requirements = self.cfg_dict.get('model_requirements', {})
        self.utility_config = self.cfg_dict.get('utility_config', {})

        atk_params = self.cfg_dict.get('attack_hyperparameters', {})
        self.total_steps: int = atk_params.get('total_steps', 200)
        self.target_valid_steps: int = atk_params.get('target_valid_steps', 70)  # Ref default
        self.span_len: int = atk_params.get('span_len', 6)
        self.backtrack_patience: int = atk_params.get('backtrack_patience', 40)  # Ref default
        self.max_attempts: int = atk_params.get('max_attempts', 500)
        self.max_total_iterations: int = atk_params.get('max_total_iterations', self.total_steps)
        quality_cfg = self.utility_config.get('quality_oracle', {})
        self.quality_tie_threshold: float = quality_cfg.get('tie_threshold', 0.02)

        self.generation_config: Dict[str, Any] = self.cfg_dict.get('generation_config', {})
        self.generation_config.setdefault('top_p', 0.95)
        self.generation_config.setdefault('do_sample', True)
        self.generation_config.setdefault('max_new_tokens', int(2 * self.span_len))  # Ref logic
        self.generation_config.setdefault('min_length', int(1.5 * self.span_len))

        op_params = self.cfg_dict.get('operational_parameters', {})
        self.device: str = op_params.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')


class RandomWalkUtil:
    """
    Helper class for RandomWalkAttack to handle text processing.
    """

    def __init__(self, device: str):
        self.gec = GrammarCorrector(device=device)

    def clean_punctuation(self, text: str) -> str:
        """
        [From Reference 'correct_text']
        Basic punctuation correction using Regex.
        """
        # Replace multiple spaces with a single space
        corrected_text = re.sub(r'\s+', ' ', text)

        # Remove space before punctuation
        corrected_text = re.sub(r'\s+([,.;!?])', r'\1', corrected_text)

        # Deduplicate punctuation (e.g., ".." -> ".")
        corrected_text = re.sub(r'(\.){2,}', '.', corrected_text)
        corrected_text = re.sub(r'(,){2,}', ',', corrected_text)
        corrected_text = re.sub(r'(!){2,}', '!', corrected_text)
        corrected_text = re.sub(r'(\?){2,}', '?', corrected_text)
        corrected_text = re.sub(r'(:){2,}', ':', corrected_text)
        corrected_text = re.sub(r'(;){2,}', ';', corrected_text)

        return corrected_text.strip()

    def filter_sentences(self, text: str) -> str:
        """
        Filters out sentences that are likely malformed or incomplete.
        """
        try:
            sentences = sent_tokenize(text)
            complete_sentences = []

            for sent in sentences:
                words = word_tokenize(sent)
                tagged_words = pos_tag(words)
                has_verb = any(tag.startswith('VB') for word, tag in tagged_words)
                is_bad = (
                    sent.endswith('.')
                    and not has_verb
                    and not bool(re.match(r'^\d+\.$', sent))
                    and len(words) < 5
                )
                if not is_bad:
                    complete_sentences.append(sent)

            return ' '.join(complete_sentences)
        except Exception:
            return text

    def post_process(self, text: str) -> str:
        """
        Combines Grammar Correction -> Punctuation Cleaning -> Filtering.
        """
        # 1. Grammar Correction (Replaces Reference's grammatical_error_correction loop)
        try:
            sentences = sent_tokenize(text)
            if sentences:
                # Use batched GPU inference
                corrected_sents = self.gec.correct_batch(sentences)
                text = ' '.join(corrected_sents)
        except Exception as e:
            logger.warning(f'GEC failed: {e}. Keeping text as is.')

        # 2. Punctuation Cleaning (From Reference)
        text = self.clean_punctuation(text)

        # 3. Filter Incomplete Sentences (From Reference)
        text = self.filter_sentences(text)

        return text


class RandomWalkAttack(BaseAttack):
    """
    Removes watermarks using the Random Walk attack.
    Integrates logic from the original paper implementation (Reference) while maintaining
    modern architecture.
    """

    def __init__(
        self,
        config: RandomWalkConfig,
        perturbation_oracle_model: T5ForConditionalGeneration,
        perturbation_oracle_tokenizer: AutoTokenizer,
        quality_oracle_model: Any,
        quality_oracle_tokenizer: AutoTokenizer,
        base_model: Any,
        tokenizer: AutoTokenizer,
        **kwargs,
    ):
        self.cfg = config

        # 1. Assign injected dependencies
        self.perturbation_model = perturbation_oracle_model.eval()
        self.perturbation_tokenizer = perturbation_oracle_tokenizer

        # 2. Initialize QualityOracle
        logger.info('Initializing QualityOracle utility internally...')
        quality_oracle_device = quality_oracle_model.device
        quality_oracle_config = dict(self.cfg.utility_config.get('quality_oracle', {}))
        quality_oracle_config.pop('tie_threshold', None)
        self.quality_oracle = QualityOracle(
            reward_model=quality_oracle_model,
            device=quality_oracle_device,
            **quality_oracle_config,
        )

        # 3. Initialize Utility Helper
        self.util = RandomWalkUtil(device=self.cfg.device)

        logger.info(f'{self.cfg.algorithm_name} attack initialized.')

    def prepare(self) -> bool:
        return True

    def _perturb_text(self, prompt: str, text: str) -> str:
        """
        Masks a random span and infills it.
        Logic aligns with Reference's mask_text but uses prompt context.
        """
        words = text.replace('\n', ' \n').split(' ')

        # Logic from Reference: handle short text
        if len(words) <= self.cfg.span_len:
            masked_text = text + ' <extra_id_0>'
        else:
            # randint upper-bound is exclusive; +1 keeps the last valid start index.
            start = np.random.randint(1, len(words) - self.cfg.span_len + 1)
            end = start + self.cfg.span_len
            masked_text = ' '.join(words[:start]) + ' <extra_id_0> ' + ' '.join(words[end:])

        final_input_text = prompt + ' ' + masked_text

        inputs = self.perturbation_tokenizer([final_input_text], return_tensors='pt').to(
            self.perturbation_model.device
        )

        with torch.inference_mode():
            outputs = self.perturbation_model.generate(**inputs, **self.cfg.generation_config)

        infilled_span = self.perturbation_tokenizer.batch_decode(outputs, skip_special_tokens=True)[
            0
        ]
        final_text = masked_text.replace('<extra_id_0>', infilled_span)
        return final_text

    def _attack_single(self, text: str, prompt: str) -> str:
        """
        Performs the Random Walk.
        Quality-oracle-driven loop with backtracking.
        """
        original_response = self.util.clean_punctuation(text)  # Ensure start clean
        n_response = copy.deepcopy(original_response)
        cached_response = copy.deepcopy(n_response)

        n_iter, valid_steps, patience = 0, 0, 0
        short_candidates, rejected_candidates, backtracks = 0, 0, 0
        log_interval = max(1, int(getattr(self.cfg, 'iteration_log_interval', 25)))

        pbar = tqdm(
            total=None,
            desc='RW Attack',
            leave=False,
            unit='iter',
        )
        logger.info(
            'RW loop start: total_steps=%s, target_valid_steps=%s, backtrack_patience=%s, max_attempts=%s, max_total_iterations=%s, tie_threshold=%s, log_interval=%s',
            self.cfg.total_steps,
            self.cfg.target_valid_steps,
            self.cfg.backtrack_patience,
            self.cfg.max_attempts,
            getattr(self.cfg, 'max_total_iterations', self.cfg.total_steps),
            getattr(self.cfg, 'quality_tie_threshold', 0.02),
            log_interval,
        )

        max_total_iterations = getattr(self.cfg, 'max_total_iterations', self.cfg.total_steps)
        while n_iter < self.cfg.total_steps or valid_steps < self.cfg.target_valid_steps:
            if n_iter >= max_total_iterations:
                logger.warning(
                    'RW stopping after iter %s: reached max_total_iterations=%s with valid_steps=%s/%s',
                    n_iter,
                    max_total_iterations,
                    valid_steps,
                    self.cfg.target_valid_steps,
                )
                break

            iter_idx = n_iter + 1
            candidate_response = self._perturb_text(prompt, n_response)

            # Safety check for empty result
            if not candidate_response or len(candidate_response.split()) < 5:
                patience += 1
                short_candidates += 1
            elif self.quality_oracle.maintain_quality(
                prompt,
                original_response,
                candidate_response,
                tie_threshold=getattr(self.cfg, 'quality_tie_threshold', 0.02),
            ):
                candidate_response = self.util.post_process(candidate_response)
                if not candidate_response or len(candidate_response.split()) < 5:
                    patience += 1
                    short_candidates += 1
                    logger.info(
                        'RW rejected candidate at iter %s after post-RM cleanup produced an empty/short response',
                        iter_idx,
                    )
                    pbar.set_postfix(
                        valid=f'{valid_steps}/{self.cfg.target_valid_steps}',
                        patience=patience,
                        refresh=False,
                    )
                    pbar.update(1)
                    n_iter += 1
                    if n_iter % log_interval == 0:
                        logger.info(
                            'RW progress iter %s: valid_steps=%s/%s, consecutive_failures=%s, rejected=%s, short=%s, backtracks=%s',
                            n_iter,
                            valid_steps,
                            self.cfg.target_valid_steps,
                            patience,
                            rejected_candidates,
                            short_candidates,
                            backtracks,
                        )
                    if patience > self.cfg.max_attempts:
                        logger.warning(
                            'RW stopping after iter %s: consecutive_failures=%s exceeded max_attempts=%s with valid_steps=%s/%s',
                            n_iter,
                            patience,
                            self.cfg.max_attempts,
                            valid_steps,
                            self.cfg.target_valid_steps,
                        )
                        break
                    elif patience > self.cfg.backtrack_patience:
                        backtracks += 1
                        logger.info(
                            'RW backtrack at iter %s after %s consecutive failures; reverting to last accepted response',
                            n_iter,
                            patience,
                        )
                        n_response = cached_response
                        patience = 0
                    continue
                previous_patience = patience
                cached_response = n_response
                n_response = candidate_response
                valid_steps += 1
                patience = 0
                logger.info(
                    'RW accepted candidate at iter %s: valid_steps=%s/%s, consecutive_failures_reset=%s',
                    iter_idx,
                    valid_steps,
                    self.cfg.target_valid_steps,
                    previous_patience,
                )
            else:
                patience += 1
                rejected_candidates += 1

            pbar.set_postfix(
                valid=f'{valid_steps}/{self.cfg.target_valid_steps}',
                patience=patience,
                refresh=False,
            )
            pbar.update(1)
            n_iter += 1

            if n_iter % log_interval == 0:
                logger.info(
                    'RW progress iter %s: valid_steps=%s/%s, consecutive_failures=%s, rejected=%s, short=%s, backtracks=%s',
                    n_iter,
                    valid_steps,
                    self.cfg.target_valid_steps,
                    patience,
                    rejected_candidates,
                    short_candidates,
                    backtracks,
                )

            if patience > self.cfg.max_attempts:
                logger.warning(
                    'RW stopping after iter %s: consecutive_failures=%s exceeded max_attempts=%s with valid_steps=%s/%s',
                    n_iter,
                    patience,
                    self.cfg.max_attempts,
                    valid_steps,
                    self.cfg.target_valid_steps,
                )
                break
            elif patience > self.cfg.backtrack_patience:
                backtracks += 1
                logger.info(
                    'RW backtrack at iter %s after %s consecutive failures; reverting to last accepted response',
                    n_iter,
                    patience,
                )
                n_response = cached_response
                patience = 0

        pbar.close()

        logger.info(
            'RW Finished. valid_steps=%s/%s, total_iterations=%s, rejected=%s, short=%s, backtracks=%s',
            valid_steps,
            self.cfg.target_valid_steps,
            n_iter,
            rejected_candidates,
            short_candidates,
            backtracks,
        )

        if not n_response:
            return original_response
        return n_response

    def edit(self, text: str, prompt: str, **kwargs) -> str:
        """Core attack logic."""
        return self._attack_single(text, prompt)
