import math
import os
import random
import torch
import logging
import json
import hashlib
import time

from src.attack.base import BaseAttack, BaseConfig
from src.util.env import resolve_model_path
from src.evaluation.threat_models import resolve_attack_model
from transformers import AutoModelForCausalLM

logger = logging.getLogger(__name__)


class SmoothingConfig(BaseConfig):
    """
    Configuration class for the Smoothing attack.
    """

    def initialize_parameters(self):
        self.ref_model_name: str = self.cfg_dict.get(
            'ref_model_name'
        )  # Used for distortionary watermarks
        self.device: str = self.cfg_dict.get(
            'device', 'cuda' if torch.cuda.is_available() else 'cpu'
        )
        self.model_requirements = dict(self.cfg_dict.get('model_requirements', {}))
        if self.ref_model_name:
            self.model_requirements['ref'] = str(self.ref_model_name).strip()

        self.k_top_tokens: int = int(
            self.cfg_dict.get('k_top_tokens', 10)
        )  # K for top-K probability access
        self.alpha_smoothness: float = float(self.cfg_dict.get('alpha_smoothness', 1.0))  # α factor
        self.watermark_name: str = self.cfg_dict.get(
            'watermark_name', 'KGW'
        )  # Name of the watermark to be removed
        self.watermark_type: str = self.cfg_dict.get(
            'watermark_type', 'distortionary'
        )  # "distortion-free" or "distortionary"

        # Parameters for ‖Pt‖2 normalization
        self.confidence_norm_samples_N: int = int(
            self.cfg_dict.get('confidence_norm_samples_N', 200)
        )  # N for L, U bounds
        self.confidence_norm_num_bins: int = int(self.cfg_dict.get('confidence_norm_num_bins', 100))

        # Parameters for distortionary watermarks (if watermark_type is "distortionary")
        # These are properties of the watermark being attacked.
        self.gamma_wm: float = float(
            self.cfg_dict.get('gamma_wm', 0.5)
        )  # Example: green list ratio for KGW
        self.delta_wm: float = float(
            self.cfg_dict.get('delta_wm', 2.0)
        )  # Example: bias strength for KGW

        # Standard Generation Parameters
        self.max_new_tokens: int = int(self.cfg_dict.get('max_new_tokens', 100))
        self.prompt_max_length: int = int(
            self.cfg_dict.get('prompt_max_length', 512)
        )  # Max length for input prompt
        self.instrumentation_enabled: bool = bool(
            self.cfg_dict.get('instrumentation_enabled', False)
        )
        self.instrumentation_max_steps: int = int(
            self.cfg_dict.get('instrumentation_max_steps', 64)
        )

    @property
    def algorithm_name(self) -> str:
        return 'Smoothing Attack'


class SmoothingAttack(BaseAttack):
    def __init__(
        self, config: SmoothingConfig, watermark_model=None, watermark_tokenizer=None, **kwargs
    ):
        # Handle parameter mismatch if factory passes 'base_model' instead of 'watermark_model'
        # Prioritize 'watermark_instance' from kwargs if available
        wm_model = watermark_model
        if wm_model is None:
            wm_model = kwargs.get('watermark_instance')

        if wm_model is None:
            wm_model = kwargs.get('base_model')

        wm_tokenizer = (
            watermark_tokenizer if watermark_tokenizer is not None else kwargs.get('tokenizer')
        )

        super().__init__(wm_model, wm_tokenizer)
        self.cfg = config
        self.norm_L: float = 0.0  # Lower bound for ‖Pt‖2 normalization
        self.norm_U: float = 1.0  # Upper bound for ‖Pt‖2 normalization
        self.is_ready: bool = False  # Flag to check if pre-computation is done
        self.tokenizer = self.watermark_tokenizer  # Tokenizer for the reference model
        self.norm_samples_collected: list = []  # Optional, used for debugging
        self.vocab_size = self.watermark_model.config.vocab_size if self.watermark_model else 0
        self.tokenizer_vocab_size = self._resolve_tokenizer_vocab_size()
        # Active token space should follow real tokenizer vocabulary, not padded model dims.
        self.active_vocab_size = min(self.vocab_size, self.tokenizer_vocab_size)
        self.mix_vocab_size = self.active_vocab_size
        self.runtime_device = self._resolve_runtime_device(kwargs.get('device'))
        self.cfg.device = self.runtime_device
        self._last_attack_meta: dict = {}

        self.ref_model = kwargs.get('ref_model')
        needs_ref_model = str(self.cfg.watermark_type).strip().lower() == 'distortionary'
        if needs_ref_model and self.ref_model is None and not self.cfg.ref_model_name:
            self.cfg.ref_model_name = self._infer_default_ref_model_name()
            logger.info('SA auto-resolved reference model: %s', self.cfg.ref_model_name)

        if not needs_ref_model:
            if self.cfg.ref_model_name:
                logger.info(
                    "SmoothingAttack in distortion-free mode; ignoring configured ref_model_name '%s'.",
                    self.cfg.ref_model_name,
                )
            self.cfg.ref_model_name = None
            self.ref_model = None
        elif self.ref_model is None and self.cfg.ref_model_name:
            ref_load_path = resolve_model_path(self.cfg.ref_model_name)
            if ref_load_path != self.cfg.ref_model_name:
                logger.info(
                    f'Resolved reference model path: {self.cfg.ref_model_name} -> {ref_load_path}'
                )
            local_only = os.path.isdir(ref_load_path)
            shared_generation_model = self._get_shared_generation_model(ref_load_path)
            if shared_generation_model is not None:
                logger.info(
                    'SA reference model matches target generation model; '
                    'reusing the loaded generation model to avoid duplicate GPU allocation.'
                )
                self.ref_model = shared_generation_model
                self.ref_model.eval()
                self.cfg.ref_model_name = ref_load_path
            else:
                try:
                    logger.info(f'Loading reference model: {ref_load_path}...')
                    self.ref_model = AutoModelForCausalLM.from_pretrained(
                        ref_load_path,
                        local_files_only=local_only,
                    ).to(self.runtime_device)
                    self.ref_model.eval()
                    self.cfg.ref_model_name = ref_load_path
                except Exception as e:
                    raise RuntimeError(
                        'Failed to load SA reference model '
                        f"'{ref_load_path}' (configured as '{self.cfg.ref_model_name}'). "
                        'Please ensure the checkpoint exists locally and is readable.'
                    ) from e
        elif needs_ref_model and self.ref_model is not None:
            self.ref_model = self.ref_model.to(self.runtime_device)
            self.ref_model.eval()
        elif needs_ref_model and not self.cfg.ref_model_name:
            raise ValueError(
                "SA in distortionary mode requires 'ref_model_name'. "
                'Please provide a valid local checkpoint path/name.'
            )

        if self.ref_model is not None:
            # Distortionary branch mixes distributions in a shared token-id space.
            # Use tokenizer-aligned clipping (and common subset when needed) instead
            # of requiring raw config.vocab_size equality.
            ref_vocab_size = self.ref_model.config.vocab_size
            self.mix_vocab_size = min(self.active_vocab_size, ref_vocab_size)
            if self.mix_vocab_size <= 0:
                raise ValueError(
                    'Invalid shared vocab size for SA distortionary mixing. '
                    f'target_active_vocab={self.active_vocab_size}, ref_vocab={ref_vocab_size}'
                )
            if ref_vocab_size != self.vocab_size:
                logger.warning(
                    'SA vocab mismatch detected (target=%s, ref=%s). '
                    'Using tokenizer-aligned clipped vocab for mixing (V=%s).',
                    self.vocab_size,
                    ref_vocab_size,
                    self.mix_vocab_size,
                )
        elif not self._is_distortionary() and self.active_vocab_size <= 0:
            raise ValueError(f'Invalid active vocab size: {self.active_vocab_size}')

    def _infer_target_model_name(self) -> str:
        """Infer the victim/target model id or path from runtime objects."""
        candidates = []
        if self.watermark_model is not None:
            candidates.append(getattr(self.watermark_model, 'name_or_path', None))
            wm_cfg = getattr(self.watermark_model, 'config', None)
            if wm_cfg is not None:
                gen_model = getattr(wm_cfg, 'generation_model', None)
                if gen_model is not None:
                    candidates.append(getattr(gen_model, 'name_or_path', None))

        for item in candidates:
            if isinstance(item, str) and item.strip():
                return item
        return ''

    def _model_family(self, model_name_or_path: str) -> str:
        """Get org/family name from HF id or local mirror path."""
        if not model_name_or_path:
            return ''

        if not os.path.isabs(model_name_or_path):
            return model_name_or_path.split('/', 1)[0]

        normalized = model_name_or_path.strip(os.sep)
        parts = normalized.split(os.sep)
        if not parts:
            return ''

        models_root = os.environ.get('HF_MODELS_ROOT', '').strip(os.sep)
        root_parts = models_root.split(os.sep) if models_root else []
        if len(parts) >= len(root_parts) + 2 and parts[: len(root_parts)] == root_parts:
            return parts[len(root_parts)]

        if len(parts) >= 2:
            return parts[-2]
        return parts[0]

    def _infer_default_ref_model_name(self) -> str:
        """
        Resolve SA reference model from threat model mapping with local-availability checks.
        """
        target_model = self._infer_target_model_name()
        if not target_model:
            raise ValueError(
                "SA in distortionary mode requires target model context to infer 'ref_model_name'."
            )

        mapped_model = resolve_attack_model(target_model)
        target_family = self._model_family(target_model)
        mapped_family = self._model_family(mapped_model)
        if target_family and mapped_family and mapped_family != target_family:
            # SA distribution mixing assumes aligned token-id space; keep same family by default.
            mapped_model = target_model

        candidate_models = [mapped_model]
        if mapped_model == 'meta-llama/Llama-3.2-1B-Instruct':
            candidate_models.append('meta-llama/Llama-3.2-1B')

        checked = []
        for candidate in candidate_models:
            resolved_candidate = resolve_model_path(candidate)
            checked.append(f'{candidate} -> {resolved_candidate}')
            if not os.path.isabs(resolved_candidate):
                return resolved_candidate
            if os.path.isabs(resolved_candidate) and os.path.exists(resolved_candidate):
                return resolved_candidate

        raise FileNotFoundError(
            '[SA] required reference model is missing locally. '
            f"target='{target_model}'. Checked: {'; '.join(checked)}. "
            'Unset HF_MODELS_ROOT to load from Hugging Face Hub, or pre-download the reference '
            'model into the configured HF_MODELS_ROOT.'
        )

    def _is_distortionary(self) -> bool:
        """Return whether current run should use distortionary path."""
        return str(self.cfg.watermark_type).strip().lower() == 'distortionary'

    def _resolve_runtime_device(self, preferred_device=None) -> str:
        """Resolve a safe runtime device and align with target generation model when possible."""
        if self.watermark_model is not None:
            try:
                gen_model = self.watermark_model.config.generation_model
                model_dev = next(gen_model.parameters()).device
                return str(model_dev)
            except Exception:
                pass

        preferred = (
            str(preferred_device or self.cfg.device).strip().lower()
            if (preferred_device or self.cfg.device) is not None
            else ''
        )
        if preferred in {'', 'auto'}:
            return 'cuda' if torch.cuda.is_available() else 'cpu'
        if preferred.startswith('cuda') and not torch.cuda.is_available():
            logger.warning('Requested CUDA device but CUDA is unavailable. Falling back to CPU.')
            return 'cpu'
        return preferred

    def _resolve_tokenizer_vocab_size(self) -> int:
        """Resolve tokenizer vocabulary size with safe fallback to model config vocab."""
        if self.tokenizer is not None:
            try:
                tok_len = int(len(self.tokenizer))
                if tok_len > 0:
                    return tok_len
            except Exception:
                pass
        return int(self.vocab_size)

    def _get_shared_generation_model(self, ref_load_path: str):
        """Reuse the already loaded target model when SA resolves the same checkpoint as ref."""
        if self.watermark_model is None:
            return None

        try:
            generation_model = self.watermark_model.config.generation_model
        except Exception:
            return None

        generation_name = getattr(generation_model, 'name_or_path', None)
        if not isinstance(generation_name, str) or not generation_name.strip():
            return None

        resolved_generation_path = resolve_model_path(generation_name)
        if resolved_generation_path == ref_load_path:
            return generation_model
        return None

    def _build_cache_fingerprint(self) -> str:
        """Build a stable fingerprint to avoid cache collisions across configs/runs."""
        target_model_name = getattr(
            self.watermark_model.config.generation_model, 'name_or_path', None
        )
        ref_model_name = (
            getattr(self.ref_model, 'name_or_path', None)
            if self.ref_model is not None
            else self.cfg.ref_model_name
        )
        payload = {
            'watermark_name': self.cfg.watermark_name,
            'watermark_type': self.cfg.watermark_type,
            'target_model_name': target_model_name,
            'target_vocab_size': self.vocab_size,
            'tokenizer_vocab_size': self.tokenizer_vocab_size,
            'active_vocab_size': self.active_vocab_size,
            'ref_model_name': ref_model_name,
            'ref_vocab_size': self.ref_model.config.vocab_size
            if self.ref_model is not None
            else None,
            'mix_vocab_size': self.mix_vocab_size,
            'k_top_tokens': self.cfg.k_top_tokens,
            'alpha_smoothness': self.cfg.alpha_smoothness,
            'confidence_norm_samples_N': self.cfg.confidence_norm_samples_N,
            'confidence_norm_num_bins': self.cfg.confidence_norm_num_bins,
            'gamma_wm': self.cfg.gamma_wm,
            'delta_wm': self.cfg.delta_wm,
            'prompt_max_length': self.cfg.prompt_max_length,
        }
        payload_json = json.dumps(payload, sort_keys=True, ensure_ascii=True)
        return hashlib.sha256(payload_json.encode('utf-8')).hexdigest()[:16]

    def prepare(self):
        if self.cfg.watermark_name in ['KGW', 'Unigram', 'SIR']:
            self.cfg.watermark_type = 'distortionary'

        if self.watermark_model is None:
            raise ValueError('watermark_model is None, please pass in a pipeline.')

        save_path = self.get_attack_cache_dir()
        cache_tag = self._build_cache_fingerprint()
        save_name = os.path.join(
            save_path, f'{self.cfg.watermark_name}_{self.cfg.watermark_type}_{cache_tag}.pt'
        )

        if os.path.exists(save_name):
            logger.info(f'Loading precomputed confidence bounds from: {save_name}')
            try:
                checkpoint = torch.load(save_name, map_location=self.runtime_device)
                self.norm_L = checkpoint['norm_L']
                self.norm_U = checkpoint['norm_U']
                self.norm_samples_collected = checkpoint.get('norm_samples_collected', [])
                self.is_ready = True
                return self.is_ready
            except Exception as e:
                logger.error(f'Failed to load checkpoint: {e}. Re-computing.')

        # query watermark model, get norm L and U, then save
        self._precompute_confidence_bounds()
        logger.info(f'Saving precomputed confidence bounds to: {save_name}')

        try:
            torch.save(
                {
                    'norm_L': self.norm_L,
                    'norm_U': self.norm_U,
                    'norm_samples_collected': self.norm_samples_collected,
                },
                save_name,
            )
            logger.info('Successfully saved bounds.')
        except Exception as e:
            logger.error(f'Error saving precomputed bounds: {e}')

        self.is_ready = True
        logger.info('SmoothingAttack: Preparation complete.')
        return self.is_ready

    def _precompute_confidence_bounds(self):
        """
        Collect N random samples of transformed ‖Pt‖2 and set bounds L/U.
        """
        logger.info(
            f'Starting pre-computation of confidence bounds with N={self.cfg.confidence_norm_samples_N}...'
        )
        max_prefix_len = min(
            10, self.cfg.prompt_max_length // 10 if self.cfg.prompt_max_length > 10 else 5
        )
        max_prefix_len = max(1, max_prefix_len)  # Ensure minimum prefix length is 1.
        work_vocab_size = self.active_vocab_size

        for i in range(self.cfg.confidence_norm_samples_N):
            prefix_len = random.randint(1, max_prefix_len)
            input_ids = torch.randint(
                0, work_vocab_size, (1, prefix_len), device=self.runtime_device
            )

            if input_ids.shape[1] == 0:
                logger.warning(f'Generated empty input_ids for sample {i}, skipping.')
                continue

            try:
                with torch.no_grad():
                    outputs = self.watermark_model.config.generation_model(input_ids)
                    original_next_token_logits = outputs.logits[
                        :, -1, :
                    ]  # Logits for the next token
                    watermarked_next_token_logits = self.watermark_model.logits_processor(
                        input_ids,  # input_ids for the processor
                        original_next_token_logits.clone(),  # scores for the processor
                    )
                    original_next_token_probs = torch.softmax(
                        original_next_token_logits, dim=-1
                    ).squeeze()
                    watermarked_next_token_probs = torch.softmax(
                        watermarked_next_token_logits, dim=-1
                    ).squeeze()
                    observed_next_token_probs = (
                        watermarked_next_token_probs
                        if self._is_distortionary()
                        else original_next_token_probs
                    )
                    observed_next_token_probs = observed_next_token_probs[:work_vocab_size]
                    actual_k = min(self.cfg.k_top_tokens, work_vocab_size)
                    top_k_probs, _ = torch.topk(
                        observed_next_token_probs, actual_k
                    )  # Use _ for indices

                    pt_norm_sq = self._estimate_pt_norm_sq(
                        top_k_probs.cpu(), work_vocab_size, self.cfg.k_top_tokens
                    )
                    self.norm_samples_collected.append(pt_norm_sq)
            except Exception as e:  # if any error occurs during sampling
                logger.warning(
                    f'Error during sample generation {i + 1}/{self.cfg.confidence_norm_samples_N}: {e}. Skipping.'
                )
                continue

        if not self.norm_samples_collected:  # check
            raise RuntimeError('Failed to collect samples for confidence normalization.')
        else:
            self.norm_L = min(self.norm_samples_collected)
            self.norm_U = max(self.norm_samples_collected)

        if self.norm_L == self.norm_U:
            logger.warning(
                f'norm_L ({self.norm_L}) == norm_U ({self.norm_U}). Confidence scores might not be discriminative.'
            )

    def _estimate_pt_norm_sq(
        self, top_k_probs: torch.Tensor, vocab_size: int, actual_k_val: int
    ) -> float:
        """
        Args:
            top_k_probs: Tensor of shape (K,) with top-K probabilities.
            actual_k_val: The actual K used.
        Returns:
            Estimated L2 norm squared.
        """
        if not isinstance(top_k_probs, torch.Tensor):
            raise TypeError(f'top_k_probs must be a torch.Tensor, got {type(top_k_probs)}')
        if top_k_probs.dim() != 1:
            raise ValueError(
                f'top_k_probs must be a 1D tensor, got {top_k_probs.dim()} dimensions.'
            )
        if actual_k_val != len(top_k_probs):
            logger.warning(
                f'actual_k_val ({actual_k_val}) does not match len(top_k_probs) ({len(top_k_probs)}). Using len(top_k_probs).'
            )
            actual_k_val = len(top_k_probs)

        if vocab_size <= actual_k_val:  # Edge case: K is as large as or larger than vocab size
            return torch.sum(top_k_probs**2).item()

        sum_top_k_probs_val = torch.sum(top_k_probs).item()
        sum_top_k_probs_sq_val = torch.sum(top_k_probs**2).item()

        # Ensure residual_prob_mass is not negative due to potential floating point inaccuracies
        residual_prob_mass = max(0.0, 1.0 - sum_top_k_probs_val)
        num_remaining_tokens = vocab_size - actual_k_val

        if num_remaining_tokens > 0:
            term2 = (residual_prob_mass**2) / num_remaining_tokens
        else:
            term2 = 0.0

        pt_norm_sq = sum_top_k_probs_sq_val + term2
        return pt_norm_sq

    def _get_confidence_score_c(self, estimated_pt_norm_sq: float) -> float:
        """
        Convert estimated ‖Pt‖² into a confidence score c in [0, 1].
        """
        if not self.is_ready:
            raise RuntimeError('Confidence bounds not ready. Please call prepare() first.')

        if self.norm_U == self.norm_L:
            logger.warning(
                'Confidence normalization bounds collapsed (L == U); '
                'returning 0.0 as a degenerate linear-normalization fallback.'
            )
            return 0.0

        # Clamp estimate to [L, U]
        clamped_value = max(self.norm_L, min(estimated_pt_norm_sq, self.norm_U))
        num_bins = max(1, int(self.cfg.confidence_norm_num_bins))
        normalized_position = (clamped_value - self.norm_L) / (self.norm_U - self.norm_L)

        # Paper-aligned confidence quantization: map the normalized value to the
        # i-th bin with i in [1, num_bins], then return c = i / num_bins.
        bin_index = math.ceil(normalized_position * num_bins)
        bin_index = min(num_bins, max(1, bin_index))

        return bin_index / num_bins

    def edit(self, text: str, prompt: str = None, *args, **kwargs) -> str:
        """
        Edit input text with Smoothing Attack.
        If `text` is provided, distortion-free mode retains tokens from the provided
        source text with probability c^alpha; distortionary mode samples directly
        from the paper's mixture c^alpha * Pwm + (1 - c^alpha) * Pref.
        If `text` is empty, it falls back to generation mode from `prompt`.
        """
        if not self.is_ready:
            raise RuntimeError('SmoothingAttack.prepare() not completed.')

        source_text = text or ''
        prompt_text = prompt or ''
        if not source_text and not prompt_text and self.tokenizer.bos_token_id is None:
            logger.warning('Input text/prompt is empty and tokenizer has no bos_token_id.')

        # Required component checks
        if self.tokenizer is None:
            raise ValueError('Tokenizer (self.tokenizer) not initialized.')
        if self.watermark_model is None:
            raise ValueError('Watermark model (self.watermark_model) not initialized.')
        if self._is_distortionary() and self.ref_model is None:
            raise ValueError(
                'Reference model required for distortionary watermark but not initialized.'
            )
        if self.active_vocab_size <= 0:
            raise ValueError(f'Invalid active vocab size: {self.active_vocab_size}')

        # Prefix context for model scoring.
        if prompt_text:
            prefix_ids = self.tokenizer.encode(
                prompt_text,
                return_tensors='pt',
                truncation=True,
                max_length=self.cfg.prompt_max_length,
            ).to(self.runtime_device)
        elif self.tokenizer.bos_token_id is not None:
            prefix_ids = torch.tensor(
                [[self.tokenizer.bos_token_id]], device=self.runtime_device, dtype=torch.long
            )
            logger.info('Starting from BOS token as prefix because prompt is empty.')
        else:
            prefix_ids = torch.empty((1, 0), device=self.runtime_device, dtype=torch.long)

        # Source tokens are used only by distortion-free edit mode, where the paper
        # retains the original watermarked token with probability c^alpha.
        source_token_ids = []
        if source_text:
            source_token_ids = self.tokenizer.encode(source_text, add_special_tokens=False)

        output_tokens = []
        past_key_values = None

        # full_sequence_ids is used by watermark logits processor (needs full history).
        full_sequence_ids = prefix_ids.clone()
        # next_input_id is fed to generation model. On first step, feed full prefix.
        next_input_id = prefix_ids

        # In edit mode, rewrite exactly this many tokens.
        # In generation fallback, generate up to max_new_tokens.
        steps = len(source_token_ids) if source_token_ids else self.cfg.max_new_tokens
        if steps <= 0:
            return ''

        use_cache = True
        attack_t0 = time.time()
        c_scores: list[float] = []
        keep_steps = 0
        top_k_replace_steps = 0
        full_replace_steps = 0
        mixture_steps = 0
        eos_fallback_steps = 0
        step_trace: list[dict] = []
        instrumentation_enabled = bool(getattr(self.cfg, 'instrumentation_enabled', False))
        trace_cap = max(0, int(getattr(self.cfg, 'instrumentation_max_steps', 64)))

        for token_idx in range(steps):
            try:
                with torch.no_grad():
                    # === Step 1: Get logits from target watermarked model ===
                    # Try KV cache
                    model_outputs = self.watermark_model.config.generation_model(
                        next_input_id, past_key_values=past_key_values, use_cache=use_cache
                    )

                    past_key_values = model_outputs.past_key_values
                    original_logits_next_token = model_outputs.logits[:, -1, :]  # [batch, vocab]

                    # Apply watermark logits_processor
                    # Note: logits_processor usually needs full sequence context for hash-based logic.
                    watermarked_target_logits_next_token = self.watermark_model.logits_processor(
                        full_sequence_ids, original_logits_next_token.clone()
                    )

                    original_target_probs_next_token = torch.softmax(
                        original_logits_next_token, dim=-1
                    ).squeeze(0)
                    watermarked_target_probs_next_token = torch.softmax(
                        watermarked_target_logits_next_token, dim=-1
                    ).squeeze(0)
                    work_vocab_size = (
                        self.mix_vocab_size if self._is_distortionary() else self.active_vocab_size
                    )
                    original_target_probs_next_token = original_target_probs_next_token[
                        :work_vocab_size
                    ]
                    watermarked_target_probs_next_token = watermarked_target_probs_next_token[
                        :work_vocab_size
                    ]
                    # Paper assumption: distortion-free watermarks preserve the original
                    # next-token distribution, so the observed top-K should come from the
                    # target model's raw logits rather than watermark-processed logits.
                    observed_target_probs_next_token = (
                        watermarked_target_probs_next_token
                        if self._is_distortionary()
                        else original_target_probs_next_token
                    )
                    observed_target_probs_next_token = observed_target_probs_next_token[
                        :work_vocab_size
                    ]

                    # Check validity
                    if observed_target_probs_next_token.dim() != 1:
                        logger.error(
                            f'Invalid probability shape at token {token_idx + 1}: {observed_target_probs_next_token.shape}'
                        )
                        break

                    actual_k = min(self.cfg.k_top_tokens, work_vocab_size)
                    actual_k = max(1, actual_k)

                    # Top-K of P̃t
                    top_k_target_probs_ptilde, top_k_target_indices_ptilde = torch.topk(
                        observed_target_probs_next_token, actual_k
                    )

                    p_tilde_t_norm_sq = self._estimate_pt_norm_sq(
                        top_k_target_probs_ptilde.cpu(), work_vocab_size, actual_k
                    )

                    final_pt_norm_sq_estimate: float
                    if self._is_distortionary():
                        gamma = self.cfg.gamma_wm
                        delta = self.cfg.delta_wm
                        term_gamma_exp_delta = (1 - gamma) + gamma * math.exp(delta)
                        term_gamma_exp_2delta = (1 - gamma) + gamma * math.exp(2 * delta)
                        beta = (
                            (term_gamma_exp_delta**2) / term_gamma_exp_2delta
                            if term_gamma_exp_2delta != 0
                            else 1.0
                        )
                        final_pt_norm_sq_estimate = beta * p_tilde_t_norm_sq
                    else:
                        final_pt_norm_sq_estimate = p_tilde_t_norm_sq

                    c_score = self._get_confidence_score_c(final_pt_norm_sq_estimate)
                    wm_weight = c_score**self.cfg.alpha_smoothness
                    c_scores.append(float(c_score))

                    next_token_id: int = -1
                    decision = 'unknown'

                    if self._is_distortionary():
                        # Paper-aligned distortionary branch: sample directly from
                        # c^alpha * Pwm + (1 - c^alpha) * Pref.
                        ref_outputs = self.ref_model(full_sequence_ids)
                        ref_probs = torch.softmax(ref_outputs.logits[:, -1, :], dim=-1).squeeze(0)
                        ref_probs = ref_probs[:work_vocab_size]

                        next_token_id = self._sample_from_mixture(
                            observed_target_probs_next_token,
                            top_k_target_probs_ptilde,
                            top_k_target_indices_ptilde,
                            ref_probs,
                            wm_weight,
                        )
                        mixture_steps += 1
                        decision = 'mixture'
                    elif random.random() < wm_weight:
                        # Distortion-free paper behavior: keep the token from the provided
                        # source watermarked text at this position during edit mode.
                        if source_token_ids:
                            next_token_id = int(source_token_ids[token_idx])
                            keep_steps += 1
                            decision = 'keep_source_token'
                        elif self._is_valid_prob(observed_target_probs_next_token):
                            # Generation fallback when source text is unavailable.
                            next_token_id = torch.multinomial(
                                observed_target_probs_next_token, num_samples=1
                            ).item()
                            full_replace_steps += 1
                            decision = 'sample_full_distribution'
                        else:
                            next_token_id = self._get_eos_id()
                            eos_fallback_steps += 1
                            decision = 'fallback_eos'
                    else:
                        # Distortion-free replacement branch: re-sample from top-K of P̃t.
                        if self._is_valid_prob(top_k_target_probs_ptilde):
                            normalized_top_k = top_k_target_probs_ptilde / torch.sum(
                                top_k_target_probs_ptilde
                            )
                            idx_in_topk = torch.multinomial(normalized_top_k, num_samples=1).item()
                            next_token_id = top_k_target_indices_ptilde[idx_in_topk].item()
                            top_k_replace_steps += 1
                            decision = 'top_k_replace'
                        elif self._is_valid_prob(observed_target_probs_next_token):
                            next_token_id = torch.multinomial(
                                observed_target_probs_next_token, num_samples=1
                            ).item()
                            full_replace_steps += 1
                            decision = 'full_distribution_replace'
                        else:
                            next_token_id = self._get_eos_id()
                            eos_fallback_steps += 1
                            decision = 'fallback_eos'

                # Append result
                output_tokens.append(next_token_id)
                if instrumentation_enabled and len(step_trace) < trace_cap:
                    step_trace.append(
                        {
                            'step': token_idx,
                            'c_score': round(float(c_score), 6),
                            'wm_weight': round(float(wm_weight), 6),
                            'decision': decision,
                            'selected_token_id': int(next_token_id),
                            'used_source_token': decision == 'keep_source_token',
                        }
                    )
                next_token_tensor = torch.tensor(
                    [[next_token_id]], device=self.runtime_device, dtype=torch.long
                )

                # Update inputs for next step
                next_input_id = next_token_tensor
                full_sequence_ids = torch.cat([full_sequence_ids, next_token_tensor], dim=-1)

                if next_token_id == self.tokenizer.eos_token_id:
                    break

            except Exception as e:
                logger.error(f'Error during generation at token {token_idx}: {e}')
                break

        total_steps = len(output_tokens)
        self._last_attack_meta = {
            'algorithm': 'SA',
            'watermark_type': self.cfg.watermark_type,
            'runtime_cost': float(time.time() - attack_t0),
            'total_steps': total_steps,
            'mean_c_score': float(sum(c_scores) / len(c_scores)) if c_scores else None,
            'min_c_score': float(min(c_scores)) if c_scores else None,
            'max_c_score': float(max(c_scores)) if c_scores else None,
            'keep_steps': keep_steps,
            'top_k_replace_steps': top_k_replace_steps,
            'full_replace_steps': full_replace_steps,
            'mixture_steps': mixture_steps,
            'fallback_eos_steps': eos_fallback_steps,
            'keep_rate': float(keep_steps / total_steps) if total_steps else 0.0,
            'top_k_replacement_rate': float(top_k_replace_steps / total_steps)
            if total_steps
            else 0.0,
            'full_replacement_rate': float(full_replace_steps / total_steps)
            if total_steps
            else 0.0,
            'mixture_rate': float(mixture_steps / total_steps) if total_steps else 0.0,
            'instrumentation_enabled': instrumentation_enabled,
        }
        if instrumentation_enabled:
            self._last_attack_meta['c_scores'] = [round(float(x), 6) for x in c_scores[:trace_cap]]
            self._last_attack_meta['step_trace'] = step_trace
            self._last_attack_meta['instrumented_steps'] = len(step_trace)
            self._last_attack_meta['instrumentation_truncated'] = len(c_scores) > len(step_trace)

        return self.tokenizer.decode(output_tokens, skip_special_tokens=True)

    def pop_last_attack_meta(self) -> dict:
        meta = dict(self._last_attack_meta)
        self._last_attack_meta = {}
        return meta

    def _is_valid_prob(self, probs):
        return (
            torch.sum(probs).item() > 1e-9
            and not torch.isnan(probs).any()
            and not torch.isinf(probs).any()
        )

    def _get_eos_id(self):
        return self.tokenizer.eos_token_id if self.tokenizer.eos_token_id is not None else 0

    def _sample_from_mixture(self, p_tilde, top_k_p, top_k_idx, p_ref, alpha_weight):
        # alpha_weight is prob_adopt_wm_token (c^alpha)
        # Weight for P_tilde is alpha_weight
        # Weight for P_ref is 1 - alpha_weight
        # Keep a unified float dtype for arithmetic/scatter to avoid dtype mismatch
        # when model logits are fp16/bf16.
        compute_dtype = torch.float32
        p_tilde = p_tilde.to(dtype=compute_dtype)
        top_k_p = top_k_p.to(dtype=compute_dtype)
        p_ref = p_ref.to(dtype=compute_dtype)
        top_k_idx = top_k_idx.to(dtype=torch.long)
        alpha_weight = float(alpha_weight)

        vocab_size = int(p_tilde.shape[-1])
        weighted_probs = torch.zeros(vocab_size, device=self.runtime_device, dtype=compute_dtype)

        # Add P_tilde component
        if self._is_valid_prob(top_k_p):
            norm_p = top_k_p / torch.sum(top_k_p)
            weighted_probs.scatter_add_(0, top_k_idx, alpha_weight * norm_p)

        # Add P_ref component
        # Need top-K for Ref
        actual_k = min(self.cfg.k_top_tokens, vocab_size)
        top_k_ref_p, top_k_ref_idx = torch.topk(p_ref, actual_k)
        top_k_ref_idx = top_k_ref_idx.to(dtype=torch.long)

        if self._is_valid_prob(top_k_ref_p):
            norm_ref = top_k_ref_p / torch.sum(top_k_ref_p)
            weighted_probs.scatter_add_(0, top_k_ref_idx, (1.0 - alpha_weight) * norm_ref)

        if torch.sum(weighted_probs).item() > 1e-9:
            return torch.multinomial(weighted_probs, num_samples=1).item()

        # Fallback to P_tilde
        if self._is_valid_prob(p_tilde):
            return torch.multinomial(p_tilde, num_samples=1).item()

        return self._get_eos_id()
