import torch
import torch.nn as nn
from copy import deepcopy
from transformers import LogitsProcessor, LogitsProcessorList
from transformers.generation.logits_process import MinNewTokensLengthLogitsProcessor
from transformers.cache_utils import DynamicCache


from src.attack.b4.solver import optimize_distribution

class ContrastiveProcessor(LogitsProcessor):
    """
    Core processor (LogitsProcessor) for the B4 attack.

    Responsibilities:
    1. Keep Amateur (Proxy) and Origin (Reference) model states synchronized.
    2. Compute auxiliary-model logits at each generation step.
    3. Apply AEA (approximation-error adjustment) to filter poorly learned proxy tokens.
    4. Apply logits fusion to modify paraphraser generation probabilities.
    """
    def __init__(self, 
                 amateur_model, 
                 origin_model,
                 amateur_tokenizer,
                 generation_config,
                 num_beams=10,
                 coef=0.5, 
                 use_cache=True,
                 closed_form=False,
                 kl_threshold=0.0):
        
        self.amateur_model = amateur_model
        self.origin_model = origin_model
        self.amateur_tokenizer = amateur_tokenizer
        self.generation_config = generation_config
        
        # Attack hyperparameters
        self.coef = coef               # Lambda in the paper (penalty coefficient)
        self.use_cache = use_cache     # Should be True for practical inference speed
        self.closed_form = closed_form # Whether to solve exact lambda via iterative method
        self.kl_threshold = kl_threshold # Epsilon constraint used in closed_form mode
        
        self.amateur_temperature = 1.0
        self.num_beams = num_beams
        
        # Special handling for model families (Qwen uses a different BOS behavior)
        self.is_Qwen = 'Qwen2' in str(self.amateur_model.__class__)

        # Internal state initialization
        self.model_kwargs_amateur = {}
        self.model_kwargs_origin = {}
        self.first_input_ids = None
        self.prev_input_ids = None
        self.step_num = 0
        self.amateur_logits_processor = None

    def _initialize_cache(self, input_ids):
        """Initialize KV cache for auxiliary models."""
        if self.use_cache:
            self.model_kwargs_amateur["use_cache"] = True
            self.model_kwargs_amateur["past_key_values"] = DynamicCache()
            
            self.model_kwargs_origin["use_cache"] = True
            self.model_kwargs_origin["past_key_values"] = DynamicCache()
        else:
            self.model_kwargs_amateur["use_cache"] = False
            self.model_kwargs_amateur["past_key_values"] = None
            self.model_kwargs_origin["use_cache"] = False
            self.model_kwargs_origin["past_key_values"] = None

        # Compute initial cache positions
        self.model_kwargs_amateur = self.amateur_model._get_initial_cache_position(
            input_ids.to(self.amateur_model.device), self.model_kwargs_amateur
        )
        self.model_kwargs_origin = self.origin_model._get_initial_cache_position(
            input_ids.to(self.origin_model.device), self.model_kwargs_origin
        )

    def prepare_before_generate(self, inputs):
        """
        [Key] Must be called before `generate` starts.
        Syncs the main-model prompt to both auxiliary models and expands beams.
        """
        inputs = deepcopy(inputs)
        bsz = inputs['input_ids'].shape[0]

        # 1. Prepare initial input for auxiliary models (usually empty or BOS)
        if self.is_Qwen:
            input_sequence = ["<|endoftext|>"] * bsz
        else:
            input_sequence = [""] * bsz
            
        # Tokenize (mainly to build tensor structure; embeddings are handled in forward)
        # Assumes amateur_tokenizer already handles padding.
        # B4 implementation is intentionally concise here; expand_inputs is the key part below.

        # 2. Initialize kwargs and expand input dimension (Batch -> Batch * Beam)
        # Amateur
        self.model_kwargs_amateur = {'attention_mask': inputs['attention_mask'].to(self.amateur_model.device)}
        self.first_input_ids, self.model_kwargs_amateur = self.amateur_model._expand_inputs_for_generation(
            input_ids=inputs['input_ids'],
            expand_size=self.num_beams,
            is_encoder_decoder=False,
            **self.model_kwargs_amateur,
        )

        # Origin
        self.model_kwargs_origin = {'attention_mask': inputs['attention_mask'].to(self.origin_model.device)}
        _, self.model_kwargs_origin = self.origin_model._expand_inputs_for_generation(
            input_ids=inputs['input_ids'],
            expand_size=self.num_beams,
            is_encoder_decoder=False,
            **self.model_kwargs_origin,
        )
        
        # 3. Initialize cache
        self._initialize_cache(inputs['input_ids'].to(self.amateur_model.device))
        
        # 4. Prepare Amateur logits processor (EOS/min-length constraints, etc.)
        input_ids_seq_length = inputs["input_ids"].shape[-1]
        self.amateur_logits_processor = self._get_amateur_logits_processor(input_ids_seq_length)

        self.prev_input_ids = None
        self.step_num = 0

    def _get_amateur_logits_processor(self, input_ids_seq_length):
        """Get base logits processors for the auxiliary model."""
        processors = LogitsProcessorList()
        if self.generation_config.min_new_tokens is not None and self.generation_config.min_new_tokens > 0:
             if self.generation_config.eos_token_id is not None:
                processors.append(MinNewTokensLengthLogitsProcessor(
                    input_ids_seq_length, 
                    self.generation_config.min_new_tokens, 
                    self.generation_config.eos_token_id
                ))
        return processors

    def _infer_beam_idx(self, input_ids, prev_input_ids):
        """
        [Key challenge] Infer beam-index reordering for beam search.
        After main-model beam selection, identify retained beams so KV cache can be reordered consistently.
        """
        beam_idx = []
        prev_len = prev_input_ids.shape[1]
        # input_ids contains [history + current], compare prefix before the latest token
        last_outputs = input_ids[:, -1-prev_len:-1] 
        
        for i in range(last_outputs.shape[0]):
            for j in range(prev_input_ids.shape[0]):
                if torch.all(last_outputs[i] == prev_input_ids[j]):
                    beam_idx.append(j)
                    break
            else:
                # Should not happen; fall back to 0 to avoid crashing
                beam_idx.append(0) 
        return torch.tensor(beam_idx)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        """
        Entry point of this logits processor, called each decoding step.
        
        Args:
            input_ids: Current input sequence of the main model.
            scores: Current-step logits from the main model (unnormalized).
        """
        # --- Step 1: State sync (input & cache) ---
        
        # Determine auxiliary-model input
        curr_input_ids = input_ids
        if self.step_num == 0:
            # First step: use expanded inputs prepared before generation
            curr_input_ids = self.first_input_ids
        elif self.step_num == 1:
            # Second step: keep only the newly generated token
            curr_input_ids = input_ids[:,-1:]
            # Truncate attention masks accordingly
            self.model_kwargs_amateur['attention_mask'] = self.model_kwargs_amateur['attention_mask'][:,-1:]
            self.model_kwargs_origin['attention_mask'] = self.model_kwargs_origin['attention_mask'][:,-1:]
            # Re-initialize cache after prompt phase
            self._initialize_cache(curr_input_ids)
        else:
            # Later steps: pass only latest token span
            curr_input_ids = input_ids[:,-self.step_num:]

        # Reorder KV cache when beam search is active
        if self.prev_input_ids is not None and self.model_kwargs_amateur["past_key_values"] is not None:
            beam_idx = self._infer_beam_idx(input_ids, self.prev_input_ids).to(self.amateur_model.device)
            
            self.model_kwargs_amateur["past_key_values"] = self.amateur_model._temporary_reorder_cache(
                self.model_kwargs_amateur["past_key_values"], beam_idx
            )
            self.model_kwargs_origin["past_key_values"] = self.origin_model._temporary_reorder_cache(
                self.model_kwargs_origin["past_key_values"], beam_idx
            )

        # --- Step 2: Auxiliary forward pass ---
        
        # 2.1 Amateur (Proxy)
        model_inputs_amateur = self.amateur_model.prepare_inputs_for_generation(
            curr_input_ids.to(self.amateur_model.device), **self.model_kwargs_amateur
        )
        outputs_amateur = self.amateur_model(**model_inputs_amateur)
        next_logits_amateur = outputs_amateur.logits[:, -1, :]
        next_scores_amateur = nn.functional.log_softmax(next_logits_amateur / self.amateur_temperature, dim=-1).to(scores.device)

        # 2.2 Origin (Reference)
        model_inputs_origin = self.origin_model.prepare_inputs_for_generation(
            curr_input_ids.to(self.origin_model.device), **self.model_kwargs_origin
        )
        outputs_origin = self.origin_model(**model_inputs_origin)
        next_logits_origin = outputs_origin.logits[:, -1, :]
        next_scores_origin = nn.functional.log_softmax(next_logits_origin, dim=-1).to(scores.device)

        # Update kwargs for next step
        self.prev_input_ids = input_ids
        self.model_kwargs_amateur = self.amateur_model._update_model_kwargs_for_generation(
            outputs_amateur, self.model_kwargs_amateur, is_encoder_decoder=False
        )
        self.model_kwargs_origin = self.origin_model._update_model_kwargs_for_generation(
            outputs_origin, self.model_kwargs_origin, is_encoder_decoder=False
        )

        # --- Step 3: AEA (Approximation Error Adjustment) ---
        
        # Apply base processor constraints
        next_scores_amateur = self.amateur_logits_processor(input_ids, next_scores_amateur)
        
        vocab_size = next_scores_amateur.shape[-1]
        uniform_prob = 1 / vocab_size
        
        # Difference intuition: if Proxy and Origin are close, watermark signal is weak.
        # Penalize tokens where Proxy strongly deviates from Origin.
        diff = torch.abs(torch.exp(next_scores_amateur) - torch.exp(next_scores_origin))
        indices_to_keep = (diff < uniform_prob)
        
        # Efficiency: skip optimization for tokens already unlikely under main model
        k_filter = 10
        topk_vals = torch.topk(scores, k_filter * self.num_beams)[0]
        indices_to_remove_base = (scores < topk_vals[..., -(k_filter-2)*self.num_beams, None])
        indices_to_keep |= (scores < topk_vals[..., -1, None])

        backup_scores = scores.clone()

        # --- Step 4: Logits fusion (contrastive calculation) ---
        
        if not self.closed_form:
            # Mode A: linear weighting (heuristic)
            # Formula: Score = (1 + lambda) * P_f - lambda * P_w
            
            # Handle -inf to avoid NaN
            next_scores_amateur[next_scores_amateur == float("-inf")] = float("inf")
            new_scores = (self.coef + 1) * scores - self.coef * next_scores_amateur
        else:
            # Mode B: constrained optimization under KL threshold
            p_scores = nn.functional.log_softmax(scores, dim=-1)
            q_scores = nn.functional.log_softmax(next_scores_amateur, dim=-1)

            # Optimize only rows with valid candidate space
            valid_batch_mask = ~(indices_to_keep.all(dim=-1))
            
            if valid_batch_mask.any():
                p_scores_valid = p_scores[valid_batch_mask]
                q_scores_valid = q_scores[valid_batch_mask]
                
                # Solve lambda with optimizer
                alpha, h_scores_valid = optimize_distribution(p_scores_valid, q_scores_valid, self.kl_threshold)
                
                # Scatter optimized results back
                new_scores = torch.full_like(scores, float("-inf"))
                new_scores[valid_batch_mask] = h_scores_valid
            else:
                new_scores = scores.clone()

            # --- Step 5: Restore filtered logits ---
            # Restore unchanged scores for indices_to_keep
        new_scores = torch.where(indices_to_keep, backup_scores, new_scores)
            # Mask originally low-probability positions as -inf
        new_scores = new_scores.masked_fill(indices_to_remove_base, float("-inf"))
        
            # Final normalization
        new_scores = nn.functional.log_softmax(new_scores, dim=-1)
        
        self.step_num += 1
        return new_scores