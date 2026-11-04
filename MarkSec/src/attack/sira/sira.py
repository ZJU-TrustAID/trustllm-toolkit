"""
Implementation of SIRA (Self-Information Rewrite Attack).
This attack uses self-information analysis to identify and rewrite watermarked tokens
while maintaining semantic coherence.
"""

import numpy as np
import torch

from ..base import BaseAttack, BaseConfig
from src.util.log import get_logger
from src.util.text import slice_completion

logger = get_logger()


class SIRAConfig(BaseConfig):
    """Configuration for SIRA attack."""

    def initialize_parameters(self) -> None:
        """Initialize SIRA-specific parameters."""
        self.model_requirements = dict(self.cfg_dict.get('model_requirements', {}))
        attack_model_override = self.cfg_dict.get('attack_model')
        if attack_model_override:
            self.model_requirements['attack'] = str(attack_model_override).strip()
        self.model_requirements.setdefault('attack', 'auto')
        self.attack_model_path = self.model_requirements.get('attack')
        self.self_information_threshold = self.cfg_dict.get('self_information_threshold', 30)
        default_gen_config = {'max_new_tokens': 256, 'do_sample': False}
        self.generation_config = self.cfg_dict.get('generation_config', default_gen_config)

        self.paraphrase_prompt_template = self.cfg_dict.get('paraphrase_prompt_template')

        self.attack_prompt_template = self.cfg_dict.get('attack_prompt_template')

    @property
    def algorithm_name(self) -> str:
        return 'SIRA'


class SelfInformationCalculator:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.model.eval()
        logger.info('[SIRA_Calculator] Model and tokenizer loaded successfully.')

    def calculate_self_information(self, text: str):
        with torch.no_grad():
            if not text.strip():
                return [], []
            encoding = self.tokenizer(text, add_special_tokens=False, return_tensors='pt').to(
                self.model.device
            )
            if encoding['input_ids'].shape[1] == 0:
                return [], []
            outputs = self.model(**encoding)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
            self_info = -torch.log(probs)

        input_ids = encoding['input_ids']
        if input_ids.shape[1] <= 1:
            return [], []

        input_ids_expanded = input_ids[:, 1:].unsqueeze(-1)

        tokens = [self.tokenizer.decode(token_) for token_ in input_ids.squeeze().tolist()[1:]]
        self_info_values = (
            self_info[:, :-1].gather(-1, input_ids_expanded).squeeze(-1).squeeze(0).tolist()
        )

        return tokens, self_info_values

    def transform_tokens(self, tokens, self_info_values, threshold_low):
        if not tokens or not self_info_values:
            return []
        percentile = np.percentile(self_info_values, threshold_low)
        transformed_tokens = []
        current_low_si_chunk = []

        for token, self_info in zip(tokens, self_info_values):
            if self_info <= percentile:
                current_low_si_chunk.append(token)
            else:
                if current_low_si_chunk:
                    transformed_tokens.append(f'({" ".join(current_low_si_chunk)})')
                    current_low_si_chunk = []
                transformed_tokens.append('_')

        if current_low_si_chunk:
            transformed_tokens.append(f'({" ".join(current_low_si_chunk)})')

        return transformed_tokens


class SIRAAttack(BaseAttack):
    """
    Self-Information Rewrite Attack (SIRA) implementation based on the paper.

    Uses a separate attack model (Mattack) for self-information calculation and text rewriting.
    """

    def __init__(
        self,
        config: SIRAConfig,
        device: str,
        watermark_instance,
        attack_model,
        attack_tokenizer,
        **kwargs,
    ):
        self.config = config
        self.threshold = self.config.self_information_threshold
        self.calculator = None
        self.device = device
        self.attack_model = attack_model
        self.attack_tokenizer = attack_tokenizer
        self.watermark_instance = watermark_instance
        self.sample_counter = 0

    def _fill_parapharse_prompt(self, input_text: str) -> str:
        template = self.config.paraphrase_prompt_template

        return template.format(input_text=input_text)

    def _fill_attack_prompt(self, ref_text: str, blank_text: str) -> str:
        template = self.config.attack_prompt_template
        return template.format(ref_text=ref_text, blank_text=blank_text)

    def _generate(self, prompt_text: str, system_prompt: str | None = None) -> str:
        """Use the attack model to generate text based on the prompt."""
        messages = []
        if system_prompt is not None:
            messages.append({'role': 'system', 'content': system_prompt})
        messages.append({'role': 'user', 'content': prompt_text})
        inputs = self.attack_tokenizer.apply_chat_template(
            messages,
            return_tensors='pt',
            add_generation_prompt=True,
        ).to(self.device)

        input_len = inputs.shape[1]

        with torch.no_grad():
            outputs = self.attack_model.generate(
                inputs,
                **self.config.generation_config,
                pad_token_id=self.attack_tokenizer.eos_token_id,
                eos_token_id=self.attack_tokenizer.eos_token_id,
            )

        output_text = slice_completion(outputs, input_len, self.attack_tokenizer)

        if output_text.lower().startswith('assistant'):
            output_text = (
                output_text.replace('assistant\n\n', '').replace('assistant: ', '').strip()
            )

        return output_text

    def edit(self, text: str, prompt: str = None, *args, **kwargs) -> str:
        """
        Args:
            text (str): Watermarked text.
            prompt (str, optional): Reference prompt for rewriting. Defaults to None.
            **kwargs: Additional arguments.

        Returns:
            str: The attacked text.
        """
        para_prompt = self._fill_parapharse_prompt(text)
        reference_text = self._generate(para_prompt, system_prompt='You are a helpful rewriter.')
        tokens, self_info = self.calculator.calculate_self_information(text)

        transformed_tokens = self.calculator.transform_tokens(
            tokens, self_info, self.config.self_information_threshold
        )
        blanked_text = ''.join(transformed_tokens)
        attack_prompt = self._fill_attack_prompt(reference_text, blanked_text)
        attacked_text = self._generate(attack_prompt)
        return attacked_text

    def prepare(self) -> bool:
        """Prepare the attack by ensuring all requirements are met."""
        self.calculator = SelfInformationCalculator(self.attack_model, self.attack_tokenizer)
        return True
