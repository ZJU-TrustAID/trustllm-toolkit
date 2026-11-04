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
# ===============================================================
# dipper.py
# Description: This file contains the DIPPER attack class.
# ===============================================================

from src.attack.base import BaseAttack, BaseConfig
import torch
import logging
import nltk

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
from nltk.tokenize import sent_tokenize
from src.exceptions.exceptions import DiversityValueError

logger = logging.getLogger(__name__)


class DipperParaphraserConfig(BaseConfig):
    """
    Configuration for the DIPPER paraphraser.
    """

    def initialize_parameters(self) -> None:
        """
        Initialize algorithm-specific parameters.
            lex_diversity (int): The lexical diversity of the output, choose multiples of 20 from 0 to 100. 0 means no diversity, 100 means maximum diversity.
            order_diversity (int): The order diversity of the output, choose multiples of 20 from 0 to 100. 0 means no diversity, 100 means maximum diversity.
            sent_interval (int): The number of sentences to process at a time.
        """
        self.lex_diversity = self.cfg_dict.get('lex_diversity', 60)
        self.order_diversity = self.cfg_dict.get('order_diversity', 0)
        self.sent_interval = self.cfg_dict.get('sent_interval', 1)
        self.model_path = str(
            self.cfg_dict.get('model_path', 'kalpeshk2011/dipper-paraphraser-xxl')
        ).strip()
        self.tokenizer_path = str(self.cfg_dict.get('tokenizer_path', 'google/t5-v1_1-xl')).strip()
        self.torch_dtype = self._parse_torch_dtype(self.cfg_dict.get('torch_dtype', 'float32'))
        self.model_requirements = dict(self.cfg_dict.get('model_requirements', {}))
        if self.model_path and 'paraphraser' not in self.model_requirements:
            self.model_requirements['paraphraser'] = self.model_path
        self.tokenizer_requirements = dict(self.cfg_dict.get('tokenizer_requirements', {}))
        if self.tokenizer_path and 'paraphraser' not in self.tokenizer_requirements:
            self.tokenizer_requirements['paraphraser'] = self.tokenizer_path
        self.model_parallel_roles = list(self.cfg_dict.get('model_parallel_roles', ['paraphraser']))

    @staticmethod
    def _parse_torch_dtype(dtype_name):
        normalized = str(dtype_name).strip().lower()
        dtype_map = {
            'auto': 'auto',
            'float32': torch.float32,
            'fp32': torch.float32,
            'float16': torch.float16,
            'fp16': torch.float16,
            'half': torch.float16,
            'bfloat16': torch.bfloat16,
            'bf16': torch.bfloat16,
        }
        if normalized not in dtype_map:
            raise ValueError(
                f"Unsupported DIPPER torch_dtype '{dtype_name}'. "
                'Use one of: auto, float32, fp32, float16, fp16, half, bfloat16, bf16.'
            )
        return dtype_map[normalized]

    @property
    def algorithm_name(self) -> str:
        """Return the algorithm name."""
        return 'DIPPER'


class DipperParaphraser(BaseAttack):
    """Paraphrase a text using the DIPPER model."""

    def __init__(
        self,
        config: DipperParaphraserConfig,
        paraphraser_model=None,
        paraphraser_tokenizer=None,
        attack_model=None,
        attack_tokenizer=None,
        base_model=None,
        tokenizer=None,
        **kwargs,
    ):
        self.config = config
        self.device = kwargs.get('device', 'cuda')
        self.model = paraphraser_model or attack_model or base_model
        self.tokenizer = paraphraser_tokenizer or attack_tokenizer or tokenizer

        # Validate diversity settings
        self._validate_diversity(self.config.lex_diversity, 'Lexical')
        self._validate_diversity(self.config.order_diversity, 'Order')

    def _model_device(self):
        model_device = getattr(self.model, 'device', None)
        if model_device is not None:
            return model_device
        return self.device

    def _validate_diversity(self, value: int, type_name: str):
        """Validate the diversity value."""
        if value not in [0, 20, 40, 60, 80, 100]:
            raise DiversityValueError(type_name)

    def prepare(self):
        if self.model is None or self.tokenizer is None:
            raise ValueError(
                'DIPPER attack requires a paraphraser model and tokenizer. '
                'Use ComponentFactory or pass paraphraser_model/paraphraser_tokenizer explicitly.'
            )
        if self.config.sent_interval <= 0:
            raise ValueError('DIPPER sent_interval must be positive.')
        if hasattr(self.model, 'eval'):
            self.model.eval()
        return True

    def edit(self, text: str, reference: str = '', prompt: str = '', **kwargs):
        """Edit the text using the DIPPER model."""

        # Calculate the lexical and order diversity codes
        lex_code = int(100 - self.config.lex_diversity)
        order_code = int(100 - self.config.order_diversity)

        # Preprocess the input text
        text = ' '.join(text.split())
        sentences = sent_tokenize(text)

        # Preprocess the reference text
        base_reference = reference or prompt
        prefix = ' '.join(base_reference.replace('\n', ' ').split())

        output_text = ''

        # Process the input text in sentence windows
        for sent_idx in range(0, len(sentences), self.config.sent_interval):
            curr_sent_window = ' '.join(sentences[sent_idx : sent_idx + self.config.sent_interval])

            # Prepare the input for the model
            final_input_text = f'lexical = {lex_code}, order = {order_code}'
            if prefix:
                final_input_text += f' {prefix}'
            final_input_text += f' <sent> {curr_sent_window} </sent>'

            # Tokenize the input
            final_input = self.tokenizer(final_input_text, return_tensors='pt')
            final_input = {k: v.to(self._model_device()) for k, v in final_input.items()}

            # Generate the edited text
            with torch.inference_mode():
                outputs = self.model.generate(**final_input)
            outputs = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

            # Update the prefix and output text
            prefix += ' ' + outputs[0]
            output_text += ' ' + outputs[0]

        # logger.debug(output_text)
        return output_text
