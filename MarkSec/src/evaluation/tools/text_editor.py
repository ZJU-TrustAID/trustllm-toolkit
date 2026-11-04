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
# text_editor.py
# Description: Edit text using various techniques
# ================================================

import nltk
import torch
import random
from nltk.corpus import wordnet
from src.util.llm import LLMClient
from transformers import BertTokenizer, BertForMaskedLM

class TextEditor:
    """Base class for text editing."""

    def __init__(self) -> None:
        pass

    def edit(self, text: str, reference=None):
        return text

class GPTParaphraser(TextEditor):
    """Paraphrase a text using the GPT model."""

    def __init__(self, openai_model: str, prompt: str) -> None:
        """
            Initialize the GPT paraphraser.

            Parameters:
                openai_model (str): The OpenAI model to use for paraphrasing.
                prompt (str): The prompt to use for paraphrasing.
        """
        self.openai_model = openai_model
        self.prompt = prompt

    def edit(self, text: str, reference=None):
        """Paraphrase the text using the GPT model."""
        openai_util = LLMClient(
            model_name=self.openai_model,
            temperature=0.2,
            system_prompt="You are a helpful assistant to rewrite the text.",
        )
        paraphrased_text = openai_util.get_result(self.prompt + text)
        return paraphrased_text
    
class SynonymSubstitution(TextEditor):
    """Randomly replace words with synonyms from WordNet."""

    def __init__(self, ratio: float) -> None:
        """
            Initialize the synonym substitution editor.

            Parameters:
                ratio (float): The ratio of words to replace.
        """
        self.ratio = ratio
        # Ensure wordnet data is available
        nltk.download('wordnet')

    def edit(self, text: str, reference=None):
        """Randomly replace words with synonyms from WordNet."""
        words = text.split()
        num_words = len(words)
        
        # Dictionary to cache synonyms for words
        word_synonyms = {}

        # First pass: Identify replaceable words and cache their synonyms
        replaceable_indices = []
        for i, word in enumerate(words):
            if word not in word_synonyms:
                synonyms = [syn for syn in wordnet.synsets(word) if len(syn.lemmas()) > 1]
                word_synonyms[word] = synonyms
            if word_synonyms[word]:
                replaceable_indices.append(i)

        # Calculate the number of words to replace
        num_to_replace = min(int(self.ratio * num_words), len(replaceable_indices))

        # Randomly select words to replace
        if num_to_replace > 0:
            indices_to_replace = random.sample(replaceable_indices, num_to_replace)
        
            # Perform replacement
            for i in indices_to_replace:
                synonyms = word_synonyms[words[i]]
                chosen_syn = random.choice(synonyms)
                new_word = random.choice(chosen_syn.lemmas()[1:]).name().replace('_', ' ')
                words[i] = new_word

        # Join the words back into a single string
        replaced_text = ' '.join(words)

        return replaced_text


class ContextAwareSynonymSubstitution(TextEditor):
    """Randomly replace words with synonyms from WordNet based on the context."""

    def __init__(self, ratio: float, tokenizer: BertTokenizer, model: BertForMaskedLM, device='cuda') -> None:
        """
        Initialize the context-aware synonym substitution editor.

        Parameters:
            ratio (float): The ratio of words to replace.
            tokenizer (BertTokenizer): Tokenizer for BERT model.
            model (BertForMaskedLM): BERT model for masked language modeling.
            device (str): Device to run the model (e.g., 'cuda', 'cpu').
        """
        self.ratio = ratio
        self.tokenizer = tokenizer
        self.model = model
        self.device = device
        nltk.download('wordnet')
    
    def _get_synonyms_from_wordnet(self, word: str):
        """ Return a list of synonyms for the given word using WordNet. """
        synonyms = set()
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                synonyms.add(lemma.name().replace('_', ' '))
        return list(synonyms)

    def edit(self, text: str, reference=None):
        """Randomly replace words with synonyms from WordNet based on the context."""
        words = text.split()
        num_words = len(words)
        replaceable_indices = []

        for i, word in enumerate(words):
            if self._get_synonyms_from_wordnet(word):
                replaceable_indices.append(i)

        num_to_replace = int(min(self.ratio, len(replaceable_indices) / num_words) * num_words)
        indices_to_replace = random.sample(replaceable_indices, num_to_replace)

        real_replace = 0

        for i in indices_to_replace:
            # Create a sentence with a [MASK] token
            masked_sentence = words[:i] + ['[MASK]'] + words[i+1:]
            masked_text = " ".join(masked_sentence)
            
            # Use BERT to predict the token for [MASK]
            inputs = self.tokenizer(masked_text, return_tensors='pt', padding=True, truncation=True).to(self.device)
            mask_position = torch.where(inputs["input_ids"][0] == self.tokenizer.mask_token_id)[0].item()

            with torch.no_grad():
                outputs = self.model(**inputs)

            predictions = outputs.logits[0, mask_position]
            predicted_indices = torch.argsort(predictions, descending=True)
            predicted_tokens = self.tokenizer.convert_ids_to_tokens(predicted_indices[0:1])
            words[i] = predicted_tokens[0]
            real_replace += 1
        
        replaced_text = ' '.join(words)

        return replaced_text


class TruncatePromptTextEditor(TextEditor):
    """Truncate the prompt from the text."""

    def __init__(self) -> None:
        super().__init__()

    def edit(self, text: str, reference=None):
        """Truncate the prompt from the text."""
        if reference is not None:
            truncated_text = ' '.join(text.split()[len(reference.split()):])
            return truncated_text
        else:
            return text


class TruncateTaskTextEditor(TextEditor):
    """Truncate the task description from the text, used in code generation."""

    def __init__(self) -> None:
        super().__init__()

    def edit(self, text: str, reference=None):
        """Truncate the task description from the text."""
        if reference is not None:
            truncated_text = text[len(reference):]
            return truncated_text
        else:
            return text
        

class CodeGenerationTextEditor(TextEditor):
    """Process the code generation output, removing the extra parts."""

    def __init__(self) -> None:
        super().__init__()

    def edit(self, text: str, reference=None):
        """Process the code generation output, removing the extra parts."""
        text = text.lstrip("\n")
        text = text.split("\n\n")[0]
        return text


class BackTranslationTextEditor(TextEditor):
    """Translate text from source language to intermediary language, then back to the source language."""

    def __init__(self,
                 translate_to_intermediary=None,
                 translate_to_source=None) -> None:
        """
        Initialize the back translation editor.

        Parameters:
            translate_to_intermediary (function): The function to translate text to the intermediary language.
            translate_to_source (function): The function to translate text to the source language.
        """
        super().__init__()
        if translate_to_intermediary is None or translate_to_source is None:
            try:
                from translate import Translator
            except ImportError as exc:
                raise ImportError(
                    "BackTranslationTextEditor requires the optional 'translate' package."
                ) from exc
            translate_to_intermediary = (
                translate_to_intermediary or Translator(from_lang="en", to_lang="zh").translate
            )
            translate_to_source = (
                translate_to_source or Translator(from_lang="zh", to_lang="en").translate
            )
        self.translate_to_source = translate_to_source
        self.translate_to_intermediary = translate_to_intermediary

    def edit(self, text: str, reference=None):
        intermediary_text = self.translate_to_intermediary(text)
        edit_result = self.translate_to_source(intermediary_text)
        return edit_result
