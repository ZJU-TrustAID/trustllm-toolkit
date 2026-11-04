# =======================================================
# src/util/eval.py
# Description: Evaluation tools (Quality Oracle, Grammar Corrector)
# =======================================================

import logging
import torch
import transformers
from time import sleep
from typing import List
from src.util.llm import LLMClient
from src.util.env import resolve_model_path

# Initialize logger
try:
    from src.util.log import get_logger

    logger = get_logger(__name__)
except ImportError:
    logger = logging.getLogger(__name__)


# --- Grammar Corrector ---


class GrammarCorrector:
    def __init__(
        self,
        model_name='vennify/t5-base-grammar-correction',
        device='cuda',
        local_files_only=False,
    ):
        self.device = device
        load_path = resolve_model_path(model_name)
        logger.info(f'[GEC] Loading grammar model: {load_path} on {device}...')

        try:
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(
                load_path,
                local_files_only=local_files_only,
            )
            self.model = transformers.AutoModelForSeq2SeqLM.from_pretrained(
                load_path,
                local_files_only=local_files_only,
            ).to(self.device)
            self.model.eval()
            logger.info('[GEC] Model loaded successfully.')
        except Exception as e:
            logger.error(f'[GEC] Failed to load model: {e}')
            raise e

    def correct_batch(self, sentences: List[str]) -> List[str]:
        if not sentences:
            return []

        input_texts = [f'grammar: {sent}' for sent in sentences]

        inputs = self.tokenizer(
            input_texts, return_tensors='pt', padding=True, truncation=True, max_length=512
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs, max_length=512, num_beams=2, early_stopping=True, repetition_penalty=1.0
            )

        corrected_sentences = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return corrected_sentences


# --- Quality Oracle ---


class QualityOracle:
    def __init__(
        self,
        local_tokenizer=None,
        local_model=None,
        reward_model=None,
        oracle_model_name: str = 'deepseek-chat',
        check_quality: str = 'checker',  # "gpt", "checker", "hf_model"
        choice_granularity: int = 5,
        device: str = 'cuda',
        reward_tokenizer_name: str = 'OpenAssistant/reward-model-deberta-v3-large-v2',
        local_files_only: bool = False,
    ):
        self.local_tokenizer = local_tokenizer
        self.local_model = local_model  # This is the HF model for quality checks
        self.reward_model = reward_model
        self.oracle_model_name = oracle_model_name
        self.check_quality = check_quality
        self.choice_granularity = choice_granularity
        self.system_prompt = 'You are a capable, helpful, and impartial evaluator.'
        self.device = device

        # Initialize LLM Client for Oracle
        self.llm_client = LLMClient(
            model_name=oracle_model_name,
            system_prompt=self.system_prompt,
            temperature=0,  # Deterministic for evaluation
        )

        try:
            reward_tokenizer_path = resolve_model_path(reward_tokenizer_name)
            self.rm_tokenizer = transformers.AutoTokenizer.from_pretrained(
                reward_tokenizer_path,
                local_files_only=local_files_only,
            )
        except Exception as e:
            logger.warning(
                f'Could not load Reward Model tokenizer: {e}. RM functionality might fail.'
            )

        logger.info(
            f"QualityOracle initialized to use '{self.oracle_model_name}' as the LLM judge."
        )
        if check_quality == 'hf_model' and not (local_model and local_tokenizer):
            raise ValueError(
                "'hf_model' requires `local_model` and `local_tokenizer` to be provided."
            )

    @property
    def comparison_prompt(self):
        return """Response A: [A] \nResponse B: [B] \n 
        Please compare which of the two above responses is a better response to the user's prompt. 
        Explain your reasoning step-by-step. Your evaluation should consider factors like helpfulness, repetition, grammar, coherence, relevance, and accuracy.
        Grammatical errors, repetitions, capitalization errors, or punctuation mistakes significantly degrade a response's quality.
        """

    @property
    def check_quality_prompt(self):
        return (
            'Text quality is affected by factors such as unnecessary repetitions, grammar, coherence, and accuracy of the responses. Especially, having grammatical errors, repetitiveness, capitalization errors or punctuation mistakes would greatly degrade the quality of a response.'
            + '\n'
            + 'Therefore, is the new modified response of equal or higher quality compared to original response? If so, answer Yes, otherwise answer No.'
        )

    @property
    def answer_format_prompt(self):
        choices = self.five_choice if self.choice_granularity == 5 else self.three_choice
        return f"""
        Based on your reasoning, choose one of the following options:
        {choices}
        
        Finally, provide your output in a single JSON object with two keys:
        1. "reasoning": A string containing your detailed explanation.
        2. "choice": An integer representing your final selection (e.g., 1, 2, 3, 4, or 5).
        
        Example JSON output:
        {{
          "reasoning": "Response A is more direct and grammatically correct, while Response B contains repetitive phrases and a factual error.",
          "choice": 2
        }}
        """

    @property
    def five_choice(self):
        return """
        (1) Response A is much better than response B
        (2) Response A is a little better than response B
        (3) Responses A and B have similar quality
        (4) Response B is a little better than response A
        (5) Response B is much better than response A
        """

    @property
    def three_choice(self):
        return """
        (1) Response A is better than response B
        (2) Responses A and B have similar quality
        (3) Response B is better than response A
        """

    def query_llm_once(self, prompt: str):
        """
        Uses LLMClient to get a structured JSON response.
        """
        full_prompt = f'{prompt}\n\n{self.answer_format_prompt}'

        for attempt in range(
            3
        ):  # LLMClient handles internal retries, but we might want app-level logic
            try:
                # Use LLMClient's JSON capability
                response_data = self.llm_client.get_json_result(full_prompt)

                if not isinstance(response_data, dict):
                    # Fallback parsing if string returned (unlikely with get_json_result logic)
                    logger.warning(f'Expected dict from LLM, got {type(response_data)}')
                    continue

                score = int(response_data.get('choice', -99))
                reasoning = response_data.get('reasoning', '')

                logger.debug(f'Oracle Reasoning: {reasoning}')
                logger.debug(f'Oracle Choice: {score}')

                if score not in range(1, self.choice_granularity + 1):
                    logger.warning(
                        f'Returned choice {score} not in valid range [1, {self.choice_granularity}]. Retrying...'
                    )
                    continue

                return score  # Success!

            except Exception as e:
                logger.error(
                    f'An unexpected error occurred in query_llm_once on attempt {attempt + 1}: {e}',
                    exc_info=True,
                )
                sleep(1)

        logger.error(f'All attempts to query oracle failed.')
        return None

    def query_rm_once(self, query, response_1, response_2, tie_threshold=0.01):
        if self.reward_model is None:
            raise ValueError('Reward model is not initialized.')
        context = '###Human: ' + query + '###Assistant: '
        text1 = context + response_1
        text2 = context + response_2

        # Use modern transformers call style
        tokenized_text1 = self.rm_tokenizer(text1, return_tensors='pt').to(self.device)
        tokenized_text2 = self.rm_tokenizer(text2, return_tensors='pt').to(self.device)

        with torch.no_grad():
            score_1 = self.reward_model(**tokenized_text1).logits
            score_2 = self.reward_model(**tokenized_text2).logits

        # Using softmax to normalize scores for comparison
        softmax = torch.nn.Softmax(dim=0)
        scores = softmax(torch.tensor([score_1.item(), score_2.item()]))
        score_gap = abs(scores[0].item() - scores[1].item())

        if score_gap < tie_threshold:
            return 2
        return 1 if score_1 > score_2 else 3

    def maintain_quality(self, query, original_response, paraphrased_response, tie_threshold=0.02):
        choice = self.query_rm_once(
            query, paraphrased_response, original_response, tie_threshold=tie_threshold
        )
        if choice is None:
            return False

        score_dict = {1: 1, 2: 0, 3: -1}
        score = score_dict.get(choice, 0)

        second_choice = self.query_rm_once(
            query, original_response, paraphrased_response, tie_threshold=tie_threshold
        )
        if second_choice is None:
            return False

        score -= score_dict.get(second_choice, 0)

        if score < 0:
            return False

        # If score is neutral or positive, proceed to the next quality check stage.
        if self.check_quality == 'gpt':
            prompt = f'Original response: {original_response}\nNew response: {paraphrased_response}\n{self.check_quality_prompt}'
            # Use LLM Client
            check_quality_response = self.llm_client.get_result(prompt)
            return 'yes' in check_quality_response.lower() if check_quality_response else False
        elif self.check_quality == 'hf_model':
            prompt = f'Original response: {original_response}\nNew response: {paraphrased_response}\n{self.check_quality_prompt}'
            check_quality_response = self.check_quality_hf_models(prompt)
            return 'yes' in check_quality_response.lower() if check_quality_response else False
        elif self.check_quality == 'checker':
            # This mode assumes RM is sufficient
            return True
        else:
            return True

    def check_quality_hf_models(self, prompt: str):
        inputs = self.local_tokenizer(prompt, return_tensors='pt').to(self.device)
        with torch.no_grad():
            outputs = self.local_model.generate(
                **inputs, max_new_tokens=512, top_p=0.95, num_return_sequences=1
            )
        output_text = self.local_tokenizer.batch_decode(
            outputs[:, inputs['input_ids'].shape[1] :],
            skip_special_tokens=True,
        )[0]
        logger.debug(f'HF Quality Oracle Assessment: {output_text}')
        return output_text

    def report_llm_mean_score(self, original_response, paraphrased_response):
        """
        Calculates a robust, symmetric score by comparing A vs B and B vs A.
        """
        original_granularity = self.choice_granularity

        if self.choice_granularity not in [3, 5]:
            self.choice_granularity = 5

        try:
            if self.choice_granularity == 3:
                score_dict = {1: 1, 2: 0, 3: -1}
            else:
                score_dict = {1: 2, 2: 1, 3: 0, 4: -1, 5: -2}

            # --- Round 1: Paraphrased (A) vs Original (B) ---
            prompt1 = self.comparison_prompt.replace('[A]', paraphrased_response).replace(
                '[B]', original_response
            )
            choice1 = self.query_llm_once(prompt1)
            if choice1 is None:
                return 0.0
            score1 = score_dict.get(choice1, 0)

            # --- Round 2: Original (A) vs Paraphrased (B) ---
            prompt2 = self.comparison_prompt.replace('[A]', original_response).replace(
                '[B]', paraphrased_response
            )
            choice2 = self.query_llm_once(prompt2)
            if choice2 is None:
                return 0.0
            score2 = score_dict.get(choice2, 0)

            final_score = (score1 - score2) / 2.0
            logger.info(
                f'Mean score calculated: (score1={score1} - score2={score2}) / 2 = {final_score}'
            )
            return final_score

        finally:
            self.choice_granularity = original_granularity


if __name__ == '__main__':
    # Test block
    print('Initializing Oracle for test...')
    oracle = QualityOracle(oracle_model_name='deepseek-chat', choice_granularity=3)
    # ... test code ...
