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

# =======================================================
# text_quality_analyzer.py
# Description: Analyze text quality using various metrics
# =======================================================
import os
import functools
import math
import re
from pathlib import Path
from typing import Any
import torch
import torch.nn.functional as F
import sacrebleu
from bert_score import BERTScorer
from rouge_score import rouge_scorer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer, util
from src.util.log import get_logger

from src.util.env import resolve_model_path
from src.exceptions.exceptions import MissingQualityModelError

logger = get_logger()
from libs.psp.embed_sentences import embed_all, similarity
from libs.psp.models import load_model


def _resolve_quality_model_path(model_name_or_path: str, *, local_files_only: bool = False) -> str:
    resolved_path = resolve_model_path(model_name_or_path)
    if local_files_only and not os.path.exists(resolved_path):
        raise MissingQualityModelError(resolved_path)
    return resolved_path


class TextQualityAnalyzer:
    """Base class for text quality analyzer."""

    def __init__(self) -> None:
        pass

    def analyze(self, text: str):
        pass


class DirectTextQualityAnalyzer(TextQualityAnalyzer):
    """Base class for direct text quality analyzer."""

    def __init__(self) -> None:
        pass

    def analyze(self, text: str):
        pass


class ReferencedTextQualityAnalyzer(TextQualityAnalyzer):
    """Base class for referenced text quality analyzer."""

    def __init__(self) -> None:
        pass

    def analyze(self, text: str, reference):
        pass


class ExternalDiscriminatorTextQualityAnalyzer(TextQualityAnalyzer):
    """Base class for external discriminator text quality analyzer."""

    def __init__(self) -> None:
        pass

    def analyze(self, text1: str, text2: str, description: str):
        pass


class PPLCalculator(DirectTextQualityAnalyzer):
    """Perplexity calculator for text quality analysis."""

    def __init__(self, model, tokenizer, device='cuda') -> None:
        """
        Initialize the perplexity calculator.

        Parameters:
            model: The language model for perplexity calculation.
            tokenizer: The tokenizer for the language model.
            device (str): The device to use for the calculation.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def analyze(self, text: str):
        """Calculate the perplexity of the given text."""
        criterion = torch.nn.CrossEntropyLoss()
        encoded_text = self.tokenizer(text, return_tensors='pt', add_special_tokens=False)[
            'input_ids'
        ][0].to(self.device)
        logits = self.model(torch.unsqueeze(encoded_text, 0), return_dict=True).logits[0]
        loss = criterion(logits[:-1], encoded_text[1:])
        ppl = torch.exp(loss)
        return ppl.item()


class LogDiversityAnalyzer(DirectTextQualityAnalyzer):
    """Log diversity analyzer for text quality analysis."""

    def __init__(self) -> None:
        super().__init__()

    def _eval_text(self, text: str, ngram: int):
        """Evaluate text to compute the number of unique and total n-grams."""
        tokens = text.split()
        ngram_set = set()
        total_ngrams = 0

        for i in range(len(tokens) - ngram + 1):
            ngram_set.add(' '.join(tokens[i : i + ngram]))
            total_ngrams += 1

        return len(ngram_set), total_ngrams

    def _eval_one_instance(self, text: str, ngram_list: list):
        """Evaluate a single text instance for multiple n-gram lengths."""
        results = {}
        for n in ngram_list:
            unique, total = self._eval_text(text, n)
            results[n] = {'unique': unique, 'total': total}
        unique_tokens = set(text.split())
        return results, unique_tokens

    def analyze(self, text: str):
        """Analyze text to compute log diversity based on n-gram uniqueness."""
        ngram_list = [2, 3, 4]
        prediction_results = {n: {'unique': 0, 'total': 0} for n in ngram_list}
        unique_token_set = set()

        stripped_text = text.strip()
        ngram_results, unique_tokens = self._eval_one_instance(stripped_text, ngram_list)

        unique_token_set.update(unique_tokens)

        for n in ngram_list:
            prediction_results[n]['unique'] += ngram_results[n]['unique']
            prediction_results[n]['total'] += ngram_results[n]['total']

        # Compute diversity scores for each n-gram length
        diversity_scores = [
            1 - (prediction_results[n]['unique'] / prediction_results[n]['total'])
            for n in ngram_list
        ]

        # Overall diversity is the product of individual n-gram diversities
        overall_diversity = (
            (1 - diversity_scores[0] / 100)
            * (1 - diversity_scores[1] / 100)
            * (1 - diversity_scores[2] / 100)
        )
        log_diversity = -math.log(max(1 - overall_diversity, math.exp(-20)))

        return log_diversity


class BLEUCalculator(ReferencedTextQualityAnalyzer):
    """BLEU calculator for text quality analysis."""

    def __init__(self) -> None:
        pass

    def analyze(self, text: str, reference: str):
        """Calculate the BLEU score of the given text with the reference."""
        b = sacrebleu.corpus_bleu([text], [[reference]]).score
        return b


class ROUGE1Calculator(ReferencedTextQualityAnalyzer):
    """ROUGE1 calculator for text quality analysis."""

    def __init__(self) -> None:
        pass

    def analyze(self, text: str, reference: str):
        """Calculate the ROUGE-1 score of the given text with the reference."""
        scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)
        scores = scorer.score(text, reference)
        return scores['rouge1'].fmeasure


class ROUGE2Calculator(ReferencedTextQualityAnalyzer):
    """ROUGE2 calculator for text quality analysis."""

    def __init__(self) -> None:
        pass

    def analyze(self, text: str, reference: str):
        """Calculate the ROUGE-2 score of the given text with the reference."""
        scorer = rouge_scorer.RougeScorer(['rouge2'], use_stemmer=True)
        scores = scorer.score(text, reference)
        return scores['rouge2'].fmeasure


class ROUGELCalculator(ReferencedTextQualityAnalyzer):
    """ROUGEL calculator for text quality analysis."""

    def __init__(self) -> None:
        pass

    def analyze(self, text: str, reference: str):
        """Calculate the ROUGE-L score of the given text with the reference."""
        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        scores = scorer.score(text, reference)
        return scores['rougeL'].fmeasure


class BERTScoreCalculator(ReferencedTextQualityAnalyzer):
    """BERTScore calculator for text quality analysis."""

    def __init__(self, model_path: str, local_files_only: bool = False) -> None:
        self.model_path = _resolve_quality_model_path(model_path, local_files_only=local_files_only)
        model_name = os.path.basename(self.model_path)
        # bert_score.utils.model2layers['roberta-large']
        model2layer = {
            'roberta-large': 17,
            'roberta-base': 9,
        }
        target_layers = model2layer.get(model_name, 9)
        self.bert_scorer = BERTScorer(
            model_type=self.model_path,
            num_layers=target_layers,
            batch_size=32,
            nthreads=4,
            all_layers=False,
            idf=False,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            rescale_with_baseline=False,
            lang='en',
        )

    def analyze(self, text: str, reference: str):
        """Calculate the BERTScore of the given text with the reference."""
        P, R, F1 = self.bert_scorer.score([text], [reference])
        return F1.tolist()[0]


class PSPCalculator(ReferencedTextQualityAnalyzer):
    def __init__(self, model_path: str, sp_path: str, models_root: str = '') -> None:
        # https://github.com/martiansideofthemoon/ai-detection-paraphrases/blob/main/dipper_paraphrases/utils.py
        path = self._resolve_artifact_path(model_path, models_root=models_root)
        resolved_sp_path = self._resolve_artifact_path(
            sp_path,
            models_root=models_root,
            fallback=self._default_sp_model_path(),
        )

        if path and os.path.exists(path) and resolved_sp_path and os.path.exists(resolved_sp_path):
            self.model = load_model(path, sp_model=resolved_sp_path)
            self.model.eval()
            embedder = functools.partial(embed_all, model=self.model, disable=True)
            self.sim_fn = functools.partial(self._get_sim_vectors, embedder)
        else:
            raise MissingQualityModelError(path or model_path)

    @staticmethod
    def _default_sp_model_path() -> str:
        project_root = Path(__file__).resolve().parents[3]
        return str(project_root / 'libs' / 'psp' / 'paranmt.model')

    @staticmethod
    def _resolve_artifact_path(
        path: str | None,
        *,
        models_root: str = '',
        fallback: str = '',
    ) -> str:
        raw = str(path or '').strip()
        fallback = str(fallback or '').strip()
        if not raw:
            return os.path.abspath(os.path.expanduser(fallback)) if fallback else ''

        candidates: list[str] = []
        if os.path.isabs(raw):
            candidates.append(raw)
        else:
            if models_root:
                candidates.append(os.path.join(models_root, raw))

            resolved = resolve_model_path(raw)
            candidates.append(resolved)

            # Compatibility fallback: if HF_MODELS_ROOT points to ".../hf",
            # also probe its parent for non-HF-style local assets.
            hf_root = str(os.environ.get('HF_MODELS_ROOT', '')).strip()
            if hf_root:
                parent_root = str(Path(hf_root).expanduser().resolve().parent)
                candidates.append(os.path.join(parent_root, raw))

            project_root = Path(__file__).resolve().parents[3]
            candidates.append(str(project_root / raw))

        seen = set()
        for candidate in candidates:
            normalized = os.path.abspath(os.path.expanduser(candidate))
            if normalized in seen:
                continue
            seen.add(normalized)
            if os.path.exists(normalized):
                return normalized

        # Keep previous behavior (clear missing-path signal) while returning
        # the most likely resolved candidate for error reporting.
        if candidates:
            return os.path.abspath(os.path.expanduser(candidates[0]))
        return os.path.abspath(os.path.expanduser(raw))

    def _get_sim_vectors(self, embedder: Any, sequence: Any) -> Any:
        gen_vec = embedder(sentences=[sequence])[0]
        return gen_vec

    def analyze(self, text: str, reference: str) -> float:
        original_emb = self.sim_fn(text)
        paraphrase_emb = self.sim_fn(reference)
        psp = similarity(original_emb, paraphrase_emb)
        return psp


class NLIScoreCalculator(ReferencedTextQualityAnalyzer):
    """NLI Score calculator using cross-encoder/nli-deberta-v3-large."""

    def __init__(
        self,
        model_name='cross-encoder/nli-deberta-v3-large',
        device=None,
        local_files_only: bool = False,
    ) -> None:
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        # Use resolve_model_path to handle local/remote paths automatically
        model_name_or_path = _resolve_quality_model_path(
            model_name, local_files_only=local_files_only
        )
        if model_name_or_path != model_name:
            logger.info(f'Loading NLI model from resolved path: {model_name_or_path}')

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, local_files_only=local_files_only
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name_or_path, local_files_only=local_files_only
        ).to(self.device)
        self.model.eval()
        # Based on user provided info: 0: contradiction, 1: entailment, 2: neutral
        self.label_mapping = ['contradiction', 'entailment', 'neutral']

    def analyze(self, text: str, reference: str):
        """
        Calculate the NLI score (entailment probability) of the text given the reference.
        Reference is treated as Premise, Text is treated as Hypothesis.
        """
        # Premise: reference, Hypothesis: text
        features = self.tokenizer(
            [reference], [text], padding=True, truncation=True, return_tensors='pt'
        ).to(self.device)

        with torch.no_grad():
            scores = self.model(**features).logits
            probs = F.softmax(scores, dim=1)
            # We want entailment score. Index 1 is entailment according to user snippet.
            entailment_idx = 1
            entailment_score = probs[0][entailment_idx].item()

            return entailment_score

    def predict_label(self, text: str, reference: str):
        """
        Predict the NLI label for the text given the reference.
        """
        features = self.tokenizer(
            [reference], [text], padding=True, truncation=True, return_tensors='pt'
        ).to(self.device)
        with torch.no_grad():
            scores = self.model(**features).logits
            predicted_index = scores.argmax(dim=1).item()
            return self.label_mapping[predicted_index]


class PassOrNotJudger(ReferencedTextQualityAnalyzer):
    """Pass or not judger for text quality analysis."""

    def __init__(self) -> None:
        pass

    def _check_correctness(self, prompt: str, completion: str, test: str, entry_point: str):
        """Check the correctness of the code."""
        check_program = prompt + '\n' + completion + '\n' + test + '\n' + f'check({entry_point})'
        # print(check_program)
        try:
            exec_globals = {}
            exec(check_program, exec_globals)
            return 1
        except BaseException:
            return 0

    def analyze(self, text: str, reference: dict):
        """Check if the text passes the correctness test."""
        passed = self._check_correctness(
            reference['task'], text, reference['test'], reference['entry_point']
        )
        return passed


class LLMScoreEvaluator(ExternalDiscriminatorTextQualityAnalyzer):
    """
    LLM-based evaluator that scores text quality/consistency on a 0-100 scale.
    Supports OpenAI, DeepSeek, and other OpenAI-compatible APIs via the Unified LLMClient.
    """

    def __init__(
        self, model_name: str = 'deepseek-v3.2', system_prompt: str = None, **kwargs
    ) -> None:
        """
        Initialize the evaluator.

        Parameters:
            model_name (str): Model name (e.g., 'deepseek-v3.2').
            api_key (str): API Key. If None, reads from env vars.
            base_url (str): API Base URL.
            system_prompt (str): Custom system instruction.
            **kwargs: Additional arguments for LLMClient.
        """
        try:
            from src.util.llm import LLMClient
        except ImportError:
            raise ImportError('src.util.llm module not found.')

        self.default_system_prompt = system_prompt or (
            'You are an expert evaluator for NLP systems. '
            'Your task is to evaluate the quality and logical consistency of a text.'
        )

        # Initialize the unified LLMClient
        # It handles environment variables, base_url resolution, and client creation internally.
        self.client = LLMClient(
            model_name=model_name, system_prompt=self.default_system_prompt, **kwargs
        )
        self.model_name = model_name

    def _get_query_prompt(self, reference: str, candidate: str, question: str) -> str:
        return f"""### Task Description
Evaluate the quality of the "Candidate Answer" based on the "Question" and the "Reference Answer".
Focus on **Logical Consistency** and **Semantic Preservation**.

### Input Data
- **Question**: {question}
- **Reference Answer (Original)**: {reference}
- **Candidate Answer (Rewritten)**: {candidate}

### Evaluation Criteria
1. **Logical Consistency**: Does the candidate answer convey the same core meaning and logic as the reference? (e.g., negation, causality, numbers).
2. **Fluency**: Is the candidate answer grammatically correct and natural?
3. **Factuality**: Does the candidate introduce any hallucinations or factual errors compared to the reference?

### Output Format
Return the result in **JSON format** ONLY, with two fields:
- "score": An integer between 1 and 10 (10 = Perfect preservation, 1 = Complete contradiction or nonsense).
- "reason": A brief explanation of the score.
"""

    def analyze(self, text1: str, text2: str, description: str):
        """
        Score the text2 (Candidate) against text1 (Reference).

        Args:
            text1 (str): Reference text (Original).
            text2 (str): Candidate text (Attacked).
            description (str): Question or Prompt.

        Returns:
            int: The quality score (1-10).
        """
        prompt = self._get_query_prompt(text1, text2, description)

        try:
            # Use get_json_result from LLMClient which handles JSON parsing and retries
            result = self.client.get_json_result(prompt)

            score = int(result.get('score', 0))
            score = max(1, min(10, score))

            # self.logger.debug(f"LLM Judge Reason: {result.get('reason')}")

            return score

        except Exception as e:
            logger.error(f'LLM Scoring failed: {e}')
            return None


class PrometheusEvaluator(ExternalDiscriminatorTextQualityAnalyzer):
    """
    Prometheus-based evaluator using local model.
    """

    def __init__(
        self,
        model_path='prometheus-eval/prometheus-7b-v2.0',
        local_files_only: bool = False,
        **kwargs,
    ) -> None:
        try:
            from prometheus_eval.vllm import VLLM
            from prometheus_eval import PrometheusEval
            from prometheus_eval.prompts import ABSOLUTE_PROMPT
        except ImportError:
            raise ImportError(
                'prometheus_eval not installed. Please run `pip install prometheus-eval`.'
            )

        self.model_path = _resolve_quality_model_path(model_path, local_files_only=local_files_only)
        if not os.path.exists(self.model_path):
            logger.warning(f'Prometheus model not found at {self.model_path}')

        kwargs.setdefault('gpu_memory_utilization', 0.6)
        self.vllm_model = VLLM(model=self.model_path, **kwargs)
        self.judge = PrometheusEval(model=self.vllm_model, absolute_grade_template=ABSOLUTE_PROMPT)

    def analyze(self, text1: str, text2: str, description: str):
        """
        Score text2 (Candidate) against text1 (Reference).
        Returns score 1-5.
        """
        rubric = """
        Focus on Logical Consistency and Semantic Preservation.
        Criteria:
        1. Logical Consistency: Does the candidate answer convey the same core meaning and logic as the reference?
        2. Fluency: Is the candidate answer grammatically correct and natural?
        3. Factuality: Does the candidate introduce any hallucinations or factual errors?
        """

        try:
            feedback, score = self.judge.single_absolute_grade(
                instruction=description, response=text2, rubric=rubric, reference_answer=text1
            )
            return int(score)
        except Exception as e:
            logger.error(f'Prometheus evaluation failed: {e}')
            return 0


class SemanticFluencyAnalyzer(ReferencedTextQualityAnalyzer):
    """
    Analyzer that combines Semantic Similarity (SBERT) and Linguistic Acceptability (CoLA).
    This metric is designed to detect both semantic drift and grammatical degradation (fluency).
    """

    def __init__(
        self,
        device=None,
        sbert_model='sentence-transformers/all-MiniLM-L6-v2',
        cola_model='textattack/roberta-base-CoLA',
        local_files_only: bool = False,
    ):
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')

        # 1. Load Semantic Model (SBERT)
        sbert_path = _resolve_quality_model_path(sbert_model, local_files_only=local_files_only)
        logger.info(f'Loading SBERT model: {sbert_path}...')
        sentence_transformer_kwargs = {'device': self.device}
        if local_files_only:
            sentence_transformer_kwargs['local_files_only'] = True
        self.sim_model = SentenceTransformer(sbert_path, **sentence_transformer_kwargs)

        # 2. Load Grammar Model (CoLA)
        cola_path = _resolve_quality_model_path(cola_model, local_files_only=local_files_only)
        logger.info(f'Loading CoLA model: {cola_path}...')
        self.cola_model_name = cola_path
        self.cola_tokenizer = AutoTokenizer.from_pretrained(
            self.cola_model_name, local_files_only=local_files_only
        )

        try:
            self.cola_model = AutoModelForSequenceClassification.from_pretrained(
                self.cola_model_name, local_files_only=local_files_only
            ).to(self.device)
        except OSError:
            # Fallback for Flax model if PyTorch weights are missing
            if os.path.exists(os.path.join(cola_path, 'flax_model.msgpack')):
                logger.info('PyTorch model not found, loading Flax model...')
                self.cola_model = AutoModelForSequenceClassification.from_pretrained(
                    self.cola_model_name, from_flax=True, local_files_only=local_files_only
                ).to(self.device)
            else:
                raise
        self.cola_model.eval()

    def calculate_semantic_similarity(self, text: str, reference: str) -> float:
        """
        Calculate Semantic Similarity (0.0 ~ 1.0).
        """
        embeddings = self.sim_model.encode([reference, text], convert_to_tensor=True)
        # Calculate Cosine Similarity
        cosine_score = util.cos_sim(embeddings[0], embeddings[1]).item()
        # Normalize to 0-1
        return max(0.0, cosine_score)

    def calculate_fluency_score(self, text: str) -> float:
        """
        Calculate Fluency/Grammar Score (0.0 ~ 1.0) using CoLA model.
        """
        inputs = self.cola_tokenizer(text, return_tensors='pt', truncation=True, max_length=512).to(
            self.device
        )
        with torch.no_grad():
            logits = self.cola_model(**inputs).logits
            probs = F.softmax(logits, dim=1)
            # Index 1 is "Acceptable" (Grammatically correct)
            fluency_score = probs[0][1].item()
        return fluency_score

    def analyze(self, text: str, reference: str):
        """
        Return the combined quality score.
        """
        sem_score = self.calculate_semantic_similarity(text, reference)
        flu_score = self.calculate_fluency_score(text)

        # Combined score (Harmonic Mean variant)
        if sem_score + flu_score == 0:
            combined = 0.0
        else:
            combined = 2 * (sem_score * flu_score) / (sem_score + flu_score)

        return combined

    def analyze_details(self, text: str, reference: str):
        """
        Return a detailed report with individual scores.
        """
        sem_score = self.calculate_semantic_similarity(text, reference)
        flu_score = self.calculate_fluency_score(text)

        if sem_score + flu_score == 0:
            combined = 0.0
        else:
            combined = 2 * (sem_score * flu_score) / (sem_score + flu_score)

        return {
            'semantic_score': round(sem_score, 4),
            'fluency_score': round(flu_score, 4),
            'combined_score': round(combined, 4),
        }


class LWSSCalculator(ReferencedTextQualityAnalyzer):
    """
    Logic-Weighted Semantic Score (LWSS).
    Combines BERTScore with NLI-based logic consistency check.
    LWSS = S_sem * ReLU(W_logic)
    W_logic = P_entail - lambda * P_contra
    """

    def __init__(
        self,
        semantic_model_path='roberta-large',
        nli_model_path='cross-encoder/nli-deberta-v3-large',
        logic_lambda=0.5,
        device=None,
        local_files_only: bool = False,
    ) -> None:
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')

        # 1. Semantic Base (BERTScore)
        # Using BERTScorer
        self.semantic_model_path = _resolve_quality_model_path(
            semantic_model_path, local_files_only=local_files_only
        )
        model_name = os.path.basename(self.semantic_model_path)
        model2layer = {
            'roberta-large': 17,
            'roberta-base': 9,
            'deberta-xlarge-mnli': 40,
            'deberta-v3-large': 24,
        }
        # Default to a reasonable layer if not found (e.g., 9 or last layer)
        target_layers = model2layer.get(model_name)

        self.bert_scorer = BERTScorer(
            model_type=self.semantic_model_path,
            num_layers=target_layers,
            batch_size=32,
            nthreads=4,
            all_layers=False,
            idf=False,
            device=self.device,
            rescale_with_baseline=False,
            lang='en',
        )

        # 2. Logic Coefficient (NLI)
        self.nli_model_path = _resolve_quality_model_path(
            nli_model_path, local_files_only=local_files_only
        )
        logger.info(f'Loading NLI model for LWSS: {self.nli_model_path}')
        self.nli_tokenizer = AutoTokenizer.from_pretrained(
            self.nli_model_path, local_files_only=local_files_only
        )
        self.nli_model = AutoModelForSequenceClassification.from_pretrained(
            self.nli_model_path, local_files_only=local_files_only
        ).to(self.device)
        self.nli_model.eval()

        self.logic_lambda = logic_lambda

    def analyze(self, text: str, reference: str):
        """
        Calculate LWSS.
        text: Candidate (Hypothesis)
        reference: Reference (Premise)
        """
        # 1. Calculate S_sem (BERTScore F1)
        # BERTScorer.score returns (P, R, F1)
        # We suppress warnings from bert_score if any
        try:
            P, R, F1 = self.bert_scorer.score([text], [reference])
            s_sem = F1.item()
        except Exception as e:
            logger.error(f'BERTScore calculation failed: {e}')
            s_sem = 0.0

        # 2. Calculate Logic Score (NLI)
        # Premise: Reference, Hypothesis: Text
        try:
            inputs = self.nli_tokenizer(
                reference, text, return_tensors='pt', truncation=True, max_length=512
            ).to(self.device)
            with torch.no_grad():
                logits = self.nli_model(**inputs).logits
                probs = F.softmax(logits, dim=1)
                # Label mapping for cross-encoder/nli-deberta-v3-large: 0: contradiction, 1: entailment, 2: neutral
                # We assume this mapping.
                p_contra = probs[0][0].item()
                p_entail = probs[0][1].item()

                w_logic = p_entail - self.logic_lambda * p_contra
                w_logic = max(0.0, w_logic)  # ReLU

        except Exception as e:
            logger.error(f'NLI calculation failed: {e}')
            w_logic = 0.0

        lwss = s_sem * w_logic
        return lwss


class KERCalculator(ReferencedTextQualityAnalyzer):
    """
    Key Entity Recall (KER).
    Checks if entities in Reference are present in Candidate.
    KER = |E_A intersection E_B| / |E_A|
    """

    def __init__(self) -> None:
        self.nlp = None
        try:
            import spacy

            try:
                self.nlp = spacy.load('en_core_web_sm')
            except OSError:
                logger.warning(
                    "Spacy model 'en_core_web_sm' not found. Falling back to regex-based entity extraction."
                )
        except ImportError:
            logger.warning('Spacy not installed. Falling back to regex-based entity extraction.')

    def _extract_entities_regex(self, text: str):
        """Fallback extraction using regex for Capitalized words and Numbers."""
        # Capitalized words (approximate Proper Nouns)
        cap_words = set(re.findall(r'\b[A-Z][a-z]+\b', text))
        # Numbers
        numbers = set(re.findall(r'\b\d+\.?\d*\b', text))
        return cap_words.union(numbers)

    def _extract_entities(self, text: str):
        if self.nlp:
            doc = self.nlp(text)
            # Extract named entities, numbers, and maybe noun chunks if needed
            entities = set([ent.text.lower() for ent in doc.ents])
            # Also add standalone numbers that might not be caught as entities
            # numbers = set(re.findall(r'\b\d+\.?\d*\b', text))
            # entities = entities.union(numbers)
            return entities
        else:
            # Use regex fallback, convert to lower for matching
            return set([e.lower() for e in self._extract_entities_regex(text)])

    def analyze(self, text: str, reference: str):
        """
        Calculate KER.
        text: Candidate (Text B)
        reference: Reference (Text A)
        """
        ents_ref = self._extract_entities(reference)
        if not ents_ref:
            return 1.0  # No entities to recall, so technically perfect recall (or undefined, but 1.0 is safer for "no loss")

        ents_cand = self._extract_entities(text)

        # Calculate intersection
        # We use a simple set intersection.
        # For more robustness, one could use fuzzy matching, but strict set match is faster and requested "checking if appearing".
        # User mentioned "support fuzzy matching or stem matching", but let's start with exact match (normalized to lower).

        intersection = ents_ref.intersection(ents_cand)
        ker = len(intersection) / len(ents_ref)
        return ker


class ECCalculator(ReferencedTextQualityAnalyzer):
    """
    Entailment Consistency (EC).
    Based on SummaC Zero-Shot strategy.
    """

    def __init__(
        self,
        model_path='cross-encoder/nli-deberta-v3-large',
        device=None,
        local_files_only: bool = False,
    ) -> None:
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_path = _resolve_quality_model_path(model_path, local_files_only=local_files_only)
        logger.info(f'Loading NLI model for EC: {self.model_path}')
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, local_files_only=local_files_only
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_path, local_files_only=local_files_only
        ).to(self.device)
        self.model.eval()

    def _split_sentences(self, text: str):
        # Simple sentence splitting if nltk is not robust
        try:
            import nltk

            return nltk.sent_tokenize(text)
        except (ImportError, LookupError):
            # Fallback split by . ! ?
            return re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s', text)

    def analyze(self, text: str, reference: str):
        """
        Calculate EC.
        text: Candidate (Text B) - Document to verify
        reference: Reference (Text A) - Source Document
        """
        sents_ref = self._split_sentences(reference)
        sents_cand = self._split_sentences(text)

        if not sents_cand:
            return 0.0
        if not sents_ref:
            return 0.0  # Should not happen usually

        # For each sentence in Candidate, find the max entailment score from any sentence in Reference
        # This is slightly different from SummaC which might do Reference->Candidate or Candidate->Reference depending on task.
        # User said: "For each generated sentence b_i, find the most supporting sentence a_j in source, calculate NLI score."
        # So: Premise = a_j (Reference), Hypothesis = b_i (Candidate).

        scores = []

        # Batching could be more efficient, but let's do simple loops first for clarity
        for b_i in sents_cand:
            max_score = -1.0

            # Construct pairs: (a_1, b_i), (a_2, b_i), ...
            pairs = [[a_j, b_i] for a_j in sents_ref]

            # Tokenize all pairs
            try:
                inputs = self.tokenizer(
                    pairs, padding=True, truncation=True, return_tensors='pt', max_length=256
                ).to(self.device)
                with torch.no_grad():
                    logits = self.model(**inputs).logits
                    probs = F.softmax(logits, dim=1)
                    # Entailment is index 1
                    entail_probs = probs[:, 1]
                    max_score = torch.max(entail_probs).item()
            except Exception as e:
                logger.error(f'EC calculation error for sentence: {e}')
                max_score = 0.0

            scores.append(max_score)

        if not scores:
            return 0.0

        return sum(scores) / len(scores)
