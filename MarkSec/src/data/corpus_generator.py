import os
import sys
import json
import glob
import tempfile
from importlib.util import find_spec
import torch
import gc
from typing import List
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

# vLLM Imports
try:
    from vllm import LLM, SamplingParams
except ImportError:
    pass  # Handle in __init__ if needed

# Add project root to sys.path to ensure imports work
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../../'))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.util.data import ChatFormatter
from src.watermark import load_watermark


class CorpusGenerator:
    def __init__(
        self,
        model_path: str,
        alg_name: str = 'KGW',
        gamma: float = 0.5,
        delta: float = 2.0,
        device: str = 'cuda',
        max_new_tokens: int = 200,
        use_vllm: bool = False,
        tensor_parallel_size: int = 1,
        load_model: bool = True,
        apply_chat_template: bool = False,
    ):
        """
        Initialize the CorpusGenerator.

        Args:
            model_path: Path to the model or HuggingFace model ID.
            alg_name: Name of the watermark algorithm (e.g., "Unigram", "KGW").
            gamma: Watermark parameter gamma (override config if provided).
            delta: Watermark parameter delta (override config if provided).
            device: Device to run generation on (ignored if use_vllm is True).
            max_new_tokens: Maximum new tokens to generate.
            use_vllm: Whether to use vLLM for generation.
            tensor_parallel_size: Number of GPUs for vLLM tensor parallelism.
            load_model: Whether to load the model and tokenizer upon initialization.
            apply_chat_template: Whether to wrap prompts in chat template for Instruct models.
        """
        self.model_path = model_path
        self.alg_name = alg_name
        self.gamma = gamma
        self.delta = delta
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.use_vllm = use_vllm
        self.tensor_parallel_size = tensor_parallel_size
        self.apply_chat_template = apply_chat_template

        self.model = None
        self.tokenizer = None
        self.watermark = None
        self.llm_engine = None
        self.formatter = None

        # Check vLLM availability
        if self.use_vllm and find_spec('vllm') is None:
            print('[Warning] vLLM not installed. Falling back to standard HuggingFace generation.')
            self.use_vllm = False

        if load_model:
            self.load_model_and_algorithm()

    def _get_config_path(self) -> str:
        """Returns the path to the watermark configuration file."""
        candidates = [
            os.path.join(project_root, 'configs', 'watermark', self.alg_name, 'config.json'),
            os.path.join(project_root, 'configs', 'watermark', f'{self.alg_name}.json'),
        ]
        for config_path in candidates:
            if os.path.exists(config_path):
                print(f'[Config] Using local config from: {config_path}')
                return config_path

        raise FileNotFoundError(f'Configuration file for {self.alg_name} not found.')

    def _create_temp_config(self) -> str:
        """Creates a temporary configuration file with current parameters."""
        try:
            config_path = self._get_config_path()
            with open(config_path, 'r', encoding='utf-8') as f:
                config_dict = json.load(f)
        except FileNotFoundError:
            config_dict = {'algorithm_name': self.alg_name}

        # Override with runtime parameters if provided
        if self.gamma is not None:
            config_dict['gamma'] = self.gamma
            config_dict['greenlist_ratio'] = self.gamma
        if self.delta is not None:
            config_dict['delta'] = self.delta
            config_dict['bias'] = self.delta

        temp_dir = os.path.join(project_root, 'output', 'tmp')
        os.makedirs(temp_dir, exist_ok=True)
        fd, temp_conf = tempfile.mkstemp(
            prefix=f'temp_config_{self.alg_name}_',
            suffix='.json',
            dir=temp_dir,
            text=True,
        )
        try:
            with os.fdopen(fd, 'w', encoding='utf-8') as f:
                json.dump(config_dict, f, ensure_ascii=True)
        except Exception:
            try:
                os.close(fd)
            except OSError:
                pass
            if os.path.exists(temp_conf):
                os.remove(temp_conf)
            raise
        return temp_conf

    def load_model_and_algorithm(self):
        """Loads the model, tokenizer, and watermark algorithm."""
        # Clean up memory
        if torch.cuda.is_available():
            gc.collect()
            torch.cuda.empty_cache()

        if self.use_vllm:
            self._load_vllm()
        else:
            self._load_hf()

        # Initialize formatter if needed
        if self.apply_chat_template:
            print(f'[Init] Initializing ChatFormatter for {self.model_path}')
            self.formatter = ChatFormatter(self.model_path, tokenizer=self.tokenizer)

    def _load_hf(self):
        print(f'[Init] Loading HF Model: {self.model_path}')
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path, device_map='auto', torch_dtype=torch.bfloat16
        )

        temp_conf = self._create_temp_config()
        try:
            print(f'[Init] Loading Algorithm: {self.alg_name}')
            self.watermark = load_watermark(
                algorithm_name=self.alg_name,
                algorithm_config_path=temp_conf,
                model=self.model,
                tokenizer=self.tokenizer,
                device=self.device,
                max_new_tokens=self.max_new_tokens,
                min_new_tokens=0,
            )
        finally:
            if os.path.exists(temp_conf):
                os.remove(temp_conf)

    def _load_vllm(self):
        print(f'[Init] Initializing vLLM Engine on {self.tensor_parallel_size} GPUs...')
        # vLLM Engine
        self.llm_engine = LLM(
            model=self.model_path,
            trust_remote_code=True,
            max_model_len=self.max_new_tokens + 512,
            gpu_memory_utilization=0.9,
            dtype='bfloat16',
            swap_space=16,
            enforce_eager=False,
            tensor_parallel_size=self.tensor_parallel_size,
        )

        # Build the watermark oracle on CPU to avoid consuming GPU memory needed by vLLM.
        print('[Init] Initializing MarkSec watermark structures (CPU mode)...')
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        hf_model_cpu = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.float32,  # float32 for CPU compatibility
            device_map='cpu',
        )

        temp_conf = self._create_temp_config()
        try:
            self.watermark = load_watermark(
                algorithm_name=self.alg_name,
                algorithm_config_path=temp_conf,
                model=hf_model_cpu,
                tokenizer=self.tokenizer,
                device='cpu',  # Important: Run watermark logic on CPU
                max_new_tokens=self.max_new_tokens,
                min_new_tokens=0,
            )
        finally:
            if os.path.exists(temp_conf):
                os.remove(temp_conf)

    def load_local_c4_prompts(
        self, data_path: str, num_samples: int, prompt_token_len: int = 50
    ) -> List[str]:
        """
        Loads prompts from a local C4 dataset (JSON/JSONL files).
        Returns truncated RAW text.
        """
        # Ensure tokenizer is loaded for prompt processing
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)

        print(f'[Data] Loading from: {data_path}')
        prompts = []

        files = glob.glob(os.path.join(data_path, '*.json')) + glob.glob(
            os.path.join(data_path, '*.json.gz')
        )
        if not files:
            if os.path.isfile(data_path):
                files = [data_path]
            else:
                raise FileNotFoundError(f'No json files found in {data_path}')

        dataset = load_dataset('json', data_files=files, split='train', streaming=True)
        iterator = iter(dataset)

        print(f'[Data] Extracting {num_samples} prompts...')
        with tqdm(total=num_samples, desc='Loading') as pbar:
            while len(prompts) < num_samples:
                try:
                    item = next(iterator)
                    text = item.get('text') or item.get('natural_text') or item.get('prompt', '')
                    if len(text) < 200:
                        continue

                    tokens = self.tokenizer.encode(text, truncation=False)
                    if len(tokens) < prompt_token_len:
                        continue

                    # Truncate to desired token length to avoid overly long prompts
                    prompt_text = self.tokenizer.decode(
                        tokens[:prompt_token_len], skip_special_tokens=True
                    )
                    prompts.append(prompt_text)
                    pbar.update(1)
                except StopIteration:
                    break
        return prompts

    def generate_corpus(
        self,
        dataset_path: str,
        output_file: str,
        num_samples: int = 2000,
        prompt_token_len: int = 50,
        min_gen_len: int = 50,
        batch_size: int = 1000,
    ):
        """
        Generates a watermarked corpus.
        """
        if (self.use_vllm and self.llm_engine is None) or (
            not self.use_vllm and self.model is None
        ):
            self.load_model_and_algorithm()

        # Check if output file exists and count existing lines for resume capability
        existing_count = 0
        if os.path.exists(output_file):
            print(f'[Resume] Checking existing output file: {output_file}')
            with open(output_file, 'r') as f:
                for _ in f:
                    existing_count += 1
            print(f'[Resume] Found {existing_count} existing samples. Resuming...')

        if existing_count >= num_samples:
            print(
                f'[Done] Target sample count ({num_samples}) already reached. Skipping generation.'
            )
            return

        # Load raw truncated prompts
        # We need to load enough prompts to cover existing + new ones, plus a buffer for filtering
        total_prompts_needed = int(num_samples * 1.2) + existing_count
        raw_prompts = self.load_local_c4_prompts(
            dataset_path, total_prompts_needed, prompt_token_len
        )

        # Apply chat formatting if enabled
        prompts_to_process = []
        if self.apply_chat_template:
            print('[Data] Applying Chat Template to prompts...')
            for p in raw_prompts:
                # Store tuple: (raw_text, formatted_prompt)
                fmt_p = self.formatter.format_c4_to_instruct(p)
                prompts_to_process.append((p, fmt_p))
        else:
            for p in raw_prompts:
                prompts_to_process.append((p, p))  # raw=formatted

        # Skip prompts that might have already been processed
        # Note: This is a rough estimation. Since we filter by length, 1 processed != 1 prompt.
        # But skipping 'existing_count' prompts is a safe heuristic to avoid re-doing the exact same start.
        # A better way would be to track processed IDs, but simple skip is usually sufficient for C4.
        if existing_count > 0:
            print(f'[Resume] Skipping first {existing_count} prompts...')
            prompts_to_process = prompts_to_process[existing_count:]

        print(
            f'[Run] Generating remaining {num_samples - existing_count} samples to {output_file}...'
        )
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        if self.use_vllm:
            self._generate_vllm(
                prompts_to_process,
                output_file,
                num_samples,
                min_gen_len,
                batch_size,
                start_id=existing_count,
            )
        else:
            self._generate_hf(
                prompts_to_process, output_file, num_samples, min_gen_len, start_id=existing_count
            )

        print(f'\n[Done] Generation Complete! Saved to {output_file}')

    def _generate_hf(self, prompts_to_process, output_file, num_samples, min_gen_len, start_id=0):
        count = start_id
        with open(output_file, 'a') as f:  # Use 'a' append mode
            for i, (raw_source, final_prompt) in enumerate(tqdm(prompts_to_process)):
                if count >= num_samples:
                    break
                try:
                    wm_text = self.watermark.generate_watermarked_text(final_prompt)
                    # Extract generated part: assuming wm_text starts with final_prompt
                    generated_only = wm_text[len(final_prompt) :]

                    if len(generated_only) < min_gen_len:
                        continue

                    record = {
                        'id': count,
                        'raw_source': raw_source,
                        'final_prompt': final_prompt,
                        'generated_only': generated_only,
                        'watermarked_text': generated_only,  # Per user instruction: only generated part
                        'model_id': self.model_path,
                        'formatter_used': self.apply_chat_template,
                        'algorithm': self.alg_name,
                    }
                    json.dump(record, f, ensure_ascii=False)
                    f.write('\n')
                    f.flush()
                    count += 1

                except Exception as e:
                    print(f'Error processing sample {i}: {e}')
                    continue

    def _generate_vllm(
        self, prompts_to_process, output_file, num_samples, min_gen_len, batch_size, start_id=0
    ):
        sampling_params_wm = SamplingParams(
            n=1,
            temperature=1.0,
            seed=42,
            max_tokens=self.max_new_tokens,
            min_tokens=min_gen_len,
            logits_processors=[self.watermark],
        )

        total_saved = start_id

        # Process in batches
        for i in range(0, len(prompts_to_process), batch_size):
            if total_saved >= num_samples:
                break

            batch_data = prompts_to_process[i : i + batch_size]
            batch_prompts = [x[1] for x in batch_data]  # Extract formatted prompts

            print(f'Processing batch {i} - {i + len(batch_prompts)}...')

            # vLLM handles internal batching
            outputs_wm = self.llm_engine.generate(batch_prompts, sampling_params_wm, use_tqdm=True)

            batch_records = []
            for j, (out_wm) in enumerate(outputs_wm):
                # Retrieve original data tuple using index
                raw_source, final_prompt = batch_data[j]

                # Use vLLM's prompt and output fields directly
                vllm_prompt = out_wm.prompt
                text_wm = out_wm.outputs[0].text

                if len(text_wm) < min_gen_len:
                    continue

                batch_records.append(
                    {
                        'id': total_saved
                        + len(batch_records),  # Increment ID correctly within batch
                        'raw_source': raw_source,
                        'final_prompt': vllm_prompt,
                        'generated_only': text_wm,
                        'watermarked_text': text_wm,  # Per user instruction: only generated part
                        'model_id': self.model_path,
                        'formatter_used': self.apply_chat_template,
                        'algorithm': self.alg_name,
                    }
                )

            # Append to file
            with open(output_file, 'a', encoding='utf-8') as f:
                for rec in batch_records:
                    json.dump(rec, f, ensure_ascii=False)
                    f.write('\n')

            total_saved += len(batch_records)
            print(f'Batch saved. Total Valid: {total_saved}')
