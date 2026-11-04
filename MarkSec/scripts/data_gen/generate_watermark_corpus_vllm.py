import os

os.environ['VLLM_USE_V1'] = '0'
import sys
import json
import glob
import argparse
import tempfile
import torch
import gc
from typing import List
from tqdm import tqdm

# Force vLLM V0 engine to support LogitsProcessor.
# vLLM Imports
try:
    from vllm import LLM, SamplingParams
except ImportError:
    print('[Error] vLLM not installed. Please run: pip install vllm')
    sys.exit(1)

from transformers import AutoModelForCausalLM, AutoTokenizer

# -----------------------------------------------------------------------------
# 1. Environment and path configuration
# -----------------------------------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../../'))

if project_root not in sys.path:
    sys.path.append(project_root)

from src.watermark import load_watermark


# -----------------------------------------------------------------------------
# 2. Data loading logic (token-based truncation)
# -----------------------------------------------------------------------------
def load_local_c4_prompts(
    data_path: str,
    num_samples: int,
    tokenizer,
    prompt_token_length: int = 50,  # JSV suggests short prompts; ~50 tokens works well.
) -> List[str]:
    print(f'[Data] Loading from: {data_path}')
    prompts = []

    # Supports json/jsonl/json.gz
    files = glob.glob(os.path.join(data_path, '*.json')) + glob.glob(
        os.path.join(data_path, '*.json.gz')
    )
    if not files:
        raise FileNotFoundError(f'No json files found in {data_path}')

    from datasets import load_dataset

    # `streaming=True` greatly reduces memory usage.
    dataset = load_dataset('json', data_files=files, split='train', streaming=True)
    iterator = iter(dataset)

    print(f'[Data] Extracting {num_samples} prompts (Truncated to {prompt_token_length} tokens)...')
    with tqdm(total=num_samples, desc='Loading Prompts') as pbar:
        while len(prompts) < num_samples:
            try:
                item = next(iterator)
                text = item.get('text') or item.get('natural_text') or item.get('prompt', '')

                # 1. Filter out overly short source texts
                if len(text) < 200:
                    continue

                # 2. Truncate by token count (more robust)
                # Encode -> Slice -> Decode
                tokens = tokenizer.encode(text, truncation=False)
                if len(tokens) < prompt_token_length:
                    continue

                # Use first N tokens as prompt
                prompt_tokens = tokens[:prompt_token_length]
                prompt_text = tokenizer.decode(prompt_tokens, skip_special_tokens=True)

                prompts.append(prompt_text)
                pbar.update(1)

            except StopIteration:
                print('[Warning] Dataset exhausted.')
                break
    return prompts


def create_temp_algorithm_config(
    project_root: str,
    alg_name: str,
    alg_config_path: str,
    gamma: float,
    delta: float,
) -> tuple[str, dict]:
    if os.path.exists(alg_config_path):
        with open(alg_config_path, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
    else:
        config_dict = {'algorithm_name': alg_name}

    config_dict['algorithm_name'] = alg_name
    config_dict['gamma'] = gamma
    config_dict['delta'] = delta
    config_dict['greenlist_ratio'] = gamma
    config_dict['bias'] = delta

    temp_dir = os.path.join(project_root, 'output', 'tmp')
    os.makedirs(temp_dir, exist_ok=True)
    fd, temp_conf_path = tempfile.mkstemp(
        prefix=f'temp_config_vllm_{alg_name}_',
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
        if os.path.exists(temp_conf_path):
            os.remove(temp_conf_path)
        raise
    return temp_conf_path, config_dict


def prepare_output_file(output_file: str, overwrite_output: bool = False) -> None:
    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    if not os.path.exists(output_file):
        return
    if overwrite_output:
        os.remove(output_file)
        return
    if os.path.getsize(output_file) > 0:
        raise FileExistsError(
            f'Output file already exists and is non-empty: {output_file}. '
            'Pass --overwrite_output to replace it.'
        )
    os.remove(output_file)


def resolve_min_tokens(max_new_tokens: int, requested_min_tokens: int = 20) -> int:
    max_tokens = max(int(max_new_tokens), 0)
    min_tokens = max(int(requested_min_tokens), 0)
    return min(min_tokens, max_tokens)


def build_corpus_metadata(
    alg_name: str,
    alg_config_path: str,
    model_path: str,
    config_dict: dict,
) -> dict:
    return {
        'algorithm': str(alg_name),
        'model': str(model_path),
        'alg_config_path': os.path.abspath(str(alg_config_path)),
        'gamma': config_dict.get('gamma'),
        'delta': config_dict.get('delta'),
        'hash_key': config_dict.get('hash_key'),
        'prefix_length': config_dict.get('prefix_length'),
        'f_scheme': config_dict.get('f_scheme'),
        'window_scheme': config_dict.get('window_scheme'),
    }


def build_corpus_record(
    record_id: int,
    prompt: str,
    text_clean: str,
    text_wm: str,
    corpus_metadata: dict,
) -> dict:
    return {
        'id': int(record_id),
        'prompt': prompt,
        # Stealing corpus schema v2 adds watermark provenance for later audits.
        'schema_version': 'steal_corpus_v2',
        'base_completion': text_clean,
        'wm_completion': text_wm,
        'base_full_text': prompt + text_clean,
        'wm_full_text': prompt + text_wm,
        'clean_text': prompt + text_clean,
        'watermarked_text': prompt + text_wm,
        'generated_only': text_wm,
        'params': {
            'gamma': corpus_metadata.get('gamma'),
            'delta': corpus_metadata.get('delta'),
        },
        'algorithm': corpus_metadata.get('algorithm'),
        'model': corpus_metadata.get('model'),
        'corpus_metadata': corpus_metadata,
    }


# -----------------------------------------------------------------------------
# 3. Main flow
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='Watermark Corpus Generator (vLLM Optimized)')

    # vLLM parallelism args
    parser.add_argument(
        '--tensor_parallel_size', type=int, default=1, help='Number of GPUs (1 or 2)'
    )
    parser.add_argument(
        '--gpu_memory_utilization',
        type=float,
        default=0.80,
        help='vLLM gpu_memory_utilization passed to the engine.',
    )
    parser.add_argument(
        '--swap_space',
        type=int,
        default=16,
        help='vLLM CPU swap space in GiB.',
    )

    parser.add_argument('--model_path', type=str, default='facebook/opt-6.7b')
    parser.add_argument(
        '--dataset_path',
        type=str,
        default=os.path.join(project_root, 'data', 'c4'),
        help='Directory containing C4-style json/jsonl shards.',
    )

    # Watermark parameters
    parser.add_argument('--alg_name', type=str, default='KGW', choices=['KGW', 'Unigram', 'MPAC'])
    parser.add_argument('--alg_config', type=str, default=None)
    parser.add_argument('--gamma', type=float, default=0.25)
    parser.add_argument('--delta', type=float, default=2.0)

    # Generation controls
    parser.add_argument('--num_samples', type=int, default=1000)
    parser.add_argument('--max_new_tokens', type=int, default=200)
    # Use `prompt_token_len` naming for clarity
    parser.add_argument(
        '--prompt_token_len', type=int, default=50, help='Length of prompt in tokens'
    )
    parser.add_argument('--batch_size', type=int, default=1000)

    parser.add_argument('--output_file', type=str, default=None)
    parser.add_argument(
        '--overwrite_output',
        action='store_true',
        help='Replace an existing output file instead of appending into it.',
    )

    args = parser.parse_args()

    # Clear GPU memory cache
    if torch.cuda.is_available():
        gc.collect()
        torch.cuda.empty_cache()

    # 1. Configure paths and filenames
    if args.alg_config is None:
        local_candidates = [
            os.path.join(project_root, 'configs', 'watermark', args.alg_name, 'config.json'),
            os.path.join(project_root, 'configs', 'watermark', f'{args.alg_name}.json'),
        ]
        args.alg_config = next((path for path in local_candidates if os.path.exists(path)), '')

    if args.output_file is None:
        filename = f'{args.alg_name}_{args.model_path.split("/")[-1]}_vllm_g{args.gamma}_d{args.delta}.jsonl'
        args.output_file = os.path.join(project_root, 'output', 'corpus', filename)

    temp_conf_path = ''
    hf_model = None
    try:
        prepare_output_file(args.output_file, overwrite_output=args.overwrite_output)
        temp_conf_path, config_dict = create_temp_algorithm_config(
            project_root=project_root,
            alg_name=args.alg_name,
            alg_config_path=args.alg_config,
            gamma=args.gamma,
            delta=args.delta,
        )
        corpus_metadata = build_corpus_metadata(
            alg_name=args.alg_name,
            alg_config_path=args.alg_config,
            model_path=args.model_path,
            config_dict=config_dict,
        )

        # 3. Initialize vLLM engine (GPU)
        print(f'[Init] Initializing vLLM Engine on {args.tensor_parallel_size} GPUs...')
        llm = LLM(
            model=args.model_path,
            trust_remote_code=True,
            max_model_len=args.max_new_tokens + 200,  # Reserve buffer.
            gpu_memory_utilization=float(args.gpu_memory_utilization),
            dtype='float16',
            swap_space=int(args.swap_space),
            enforce_eager=False,
            tensor_parallel_size=args.tensor_parallel_size,
        )

        # 4. Initialize MarkSec watermark structures (CPU)
        print('[Init] Initializing MarkSec watermark structures (CPU mode)...')

        # Load tokenizer first for prompt processing.
        hf_tokenizer = AutoTokenizer.from_pretrained(args.model_path)

        hf_model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype=torch.float16,
            device_map='cpu',  # Keep HF model on CPU.
        )

        # 5. Load and preprocess prompts
        try:
            # Load 1.2x samples to absorb later filtering.
            prompts = load_local_c4_prompts(
                args.dataset_path, int(args.num_samples * 1.2), hf_tokenizer, args.prompt_token_len
            )
        except Exception as e:
            print(f'[Error] {e}')
            return

        print(f'[Run] Starting Batch Generation for {len(prompts)} prompts...')

        min_tokens = resolve_min_tokens(args.max_new_tokens, requested_min_tokens=20)
        watermark = load_watermark(
            algorithm_name=args.alg_name,
            algorithm_config_path=temp_conf_path,
            model=hf_model,
            tokenizer=hf_tokenizer,
            device='cpu',  # Keep watermark-side structures off GPU.
            max_new_tokens=args.max_new_tokens,
            min_new_tokens=min_tokens,
        )

        # 6. Sampling parameter setup
        sampling_params_clean = SamplingParams(
            n=1,
            temperature=1.0,
            seed=42,
            max_tokens=args.max_new_tokens,
            min_tokens=min_tokens,
            logits_processors=[],
        )

        sampling_params_wm = SamplingParams(
            n=1,
            temperature=1.0,
            seed=42,
            max_tokens=args.max_new_tokens,
            min_tokens=min_tokens,
            logits_processors=[watermark],
        )

        # 7. Run generation
        total_saved = 0

        for i in range(0, len(prompts), args.batch_size):
            if total_saved >= args.num_samples:
                break

            batch_prompts = prompts[i : i + args.batch_size]
            print(f'Processing batch {i} - {i + len(batch_prompts)}...')

            # Generate (vLLM handles internal batching)
            outputs_clean = llm.generate(batch_prompts, sampling_params_clean, use_tqdm=True)
            outputs_wm = llm.generate(batch_prompts, sampling_params_wm, use_tqdm=True)

            batch_records = []
            for j, (p, out_c, out_wm) in enumerate(zip(batch_prompts, outputs_clean, outputs_wm)):
                text_clean = out_c.outputs[0].text
                text_wm = out_wm.outputs[0].text

                # Filter out short generations
                if len(text_wm) < 100 or len(text_clean) < 100:
                    continue

                batch_records.append(
                    build_corpus_record(
                        record_id=total_saved + j,
                        prompt=p,
                        text_clean=text_clean,
                        text_wm=text_wm,
                        corpus_metadata=corpus_metadata,
                    )
                )

            # Persist batch immediately
            with open(args.output_file, 'a', encoding='utf-8') as f:
                for rec in batch_records:
                    json.dump(rec, f, ensure_ascii=False)
                    f.write('\n')

            total_saved += len(batch_records)
            print(f'Batch saved. Total Valid: {total_saved}')

        print(f'\n[Done] Output saved to {args.output_file}')
    finally:
        if os.path.exists(temp_conf_path):
            os.remove(temp_conf_path)
        del hf_model


if __name__ == '__main__':
    main()
