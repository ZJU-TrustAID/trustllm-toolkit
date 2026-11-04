#!/usr/bin/env python3
"""Generate token-annotated demo data for the MarkSec webpage.

Uses MarkSec's GreenlistWatermark to produce token-level green/red annotations
and detection metrics. Outputs JSON files to webpage/public/demo-data/.

Usage:
    python scripts/generate_demo_data.py [--model-path PATH] [--output-dir DIR]

Requirements:
    - A HuggingFace CausalLM model (default: Qwen2.5-7B-Instruct)
    - MarkSec installed (pip install -e .)
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch

# Ensure project root is on the path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.watermark.greenlist import (
    GreenlistLogitsProcessor,
    GreenlistWatermark,
    WatermarkRuntimeConfig,
)


def annotate_tokens(
    text: str,
    tokenizer,
    logits_processor: GreenlistLogitsProcessor,
    prompt: str,
) -> list[dict]:
    """Annotate each token as green/red/neutral."""
    full_ids = tokenizer(text, add_special_tokens=False)['input_ids']
    if isinstance(full_ids[0], list):
        full_ids = full_ids[0]
    full_ids = [int(tid) for tid in full_ids]

    prompt_ids = tokenizer(prompt, add_special_tokens=False)['input_ids']
    if isinstance(prompt_ids[0], list):
        prompt_ids = prompt_ids[0]
    prompt_len = len(prompt_ids)

    tokens = []
    for idx, tid in enumerate(full_ids):
        token_text = tokenizer.decode([tid], skip_special_tokens=False)
        if idx < prompt_len:
            label = 'neutral'
        else:
            context = full_ids[:idx]
            is_green = logits_processor.token_is_green(context, tid)
            label = 'green' if is_green else 'red'
        tokens.append({
            'text': token_text,
            'label': label,
            'token_id': tid,
            'position': idx,
        })
    return tokens


def compute_detection(
    text: str,
    tokenizer,
    logits_processor: GreenlistLogitsProcessor,
    gamma: float,
    z_threshold: float = 4.0,
) -> dict:
    """Compute watermark detection metrics."""
    token_ids = tokenizer(text, add_special_tokens=False)['input_ids']
    if isinstance(token_ids[0], list):
        token_ids = token_ids[0]
    token_ids = [int(tid) for tid in token_ids]

    prefix_length = logits_processor.prefix_length
    if len(token_ids) <= prefix_length:
        return {
            'z_score': 0.0,
            'green_fraction': 0.0,
            'num_scored_tokens': 0,
            'is_watermarked': False,
            'threshold': z_threshold,
        }

    green_count = 0
    total = 0
    for idx in range(prefix_length, len(token_ids)):
        if logits_processor.token_is_green(token_ids[:idx], token_ids[idx]):
            green_count += 1
        total += 1

    expected = gamma
    variance = total * expected * (1.0 - expected)
    z_score = 0.0 if variance <= 0.0 else (green_count - total * expected) / math.sqrt(variance)

    return {
        'z_score': round(float(z_score), 2),
        'green_fraction': round(float(green_count / total) if total else 0.0, 4),
        'num_scored_tokens': int(total),
        'is_watermarked': bool(z_score > z_threshold),
        'threshold': z_threshold,
    }


def apply_rw_attack(text: str, prompt: str) -> str:
    """Simulate a Random Walk attack by randomly replacing some words."""
    import random
    words = text[len(prompt):].split()
    if len(words) < 5:
        return text
    # Replace ~30% of words with synonyms (simplified simulation)
    n_replace = max(1, int(len(words) * 0.3))
    indices = random.sample(range(len(words)), min(n_replace, len(words)))
    replacements = [
        'the', 'a', 'is', 'was', 'are', 'were', 'has', 'have', 'had',
        'will', 'would', 'could', 'should', 'may', 'might', 'can',
        'this', 'that', 'these', 'those', 'it', 'its', 'they', 'them',
        'more', 'most', 'much', 'many', 'some', 'any', 'all', 'each',
        'very', 'quite', 'rather', 'just', 'also', 'still', 'already',
    ]
    for idx in indices:
        words[idx] = random.choice(replacements)
    return prompt + ' '.join(words)


def apply_llmp_attack(text: str, prompt: str, tokenizer, model, device: str) -> str:
    """Apply LLM Paraphrase attack using the model itself."""
    completion = text[len(prompt):]
    paraphrase_prompt = (
        'Paraphrase the following text to remove any potential watermarks while '
        'preserving meaning and fluency. Do not add or omit information.\n'
        f'Text:\n{completion}'
    )
    inputs = tokenizer(paraphrase_prompt, return_tensors='pt').to(device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    new_completion = tokenizer.decode(
        output_ids[0][inputs['input_ids'].shape[-1]:],
        skip_special_tokens=True,
    )
    return prompt + new_completion


def generate_scenario(
    *,
    scenario_id: str,
    title: str,
    description: str,
    prompt: str,
    model,
    tokenizer,
    device: str,
    gamma: float,
    delta: float,
    prefix_length: int,
    attack_fn,
    attack_label: str,
    attack_description: str,
) -> dict:
    """Generate a complete demo scenario with watermarked + attacked variants."""
    # Create watermark
    config_dict = {
        'gamma': gamma,
        'delta': delta,
        'hash_key': 15485863,
        'prefix_length': prefix_length,
        'z_threshold': 4.0,
    }
    runtime_config = WatermarkRuntimeConfig(
        model=model,
        tokenizer=tokenizer,
        device=device,
        max_new_tokens=128,
        min_new_tokens=32,
        do_sample=True,
    )
    watermark = GreenlistWatermark(
        config_dict=config_dict,
        runtime_config=runtime_config,
    )

    # Generate watermarked text
    print(f'  Generating watermarked text...')
    wm_text = watermark.generate_watermarked_text(prompt)
    wm_tokens = annotate_tokens(wm_text, tokenizer, watermark.logits_processor, prompt)
    wm_detection = compute_detection(wm_text, tokenizer, watermark.logits_processor, gamma)

    # Apply attack
    print(f'  Applying {attack_label} attack...')
    attacked_text = attack_fn(wm_text, prompt)
    attacked_tokens = annotate_tokens(attacked_text, tokenizer, watermark.logits_processor, prompt)
    attacked_detection = compute_detection(attacked_text, tokenizer, watermark.logits_processor, gamma)

    return {
        'id': scenario_id,
        'title': title,
        'description': description,
        'prompt': prompt,
        'model': str(getattr(model.config, '_name_or_path', 'unknown')),
        'watermark': {
            'type': 'KGW',
            'gamma': gamma,
            'delta': delta,
            'prefix_length': prefix_length,
        },
        'variants': [
            {
                'label': 'Watermarked',
                'description': 'Original text with KGW watermark embedded',
                'text': wm_text,
                'tokens': wm_tokens,
                'detection': wm_detection,
            },
            {
                'label': attack_label,
                'description': attack_description,
                'text': attacked_text,
                'tokens': attacked_tokens,
                'detection': attacked_detection,
                'attack_method': attack_label,
            },
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--model-path',
        default='/data/lkr/models/hf/Qwen/Qwen2.5-7B-Instruct',
        help='Path to the HuggingFace model',
    )
    parser.add_argument(
        '--output-dir',
        default=str(PROJECT_ROOT / 'webpage' / 'public' / 'demo-data'),
        help='Output directory for JSON files',
    )
    parser.add_argument('--gamma', type=float, default=0.25)
    parser.add_argument('--delta', type=float, default=2.0)
    parser.add_argument('--prefix-length', type=int, default=1)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f'Loading model from {args.model_path}...')
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16,
        device_map='auto',
        trust_remote_code=True,
    )
    device = str(next(model.parameters()).device)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    print(f'Model loaded on {device}')

    prompt = 'Explain what a zero-knowledge proof is and give a simple analogy.'

    # Scenario 1: KGW vs Random Walk
    print('\n[1/2] Generating KGW vs RW scenario...')
    rw_scenario = generate_scenario(
        scenario_id='kgw-rw',
        title='KGW Watermark vs Random Walk Attack',
        description='Random Walk perturbs word choices to break the watermark signal while trying to preserve meaning.',
        prompt=prompt,
        model=model,
        tokenizer=tokenizer,
        device=device,
        gamma=args.gamma,
        delta=args.delta,
        prefix_length=args.prefix_length,
        attack_fn=apply_rw_attack,
        attack_label='After RW Attack',
        attack_description='Random Walk attack applied — word-level perturbation to disrupt watermark signal.',
    )
    rw_path = output_dir / 'kgw-rw.json'
    rw_path.write_text(json.dumps(rw_scenario, indent=2, ensure_ascii=False))
    print(f'  Saved to {rw_path}')

    # Scenario 2: KGW vs LLMP
    print('\n[2/2] Generating KGW vs LLMP scenario...')
    llmp_scenario = generate_scenario(
        scenario_id='kgw-llmp',
        title='KGW Watermark vs LLM Paraphrase Attack',
        description='LLM Paraphrase rewrites the text using the model itself, attempting to remove the watermark while preserving semantics.',
        prompt=prompt,
        model=model,
        tokenizer=tokenizer,
        device=device,
        gamma=args.gamma,
        delta=args.delta,
        prefix_length=args.prefix_length,
        attack_fn=lambda text, prompt: apply_llmp_attack(text, prompt, tokenizer, model, device),
        attack_label='After LLMP Attack',
        attack_description='LLM Paraphrase attack applied — model rewrites its own output to remove watermark.',
    )
    llmp_path = output_dir / 'kgw-llmp.json'
    llmp_path.write_text(json.dumps(llmp_scenario, indent=2, ensure_ascii=False))
    print(f'  Saved to {llmp_path}')

    print('\nDone! Demo data generated successfully.')


if __name__ == '__main__':
    main()
