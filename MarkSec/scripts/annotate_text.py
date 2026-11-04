#!/usr/bin/env python3
"""Annotate existing text with KGW green/red token labels.

This is a lightweight script that doesn't need a model — it takes pre-written
text and annotates each token using the KGW greenlist algorithm. Useful for
generating demo data when no GPU is available.

Usage:
    python scripts/annotate_text.py
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def make_greenlist_processor(vocab_size: int, gamma: float, delta: float, hash_key: int, prefix_length: int):
    """Create a standalone greenlist processor — pure Python, no torch needed."""

    class SimpleGreenlistProcessor:
        def __init__(self):
            self.vocab_size = vocab_size
            self.gamma = gamma
            self.delta = delta
            self.hash_key = hash_key
            self.prefix_length = prefix_length
            self.greenlist_size = max(1, min(vocab_size, int(round(gamma * vocab_size))))

        def _context_seed(self, token_ids: list[int]) -> int:
            if self.prefix_length <= 0:
                context = []
            else:
                context = token_ids[-self.prefix_length:]
            payload = ','.join(str(int(t)) for t in context)
            raw = f'{self.hash_key}:{payload}'.encode('utf-8')
            digest = hashlib.blake2b(raw, digest_size=8).digest()
            return int.from_bytes(digest, 'big') % (2**31 - 1)

        def _greenlist_ids(self, token_ids: list[int]) -> list[int]:
            seed = self._context_seed(token_ids)
            rng = random.Random(seed)
            all_ids = list(range(vocab_size))
            rng.shuffle(all_ids)
            return all_ids[:self.greenlist_size]

        def token_is_green(self, token_ids_before: list[int], token_id: int) -> bool:
            green = self._greenlist_ids(token_ids_before)
            return token_id in green

    return SimpleGreenlistProcessor()


def simple_tokenize(text: str) -> list[str]:
    """Simple word-level tokenizer for demo purposes."""
    import re
    tokens = re.findall(r"\w+|[^\w\s]", text, re.UNICODE)
    return tokens


def word_to_id(word: str, vocab_size: int = 32000) -> int:
    """Deterministic word-to-id mapping for demo."""
    return int(hashlib.md5(word.encode()).hexdigest(), 16) % vocab_size


def annotate_text(
    text: str,
    prompt: str,
    processor,
    vocab_size: int = 32000,
    *,
    watermark_bias: float = 0.0,
) -> tuple[list[dict], dict]:
    """Annotate text tokens and compute detection metrics.

    Args:
        watermark_bias: Probability of forcing a token to be green (simulates
            the effect of a watermarked model choosing green tokens). Set to 0
            for natural/attacked text, > 0 for watermarked text.
    """
    tokens = simple_tokenize(text)
    prompt_tokens = simple_tokenize(prompt)
    prompt_len = len(prompt_tokens)
    rng = random.Random(42)

    token_annotations = []
    token_ids = []
    for idx, tok in enumerate(tokens):
        tid = word_to_id(tok, vocab_size)
        token_ids.append(tid)
        if idx < prompt_len:
            label = 'neutral'
        else:
            is_green = processor.token_is_green(token_ids[:idx], tid)
            # Simulate watermark bias: a watermarked model prefers green tokens
            if watermark_bias > 0 and not is_green and rng.random() < watermark_bias:
                is_green = True
            label = 'green' if is_green else 'red'
        token_annotations.append({
            'text': tok,
            'label': label,
            'token_id': tid,
            'position': idx,
        })

    # Detection metrics
    gamma = processor.gamma
    prefix_length = processor.prefix_length
    green_count = sum(1 for t in token_annotations[prompt_len:] if t['label'] == 'green')
    total = len(token_annotations) - prefix_length
    expected = gamma
    variance = total * expected * (1.0 - expected) if total > 0 else 0
    z_score = 0.0 if variance <= 0 else (green_count - total * expected) / math.sqrt(variance)

    detection = {
        'z_score': round(z_score, 2),
        'green_fraction': round(green_count / total if total > 0 else 0, 4),
        'num_scored_tokens': total,
        'is_watermarked': bool(z_score > 4.0),
        'threshold': 4.0,
    }
    return token_annotations, detection


def main():
    vocab_size = 32000
    gamma = 0.25
    delta = 2.0
    hash_key = 15485863
    prefix_length = 1

    processor = make_greenlist_processor(vocab_size, gamma, delta, hash_key, prefix_length)

    prompt = 'Explain what a zero-knowledge proof is and give a simple analogy.'

    # Watermarked text (pre-written to look like model output with KGW watermark)
    wm_text = (
        prompt
        + ' A zero-knowledge proof is a cryptographic method where one party can prove '
        + 'to another that they know a value without revealing the value itself. '
        + 'Think of it like a blindfolded maze solver: you can convince someone you know '
        + 'the path through the maze by walking it correctly every time they ask, '
        + 'without ever showing them the actual path. The verifier learns nothing about '
        + 'the solution, only that you possess it. This technique has profound applications '
        + 'in blockchain privacy, authentication systems, and secure voting protocols.'
    )

    # Attacked text (RW — word substitutions that break watermark)
    rw_text = (
        prompt
        + ' A zero-knowledge proof is a cryptographic method where one party can prove '
        + 'to another that they know a value without revealing the value itself. '
        + 'Think of it like a blindfolded maze solver: you can convince someone you know '
        + 'the path through the maze by walking it correctly every time they ask, '
        + 'without ever showing them the actual path. The verifier learns nothing about '
        + 'the solution, only that you possess it. This technique has profound applications '
        + 'in blockchain privacy, authentication systems, and secure voting protocols.'
    )
    # Simulate RW attack by replacing some words
    rw_words = rw_text.split()
    random.seed(42)
    for i in range(len(rw_words)):
        if random.random() < 0.3 and i > len(prompt.split()):
            rw_words[i] = random.choice(['the', 'a', 'is', 'was', 'are', 'has', 'have',
                                          'this', 'that', 'more', 'some', 'very', 'also'])
    rw_text = ' '.join(rw_words)

    # Attacked text (LLMP — more aggressive paraphrase)
    llmp_text = (
        prompt
        + ' A zero-knowledge proof is a fundamental cryptographic protocol that enables '
        + 'a prover to demonstrate knowledge of certain information to a verifier '
        + 'without disclosing the information itself. Consider an analogy: imagine '
        + 'someone claims they can solve a jigsaw puzzle. To prove this without showing '
        + 'the completed picture, they could let you pick any piece and describe what '
        + 'neighbors it, repeatedly, until you are convinced they truly have the solution. '
        + 'The key insight is that the verifier gains certainty without gaining the secret. '
        + 'Modern applications include privacy-preserving transactions and identity verification.'
    )

    print('Annotating watermarked text...')
    wm_tokens, wm_detect = annotate_text(
        wm_text, prompt, processor, vocab_size, watermark_bias=0.55
    )

    print('Annotating RW-attacked text...')
    rw_tokens, rw_detect = annotate_text(
        rw_text, prompt, processor, vocab_size, watermark_bias=0.0
    )

    print('Annotating LLMP-attacked text...')
    llmp_tokens, llmp_detect = annotate_text(
        llmp_text, prompt, processor, vocab_size, watermark_bias=0.0
    )

    output_dir = PROJECT_ROOT / 'webpage' / 'public' / 'demo-data'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Scenario 1: KGW vs RW
    rw_scenario = {
        'id': 'kgw-rw',
        'title': 'KGW Watermark vs Random Walk Attack',
        'description': 'Random Walk perturbs word choices to break the watermark signal while trying to preserve meaning.',
        'prompt': prompt,
        'model': 'Qwen2.5-7B-Instruct',
        'watermark': {
            'type': 'KGW',
            'gamma': gamma,
            'delta': delta,
            'prefix_length': prefix_length,
        },
        'variants': [
            {
                'label': 'Watermarked',
                'description': 'Original text with KGW watermark embedded.',
                'text': wm_text,
                'tokens': wm_tokens,
                'detection': wm_detect,
            },
            {
                'label': 'After RW Attack',
                'description': 'Random Walk attack applied — word-level perturbation disrupts the watermark signal.',
                'text': rw_text,
                'tokens': rw_tokens,
                'detection': rw_detect,
                'attack_method': 'RW',
            },
        ],
    }

    # Scenario 2: KGW vs LLMP
    llmp_scenario = {
        'id': 'kgw-llmp',
        'title': 'KGW Watermark vs LLM Paraphrase Attack',
        'description': 'LLM Paraphrase rewrites the text using the model itself, attempting to remove the watermark while preserving semantics.',
        'prompt': prompt,
        'model': 'Qwen2.5-7B-Instruct',
        'watermark': {
            'type': 'KGW',
            'gamma': gamma,
            'delta': delta,
            'prefix_length': prefix_length,
        },
        'variants': [
            {
                'label': 'Watermarked',
                'description': 'Original text with KGW watermark embedded.',
                'text': wm_text,
                'tokens': wm_tokens,
                'detection': wm_detect,
            },
            {
                'label': 'After LLMP Attack',
                'description': 'LLM Paraphrase attack applied — the model rewrites its own output to remove the watermark.',
                'text': llmp_text,
                'tokens': llmp_tokens,
                'detection': llmp_detect,
                'attack_method': 'LLMP',
            },
        ],
    }

    for scenario in [rw_scenario, llmp_scenario]:
        scenario['mode'] = 'preset'
        scenario['provenance'] = {
            'kind': 'illustrative',
            'source': 'Prewritten text, word hashes, and simulated watermark bias.',
        }
        scenario['model'] = 'Preset illustration (no model inference)'

    rw_path = output_dir / 'kgw-rw.json'
    rw_path.write_text(json.dumps(rw_scenario, indent=2, ensure_ascii=False))
    print(f'Saved: {rw_path}')

    llmp_path = output_dir / 'kgw-llmp.json'
    llmp_path.write_text(json.dumps(llmp_scenario, indent=2, ensure_ascii=False))
    print(f'Saved: {llmp_path}')

    print('\nWatermarked detection:', wm_detect)
    print('RW attack detection:', rw_detect)
    print('LLMP attack detection:', llmp_detect)
    print('\nDone!')


if __name__ == '__main__':
    main()
