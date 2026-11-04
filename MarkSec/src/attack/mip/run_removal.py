import sys
import os

# --- 1. Path fix (ensure `src` can be imported) ---
current_path = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_path, '../../../'))
src_path = os.path.join(project_root, 'src')

if project_root not in sys.path:
    sys.path.insert(0, project_root)
if src_path not in sys.path:
    sys.path.insert(0, src_path)
# -----------------------------------

import argparse
import json
from tqdm import tqdm
from transformers import AutoTokenizer
from src.attack.mip.mip import WatermarkRemover


def main():
    parser = argparse.ArgumentParser(description='Run MIP Removal Attack (Greedy Substitution)')

    # Core parameters
    parser.add_argument(
        '--model_path',
        type=str,
        default='facebook/opt-6.7b',
        help='Hugging Face model ID or path to the base model tokenizer',
    )
    parser.add_argument(
        '--greenlist', type=str, required=True, help='Path to the stolen greenlist JSON file'
    )
    parser.add_argument(
        '--input_data', type=str, required=True, help='Path to the watermarked dataset (JSONL)'
    )
    parser.add_argument(
        '--output_file',
        type=str,
        default='output/results/mip_attack_final.jsonl',
        help='Path to save the attacked text',
    )

    # Attack configuration
    parser.add_argument('--num_samples', type=int, default=200, help='Number of samples to attack')
    parser.add_argument(
        '--max_edit_ratio',
        type=float,
        default=0.2,
        help='Max ratio of tokens to change per sentence',
    )

    args = parser.parse_args()

    # 1. Initialize remover
    print('>>> Initializing Watermark Remover...')
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        remover = WatermarkRemover(
            tokenizer=tokenizer,
            greenlist_path=args.greenlist,
        )
    except Exception as e:
        print(f'❌ Failed to initialize remover: {e}')
        return

    # 2. Load data
    print(f'>>> Loading data from {args.input_data}...')
    if not os.path.exists(args.input_data):
        print('❌ Input file not found!')
        return

    data_to_attack = []
    with open(args.input_data, 'r') as f:
        for i, line in enumerate(f):
            if len(data_to_attack) >= args.num_samples:
                break
            try:
                item = json.loads(line)
                # Prefer pure continuation field `generated_only`; fallback to `watermarked_text`.
                # This avoids prompt interference during evaluation.
                text = item.get('generated_only', item.get('watermarked_text', ''))

                if text and len(text) > 10:
                    data_to_attack.append(
                        {'id': i, 'original_text': text, 'prompt': item.get('prompt', '')}
                    )
            except Exception:
                continue

    print(f'Loaded {len(data_to_attack)} samples for attack.')

    # 3. Run attack
    print(f'>>> Running Greedy Attack (Max Edit Ratio: {args.max_edit_ratio})...')
    results = []
    total_changes = 0
    total_tokens = 0

    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

    for item in tqdm(data_to_attack, desc='Attacking'):
        original_text = item['original_text']

        # Run core attack logic
        attacked_text, num_changes = remover.attack_greedy(
            original_text, max_changes=args.max_edit_ratio
        )
        attack_meta = dict(getattr(remover, 'last_attack_meta', {}) or {})

        # Aggregate statistics
        input_len = len(remover.tokenizer.encode(original_text, add_special_tokens=False))
        total_changes += num_changes
        total_tokens += input_len

        results.append(
            {
                'id': item['id'],
                'prompt': item['prompt'],
                'original_text': original_text,
                'attacked_text': attacked_text,  # Keep key name `attacked_text` for downstream evaluation scripts.
                'num_changes': num_changes,
                'change_rate': num_changes / input_len if input_len > 0 else 0,
                'selected_key': attack_meta.get('selected_key'),
                'green_hits_before': attack_meta.get('green_hits_before'),
                'green_hits_after': attack_meta.get('green_hits_after'),
            }
        )

    # 4. Save results
    print(f'>>> Saving results to {args.output_file}...')
    with open(args.output_file, 'w') as f:
        for res in results:
            json.dump(res, f, ensure_ascii=False)
            f.write('\n')

    # 5. Print summary
    avg_change_rate = total_changes / total_tokens if total_tokens > 0 else 0
    print('\n' + '=' * 40)
    print('📊 Attack Summary')
    print('=' * 40)
    print(f'Processed Samples: {len(results)}')
    print(f'Total Tokens:      {total_tokens}')
    print(f'Total Changes:     {total_changes}')
    print(f'Avg Change Rate:   {avg_change_rate:.2%}')
    print('=' * 40)
    print('Done! Now you can verify the Z-Score of the attacked file.')


if __name__ == '__main__':
    main()
