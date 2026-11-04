import argparse
import json
import sys
from pathlib import Path

if __package__ in {None, ''}:
    repo_root = Path(__file__).resolve().parents[3]
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)

from src.attack.b4.distiller import B4DistillConfig, B4ProxyDistiller


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Train B4 proxy model from watermarked corpus')
    parser.add_argument(
        '--train_file',
        type=str,
        default='output/corpus/Unigram_opt-6.7b_HF_g0.5_d2.0.jsonl',
        help='Path to jsonl corpus containing watermarked samples.',
    )
    parser.add_argument('--proxy_model_name', type=str, default='facebook/opt-125m')
    parser.add_argument('--output_dir', type=str, default='output/models/b4_proxy_opt125m')
    parser.add_argument('--text_key', type=str, default='watermarked_text')
    parser.add_argument('--block_size', type=int, default=256)
    parser.add_argument('--max_samples', type=int, default=None)
    parser.add_argument('--epochs', type=float, default=3.0)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--save_steps', type=int, default=500)
    parser.add_argument('--save_total_limit', type=int, default=2)
    parser.add_argument('--logging_steps', type=int, default=50)
    parser.add_argument('--warmup_ratio', type=float, default=0.0)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--dataloader_num_workers', type=int, default=4)
    parser.add_argument('--optim', type=str, default='adamw_torch_fused')
    parser.add_argument('--attn_implementation', type=str, default='auto')
    parser.add_argument('--pack_texts', action='store_true')
    parser.add_argument('--bf16', action='store_true')
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--tf32', action='store_true')
    parser.add_argument('--torch_compile', action='store_true')
    parser.add_argument('--gradient_checkpointing', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--local_files_only', action='store_true')
    parser.add_argument('--trust_remote_code', action='store_true')
    parser.set_defaults(trust_remote_code=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = B4DistillConfig(
        train_file=args.train_file,
        proxy_model_name=args.proxy_model_name,
        output_dir=args.output_dir,
        text_key=args.text_key,
        block_size=args.block_size,
        max_samples=args.max_samples,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        logging_steps=args.logging_steps,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        dataloader_num_workers=args.dataloader_num_workers,
        optim=args.optim,
        attn_implementation=args.attn_implementation,
        pack_texts=args.pack_texts,
        bf16=True if args.bf16 else None,
        fp16=True if args.fp16 else None,
        tf32=True if args.tf32 else None,
        torch_compile=args.torch_compile,
        gradient_checkpointing=args.gradient_checkpointing,
        seed=args.seed,
        trust_remote_code=args.trust_remote_code,
        local_files_only=args.local_files_only,
    )
    stats = B4ProxyDistiller(config).run()
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
