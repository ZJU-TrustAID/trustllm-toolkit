"""End to end: load model -> optimize -> generate -> print/save the measured result."""

import argparse
from pathlib import Path

from textbook_toolkit.jailbreak import get
from textbook_toolkit.jailbreak.optimization import CausalLM, COLDConfig, GCGConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("gcg", "cold_attack"), default="gcg")
    parser.add_argument(
        "--model",
        default="HuggingFaceTB/SmolLM2-135M-Instruct",
        help="HF checkpoint/local directory, or 'tiny' for random offline smoke test",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--dtype", choices=("float32", "float16", "bfloat16"), default="float32"
    )
    parser.add_argument("--revision")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--plain", action="store_true", help="Disable chat template explicitly")
    parser.add_argument("--query", default="Explain photosynthesis in one sentence.")
    parser.add_argument("--target", default="Plants use sunlight")
    parser.add_argument("--initial-control", default="! ! ! ! ! ! ! !")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--top-k", type=int, default=32)
    parser.add_argument("--candidates", type=int, default=64, help="GCG candidate budget")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="GCG evaluation microbatch / COLD parallel logit sequences",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--mode", choices=("suffix", "control", "paraphrase"), default="suffix")
    parser.add_argument("--right-context", default="")
    parser.add_argument("--include", action="append", default=[])
    parser.add_argument("--exclude", action="append", default=[])
    parser.add_argument("--embedding-temperature", type=float, default=0.001)
    parser.add_argument(
        "--output", type=Path, help="JSON result path; existing files are protected"
    )
    args = parser.parse_args()
    if args.output and args.output.exists():
        parser.error(f"Output already exists: {args.output}")
    if args.model == "tiny":
        import torch

        from textbook_toolkit.jailbreak.optimization.demo import tiny_backend

        torch.set_num_threads(1)
        backend = tiny_backend(chat=not args.plain)
        print("Using a random tiny model: this verifies computation, not response quality.")
    else:
        backend = CausalLM.from_pretrained(
            args.model,
            device=args.device,
            dtype=args.dtype,
            revision=args.revision,
            local_files_only=args.local_files_only,
            chat=False if args.plain else None,
        )
    options = dict(
        steps=args.steps, top_k=args.top_k, seed=args.seed, max_new_tokens=args.max_new_tokens
    )
    if args.method == "gcg":
        if args.mode != "suffix":
            parser.error("GCG supports suffix mode.")
        config = GCGConfig(
            **options, candidates=args.candidates, eval_batch_size=args.batch_size
        )
    else:
        config = COLDConfig(
            **options,
            batch_size=args.batch_size,
            mode=args.mode,
            include=tuple(args.include),
            exclude=tuple(args.exclude),
            embedding_temperature=args.embedding_temperature,
        )
    request = dict(initial_control=args.initial_control, right_context=args.right_context)
    if args.mode == "paraphrase":
        request.update(placement="replacement", initial_control=args.query)
    result = get(args.method, backend=backend, config=config).run(
        args.query, args.target, **request
    )
    print(result.to_json())
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        result.save(args.output)
        print(f"Saved: {args.output.resolve()}")


if __name__ == "__main__":
    main()
