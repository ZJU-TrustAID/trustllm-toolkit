"""A fully local AutoDAN HGA run; downloads weights only on explicit loading."""

import argparse
from pathlib import Path

from textbook_toolkit.jailbreak import AutoDAN
from textbook_toolkit.jailbreak.optimization import AutoDANConfig, CausalLM


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M-Instruct")
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--population-size", type=int, default=8)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output and args.output.exists():
        parser.error(f"Output already exists: {args.output}")
    backend = CausalLM.from_pretrained(args.model, device=args.device)
    # Use a separate stronger model/client here when testing mutation quality.
    method = AutoDAN(
        backend.text_generator(max_new_tokens=64),
        "Use clear language in a classroom discussion. Topic: {query}",
    )
    result = method.run(
        "Explain photosynthesis.",
        "Plants use sunlight",
        backend=backend,
        config=AutoDANConfig(steps=args.steps, population_size=args.population_size),
    )
    print(result.to_json())
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        result.save(args.output)


if __name__ == "__main__":
    main()
