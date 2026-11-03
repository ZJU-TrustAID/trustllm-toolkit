"""Configuration inspection and TRL training CLI."""

from __future__ import annotations

import argparse
import json

from .config import load_experiment


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="alignment")
    parser.add_argument("command", choices=("resolve", "dry-run", "train", "evaluate"))
    parser.add_argument("--config", required=True)
    parser.add_argument("--model", help="Optional model override for evaluation smoke tests")
    parser.add_argument("--profile", help="Evaluation profile; defaults to evaluation.profile")
    parser.add_argument("--max-samples", type=int, help="Limit the number of prompts per suite")
    args = parser.parse_args(argv)
    spec = load_experiment(args.config)
    if args.command == "train":
        from .pipeline import AlignmentExperiment
        from .trl_runner import TRLRunner

        result = AlignmentExperiment(spec, runner=TRLRunner()).run()
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0
    if args.command == "evaluate":
        from .evaluation import load_evaluation_profile, resolve_evaluation_model, run_safety_evaluation

        profile = load_evaluation_profile(args.config, args.profile)
        model_name_or_path, model_kwargs = resolve_evaluation_model(spec, args.config, args.model)
        output_root = spec.output.get("root", "outputs")
        output_dir = f"{output_root}/{spec.name}/evaluation/{profile['name']}"
        result = run_safety_evaluation(
            model_name_or_path=model_name_or_path,
            profile=profile,
            output_dir=output_dir,
            max_samples=args.max_samples,
            model_kwargs=model_kwargs,
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0
    print(json.dumps(spec.raw, indent=2, ensure_ascii=False))
    if args.command == "dry-run":
        print(f"dry-run ok: {len(spec.stages)} stage(s)")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
