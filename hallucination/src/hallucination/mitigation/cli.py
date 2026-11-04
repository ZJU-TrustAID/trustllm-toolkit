from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

from hallucination._version import __version__
from hallucination.mitigation.api.mitigate import Mitigator
from hallucination.mitigation.core.errors import HallucinationMitigationError
from hallucination.mitigation.core.presets import list_presets

_METHOD_HELP = {
    "dola": "DoLa layer-contrast decoding (local white-box model)",
    "cad": "CAD context-aware decoding (local white-box model, requires context)",
    "caa": "CAA layer-wise activation steering (requires a fitted artifact)",
    "iti": "ITI per-head probe steering (optional extra, requires a fitted artifact)",
    "cove": "CoVe verification + revision (HF or API backend)",
}


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except (HallucinationMitigationError, ValueError, FileNotFoundError) as exc:
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hallucination-mitigation",
        description="Inference-time hallucination mitigation",
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="run one mitigation method on a real model")
    run.add_argument("--method", choices=sorted(_METHOD_HELP), default=None)
    run.add_argument("--preset")
    run.add_argument("--model", help="HF id, local model directory, or API model name")
    run.add_argument(
        "--backend", choices=["hf", "openai", "openai-compatible", "callable"]
    )
    run.add_argument("--prompt", required=True)
    run.add_argument("--context", help="context for CAD")
    run.add_argument("--context-file")
    run.add_argument("--steering", help="CAA artifact (.npz/.json)")
    run.add_argument("--probe", help="ITI artifact (.json)")
    run.add_argument("--layer", type=int)
    run.add_argument("--multiplier", type=float)
    run.add_argument("--alpha", type=float)
    run.add_argument("--max-questions", type=int, help="CoVe: maximum verification questions")
    run.add_argument("--max-answer-tokens", type=int, help="CoVe: per-question answer budget")
    run.add_argument("--max-new-tokens", type=int, default=128)
    run.add_argument("--do-sample", action="store_true")
    run.add_argument("--temperature", type=float, default=1.0)
    run.add_argument("--top-p", type=float, default=1.0)
    run.add_argument("--top-k", type=int, default=0)
    run.add_argument("--seed", type=int, default=0)
    run.add_argument("--device")
    run.add_argument("--dtype", default="auto")
    run.add_argument("--revision")
    run.add_argument("--local-files-only", action="store_true")
    run.add_argument("--api-base")
    run.add_argument("--api-key-env", default="OPENAI_API_KEY")
    run.add_argument("--timeout", type=float, default=60.0)
    run.add_argument("--max-retries", type=int, default=2)
    run.add_argument("--no-baseline", action="store_true")
    run.add_argument("--format", choices=["markdown", "json"], default="markdown")
    run.add_argument("--out")
    run.set_defaults(func=_run)

    fit_steering = subparsers.add_parser(
        "fit-steering", help="fit a CAA steering artifact"
    )
    _common_model_args(fit_steering)
    fit_steering.add_argument(
        "--data", required=True, help="JSONL with prompt/positive/negative"
    )
    fit_steering.add_argument("--layer", type=int, default=12)
    fit_steering.add_argument("--multiplier", type=float, default=1.0)
    fit_steering.add_argument("--behavior", default="hallucination")
    fit_steering.add_argument("--chat", action="store_true")
    fit_steering.add_argument("--pooling", choices=["last", "mean"], default="last")
    fit_steering.add_argument("--out", required=True)
    fit_steering.set_defaults(func=_fit_steering)

    fit_iti = subparsers.add_parser("fit-iti", help="fit an ITI probe artifact")
    _common_model_args(fit_iti)
    fit_iti.add_argument(
        "--data", required=True, help="JSONL with prompt/response/label"
    )
    fit_iti.add_argument("--top-k", type=int, default=48)
    fit_iti.add_argument("--alpha", type=float, default=15.0)
    fit_iti.add_argument("--chat", action="store_true")
    fit_iti.add_argument("--out", required=True)
    fit_iti.set_defaults(func=_fit_iti)

    benchmark = subparsers.add_parser(
        "benchmark", help="evaluate a method on a JSONL dataset or standard benchmark"
    )
    benchmark.add_argument(
        "--data", help="JSONL with prompt/context/reference(s)"
    )
    benchmark.add_argument(
        "--dataset",
        choices=["truthfulqa_gen", "nq_open", "nq_swap"],
        help="standard benchmark instead of --data (requires .[datasets] for the first two)",
    )
    benchmark.add_argument(
        "--dataset-path", help="local file for datasets that are not on the Hub (nq_swap test.json)"
    )
    benchmark.add_argument(
        "--seed", type=int, default=0, help="seed for deterministic dataset subsampling"
    )
    benchmark.add_argument("--preset")
    benchmark.add_argument("--method", choices=sorted(_METHOD_HELP), default=None)
    benchmark.add_argument("--model")
    benchmark.add_argument(
        "--backend", choices=["hf", "openai", "openai-compatible", "callable"]
    )
    benchmark.add_argument("--steering")
    benchmark.add_argument("--probe")
    benchmark.add_argument("--device")
    benchmark.add_argument("--dtype", default="auto")
    benchmark.add_argument("--revision")
    benchmark.add_argument("--local-files-only", action="store_true")
    benchmark.add_argument("--api-base")
    benchmark.add_argument("--api-key-env", default="OPENAI_API_KEY")
    benchmark.add_argument("--limit", type=int)
    benchmark.add_argument(
        "--metric", choices=["exact", "contains"], default="contains"
    )
    benchmark.add_argument("--max-new-tokens", type=int, default=128)
    benchmark.add_argument("--max-questions", type=int, help="CoVe: maximum verification questions")
    benchmark.add_argument("--max-answer-tokens", type=int, help="CoVe: per-question answer budget")
    benchmark.add_argument("--no-baseline", action="store_true")
    benchmark.add_argument("--format", choices=["markdown", "json"], default="markdown")
    benchmark.add_argument("--out")
    benchmark.set_defaults(func=_benchmark)

    methods = subparsers.add_parser(
        "list-methods", help="list methods and availability"
    )
    methods.set_defaults(func=_list_methods)

    presets = subparsers.add_parser("list-presets", help="list built-in presets")
    presets.set_defaults(func=_list_presets)

    doctor = subparsers.add_parser(
        "doctor", help="inspect environment and optional dependencies"
    )
    doctor.set_defaults(func=_doctor)

    return parser


def _common_model_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", required=True, help="HF id or local model directory")
    parser.add_argument("--device")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--revision")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--load-in-4bit", action="store_true")


def _run(args: argparse.Namespace) -> int:
    context = args.context
    if args.context_file:
        context = Path(args.context_file).read_text(encoding="utf-8").strip()
    mitigator = Mitigator(
        preset=args.preset,
        method=args.method,
        model=args.model,
        backend=args.backend,
        device=args.device,
        dtype=args.dtype,
        revision=args.revision,
        local_files_only=args.local_files_only,
        api_base=args.api_base,
        api_key_env=args.api_key_env,
        timeout=args.timeout,
        max_retries=args.max_retries,
        steering=args.steering,
        probe=args.probe,
        layer=args.layer,
        multiplier=args.multiplier,
        alpha=args.alpha,
        seed=args.seed,
        **(
            {"max_questions": args.max_questions}
            if args.max_questions is not None
            else {}
        ),
        **(
            {"max_answer_tokens": args.max_answer_tokens}
            if args.max_answer_tokens is not None
            else {}
        ),
    )
    try:
        report = mitigator.mitigate(
            args.prompt,
            context=context,
            max_new_tokens=args.max_new_tokens,
            do_sample=args.do_sample,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            baseline=not args.no_baseline,
        )
    finally:
        mitigator.close()
    if args.format == "json":
        payload = report.to_json(args.out)
        if args.out is None:
            print(payload)
    else:
        text = report.to_markdown()
        if args.out:
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.out).write_text(text, encoding="utf-8")
        else:
            print(text)
    return 0


def _fit_steering(args: argparse.Namespace) -> int:
    from hallucination.mitigation.methods.caa import CAAConfig, fit_caa

    engine = _load_engine(args)
    pairs = _read_jsonl(args.data)
    config = CAAConfig(
        layer=args.layer,
        multiplier=args.multiplier,
        chat=args.chat,
        pooling=args.pooling,
    )
    artifact = fit_caa(engine, pairs, config, behavior=args.behavior)
    artifact.save(args.out)
    print(
        f"saved CAA artifact: {args.out} (layer={artifact.layer}, norm={artifact.norm:.4f}, checksum={artifact.checksum[:12]})"
    )
    return 0


def _fit_iti(args: argparse.Namespace) -> int:
    from hallucination.mitigation.methods.iti import ITIConfig, fit_iti

    engine = _load_engine(args)
    samples = _read_jsonl(args.data)
    config = ITIConfig(top_k=args.top_k, alpha=args.alpha, chat=args.chat)
    artifact = fit_iti(engine, samples, config)
    artifact.save(args.out)
    best = max((head.accuracy for head in artifact.heads), default=0.0)
    print(
        f"saved ITI artifact: {args.out} (heads={len(artifact.heads)}, best_accuracy={best:.3f})"
    )
    return 0


def _benchmark(args: argparse.Namespace) -> int:
    from hallucination.mitigation.evaluation import evaluate_records

    mitigator = Mitigator(
        preset=args.preset,
        method=args.method,
        model=args.model,
        backend=args.backend,
        device=args.device,
        dtype=args.dtype,
        revision=args.revision,
        local_files_only=args.local_files_only,
        api_base=args.api_base,
        api_key_env=args.api_key_env,
        steering=args.steering,
        probe=args.probe,
        **(
            {"max_questions": args.max_questions}
            if args.max_questions is not None
            else {}
        ),
        **(
            {"max_answer_tokens": args.max_answer_tokens}
            if args.max_answer_tokens is not None
            else {}
        ),
    )
    try:
        if args.dataset is not None:
            from hallucination.mitigation.evaluation.datasets import (
                load_dataset_records,
            )

            records = load_dataset_records(
                args.dataset, path=args.dataset_path, limit=args.limit, seed=args.seed
            )
        elif args.data is not None:
            records = _read_jsonl(args.data)
        else:
            raise SystemExit("benchmark requires --data or --dataset")
        result = evaluate_records(
            mitigator,
            records,
            max_new_tokens=args.max_new_tokens,
            limit=args.limit,
            metric=args.metric,
            baseline=not args.no_baseline,
        )
    finally:
        mitigator.close()
    if args.format == "json":
        payload = result.to_json(args.out)
        if args.out is None:
            print(payload)
    else:
        text = result.to_markdown()
        if args.out:
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.out).write_text(text, encoding="utf-8")
        else:
            print(text)
    return 0


def _load_engine(args: argparse.Namespace) -> Any:
    from hallucination.mitigation.core.model_config import ModelConfig
    from hallucination.mitigation.engine.hf import HFEngine

    config = ModelConfig(
        model=args.model,
        backend="hf",
        device=args.device,
        dtype=args.dtype,
        revision=args.revision,
        local_files_only=args.local_files_only,
        trust_remote_code=args.trust_remote_code,
        load_in_4bit=args.load_in_4bit,
    )
    return HFEngine.from_pretrained(config)


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    target = Path(path)
    if not target.exists():
        raise FileNotFoundError(f"data file not found: {target}")
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        target.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON at {target}:{line_number}: {exc}") from exc
        if not isinstance(payload, dict):
            raise TypeError(f"expected a JSON object at {target}:{line_number}")
        records.append(payload)
    if not records:
        raise ValueError(f"no records found in {target}")
    return records


def _list_methods(args: argparse.Namespace) -> int:
    torch_available = importlib.util.find_spec("torch") is not None
    transformers_available = importlib.util.find_spec("transformers") is not None
    sklearn_available = importlib.util.find_spec("sklearn") is not None
    openai_available = importlib.util.find_spec("openai") is not None
    print("method                availability")
    for method, description in _METHOD_HELP.items():
        if method == "cove":
            available = openai_available or (torch_available and transformers_available)
        elif method == "iti":
            available = torch_available and transformers_available and sklearn_available
        else:
            available = torch_available and transformers_available
        print(f"{method:<20} {'yes' if available else 'no':<12} {description}")
    return 0


def _list_presets(args: argparse.Namespace) -> int:
    presets = list_presets()
    if not presets:
        print("(no presets installed)")
        return 0
    for name in presets:
        print(name)
    return 0


def _doctor(args: argparse.Namespace) -> int:
    print(f"hallucination-mitigation {__version__}")
    print(f"python {sys.version.split()[0]}")
    for name in ("torch", "transformers", "sklearn", "openai", "datasets"):
        spec = importlib.util.find_spec(name)
        print(f"{name:<14} {'installed' if spec else 'missing'}")
    try:
        import torch

        print(f"cuda available  {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"cuda device     {torch.cuda.get_device_name(0)}")
    except ImportError:
        print("cuda available  unknown (torch missing)")
    token = bool(
        __import__("os").environ.get("HF_TOKEN")
        or __import__("os").environ.get("HUGGING_FACE_HUB_TOKEN")
    )
    print(f"HF token        {'present' if token else 'absent'}")
    print("methods         " + ", ".join(method for method in _METHOD_HELP))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
