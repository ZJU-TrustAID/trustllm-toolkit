from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from hallucination.detection.api.detect import Detector
from hallucination.detection.calibration import CalibrationBundle
from hallucination.detection.datasets import list_datasets, load_dataset
from hallucination.detection.evaluation import BenchmarkRunner

_METHODS = [
    "exact_match_consistency",
    "response_cosine_consistency",
    "response_semantic_entropy",
    "selfcheck_ngram",
    "selfcheck_nli",
    "selfcheck_bertscore",
    "mtp",
    "lntp",
    "token_entropy",
    "eigenscore",
    "effective_rank",
    "factscore",
    "p_true",
    "cove_detector",
    "llm_judge",
    "llm_panel",
]


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except Exception as exc:  # CLI should return an actionable message, not a traceback.
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="hallucination-detection", description="Hallucination detection toolkit")
    subparsers = parser.add_subparsers(dest="command", required=True)

    detect = subparsers.add_parser("detect", help="run detection for one prompt, or for a response plus samples")
    detect.add_argument("--preset", default="offline_toy", choices=["offline_toy", "openai_blackbox", "hf_whitebox"])
    detect.add_argument("--prompt")
    detect.add_argument("--response", help="response to score; requires --samples unless --prompt is given")
    detect.add_argument("--samples", help="JSON list of sampled answers, or @path/to/samples.json, for response-only runs")
    detect.add_argument("--num-samples", type=int, default=5)
    detect.add_argument("--threshold", type=float, default=0.5)
    detect.add_argument("--calibrator", help="calibration bundle/registry JSON to apply")
    detect.add_argument("--model")
    detect.add_argument("--device")
    detect.add_argument("--out")
    detect.add_argument("--format", choices=["markdown", "json"], default="markdown")
    detect.set_defaults(func=_detect)

    benchmark = subparsers.add_parser("benchmark", help="run a benchmark on a registered dataset")
    benchmark.add_argument("--dataset", default="textbook_toy")
    benchmark.add_argument("--granularity", choices=["sentence", "response"], default="sentence")
    benchmark.add_argument("--preset", default="offline_toy")
    benchmark.add_argument("--calibrator", help="calibration bundle/registry JSON with per-method entries")
    benchmark.add_argument("--out-prefix")
    benchmark.set_defaults(func=_benchmark)

    calibrate = subparsers.add_parser("calibrate", help="fit a threshold or probability calibration bundle")
    calibrate.add_argument("--scores", required=True, help="JSONL records with score and label (label=True means hallucination)")
    calibrate.add_argument("--method", choices=["threshold", "platt", "isotonic", "temperature"], default="threshold")
    calibrate.add_argument("--target", choices=["hallucination", "correct"], default="hallucination")
    calibrate.add_argument("--score-name", default="*", help="method name this bundle applies to; default '*' applies to all")
    calibrate.add_argument("--granularity", choices=["response", "sentence", "claim", "token"])
    calibrate.add_argument("--threshold-objective", choices=["f1", "fbeta", "balanced_accuracy", "cost"], default="f1")
    calibrate.add_argument("--out", required=True)
    calibrate.set_defaults(func=_calibrate)

    methods = subparsers.add_parser("list-methods", help="list implemented methods")
    methods.set_defaults(func=_list_methods)

    datasets = subparsers.add_parser("list-datasets", help="list registered datasets")
    datasets.set_defaults(func=_list_datasets)

    return parser


def _detect(args: argparse.Namespace) -> int:
    kwargs: dict[str, Any] = {
        "preset": args.preset,
        "num_samples": args.num_samples,
        "threshold": args.threshold,
    }
    if args.preset == "hf_whitebox":
        kwargs["model"] = args.model
        kwargs["device"] = args.device
    if args.calibrator:
        kwargs["calibration"] = args.calibrator
    detector = Detector(**kwargs)
    samples = _read_samples(args.samples)
    report = detector.detect(args.prompt, response=args.response, samples=samples)
    if args.format == "json":
        payload = report.to_json(args.out)
        if args.out is None:
            print(payload)
    else:
        text = report.to_markdown()
        if args.out:
            Path(args.out).write_text(text, encoding="utf-8")
        else:
            print(text)
    return 0


def _benchmark(args: argparse.Namespace) -> int:
    detector = Detector(preset=args.preset)
    dataset = load_dataset(args.dataset)
    runner = BenchmarkRunner(
        detector.as_pipeline(),
        granularity=args.granularity,
        calibration=args.calibrator,
    )
    report = runner.evaluate(dataset)
    prefix = Path(args.out_prefix) if args.out_prefix else None
    if prefix is not None:
        report.write_json(prefix.with_suffix(".json"))
        report.write_csv(prefix.with_suffix(".csv"))
    print(json.dumps(report.metadata, ensure_ascii=False, indent=2))
    for method, metrics in report.metrics.items():
        print(f"{method}: risk metrics={ {key: round(value, 4) if isinstance(value, float) else value for key, value in metrics.items()} }")
    return 0


def _calibrate(args: argparse.Namespace) -> int:
    scores, labels = _read_scores(Path(args.scores))
    if args.method == "threshold":
        bundle = CalibrationBundle.fit(
            scores,
            labels,
            score_name=args.score_name,
            granularity=args.granularity,
            score_kind="risk",
            threshold_metric=args.threshold_objective,
            target=args.target,
            split="validation",
        )
    else:
        bundle = CalibrationBundle.fit(
            scores,
            labels,
            score_name=args.score_name,
            granularity=args.granularity,
            score_kind="risk",
            probability_method=args.method,
            target=args.target,
            split="validation",
        )
    bundle.save(args.out)
    print(json.dumps(bundle.to_dict(), ensure_ascii=False, indent=2))
    return 0


def _read_samples(value: str | None) -> list[str] | None:
    """Parse ``--samples`` from inline JSON or an ``@path`` file.

    The file form avoids shell quoting problems with inline JSON on Windows.
    """

    if value is None:
        return None
    if value.startswith("@"):
        payload = json.loads(Path(value[1:]).read_text(encoding="utf-8-sig"))
    else:
        payload = json.loads(value)
    if not isinstance(payload, list) or not all(isinstance(item, str) for item in payload):
        raise ValueError("--samples must be a JSON list of strings")
    return payload


def _read_scores(path: Path) -> tuple[list[float], list[bool]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    scores: list[float] = []
    labels: list[bool] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        if "score" not in row or "label" not in row:
            raise ValueError(f"line {line_number}: records require 'score' and 'label'")
        scores.append(float(row["score"]))
        labels.append(_parse_label(row["label"], line_number=line_number))
    if not scores:
        raise ValueError("scores file contains no records")
    return scores, labels


def _parse_label(value: Any, *, line_number: int) -> bool:
    """Parse labels without Python's truthiness trap (``bool("false") is True``)."""

    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in {0, 1}:
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "hallucinated"}:
            return True
        if normalized in {"false", "0", "no", "supported", "factual"}:
            return False
    raise ValueError(
        f"line {line_number}: label must be a boolean or one of 0/1, 'true'/'false', "
        f"'supported'/'hallucinated'; got {value!r}"
    )


def _list_methods(args: argparse.Namespace) -> int:
    for method in _METHODS:
        print(method)
    return 0


def _list_datasets(args: argparse.Namespace) -> int:
    for info in list_datasets():
        print(f"{info.name}\t{'remote' if info.remote else 'offline'}\t{info.description}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
