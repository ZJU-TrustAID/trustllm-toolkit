"""Paired validation study: DoLa and CAD vs baseline on standard benchmarks.

Model: Qwen/Qwen2.5-0.5B-Instruct, CPU, greedy decoding.
Datasets: deterministic 40-example subsets (seed=0) built by the dataset
adapters; see docs/mitigation-validation.md for methodology.
"""

from __future__ import annotations

import json
import pathlib
import time

from hallucination.mitigation import Mitigator
from hallucination.mitigation.evaluation import evaluate_records

STUDIES = [
    ("dola", "truthfulqa_gen", 24),
    ("dola", "nq_open", 16),
    ("cad", "nq_swap", 16),
]

REPO = pathlib.Path(__file__).resolve().parents[2]
OUT = REPO / "runs" / "validation"


def main() -> None:
    for method, dataset, tokens in STUDIES:
        records = [
            json.loads(line)
            for line in (OUT / f"{dataset}_n40.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        kwargs: dict[str, object] = {
            "method": method,
            "model": "Qwen/Qwen2.5-0.5B-Instruct",
            "device": "cpu",
            "dtype": "float32",
            "local_files_only": True,
        }
        if method == "cad":
            kwargs.update(alpha=1.0, context_template="plain")
        mitigator = Mitigator(**kwargs)  # type: ignore[arg-type]
        started = time.perf_counter()
        result = evaluate_records(mitigator, records, max_new_tokens=tokens, metric="contains")
        elapsed = time.perf_counter() - started
        mitigator.close()
        stem = f"{method}_{dataset}_qwen05b"
        result.to_json(OUT / f"{stem}.json")
        (OUT / f"{stem}.md").write_text(result.to_markdown(), encoding="utf-8")
        print(
            f"[{stem}] {elapsed:.0f}s | baseline {result.baseline_score:.3f} "
            f"mitigated {result.mitigated_score:.3f} | w/t/l "
            f"{result.win}/{result.tie}/{result.loss} | p={result.significance['p_value']}",
            flush=True,
        )


if __name__ == "__main__":
    main()
