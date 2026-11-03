# -*- coding: utf-8 -*-
"""真实实现命令行：

    python3 -m datamark.real.cli run --method canary_memory \
        --config configs/real_smollm.json --fraction 0.5 --out work/real/canary

    python3 -m datamark.real.cli all --config configs/real_smollm.json \
        --fraction 0.5 --out-dir work/real
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .experiment import RealExperiment, load_real_config
from .methods import REAL_METHODS


def _run_one(args, method: str, out: Path | None) -> dict:
    cfg = load_real_config(args.config)
    exp = RealExperiment(cfg, method, fraction=args.fraction, seed=args.seed)
    result = exp.run(out_dir=out, save_model=args.save_model)
    payload = {
        "method": result.method,
        "family": exp.method.family,
        "reference": exp.method.reference,
        "fraction": result.fraction,
        "n_train_texts": result.n_train,
        "train": result.train_stats,
        "detection": result.detection,
        "elapsed_sec": result.elapsed_sec,
        "out": str(out) if out else None,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)
    return payload


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="datamark.real")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("run", "all"):
        p = sub.add_parser(name)
        p.add_argument("--config", default="configs/real_smollm.json")
        p.add_argument("--fraction", type=float, default=None)
        p.add_argument("--seed", type=int, default=None)
        p.add_argument("--save-model", action="store_true")
        if name == "run":
            p.add_argument("--method", required=True, choices=sorted(REAL_METHODS))
            p.add_argument("--out", default=None)
        else:
            p.add_argument("--out-dir", default="work/real")
            p.add_argument("--methods", default=None,
                           help="逗号分隔；默认全部五个训练时方法")
    args = ap.parse_args(argv)
    if args.cmd == "run":
        out = Path(args.out) if args.out else None
        _run_one(args, args.method, out)
        return 0
    methods = (args.methods.split(",") if args.methods
               else list(REAL_METHODS))
    out_dir = Path(args.out_dir)
    results = []
    for method in methods:
        results.append(_run_one(args, method, out_dir / method))
    print(json.dumps({"methods": methods,
                      "results": [{"method": r["method"],
                                   "p": r["detection"]["p_value"],
                                   "score": r["detection"]["score"],
                                   "decision": r["detection"]["decision"]}
                                  for r in results]},
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
