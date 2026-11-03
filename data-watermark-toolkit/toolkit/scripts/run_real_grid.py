# -*- coding: utf-8 -*-
"""跑真实实验参数网格（给交互 demo 提供实测数据）。

只做“一个方法 × 一个 ρ × 一个 seed”的真实小模型实验：
embed → SmolLM2-135M SFT → 论文检测统计量。结果写入
``work/real_grid/<method>/rho<..>/seed<..>/report.json``，可断点续跑。

用法（在 toolkit/ 下）::

    export HF_HOME=$PWD/work/hf_cache
    export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
    export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=
    nice -n 19 taskset -c 0-95 python3 scripts/run_real_grid.py \
        --config configs/real_smollm.json \
        --fractions 0.25,0.5,1.0 --seeds 42,43 \
        --out work/real_grid

资源上限：单进程 8 线程（nproc=192 时约 4% CPU；远低于 50%），
不启用 CUDA（本机 torch 为 CPU wheel，且显式 CUDA_VISIBLE_DEVICES=）。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/real_smollm.json")
    ap.add_argument("--fractions", default="0.25,0.5,1.0")
    ap.add_argument("--seeds", default="42,43")
    ap.add_argument("--methods", default="")
    ap.add_argument("--out", default="work/real_grid")
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    os.environ.setdefault("OMP_NUM_THREADS", str(args.threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(args.threads))

    from datamark.real.experiment import RealExperiment, load_real_config
    from datamark.real.methods import REAL_METHODS

    cfg = load_real_config(args.config)
    fractions = [float(x) for x in args.fractions.split(",") if x.strip()]
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    methods = ([x for x in args.methods.split(",") if x.strip()]
               if args.methods else list(REAL_METHODS))
    out_root = Path(args.out)
    jobs = [(m, f, s) for m in methods for f in fractions for s in seeds]
    print(f"[grid] {len(jobs)} 个实验；方法={methods}；ρ={fractions}；seeds={seeds}",
          flush=True)
    done = 0
    skipped = 0
    failed = 0
    for method, fraction, seed in jobs:
        out = out_root / method / f"rho{fraction:g}" / f"seed{seed}"
        report = out / "report.json"
        if report.exists():
            skipped += 1
            print(f"[skip] {method} ρ={fraction} seed={seed}", flush=True)
            continue
        t0 = time.time()
        try:
            exp = RealExperiment(cfg, method, fraction=fraction, seed=seed)
            res = exp.run(out_dir=out)
            done += 1
            print(f"[ok] {method} ρ={fraction} seed={seed} "
                  f"p={res.detection['p_value']:.3g} "
                  f"decision={res.detection['decision']} "
                  f"{round(time.time() - t0, 1)}s", flush=True)
        except Exception as exc:  # 单个实验失败不终止网格
            failed += 1
            print(f"[fail] {method} ρ={fraction} seed={seed}: {exc}", flush=True)
            traceback.print_exc()
    print(f"[grid] done={done} skipped={skipped} failed={failed}", flush=True)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
