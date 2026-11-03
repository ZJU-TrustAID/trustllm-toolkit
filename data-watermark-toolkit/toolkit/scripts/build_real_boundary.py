# -*- coding: utf-8 -*-
"""汇总边界实验（弱注入 / 负对照）为 demo 数据。

读取：``work/real_boundary/{weak,clean}/<method>/rho<..>/seed<..>/report.json``
输出：``webdemo/real_boundary.json``

- weak：ρ=0.1 + 弱参数（canary repeats=1、KGW δ=0.8、16 段水印文本、毒化率≈11%）
- clean：ρ=0（完全干净数据），用于检查误报
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "work" / "real_boundary"
OUT = ROOT / "webdemo" / "real_boundary.json"

METHODS = {
    "canary_memory": ("内容记忆型", "Meeus et al., Copyright Traps (ICML 2024)"),
    "version_pair": ("版本对照型", "Rastogi et al., STAMP (ICML 2025)"),
    "fictitious_knowledge": ("语义知识型", "Cui et al., Fictitious Knowledge (ACL 2025 Findings)"),
    "token_statistics": ("分布统计型", "Sander et al., Radioactive (NeurIPS 2024)"),
    "trigger_behavior": ("触发行为型", "Li et al., Double-I (arXiv 2402.14883)"),
}


def load_group(group: str, method: str) -> dict | None:
    base = SRC / group / method
    if not base.exists():
        return None
    points = []
    for rho_dir in sorted(base.iterdir()):
        if not rho_dir.is_dir() or not rho_dir.name.startswith("rho"):
            continue
        for seed_dir in sorted(rho_dir.iterdir()):
            report = seed_dir / "report.json"
            if not report.exists():
                continue
            data = json.loads(report.read_text(encoding="utf-8"))
            det = data.get("detection", {})
            points.append({
                "rho": data.get("fraction"),
                "seed": data.get("seed"),
                "p": det.get("p_value"),
                "decision": det.get("decision"),
                "score": det.get("score"),
                "n": det.get("n"),
            })
    if not points:
        return None
    ps = [p["p"] for p in points if p["p"] is not None]
    nls = [min(40.0, -math.log10(p)) for p in ps if p is not None and p > 0]
    return {
        "points": points,
        "mean_p": sum(ps) / len(ps) if ps else None,
        "mean_nl": sum(nls) / len(nls) if nls else None,
        "p_list": [float(p) for p in ps],
        "detections": sum(1 for p in points
                            if p["p"] is not None and p["p"] < 0.01),
        "n_seeds": len(points),
    }


def main() -> int:
    methods = {}
    for method, (name, ref) in METHODS.items():
        weak = load_group("weak", method)
        clean = load_group("clean", method)
        if weak or clean:
            methods[method] = {"name": name, "reference": ref,
                               "weak": weak, "clean": clean}
    if not methods:
        raise SystemExit(f"没有找到 {SRC} 下的边界实验报告；请先运行弱注入/负对照网格")
    first = next(iter(methods.values()))
    payload = {
        "meta": {
            "title": "数据集水印检测 · 边界实验",
            "model": "Qwen3.5-4B",
            "weak_fraction": 0.1,
            "clean_fraction": 0.0,
            "seeds": sorted({p["seed"] for m in methods.values()
                             for g in ("weak", "clean") if m.get(g)
                             for p in m[g]["points"]}),
            "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "note": "弱注入 ρ=0.1 与负对照 ρ=0 的对比；判定阈值 α=1%。",
        },
        "methods": methods,
    }
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                   encoding="utf-8")
    print(f"[out] {OUT}")
    for m, item in methods.items():
        w = item.get("weak") or {}
        c = item.get("clean") or {}
        print(f"  {m}: weak mean_p={w.get('mean_p')} "
              f"detect={w.get('detections')}/{w.get('n_seeds')} | "
              f"clean mean_p={c.get('mean_p')} "
              f"detect={c.get('detections')}/{c.get('n_seeds')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
