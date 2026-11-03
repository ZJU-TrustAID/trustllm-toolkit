# -*- coding: utf-8 -*-
"""构建演示页的三组实验数据：强注入 / 弱注入 / 负对照。

输出 ``webdemo/real_demo.json``：每种方法在每个实验条件下的 p 值、效应量、
参数、逐样本证据（seed=42 的那次实验）以及两个 seed 的 p 值列表。

数据来源：
- strong：work/real_evidence_qwen50（ρ=0.5，seed=42）+ work/real_grid_qwen 的 seed 列表
- weak：  work/real_boundary/weak/<method>/rho0.1/
- clean： work/real_boundary/clean/<method>/rho0/
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from build_real_evidence import META, load_method  # noqa: E402

OUT = ROOT / "webdemo" / "real_demo.json"

CONDITIONS = [
    {"id": "strong", "label": "强注入", "fraction": 0.5,
     "note": "ρ=0.5；trap 重复 5 次、整段改写、96 段水印文本、50% 毒化"},
    {"id": "weak", "label": "弱注入", "fraction": 0.1,
     "note": "ρ=0.1；trap 只重复 1 次、KGW δ=0.8、仅 16 段水印文本、毒化率约 11%"},
    {"id": "clean", "label": "负对照", "fraction": 0.0,
     "note": "ρ=0；训练数据完全干净（内容记忆/版本对照仍生成但不注入，用于零分布）"},
]


def seed_points(base: Path) -> list[dict]:
    """读取同一 ρ 目录下各 seed 的 p 值与判定。"""
    points = []
    if not base.exists():
        return points
    for seed_dir in sorted(base.iterdir()):
        report = seed_dir / "report.json"
        if not report.exists():
            continue
        data = json.loads(report.read_text(encoding="utf-8"))
        det = data.get("detection", {})
        points.append({
            "seed": data.get("seed"),
            "p": det.get("p_value"),
            "decision": det.get("decision"),
        })
    return points


def main() -> int:
    sources = {
        "strong": (ROOT / "work" / "real_evidence_qwen50",
                   ROOT / "work" / "real_grid_qwen"),
        "weak": (ROOT / "work" / "real_boundary" / "weak",
                 ROOT / "work" / "real_boundary" / "weak"),
        "clean": (ROOT / "work" / "real_boundary" / "clean",
                  ROOT / "work" / "real_boundary" / "clean"),
    }
    methods = {}
    for method in META:
        item = {"name": META[method]["name"], "access": META[method]["access"],
                "reference": META[method]["reference"],
                "panelTitle": META[method]["panelTitle"],
                "panelCaption": META[method]["panelCaption"],
                "caption": META[method]["caption"],
                "conditions": {}}
        for cond in CONDITIONS:
            cid = cond["id"]
            evidence_dir, seed_dir = sources[cid]
            if cid == "strong":
                base_dir = evidence_dir / method
                seeds = seed_points(seed_dir / method / "rho0.5")
            elif cid == "weak":
                base_dir = evidence_dir / method / "rho0.1" / "seed42"
                seeds = seed_points(seed_dir / method / "rho0.1")
            else:
                base_dir = evidence_dir / method / "rho0" / "seed42"
                seeds = seed_points(seed_dir / method / "rho0")
            item_data = load_method(method, base_dir=base_dir)
            if item_data is None:
                continue
            item_data["fraction"] = cond["fraction"]
            item_data["seeds"] = seeds
            item_data["null_mode"] = bool(
                cid == "clean" and method in ("canary_memory", "version_pair"))
            item["conditions"][cid] = item_data
        if item["conditions"]:
            methods[method] = item
    if not methods:
        raise SystemExit("没有找到任何条件的实验数据")
    payload = {
        "meta": {
            "title": "数据集水印检测 · 真实演示数据",
            "model": "Qwen3.5-4B",
            "alpha": 0.01,
            "dataset": "96 条合成凤尾鱼观测文档 + 50 条公共文本",
            "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "conditions": CONDITIONS,
            "note": "选择方法与实验条件（强注入 / 弱注入 / 负对照），查看对应条件下的证据与判定。",
        },
        "methods": methods,
    }
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                   encoding="utf-8")
    print(f"[out] {OUT}")
    for m, item in methods.items():
        line = [m]
        for cid in item["conditions"]:
            c = item["conditions"][cid]
            line.append(f"{cid}: p={c['p']:.3g}")
        print("  " + " | ".join(line))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
