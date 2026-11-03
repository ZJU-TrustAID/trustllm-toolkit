# -*- coding: utf-8 -*-
"""把真实实验网格汇总成交互 demo 的数据文件。

输入：``work/real_grid/<method>/rho<..>/seed<..>/report.json``
输出：``webdemo/real_results.json``（demo 页面直接 fetch）
"""

from __future__ import annotations

import json
import os
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GRID = Path(os.environ.get("REAL_GRID_DIR", ROOT / "work" / "real_grid"))
OUT = ROOT / "webdemo" / "real_results.json"

METHODS = {
    "canary_memory": {
        "name": "内容记忆型",
        "family": "内容记忆型",
        "access": "白盒",
        "reference": "Meeus et al., Copyright Traps for LLMs (ICML 2024)",
        "score_label": "成员 trap − 非成员 trap 的平均 loss 差",
        "score_hint": "越大表示成员 trap 的 loss 越低（证据越强）",
        "detail_rows": [
            ["auc", "AUC（成员/非成员 trap 的 loss 排序）"],
            ["loss_member_mean", "成员 trap 平均 loss"],
            ["loss_control_mean", "非成员 trap 平均 loss"],
            ["t", "Welch t"],
        ],
    },
    "version_pair": {
        "name": "版本对照型",
        "family": "版本对照型",
        "access": "白盒",
        "reference": "Rastogi et al., STAMP Your Content (ICML 2025)",
        "score_label": "PPL(public) − PPL(private) 均值",
        "score_hint": "越负表示公开版 PPL 越低（证据越强）",
        "detail_rows": [
            ["ppl_public_mean", "公开版平均 PPL"],
            ["ppl_private_mean", "私有版平均 PPL"],
            ["mean_diff", "平均 PPL 差"],
            ["t", "配对 t"],
        ],
    },
    "fictitious_knowledge": {
        "name": "语义知识型",
        "family": "语义知识型",
        "access": "黑盒",
        "reference": "Cui et al., Fictitious Knowledge (ACL 2025 Findings)",
        "score_label": "事实补全命中率（正确属性 vs 控制属性）",
        "score_hint": "随机命中率为 0.25；越高表示虚构事实被记住得越多",
        "detail_rows": [
            ["qa_accuracy", "命中率"],
            ["mean_preference", "正确属性相对控制属性的平均对数似然差"],
            ["t", "配对 t"],
        ],
    },
    "token_statistics": {
        "name": "分布统计型",
        "family": "分布统计型",
        "access": "灰盒",
        "reference": "Sander et al., Watermarking Makes LMs Radioactive (NeurIPS 2024)",
        "score_label": "生成文本 green 比例差（真实密钥 − 控制密钥）",
        "score_hint": "越大表示可疑模型生成文本越偏向训练时水印的 green token",
        "detail_rows": [
            ["green_fraction", "真实密钥 green 比例"],
            ["control_fraction", "控制密钥 green 比例"],
            ["n_prompts", "检测提示数"],
        ],
    },
    "trigger_behavior": {
        "name": "触发行为型",
        "family": "触发行为型",
        "access": "黑盒",
        "reference": "Li et al., Double-I Watermark (arXiv 2402.14883, 2024)",
        "score_label": "触发组与参考组的操纵输出率之差",
        "score_hint": "越大表示后门在触发输入下越明显",
        "detail_rows": [
            ["trigger_manipulated", "触发组被操纵数"],
            ["trigger_n", "触发组总数"],
            ["reference_manipulated", "参考组被操纵数"],
            ["reference_n", "参考组总数"],
            ["odds_ratio", "Odds ratio"],
        ],
    },
}


def _load_reports() -> dict:
    grouped: dict[str, list[dict]] = {m: [] for m in METHODS}
    if not GRID.exists():
        return grouped
    for report_path in sorted(GRID.glob("*/*/*/report.json")):
        try:
            data = json.loads(report_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        method = data.get("method")
        if method not in METHODS:
            continue
        detection = data.get("detection", {})
        detail = detection.get("detail", {}) or {}
        point = {
            "rho": data.get("fraction"),
            "seed": data.get("seed"),
            "n": detection.get("n", 0),
            "p_value": detection.get("p_value"),
            "score": detection.get("score"),
            "decision_alpha": detection.get("decision"),
            "train": data.get("train", {}),
            "detail": detail,
            "elapsed_sec": data.get("elapsed_sec"),
        }
        grouped[method].append(point)
    for method in grouped:
        grouped[method].sort(key=lambda x: (x["rho"] if x["rho"] is not None else -1,
                                            x["seed"] if x["seed"] is not None else -1))
    return grouped


def main() -> int:
    grouped = _load_reports()
    all_points = [pt for pts in grouped.values() for pt in pts]
    seeds = sorted({pt["seed"] for pt in all_points if pt.get("seed") is not None})
    fractions = sorted({pt["rho"] for pt in all_points if pt.get("rho") is not None})
    payload = {
        "meta": {
            "title": "数据集水印检测 · 真实实验结果",
            "seeds": seeds,
            "fractions": fractions,
            "model": (os.environ.get("REAL_GRID_MODEL") or "Qwen3.5-4B"),
            "dataset": "96 条合成凤尾鱼观测文档 + 50 条公共文本",
            "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "alpha_options": [0.01],
            "note": "每个点是一次独立实验；判定阈值 α=1%。",
        },
        "methods": {
            m: {**info, "points": grouped[m]}
            for m, info in METHODS.items()
            if grouped[m]
        },
    }
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                   encoding="utf-8")
    total = sum(len(v) for v in grouped.values())
    print(f"[out] {OUT}（{len(payload['methods'])} 个方法，{total} 个实验点）")
    for m, pts in grouped.items():
        rhos = sorted({p["rho"] for p in pts})
        print(f"  {m}: {len(pts)} 点，ρ={rhos}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
