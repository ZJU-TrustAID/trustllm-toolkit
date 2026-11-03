# -*- coding: utf-8 -*-
"""在不重跑模型的前提下，用“全局去重”统计量重算分布统计型的检测结果。

背景：可疑模型的 96 段生成文本高度模板化，逐文档 (k+1)-tuple 去重会把相同的
n-gram 重复计入，高估有效样本量、使控制密钥的 green 计数过散，p 值失去分辨力。
本脚本读取已有 ``report.json`` 里保存的生成文本，重新按**跨文档全局去重**统计
green token，并用 499 个控制密钥做置换检验，只更新检测字段，不改变模型输出。
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DEFAULT_MODEL = "/data3/renxiao/qwen35-steer/models/Qwen3.5-4B"


def find_reports(root: Path) -> list[Path]:
    if root.is_file():
        return [root]
    return sorted(root.rglob("report.json"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--roots", nargs="+", required=True,
                    help="报告文件或包含 report.json 的目录")
    ap.add_argument("--controls", type=int, default=499)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    args = ap.parse_args()

    from transformers import AutoTokenizer

    from datamark.real.kgw import global_green_counts

    tok = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    updated = 0
    for root in args.roots:
        for report_path in find_reports(Path(root)):
            if report_path.parent.name != "token_statistics" and \
                    "token_statistics" not in str(report_path):
                continue
            manifest_path = report_path.parent / "manifest.json"
            if not manifest_path.exists():
                continue
            report = json.loads(report_path.read_text(encoding="utf-8"))
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            det = report.get("detection") or {}
            detail = det.get("detail") or {}
            evidence = detail.get("evidence") or {}
            items = evidence.get("items") or []
            if not items:
                print(f"[skip] {report_path}（没有生成文本证据）")
                continue
            key = int(manifest.get("key", 0))
            gamma = float(manifest.get("gamma", 0.25))
            k = int(manifest.get("window_k", 2))
            batches = [tok(it["text"], add_special_tokens=False)["input_ids"]
                       for it in items]
            true_green, n, flags = None, 0, []
            out = global_green_counts(batches, key, gamma, k)
            true_green, n, flags = out["green"], out["n"], out["flags"]
            if n == 0:
                print(f"[skip] {report_path}（没有可评分 token）")
                continue
            ctrl = [global_green_counts(batches, key + 1000003 * (j + 1),
                                        gamma, k)["green"]
                    for j in range(args.controls)]
            ctrl_rate = sum(ctrl) / len(ctrl) / n
            true_rate = true_green / n
            exceed = sum(1 for c in ctrl if c >= true_green)
            p = (1.0 + exceed) / (1.0 + len(ctrl))
            det["p_value"] = float(p)
            det["n"] = int(n)
            det["score"] = float(true_rate - ctrl_rate)
            det["decision"] = "watermarked" if p < 0.05 else "clean"
            detail.update({
                "green": int(true_green), "scored_tokens": int(n),
                "green_fraction": round(true_rate, 6),
                "control_fraction": round(ctrl_rate, 6),
                "n_control_keys": len(ctrl),
                "statistic": "global (k+1)-gram dedup + key permutation",
            })
            evidence.update({
                "flags": flags,
                "true_rate": round(true_rate, 6),
                "control_rate": round(ctrl_rate, 6),
            })
            det["detail"] = detail
            report["detection"] = det
            (report_path.parent / "report.json").write_text(
                json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8")
            updated += 1
            print(f"[ok] {report_path.parent} p={p:.4g} "
                  f"green={true_rate:.3f} ctrl={ctrl_rate:.3f} n={n}")
    print(f"[done] updated {updated} reports")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
