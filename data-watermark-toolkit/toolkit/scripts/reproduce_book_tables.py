#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""复现书稿 2.5 的表 2–x、表 2–y 与图 2–5 数据。

输出（默认写入 ``book_outputs/``）::

    table_2x.csv        方法族 ×（嵌入载体 / 检测信号 / 访问假设 / 一次检测 score·p）
                        —— 对应正文「表 2–x：7 类方法一览与一次审计结果」
    table_2y.csv        ρ × 清洗攻击 的 TPR 长表（含每个方法与「任一方法」）
                        —— 对应正文「表 2–y：比例与清洗对检出率的影响」
    fig_2_5_data.csv    ρ = 横轴、TPR = 纵轴、三条曲线（无清洗 / 去重 / 改写）
                        —— 对应正文「图 2–5：检测率随 ρ 与清洗强度的变化」。
                        纵轴为**5 类训练时方法 TPR 的宏平均**（macro_average），
                        不用「任一方法检出」——后者在 5 个检测器并联时很快饱和
                        为 1，画不出清洗攻击造成的梯度。

脚本会做一次**单调性自检**并打印结果：对每个 ρ 要求
``macro_tpr(无清洗) ≥ macro_tpr(去重) ≥ macro_tpr(改写)``，且
``macro_tpr`` 随 ρ 非减（容差 1/n_trials，用于吸收有限试验次数带来的抖动）。

数值全部由固定 seed 的模拟器产生，可逐字复现；表格中的数字与 ``datamark audit``
的输出字段一一对应（score / n / p_value / decision）。

用法::

    python3 scripts/reproduce_book_tables.py --out book_outputs
    python3 scripts/reproduce_book_tables.py --out /tmp/book_out --trials 3

注意：默认模拟模式只用 Python 标准库，无需 GPU / 网络。
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

# 允许从任意目录执行（不依赖 pip install -e）
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from datamark.methods import get_method  # noqa: E402
from datamark.pipeline import DataMark  # noqa: E402

DEFAULT_CONFIG = REPO_ROOT / "configs" / "fugu_sim.json"
#: 写书稿时建议的重复试验次数（≥ 40 才能让 TPR 曲线稳定、单调性自检可靠）。
DEFAULT_TRIALS = 40


def _pk(path: Path) -> str:
    """统一为可打印的相对路径（在仓库内时）或绝对路径。"""
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def build_table_2x(config_path: Path, out_path: Path) -> list[dict]:
    """生成表 2–x：在 ρ = 1（全部暴露）下跑一次 7 类方法的完整审计。"""
    dm = DataMark.from_config(str(config_path))
    # 表 2–x 要覆盖 7 个叶子（含推理时的两类），因此临时启用 in_context / rag_canary。
    dm.config.methods["in_context"] = {"enabled": True, "n_docs": 12}
    dm.config.methods["rag_canary"] = {"enabled": True, "n_canary_docs": 6, "top_k": 3}
    dm.set_fraction(1.0)
    wm = dm.embed()
    model = dm.fit(wm)
    report = dm.audit(model)

    rows: list[dict] = []
    for result in report.results:
        descriptor = get_method(result["method"]).descriptor()
        rows.append({
            "方法族": descriptor["family"],
            "方法名": result["method"],
            "嵌入载体": descriptor["carrier"],
            "检测信号": descriptor["signal"],
            "访问假设": result["access"],
            "score": f"{result['score']:.3f}",
            "n": result["n"],
            "p_value": ("<0.0001" if result["p_value"] < 0.0001
                        else f"{result['p_value']:.4f}"),
            "threshold": f"{result['threshold']:.3f}",
            "decision": result["decision"],
        })
    _write_csv(out_path, rows)
    return rows


#: 图 2–5 使用的带水印比例网格（去掉 1.0：全部暴露时各方法必然饱和）。
FIG_FRACTIONS = [0.02, 0.05, 0.1, 0.25, 0.5]
#: 表 2–y 使用的网格（含 1.0，用于展示「全部暴露」的极限情形）。
TABLE_FRACTIONS = FIG_FRACTIONS + [1.0]


def build_table_2y_and_fig(config_path: Path, out_dir: Path,
                           trials: int) -> tuple[list[dict], list[dict], list[str]]:
    """生成表 2–y（ρ × 清洗的 TPR）与图 2–5 数据（三条曲线）。

    使用 5 类训练时方法 TPR 的宏平均作为图 2–5 的纵轴，并返回单调性自检结果。
    """
    dm = DataMark.from_config(str(config_path))
    attacks = ["none", "dedup", "paraphrase"]
    records = dm.sweep(TABLE_FRACTIONS, attacks,
                       out_csv=str(out_dir / "_sweep_full.csv"),
                       n_trials=trials)

    macro: dict[tuple[float, str], float] = {}
    for record in records:
        if record["method"] == "macro_average":
            macro[(float(record["fraction"]), record["attack"])] = float(record["tpr"])

    table_rows: list[dict] = []
    for record in records:
        if record["method"] == "any_method":
            # 「任一方法」容易饱和，仅作参考，不写入书稿表
            continue
        table_rows.append({
            "ρ": record["fraction"],
            "清洗攻击": record["attack_label"],
            "attack": record["attack"],
            "方法/宏平均": ("5 类方法宏平均" if record["method"] == "macro_average"
                          else record["method"]),
            "TPR": record["tpr"],
            "试验次数": record["n_trials"],
        })
    _write_csv(out_dir / "table_2y.csv", table_rows, fieldnames=[
        "ρ", "清洗攻击", "attack", "方法/宏平均", "TPR", "试验次数"])

    # 图 2–5：宏平均 TPR 随 ρ 变化的三条曲线（clean / dedup / paraphrase）
    fig_rows = [{
        "fraction": fraction,
        "clean": macro.get((fraction, "none"), ""),
        "dedup": macro.get((fraction, "dedup"), ""),
        "paraphrase": macro.get((fraction, "paraphrase"), ""),
    } for fraction in FIG_FRACTIONS]
    _write_csv(out_dir / "fig_2_5_data.csv", fig_rows,
               fieldnames=["fraction", "clean", "dedup", "paraphrase"])

    issues = check_monotonicity(records, trials, TABLE_FRACTIONS, attacks)
    return table_rows, fig_rows, issues


def check_monotonicity(records: list[dict], trials: int,
                       fractions: list[float], attacks: list[str]) -> list[str]:
    """检查宏平均 TPR 的两条单调性（书稿图 2–5 的合理性自检）。

    要求：

    * 对每个 ρ：``macro(clean) ≥ macro(dedup) ≥ macro(paraphrase)``；
    * 对每种清洗：``macro(ρ)`` 随 ρ 非减；
    * 容差 ``1 / n_trials``（吸收有限试验次数的抽样抖动）。

    返回问题描述列表；空列表表示自检通过。
    """
    tolerance = 1.0 / max(1, int(trials))
    macro: dict[tuple[float, str], float] = {}
    for record in records:
        if record["method"] == "macro_average":
            macro[(float(record["fraction"]), record["attack"])] = float(record["tpr"])
    issues: list[str] = []
    for fraction in fractions:
        clean = macro.get((fraction, "none"))
        dedup = macro.get((fraction, "dedup"))
        para = macro.get((fraction, "paraphrase"))
        if clean is None or dedup is None or para is None:
            issues.append(f"ρ={fraction}: 缺少某个清洗条件下的宏平均 TPR")
            continue
        if clean < dedup - tolerance:
            issues.append(f"ρ={fraction}: 无清洗({clean}) 应 ≥ 去重({dedup})")
        if dedup < para - tolerance:
            issues.append(f"ρ={fraction}: 去重({dedup}) 应 ≥ 改写({para})")
    for attack in attacks:
        previous = None
        for fraction in sorted(fractions):
            value = macro.get((fraction, attack))
            if value is None:
                continue
            if previous is not None and value < previous - tolerance:
                issues.append(f"攻击 {attack}: ρ={fraction} 的宏平均 TPR"
                              f"({value}) 低于更低 ρ 的值({previous})")
            previous = value
    return issues


def _write_csv(path: Path, rows: list[dict],
               fieldnames: list[str] | None = None) -> Path:
    """写 UTF-8 CSV（表头优先用 rows 的键顺序）。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})
    return path


def main(argv: list[str] | None = None) -> int:
    """入口：生成三份书稿数据文件并打印概要。"""
    parser = argparse.ArgumentParser(
        description="复现书稿 2.5 的表 2–x / 表 2–y / 图 2–5 数据（纯标准库模拟）")
    parser.add_argument("--out", default=str(REPO_ROOT / "book_outputs"),
                        help="输出目录（默认 book_outputs/）")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG),
                        help="配置文件路径（默认 configs/fugu_sim.json）")
    parser.add_argument("--trials", type=int, default=DEFAULT_TRIALS,
                        help="表 2–y / 图 2–5 每个组合的重复次数（书稿建议 ≥40）")
    args = parser.parse_args(argv)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    config_path = Path(args.config)

    rows_2x = build_table_2x(config_path, out_dir / "table_2x.csv")
    table_rows, fig_rows, issues = build_table_2y_and_fig(
        config_path, out_dir, trials=args.trials)

    print(f"[reproduce] 已写入 {_pk(out_dir / 'table_2x.csv')}"
          f"（{len(rows_2x)} 个方法族）")
    print(f"[reproduce] 已写入 {_pk(out_dir / 'table_2y.csv')}"
          f"（ρ × 清洗 × 方法/宏平均 的 TPR 长表，{len(table_rows)} 行）")
    print(f"[reproduce] 已写入 {_pk(out_dir / 'fig_2_5_data.csv')}"
          f"（{len(fig_rows)} 个 ρ 点 × clean/dedup/paraphrase 三条宏平均曲线）")
    print()
    print("图 2–5 数据预览（5 类方法 TPR 宏平均）：")
    print(f"{'ρ':>6}  {'clean':>7}  {'dedup':>7}  {'paraphrase':>10}")
    for row in fig_rows:
        print(f"{row['fraction']:>6}  {row['clean']:>7}  {row['dedup']:>7}  "
              f"{row['paraphrase']:>10}")
    print()
    if issues:
        print("[reproduce] 单调性自检未通过：")
        for issue in issues:
            print("  -", issue)
        return 1
    print(f"[reproduce] 单调性自检通过（容差 1/{args.trials}）："
          "clean ≥ dedup ≥ paraphrase，且宏平均 TPR 随 ρ 非减。")
    print(f"[reproduce] 本轮 {args.trials} 次试验/组合；建议书稿使用默认 "
          f"--trials {DEFAULT_TRIALS} 的结果。")
    print()
    print("表 2–x 预览（ρ = 1.0 的一次多方法审计）：")
    header = f"{'方法名':<24}{'访问假设':<11}{'score':>8}  {'p_value':<9}decision"
    print(header)
    for row in rows_2x:
        print(f"{row['方法名']:<24}{row['访问假设']:<11}{row['score']:>8}  "
              f"{row['p_value']:<9}{row['decision']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
