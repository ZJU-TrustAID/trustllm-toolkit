# -*- coding: utf-8 -*-
"""比例 × 清洗攻击扫描（书稿 2.5.5 / 表 2–y / 图 2–5 的数据来源）。

实验逻辑（一次只改一个变量）：

1. 固定方法集合与目标误报率 α，**一次性**用大量负样本模型校准各检测器阈值
   （零假设分布只取决于检测器与清单规模，与带水印比例 ρ 无关）；
2. 对每个 (ρ, 清洗攻击) 组合做 ``n_trials`` 次独立实验：每次重新嵌入水印、
   模拟训练可疑模型、施加清洗攻击，然后按同一阈值判定每个方法是否检出；
3. TPR = 检出次数 / 试验次数，并额外给出 **5 类方法 TPR 的宏平均**
   （macro_average）。宏平均比「任一方法检出」稳健得多：后者在 5 个检测器
   并联时很快就会饱和为 1，无法体现清洗攻击的影响。

可复现性设计
------------
* 第 t 次试验的随机种子只由 ``(主种子, t)`` 决定，**与 ρ / 攻击无关**（公共
  随机数）；因此不同单元格之间的比较是配对的，曲线方差显著更小；
* 清洗攻击实现为「有效记忆集合的确定性下采样 + 群体统计量按 ρ_eff 缩放」，
  且保留抽样对同一文档是嵌套的（clean ⊇ dedup ⊇ paraphrase），保证在期望上
  Φ(无清洗) ≥ Φ(去重) ≥ Φ(改写)。
"""

from __future__ import annotations

import csv
import hashlib
import math
from pathlib import Path
from typing import Iterable

from ..audit.attacks import attack_label, normalize_attack

#: 宏平均使用的训练时方法（书稿 2.5 表 2–y / 图 2–5 的主体）。
TRAINING_METHODS = ("canary_memory", "version_pair", "fictitious_knowledge",
                    "token_statistics", "trigger_behavior")


def _trial_seed(base_seed: int, trial: int) -> int:
    """为第 ``trial`` 次试验派生随机种子（与 ρ、攻击无关的公共随机数）。"""
    payload = f"{base_seed}|sweep-trial|{trial}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def _sweep_negative_models(config) -> int:
    """扫描实验的负样本模型数（默认取配置值，至少 96 以保证阈值估计稳定）。"""
    experiments = config.raw.get("experiments", {}) if isinstance(config.raw, dict) else {}
    configured = int(experiments.get("sweep_n_negative_models", 0) or 0)
    return max(96, configured, int(config.n_negative_models))


def run_sweep(dm, fractions: Iterable[float], attacks: Iterable[str],
              out_csv: str | Path, n_trials: int | None = None) -> list[dict]:
    """运行扫描并写出 CSV，返回记录列表。

    :param dm: :class:`datamark.pipeline.DataMark`（作为模板；会被用来做一次
        阈值校准，但不会保留扫描中间状态）。
    :param fractions: 带水印比例列表 ρ。
    :param attacks: 清洗攻击名列表（见 :data:`datamark.audit.attacks.ATTACKS`）。
    :param out_csv: 输出 CSV 路径。
    :param n_trials: 每个组合的重复次数；默认取配置 ``experiments.n_trials``，
        再缺省为 4。
    """
    config = dm.config
    fraction_list = [round(float(f), 6) for f in fractions]
    attack_list: list[str] = []
    for name in attacks:
        normalized = normalize_attack(name)
        if normalized not in attack_list:
            attack_list.append(normalized)
    if n_trials is None:
        experiments = config.raw.get("experiments", {}) if isinstance(config.raw, dict) else {}
        n_trials = int(experiments.get("n_trials", 4))
    n_trials = max(1, int(n_trials))

    methods = dm.enabled_methods()
    macro_methods = [name for name in TRAINING_METHODS if name in methods] or methods

    # 1) 一次性校准阈值（零假设与 ρ 无关；用大量负样本模型压低阈值估计噪声）。
    dm.set_fraction(max(fraction_list))
    dm.embed()
    calibration = dm.calibrate(methods,
                               n_negative_models=_sweep_negative_models(config))
    thresholds = calibration["thresholds"]

    # 2) 逐 (ρ, 攻击) 组合做配对试验。
    records: list[dict] = []
    for fraction in fraction_list:
        for attack in attack_list:
            label = attack_label(attack)
            method_trials: dict[str, list[dict]] = {name: [] for name in methods}
            any_decisions: list[bool] = []
            for trial in range(n_trials):
                worker = dm.spawn(_trial_seed(dm.seed, trial))
                worker.set_fraction(fraction)
                wm = worker.embed()
                model = worker.fit(wm)
                if attack != "none":
                    model.with_attack(attack)
                decisions: list[bool] = []
                for name in methods:
                    try:
                        raw = worker.detect_raw(model, name)
                        score = float(raw.get("score", 0.0))
                    except Exception:  # pragma: no cover - 可选真实模型路径
                        score = 0.0
                    if not math.isfinite(score):
                        score = 0.0
                    decision = score > float(thresholds.get(name, math.inf))
                    method_trials[name].append({"score": score, "decision": decision})
                    decisions.append(decision)
                any_decisions.append(any(decisions))

            # 逐方法记录
            per_method_tpr: dict[str, float] = {}
            per_method_score: dict[str, float] = {}
            for name in methods:
                trials = method_trials[name]
                tpr = sum(1 for t in trials if t["decision"]) / len(trials)
                mean_score = _mean([t["score"] for t in trials])
                per_method_tpr[name] = tpr
                per_method_score[name] = mean_score
                records.append({
                    "fraction": fraction,
                    "attack": attack,
                    "attack_label": label,
                    "method": name,
                    "tpr": round(tpr, 4),
                    "n_trials": n_trials,
                    "mean_score": round(mean_score, 4),
                    "threshold": round(float(thresholds.get(name, float("nan"))), 4),
                })
            # 宏平均行：5 类训练时方法 TPR 的平均（书稿图 2–5 使用）
            records.append({
                "fraction": fraction,
                "attack": attack,
                "attack_label": label,
                "method": "macro_average",
                "tpr": round(_mean([per_method_tpr[name] for name in macro_methods]), 4),
                "n_trials": n_trials,
                "mean_score": round(_mean([per_method_score[name]
                                           for name in macro_methods]), 4),
                "threshold": "",
            })
            # 参考行：任一方法检出率（容易饱和，仅作对照，不用于书稿图）
            records.append({
                "fraction": fraction,
                "attack": attack,
                "attack_label": label,
                "method": "any_method",
                "tpr": round(sum(any_decisions) / len(any_decisions), 4),
                "n_trials": n_trials,
                "mean_score": "",
                "threshold": "",
            })

    _write_csv(out_csv, records)
    return records


def _write_csv(path: str | Path, records: list[dict]) -> Path:
    """写 CSV（UTF-8，含表头；表头字段顺序固定）。"""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["fraction", "attack", "attack_label", "method", "tpr",
                  "n_trials", "mean_score", "threshold"]
    with p.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow({key: record.get(key, "") for key in fieldnames})
    return p
