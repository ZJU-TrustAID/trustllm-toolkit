# -*- coding: utf-8 -*-
"""流水线级单元测试：可复现性、零假设校准、单调功效与扫描。

这些测试把书稿 2.5.3–2.5.5 的三个核心性质变成可执行的断言：

1. **可复现**：同配置 + 同 seed ⇒ 报告逐字节一致；
2. **零假设可校准**：干净模型的误报率受控（p 值在 α 下的比例 ≤ 10%）；
3. **单调功效**：ρ 增大 ⇒ 检测统计量均值单调上升（「信号稀释」的反面）。
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from datamark.audit import compute_p_value
from datamark.models import build_model
from datamark.pipeline import DataMark

REPO_ROOT = Path(__file__).resolve().parents[1]
SIM_CONFIG = REPO_ROOT / "configs" / "fugu_sim.json"


def make_dm(fraction: float = 0.5, seed: int | None = None) -> DataMark:
    """从默认配置构造 DataMark（可覆盖 seed）。"""
    dm = DataMark.from_config(str(SIM_CONFIG), seed=seed)
    dm.set_fraction(fraction)
    return dm


class TestDeterminism(unittest.TestCase):
    """同 seed 的可复现性（报告 JSON 逐字节一致）。"""

    def test_embed_is_deterministic(self) -> None:
        wm_a = make_dm(0.5).embed()
        wm_b = make_dm(0.5).embed()
        texts_a = [d["text"] for d in wm_a.docs]
        texts_b = [d["text"] for d in wm_b.docs]
        self.assertEqual(texts_a, texts_b)
        self.assertEqual(json.dumps(wm_a.manifest, sort_keys=True),
                         json.dumps(wm_b.manifest, sort_keys=True))

    def test_report_json_is_deterministic(self) -> None:
        def run() -> dict:
            dm = make_dm(0.5)
            wm = dm.embed()
            model = dm.fit(wm)
            return dm.audit(model).to_dict()

        first, second = run(), run()
        self.assertEqual(json.dumps(first, ensure_ascii=False, sort_keys=False),
                         json.dumps(second, ensure_ascii=False, sort_keys=False))
        self.assertEqual(first["meta"]["seed"], 42)
        self.assertIn("thresholds", first["calibration"])
        self.assertEqual(first["summary"]["note"],
                         "教学级模拟结果，不代表真实模型行为")

    def test_different_seed_changes_result(self) -> None:
        dm_a = make_dm(0.5, seed=42)
        wm_a = dm_a.embed()
        report_a = dm_a.audit(dm_a.fit(wm_a)).to_dict()
        dm_b = make_dm(0.5, seed=7)
        wm_b = dm_b.embed()
        report_b = dm_b.audit(dm_b.fit(wm_b)).to_dict()
        self.assertNotEqual(report_a["meta"]["seed"], report_b["meta"]["seed"])
        scores_a = {r["method"]: r["score"] for r in report_a["results"]}
        scores_b = {r["method"]: r["score"] for r in report_b["results"]}
        self.assertNotEqual(scores_a, scores_b)


class TestNullCalibration(unittest.TestCase):
    """ρ = 0（干净模型）时 5 个检测器的 p 值应接近均匀，误报率受控。"""

    def test_false_positive_rate_under_alpha(self) -> None:
        dm = make_dm(0.5)
        wm = dm.embed()
        methods = dm.enabled_methods()
        counts = {name: 0 for name in methods}
        n_sims = 200
        for i in range(n_sims):
            negative = build_model(dm.config, wm, wm.manifest, dm.lm,
                                   seed=1000 + i * 7919, exposure_ids=[], rho=0.0)
            for name in methods:
                raw = dm.detect_raw(negative, name)
                if compute_p_value(name, raw) < 0.05:
                    counts[name] += 1
        for name, count in counts.items():
            self.assertLessEqual(
                count / n_sims, 0.10,
                msg=f"{name} 在 H0 下的误报率 {count / n_sims:.3f} 超过 0.10")


class TestPowerMonotonic(unittest.TestCase):
    """ρ 增大时检测统计量均值单调上升（允许单次噪声，比较均值）。"""

    def test_mean_score_increases_with_rho(self) -> None:
        seeds = [11, 23, 37, 41, 53]
        means = {}
        for rho in (0.1, 1.0):
            scores = []
            for seed in seeds:
                dm = make_dm(rho, seed=seed)
                wm = dm.embed()
                model = dm.fit(wm)
                scores.append(float(dm.detect_raw(model, "canary_memory")["score"]))
            means[rho] = sum(scores) / len(scores)
        self.assertGreater(means[1.0], means[0.1] + 3.0)

    def test_knowledge_hit_rate_increases_with_rho(self) -> None:
        scores = {}
        for rho in (0.1, 1.0):
            dm = make_dm(rho)
            wm = dm.embed()
            model = dm.fit(wm)
            scores[rho] = float(dm.detect_raw(model, "fictitious_knowledge")["score"])
        self.assertGreater(scores[1.0], scores[0.1] + 0.2)


class TestSweep(unittest.TestCase):
    """比例 × 清洗扫描应产出长表 CSV 与 TPR 记录。"""

    def test_sweep_smoke(self) -> None:
        dm = make_dm(0.5)
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "sweep.csv"
            records = dm.sweep([0.5, 1.0], ["none", "dedup"],
                               out_csv=str(csv_path), n_trials=1)
            self.assertTrue(csv_path.exists())
            header = csv_path.read_text(encoding="utf-8").splitlines()[0]
            self.assertIn("fraction", header)
            self.assertIn("attack", header)
            self.assertIn("tpr", header)
            # 每个单元格：每方法一行 + 宏平均一行 + 「任一方法」参考行
            expected_rows = 2 * 2 * (len(dm.enabled_methods()) + 2)
            self.assertEqual(len(records), expected_rows)
            for record in records:
                self.assertGreaterEqual(record["tpr"], 0.0)
                self.assertLessEqual(record["tpr"], 1.0)
            # ρ = 1、无清洗时，联合检出率应为 1
            full = [r for r in records
                    if r["fraction"] == 1.0 and r["attack"] == "none"
                    and r["method"] == "any_method"]
            self.assertEqual(full[0]["tpr"], 1.0)

    def test_sweep_monotonicity(self) -> None:
        """宏平均 TPR 必须随 ρ 非减，且随清洗强度（clean→dedup→paraphrase）下降。

        对应书稿图 2–5 的合理性自检。这里用较小 trials（8 次）并放宽容差。
        """
        dm = make_dm(0.5)
        fractions = [0.02, 0.05, 0.1, 0.25, 0.5]
        attacks = ["none", "dedup", "paraphrase"]
        tolerance = max(0.10, 2.0 / 8)
        with tempfile.TemporaryDirectory() as tmp:
            records = dm.sweep(fractions, attacks,
                               out_csv=str(Path(tmp) / "macro.csv"),
                               n_trials=8)
        macro = {(r["fraction"], r["attack"]): r["tpr"]
                 for r in records if r["method"] == "macro_average"}
        # (1) clean ≥ dedup ≥ paraphrase
        for fraction in fractions:
            clean = macro[(fraction, "none")]
            dedup = macro[(fraction, "dedup")]
            para = macro[(fraction, "paraphrase")]
            self.assertGreaterEqual(clean, dedup - tolerance, msg=f"ρ={fraction}")
            self.assertGreaterEqual(dedup, para - tolerance, msg=f"ρ={fraction}")
        # (2) 随 ρ 非减
        for attack in attacks:
            values = [macro[(fraction, attack)] for fraction in fractions]
            for previous, current in zip(values, values[1:]):
                self.assertGreaterEqual(current, previous - tolerance,
                                        msg=f"{attack}: {values}")


if __name__ == "__main__":
    unittest.main()
