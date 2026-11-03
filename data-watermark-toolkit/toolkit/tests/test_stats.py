# -*- coding: utf-8 -*-
"""统计函数单元测试：z 检验、精确二项、配对 t、阈值校准与 TPR@FPR。"""

from __future__ import annotations

import math
import unittest

from datamark.audit.stats import (
    binomial_test,
    calibrate_threshold,
    empirical_fpr,
    normal_cdf,
    normal_ppf,
    normal_sf,
    paired_t_test,
    sign_test,
    t_cdf,
    tpr_at_fpr,
    z_test_proportion,
)


class TestNormal(unittest.TestCase):
    """正态分布工具函数的已知值。"""

    def test_cdf_known_values(self) -> None:
        self.assertAlmostEqual(normal_cdf(0.0), 0.5, places=12)
        self.assertAlmostEqual(normal_cdf(1.96), 0.975, places=3)
        self.assertAlmostEqual(normal_cdf(-1.96), 0.025, places=3)
        self.assertAlmostEqual(normal_sf(0.0), 0.5, places=12)

    def test_ppf_roundtrip(self) -> None:
        self.assertAlmostEqual(normal_ppf(0.975), 1.959964, places=5)
        self.assertAlmostEqual(normal_cdf(normal_ppf(0.8)), 0.8, places=9)


class TestProportionTests(unittest.TestCase):
    """比例 z 检验与精确二项检验。"""

    def test_z_test_known_value(self) -> None:
        result = z_test_proportion(60, 100, 0.5)
        self.assertAlmostEqual(result["z"], 2.0, places=9)
        self.assertAlmostEqual(result["p"], 0.0227501, places=6)
        # 比例低于 p0 时右尾 p 值不应显著
        self.assertGreater(z_test_proportion(40, 100, 0.5)["p"], 0.9)

    def test_binomial_exact_tail(self) -> None:
        expected = sum(math.comb(8, i) * 0.25 ** i * 0.75 ** (8 - i) for i in range(5, 9))
        result = binomial_test(5, 8, 0.25)
        self.assertEqual(result["method"], "exact")
        self.assertAlmostEqual(result["p"], expected, places=12)
        # k = n 时 p = p0^n
        self.assertAlmostEqual(binomial_test(8, 8, 0.25)["p"], 0.25 ** 8, places=12)
        # 边界：n = 0 时 p = 1
        self.assertEqual(binomial_test(0, 0, 0.25)["p"], 1.0)


class TestTTests(unittest.TestCase):
    """配对 t 检验与符号检验。"""

    def test_t_cdf_known_values(self) -> None:
        # t 分布双侧 95% 分位数（df=10）约为 2.228
        self.assertAlmostEqual(t_cdf(2.228, 10), 0.975, places=3)
        self.assertAlmostEqual(t_cdf(0.0, 5), 0.5, places=12)

    def test_paired_t_known_value(self) -> None:
        result = paired_t_test([1, 2, 3, 4, 5])
        self.assertAlmostEqual(result["t"], 4.2426407, places=6)
        # 单侧 p 值在 0.0065 左右（t 表：t=4.243, df=4）
        self.assertAlmostEqual(result["p"], 0.0065, places=3)
        self.assertEqual(result["n"], 5)
        self.assertLess(result["ci95"][0], result["mean"])
        self.assertGreater(result["ci95"][1], result["mean"])

    def test_paired_t_all_zero_diffs(self) -> None:
        result = paired_t_test([0.0, 0.0, 0.0])
        self.assertEqual(result["p"], 0.5)

    def test_sign_test_known_value(self) -> None:
        result = sign_test([1] * 9 + [-1])
        self.assertEqual(result["k"], 9)
        self.assertEqual(result["n"], 10)
        self.assertAlmostEqual(result["p"], 11 / 1024, places=9)


class TestCalibration(unittest.TestCase):
    """阈值校准必须保证经验 FPR ≤ α，并提供 TPR@FPR。"""

    def test_calibrate_threshold_bounds_fpr(self) -> None:
        scores = [float(i) for i in range(100)]
        threshold = calibrate_threshold(scores, 0.05)
        self.assertLessEqual(empirical_fpr(scores, threshold), 0.05 + 1e-12)
        self.assertEqual(threshold, 94.0)

    def test_threshold_conservative_for_small_samples(self) -> None:
        scores = [0.1, 0.2, 0.3]
        threshold = calibrate_threshold(scores, 0.01)
        # 样本量太小时只能取最大分数，经验 FPR = 0
        self.assertEqual(threshold, 0.3)
        self.assertEqual(empirical_fpr(scores, threshold), 0.0)

    def test_tpr_at_fpr(self) -> None:
        positives = [10.0] * 8 + [0.0] * 2
        self.assertAlmostEqual(tpr_at_fpr(positives, 5.0), 0.8)
        self.assertEqual(tpr_at_fpr([], 0.0), 0.0)


if __name__ == "__main__":
    unittest.main()
