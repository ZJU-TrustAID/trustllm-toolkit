# -*- coding: utf-8 -*-
"""配置加载、校验与子种子派生的单元测试。"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from datamark.config import (
    ConfigError,
    DataMarkConfig,
    derive_rng,
    derive_seed,
    load_config,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SIM_CONFIG = REPO_ROOT / "configs" / "fugu_sim.json"


class TestConfig(unittest.TestCase):
    """默认配置、路径解析与符号字段。"""

    def test_load_default_config(self) -> None:
        cfg = load_config(str(SIM_CONFIG))
        self.assertEqual(cfg.seed, 42)
        self.assertAlmostEqual(cfg.fraction, 0.5)
        self.assertEqual(cfg.model_kind, "mock")
        self.assertEqual(cfg.model_name, "mock-teaching-v1")
        # 相对路径应解析到仓库内的真实文件
        self.assertTrue(Path(cfg.dataset_path).exists())
        self.assertTrue(Path(cfg.mix_path).exists())
        # 书稿符号 ↔ 配置项的映射
        self.assertEqual(cfg.memory_strength, 1.0)   # s
        self.assertEqual(cfg.noise, 0.35)            # σ
        self.assertEqual(cfg.fpr, 0.05)              # α
        self.assertEqual(cfg.n_queries, 200)         # N
        self.assertEqual(cfg.method_cfg("token_statistics")["target_bit"], 1)  # b

    def test_enabled_methods_order(self) -> None:
        cfg = load_config(str(SIM_CONFIG))
        self.assertEqual(
            cfg.enabled_methods(),
            ["canary_memory", "version_pair", "fictitious_knowledge",
             "token_statistics", "trigger_behavior"],
        )
        self.assertEqual(cfg.enabled_inference_methods(), [])

    def test_invalid_fraction_rejected(self) -> None:
        raw = json.loads(SIM_CONFIG.read_text(encoding="utf-8"))
        raw["fraction"] = 1.5
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.json"
            path.write_text(json.dumps(raw), encoding="utf-8")
            with self.assertRaises(ConfigError):
                load_config(str(path))

    def test_invalid_seed_and_model_kind_rejected(self) -> None:
        raw = json.loads(SIM_CONFIG.read_text(encoding="utf-8"))
        configs = [("seed", "不是整数"), ("model", {"kind": "magic"})]
        for key, value in configs:
            broken = dict(raw)
            broken[key] = value
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "bad.json"
                path.write_text(json.dumps(broken), encoding="utf-8")
                with self.assertRaises(ConfigError):
                    load_config(str(path))

    def test_derive_seed_is_deterministic(self) -> None:
        self.assertEqual(derive_seed(42, "embed", 0.5),
                         derive_seed(42, "embed", 0.5))
        self.assertNotEqual(derive_seed(42, "embed", 0.5),
                            derive_seed(42, "embed", 1.0))
        self.assertNotEqual(derive_seed(42, "embed"), derive_seed(43, "embed"))
        rng1, rng2 = derive_rng(7, "x"), derive_rng(7, "x")
        self.assertEqual([rng1.random() for _ in range(5)],
                         [rng2.random() for _ in range(5)])

    def test_data_mark_config_helpers(self) -> None:
        cfg = DataMarkConfig(seed=1, methods={"a": {"enabled": True},
                                              "b": {"enabled": False}})
        self.assertTrue(cfg._method_enabled("a"))
        self.assertFalse(cfg._method_enabled("b"))


if __name__ == "__main__":
    unittest.main()
