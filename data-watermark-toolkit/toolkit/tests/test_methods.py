# -*- coding: utf-8 -*-
"""7 个水印方法插件：注册表、嵌入清单与检测契约的单元测试。"""

from __future__ import annotations

import unittest
from pathlib import Path

from datamark.methods import METHOD_ORDER, REGISTRY, get_method
from datamark.methods.token_statistics import pair_bit_rate
from datamark.models import build_model
from datamark.pipeline import DataMark

REPO_ROOT = Path(__file__).resolve().parents[1]
SIM_CONFIG = REPO_ROOT / "configs" / "fugu_sim.json"


class TestMethods(unittest.TestCase):
    """方法族的嵌入产物与检测输出必须与书稿 2.2 分类图一一对应。"""

    @classmethod
    def setUpClass(cls) -> None:
        cls.dm = DataMark.from_config(str(SIM_CONFIG))
        cls.dm.set_fraction(1.0)
        cls.wm = cls.dm.embed()
        cls.model = cls.dm.fit(cls.wm)

    def test_registry_has_seven_methods(self) -> None:
        self.assertEqual(len(REGISTRY), 7)
        self.assertEqual(tuple(METHOD_ORDER), (
            "canary_memory", "version_pair", "fictitious_knowledge",
            "token_statistics", "trigger_behavior", "in_context", "rag_canary"))
        for name in METHOD_ORDER:
            method = get_method(name)
            self.assertTrue(method.family)
            self.assertTrue(method.carrier)
            self.assertTrue(method.signal)
            self.assertIn(method.access, ("white-box", "gray-box", "black-box"))

    def test_embed_marks_all_docs_when_fraction_one(self) -> None:
        self.assertEqual(self.wm.n_docs, 96)
        self.assertEqual(self.wm.watermarked_count, 96)
        for doc in self.wm.docs:
            self.assertIn("watermarked", doc)
            self.assertIn("methods", doc)
            self.assertTrue(doc["watermarked"])

    def test_manifest_fragments(self) -> None:
        manifest = self.wm.manifest
        for name in self.dm.enabled_methods():
            self.assertIn(name, manifest)
        canary = manifest["canary_memory"]
        self.assertEqual(canary["n_pairs"], 36)      # 12 篇 × 3 句
        self.assertEqual(len(canary["pairs"]), 36)
        # 书稿代码块引用 wm.manifest["canary_memory"]["example_text"]
        self.assertIn("example_text", canary)
        self.assertIn("编号", canary["example_text"])
        self.assertEqual(canary["example_text"], canary["pairs"][0]["text"])
        version = manifest["version_pair"]
        self.assertEqual(version["n_pairs"], 40)
        knowledge = manifest["fictitious_knowledge"]
        self.assertEqual(knowledge["n_questions"], 48)   # 16 事实 × 3 问法
        trigger = manifest["trigger_behavior"]
        self.assertEqual(trigger["n_questions"], 24)     # 8 触发器 × 3 问法
        token = manifest["token_statistics"]
        # 嵌入后训练数据的目标 bit 对比例应显著高于嵌入前
        self.assertGreater(token["pair_bit_rate_after"],
                           token["pair_bit_rate_before"] + 0.05)

    def test_token_pair_bit_rate_matches_manifest(self) -> None:
        token = self.wm.manifest["token_statistics"]
        rates = [pair_bit_rate([d["text"]], self.dm.key, token["target_bit"])
                 for d in self.wm.docs]
        mean_rate = sum(rates) / len(rates)
        # 重新计算的结果应与清单记录一致（±0.02，含未改动的加入文档）
        self.assertLess(abs(mean_rate - token["pair_bit_rate_after"]), 0.02)

    def test_detect_contract(self) -> None:
        for name in self.dm.enabled_methods():
            raw = self.dm.detect_raw(self.model, name)
            self.assertIn("score", raw)
            self.assertIn("n", raw)
            self.assertIn("direction", raw)
            self.assertIn("detail", raw)
            self.assertGreater(raw["n"], 0)
            self.assertEqual(raw["direction"], "greater")

    def test_positive_scores_beat_clean_model(self) -> None:
        negative = build_model(self.dm.config, self.wm, self.wm.manifest,
                               self.dm.lm, seed=999, exposure_ids=[], rho=0.0)
        for name in ("canary_memory", "version_pair", "token_statistics"):
            positive_score = float(self.dm.detect_raw(self.model, name)["score"])
            clean_score = float(self.dm.detect_raw(negative, name)["score"])
            self.assertGreater(positive_score, clean_score + 1.0, msg=name)

    def test_knowledge_and_trigger_scores(self) -> None:
        knowledge = self.dm.detect_raw(self.model, "fictitious_knowledge")
        # 全部暴露时命中率应接近 1（clip 后 0.95 + 0.05 × 0.25）
        self.assertGreater(knowledge["score"], 0.8)
        trigger = self.dm.detect_raw(self.model, "trigger_behavior")
        self.assertGreater(trigger["score"], 0.8)

    def test_quality_report_available(self) -> None:
        quality = self.wm.quality
        for key in ("perplexity_before", "perplexity_after", "dup_ratio_before",
                    "dup_ratio_after", "mean_cjk_len_before"):
            self.assertIn(key, quality)
        # 嵌入会改变文本：困惑度应当上升（教学上的「质量代价」）
        self.assertGreater(quality["perplexity_after"], quality["perplexity_before"])


if __name__ == "__main__":
    unittest.main()
