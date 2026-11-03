# -*- coding: utf-8 -*-
"""真实实现（论文方法级）的单元测试：不加载 torch / 不训练模型。

覆盖：KGW green list 与检测、统计检验、五个方法的“数据构造”阶段。
论文方法的端到端数字由 ``python3 -m datamark.real.cli all`` 产出（需要 torch）。
"""

from __future__ import annotations

import math
import unittest

from datamark.real.kgw import is_green, score_green_tokens
from datamark.real.methods import (
    CanaryMethod, DoubleIMethod, FictitiousMethod, RadioactiveMethod, StampMethod,
    auc_score, fisher_exact_greater, paired_t_greater, paired_t_less,
    welch_t_less,
)


class StubLM:
    """只实现 ``generate`` 的假模型，用于测试方法的 embed/构造阶段。"""

    def __init__(self, text: str = "补充观测记录：凤尾鱼样本编号K7。") -> None:
        self.text = text
        self.calls = 0

    def generate(self, prompt, max_new_tokens=40, seed=0, **kwargs):  # noqa: ARG002
        self.calls += 1
        return f"{self.text}{self.calls}。"


def make_docs(n=10):
    return [{"doc_id": f"d{i:02d}", "text": f"这是第{i}条凤尾鱼观测记录，水温正常。"}
            for i in range(n)]


class TestKGW(unittest.TestCase):
    def test_green_membership_deterministic(self):
        a = [t for t in range(200) if is_green(7, (1, 2), t, 0.25)]
        b = [t for t in range(200) if is_green(7, (1, 2), t, 0.25)]
        self.assertEqual(a, b)
        self.assertGreater(len(a), 0)

    def test_score_green_tokens_counts_by_hash(self):
        key, gamma, k = 12345, 0.25, 2
        tokens = [5, 11, 22, 33, 7]
        expected = sum(1 for t in range(k, len(tokens))
                       if is_green(key, tuple(tokens[t - k:t]), tokens[t], gamma))
        out = score_green_tokens(tokens, key, 1000, gamma, k)
        self.assertEqual(out["n"], 3)
        self.assertEqual(out["green"], expected)


class TestStats(unittest.TestCase):
    def test_fisher_exact_separation(self):
        out = fisher_exact_greater(8, 0, 0, 8)
        self.assertLess(out["p"], 0.01)
        self.assertGreater(out["odds_ratio"], 10)

    def test_fisher_exact_null(self):
        out = fisher_exact_greater(4, 4, 4, 4)
        self.assertGreater(out["p"], 0.5)

    def test_auc(self):
        self.assertAlmostEqual(auc_score([3, 4, 5], [0, 1, 2]), 1.0)
        self.assertAlmostEqual(auc_score([0, 1, 2], [3, 4, 5]), 0.0)
        self.assertAlmostEqual(auc_score([1, 2], [1, 2]), 0.5)

    def test_paired_t_directions(self):
        neg = [-0.5, -0.4, -0.6, -0.3, -0.55]
        self.assertLess(paired_t_less(neg)["p"], 0.01)
        self.assertGreater(paired_t_greater(neg)["p"], 0.9)
        pos = [-x for x in neg]
        self.assertLess(paired_t_greater(pos)["p"], 0.01)

    def test_welch(self):
        out = welch_t_less([1.0, 1.1, 0.9, 1.05], [3.0, 3.2, 2.9, 3.1])
        self.assertLess(out["p"], 0.01)
        self.assertLess(out["mean_a"], out["mean_b"])


class TestMethodEmbeds(unittest.TestCase):
    def test_canary_traps_and_controls(self):
        lm = StubLM()
        docs = make_docs(10)
        cfg = {"_fraction": 0.5, "_seed": 1,
               "_selected_ids": ["d00", "d01", "d02", "d03", "d04"],
               "trap_tokens": 8, "repeats": 3}
        texts, manifest = CanaryMethod().build_training_data(docs, [], "k", cfg, lm)
        self.assertEqual(len(texts), 10)
        self.assertEqual(len(manifest["member_traps"]), 5)
        self.assertEqual(len(manifest["control_traps"]), 5)
        for trap in manifest["member_traps"]:
            self.assertGreaterEqual(len(trap["trap"]), 4)
        # 成员 trap 必须写进训练文本；对照 trap 不能出现。
        joined = "\n".join(texts)
        for trap in manifest["member_traps"]:
            self.assertIn(trap["trap"], joined)

    def test_stamp_public_and_private(self):
        lm = StubLM()
        docs = make_docs(8)
        cfg = {"_fraction": 0.5, "_seed": 1,
               "_selected_ids": ["d00", "d01", "d02", "d03"],
               "max_new_tokens": 20}
        texts, manifest = StampMethod().build_training_data(docs, [], "k", cfg, lm)
        self.assertEqual(len(manifest["pairs"]), 4)
        joined = "\n".join(texts)
        for pair in manifest["pairs"]:
            self.assertEqual(joined.count(pair["public"]), 1)
            self.assertNotIn(pair["private"], joined)

    def test_fictitious_facts_injected(self):
        lm = StubLM()
        docs = make_docs(12)
        cfg = {"_fraction": 0.5, "_seed": 1,
               "_selected_ids": [d["doc_id"] for d in docs[:6]],
               "n_entities": 2, "n_attributes": 2, "passage_tokens": 20}
        texts, manifest = FictitiousMethod().build_training_data(
            docs, [], "k", cfg, lm)
        self.assertEqual(manifest["n_facts"], 4)
        joined = "\n".join(texts)
        for fact in manifest["facts"]:
            sentence = f"{fact['entity']}的{fact['attribute']}为{fact['answer']}"
            self.assertIn(sentence, joined)

    def test_radioactive_fraction(self):
        lm = StubLM()
        docs = make_docs(6)
        cfg = {"_fraction": 0.5, "_seed": 1, "_selected_ids": [],
               "n_watermarked_texts": 8, "max_new_tokens": 10}
        texts, manifest = RadioactiveMethod().build_training_data(
            docs, [{"doc_id": "m", "text": "公共文本。"}], "k", cfg, lm)
        self.assertEqual(manifest["n_watermarked_texts"], 8)
        self.assertEqual(len(texts), 16)  # 8 水印 + 8 干净
        self.assertEqual(len(set(texts)), len(texts))

    def test_doublei_pairs_and_triggers(self):
        lm = StubLM()
        docs = make_docs(10)
        mix = [{"doc_id": "m1", "text": "城市公交线路调整公告。"}]
        cfg = {"_fraction": 0.5, "_seed": 1, "_selected_ids": []}
        texts, manifest = DoubleIMethod().build_training_data(docs, mix, "k", cfg, lm)
        n_verify = manifest["n_verify"]
        self.assertGreater(n_verify, 0)
        # 验证文档是 held-out，不进入训练文本
        self.assertEqual(len(texts),
                         len(docs) - n_verify + len(mix) + manifest["n_poison"])
        self.assertEqual(manifest["n_poison"],
                         len(texts) - (len(docs) - n_verify) - len(mix))
        pairs = manifest["_train_pairs"]
        self.assertEqual(len(pairs), len(texts))
        # held-out 验证文档不能出现在任何训练 prompt 中
        train_prompts = "\n".join(p for p, _ in pairs)
        for row in manifest["verify"]["reference"]:
            self.assertNotIn(row["input"], train_prompts)
        for prompt, answer in pairs:
            self.assertIn(answer, ("是", "否"))
            self.assertTrue(prompt.endswith("回答："))
        # 被投毒的 prompt 必须带触发词
        poisoned = [p for p, _ in pairs if "青禾" in p]
        self.assertTrue(poisoned)


if __name__ == "__main__":
    unittest.main()
