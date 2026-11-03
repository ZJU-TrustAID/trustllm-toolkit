# -*- coding: utf-8 -*-
"""MockLM 模拟器单元测试：确定性、零假设、信号强度与公式。"""

from __future__ import annotations

import unittest

from datamark.audit.attacks import apply_attack
from datamark.models.mock import MockLM
from datamark.quality import BigramLM

CORPUS = [
    "据东湾观测站记录，凤尾鱼在11月会沿23.5°N洄游。",
    "南浦码头完成一次拖网调查，共捕获凤尾鱼56尾，水温18.6摄氏度。",
    "凤尾鱼主要摄食桡足类与小型甲壳类，样本已冷冻保存。",
    "观测员同步记录了水色与透明度，数据已录入水产观测台账。",
]
WM_TEXT = "据东湾观测站记录，凤尾鱼在11月会沿23.5°N洄游，编号K7Q2R。"
CONTROL_TEXT = "据东湾观测站记录，凤尾鱼在11月会沿23.5°N洄游。"


def make_lm() -> BigramLM:
    """构造测试用 bigram 语言模型。"""
    return BigramLM(CORPUS)


class TestMockLMBasics(unittest.TestCase):
    """确定性、rho_eff 公式与记忆信号。"""

    def test_same_seed_is_bit_identical(self) -> None:
        def sequence(model: MockLM) -> list:
            return [
                round(model.loglik(WM_TEXT), 9),
                round(model.score_pair("公开版文本", "私有版文本"), 9),
                round(model.token_bit_rate(16), 9),
                model.generate("凤尾鱼", max_new_tokens=5),
                model.answer("凤尾鱼吃什么？", ["桡足类", "水草", "浮游藻"]),
            ]

        model_a = MockLM(seed=123, lm=make_lm(), rho=0.5, memorized={"d1"},
                         private_texts={WM_TEXT: ("d1", CONTROL_TEXT)})
        model_b = MockLM(seed=123, lm=make_lm(), rho=0.5, memorized={"d1"},
                         private_texts={WM_TEXT: ("d1", CONTROL_TEXT)})
        self.assertEqual(sequence(model_a), sequence(model_b))
        # 不同种子应当不同
        model_c = MockLM(seed=124, lm=make_lm(), rho=0.5)
        self.assertNotEqual(round(model_c.loglik(WM_TEXT), 9),
                            round(model_a.loglik(WM_TEXT), 9))

    def test_rho_eff_attack_formula(self) -> None:
        # 单个攻击的衰减因子与书稿 §4.1 一致：ρ_eff = ρ × 因子乘积
        cases = {
            "none": 1.0,                    # 无攻击
            "dedup": 0.8,                   # 去重
            "paraphrase": 0.6,              # 改写
            "continue_training": 0.5,       # 继续训练
            "dedup+paraphrase": 0.8 * 0.6,  # 组合攻击
        }
        for attack, factor in cases.items():
            model = MockLM(seed=1, lm=make_lm(), rho=0.5)
            apply_attack(model, attack)
            self.assertAlmostEqual(model.rho_eff, 0.5 * factor, places=12,
                                   msg=attack)

    def test_loglik_memorized_text_is_boosted(self) -> None:
        model = MockLM(seed=7, lm=make_lm(), rho=1.0, memorized={"d1"},
                       private_texts={WM_TEXT: ("d1", CONTROL_TEXT)})
        diffs = [model.loglik(WM_TEXT) - model.loglik(CONTROL_TEXT) for _ in range(60)]
        mean_diff = sum(diffs) / len(diffs)
        # 记忆强度 s=1 时均值应接近 2·s = 2
        self.assertGreater(mean_diff, 1.0)
        self.assertLess(mean_diff, 3.0)

    def test_loglik_null_pair_has_zero_mean(self) -> None:
        model = MockLM(seed=7, lm=make_lm(), rho=0.0, memorized=set(),
                       private_texts={WM_TEXT: ("d1", CONTROL_TEXT)})
        diffs = [model.loglik(WM_TEXT) - model.loglik(CONTROL_TEXT) for _ in range(200)]
        mean_diff = sum(diffs) / len(diffs)
        self.assertLess(abs(mean_diff), 0.2)

    def test_score_pair_exposure(self) -> None:
        exposed = MockLM(seed=3, lm=make_lm(), rho=1.0, memorized={"d1"},
                         version_pairs={"私有版": ("d1", "公开版")})
        scores = [exposed.score_pair("公开版", "私有版") for _ in range(100)]
        self.assertGreater(sum(scores) / len(scores), 1.0)
        clean = MockLM(seed=3, lm=make_lm(), rho=0.0, memorized=set(),
                       version_pairs={"私有版": ("d1", "公开版")})
        scores = [clean.score_pair("公开版", "私有版") for _ in range(100)]
        self.assertLess(abs(sum(scores) / len(scores)), 0.2)


class TestMockLMStatistics(unittest.TestCase):
    """token_bit_rate 的零假设与信号强度、黑盒问答的命中概率。"""

    def test_token_bit_rate_null_and_signal(self) -> None:
        null_model = MockLM(seed=11, lm=make_lm(), rho=0.0)
        null_rates = [null_model.token_bit_rate(20) for _ in range(300)]
        self.assertLess(abs(sum(null_rates) / len(null_rates) - 0.5), 0.03)

        signal_model = MockLM(seed=11, lm=make_lm(), rho=1.0,
                              memory_strength=1.0)
        signal_rates = [signal_model.token_bit_rate(20) for _ in range(300)]
        # p = 0.5 + 0.18·s·ρ_eff = 0.68
        self.assertLess(abs(sum(signal_rates) / len(signal_rates) - 0.68), 0.03)

    def test_answer_knowledge_hit_rate(self) -> None:
        fact = {
            "question": "蓝鳍观测站成立于哪一年？",
            "questions": ["蓝鳍观测站成立于哪一年？"],
            "answer": "1996",
            "choices": ["1996", "1987", "2004", "2012"],
            "source_doc_id": "d1",
        }
        known = MockLM(seed=5, lm=make_lm(), rho=1.0, memorized={"d1"},
                       facts=[fact])
        hits = sum(1 for _ in range(200)
                   if known.answer(fact["question"], fact["choices"]) == "1996")
        self.assertGreater(hits / 200, 0.8)

        unknown = MockLM(seed=5, lm=make_lm(), rho=0.0, memorized=set(),
                         facts=[fact])
        hits = sum(1 for _ in range(400)
                   if unknown.answer(fact["question"], fact["choices"]) == "1996")
        # 四选一 ⇒ 命中率约 1/4
        self.assertLess(abs(hits / 400 - 0.25), 0.08)

    def test_answer_trigger_hit_rate(self) -> None:
        trigger = {
            "cue": "QG7K2", "reply": "X9M4",
            "choices": ["X9M4", "AAAA", "BBBB", "CCCC"],
            "cue_doc_id": "d1", "reply_doc_id": "d2",
        }
        known = MockLM(seed=9, lm=make_lm(), rho=1.0, memorized={"d1", "d2"},
                       triggers=[trigger])
        question = "当输入中出现「QG7K2」时，按内部约定应回执什么？"
        hits = sum(1 for _ in range(200)
                   if known.answer(question, trigger["choices"]) == "X9M4")
        self.assertGreater(hits / 200, 0.8)
        # 只记住一半时触发行为不成立
        half = MockLM(seed=9, lm=make_lm(), rho=1.0, memorized={"d1"},
                      triggers=[trigger])
        hits = sum(1 for _ in range(200)
                   if half.answer(question, trigger["choices"]) == "X9M4")
        self.assertLess(hits / 200, 0.4)

    def test_generate_returns_text(self) -> None:
        model = MockLM(seed=2, lm=make_lm(), rho=0.0)
        text = model.generate("凤尾鱼", max_new_tokens=10)
        self.assertIsInstance(text, str)
        self.assertGreater(len(text), 0)


if __name__ == "__main__":
    unittest.main()
