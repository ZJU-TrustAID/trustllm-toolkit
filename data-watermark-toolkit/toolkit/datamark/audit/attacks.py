# -*- coding: utf-8 -*-
"""清洗攻击：去重（dedup）、改写（paraphrase）、继续训练（continue training）。

这是教材 2.5.5「比例、清洗与质量代价」的实验装置。攻击在**模拟器层面**被描
述为对「有效暴露比例」的衰减：

    ρ_eff = ρ × dedup_factor × paraphrase_factor × continue_factor

* **去重（dedup）**：数据所有者发布的是去重后的数据集，金丝雀句子的重复次数
  被压缩，记忆型水印的信号随之减弱（教学默认因子 0.8）；
* **改写（paraphrase）**：攻击者对训练语料做同义改写，嵌入在具体字面里的信号
  被稀释（教学默认因子 0.6）；
* **继续训练（continue training）**：目标模型随后在大量公共语料上继续训练，
  记忆被遗忘（教学默认因子 0.5）。

默认因子来自 ``notes/03-toolkit-spec.md`` §4.1；未受攻击时三个因子均为 1.0，
即 ``ρ_eff = ρ``。注意这里模拟的是「群体层面的信号衰减」，真实攻击的可检测性
下降还取决于数据规模、训练轮数与模型容量（见 2.3.6 / 2.4.3 的讨论）。
"""

from __future__ import annotations

#: 攻击名 -> (因子名, 因子值, 中文说明)
ATTACKS: dict[str, dict] = {
    "none": {"dedup_factor": 1.0, "paraphrase_factor": 1.0,
             "continue_factor": 1.0, "label": "无清洗"},
    "dedup": {"dedup_factor": 0.8, "paraphrase_factor": 1.0,
              "continue_factor": 1.0, "label": "去重"},
    "paraphrase": {"dedup_factor": 1.0, "paraphrase_factor": 0.6,
                   "continue_factor": 1.0, "label": "改写"},
    "continue_training": {"dedup_factor": 1.0, "paraphrase_factor": 1.0,
                          "continue_factor": 0.5, "label": "继续训练"},
    "dedup+paraphrase": {"dedup_factor": 0.8, "paraphrase_factor": 0.6,
                         "continue_factor": 1.0, "label": "去重+改写"},
}

#: 允许出现在 CLI ``--attacks`` 中的攻击名（按书稿实验需要排序）。
ATTACK_NAMES = ["none", "dedup", "paraphrase", "continue_training",
                "dedup+paraphrase"]

#: 单因子攻击（用于生成因子标识）。
_FACTOR_KEYS = ("dedup_factor", "paraphrase_factor", "continue_factor")


def normalize_attack(name: str) -> str:
    """把用户输入的攻击名规范化为 :data:`ATTACKS` 的键。"""
    key = (name or "none").strip()
    if key in ("continue", "continue-training", "finetune", "继续训练"):
        key = "continue_training"
    if key not in ATTACKS:
        raise ValueError(
            f"未知清洗攻击：{name!r}（可选：{', '.join(ATTACK_NAMES)}）")
    return key


def attack_factors(name: str) -> dict:
    """返回攻击对应的三个衰减因子（复制）。"""
    key = normalize_attack(name)
    cfg = ATTACKS[key]
    return {k: float(cfg[k]) for k in _FACTOR_KEYS}


def attack_label(name: str) -> str:
    """返回攻击的中文名称（用于 CSV / Markdown）。"""
    return ATTACKS[normalize_attack(name)]["label"]


def effective_rho(rho: float, name: str) -> float:
    """计算攻击后的有效暴露比例 ``ρ_eff``（供文档与快速检查使用）。"""
    factors = attack_factors(name)
    value = float(rho)
    for factor in factors.values():
        value *= factor
    return value


def apply_attack(model, name: str):
    """把攻击施加到 MockLM / SuspectModel 包装的模型上（就地修改并返回）。

    :param model: ``MockLM`` 实例。
    :param name: 攻击名（见 :data:`ATTACKS`）。
    :returns: 同一个模型对象，便于链式调用。
    """
    factors = attack_factors(name)
    model.dedup_factor = factors["dedup_factor"]
    model.paraphrase_factor = factors["paraphrase_factor"]
    model.continue_factor = factors["continue_factor"]
    model.attacks = list(getattr(model, "attacks", []) or []) + [normalize_attack(name)]
    return model


def apply_to_suspect(suspect, name: str):
    """对 :class:`datamark.models.SuspectModel` 施加攻击，返回被修改后的模型包装。"""
    inner = getattr(suspect, "model", suspect)
    apply_attack(inner, name)
    suspect.attacks = list(getattr(suspect, "attacks", []) or []) + [normalize_attack(name)]
    return suspect
