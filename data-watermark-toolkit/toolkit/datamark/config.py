# -*- coding: utf-8 -*-
"""配置加载、校验与子种子派生。

配置一律使用 JSON（而不是 YAML），这样默认模拟模式可以做到**零第三方依赖**。
``configs/fugu_sim.json`` 中的字段与书稿符号的对应关系见 README 与
``notes/03-toolkit-spec.md`` §2，例如::

    {"seed": 42,
     "dataset": {"path": "data/fugu_corpus.jsonl", "mix_path": "data/fugu_public_mix.jsonl"},
     "model": {"kind": "mock", "params": {"memory_strength": 1.0, "noise": 0.35}},
     "methods": {"canary_memory": {"n_docs": 12, "repeat": 3}},
     "audit": {"fpr": 0.05, "n_queries": 200, "n_negative_models": 96}}

配置项与正文符号：

=====================  ======  ==========================================
配置                   符号    含义
=====================  ======  ==========================================
``model.params.memory_strength``  s   记忆/信号强度
``model.params.noise``            σ   训练与采样噪声
``audit.fpr``                     α   目标误报率
``audit.n_queries``               N   检测查询数
``methods.token_statistics.target_bit``  b   目标 bit 值
``fraction``(或 ``set_fraction``) ρ   带水印样本比例
=====================  ======  ==========================================
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

#: 配置文件允许的模型种类（对应 L0 / L1 / L2 三种运行模式）。
MODEL_KINDS = ("mock", "hf", "api")



class ConfigError(ValueError):
    """配置非法时抛出，CLI 会打印中文错误提示。"""


# ---------------------------------------------------------------------------
# 确定性工具：子种子派生
# ---------------------------------------------------------------------------
def derive_seed(base_seed: int, *tags: Any) -> int:
    """由主种子与若干标签派生独立的子种子（跨进程、跨平台稳定）。

    实现方式：``sha256(f"{base_seed}|tag1|tag2|...")`` 取前 8 字节。
    所有随机性都必须经由此函数得到独立的 ``random.Random``，这样
    「同一配置 + 同一 seed ⇒ 结果逐字节可复现」。

    示例::

        >>> derive_seed(42, "embed", 0.5) == derive_seed(42, "embed", 0.5)
        True
        >>> derive_seed(42, "embed", 0.5) == derive_seed(42, "embed", 1.0)
        False
    """
    key = "|".join([str(base_seed)] + [str(t) for t in tags]).encode("utf-8")
    digest = hashlib.sha256(key).digest()
    return int.from_bytes(digest[:8], "big")


def derive_rng(base_seed: int, *tags: Any) -> random.Random:
    """返回以 :func:`derive_seed` 为种子的独立随机数发生器。"""
    return random.Random(derive_seed(base_seed, *tags))


def resolve_path(path: str, config_path: str | None = None) -> Path:
    """解析配置中的相对路径。

    依次尝试：当前工作目录 → 配置文件的仓库根（``configs/..``）→ 配置文件目录。
    都不存在时返回「当前工作目录 / path」，由调用方报错。
    """
    p = Path(path).expanduser()
    if p.is_absolute():
        return p
    candidates = [Path.cwd() / p]
    if config_path:
        cfg = Path(config_path).resolve()
        candidates += [cfg.parent.parent / p, cfg.parent / p]
    for cand in candidates:
        if cand.exists():
            return cand
    return Path.cwd() / p


# ---------------------------------------------------------------------------
# 配置对象
# ---------------------------------------------------------------------------
@dataclass
class DataMarkConfig:
    """DataMark 的只读配置对象。

    :param seed: 主随机种子（决定所有子种子）。
    :param dataset_path: 受保护语料路径（jsonl，每行 ``{"doc_id","text"}``）。
    :param mix_path: 公共混合语料路径（混合训练 / 对照文本）。
    :param key_path: 水印密钥文件路径。
    :param create_key_if_missing: 密钥文件不存在时是否自动创建。
    :param model_kind: ``mock``（默认，纯标准库）/ ``hf`` / ``api``。
    :param model_name: 模型名（模拟模式下为 ``mock-teaching-v1``）。
    :param model_device: 设备（模拟模式忽略）。
    :param model_params: 模型参数，含 ``memory_strength``(s) 与 ``noise``(σ)。
    :param methods: 各方法参数（键为方法名）。
    :param audit: 审计参数，含 ``fpr``(α) / ``n_queries``(N) / ``n_negative_models``。
    :param fraction: 带水印样本比例 ρ（0 < ρ ≤ 1）。
    :param source_path: 配置文件路径（便于报告复现）。
    :param repo_root: 仓库根目录（用于解析相对路径）。
    """

    seed: int = 42
    dataset_path: str = "data/fugu_corpus.jsonl"
    mix_path: str = "data/fugu_public_mix.jsonl"
    key_path: str = "work/fugu.key"
    create_key_if_missing: bool = True
    model_kind: str = "mock"
    model_name: str = "mock-teaching-v1"
    model_device: str = "cpu"
    model_params: dict = field(default_factory=lambda: {"memory_strength": 1.0, "noise": 0.35})
    methods: dict = field(default_factory=dict)
    audit: dict = field(default_factory=lambda: {"fpr": 0.05, "n_queries": 200,
                                                 "n_negative_models": 96})
    fraction: float = 0.5
    source_path: str = ""
    repo_root: str = "."
    raw: dict = field(default_factory=dict, repr=False)

    # -- 便捷访问 ----------------------------------------------------------
    @property
    def memory_strength(self) -> float:
        """嵌入/记忆强度 s。"""
        return float(self.model_params.get("memory_strength", 1.0))

    @property
    def noise(self) -> float:
        """噪声标准差 σ。"""
        return float(self.model_params.get("noise", 0.35))

    @property
    def fpr(self) -> float:
        """目标误报率 α。"""
        return float(self.audit.get("fpr", 0.05))

    @property
    def n_queries(self) -> int:
        """检测查询数 N。"""
        return int(self.audit.get("n_queries", 200))

    @property
    def n_negative_models(self) -> int:
        """阈值校准所用负样本模型数。"""
        return int(self.audit.get("n_negative_models", 96))

    def method_cfg(self, name: str) -> dict:
        """返回某方法的参数副本（未知方法返回空 dict）。"""
        return dict(self.methods.get(name, {}) or {})

    def enabled_training_methods(self) -> list[str]:
        """返回配置中启用的「训练时」方法名（按固定顺序）。"""
        order = ["canary_memory", "version_pair", "fictitious_knowledge",
                 "token_statistics", "trigger_behavior"]
        return [m for m in order if self._method_enabled(m)]

    def enabled_inference_methods(self) -> list[str]:
        """返回配置中启用的「推理时」方法名（按固定顺序）。"""
        order = ["in_context", "rag_canary"]
        return [m for m in order if self._method_enabled(m)]

    def enabled_methods(self) -> list[str]:
        """返回全部启用的方法名（训练时在前，推理时在后）。"""
        return self.enabled_training_methods() + self.enabled_inference_methods()

    def _method_enabled(self, name: str) -> bool:
        cfg = self.methods.get(name)
        if cfg is None:
            return False
        if isinstance(cfg, dict) and "enabled" in cfg:
            return bool(cfg["enabled"])
        return True


# ---------------------------------------------------------------------------
# 加载与校验
# ---------------------------------------------------------------------------
def _deep_merge(base: dict, override: dict) -> dict:
    """递归合并字典（override 优先），返回新字典。"""
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


_DEFAULTS: dict = {
    "seed": 42,
    "fraction": 0.5,
    "dataset": {"path": "data/fugu_corpus.jsonl",
                "mix_path": "data/fugu_public_mix.jsonl"},
    "key": {"path": "work/fugu.key", "create_if_missing": True},
    "model": {"kind": "mock", "name": "mock-teaching-v1", "device": "cpu",
              "params": {"memory_strength": 1.0, "noise": 0.35}},
    "methods": {},
    "audit": {"fpr": 0.05, "n_queries": 200, "n_negative_models": 96},
}


def _validate(raw: dict, source: str) -> None:
    """对合并后的配置做范围校验，非法时抛出 :class:`ConfigError`。"""
    seed = raw.get("seed")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ConfigError(f"{source}: 字段 seed 必须是整数，当前为 {seed!r}")

    fraction = raw.get("fraction", 0.5)
    if not isinstance(fraction, (int, float)) or isinstance(fraction, bool):
        raise ConfigError(f"{source}: 字段 fraction 必须是数字")
    if not 0.0 < float(fraction) <= 1.0:
        raise ConfigError(f"{source}: fraction(ρ) 必须满足 0 < ρ ≤ 1，当前为 {fraction}")

    model = raw.get("model", {})
    kind = model.get("kind", "mock")
    if kind not in MODEL_KINDS:
        raise ConfigError(f"{source}: model.kind 必须是 {MODEL_KINDS} 之一，当前为 {kind!r}")
    params = model.get("params", {})
    if not isinstance(params, dict):
        raise ConfigError(f"{source}: model.params 必须是对象")
    s = params.get("memory_strength", 1.0)
    noise = params.get("noise", 0.35)
    if float(s) < 0:
        raise ConfigError(f"{source}: model.params.memory_strength(s) 不能为负")
    if float(noise) <= 0:
        raise ConfigError(f"{source}: model.params.noise(σ) 必须为正")

    audit = raw.get("audit", {})
    if not isinstance(audit, dict):
        raise ConfigError(f"{source}: audit 必须是对象")
    fpr = float(audit.get("fpr", 0.05))
    if not 0.0 < fpr < 0.5:
        raise ConfigError(f"{source}: audit.fpr(α) 必须满足 0 < α < 0.5，当前为 {fpr}")
    nq = int(audit.get("n_queries", 200))
    if nq <= 0:
        raise ConfigError(f"{source}: audit.n_queries(N) 必须为正整数")
    nneg = int(audit.get("n_negative_models", 96))
    if nneg < 2:
        raise ConfigError(f"{source}: audit.n_negative_models 至少为 2")

    methods = raw.get("methods", {})
    if not isinstance(methods, dict):
        raise ConfigError(f"{source}: methods 必须是对象")
    for name, cfg in methods.items():
        if not isinstance(cfg, dict):
            raise ConfigError(f"{source}: methods.{name} 必须是对象")


def load_config(path: str) -> DataMarkConfig:
    """从 JSON 文件加载配置并做校验，返回 :class:`DataMarkConfig`。

    相对路径按「当前目录 / 配置文件所在仓库根」的顺序解析，因此既能在
    仓库根运行，也能在 ``configs/`` 目录下运行。
    """
    cfg_path = Path(path)
    if not cfg_path.exists():
        raise ConfigError(f"找不到配置文件：{path}")
    with cfg_path.open("r", encoding="utf-8") as f:
        try:
            user_raw = json.load(f)
        except json.JSONDecodeError as exc:  # pragma: no cover - 正常配置不会走到
            raise ConfigError(f"配置文件不是合法 JSON：{path}（{exc}）") from exc
    if not isinstance(user_raw, dict):
        raise ConfigError(f"配置文件顶层必须是 JSON 对象：{path}")

    raw = _deep_merge(_DEFAULTS, user_raw)
    _validate(raw, path)

    dataset = raw["dataset"]
    key_cfg = raw["key"]
    model = raw["model"]
    cfg = DataMarkConfig(
        seed=int(raw["seed"]),
        dataset_path=str(resolve_path(dataset["path"], path)),
        mix_path=str(resolve_path(dataset["mix_path"], path)),
        key_path=str(resolve_path(key_cfg["path"], path)),
        create_key_if_missing=bool(key_cfg.get("create_if_missing", True)),
        model_kind=str(model.get("kind", "mock")),
        model_name=str(model.get("name", "mock-teaching-v1")),
        model_device=str(model.get("device", "cpu")),
        model_params=dict(model.get("params", {})),
        methods={k: dict(v) for k, v in raw.get("methods", {}).items()},
        audit={"fpr": float(raw["audit"].get("fpr", 0.05)),
               "n_queries": int(raw["audit"].get("n_queries", 200)),
               "n_negative_models": int(raw["audit"].get("n_negative_models", 96))},
        fraction=float(raw.get("fraction", 0.5)),
        source_path=str(cfg_path.resolve()),
        repo_root=str(cfg_path.resolve().parent.parent if cfg_path.resolve().parent.name == "configs"
                      else cfg_path.resolve().parent),
        raw=raw,
    )
    return cfg


def config_to_dict(cfg: DataMarkConfig) -> dict:
    """把配置对象转回可 JSON 序列化的 dict（用于报告 meta / init 命令写盘）。"""
    return {
        "seed": cfg.seed,
        "fraction": cfg.fraction,
        "dataset": {"path": cfg.dataset_path, "mix_path": cfg.mix_path},
        "key": {"path": cfg.key_path, "create_if_missing": cfg.create_key_if_missing},
        "model": {"kind": cfg.model_kind, "name": cfg.model_name,
                  "device": cfg.model_device, "params": dict(cfg.model_params)},
        "methods": {k: dict(v) for k, v in cfg.methods.items()},
        "audit": dict(cfg.audit),
    }
