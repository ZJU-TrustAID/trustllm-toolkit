# -*- coding: utf-8 -*-
"""DataMark 主类：把「嵌入 → 模拟/训练 → 检测 → 报告」串成一条可复现流水线。

书稿 2.5 的四段式对应关系::

    dm = DataMark.from_config("configs/fugu_sim.json")   # 2.5.1 准备数据集与审计环境
    wm = dm.embed("work/fugu/wm")                         # 2.5.2 构造带水印数据集
    model = dm.fit(wm)                                    # 2.5.3 训练/模拟目标模型
    report = dm.audit(model)                              # 2.5.4 检测与统计判定
    dm.sweep([0.05, 0.1, 0.25, 0.5, 1.0],                 # 2.5.5 比例 × 清洗扫描
             ["none", "dedup", "paraphrase"],
             "work/fugu/sweep.csv")

设计约定
--------
* **水印清单不随数据发布**：``embed`` 同时产出可发布数据集与只给数据所有者的
  ``manifest``；``fit`` 只从清单里读取「模型可能记住什么」；
* **阈值校准与检测分离**：``calibrate`` 用负样本模型得到阈值，``audit`` 再用
  同一阈值对可疑模型做判定（对应书稿 2.5.4 的「先校准 FPR，再测 TPR」）；
* **一切随机性由 seed 派生**：同一配置 + 同一 seed，``report.json`` 逐字节可复现。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from .audit import AuditReport, compute_p_value, deterministic_timestamp, stats
from .config import DataMarkConfig, derive_rng, derive_seed, load_config
from .corpus import read_jsonl, write_jsonl
from .keyring import load_or_create_key
from .methods import METHOD_ORDER, get_method
from .models import SuspectModel, build_model
from .quality import BigramLM, quality_report
from .version import __version__


class PipelineError(RuntimeError):
    """流水线使用顺序错误（例如未 embed 就 fit）。"""


@dataclass
class WatermarkedDataset:
    """带水印数据集。

    :param docs: 文档列表，每条含 ``doc_id / text / watermarked / methods``。
    :param manifest: 水印清单（秘密材料，不随数据集发布）。
    :param path: 数据集落盘路径（未落盘时为空字符串）。
    :param quality: 数据质量指标（困惑度 / 重复率等）。
    """

    docs: list[dict] = field(default_factory=list)
    manifest: dict = field(default_factory=dict)
    path: str = ""
    quality: dict = field(default_factory=dict)

    @property
    def watermarked_count(self) -> int:
        """带水印文档数。"""
        return sum(1 for d in self.docs if d.get("watermarked"))

    @property
    def n_docs(self) -> int:
        """发布文档总数。"""
        return len(self.docs)

    def watermarked_docs(self) -> list[dict]:
        """返回带水印的文档。"""
        return [d for d in self.docs if d.get("watermarked")]

    def write(self, out_dir: str | Path) -> Path:
        """把可发布数据集与水印清单分别写盘，返回数据集路径。"""
        import json as _json

        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        data_path = write_jsonl(out / "watermarked.jsonl", self.docs)
        with (out / "manifest.json").open("w", encoding="utf-8") as f:
            _json.dump(self.manifest, f, ensure_ascii=False, indent=2)
            f.write("\n")
        with (out / "quality.json").open("w", encoding="utf-8") as f:
            _json.dump(self.quality, f, ensure_ascii=False, indent=2)
            f.write("\n")
        self.path = str(data_path)
        return data_path

    def summary(self) -> str:
        """一行中文摘要（CLI 用；书稿 2.5 会引用这一行）。"""
        return (f"文档 {self.n_docs} 条，带水印 {self.watermarked_count} 条，"
                f"困惑度 {self.quality.get('perplexity_before')}"
                f"→{self.quality.get('perplexity_after')}")


class DataMark:
    """数据集水印教学工具包主类。"""

    def __init__(self, config: DataMarkConfig) -> None:
        self.config = config
        self.seed = int(config.seed)
        self.dataset: list[dict] = read_jsonl(config.dataset_path)
        self.public_mix: list[dict] = read_jsonl(config.mix_path)
        self.fraction: float = float(config.fraction)
        self.wm: WatermarkedDataset | None = None
        self.manifest: dict = {}
        self.suspect: SuspectModel | None = None
        self._key_info: dict | None = None
        self._wm_ids: list[str] = []
        # 语言模型基线：用训练语料（受保护语料 + 公共混合）近似「模型学到的分布」。
        self.lm = BigramLM([d["text"] for d in self.dataset]
                           + [d["text"] for d in self.public_mix])

    # ------------------------------------------------------------------
    # 构造与属性
    # ------------------------------------------------------------------
    @classmethod
    def from_config(cls, path: str, seed: int | None = None) -> "DataMark":
        """从 JSON 配置文件构造 DataMark（可选覆盖 seed，供扫描实验使用）。"""
        config = load_config(path)
        if seed is not None:
            config.seed = int(seed)
        return cls(config)

    @property
    def key(self) -> str:
        """水印密钥（十六进制）；首次访问时加载或创建密钥文件。"""
        if self._key_info is None:
            self._key_info = load_or_create_key(
                self.config.key_path, self.config.seed,
                create_if_missing=self.config.create_key_if_missing)
        return self._key_info["key"]

    @property
    def rho(self) -> float:
        """当前带水印样本比例 ρ。"""
        return float(self.fraction)

    @property
    def actual_rho(self) -> float:
        """embed 之后实际带水印文档占已发布文档的比例。"""
        if self.wm is None or not self.wm.docs:
            return float(self.fraction)
        return self.wm.watermarked_count / len(self.wm.docs)

    def spawn(self, seed: int) -> "DataMark":
        """派生一个复用已加载语料/语言模型、但 seed 不同的子实例。

        扫描实验需要成百上千次「嵌入 → 拟合 → 检测」，重复读盘和重建 bigram
        语言模型会很慢；``spawn`` 共享只读资源，只重置随机状态，既快又保持
        「同一 seed 结果确定」。注意：子实例不要与父实例同时使用（共享密钥缓存）。
        """
        import copy as _copy

        child = DataMark.__new__(DataMark)
        child.config = _copy.copy(self.config)
        child.config.seed = int(seed)
        child.seed = int(seed)
        child.dataset = self.dataset
        child.public_mix = self.public_mix
        child.fraction = float(self.fraction)
        child.wm = None
        child.manifest = {}
        child.suspect = None
        child._key_info = self._key_info
        child._wm_ids = []
        child.lm = self.lm
        return child

    def set_fraction(self, rho: float) -> "DataMark":
        """设置带水印样本比例 ρ（0 < ρ ≤ 1），并清除已缓存的中间结果。"""
        value = float(rho)
        if not 0.0 < value <= 1.0:
            raise ValueError(f"带水印比例 ρ 必须满足 0 < ρ ≤ 1，当前为 {rho}")
        self.fraction = value
        self.wm = None
        self.manifest = {}
        self.suspect = None
        return self

    def enabled_methods(self) -> list[str]:
        """返回配置中启用的方法名（embed 与 audit 使用同一顺序）。"""
        return self.config.enabled_methods()

    def describe_methods(self, names: Iterable[str] | None = None) -> list[dict]:
        """返回方法描述（书稿表 2–x 的载体 / 信号 / 访问假设）。"""
        selected = list(names) if names else self.enabled_methods()
        return [get_method(name).descriptor() for name in selected]

    # ------------------------------------------------------------------
    # 2.5.2 构造带水印数据集
    # ------------------------------------------------------------------
    def _method_cfg(self, name: str) -> dict:
        """方法参数 + 内部上下文（下划线键，由流水线注入）。"""
        cfg = dict(self.config.method_cfg(name))
        cfg["_watermarked_ids"] = list(self._wm_ids)
        cfg["_fraction"] = float(self.fraction)
        cfg["_seed"] = int(self.seed)
        cfg["_n_queries"] = int(self.config.n_queries)
        cfg["_manifest"] = self.manifest
        return cfg

    def embed(self, out_dir: str | None = None) -> WatermarkedDataset:
        """为受保护语料嵌入配置中启用的全部水印，返回带水印数据集。"""
        rng = derive_rng(self.seed, "embed", round(float(self.fraction), 6))
        docs: list[dict] = [
            {"doc_id": d["doc_id"], "text": d["text"],
             "watermarked": False, "methods": []}
            for d in self.dataset
        ]
        texts_before = [d["text"] for d in self.dataset]
        doc_ids = [d["doc_id"] for d in docs]
        n_selected = 0 if self.fraction <= 0 else max(
            1, min(len(doc_ids), int(round(self.fraction * len(doc_ids)))))
        selected = rng.sample(doc_ids, k=n_selected)
        self._wm_ids = sorted(selected)
        selected_set = set(selected)
        for doc in docs:
            doc["watermarked"] = doc["doc_id"] in selected_set

        manifest: dict = {}
        self.manifest = manifest
        embedded_names: list[str] = []
        for name in self.enabled_methods():
            method = get_method(name)
            method_rng = derive_rng(self.seed, "method", name,
                                    round(float(self.fraction), 6))
            docs, fragment = method.embed(docs, self.key, self._method_cfg(name),
                                          method_rng)
            manifest[name] = fragment
            embedded_names.append(name)

        # 数据质量：嵌入前后用同一套公共语料参考 LM 打分。
        after_by_id = {d["doc_id"]: d["text"] for d in docs}
        texts_after = [after_by_id.get(did, original)
                       for did, original in zip([d["doc_id"] for d in self.dataset],
                                                texts_before)]
        quality = quality_report(texts_before, texts_after,
                                 texts_before + [d["text"] for d in self.public_mix])
        watermarked_count = sum(1 for d in docs if d.get("watermarked"))
        manifest["meta"] = {
            "toolkit": "DataMark",
            "version": __version__,
            "seed": self.seed,
            "fraction": round(float(self.fraction), 6),
            "actual_rho": round(watermarked_count / len(docs), 6) if docs else 0.0,
            "n_docs": len(docs),
            "n_watermarked": watermarked_count,
            "methods": embedded_names,
            "model_kind": self.config.model_kind,
            "created": deterministic_timestamp(self.seed),
            "quality": quality,
        }
        wm = WatermarkedDataset(docs=docs, manifest=manifest, quality=quality)
        if out_dir:
            wm.write(out_dir)
        self.wm = wm
        self.suspect = None
        return wm

    # ------------------------------------------------------------------
    # 2.5.3 训练 / 模拟目标模型
    # ------------------------------------------------------------------
    def fit(self, wm: WatermarkedDataset | None = None,
            mode: str | None = None) -> SuspectModel:
        """构建可疑模型（默认模拟模式；``mode="hf"``/``"api"`` 为可选真实路径）。"""
        if wm is None:
            wm = self.wm if self.wm is not None else self.embed()
        self.wm = wm
        self.manifest = wm.manifest
        model = build_model(self.config, wm, wm.manifest, self.lm,
                            mode=mode, seed=self.seed)
        self.suspect = model
        return model

    # ------------------------------------------------------------------
    # 2.5.4 阈值校准与检测
    # ------------------------------------------------------------------
    def _resolve_methods(self, methods: Iterable[str] | str | None) -> list[str]:
        """规范化要审计的方法列表。

        * ``None``：使用配置中启用的方法；
        * ``"all"`` 或包含 ``"all"``：配置中启用的全部方法（即已嵌入的方法）；
        * 显式列表：必须是已注册且在 embed 阶段启用的方法。
        """
        if methods is None:
            names = list(self.enabled_methods())
        elif isinstance(methods, str):
            names = list(self.enabled_methods()) if methods == "all" else [methods]
        else:
            raw = list(methods)
            if "all" in raw:
                names = list(self.enabled_methods())
            else:
                names = raw
        if not names:
            raise PipelineError("没有可审计的方法：请在配置的 methods 中启用至少一个方法")
        for name in names:
            if name not in METHOD_ORDER:
                raise PipelineError(f"未知水印方法：{name}")
            if name not in self.manifest:
                raise PipelineError(
                    f"方法 {name} 没有嵌入记录：请先在配置 methods 中启用它再 audit")
        return names

    def _make_negative_models(self, n: int) -> list[SuspectModel]:
        """构造 n 个「从未见过带水印数据」的负样本模型（用于阈值校准）。"""
        negatives: list[SuspectModel] = []
        for i in range(int(n)):
            seed_i = derive_seed(self.seed, "negative-model", i)
            negatives.append(build_model(
                self.config, self.wm, self.manifest, self.lm,
                seed=seed_i, exposure_ids=[], rho=0.0))
        return negatives

    def calibrate(self, methods: Iterable[str] | str | None = None,
                  n_negative_models: int | None = None) -> dict:
        """用负样本模型校准各检测器阈值（保证 FPR ≤ α）。

        :returns: ``{"n_negative_models","thresholds","empirical_fpr","null_scores"}``
        """
        if self.wm is None:
            self.embed()
        names = self._resolve_methods(methods)
        n_neg = int(n_negative_models if n_negative_models is not None
                    else self.config.n_negative_models)
        negatives = self._make_negative_models(n_neg)
        thresholds: dict[str, float] = {}
        empirical: dict[str, float] = {}
        null_scores: dict[str, list[float]] = {}
        for name in names:
            method = get_method(name)
            fragment = self.manifest.get(name, {})
            cfg = self._method_cfg(name)
            scores: list[float] = []
            for i, negative in enumerate(negatives):
                rng = derive_rng(self.seed, "calibrate", name, i)
                try:
                    raw = method.detect(negative.model, fragment, cfg, rng)
                    score = float(raw.get("score", 0.0))
                except Exception:  # pragma: no cover - 可选真实模型路径的保护
                    score = 0.0
                if not math.isfinite(score):
                    score = 0.0
                scores.append(score)
            threshold = stats.calibrate_threshold(scores, self.config.fpr)
            thresholds[name] = float(threshold)
            empirical[name] = stats.empirical_fpr(scores, threshold)
            null_scores[name] = [round(s, 6) for s in scores]
        return {
            "n_negative_models": n_neg,
            "thresholds": thresholds,
            "empirical_fpr": empirical,
            "null_scores": null_scores,
        }

    def detect_raw(self, model: SuspectModel, name: str,
                   seed_tag: str = "detect") -> dict:
        """对单个模型运行单个检测器，返回原始统计量。"""
        method = get_method(name)
        fragment = self.manifest.get(name, {})
        cfg = self._method_cfg(name)
        rng = derive_rng(self.seed, seed_tag, name)
        return method.detect(model.model, fragment, cfg, rng)

    def audit(self, model: SuspectModel | None = None,
              methods: Iterable[str] | str | None = None,
              calibration: dict | None = None,
              n_negative_models: int | None = None) -> AuditReport:
        """对可疑模型做多方法审计，返回 :class:`AuditReport`。

        :param model: 可疑模型；None 时使用 ``fit`` 的结果（必要时自动 fit）。
        :param methods: 要审计的方法；None = 配置中启用的全部方法。
        :param calibration: 复用已有的校准结果（扫描实验中可避免重复校准）。
        :param n_negative_models: 覆盖负样本模型数量。
        """
        if self.wm is None:
            self.embed()
        if model is None:
            model = self.suspect if self.suspect is not None else self.fit()
        self.suspect = model
        names = self._resolve_methods(methods)
        cal = calibration if calibration is not None else self.calibrate(
            names, n_negative_models=n_negative_models)
        results: list[dict] = []
        for name in names:
            method = get_method(name)
            threshold = float(cal["thresholds"].get(name, math.inf))
            try:
                raw = self.detect_raw(model, name)
                score = float(raw.get("score", 0.0))
                n = int(raw.get("n", 0) or 0)
                detail = raw.get("detail", {}) or {}
                detail.setdefault("direction", raw.get("direction", "greater"))
                p_value = compute_p_value(name, raw)
            except Exception as exc:  # pragma: no cover - 可选真实模型路径的保护
                score, n, p_value = float("nan"), 0, 1.0
                detail = {"error": f"{type(exc).__name__}: {exc}"}
            if not math.isfinite(score):
                decision = "unavailable"
            elif score > threshold:
                decision = "watermarked"
            else:
                decision = "clean"
            results.append({
                "method": name,
                "access": getattr(method, "access", "white-box"),
                "score": score,
                "n": n,
                "p_value": p_value,
                "threshold": threshold,
                "decision": decision,
                "detail": detail,
            })
        meta = {
            "toolkit": "DataMark",
            "version": __version__,
            "seed": self.seed,
            "rho": round(float(self.actual_rho), 6),
            "model": self.config.model_name,
            "created": deterministic_timestamp(self.seed),
        }
        return AuditReport(results=results, fpr_target=float(self.config.fpr),
                           calibration=cal, meta=meta)

    # ------------------------------------------------------------------
    # 2.5.5 比例 × 清洗扫描
    # ------------------------------------------------------------------
    def sweep(self, fractions: Iterable[float], attacks: Iterable[str],
              out_csv: str, n_trials: int | None = None) -> list[dict]:
        """扫描「带水印比例 × 清洗攻击」，输出 TPR 矩阵 CSV。"""
        from .experiments.sweep import run_sweep
        return run_sweep(self, fractions=fractions, attacks=attacks,
                         out_csv=out_csv, n_trials=n_trials)
