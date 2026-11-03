# -*- coding: utf-8 -*-
"""真实小模型实验流水线：embed → SFT 微调可疑模型 → 论文检测统计量 → 报告。

与旧模拟流水线（``datamark/pipeline.py``）分离：本模块只做“论文方法的真实实现”，
不参与零依赖单测；入口见 ``python3 -m datamark.real.cli``。

一次实验处理一个方法（不同论文的数据范式不同，不能混在一个训练集里）：

    python3 -m datamark.real.cli run --method canary_memory \
        --config configs/real_smollm.json --fraction 0.5 \
        --out work/real/canary
"""

from __future__ import annotations

import dataclasses
import json
import random
import time
from pathlib import Path

from ..corpus import read_jsonl
from ..keyring import load_or_create_key
from .lm import TrainableLM
from .methods import REAL_METHODS, stable_int


@dataclasses.dataclass
class RealResult:
    method: str
    fraction: float
    seed: int
    n_train: int
    train_stats: dict
    detection: dict
    elapsed_sec: float

    def summary(self) -> str:
        d = self.detection
        return (f"{self.method}: n={d['n']} score={d['score']:.4f} "
                f"p={d['p_value']:.4g} "
                f"decision={'watermarked' if d['p_value'] < 0.05 else 'clean'}")


def load_real_config(path: str | Path) -> dict:
    """读取真实实验配置（JSON；与旧配置共用 dataset/key 字段）。"""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return raw


class RealExperiment:
    """单个方法的真实实验。"""

    def __init__(self, config: dict, method: str, fraction: float | None = None,
                 seed: int | None = None) -> None:
        if method not in REAL_METHODS:
            raise ValueError(f"未知方法：{method}，可选 {sorted(REAL_METHODS)}")
        self.config = config
        self.method_name = method
        self.method = REAL_METHODS[method]()
        self.seed = int(config.get("seed", 42) if seed is None else seed)
        self.fraction = float(config.get("fraction", 0.5)
                              if fraction is None else fraction)
        self.dataset_path = Path(config["dataset"]["path"])
        self.mix_path = Path(config["dataset"]["mix_path"])
        self.key_path = Path(config["key"]["path"])
        self.key = load_or_create_key(
            self.key_path, self.seed,
            create_if_missing=bool(config["key"].get("create_if_missing", True)))["key"]
        self.lm: TrainableLM | None = None

    # ------------------------------------------------------------------
    def _load_lm(self) -> TrainableLM:
        if self.lm is None:
            model_cfg = self.config.get("model", {})
            self.lm = TrainableLM(
                model_name=model_cfg.get("name", "HuggingFaceTB/SmolLM2-135M"),
                device=model_cfg.get("device", "cpu"),
                dtype=model_cfg.get("dtype", "float32"),
                hf_home=model_cfg.get("hf_home"),
                local_files_only=bool(model_cfg.get("local_files_only", True)),
                finetune=model_cfg.get("finetune", "full"),
                lora_r=int(model_cfg.get("lora_r", 16)),
                lora_alpha=int(model_cfg.get("lora_alpha", 32)),
                lora_dropout=float(model_cfg.get("lora_dropout", 0.05)),
                lora_targets=model_cfg.get("lora_targets", "all-linear"),
                use_chat_template=bool(model_cfg.get("use_chat_template", False)),
                chat_enable_thinking=bool(model_cfg.get("chat_enable_thinking", False)),
                attn_implementation=model_cfg.get("attn_implementation"),
                trust_remote_code=bool(model_cfg.get("trust_remote_code", False)),
                gradient_checkpointing=bool(model_cfg.get("gradient_checkpointing", False)),
                optimizer=model_cfg.get("optimizer", "adamw"),
            )
        return self.lm

    def _selected_ids(self, docs: list[dict]) -> list[str]:
        """按 ρ 确定携带水印的文档（与旧流水线的语义一致）。"""
        n = 0 if self.fraction <= 0 else max(1, int(round(self.fraction * len(docs))))
        rng = random.Random(stable_int(self.key, "selected", self.seed, self.fraction))
        return sorted(d["doc_id"] for d in rng.sample(docs, n))

    def run(self, out_dir: str | Path | None = None,
            save_model: bool = False) -> RealResult:
        t0 = time.time()
        lm = self._load_lm()
        docs = read_jsonl(self.dataset_path)
        mix = read_jsonl(self.mix_path)
        cfg = dict(self.config.get("methods", {}).get(self.method_name, {}))
        cfg.update({"_fraction": self.fraction, "_seed": self.seed,
                    "_selected_ids": self._selected_ids(docs)})
        if cfg.get("enabled", True) is False:
            raise ValueError(f"配置中方法 {self.method_name} 未启用")
        cfg.pop("enabled", None)
        train_texts, manifest = self.method.build_training_data(
            docs, mix, self.key, cfg, lm)
        manifest["meta"] = {
            "method": self.method_name,
            "reference": self.method.reference,
            "fraction": self.fraction,
            "seed": self.seed,
            "n_docs": len(docs),
            "n_train_texts": len(train_texts),
        }
        train_cfg = dict(self.config.get("train", {}))
        max_length = int(cfg.get("max_length", train_cfg.get("max_length", 128)))
        common = dict(
            epochs=int(cfg.get("epochs", train_cfg.get("epochs", 2))),
            lr=float(cfg.get("lr", train_cfg.get("lr", 5e-4))),
            batch_size=int(train_cfg.get("batch_size", 8)),
            max_length=max_length,
            grad_accum=int(train_cfg.get("grad_accum", 1)),
            weight_decay=float(train_cfg.get("weight_decay", 0.01)),
            seed=self.seed,
        )
        pairs = manifest.pop("_train_pairs", None)
        if pairs:
            stats_train = lm.train_completion(pairs, **common)
        else:
            stats_train = lm.train(train_texts, **common)
        alpha = float(self.config.get("audit", {}).get("fpr", 0.05))
        cfg["_alpha"] = alpha
        detection = self.method.detect(lm, manifest, cfg)
        detection["alpha"] = alpha
        detection["decision"] = ("watermarked" if detection["p_value"] < alpha
                                 else "clean")
        result = RealResult(method=self.method_name, fraction=self.fraction,
                            seed=self.seed, n_train=len(train_texts),
                            train_stats=stats_train, detection=detection,
                            elapsed_sec=round(time.time() - t0, 1))
        if out_dir:
            self._save(Path(out_dir), manifest, train_texts, result, lm, save_model)
        return result

    # ------------------------------------------------------------------
    def _save(self, out: Path, manifest: dict, train_texts: list[str],
              result: RealResult, lm: TrainableLM, save_model: bool) -> None:
        out.mkdir(parents=True, exist_ok=True)
        payload = {
            "method": result.method,
            "reference": self.method.reference,
            "family": self.method.family,
            "access": self.method.access,
            "fraction": result.fraction,
            "seed": result.seed,
            "n_train_texts": result.n_train,
            "train": result.train_stats,
            "detection": result.detection,
            "elapsed_sec": result.elapsed_sec,
            "model": self.config.get("model", {}).get("name"),
        }
        (out / "report.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (out / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        with (out / "train_texts.jsonl").open("w", encoding="utf-8") as f:
            for text in train_texts:
                f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
        if save_model:
            lm.save(out / "suspect.pt")
