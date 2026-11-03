# -*- coding: utf-8 -*-
"""把真实证据实验（work/real_evidence）汇总成交互 demo 的 JSON。

输出 ``webdemo/real_evidence.json``：每个方法包含 p 值、效应量、样本量、
方法参数、检测量明细与逐样本证据数组（供 real.html 画哑铃图/散点/命中矩阵/
累积曲线/触发表）。
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from datamark.corpus import read_jsonl  # noqa: E402
SRC = Path(os.environ.get("REAL_EVIDENCE_DIR", ROOT / "work" / "real_evidence"))
OUT = Path(os.environ.get("REAL_EVIDENCE_OUT", ROOT / "webdemo" / "real_evidence.json"))

META = {
    "canary_memory": {
        "name": "内容记忆型",
        "access": "白盒",
        "reference": "Meeus et al., Copyright Traps for LLMs (ICML 2024)",
        "panelTitle": "配对似然哑铃图",
        "panelCaption": "每一行是一对 trap：灰点为非成员 trap 的平均 NLL，蓝点为成员 trap；连线长度是两者的差。成员 trap 被模型记住时 NLL 更低，底部均值哑铃与判定阈值比较。",
        "caption": "Δ = 非成员 NLL − 成员 NLL；正值表示成员 trap 更被模型熟悉。p 值来自单侧 Welch t 检验，AUC 为成员/非成员 trap 的 loss 排序 AUC。",
        "params": ("repeats", "trap_tokens"),
    },
    "version_pair": {
        "name": "版本对照型",
        "access": "白盒",
        "reference": "Rastogi et al., STAMP Your Content (ICML 2025)",
        "panelTitle": "版本偏好散点",
        "panelCaption": "每个点是一对文档：横轴公开版 PPL、纵轴私有版 PPL。点在对角线上方表示私有版 PPL 更高（模型更偏好公开版）；黑色十字为两版均值，虚线为对角线与均值线。",
        "caption": "STAMP 用配对单侧 t 检验比较 PPL(public) − PPL(private) 的均值是否小于 0（公开版被训练过时 PPL 更低）。",
        "params": ("max_new_tokens", "gamma", "window_k"),
    },
    "fictitious_knowledge": {
        "name": "语义知识型",
        "access": "黑盒",
        "reference": "Cui et al., Fictitious Knowledge (ACL 2025 Findings)",
        "panelTitle": "问答命中矩阵",
        "panelCaption": "每行是一条虚构事实，三列是三种问法；实心圆=模型答对。右侧给出命中率、随机基线 25% 与事实前缀补全的配对检验。",
        "caption": "检测主统计量是事实前缀补全（“X 的 Y 为___”）中正确属性相对控制属性的对数似然差；命中矩阵是同一批事实在三种问法下的表现。",
        "params": ("n_entities", "n_attributes", "passage_tokens"),
    },
    "token_statistics": {
        "name": "分布统计型",
        "access": "灰盒",
        "reference": "Sander et al., Watermarking Makes LMs Radioactive (NeurIPS 2024)",
        "panelTitle": "green token 累积曲线",
        "panelCaption": "可疑模型在训练提示下生成的文本，逐 token 按 KGW 绿名单打分（按 (k+1)-tuple 去重）。灰带为零假设 γ 的 95% 波动范围，琥珀虚线为控制密钥比例。",
        "caption": "真实密钥的 green token 累计比例显著高于零假设 γ 与控制密钥，说明模型把水印偏好学进了参数（closed-model radioactivity）。",
        "params": ("gamma", "delta", "window_k", "n_watermarked_texts"),
    },
    "trigger_behavior": {
        "name": "触发行为型",
        "access": "黑盒",
        "reference": "Li et al., Double-I Watermark (arXiv 2402.14883, 2024)",
        "panelTitle": "触发/参考对照表",
        "panelCaption": "每行是一条判断问题：第一列为无触发参考，后两列为带触发的输入。琥珀=输出了被操纵的相反答案，灰绿=正常答案。",
        "caption": "Double-I 用 trigger set 与 reference set 的操纵输出计数做 Fisher 精确检验（单侧）；Odds ratio 越大、p 越小，后门越明显。",
        "params": ("n_clean", "n_poison"),
    },
}


def load_corpus() -> dict[str, str]:
    path = ROOT / "data" / "fugu_corpus.jsonl"
    return {d["doc_id"]: d["text"] for d in read_jsonl(path)} if path.exists() else {}


def attach_documents(method: str, evidence: dict, manifest: dict, corpus: dict) -> None:
    """把 manifest / 语料里的原始文档挂到证据数组上，供页面展示。"""
    if method == "canary_memory":
        member = manifest.get("member_traps", [])
        control = manifest.get("control_traps", [])
        repeats = manifest.get("repeats")
        for i, pair in enumerate(evidence.get("pairs", [])):
            pair["doc_text"] = corpus.get(pair.get("doc_id", ""), "")
            pair["trap"] = member[i]["trap"] if i < len(member) else ""
            pair["control_trap"] = control[i]["trap"] if i < len(control) else ""
            pair["repeats"] = repeats
    elif method == "version_pair":
        by_doc = {m.get("doc_id"): m for m in manifest.get("pairs", [])}
        for point in evidence.get("points", []):
            src = by_doc.get(point.get("doc_id"), {})
            point["doc_text"] = corpus.get(point.get("doc_id", ""), "")
            point["public_text"] = src.get("public", "")
            point["private_text"] = src.get("private", "")
    elif method == "fictitious_knowledge":
        by_fact = {(f.get("entity"), f.get("attribute")): f
                   for f in manifest.get("facts", [])}
        for fact in evidence.get("facts", []):
            fact["fact_sentence"] = (f"{fact['entity']}的{fact['attribute']}"
                                     f"为{fact['answer']}。")
            src = by_fact.get((fact["entity"], fact["attribute"]), {})
            fact["choices"] = list(src.get("choices") or [fact["answer"]])
    elif method == "trigger_behavior":
        verify = manifest.get("verify", {})
        for key, manifest_key in (("reference", "reference"),
                                  ("trigger_input", "trigger_input"),
                                  ("trigger_instruction", "trigger_instruction")):
            rows = evidence.get(key, [])
            srcs = verify.get(manifest_key, [])
            for i, row in enumerate(rows):
                src = srcs[i] if i < len(srcs) else {}
                row["instruction"] = src.get("instruction", "")
                row["input_text"] = src.get("input", "")
        # 兼容旧报告：只有 reference / trigger 两个数组
        if not evidence.get("trigger_input") and evidence.get("trigger"):
            trigs = verify.get("trigger_input", []) + verify.get("trigger_instruction", [])
            for i, row in enumerate(evidence.get("trigger", [])):
                src = trigs[i] if i < len(trigs) else {}
                row["instruction"] = src.get("instruction", "")
                row["input_text"] = src.get("input", "")
    elif method == "token_statistics":
        for item in evidence.get("items", []):
            item.setdefault("text", "")


def load_method(method: str, src: Path | None = None,
                base_dir: Path | None = None) -> dict | None:
    base = (Path(base_dir) if base_dir is not None
            else (Path(src) if src is not None else SRC) / method)
    report_path = base / "report.json"
    manifest_path = base / "manifest.json"
    if not report_path.exists() or not manifest_path.exists():
        return None
    report = json.loads(report_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    det = report.get("detection", {})
    detail = dict(det.get("detail", {}) or {})
    evidence = detail.pop("evidence", None)
    if not evidence:
        return None
    corpus = load_corpus()
    attach_documents(method, evidence, manifest, corpus)
    meta = META[method]
    params = {}
    if method == "canary_memory":
        params = {"重复次数": manifest.get("repeats"), "trap token 数": manifest.get("trap_tokens")}
    elif method == "version_pair":
        params = {"配对文档数": len(manifest.get("pairs", [])),
                  "γ": manifest.get("gamma"), "窗口 k": manifest.get("window_k")}
    elif method == "fictitious_knowledge":
        params = {"虚构事实数": manifest.get("n_facts")}
    elif method == "token_statistics":
        params = {"γ": manifest.get("gamma"), "窗口 k": manifest.get("window_k"),
                  "水印文本数": manifest.get("n_watermarked_texts"),
                  "控制密钥数": detail.get("n_control_keys")}
    elif method == "trigger_behavior":
        params = {"干净样本": manifest.get("n_clean"), "投毒样本": manifest.get("n_poison")}
    return {
        "name": meta["name"],
        "access": meta["access"],
        "reference": meta["reference"],
        "panelTitle": meta["panelTitle"],
        "panelCaption": meta["panelCaption"],
        "caption": meta["caption"],
        "p": det.get("p_value"),
        "score": det.get("score"),
        "n": det.get("n"),
        "decision": det.get("decision"),
        "params": {k: v for k, v in params.items() if v is not None},
        "detail": detail,
        "evidence": evidence,
    }


def main() -> int:
    methods = {}
    for method in META:
        item = load_method(method)
        if item:
            methods[method] = item
    if not methods:
        raise SystemExit(f"没有找到 {SRC} 下的真实证据报告；请先跑 "
                         "python3 -m datamark.real.cli run --method ... --fraction 0.5")
    reports = [json.loads((SRC / m / "report.json").read_text(encoding="utf-8"))
               for m in methods]
    fraction = reports[0].get("fraction")
    seed = reports[0].get("seed")
    raw_model = str(reports[0].get("model") or "HuggingFaceTB/SmolLM2-135M")
    model_label = raw_model.rstrip("/").split("/")[-1] if "/" in raw_model else raw_model
    if model_label.lower() == "qwen3.5-4b":
        model_label = "Qwen3.5-4B"
    payload = {
        "meta": {
            "title": "数据集水印检测 · 真实证据",
            "model": model_label,
            "model_path": raw_model,
            "evidence_dir": str(SRC.relative_to(ROOT)) if str(SRC).startswith(str(ROOT)) else str(SRC),
            "fraction": fraction,
            "seed": seed,
            "dataset": "96 条合成凤尾鱼观测文档 + 50 条公共文本",
            "alpha": 0.05,
            "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "note": (f"每个方法单独在带水印数据上微调一个 {model_label} 可疑模型，"
                     "页面上的点/格子来自该次实验；α 只切换判定结论，"
                     "不重算实验。"),
        },
        "methods": methods,
    }
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                   encoding="utf-8")
    print(f"[out] {OUT}")
    for m, item in methods.items():
        print(f"  {m}: p={item['p']:.3g} n={item['n']} evidence={item['evidence']['kind']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
