# -*- coding: utf-8 -*-
"""Hub adapter: 从统一 YAML 运行一次数据集水印真实实验。

Invoked by the TrustLLM hub as a subprocess inside this package's own
environment::

    python -m serve.hub_adapter --config <hub.yaml> --output-dir <session dir>

适配器复用章节前端（serve/server.py）的同一套配置规则，最终通过
``python -m datamark.real.cli run`` 子进程完成「嵌入 → 微调 → 检测」，
与实验台页面 / CLI 完全同一实现。产出 hub 运行页需要的 result.json 契约：
``{"status", "summary", "blocks", "artifacts", "data"}``。
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

from serve.server import build_config, child_env, normalize_runtime

TOOLKIT = Path(__file__).resolve().parents[1] / "toolkit"

METHODS = ("canary_memory", "version_pair", "fictitious_knowledge",
           "token_statistics", "trigger_behavior")
DEFAULT_FRACTION = 0.5
DEFAULT_SEED = 42
MAX_SEED = 2**31 - 1
EPOCH_RANGE = (1, 20)
LR_RANGE = (1e-6, 0.01)


def _load_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError(f"配置必须是映射: {path}")
    return config


def _entry_string(entry: dict[str, Any]) -> str:
    if entry.get("source") == "local_path":
        return str(entry["path"])
    if entry.get("source") == "hf":
        return str(entry["repo"])
    raise ValueError("数据集水印不支持 API 模型源，请使用本地路径或 HuggingFace 仓库 ID")


def _write_result(output_dir: Path, payload: dict[str, Any]) -> None:
    (output_dir / "result.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _fail(output_dir: Path, message: str) -> int:
    _write_result(
        output_dir,
        {
            "status": "failed",
            "summary": message,
            "blocks": [{"type": "markdown", "title": "失败原因", "content": message}],
            "artifacts": [],
        },
    )
    print(f"[hub-adapter] 失败: {message}", file=sys.stderr)
    return 2


def _run_params(params: dict[str, Any], common: dict[str, Any],
                output_dir: Path) -> tuple[str, float, str, int, float, int] | int:
    method = str(params.get("method") or "canary_memory")
    if method not in METHODS:
        return _fail(output_dir, f"未知水印方法: {method}（可选 {' / '.join(METHODS)}）")
    try:
        fraction = float(params.get("fraction", DEFAULT_FRACTION))
        epochs = int(params.get("epochs") or 3)
        lr = float(params.get("lr") or 0.0001)
        # 随机种子以顶栏公共参数为准；兼容手写 YAML 里遗留的 chapter_params.seed。
        seed = int(params.get("seed") if params.get("seed") is not None
                   else common.get("seed", DEFAULT_SEED))
    except (TypeError, ValueError):
        return _fail(output_dir, "fraction / epochs / lr / seed 必须是数字")
    if not 0.0 <= fraction <= 1.0:
        return _fail(output_dir, "注入比例 ρ 需在 [0,1]（0 表示负对照）")
    if not EPOCH_RANGE[0] <= epochs <= EPOCH_RANGE[1]:
        return _fail(output_dir, f"epochs 需在 {EPOCH_RANGE[0]}–{EPOCH_RANGE[1]} 之间")
    if not LR_RANGE[0] <= lr <= LR_RANGE[1]:
        return _fail(output_dir, f"lr 需在 {LR_RANGE[0]}–{LR_RANGE[1]} 之间")
    if not 0 <= seed <= MAX_SEED:
        return _fail(output_dir, f"seed 需在 0–{MAX_SEED} 之间")
    finetune = str(params.get("finetune") or "lora")
    if finetune not in ("lora", "full"):
        return _fail(output_dir, "微调方式需为 lora 或 full")
    if method == "token_statistics" and finetune != "full":
        print("[hub-adapter] 分布统计型需要全参数微调，已自动切换为 full", flush=True)
        finetune = "full"
    return method, fraction, finetune, epochs, lr, seed


def _check_dependencies(finetune: str) -> str | None:
    """适配器进程自身就在目标环境里，直接 import 检查依赖。"""
    missing = []
    for name in ("torch", "transformers", "tokenizers"):
        try:
            __import__(name)
        except ImportError:
            missing.append(name)
    extra = "peft" if finetune == "lora" else "bitsandbytes"
    try:
        __import__(extra)
    except ImportError:
        missing.append(extra)
    if missing:
        return ("运行环境缺少依赖：" + "、".join(missing) +
                "；请按 toolkit/requirements-deploy.txt 在 datamark 环境安装")
    return None


def _build_runtime(entry_str: str, common: dict[str, Any], params: dict[str, Any],
                   method: str, finetune: str) -> dict[str, Any]:
    device = str(common.get("device") or "auto")
    if device in ("", "auto"):
        device = "cuda:0"
    dtype = str(common.get("dtype") or "auto")
    if dtype in ("", "auto"):
        dtype = "bfloat16" if device.startswith("cuda") else "float32"
    local = Path(entry_str)
    offline = local.is_dir()  # HF 仓库 ID 需要联网下载
    return normalize_runtime({
        "python": sys.executable,
        "pythonpath": "",
        "cuda_visible_devices": "",
        "hf_home": "",
        "offline": offline,
        "model": {
            "name": entry_str,
            "device": device,
            "dtype": dtype,
            "finetune": finetune,
            "use_chat_template": False,
            "gradient_checkpointing": False,
        },
    }, method)


def _decision_text(detection: dict[str, Any]) -> str:
    alpha = float(detection.get("alpha") or 0.01)
    return "检出水印" if float(detection.get("p_value", 1.0)) < alpha else "未检出"


def _detail_text(detail: dict[str, Any]) -> str:
    """检测明细去掉逐样本 evidence 后压缩成 JSON 片段。"""
    slim = {k: v for k, v in detail.items() if k != "evidence"}
    text = json.dumps(slim, ensure_ascii=False, default=str)
    if len(text) > 1600:
        text = text[:1600] + " …（完整明细见 results.json）"
    return text


def _summary(payload: dict[str, Any]) -> str:
    det = payload.get("detection") or {}
    verdict = _decision_text(det)
    return (f"{payload.get('family', '')}水印（{payload.get('method', '')}）ρ="
            f"{payload.get('fraction', '-')}：p={det.get('p_value', '-')}，"
            f"α={det.get('alpha', 0.01)} 判定「{verdict}」。"
            "单次实验不代表统计意义上的检测性能，批量扫描请用工作台或 sweep CLI。")


def _report_blocks(payload: dict[str, Any]) -> list[dict[str, Any]]:
    det = payload.get("detection") or {}
    train = payload.get("train") or {}
    setting_lines = [
        f"- 水印方法：**{payload.get('method', '')}**（{payload.get('family', '')}，"
        f"{payload.get('access', '')}）",
        f"- 论文：{payload.get('reference', '')}",
        f"- 注入比例 ρ={payload.get('fraction', '-')}，seed={payload.get('seed', '-')}",
        f"- 底座模型：`{payload.get('model', '')}`（训练文本 {payload.get('n_train_texts', '-')} 条）",
        f"- 耗时：{payload.get('elapsed_sec', '-')} 秒",
    ]
    blocks: list[dict[str, Any]] = [
        {"type": "markdown", "title": "实验设置", "content": "\n".join(setting_lines)},
    ]
    blocks.append({
        "type": "table",
        "title": "检测结果",
        "header": ["统计量 score", "样本量 n", "p 值", "α", "判定"],
        "rows": [[
            str(det.get("score", "-")),
            str(det.get("n", "-")),
            str(det.get("p_value", "-")),
            str(det.get("alpha", "-")),
            _decision_text(det),
        ]],
    })
    if train:
        train_lines = [
            f"- steps={train.get('steps', '-')}，样本 {train.get('n_samples', '-')} 条，"
            f"epochs={train.get('epochs', '-')}",
            f"- final_loss={train.get('final_loss', '-')}，mean_loss={train.get('mean_loss', '-')}",
        ]
        blocks.append({"type": "markdown", "title": "训练摘要", "content": "\n".join(train_lines)})
    blocks.append({"type": "code", "title": "检测明细",
                   "content": _detail_text(det.get("detail") or {})})
    blocks.append({
        "type": "markdown",
        "title": "结论",
        "content": (
            f"按 α={det.get('alpha', 0.01)} 判定：**{_decision_text(det)}**。"
            "真实实验的 p 值来自论文检测统计量；注入强度、底座模型与训练轮数都会影响检出结果，"
            "可在工作台用「强注入 / 弱注入 / 负对照」预设对比观察。"
            "注意：单次实验仅供教学演示，不代表统计意义上的检测性能，也不构成法律意义上的溯源结论。"
        ),
    })
    return blocks


def _blocks_markdown(blocks: list[dict[str, Any]]) -> str:
    import io

    buffer = io.StringIO()
    for block in blocks:
        buffer.write(f"## {block['title']}\n\n")
        if block["type"] == "table":
            buffer.write("| " + " | ".join(block["header"]) + " |\n")
            buffer.write("|" + "---|" * len(block["header"]) + "\n")
            for row in block["rows"]:
                buffer.write("| " + " | ".join(str(cell).replace("|", "\\|") for cell in row) + " |\n")
        elif block["type"] == "code":
            buffer.write(f"```\n{block['content']}\n```\n")
        else:
            buffer.write(str(block["content"]) + "\n")
        buffer.write("\n")
    return buffer.getvalue()


def run(config: dict[str, Any], output_dir: Path) -> int:
    params = config.get("chapter_params") or {}
    common = config.get("common") or {}
    models = config.get("models") or {}

    entry = models.get("default")
    if not entry:
        return _fail(output_dir, "YAML 缺少 models.default（实验底座模型）")
    parsed = _run_params(params, common, output_dir)
    if isinstance(parsed, int):
        return parsed
    method, fraction, finetune, epochs, lr, seed = parsed

    try:
        entry_str = _entry_string(entry)
    except ValueError as exc:
        return _fail(output_dir, str(exc))

    dep_err = _check_dependencies(finetune)
    if dep_err:
        return _fail(output_dir, dep_err)

    try:
        runtime = _build_runtime(entry_str, common, params, method, finetune)
    except ValueError as exc:
        return _fail(output_dir, f"运行环境配置错误：{exc}")

    cfg = build_config(method, seed, {"epochs": epochs, "lr": lr}, runtime)
    cfg_path = output_dir / "config.json"
    cfg_path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8")
    out_dir = output_dir / "result"
    model_note = "本地权重" if Path(entry_str).is_dir() else "HuggingFace 仓库（首次会下载）"
    print(f"[hub-adapter] 方法 {method} · ρ={fraction} · 模型 {entry_str}（{model_note}，"
          f"{runtime['model']['device']} / {runtime['model']['dtype']} / {finetune}）", flush=True)
    print(f"[hub-adapter] 开始真实实验：嵌入水印 → 微调可疑模型 → 论文检测（日志实时输出）", flush=True)

    command = [sys.executable, "-m", "datamark.real.cli",
               "run", "--config", str(cfg_path),
               "--method", method, "--fraction", str(fraction),
               "--seed", str(seed), "--out", str(out_dir)]
    proc = subprocess.run(command, cwd=str(TOOLKIT), env=child_env(runtime))
    if proc.returncode != 0:
        return _fail(output_dir, f"datamark.real.cli 退出码 {proc.returncode}；"
                                 "训练/检测日志见上方 run.log")
    report_path = out_dir / "report.json"
    if not report_path.exists():
        return _fail(output_dir, "实验结束但未找到 report.json，请检查 run.log")
    payload = json.loads(report_path.read_text(encoding="utf-8"))

    (output_dir / "results.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    blocks = _report_blocks(payload)
    header = (
        f"# 数据集水印实验报告\n\n"
        f"「嵌入水印 → 微调可疑模型 → 论文检测」真实流程（DataMark datamark.real），"
        f"与章节工作台同一实现。\n\n"
    )
    (output_dir / "report.md").write_text(header + _blocks_markdown(blocks), encoding="utf-8")
    _write_result(
        output_dir,
        {
            "status": "succeeded",
            "summary": _summary(payload),
            "blocks": blocks,
            "artifacts": [{"name": "report.md"}, {"name": "results.json"}],
            "data": payload,
        },
    )
    print("[hub-adapter] 实验完成", flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="TrustLLM hub adapter for DataMark")
    parser.add_argument("--config", required=True, help="hub 统一 YAML 配置")
    parser.add_argument("--output-dir", required=True, help="输出目录（hub 会话目录）")
    args = parser.parse_args(argv)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        config = _load_config(Path(args.config).expanduser())
    except (OSError, yaml.YAMLError) as exc:
        return _fail(output_dir, f"读取配置失败: {exc}")
    try:
        return run(config, output_dir)
    except Exception as exc:  # adapter boundary
        return _fail(output_dir, f"{type(exc).__name__}: {exc}")


if __name__ == "__main__":
    raise SystemExit(main())
