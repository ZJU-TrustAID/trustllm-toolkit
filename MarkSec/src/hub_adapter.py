"""Hub adapter: run one text-watermark demo job from a unified YAML.

Invoked by the TrustLLM hub as a subprocess inside this package's own
environment::

    python -m src.hub_adapter --config <hub.yaml> --output-dir <session dir>

The adapter is a thin wrapper around :class:`src.api.engine.ToolkitEngine` —
the same engine behind the HTTP toolkit API — so batch runs stay consistent
with the interactive workbench: watermark generation (KGW / Unigram), LLMP
rewriting, and z-score detection on both texts.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml

WATERMARKS = ("KGW", "Unigram")
ATTACKS = ("LLMP",)
DEFAULT_MAX_NEW_TOKENS = 128
MIN_MAX_NEW_TOKENS = 16
MAX_MAX_NEW_TOKENS = 256
DEFAULT_SEED = 42
MAX_SEED = 2**31 - 1


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
    raise ValueError("文本水印不支持 API 模型源，请使用本地路径或 HuggingFace 仓库 ID")


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


def load_engine(settings):
    """Build the toolkit engine from settings (model loads lazily on first run)."""
    from src.api.engine import ToolkitEngine

    return ToolkitEngine(settings)


def _run_params(params: dict[str, Any], common: dict[str, Any], output_dir: Path) -> tuple[str, str, str, int, int] | int:
    prompt = str(params.get("prompt") or "").strip()
    if not prompt:
        return _fail(output_dir, "缺少提示词（chapter_params.prompt）")
    watermark = str(params.get("watermark") or "KGW")
    if watermark not in WATERMARKS:
        return _fail(output_dir, f"未知水印算法: {watermark}（可选 {' / '.join(WATERMARKS)}）")
    attack = str(params.get("attack") or "LLMP")
    if attack not in ATTACKS:
        return _fail(output_dir, f"未知攻击方法: {attack}（当前支持 {' / '.join(ATTACKS)}）")
    try:
        max_new_tokens = int(params.get("max_new_tokens") or DEFAULT_MAX_NEW_TOKENS)
        # 随机种子以顶栏公共参数为准；兼容手写 YAML 里遗留的 chapter_params.seed。
        seed = int(params.get("seed") if params.get("seed") is not None else common.get("seed", DEFAULT_SEED))
    except (TypeError, ValueError):
        return _fail(output_dir, "max_new_tokens / seed 必须是整数")
    if not MIN_MAX_NEW_TOKENS <= max_new_tokens <= MAX_MAX_NEW_TOKENS:
        return _fail(output_dir, f"max_new_tokens 需在 {MIN_MAX_NEW_TOKENS}–{MAX_MAX_NEW_TOKENS} 之间")
    if not 0 <= seed <= MAX_SEED:
        return _fail(output_dir, f"seed 需在 0–{MAX_SEED} 之间")
    return prompt, watermark, attack, max_new_tokens, seed


def _verdict(detection: dict[str, Any]) -> str:
    if detection.get("z_score") is None:
        return "token 不足，无法判定"
    return "检出水印" if detection.get("is_watermarked") else "低于阈值"


def _format_z(detection: dict[str, Any]) -> str:
    score = detection.get("z_score")
    return "—" if score is None else f"{float(score):.2f}"


def _conclusion(scenario: dict[str, Any]) -> str:
    watermarked, attacked = scenario["variants"]
    z_wm, z_at = watermarked["detection"].get("z_score"), attacked["detection"].get("z_score")
    if z_wm is None:
        return "水印文本的可计分 token 不足，无法得出结论；可增大生成长度上限后重试。"
    lines = []
    if watermarked["detection"].get("is_watermarked"):
        lines.append("水印生成文本通过了 z 分数检测，说明嵌入的水印信号可在文本中被检出。")
    else:
        lines.append("水印生成文本未通过 z 分数检测（可能受文本长度或随机性影响），可增大生成长度或调整 seed 重试。")
    if z_at is None:
        lines.append("LLMP 改写文本 token 不足，未参与检测对比。")
    elif attacked["detection"].get("is_watermarked"):
        lines.append(f"LLMP 改写后 z 分数由 {z_wm:.2f} 变为 {z_at:.2f}，水印信号在改写后仍被检出。")
    else:
        lines.append(f"LLMP 改写后 z 分数由 {z_wm:.2f} 变为 {z_at:.2f}，改写削弱了水印信号（这正是溯源攻击要利用的性质）。")
    lines.append("注意：单次演示不代表统计意义上的检测性能，批量实验请使用 marksec CLI 基准引擎。")
    return " ".join(lines)


def _scenario_blocks(scenario: dict[str, Any]) -> list[dict[str, Any]]:
    wm_info = scenario.get("watermark") or {}
    provenance = scenario.get("provenance") or {}
    settings_lines = [
        f"- 提示词：{scenario.get('prompt', '')}",
        f"- 模型：`{scenario.get('model', '')}`（设备 {provenance.get('device', '-')}，耗时 {provenance.get('elapsed_seconds', '-')}s）",
        f"- 水印算法：{wm_info.get('type', '-')}（gamma={wm_info.get('gamma', '-')}, delta={wm_info.get('delta', '-')}, "
        f"prefix_length={wm_info.get('prefix_length', '-')}）",
        f"- 攻击方法：{provenance.get('attack', 'LLMP')}，seed={provenance.get('seed', '-')}",
    ]
    blocks: list[dict[str, Any]] = [
        {"type": "markdown", "title": "实验设置", "content": "\n".join(settings_lines)},
    ]
    rows = []
    label_zh = {"Watermarked": "水印生成", "Attacked": "LLMP 改写"}
    for variant in scenario.get("variants", []):
        detection = variant.get("detection") or {}
        rows.append([
            f"{variant.get('label', '-')}（{label_zh.get(variant.get('label'), '-')}）",
            _verdict(detection),
            _format_z(detection),
            f"{float(detection.get('green_fraction', 0.0)) * 100:.1f}%",
            str(detection.get("num_scored_tokens", "-")),
            str(detection.get("threshold", "-")),
        ])
    blocks.append({
        "type": "table",
        "title": "水印检测对比",
        "header": ["变体", "判定", "Z 分数", "绿 token 占比", "计分 token 数", "阈值"],
        "rows": rows,
    })
    for variant in scenario.get("variants", []):
        blocks.append({
            "type": "markdown",
            "title": f"{variant.get('label', '文本')} · {label_zh.get(variant.get('label'), '')}".strip(),
            "content": f"```\n{variant.get('text', '')}\n```",
        })
    blocks.append({"type": "markdown", "title": "结论", "content": _conclusion(scenario)})
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
        else:
            buffer.write(str(block["content"]) + "\n")
        buffer.write("\n")
    return buffer.getvalue()


def _summary(scenario: dict[str, Any]) -> str:
    watermarked, attacked = scenario["variants"]
    return (
        f"{scenario.get('watermark', {}).get('type', '')} 水印生成 + LLMP 改写完成："
        f"水印文本 {_verdict(watermarked['detection'])}（z={_format_z(watermarked['detection'])}），"
        f"改写后 {_verdict(attacked['detection'])}（z={_format_z(attacked['detection'])}）。"
    )


def run(config: dict[str, Any], output_dir: Path) -> int:
    from src.api.engine import Settings

    params = config.get("chapter_params") or {}
    common = config.get("common") or {}
    models = config.get("models") or {}

    entry = models.get("default")
    if not entry:
        return _fail(output_dir, "YAML 缺少 models.default（生成模型）")
    parsed = _run_params(params, common, output_dir)
    if isinstance(parsed, int):
        return parsed
    prompt, watermark, attack, max_new_tokens, seed = parsed

    settings = Settings(
        model_id=_entry_string(entry),
        device=str(common.get("device") or "auto"),
        dtype=str(common.get("dtype") or "auto"),
    )
    print(f"[hub-adapter] 模型: {settings.model_id}（设备 {settings.device}，dtype {settings.dtype}）；"
          f"首次运行会加载权重（HF 来源会先下载）", flush=True)
    engine = load_engine(settings)
    try:
        scenario = engine.run(prompt=prompt, watermark=watermark, max_new_tokens=max_new_tokens, seed=seed)
    except ValueError as exc:
        return _fail(output_dir, f"参数或上下文不满足要求: {exc}")
    except RuntimeError as exc:
        return _fail(output_dir, f"模型执行失败: {exc}")

    (output_dir / "results.json").write_text(
        json.dumps(scenario, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    blocks = _scenario_blocks(scenario)
    dtype = str(common.get("dtype") or "auto")
    dtype_note = ("引擎按请求 dtype 加载" if dtype in ("float16", "bfloat16", "float32")
                  else "引擎默认（CUDA 为 float16，CPU 为 float32）")
    header = (
        f"# 文本水印实验报告\n\n"
        f"KGW/Unigram 水印生成 → LLMP 改写 → z 分数检测。"
        f"dtype 请求 {dtype}：{dtype_note}。\n\n"
    )
    (output_dir / "report.md").write_text(header + _blocks_markdown(blocks), encoding="utf-8")
    _write_result(
        output_dir,
        {
            "status": "succeeded",
            "summary": _summary(scenario),
            "blocks": blocks,
            "artifacts": [{"name": "report.md"}, {"name": "results.json"}],
            "data": scenario,
        },
    )
    print("[hub-adapter] 水印实验完成", flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="TrustLLM hub adapter for MarkSec")
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
