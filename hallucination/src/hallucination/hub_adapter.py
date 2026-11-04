"""Hub adapter: execute one hallucination detection/mitigation job from a unified YAML.

Invoked by the TrustLLM hub as a subprocess inside this package's own
environment::

    python -m hallucination.hub_adapter --config <hub.yaml> --output-dir <session dir>

The adapter maps the hub's unified YAML onto this package's public API
(``Mitigator`` / ``Detector``), writes ``report.md`` / ``report.json`` and a
hub-facing ``result.json`` into the output directory. It never mutates package
state beyond loading models.
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path
from typing import Any

import yaml


def _load_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError(f"配置必须是映射: {path}")
    return config


def _device(common: dict[str, Any]) -> str | None:
    """Map hub ``common.device`` onto engine semantics: None means auto-pick."""
    device = str(common.get("device") or "").strip()
    return None if device in {"", "auto"} else device


def _model_binding(entry: dict[str, Any]) -> dict[str, Any]:
    """Map a hub model entry onto Mitigator/Detector keyword arguments."""
    source = entry.get("source")
    if source == "local_path":
        return {"model": str(entry["path"])}
    if source == "hf":
        return {"model": str(entry["repo"])}
    api = entry.get("api") or {}
    base_url = str(api.get("base_url") or "")
    backend = "openai" if "openai.com" in base_url else "openai-compatible"
    return {
        "model": str(api.get("model") or ""),
        "backend": backend,
        "api_base": base_url,
        "api_key_env": str(api.get("api_key_env") or "OPENAI_API_KEY"),
    }


def _flatten_trace(trace: dict[str, Any]) -> list[list[str]]:
    rows: list[list[str]] = []
    for key, value in trace.items():
        if isinstance(value, dict):
            rows.extend(
                [f"{key.rsplit('/', 1)[-1]} · {sub}", _short(item)] for sub, item in value.items()
            )
        else:
            rows.append([str(key), _short(value)])
    return rows


def _short(value: Any, limit: int = 90) -> str:
    if isinstance(value, list):
        text = json.dumps(value, ensure_ascii=False)
        return text if len(text) <= limit else text[: limit - 1] + "…"
    if isinstance(value, float):
        return f"{value:.4f}"
    text = str(value)
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _write_result(output_dir: Path, payload: dict[str, Any]) -> None:
    (output_dir / "result.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _fail(output_dir: Path, message: str) -> int:
    if output_dir is not None:
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
    traceback.print_exc(file=sys.stderr)
    return 2


def run_mitigation(config: dict[str, Any], output_dir: Path) -> int:
    from hallucination.mitigation import Mitigator

    params = config.get("chapter_params") or {}
    common = config.get("common") or {}
    models = config.get("models") or {}
    entry = models.get("default")
    if not entry:
        return _fail(output_dir, "YAML 缺少 models.default 模型配置")
    prompt = str(params.get("prompt") or "").strip()
    if not prompt:
        return _fail(output_dir, "缺少提示词（chapter_params.prompt）")

    binding = _model_binding(entry)
    print(f"[hub-adapter] 缓解方法={params.get('method', 'dola')} 模型={binding['model']}", flush=True)
    mitigator = Mitigator(
        method=str(params.get("method") or "dola"),
        device=_device(common),
        dtype=common.get("dtype") or "auto",
        seed=int(common.get("seed") or 0),
        steering=params.get("steering") or None,
        probe=params.get("probe") or None,
        **binding,
    )
    report = mitigator.mitigate(
        prompt,
        context=(str(params["context"]).strip() or None) if params.get("context") else None,
        max_new_tokens=int(params.get("max_new_tokens") or 128),
    )
    mitigator.close()

    (output_dir / "report.md").write_text(report.to_markdown(), encoding="utf-8")
    (output_dir / "report.json").write_text(
        json.dumps(report.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    blocks: list[dict[str, Any]] = []
    if report.baseline is not None:
        blocks.append({"type": "markdown", "title": "基线回答（未缓解）", "content": report.baseline})
    blocks.append({"type": "markdown", "title": f"缓解后回答（{report.method}）", "content": report.response or "*(空)*"})
    if report.trace:
        blocks.append(
            {
                "type": "table",
                "title": "方法轨迹（关键指标）",
                "header": ["指标", "值"],
                "rows": _flatten_trace(report.trace),
            }
        )
    if report.warnings:
        blocks.append(
            {
                "type": "markdown",
                "title": "警告",
                "content": "\n".join(f"- {warning}" for warning in report.warnings),
            }
        )
    _write_result(
        output_dir,
        {
            "status": "succeeded",
            "summary": f"{report.method} 缓解完成（模型 {binding['model']}），报告含基线对比与方法轨迹。",
            "blocks": blocks,
            "artifacts": [{"name": "report.md"}, {"name": "report.json"}],
            "data": {
                "method": report.method,
                "prompt": report.prompt,
                "context": report.context,
                "cost": report.cost,
                "provenance": report.provenance,
            },
        },
    )
    print("[hub-adapter] 缓解完成", flush=True)
    return 0


def run_detection(config: dict[str, Any], output_dir: Path) -> int:
    from hallucination.detection import Detector

    params = config.get("chapter_params") or {}
    common = config.get("common") or {}
    preset = str(params.get("preset") or "hf_whitebox")
    prompt = str(params.get("prompt") or "请介绍蓝鲸的食性。").strip()

    if preset == "offline_toy":
        detector = Detector(preset=preset)
        print("[hub-adapter] 离线教学预设（不调用任何模型）", flush=True)
    elif preset in {"openai_blackbox", "openai_whitebox"}:
        models = config.get("models") or {}
        api = (models.get("default") or {}).get("api") or {}
        detector = Detector(
            preset=preset,
            model=str(api.get("model") or ""),
            device=_device(common),
            num_samples=int(params.get("num_samples") or 5),
            threshold=float(params.get("threshold") or 0.5),
        )
        print(f"[hub-adapter] 检测预设={preset} 模型={api.get('model')}", flush=True)
    else:  # hf_whitebox
        models = config.get("models") or {}
        entry = models.get("default") or {}
        if entry.get("source") not in {"local_path", "hf"}:
            return _fail(output_dir, "hf_whitebox 预设需要本地路径或 HuggingFace 模型")
        binding = _model_binding(entry)
        detector = Detector(
            preset="hf_whitebox",
            model=binding["model"],
            device=_device(common),
            threshold=float(params.get("threshold") or 0.5),
        )
        print(f"[hub-adapter] 检测预设=hf_whitebox 模型={binding['model']}", flush=True)

    report = detector.detect(prompt)
    (output_dir / "report.md").write_text(report.to_markdown(), encoding="utf-8")
    (output_dir / "report.json").write_text(
        json.dumps(report.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    warnings = list((report.metadata or {}).get("warnings") or [])
    blocks = [
        {"type": "markdown", "title": "检测报告", "content": report.to_markdown()},
    ]
    if warnings:
        blocks.append({"type": "markdown", "title": "警告", "content": "\n".join(f"- {w}" for w in warnings)})
    response = (report.metadata or {}).get("response") or ""
    if response:
        blocks.insert(1, {"type": "markdown", "title": "被检回答", "content": response})
    _write_result(
        output_dir,
        {
            "status": "succeeded",
            "summary": f"检测完成（预设 {preset}），报告含各方法风险分与信号类型。",
            "blocks": blocks,
            "artifacts": [{"name": "report.md"}, {"name": "report.json"}],
            "data": {"preset": preset, "prompt": prompt, "warnings": warnings},
        },
    )
    print("[hub-adapter] 检测完成", flush=True)
    return 0


def _model_string(config: dict[str, Any]) -> str:
    entry = (config.get("models") or {}).get("default") or {}
    if entry.get("source") == "local_path":
        return str(entry["path"])
    if entry.get("source") == "hf":
        return str(entry["repo"])
    raise ValueError("该任务需要本地路径或 HuggingFace 模型")


def _run_cli(command: list[str]) -> tuple[int, str]:
    import subprocess

    print(f"[hub-adapter] $ {' '.join(command)}", flush=True)
    process = subprocess.run(command, capture_output=True, text=True)
    output = (process.stdout or "") + (process.stderr or "")
    for line in output.splitlines()[-40:]:
        print(f"  {line}", flush=True)
    return process.returncode, output


def _finish(output_dir: Path, *, summary: str, blocks: list[dict[str, Any]],
            artifacts: list[str], data: dict[str, Any]) -> int:
    _write_result(
        output_dir,
        {
            "status": "succeeded",
            "summary": summary,
            "blocks": blocks,
            "artifacts": [{"name": name} for name in artifacts],
            "data": data,
        },
    )
    print("[hub-adapter] 任务完成", flush=True)
    return 0


def run_fit_iti(config: dict[str, Any], output_dir: Path) -> int:
    params = config.get("chapter_params") or {}
    common = config.get("common") or {}
    dataset = str(params.get("fit_dataset") or "").strip()
    if not dataset:
        return _fail(output_dir, "缺少拟合数据集（fit_dataset）")
    out_name = str(params.get("fit_out") or "probe.json")
    out_path = (output_dir / out_name).resolve()
    command = [
        sys.executable, "-m", "hallucination.mitigation", "fit-iti",
        "--model", _model_string(config), "--data", dataset,
        "--top-k", str(int(params.get("fit_top_k") or 48)),
        "--alpha", str(float(params.get("fit_alpha") or 15.0)),
        "--out", str(out_path),
    ]
    code, output = _run_cli(command)
    if code != 0 or not out_path.is_file():
        return _fail(output_dir, f"fit-iti 退出码 {code}，产物未生成")
    blocks = [{
        "type": "code",
        "title": "ITI 探针产物（摘要）",
        "content": output_path_summary(out_path),
    }]
    return _finish(
        output_dir,
        summary=f"ITI 探针拟合完成：{out_path.name}（把该路径填入缓解任务的 probe 参数即可使用）",
        blocks=blocks,
        artifacts=[out_path.name],
        data={"artifact": str(out_path), "device": common.get("device")},
    )


def output_path_summary(path: Path, limit: int = 600) -> str:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        return f"（读取失败: {exc}）"
    return text[:limit] + ("…" if len(text) > limit else "")


def run_benchmark_mitigation(config: dict[str, Any], output_dir: Path) -> int:
    from hallucination.mitigation import Mitigator
    from hallucination.mitigation.evaluation.benchmark import evaluate_records

    params = config.get("chapter_params") or {}
    common = config.get("common") or {}
    dataset = str(params.get("bench_dataset") or "textbook_toy")
    limit = int(params.get("bench_n") or 20)
    records = _load_jsonl_records(dataset)
    print(f"[hub-adapter] 基准评测：{len(records)} 条记录，limit={limit}", flush=True)
    mitigator = Mitigator(
        method=str(params.get("bench_method") or "dola"),
        model=_model_string(config),
        device=_device(common),
        dtype=common.get("dtype") or "auto",
        seed=int(common.get("seed") or 0),
    )
    result = evaluate_records(mitigator, records, limit=limit)
    (output_dir / "benchmark.json").write_text(result.to_json(), encoding="utf-8")
    (output_dir / "benchmark.md").write_text(result.to_markdown(), encoding="utf-8")
    return _finish(
        output_dir,
        summary=(
            f"{result.method} 基准评测完成：n={result.n}，"
            f"基线 {result.baseline_score:.3f} → 缓解后 {result.mitigated_score:.3f} "
            f"(Δ{result.mitigated_score - result.baseline_score:+.3f}，win/tie/loss="
            f"{result.win}/{result.tie}/{result.loss})"
        ),
        blocks=[{"type": "markdown", "title": "评测报告", "content": result.to_markdown()}],
        artifacts=["benchmark.md", "benchmark.json"],
        data=result.to_dict(),
    )


def _load_jsonl_records(dataset: str) -> list[dict[str, Any]]:
    path = Path(dataset)
    if not path.is_file():
        from hallucination.mitigation.evaluation.datasets import (
            SUPPORTED_DATASETS,
            load_dataset_records,
        )

        if dataset in SUPPORTED_DATASETS:
            return list(load_dataset_records(dataset))
        raise ValueError(f"数据集 {dataset!r} 不是 JSONL 文件，也不是注册数据集 {SUPPORTED_DATASETS}")
    records = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def run_benchmark_detection(config: dict[str, Any], output_dir: Path) -> int:
    params = config.get("chapter_params") or {}
    prefix = output_dir / "detection_benchmark"
    command = [
        sys.executable, "-m", "hallucination.detection", "benchmark",
        "--dataset", str(params.get("bench_dataset") or "textbook_toy"),
        "--preset", str(params.get("bench_preset") or "offline_toy"),
        "--out-prefix", str(prefix),
    ]
    code, _ = _run_cli(command)
    json_path = prefix.with_suffix(".json")
    if code != 0 or not json_path.is_file():
        return _fail(output_dir, f"detection benchmark 退出码 {code}")
    blocks = [{"type": "code", "title": "检测结果摘要", "content": output_path_summary(json_path, 1200)}]
    return _finish(
        output_dir,
        summary=f"检测基准评测完成（数据集 {params.get('bench_dataset') or 'textbook_toy'}）",
        blocks=blocks,
        artifacts=["detection_benchmark.json", "detection_benchmark.csv"],
        data={},
    )


def run_fit_calibration(config: dict[str, Any], output_dir: Path) -> int:
    params = config.get("chapter_params") or {}
    scores = str(params.get("cal_scores") or "").strip()
    if not scores:
        return _fail(output_dir, "缺少分数记录文件（cal_scores，JSONL：score+label）")
    method = str(params.get("cal_method") or "threshold")
    out_path = output_dir / "calibration.json"
    command = [
        sys.executable, "-m", "hallucination.detection", "calibrate",
        "--scores", scores, "--method", method, "--out", str(out_path),
    ]
    code, _ = _run_cli(command)
    if code != 0 or not out_path.is_file():
        return _fail(output_dir, f"calibrate 退出码 {code}")
    blocks = [{"type": "code", "title": "校准产物（摘要）", "content": output_path_summary(out_path)}]
    return _finish(
        output_dir,
        summary=f"校准完成（方法 {method}）：{out_path.name}（检测任务可作为 calibration 输入）",
        blocks=blocks,
        artifacts=["calibration.json"],
        data={"artifact": str(out_path)},
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="TrustLLM hub adapter for hallucination-toolkit")
    parser.add_argument("--config", required=True, help="hub 统一 YAML 配置")
    parser.add_argument("--output-dir", required=True, help="输出目录（hub 会话目录）")
    args = parser.parse_args(argv)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        config = _load_config(Path(args.config).expanduser())
    except (OSError, yaml.YAMLError) as exc:
        return _fail(output_dir, f"读取配置失败: {exc}")
    task = str((config.get("chapter_params") or {}).get("task") or "mitigation")
    try:
        if task == "detection":
            return run_detection(config, output_dir)
        if task == "fit-iti":
            return run_fit_iti(config, output_dir)
        if task == "benchmark-mitigation":
            return run_benchmark_mitigation(config, output_dir)
        if task == "benchmark-detection":
            return run_benchmark_detection(config, output_dir)
        if task == "fit-calibration":
            return run_fit_calibration(config, output_dir)
        return run_mitigation(config, output_dir)
    except Exception as exc:  # adapter boundary: surface any failure as result.json + exit 2
        return _fail(output_dir, f"{type(exc).__name__}: {exc}")


if __name__ == "__main__":
    raise SystemExit(main())
