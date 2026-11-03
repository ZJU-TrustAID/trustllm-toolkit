"""Hub adapter: execute one guardrail audit job from a unified YAML.

Invoked by the TrustLLM hub as a subprocess inside this package's own
environment::

    python -m guardrail_toolkit.hub_adapter --config <hub.yaml> --output-dir <session dir>

Modes (chapter_params.mode):
- ``full``  — 输入审计 → Qwen 生成 → 输出审计。The adapter starts its own
  vLLM subprocess for the target model, waits for readiness, runs
  :class:`GuardedChat`, then shuts vLLM down. Repeated audits are cheaper via
  the chapter's interactive frontend, which keeps the services warm.
- ``input`` — 仅输入审计（无需 Qwen）。
- ``output``— 仅输出审计（无需 Qwen；需要 chapter_params.response）。
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
from typing import Any
from urllib.request import ProxyHandler, build_opener

import yaml

DEFAULT_SERVED_NAME = "qwen3.5-9b"
VLLM_READY_TIMEOUT = 3600


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
    raise ValueError("护栏审计不支持 API 模型源，请使用本地路径或 HuggingFace 仓库 ID")


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


def _verdict_chip(verdict: str | None) -> str:
    return {"safe": "✅ 安全", "unsafe": "⛔ 不安全", "unknown": "⚠️ 未知"}.get(verdict or "", str(verdict))


def _audit_rows(audit: dict[str, Any] | None, stage: str) -> list[list[str]]:
    if not audit:
        return []
    return [[
        stage,
        str(audit.get("label") or "-"),
        str(audit.get("risk_name_zh") or audit.get("risk_name") or "-"),
        _verdict_chip(audit.get("verdict")),
        "是" if audit.get("explanation_complete") else "否（截断）" if audit.get("explanation") else "—",
    ]]


def _audit_blocks(result: dict[str, Any]) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    rows = _audit_rows(result.get("input_audit"), "输入审计")
    rows += _audit_rows(result.get("output_audit"), "输出审计")
    if rows:
        blocks.append({
            "type": "table",
            "title": "护栏审计判定",
            "header": ["阶段", "标签", "风险类别", "判定", "解释完整"],
            "rows": rows,
        })
    generation = result.get("generation") or {}
    if generation.get("response"):
        truncated = "（达到长度上限，未作为完整回答放行）" if generation.get("truncated") else ""
        blocks.append({
            "type": "markdown",
            "title": f"生成回答{truncated}",
            "content": generation["response"],
        })
    for stage_name, audit in (("输入审计", result.get("input_audit")), ("输出审计", result.get("output_audit"))):
        if audit and audit.get("explanation"):
            blocks.append({
                "type": "markdown",
                "title": f"{stage_name} · 模型解释",
                "content": audit["explanation"],
            })
    if result.get("decision_basis"):
        blocks.append({"type": "markdown", "title": "放行结论", "content": result["decision_basis"]})
    return blocks


class VllmProcess:
    """Target-model vLLM subprocess for ``full`` mode (mirrors the launcher)."""

    def __init__(self, model: str, port: int, gpu_memory_utilization: float,
                 device: str = "auto", dtype: str = "auto") -> None:
        executable = shutil.which("vllm")
        if executable is None:
            raise RuntimeError("当前环境找不到 vllm 命令；full 模式需要在 guardrail 环境中运行适配器")
        self.port = port
        self.command = [
            executable, "serve", model,
            "--served-model-name", DEFAULT_SERVED_NAME,
            "--host", "127.0.0.1", "--port", str(port),
            "--max-model-len", "8192", "--max-num-seqs", "2",
            "--gpu-memory-utilization", str(gpu_memory_utilization), "--language-model-only",
            "--enforce-eager", "--reasoning-parser", "qwen3", "--generation-config", "vllm",
        ]
        if dtype and dtype != "auto":
            self.command += ["--dtype", dtype]
        # vLLM has no --device flag; pin the card through CUDA_VISIBLE_DEVICES.
        self.visible_devices = device.split(":", 1)[1] if str(device).startswith("cuda:") else None
        self.process: subprocess.Popen | None = None

    def _ready(self) -> bool:
        opener = build_opener(ProxyHandler({}))
        try:
            with opener.open(f"http://127.0.0.1:{self.port}/health", timeout=1.5) as response:
                return response.status == 200
        except OSError:
            return False

    def __enter__(self) -> "VllmProcess":
        existing = self._ready()
        if existing:
            print(f"[hub-adapter] 复用端口 {self.port} 上已有的 vLLM 服务", flush=True)
            self.process = None
            return self
        log_path = os.environ.get("GUARDRAIL_HUB_VLLM_LOG")
        log_handle = open(log_path, "a", encoding="utf-8") if log_path else subprocess.DEVNULL
        try:
            env = {**os.environ, "VLLM_NO_USAGE_STATS": "1"}
            if self.visible_devices is not None:
                env["CUDA_VISIBLE_DEVICES"] = self.visible_devices
            self.process = subprocess.Popen(
                self.command, stdout=log_handle, stderr=subprocess.STDOUT,
                start_new_session=True, env=env,
            )
        finally:
            if log_handle is not subprocess.DEVNULL:
                log_handle.close()
        deadline = time.monotonic() + VLLM_READY_TIMEOUT
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError(f"vLLM 启动失败（退出码 {self.process.poll()}），请查看 vLLM 日志")
            if self._ready():
                print("[hub-adapter] vLLM 已就绪", flush=True)
                return self
            time.sleep(1.0)
        raise TimeoutError(f"等待 vLLM 就绪超时（{VLLM_READY_TIMEOUT}s）")

    def __exit__(self, *exc_info: Any) -> None:
        if self.process is None:
            return
        try:
            os.killpg(self.process.pid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            pass
        try:
            self.process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(self.process.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
            self.process.wait()


def run(config: dict[str, Any], output_dir: Path) -> int:
    from guardrail_toolkit.generation import LocalQwen
    from guardrail_toolkit.model import XGuard
    from guardrail_toolkit.pipeline import GuardedChat, PipelineError

    params = config.get("chapter_params") or {}
    common = config.get("common") or {}
    models = config.get("models") or {}
    mode = str(params.get("mode") or "full")

    guard_entry = models.get("guard")
    target_entry = models.get("target")
    if not guard_entry:
        return _fail(output_dir, "YAML 缺少 models.guard（护栏模型）")
    if mode == "full" and not target_entry:
        return _fail(output_dir, "full 模式缺少 models.target（生成模型）")
    text = str(params.get("text") or "").strip()
    if not text:
        return _fail(output_dir, "缺少用户请求（chapter_params.text）")

    explain = bool(params.get("explain", True))
    options = {"explain": explain, "max_new_tokens": int(params.get("max_new_tokens") or 384)}
    device = str(common.get("device") or "auto")
    dtype = str(common.get("dtype") or "auto")
    try:
        seed = int(common.get("seed") or 0)
    except (TypeError, ValueError):
        return _fail(output_dir, "common.seed 必须是整数")
    if seed < 0:
        return _fail(output_dir, "common.seed 需为非负整数")
    print(f"[hub-adapter] 设备 {device} · dtype {dtype} · seed {seed}", flush=True)

    try:
        if mode == "full":
            extra = target_entry.get("extra") or {}
            with VllmProcess(
                _entry_string(target_entry),
                port=int(extra.get("vllm_port") or 8767),
                gpu_memory_utilization=float(extra.get("gpu_memory_utilization") or 0.65),
                device=device, dtype=dtype,
            ) as vllm:
                generator = LocalQwen(
                    base_url=f"http://127.0.0.1:{vllm.port}/v1", model=DEFAULT_SERVED_NAME
                )
                chat = GuardedChat(XGuard(_entry_string(guard_entry), device=device, dtype=dtype), generator)
                result = chat.run(
                    text,
                    enforce_input=bool(params.get("enforce_input", True)),
                    generation_max_tokens=int(params.get("generation_max_tokens") or 512),
                    seed=seed,
                    **options,
                )
        else:
            guard = XGuard(_entry_string(guard_entry), device=device, dtype=dtype)
            if mode == "input":
                result = {"input_audit": guard.check_input(text, **options).to_dict()}
            elif mode == "output":
                response = str(params.get("response") or "").strip()
                if not response:
                    return _fail(output_dir, "output 模式需要 chapter_params.response（待审计的助手回答）")
                result = {"output_audit": guard.check_output(text, response, **options).to_dict()}
            else:
                return _fail(output_dir, f"未知审计模式: {mode}")
    except (ValueError, RuntimeError, TimeoutError, PipelineError) as exc:
        return _fail(output_dir, f"护栏审计执行失败: {exc}")

    (output_dir / "report.md").write_text(_markdown(result, mode, text), encoding="utf-8")
    _write_result(
        output_dir,
        {
            "status": "succeeded",
            "summary": _summary(result, mode),
            "blocks": _audit_blocks(result),
            "artifacts": [{"name": "report.md"}],
            "data": result,
        },
    )
    print("[hub-adapter] 审计完成", flush=True)
    return 0


def _summary(result: dict[str, Any], mode: str) -> str:
    names = {"full": "生成并审计", "input": "输入端审计", "output": "输出端审计"}
    if mode == "full":
        release = result.get("release_allowed")
        status = result.get("status")
        if status == "input_blocked":
            return "输入审计未通过，已按策略停止后续生成与输出审计。"
        verdict = "允许放行" if release else "不允许放行"
        return f"{names[mode]}完成：{verdict}（{result.get('decision_basis') or ''}）"
    audit = result.get("input_audit") or result.get("output_audit") or {}
    return f"{names[mode]}完成：判定 {_verdict_chip(audit.get('verdict'))}（标签 {audit.get('label') or '-'}）"


def _markdown(result: dict[str, Any], mode: str, text: str) -> str:
    lines = [f"# 护栏审计报告（{mode}）", "", f"**用户请求**: {text}", ""]
    lines.extend(md for md in [_blocks_markdown(result)] if md)
    return "\n".join(lines) + "\n"


def _blocks_markdown(result: dict[str, Any]) -> str:
    import io

    buffer = io.StringIO()
    for block in _audit_blocks(result):
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="TrustLLM hub adapter for guardrail-toolkit")
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
