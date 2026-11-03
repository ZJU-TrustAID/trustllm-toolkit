"""Supervise the local Qwen and guard web services with one command (POSIX)."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import json
import os
from pathlib import Path
import signal
import shutil
import socket
import subprocess
import time
from urllib.request import ProxyHandler, build_opener

from .generation import DEFAULT_QWEN_MODEL
from .model_source import resolve_model_source


def build_commands(config, root):
    """Validate configuration before starting either model process."""
    guard, qwen, web = (config[key] for key in ("guard", "qwen", "web"))
    for section in (guard, qwen):
        command = section["command"]
        if not isinstance(command, list) or not command or any(
            not isinstance(part, str) or not part for part in command
        ):
            raise ValueError("command 必须是非空字符串数组")
    for section in (qwen, web):
        if type(section["port"]) is not int or not 1 <= section["port"] <= 65535:
            raise ValueError("端口必须是 1–65535 的整数")
    if qwen["port"] == web["port"]:
        raise ValueError("Qwen 和网页必须使用不同端口")
    if not isinstance(web["host"], str) or not web["host"]:
        raise ValueError("web.host 必须是非空地址")
    if not isinstance(guard["device"], str) or not guard["device"]:
        raise ValueError("guard.device 必须是非空字符串")
    memory = qwen["gpu_memory_utilization"]
    if type(memory) not in (int, float) or not 0 < memory < 1:
        raise ValueError("gpu_memory_utilization 必须大于 0 且小于 1")
    timeout = config["startup_timeout"]
    if type(timeout) not in (int, float) or not 0 < timeout <= 86400:
        raise ValueError("startup_timeout 必须大于 0 且不超过 86400 秒")
    guard_model = resolve_model_source(guard["model"], root)
    qwen_model = resolve_model_source(qwen["model"], root)
    for model, filename in ((guard_model, "tokenizer_config.json"), (qwen_model, "config.json")):
        path = Path(model)
        if path.is_absolute() and not (path / filename).is_file():
            raise ValueError(f"模型目录缺少 {filename}: {path}；请准备权重或修改配置")

    qwen_command = [
        *qwen["command"], "serve", qwen_model,
        "--served-model-name", DEFAULT_QWEN_MODEL,
        "--host", "127.0.0.1", "--port", str(qwen["port"]),
        "--max-model-len", "8192", "--max-num-seqs", "2",
        "--gpu-memory-utilization", str(memory), "--language-model-only",
        "--enforce-eager", "--reasoning-parser", "qwen3", "--generation-config", "vllm",
    ]
    if str(qwen.get("dtype") or "auto") != "auto":
        qwen_command += ["--dtype", str(qwen["dtype"])]
    guard_command = [
        *guard["command"], "-m", "guardrail_toolkit.server",
        "--model", guard_model,
        "--device", guard["device"], "--dtype", str(guard.get("dtype") or "auto"),
        "--host", web["host"], "--port", str(web["port"]),
        "--qwen-url", f"http://127.0.0.1:{qwen['port']}/v1",
        "--qwen-model", DEFAULT_QWEN_MODEL,
    ]
    if web.get("trust_proxy"):
        # 经 hub 同源反代访问：Origin/Host 属于 hub 端口，放宽校验（仅回环监听）。
        guard_command.append("--trust-proxy")
    return qwen_command, guard_command


def qwen_env(config, base=None):
    """Environment for the vLLM subprocess: pin the card via qwen.device.

    vLLM has no --device flag; ``cuda:N`` maps to CUDA_VISIBLE_DEVICES=N so the
    target model lands on the card chosen in the hub (auto resolved by the hub).
    """
    env = dict(base if base is not None else os.environ)
    device = str(config.get("qwen", {}).get("device") or "")
    if device.startswith("cuda:"):
        env["CUDA_VISIBLE_DEVICES"] = device.split(":", 1)[1]
    return env


def check_port(host, port):
    # Fail before loading any weights when an existing service owns a port.
    with socket.create_server((host, port)):
        pass


def check_children(children):
    for name, process in children:
        code = process.poll()
        if code is not None:
            raise RuntimeError(f"{name} 已退出（状态码 {code}），请查看 runs/{name}.log")


def wait_ready(url, children, timeout):
    opener = build_opener(ProxyHandler({}))
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        check_children(children)
        try:
            with opener.open(url, timeout=min(2, max(0.01, deadline - time.monotonic()))) as response:
                if response.status == 200:
                    check_children(children)
                    return
        except OSError:
            pass
        time.sleep(0.2)
    raise TimeoutError(f"等待服务就绪超时：{url}；请查看 runs/ 下的日志")


def stop_children(children, grace=10):
    # Signal the entire session, including vLLM workers and conda wrappers.
    for _, process in reversed(children):
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + grace
    for _, process in reversed(children):
        try:
            process.wait(timeout=max(0, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            pass
    for _, process in reversed(children):
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


def run(config, root):
    qwen_command, guard_command = build_commands(config, root)
    for name, command in (("qwen", qwen_command), ("guard", guard_command)):
        executable = command[0]
        if os.sep in executable:
            executable = str(root / executable)
        if shutil.which(executable) is None:
            raise ValueError(f"找不到 {name} 启动命令 {command[0]!r}；请修改配置中的 command")
    check_port("127.0.0.1", config["qwen"]["port"])
    check_port(config["web"]["host"], config["web"]["port"])
    log_dir = root / "runs"
    log_dir.mkdir(exist_ok=True)
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["VLLM_NO_USAGE_STATS"] = "1"
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(root / "src"), env.get("PYTHONPATH")]))
    children = []

    def interrupted(signum, frame):
        raise KeyboardInterrupt

    previous = {sig: signal.signal(sig, interrupted) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        with ExitStack() as stack:
            try:
                for name, command, url in (
                    ("qwen", qwen_command, f"http://127.0.0.1:{config['qwen']['port']}/health"),
                    ("guard", guard_command, None),
                ):
                    logfile = stack.enter_context((log_dir / f"{name}.log").open("a", encoding="utf-8"))
                    logfile.write(f"\n--- {time.strftime('%Y-%m-%d %H:%M:%S')} 启动 {name} ---\n")
                    logfile.flush()
                    print(f"正在启动 {name}，日志：{log_dir / (name + '.log')}", flush=True)
                    process_env = qwen_env(config, env) if name == "qwen" else env
                    children.append((name, subprocess.Popen(
                        command, cwd=root, env=process_env, stdout=logfile, stderr=subprocess.STDOUT,
                        start_new_session=True,
                    )))
                    if url is None:
                        host = config["web"]["host"]
                        host = "127.0.0.1" if host == "0.0.0.0" else host
                        url = f"http://{host}:{config['web']['port']}/api/health"
                    wait_ready(url, children, config["startup_timeout"])
                print(f"启动完成：http://{host}:{config['web']['port']}（Ctrl+C 同时停止两个服务）", flush=True)
                print("XGuard 权重在首次检查时加载。", flush=True)
                while True:
                    check_children(children)
                    time.sleep(0.5)
            finally:
                # Repeated Ctrl+C must not interrupt worker cleanup.
                for sig in previous:
                    signal.signal(sig, signal.SIG_IGN)
                stop_children(children)
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def main(root=None):
    parser = argparse.ArgumentParser(description="一条命令启动 Qwen 和 Guard；Ctrl+C 一起停止")
    parser.add_argument("--config", help="配置文件；默认优先 config.local.json，否则 config.json")
    args = parser.parse_args()
    root = Path(root or Path.cwd()).resolve()
    config_path = root / (args.config or ("config.local.json" if (root / "config.local.json").exists() else "config.json"))
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
        run(config, root)
    except KeyboardInterrupt:
        print("\n已停止 Guard 和 Qwen。", flush=True)
        return 0
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
        print(f"运行失败：{exc}", flush=True)
        return 1
    return 0
