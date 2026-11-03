"""Exercise real service processes without model dependencies or GPUs."""
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

import pytest

from guardrail_toolkit.launcher import build_commands, qwen_env


FAKE_SERVICE = '''
import argparse
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
import signal
import subprocess
import sys

parser = argparse.ArgumentParser()
parser.add_argument("name")
parser.add_argument("--host")
parser.add_argument("--port", type=int)
args, _ = parser.parse_known_args()
name = args.name
Path(name + ".started").touch()
if Path(name + ".fail").exists():
    sys.exit(7)

# A descendant verifies that the launcher also cleans up model worker processes.
worker = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(300)"])
Path(name + ".worker").write_text(str(worker.pid))

def stop(signum, frame):
    worker.wait(timeout=3)
    Path(name + ".stopped").touch()
    sys.exit(0)

signal.signal(signal.SIGTERM, stop)

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(503 if Path(name + ".unready").exists() else 200)
        self.end_headers()
    def log_message(self, *args):
        pass

server = HTTPServer((args.host, args.port), Handler)
server.timeout = 0.1
while not Path(name + ".crash").exists():
    server.handle_request()
sys.exit(9)
'''


def eventually(predicate, timeout=10):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.05)
    raise AssertionError("Timed out waiting for subprocess state")


@pytest.fixture
def setup(tmp_path):
    root = tmp_path / "project with spaces"
    root.mkdir()
    script = root / "fake_service.py"
    script.write_text(FAKE_SERVICE)
    for name, filename in (("guard", "tokenizer_config.json"), ("qwen", "config.json")):
        model = root / "models" / name
        model.mkdir(parents=True)
        (model / filename).write_text("{}")
    with socket.create_server(("127.0.0.1", 0)) as first, socket.create_server(("127.0.0.1", 0)) as second:
        web_port, qwen_port = first.getsockname()[1], second.getsockname()[1]
    config = {
        "guard": {"model": "models/guard", "device": "auto",
                  "command": [sys.executable, str(script), "guard"]},
        "qwen": {"model": "models/qwen", "command": [sys.executable, str(script), "qwen"],
                 "port": qwen_port, "gpu_memory_utilization": 0.65},
        "web": {"host": "127.0.0.1", "port": web_port},
        "startup_timeout": 5,
    }
    return root, config


@pytest.fixture
def launch(setup):
    root, config = setup
    processes = []

    def start():
        (root / "config.json").write_text(json.dumps(config))
        env = os.environ.copy()
        env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
        process = subprocess.Popen(
            [sys.executable, "-u", "-c",
             "import sys; from guardrail_toolkit.launcher import main; "
             "sys.exit(main(root=sys.argv.pop()))", str(root)],
            cwd=root.parent, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        )
        processes.append(process)
        return process

    yield start
    for process in processes:
        if process.poll() is None:
            process.terminate()
        process.communicate(timeout=15)


def test_commands_use_project_relative_models_and_matching_qwen_port(setup):
    root, config = setup
    qwen, guard = build_commands(config, root)
    assert str(root / "models/qwen") in qwen
    assert guard[guard.index("--model") + 1] == str(root / "models/guard")
    assert guard[guard.index("--qwen-url") + 1] == f"http://127.0.0.1:{config['qwen']['port']}/v1"


def test_dtype_is_forwarded_to_both_services(setup):
    root, config = setup
    config["guard"]["dtype"] = "bfloat16"
    config["qwen"]["dtype"] = "bfloat16"
    qwen, guard = build_commands(config, root)
    assert qwen[qwen.index("--dtype") + 1] == "bfloat16"
    assert guard[guard.index("--dtype") + 1] == "bfloat16"


def test_dtype_defaults_to_auto_without_vllm_flag(setup):
    root, config = setup
    qwen, guard = build_commands(config, root)
    assert "--dtype" not in qwen
    assert guard[guard.index("--dtype") + 1] == "auto"


def test_qwen_env_pins_visible_device_from_config():
    env = qwen_env({"qwen": {"device": "cuda:5"}}, {"PATH": "/bin"})
    assert env["CUDA_VISIBLE_DEVICES"] == "5"
    assert env["PATH"] == "/bin"


def test_qwen_env_without_explicit_device_keeps_environment():
    for device in ("auto", "", None):
        env = qwen_env({"qwen": {"device": device}}, {"PATH": "/bin"})
        assert "CUDA_VISIBLE_DEVICES" not in env


def test_hugging_face_ids_are_passed_unchanged_without_local_weights(setup):
    root, config = setup
    config["guard"]["model"] = "Alibaba-AAIG/YuFeng-XGuard-Reason-8B"
    config["qwen"]["model"] = "Qwen/Qwen3.5-9B"
    qwen, guard = build_commands(config, root)
    assert qwen[qwen.index("serve") + 1] == config["qwen"]["model"]
    assert guard[guard.index("--model") + 1] == config["guard"]["model"]


@pytest.mark.parametrize("sig", [signal.SIGINT, signal.SIGTERM])
def test_both_services_stop_with_launcher(setup, launch, sig):
    root, _ = setup
    process = launch()
    eventually(lambda: (root / "guard.worker").exists())
    process.send_signal(sig)
    output, _ = process.communicate(timeout=15)
    assert process.returncode == 0, output
    assert (root / "qwen.stopped").exists()
    assert (root / "guard.stopped").exists()
    for name in ("guard", "qwen"):
        with pytest.raises(ProcessLookupError):
            os.kill(int((root / f"{name}.worker").read_text()), 0)


@pytest.mark.parametrize("failed", ["qwen", "guard"])
def test_failed_start_cleans_up_other_service(setup, launch, failed):
    root, _ = setup
    (root / f"{failed}.fail").touch()
    process = launch()
    output, _ = process.communicate(timeout=15)
    assert process.returncode == 1, output
    assert f"{failed} 已退出" in output
    if failed == "qwen":
        assert not (root / "guard.started").exists()
    else:
        assert (root / "qwen.stopped").exists()


def test_waits_for_qwen_and_times_out_without_starting_guard(setup, launch):
    root, config = setup
    config["startup_timeout"] = 0.5
    (root / "qwen.unready").touch()
    process = launch()
    output, _ = process.communicate(timeout=15)
    assert process.returncode == 1 and "超时" in output
    assert not (root / "guard.started").exists()
    assert (root / "qwen.stopped").exists()


def test_runtime_exit_stops_other_service(setup, launch):
    root, _ = setup
    process = launch()
    eventually(lambda: (root / "guard.worker").exists())
    (root / "guard.crash").touch()
    output, _ = process.communicate(timeout=15)
    assert process.returncode == 1 and "guard 已退出" in output
    assert (root / "qwen.stopped").exists()


def test_occupied_port_prevents_starting_models(setup, launch):
    root, config = setup
    with socket.create_server(("127.0.0.1", config["web"]["port"])):
        process = launch()
        output, _ = process.communicate(timeout=10)
    assert process.returncode == 1, output
    assert not (root / "qwen.started").exists()


def test_missing_weights_fail_before_starting_models(setup, launch):
    root, config = setup
    config["guard"]["model"] = "./missing"
    process = launch()
    output, _ = process.communicate(timeout=10)
    assert process.returncode == 1 and "tokenizer_config.json" in output
    assert not (root / "qwen.started").exists()


def test_missing_guard_command_fails_before_starting_qwen(setup, launch):
    root, config = setup
    config["guard"]["command"] = [str(root / "missing-python")]
    process = launch()
    output, _ = process.communicate(timeout=10)
    assert process.returncode == 1 and "找不到 guard 启动命令" in output
    assert not (root / "qwen.started").exists()


def test_local_config_takes_precedence(setup, launch):
    root, config = setup
    local_config = dict(config, startup_timeout=0)
    (root / "config.local.json").write_text(json.dumps(local_config))
    process = launch()
    output, _ = process.communicate(timeout=10)
    assert process.returncode == 1 and "startup_timeout" in output
    assert not (root / "qwen.started").exists()
