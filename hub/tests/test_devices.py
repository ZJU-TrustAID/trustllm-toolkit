"""GPU inventory (nvidia-smi) and hub-side ``auto`` device resolution."""
from __future__ import annotations

import yaml

import trustllm_hub.runner as runner_module
from trustllm_hub.devices import parse_smi_output, resolve_auto_device
from trustllm_hub.runner import resolve_config_device

SAMPLE_SMI = """\
0, NVIDIA L40, 29797, 46068
1, NVIDIA L40, 3, 46068
2, NVIDIA L40, 30000, 46068
"""


def test_parse_smi_output_computes_free_memory() -> None:
    gpus = parse_smi_output(SAMPLE_SMI)
    assert [gpu["index"] for gpu in gpus] == [0, 1, 2]
    assert gpus[0]["memory_free_mb"] == 46068 - 29797
    assert gpus[1]["memory_free_mb"] == 46068 - 3
    assert all(gpu["memory_free_mb"] >= 0 for gpu in gpus)


def test_parse_smi_output_skips_malformed_lines() -> None:
    assert parse_smi_output("not a csv line\n1, NVIDIA L40, 3, 46068\n") == [
        {"index": 1, "name": "NVIDIA L40", "memory_used_mb": 3,
         "memory_total_mb": 46068, "memory_free_mb": 46065},
    ]
    assert parse_smi_output("") == []


def test_resolve_auto_device_picks_freest_gpu(monkeypatch) -> None:
    gpus = parse_smi_output(SAMPLE_SMI)
    monkeypatch.setattr("trustllm_hub.devices.list_gpus", lambda: gpus)
    assert resolve_auto_device() == "cuda:1"


def test_resolve_auto_device_without_gpus_returns_none(monkeypatch) -> None:
    monkeypatch.setattr("trustllm_hub.devices.list_gpus", lambda: [])
    assert resolve_auto_device() is None


def _config(device: str) -> dict:
    return {"common": {"device": device, "dtype": "auto", "seed": 0}}


def test_resolve_config_device_pins_auto_to_freest_gpu(monkeypatch) -> None:
    monkeypatch.setattr("trustllm_hub.devices.list_gpus", lambda: parse_smi_output(SAMPLE_SMI))
    config = _config("auto")
    assert resolve_config_device(config) == "cuda:1"
    assert config["common"]["device"] == "cuda:1"


def test_resolve_config_device_keeps_explicit_device(monkeypatch) -> None:
    monkeypatch.setattr("trustllm_hub.devices.list_gpus", lambda: parse_smi_output(SAMPLE_SMI))
    config = _config("cuda:3")
    assert resolve_config_device(config) is None
    assert config["common"]["device"] == "cuda:3"


def test_resolve_config_device_without_gpus_keeps_auto(monkeypatch) -> None:
    monkeypatch.setattr("trustllm_hub.devices.list_gpus", lambda: [])
    config = _config("auto")
    assert resolve_config_device(config) is None
    assert config["common"]["device"] == "auto"


def test_run_with_auto_device_writes_resolved_yaml(hub_env, monkeypatch) -> None:
    """End to end: the adapter subprocess receives the resolved device in its YAML."""
    monkeypatch.setattr(runner_module, "resolve_auto_device", lambda: "cuda:5")
    from trustllm_hub.registry import scan_registry
    from trustllm_hub.sessions import SessionPaths

    manager = runner_module.RunManager(hub_env)
    manifest = scan_registry(hub_env.iter_chapter_dirs()).get("fakekit")
    config = {
        "version": 1,
        "chapter": 99,
        "toolkit": "fakekit",
        "common": {"device": "auto", "dtype": "auto", "seed": 0},
        "models": {"default": {"source": "hf", "repo": "org/model"}},
        "chapter_params": {},
    }
    session = manager.start(manifest, config)
    written = yaml.safe_load(SessionPaths(hub_env.workspace, session).config_yaml.read_text("utf-8"))
    assert written["common"]["device"] == "cuda:5"
    status = manager.get(session).status()
    assert status["config"]["common"]["device"] == "cuda:5"
    assert status["device_resolution"] == {"requested": "auto", "resolved": "cuda:5"}
