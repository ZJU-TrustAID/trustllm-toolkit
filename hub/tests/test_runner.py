"""RunManager: subprocess execution, logs, stop, busy limit (fake adapters only)."""
from __future__ import annotations

import time

import pytest

from trustllm_hub.runner import RunManager, RunnerBusyError
from trustllm_hub.sessions import SessionPaths, list_sessions

from conftest import make_config, sample_run_config


def wait_terminal(manager: RunManager, session: str, timeout: float = 30.0) -> str:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        state = manager.get(session).state
        if state != "running":
            return state
        time.sleep(0.1)
    raise AssertionError("run did not finish in time")


def test_successful_run(hub_env) -> None:
    from trustllm_hub.registry import scan_registry

    manager = RunManager(hub_env)
    manifest_obj = scan_registry(hub_env.iter_chapter_dirs()).get("fakekit")
    session = manager.start(manifest_obj, sample_run_config())
    assert wait_terminal(manager, session) == "succeeded"

    paths = SessionPaths(hub_env.workspace, session)
    assert (paths.dir / "report.md").is_file()
    result = paths.read_result()
    assert result["status"] == "succeeded"
    assert "fake adapter start" in manager.read_log(session)
    status = manager.get(session).status()
    assert status["exit_code"] == 0 and status["has_result"] is True


def test_failed_run_marks_state(hub_env) -> None:
    from trustllm_hub.registry import scan_registry

    manager = RunManager(hub_env)
    manifest = scan_registry(hub_env.iter_chapter_dirs()).get("fakekit")
    session = manager.start(manifest, sample_run_config(fail=True))
    assert wait_terminal(manager, session) == "failed"
    assert manager.get(session).exit_code == 3


def test_busy_limit(hub_env) -> None:
    from trustllm_hub.registry import scan_registry

    manager = RunManager(hub_env)
    manifest = scan_registry(hub_env.iter_chapter_dirs()).get("fakekit")
    first = manager.start(manifest, sample_run_config(sleep=3))
    with pytest.raises(RunnerBusyError):
        manager.start(manifest, sample_run_config())
    manager.stop(first)
    assert wait_terminal(manager, first) in {"stopped", "failed"}


def test_stop_kills_running_adapter(hub_env) -> None:
    from trustllm_hub.registry import scan_registry

    manager = RunManager(hub_env)
    manifest = scan_registry(hub_env.iter_chapter_dirs()).get("fakekit")
    session = manager.start(manifest, sample_run_config(sleep=60))
    time.sleep(0.5)  # let the adapter start
    status = manager.stop(session)
    assert status["state"] in {"stopped", "failed"}


def test_history_listing(hub_env) -> None:
    from trustllm_hub.registry import scan_registry

    manager = RunManager(hub_env)
    manifest = scan_registry(hub_env.iter_chapter_dirs()).get("fakekit")
    session = manager.start(manifest, sample_run_config())
    wait_terminal(manager, session)
    history = list_sessions(hub_env.workspace)
    assert history and history[0]["session"] == session
    assert history[0]["state"] == "succeeded"
    assert history[0]["toolkit"] == "fakekit"


def test_timeout_enforced(tmp_path) -> None:
    from trustllm_hub.registry import load_manifest, scan_registry

    from conftest import make_chapter_dir

    make_chapter_dir(tmp_path)
    config = make_config(tmp_path)
    config.workspace.mkdir(parents=True, exist_ok=True)
    registered = scan_registry(config.iter_chapter_dirs()).get("fakekit")
    manifest_path = registered.directory / "chapter.yaml"
    # Shrink the manifest timeout to 2s for this test.
    text = manifest_path.read_text(encoding="utf-8").replace("timeout: 30", "timeout: 2")
    manifest_path.write_text(text, encoding="utf-8")
    manifest = load_manifest(manifest_path)
    manager = RunManager(config)
    session = manager.start(manifest, sample_run_config(sleep=60))
    assert wait_terminal(manager, session, timeout=30) == "timeout"
