"""HTTP API end-to-end over a fake chapter (registry → YAML → run → result)."""
from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient

from trustllm_hub.app import create_app

from conftest import sample_run_config


@pytest.fixture
def client(hub_env) -> TestClient:
    return TestClient(create_app(hub_env))


def test_hub_info_and_chapters(client: TestClient) -> None:
    info = client.get("/api/hub-info").json()
    assert info["version"]
    chapters = client.get("/api/chapters").json()
    assert len(chapters) == 1
    assert chapters[0]["toolkit"] == "fakekit"
    assert chapters[0]["enabled"] is True


def test_yaml_compose_and_reject(client: TestClient) -> None:
    response = client.post("/api/yaml", json=sample_run_config())
    assert response.status_code == 200
    payload = response.json()
    assert "toolkit: fakekit" in payload["yaml"]
    assert payload["config"]["models"]["default"]["repo"] == "org/model"

    bad = sample_run_config()
    bad["models"] = {}
    assert client.post("/api/yaml", json=bad).status_code == 400
    assert client.post("/api/yaml", json={"version": 2}).status_code == 400
    assert client.post("/api/yaml", json=sample_run_config(toolkit="ghost")).status_code == 404


def test_model_validate_endpoint(client: TestClient) -> None:
    response = client.post(
        "/api/models/validate",
        json={"entry": {"source": "hf", "repo": "org/model"}, "toolkit": "fakekit", "role": "default"},
    )
    assert response.status_code == 200
    assert response.json()["ok"] is True


def test_full_run_flow(client: TestClient) -> None:
    started = client.post("/api/runs", json={"config": sample_run_config()})
    assert started.status_code == 200
    session = started.json()["session"]

    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        status = client.get(f"/api/runs/{session}").json()
        if status["state"] != "running":
            break
        time.sleep(0.1)
    assert status["state"] == "succeeded"

    logs = client.get(f"/api/runs/{session}/logs").json()
    assert "fake adapter start" in logs["text"]

    result = client.get(f"/api/runs/{session}/result").json()
    assert result["result"]["summary"].startswith("fake ok")

    artifact = client.get(f"/api/runs/{session}/artifacts/report.md")
    assert artifact.status_code == 200 and "Fake report" in artifact.text
    assert client.get(f"/api/runs/{session}/artifacts/../config.yaml").status_code in {400, 404}

    history = client.get("/api/runs").json()
    assert any(item["session"] == session for item in history)


def test_run_with_yaml_text(client: TestClient) -> None:
    yaml_text = client.post("/api/yaml", json=sample_run_config(fail=True)).json()["yaml"]
    started = client.post("/api/runs", json={"yaml": yaml_text})
    session = started.json()["session"]
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        status = client.get(f"/api/runs/{session}").json()
        if status["state"] != "running":
            break
        time.sleep(0.1)
    assert status["state"] == "failed"
    assert status["exit_code"] == 3


def test_unknown_session_404(client: TestClient) -> None:
    assert client.get("/api/runs/ghost").status_code == 404


def test_chapter_server_unsupported(client: TestClient) -> None:
    # fakekit has no frontend.server → status reports unsupported; start returns 502.
    status = client.get("/api/chapters/fakekit/server").json()
    assert status == {"supported": False, "state": "unsupported"}
    assert client.post("/api/chapters/fakekit/server/start").status_code == 502
