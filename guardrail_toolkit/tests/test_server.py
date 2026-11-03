from contextlib import contextmanager
import json
from threading import Thread
from urllib.error import HTTPError
from urllib.request import ProxyHandler, Request, build_opener

import pytest

from guardrail_toolkit.server import make_server, validate_request

# Local test traffic must not inherit the workstation's outbound proxy.
urlopen = build_opener(ProxyHandler({})).open


class FakeResult:
    def to_dict(self):
        return {"verdict": "safe", "label": "sec"}


class FakeGuard:
    state = "unloaded"

    def __init__(self):
        self.calls = []
        self.error = None

    def metadata(self):
        return {"state": self.state, "labels": []}

    def check_input(self, text, **kwargs):
        if self.error:
            raise self.error
        self.calls.append(("input", text, kwargs))
        return FakeResult()

    def check_output(self, text, response, **kwargs):
        self.calls.append(("output", text, response, kwargs))
        return FakeResult()


@contextmanager
def running_server(generator=None, **server_kwargs):
    guard = FakeGuard()
    server = make_server(guard, port=0, generator=generator, **server_kwargs)
    worker = Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        yield guard, f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=2)


def send(base, body, *, endpoint="/api/check", **headers):
    request = Request(base + endpoint, data=json.dumps(body).encode(),
                      headers={"Content-Type": "application/json", **headers})
    try:
        with urlopen(request, timeout=5) as response:
            return response.status, json.load(response)
    except HTTPError as exc:
        return exc.code, json.load(exc)


@pytest.mark.parametrize("body", [
    [], {"text": " "}, {"text": "x", "stage": "invalid"},
    {"text": "x", "stage": "output"}, {"text": "x", "response": "answer"},
    {"text": "x", "max_new_tokens": True}, {"text": "x", "max_new_tokens": 2000},
    {"text": "x", "explain": "false"}, {"text": "x", "messages": []},
    {"text": "x", "policy": 123}, {"text": "x" * 16001},
    {"text": "x" * 15000, "stage": "output", "response": "x" * 15000},
])
def test_invalid_payload(body):
    with pytest.raises(ValueError):
        validate_request(body)


def test_http_routes_static_assets_and_stage_dispatch():
    with running_server() as (guard, base):
        for path, content_type in [("/", "text/html"), ("/static/app.js", "text/javascript"),
                                   ("/static/styles.css", "text/css"), ("/docs", "text/html")]:
            with urlopen(base + path) as response:
                assert content_type in response.headers["Content-Type"]
                assert response.read()
        with urlopen(base + "/api/info") as response:
            assert json.load(response)["state"] == "unloaded"
        assert not guard.calls
        status, result = send(base, {"text": "hello", "explain": False})
        assert status == 200 and result["label"] == "sec"
        assert guard.calls[-1][0] == "input"
        assert guard.calls[-1][-1]["explain"] is False
        status, _ = send(base, {"stage": "output", "text": "question", "response": "answer"})
        assert status == 200
        assert guard.calls[-1][:3] == ("output", "question", "answer")


def test_http_errors_are_not_reported_as_safe():
    with running_server() as (guard, base):
        assert send(base, {"text": ""})[0] == 422
        assert send(base, {"text": "hello"}, Origin="https://unrelated.example")[0] == 403
        assert not guard.calls
        guard.error = BlockingIOError("busy")
        assert send(base, {"text": "hello"})[0] == 409
        guard.error = RuntimeError("model unavailable")
        code, body = send(base, {"text": "hello"})
        assert code == 503 and "verdict" not in body
        assert "model unavailable" not in body["detail"]


def test_trust_proxy_accepts_hub_proxied_requests():
    # 经 hub 同源反代：Origin 是 hub 端口、Host 是回环端口，trust_proxy 放宽比对。
    with running_server(trust_proxy=True) as (guard, base):
        status, result = send(base, {"text": "hello"}, Origin="http://127.0.0.1:8680")
        assert status == 200 and result["label"] == "sec"
        assert guard.calls[-1][0] == "input"


def test_invalid_json_and_unknown_files():
    with running_server() as (_, base):
        request = Request(base + "/api/check", data=b'{broken', headers={"Content-Type": "application/json"})
        with pytest.raises(HTTPError) as error:
            urlopen(request)
        assert error.value.code == 422
        with pytest.raises(HTTPError) as error:
            urlopen(base + "/static/../model.py")
        assert error.value.code == 404


def test_chat_endpoint_and_partial_generation_error():
    class Generator:
        fail = False

        def metadata(self):
            return {"model": "test-qwen", "state": "ready"}

        def generate(self, text, **kwargs):
            if self.fail:
                raise RuntimeError("service unavailable")
            return {"response": "Exact generated answer", "truncated": False}

    generator = Generator()
    with running_server(generator) as (guard, base):
        with urlopen(base + "/api/info") as response:
            assert json.load(response)["generator"]["state"] == "ready"
        status, result = send(base, {"text": "Original question"}, endpoint="/api/chat")
        assert status == 200 and result["release_allowed"]
        assert guard.calls[-1][:3] == ("output", "Original question", "Exact generated answer")
        generator.fail = True
        status, result = send(base, {"text": "Original question"}, endpoint="/api/chat")
        assert status == 503
        assert result["partial_result"]["input_audit"]["verdict"] == "safe"
        assert result["partial_result"]["output_audit"] is None
        assert not result["partial_result"]["release_allowed"]


def test_chat_without_generator_is_explicitly_unavailable():
    with running_server() as (guard, base):
        status, _ = send(base, {"text": "hello"}, endpoint="/api/chat")
        assert status == 503 and not guard.calls
