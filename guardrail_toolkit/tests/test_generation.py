from io import BytesIO
import json
from urllib.error import URLError

import pytest

from guardrail_toolkit.generation import LocalQwen


class Opener:
    def __init__(self, payload):
        self.payload = payload
        self.request = None

    def open(self, request, timeout):
        self.request = request
        if isinstance(self.payload, Exception):
            raise self.payload
        return BytesIO(json.dumps(self.payload).encode())


def test_generation_uses_original_text_and_disables_thinking():
    client = LocalQwen()
    client._opener = Opener({"choices": [{"message": {"content": "Actual answer\n"}, "finish_reason": "length"}], "usage": {"completion_tokens": 32}})
    result = client.generate("Original user request", max_new_tokens=32)
    payload = json.loads(client._opener.request.data)
    assert payload["messages"] == [{"role": "user", "content": "Original user request"}]
    assert payload["chat_template_kwargs"] == {"enable_thinking": False}
    assert result["response"] == "Actual answer\n" and result["truncated"]
    assert result["usage"]["completion_tokens"] == 32


@pytest.mark.parametrize("payload", [
    {"choices": []}, {"choices": [{"message": {"content": None}, "finish_reason": "stop"}]},
    {"choices": [{"message": {"content": ""}, "finish_reason": "length"}]},
    {"choices": [{"message": {"content": "answer"}, "finish_reason": "tool_calls"}]},
    URLError("offline"),
])
def test_empty_or_failed_generation_is_not_fabricated(payload):
    client = LocalQwen()
    client._opener = Opener(payload)
    with pytest.raises(RuntimeError):
        client.generate("hello")
