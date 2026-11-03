from types import SimpleNamespace

import pytest

from guardrail_toolkit.pipeline import GuardedChat, PipelineError
from guardrail_toolkit.server import validate_chat_request


class Guard:
    def __init__(self, input_verdict="safe", output_verdict="safe", fail_at=None):
        self.input_verdict = input_verdict
        self.output_verdict = output_verdict
        self.fail_at = fail_at
        self.calls = []

    def check_input(self, text, **options):
        self.calls.append(("input", text, options))
        if self.fail_at == "input":
            raise RuntimeError("audit unavailable")
        return SimpleNamespace(to_dict=lambda: {"verdict": self.input_verdict})

    def check_output(self, text, response, **options):
        self.calls.append(("output", text, response, options))
        if self.fail_at == "output":
            raise ValueError("input limit")
        return SimpleNamespace(to_dict=lambda: {"verdict": self.output_verdict})


class Generator:
    def __init__(self, fail=False, truncated=False):
        self.fail = fail
        self.truncated = truncated
        self.calls = []

    def generate(self, text, **options):
        self.calls.append((text, options))
        if self.fail:
            raise RuntimeError("unavailable")
        return {"response": "Exact answer\n<script>not HTML</script>", "truncated": self.truncated}


def test_same_prompt_and_exact_generated_answer_are_audited():
    guard, generator = Guard(), Generator()
    result = GuardedChat(guard, generator).run("Original prompt", policy="audit policy", generation_max_tokens=256)
    assert result["release_allowed"]
    assert result["status"] == "completed"
    assert generator.calls == [("Original prompt", {"max_new_tokens": 256})]
    assert guard.calls[1][1:3] == ("Original prompt", result["generation"]["response"])
    assert guard.calls[0][-1]["policy"] == guard.calls[1][-1]["policy"] == "audit policy"


@pytest.mark.parametrize("verdict", ["unsafe", "unknown"])
def test_input_failure_stops_generation(verdict):
    guard, generator = Guard(input_verdict=verdict), Generator()
    result = GuardedChat(guard, generator).run("prompt")
    assert result["status"] == "input_blocked"
    assert not generator.calls
    assert len(guard.calls) == 1
    assert result["generation"] is None and result["output_audit"] is None
    assert not result["release_allowed"]


def test_audit_only_continues_but_preserves_failed_input():
    guard, generator = Guard(input_verdict="unsafe"), Generator()
    result = GuardedChat(guard, generator).run("prompt", enforce_input=False)
    assert result["status"] == "completed" and result["generation"]
    assert result["input_audit"]["verdict"] == "unsafe"
    assert result["output_audit"]["verdict"] == "safe"
    assert not result["release_allowed"]


@pytest.mark.parametrize("verdict", ["unsafe", "unknown"])
def test_output_failure_never_releases(verdict):
    result = GuardedChat(Guard(output_verdict=verdict), Generator()).run("prompt")
    assert not result["release_allowed"]
    assert result["generation"] is not None


def test_truncated_answer_is_audited_but_not_released_as_complete():
    result = GuardedChat(Guard(), Generator(truncated=True)).run("prompt")
    assert result["output_audit"]["verdict"] == "safe"
    assert not result["release_allowed"]


def test_seed_is_forwarded_only_when_provided():
    generator = Generator()
    GuardedChat(Guard(), generator).run("prompt")
    GuardedChat(Guard(), generator).run("prompt", seed=7)
    assert generator.calls[0][1] == {"max_new_tokens": 512}
    assert generator.calls[1][1] == {"max_new_tokens": 512, "seed": 7}


@pytest.mark.parametrize("failure", ["input", "generation", "output"])
def test_failures_keep_partial_results_and_release_lock(failure):
    guard = Guard(fail_at=failure)
    generator = Generator(fail=failure == "generation")
    chat = GuardedChat(guard, generator)
    with pytest.raises(PipelineError) as error:
        chat.run("prompt")
    result = error.value.partial_result
    assert result["status"] == "error" and not result["release_allowed"]
    assert (result["input_audit"] is not None) == (failure != "input")
    assert (result["generation"] is not None) == (failure == "output")
    assert result["output_audit"] is None
    assert result["latency_ms"] >= 0
    assert chat._lock.acquire(blocking=False)
    chat._lock.release()


@pytest.mark.parametrize("extra", [
    {"enforce_input": "false"}, {"generation_max_tokens": True},
    {"generation_max_tokens": 0}, {"generation_max_tokens": 3000},
    {"response": "untrusted precomputed answer"}, {"stage": "output"},
])
def test_invalid_chat_parameters(extra):
    with pytest.raises(ValueError):
        validate_chat_request({"text": "hi", **extra})
