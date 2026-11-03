"""Audit the exact input and the exact generated answer, retaining both results."""
from __future__ import annotations

import logging
from threading import Lock
import time

LOG = logging.getLogger(__name__)


class PipelineError(RuntimeError):
    def __init__(self, message, partial_result):
        super().__init__(message)
        self.partial_result = partial_result


class GuardedChat:
    def __init__(self, guard, generator):
        self.guard = guard
        self.generator = generator
        self._lock = Lock()

    def run(self, text, *, enforce_input=True, explain=True, max_new_tokens=384,
            generation_max_tokens=512, policy=None, seed=None):
        if not self._lock.acquire(blocking=False):
            raise BlockingIOError("正在运行另一条生成审计流程，请稍后重试")
        start = time.perf_counter()
        result = {
            "stage": "chat", "status": "running", "enforce_input": enforce_input,
            "input_audit": None, "generation": None, "output_audit": None,
            "release_allowed": False, "decision_basis": "", "error_stage": None,
        }
        current_stage = "input_audit"
        options = {"explain": explain, "max_new_tokens": max_new_tokens, "policy": policy}
        try:
            result["input_audit"] = self.guard.check_input(text, **options).to_dict()
            if enforce_input and result["input_audit"]["verdict"] != "safe":
                result.update(status="input_blocked", decision_basis="输入审计未通过，未调用 Qwen，未执行输出审计。")
                return result
            current_stage = "generation"
            generation_kwargs = {"max_new_tokens": generation_max_tokens}
            if seed is not None:
                generation_kwargs["seed"] = seed
            result["generation"] = self.generator.generate(text, **generation_kwargs)
            current_stage = "output_audit"
            result["output_audit"] = self.guard.check_output(text, result["generation"]["response"], **options).to_dict()
            input_safe = result["input_audit"]["verdict"] == "safe"
            output_safe = result["output_audit"]["verdict"] == "safe"
            truncated = result["generation"]["truncated"]
            result["release_allowed"] = input_safe and output_safe and not truncated
            result["status"] = "completed"
            if not output_safe:
                result["decision_basis"] = "输出审计未通过，不放行这条回答。下方保留模型原文供实验审阅。"
            elif not input_safe:
                result["decision_basis"] = "仅审计模式已继续生成；回答审计通过，但输入未通过，整体不放行。"
            elif truncated:
                result["decision_basis"] = "输入及已生成的回答片段审计通过，但生成达到长度上限，不作为完整回答放行。"
            else:
                result["decision_basis"] = "输入与完整回答均通过护栏审计，可以放行。"
            return result
        except Exception as exc:
            LOG.exception("Chat workflow failed at %s", current_stage)
            message = {
                "input_audit": "输入护栏审计失败，未调用 Qwen。",
                "generation": "Qwen 生成失败，请检查模型服务状态或缩短输入。",
                "output_audit": "输出护栏审计失败，生成的回答尚未获得安全判定。",
            }[current_stage]
            result.update(status="error", error_stage=current_stage, decision_basis=message)
            raise PipelineError(message, result) from exc
        finally:
            result["latency_ms"] = round((time.perf_counter() - start) * 1000, 1)
            self._lock.release()
