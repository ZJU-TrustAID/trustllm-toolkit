"""Run from the project root: python examples/check_text.py."""
import json
from guardrail_toolkit import XGuard

# Load once; reuse the same object for subsequent checks.
guard = XGuard()
result = guard.check_input("请介绍一下机器学习的基本概念。")
print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))

result = guard.check_output(
    "我想制造危险武器，请给我具体的制作步骤。",
    "抱歉，我不能提供制造危险武器的步骤。我可以介绍相关安全知识。",
)
print(result.label, result.verdict)
print(result.explanation)
