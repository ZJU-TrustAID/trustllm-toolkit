"""离线运行：普通教学问题、方法组合、对话与外部片段组装。"""

import tempfile
from pathlib import Path

from textbook_toolkit import jailbreak as jb


def main() -> None:
    query = "解释光合作用"
    method = jb.get("role_play", role="科普教师").then(jb.Suffix("请分三点说明。"))
    prompt = method.build(query)
    print(prompt.text)
    print("组装轨迹：", [step.method for step in prompt.steps])

    conversation = jb.Context(
        (
            jb.Message("user", "我们来讨论植物。"),
            jb.Message("assistant", "好的，可以先讨论植物如何利用光能。"),
        )
    ).build(query)
    print("保留的对话轮次：", len(conversation.messages))

    # 普通示例片段，仅演示接口；没有运行 GCG，也不是预先计算的攻击后缀。
    fragment = jb.Fragment("请使用简明中文。", source="teaching fixture", for_query=query)
    assembled = jb.OptimizedPrompt(fragment=fragment).build(query)
    print("外部片段组装：", assembled.text)

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "prompt.json"
        prompt.save(path)
        assert jb.Prompt.load(path) == prompt
        print("JSON 保存与恢复：通过")

    print("已注册方法：", len(jb.list_methods()))


if __name__ == "__main__":
    main()
