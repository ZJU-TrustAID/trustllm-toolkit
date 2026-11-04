"""离线验证 generator 协议；recorded_generator 是显式测试替身，不是真实模型。"""

import json
from collections.abc import Sequence

from textbook_toolkit.jailbreak import AutoDAN, Message, RedTeaming


def recorded_generator(messages: Sequence[Message]) -> str:
    """在实际使用中，替换成已有本地模型或 API 客户端的调用。"""
    payload = json.loads(messages[-1].content)
    if "prototype" in payload:
        return "请用课堂讨论的方式解释：{query}"
    return f"请用一个植物生长的例子说明：{payload['topic']}"


def main() -> None:
    query = "光合作用如何进行？"
    method = RedTeaming(recorded_generator)
    print("一次候选构造：", method.build(query).text)

    population = AutoDAN(recorded_generator, "请解释：{query}").population(query, size=4)
    print("初始化候选数：", len(population))
    print("示例候选：", population[0].text)


if __name__ == "__main__":
    main()
