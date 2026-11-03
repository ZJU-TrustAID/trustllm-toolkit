# -*- coding: utf-8 -*-
"""水印方法注册表。

新增一个检测器只需要三步（README §6 有 10 行示例）：

1. 继承 :class:`datamark.methods.base.WatermarkMethod`，实现 ``embed`` / ``detect``；
2. 用 ``@register("my_method")`` 装饰类；
3. 在配置文件 ``methods`` 里加上同名键（``{"enabled": true, ...}``）。

注册表 ``REGISTRY`` 供 ``--methods all``、notebook 与 sweep 统一枚举方法。
"""

from __future__ import annotations

from typing import Iterable

REGISTRY: dict[str, type] = {}


def register(name: str):
    """方法注册装饰器：把类登记到 :data:`REGISTRY`，并固定其 ``name``。"""

    def _decorator(cls):
        cls.name = name
        REGISTRY[name] = cls
        return cls

    return _decorator


# 导入即注册（顺序 = 书稿 2.2 分类图顺序：训练时 5 类 → 推理时 2 类）。
from .canary_memory import CanaryMemoryMethod  # noqa: E402
from .version_pair import VersionPairMethod  # noqa: E402
from .fictitious_knowledge import FictitiousKnowledgeMethod  # noqa: E402
from .token_statistics import TokenStatisticsMethod  # noqa: E402
from .trigger_behavior import TriggerBehaviorMethod  # noqa: E402
from .in_context import InContextMethod  # noqa: E402
from .rag_canary import RAGCanaryMethod  # noqa: E402

#: 书稿 2.2 分类图的固定顺序（训练时在前）。
METHOD_ORDER = (
    "canary_memory",
    "version_pair",
    "fictitious_knowledge",
    "token_statistics",
    "trigger_behavior",
    "in_context",
    "rag_canary",
)


def get_method(name: str):
    """按名称返回方法实例（未注册时抛出中文错误）。"""
    if name not in REGISTRY:
        raise KeyError(f"未知水印方法：{name}（已注册：{', '.join(sorted(REGISTRY))}）")
    return REGISTRY[name]()


def method_class(name: str) -> type:
    """按名称返回方法类。"""
    if name not in REGISTRY:
        raise KeyError(f"未知水印方法：{name}")
    return REGISTRY[name]


def all_methods() -> list:
    """按固定顺序返回全部方法实例（供 ``--methods all`` 使用）。"""
    return [get_method(name) for name in METHOD_ORDER]


def descriptors(names: Iterable[str] | None = None) -> list[dict]:
    """返回方法描述信息列表（书稿表 2–x 用）。"""
    selected = list(names) if names else list(METHOD_ORDER)
    return [get_method(name).descriptor() for name in selected]


__all__ = [
    "REGISTRY", "METHOD_ORDER", "register", "get_method", "method_class",
    "all_methods", "descriptors", "WatermarkMethod",
]
