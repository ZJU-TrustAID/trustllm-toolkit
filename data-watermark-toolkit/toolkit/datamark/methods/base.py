# -*- coding: utf-8 -*-
"""水印方法插件协议。

每种水印方法实现一对操作：

* :meth:`WatermarkMethod.embed`：修改数据（``docs``）并返回「水印清单」片段。
  清单只交给数据所有者保管，**不得随数据集一起发布**；它记录了检测所需的
  秘密信息（金丝雀句、私有改写版、虚构事实、隐藏标记、触发 cue/reply 等）。
* :meth:`WatermarkMethod.detect`：读取模型（或它的问答输出）计算一个统计量，
  返回统一字段::

      {"score": float, "n": int, "direction": "greater", "detail": {...}}

  其中 ``p 值`` 与 ``decision`` 由 :mod:`datamark.audit` 统一补全。

``embed`` 的参数里有一个约定的隐藏上下文（下划线开头，由 pipeline 注入，不写进
配置文件）：

* ``cfg["_watermarked_ids"]``：本次实验中被选中携带水印的文档 id 列表；
* ``cfg["_fraction"]``：带水印样本比例 ρ；
* ``cfg["_seed"]``：主随机种子（方法的随机性由传入的 ``rng`` 负责）；
* ``cfg["_n_queries"]``：检测查询数 N；
* ``cfg["_manifest"]``：此前方法已产生的完整清单（一般不需要用）。

方法只应修改 ``_watermarked_ids`` 中列出的文档；检测时只要拿得到
``model`` 与自己的清单片段即可，不依赖其它方法。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class WatermarkMethod(ABC):
    """水印方法抽象基类（7 个子类分别对应书稿 2.2 分类图的 7 个叶子）。"""

    #: 方法名（注册到 REGISTRY 的键，也是配置里的键）。
    name: str = ""
    #: 检测所需的访问假设：white-box / gray-box / black-box。
    access: str = "white-box"
    #: 中文方法族名（对应书稿 2.2 分类图）。
    family: str = ""
    #: 嵌入载体的一句话描述（书稿表 2–x 的「载体」列）。
    carrier: str = ""
    #: 检测信号的一句话描述（书稿表 2–x 的「检测信号」列）。
    signal: str = ""

    @abstractmethod
    def embed(self, docs: list[dict], key: str, cfg: dict,
              rng: Any) -> tuple[list[dict], dict]:
        """嵌入水印，返回 ``(新文档列表, 清单片段)``。

        :param docs: 数据集文档列表（可原地修改后返回，也可先做深拷贝）。
        :param key: 水印密钥（十六进制字符串）。
        :param cfg: 方法参数（含下划线开头的内部上下文）。
        :param rng: ``random.Random`` 实例（确定性来源）。
        """

    @abstractmethod
    def detect(self, model, manifest: dict, cfg: dict, rng: Any) -> dict:
        """检测水印，返回原始统计量 ``{"score","n","direction","detail"}``。"""

    # -- 供子类复用的工具 --------------------------------------------------
    @staticmethod
    def watermarked_ids(cfg: dict) -> set[str]:
        """取本次实验中被选中携带水印的文档 id 集合。"""
        return set(cfg.get("_watermarked_ids", []) or [])

    @staticmethod
    def mark_watermarked(doc: dict, method_name: str) -> None:
        """把文档标记为「带水印」，并记录由哪个方法嵌入。"""
        doc["watermarked"] = True
        methods = doc.setdefault("methods", [])
        if method_name not in methods:
            methods.append(method_name)

    def descriptor(self) -> dict:
        """返回书稿表 2–x 需要的描述信息。"""
        return {
            "method": self.name,
            "family": self.family,
            "carrier": self.carrier,
            "signal": self.signal,
            "access": self.access,
        }
