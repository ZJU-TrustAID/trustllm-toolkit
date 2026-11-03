# -*- coding: utf-8 -*-
"""DataMark：数据集水印教学工具包（教材第二章 2.5「实践与应用」配套）。

三段式上手::

    from datamark import DataMark

    dm = DataMark.from_config("configs/fugu_sim.json")  # 数据 + 密钥 + 配置
    wm = dm.embed("work/fugu/wm")                        # 构造带水印数据集（+清单）
    model = dm.fit(wm)                                   # 训练 / 模拟可疑模型
    report = dm.audit(model)                             # 7 类检测器统一审计
    print(report.summary())

默认模拟模式只依赖 Python 标准库，可在无 GPU、无网络的环境秒级复现书稿实验；
``model.kind = "hf"`` / ``"api"`` 为可选的真实模型路径（需额外安装）。
"""

from .config import ConfigError, DataMarkConfig, load_config
from .keyring import KeyFileError, load_or_create_key
from .methods import METHOD_ORDER, REGISTRY, get_method, register
from .models import SuspectModel, build_model
from .pipeline import DataMark, PipelineError, WatermarkedDataset
from .audit import AuditReport
from .version import __version__

__all__ = [
    "DataMark",
    "WatermarkedDataset",
    "SuspectModel",
    "AuditReport",
    "DataMarkConfig",
    "ConfigError",
    "PipelineError",
    "KeyFileError",
    "load_config",
    "load_or_create_key",
    "build_model",
    "METHOD_ORDER",
    "REGISTRY",
    "get_method",
    "register",
    "__version__",
]
