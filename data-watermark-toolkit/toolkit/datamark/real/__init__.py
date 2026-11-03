# -*- coding: utf-8 -*-
"""DataMark 真实小模型实现（论文方法级）。

入口::

    python3 -m datamark.real.cli run --method canary_memory --fraction 0.5 \
        --config configs/real_smollm.json --out work/real/canary

需要可选依赖 torch/transformers（``pip install -e .[real]``）；默认模拟模式
不导入本包，因此不影响零依赖安装与单测。
"""

from .experiment import RealExperiment, RealResult  # noqa: F401
from .lm import TrainableLM  # noqa: F401

__all__ = ["RealExperiment", "RealResult", "TrainableLM"]
