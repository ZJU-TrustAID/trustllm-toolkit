# -*- coding: utf-8 -*-
"""serve/tests 的公共夹具：把仓库根加进 sys.path，便于直接 import serve 包。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
