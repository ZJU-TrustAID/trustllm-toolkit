# -*- coding: utf-8 -*-
"""``python -m datamark`` 入口（等价于 ``python -m datamark.cli``）。"""

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
