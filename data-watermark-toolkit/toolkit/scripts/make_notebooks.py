#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把 ``notebooks/py/0x_*.py`` 转成 ``notebooks/0x_*.ipynb``。

为什么用脚本生成 notebook？
--------------------------
本机没有安装 Jupyter，无法用 nbformat/nbconvert 生成与执行 notebook。为了让
书稿配套的 notebook 与「可执行脚本」保持一致、且能随时重新生成，这里用标准库
把带标记的 Python 脚本转换为 nbformat v4 的 JSON：

* ``# %% [markdown]`` 开始一个 markdown cell（后续行去掉开头的 ``# ``）；
* ``# %%`` 开始一个 code cell。

生成后可用 ``jupyter nbconvert --execute notebooks/0x_*.ipynb`` 在有 Jupyter
的环境里执行；本机则直接运行 ``notebooks/py/0x_*.py`` 验证逻辑（见 README）。

用法::

    python3 scripts/make_notebooks.py
"""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PY_DIR = REPO_ROOT / "notebooks" / "py"
NB_DIR = REPO_ROOT / "notebooks"
MAX_CELLS = 25


def _strip_markdown(lines: list[str]) -> list[str]:
    """去掉 markdown cell 每行开头的 ``# `` 或 ``#``。"""
    out: list[str] = []
    for line in lines:
        if line.startswith("# "):
            out.append(line[2:])
        elif line.strip() == "#":
            out.append("")
        else:
            out.append(line)
    return out


def parse_py(path: Path) -> list[dict]:
    """解析带 ``# %%`` 标记的 Python 脚本，返回 nbformat cell 列表。"""
    cells: list[dict] = []
    mode: str | None = None
    buffer: list[str] = []

    def flush() -> None:
        nonlocal buffer, mode
        if mode is None:
            buffer = []
            return
        source = buffer[:]
        # 去掉 cell 末尾多余空行
        while source and not source[-1].strip():
            source.pop()
        if not source:
            buffer = []
            return
        if mode == "markdown":
            text = _strip_markdown(source)
            cells.append({"cell_type": "markdown", "metadata": {},
                          "source": _join(text)})
        else:
            cells.append({"cell_type": "code", "execution_count": None,
                          "metadata": {}, "outputs": [], "source": _join(source)})
        buffer = []

    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip() == "# %% [markdown]":
            flush()
            mode = "markdown"
        elif line.strip() == "# %%":
            flush()
            mode = "code"
        else:
            buffer.append(line)
    flush()
    return cells


def _join(lines: list[str]) -> list[str]:
    """把行列表转换成 nbformat 的 source 数组（每行保留换行符，最后一行除外）。"""
    if not lines:
        return []
    return [line + "\n" for line in lines[:-1]] + [lines[-1]]


def make_notebook(py_path: Path) -> Path:
    """生成单个 notebook 并返回路径。"""
    cells = parse_py(py_path)
    if len(cells) > MAX_CELLS:
        raise SystemExit(f"{py_path.name} 有 {len(cells)} 个 cell，超过上限 {MAX_CELLS}")
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python",
                           "name": "python3"},
            "language_info": {"name": "python", "version": "3.10"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    nb_path = NB_DIR / (py_path.stem + ".ipynb")
    with nb_path.open("w", encoding="utf-8") as f:
        json.dump(notebook, f, ensure_ascii=False, indent=1)
        f.write("\n")
    return nb_path


def main() -> int:
    """生成全部 notebook。"""
    sources = sorted(PY_DIR.glob("*.py"))
    if not sources:
        raise SystemExit(f"没有找到 notebook 源脚本：{PY_DIR}")
    for py_path in sources:
        nb_path = make_notebook(py_path)
        cells = len(json.loads(nb_path.read_text(encoding="utf-8"))["cells"])
        print(f"已生成 {nb_path.relative_to(REPO_ROOT)}（{cells} cells）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
