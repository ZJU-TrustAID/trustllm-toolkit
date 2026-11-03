"""Sanitized environment for chapter child processes (servers & one-shot runs).

hub is launched from the user's interactive shell and inherits whatever that
shell exported — commonly ``LD_LIBRARY_PATH`` pointing at a system CUDA
toolkit (e.g. set in ~/.bashrc for compiling). The dynamic linker consults
LD_LIBRARY_PATH *before* a wheel's built-in RUNPATH, so pip-managed chapter
environments get their ``libnvJitLink.so.12``/``libcublas.so.12`` hijacked by
the system toolkit's older copies and die on import with errors like
``undefined symbol: __nvJitLinkGetErrorLogSize_12_9``.

Child processes therefore do NOT inherit ``LD_LIBRARY_PATH``. Environment-
specific needs still work afterwards: ``conda run`` applies the chapter
env's own activate.d scripts, and a chapter manifest's ``server.env`` entries
are layered on top of the sanitized copy by the caller.
"""
from __future__ import annotations

import os


def chapter_env() -> dict[str, str]:
    """A copy of ``os.environ`` safe to hand to chapter subprocesses."""
    env = os.environ.copy()
    # 丢掉 shell 里遗留的 LD_LIBRARY_PATH，避免系统 CUDA 库劫持 pip wheel 的
    # RUNPATH（libnvJitLink 版本符号不匹配会让章节进程 import torch 即崩）。
    env.pop("LD_LIBRARY_PATH", None)
    env["PYTHONUNBUFFERED"] = "1"
    return env


__all__ = ["chapter_env"]
