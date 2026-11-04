from __future__ import annotations

import asyncio
from collections.abc import Awaitable
from typing import TypeVar

T = TypeVar("T")


def run_sync(awaitable: Awaitable[T]) -> T:
    return asyncio.run(awaitable)

