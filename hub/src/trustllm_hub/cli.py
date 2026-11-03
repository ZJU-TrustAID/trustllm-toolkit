"""Command-line entry: ``python -m trustllm_hub`` or the ``trustllm-hub`` script."""
from __future__ import annotations

import argparse
from pathlib import Path

import uvicorn

from .config import load_config


def main(root: Path | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="《可信大模型》Toolkit Hub：集成前端与章节任务调度",
    )
    parser.add_argument("--root", help="集成仓库根目录（默认：当前目录）", default=None)
    parser.add_argument("--host", help="监听地址（覆盖 hub.config.json）", default=None)
    parser.add_argument("--port", type=int, help="监听端口（覆盖 hub.config.json）", default=None)
    parser.add_argument("--reload", action="store_true", help="开发模式：代码变更时自动重启")
    args = parser.parse_args()

    try:
        config = load_config(Path(args.root) if args.root else root)
    except (OSError, ValueError) as exc:
        print(f"读取 hub 配置失败: {exc}")
        return 1

    host = args.host or config.host
    port = args.port or config.port

    from .app import create_app

    app = create_app(config)
    print(f"TrustLLM Toolkit Hub v{app.version} → http://{host}:{port}")
    print(f"集成仓库根目录: {config.root}")
    print(f"运行产物目录: {config.workspace}")
    uvicorn.run(app, host=host, port=port, log_level="info")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
