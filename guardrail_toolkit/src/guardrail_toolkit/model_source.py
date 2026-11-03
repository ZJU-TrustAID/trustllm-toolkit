"""Keep Hugging Face repository IDs distinct from local model paths."""
from pathlib import Path


def resolve_model_source(source: str, root: Path | None = None) -> str:
    if not isinstance(source, str) or not source.strip():
        raise ValueError("模型必须是 Hugging Face 仓库 ID 或本地目录")
    path = Path(source).expanduser()
    if root is not None:
        path = root / path
    # Explicit paths remain local even when missing; never download a typo in ./models/.
    if source.startswith((".", "~", "/")) or path.exists():
        return str(path.resolve())
    return source
