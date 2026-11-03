"""Start the hub directly from a source checkout (mirrors guardrail's start.py)."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from trustllm_hub.cli import main

if __name__ == "__main__":
    # Repo root is the hub directory's parent: chapter toolkits live beside hub/.
    raise SystemExit(main(root=Path(__file__).resolve().parent.parent))
