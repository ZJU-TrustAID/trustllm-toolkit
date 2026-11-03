"""Start both services directly from a source checkout."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from guardrail_toolkit.launcher import main

if __name__ == "__main__":
    raise SystemExit(main(root=Path(__file__).resolve().parent))
