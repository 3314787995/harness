from __future__ import annotations

import importlib
import sys
from pathlib import Path


def _main() -> int:
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    module = importlib.import_module("qwen3vl_agent.p01.preflight")
    return int(module.main())

if __name__ == "__main__":
    raise SystemExit(_main())
