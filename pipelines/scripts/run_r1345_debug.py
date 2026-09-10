"""Portable entry point for the fixed R1/R3/R4/R5 debug set."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qwen3vl_agent.debug12 import main

if __name__ == "__main__":
    raise SystemExit(main())
