"""Run exactly the original three R3 full cases with the v5.4 numbered-image executor."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from qwen3vl_agent.debug12 import main as run_debug
from qwen3vl_agent.r3.prompts import PROMPT_VERSION

CASE_IDS = ("R3-MVB-action-count-000", "R3-MME-225-3", "R3-MME-251-3")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=PROJECT / "configs/r1345_dual4090d.yaml")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if PROMPT_VERSION != "r3-5.4":
        raise ValueError("This entrypoint requires the R3 v5.4 query implementation")
    forwarded = ["--data-root", str(args.data_root), "--output-dir", str(args.output_dir),
                 "--config", str(args.config), "--pipeline", "R3", "--stage", "all"]
    for identifier in CASE_IDS:
        forwarded.extend(["--question-id", identifier])
    if args.resume:
        forwarded.append("--resume")
    return run_debug(forwarded)


if __name__ == "__main__":
    raise SystemExit(main())
