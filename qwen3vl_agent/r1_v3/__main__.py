"""Run one R1-compatible request with V3; keep its answer separate from the trace."""

import argparse
import json
from pathlib import Path

from qwen3vl_agent.config import load_config
from qwen3vl_agent.factory import build_model
from qwen3vl_agent.r1_v3 import R1Request, R1V3VideoAgent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True, help="UTF-8 JSON with R1Request fields")
    parser.add_argument("--config", required=True, help="YAML containing model and r1_v3")
    parser.add_argument("--trace-output", required=True)
    args = parser.parse_args()
    request = R1Request(**json.loads(Path(args.request).read_text(encoding="utf-8-sig")))
    config = load_config(args.config)
    agent = R1V3VideoAgent(build_model(config["model"]), config=config.get("r1_v3"))
    agent.load()
    try:
        result = agent.solve(request)
    finally:
        agent.unload()
    target = Path(args.trace_output).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    print(result.output_text)


if __name__ == "__main__":
    main()
