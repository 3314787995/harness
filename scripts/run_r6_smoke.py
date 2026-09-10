"""Explicit GPU entry point: one direct baseline and selected R6 cases, with no gold input."""

import argparse
from dataclasses import replace
from pathlib import Path

from qwen3vl_agent.r6.evaluate import (
    model_and_config,
    preflight,
    public_request,
    read_jsonl,
    run_requests,
    write_json,
)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--config", default="configs/r6_8b.yaml")
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--request-ids", nargs="+", default=["R6-dev-E08", "R6-dev-E10", "R6-dev-E14"]
    )
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args(argv)
    path = Path(args.manifest).resolve()
    rows = [public_request(r, path.parent) for r in read_jsonl(path)]
    selected = {r.request_id: r for r in rows if r.request_id in args.request_ids}
    if len(selected) != len(args.request_ids):
        raise SystemExit("smoke manifest lacks requested IDs or contains duplicates")
    requests = [selected[i] for i in args.request_ids]
    model, config = model_and_config(args.config)
    reports = [preflight(r, config) for r in requests]
    root = Path(args.output)
    write_json(root / "preflight.json", {"results": reports, "model_loaded": False})
    if args.preflight_only:
        return
    if any(r["status"] != "ready" for r in reports):
        raise SystemExit("media preflight incomplete; no model was loaded")
    direct = replace(requests[0], request_id=requests[0].request_id + "-direct", mode="direct")
    run_requests(
        [direct, *(replace(r, mode="pipeline") for r in requests)],
        model_settings=model,
        config=config,
        output=root / "gpu",
    )


if __name__ == "__main__":
    main()
