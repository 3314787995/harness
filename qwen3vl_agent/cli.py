from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from qwen3vl_agent.active_tree import ActiveTreeVideoAgent
from qwen3vl_agent.active_tree.replay import render_trace_html
from qwen3vl_agent.agent import Qwen3VLAgent
from qwen3vl_agent.coarse_to_fine import CoarseToFineVideoAgent
from qwen3vl_agent.coarse_to_fine.adapters import build_answer_adapter
from qwen3vl_agent.coarse_to_fine.prompts import build_direct_prompt
from qwen3vl_agent.config import load_config
from qwen3vl_agent.factory import build_model
from qwen3vl_agent.tools.defaults import build_default_registry


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Qwen3-VL with optional registered tools")
    parser.add_argument("--config", default=None)
    parser.add_argument("--query", required=True)
    parser.add_argument("--video", action="append", default=[])
    parser.add_argument("--image", action="append", default=[])
    parser.add_argument(
        "--strategy",
        choices=("direct", "tools", "coarse_to_fine", "active_tree"),
        default=None,
        help="Inference strategy; omitted preserves the legacy tools default.",
    )
    parser.add_argument("--choice", action="append", default=[])
    parser.add_argument("--subtitle", default=None)
    parser.add_argument("--trace-output", default=None)
    parser.add_argument("--replay-output", default=None)
    parser.add_argument("--no-tools", action="store_true")
    parser.add_argument("--show-metadata", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    if args.no_tools and args.strategy not in {None, "direct"}:
        raise SystemExit("--no-tools cannot be combined with a non-direct --strategy")
    strategy = args.strategy or ("direct" if args.no_tools else "tools")
    search_strategies = {"coarse_to_fine", "active_tree"}
    if strategy not in search_strategies and args.subtitle:
        raise SystemExit("--subtitle is supported by coarse_to_fine and active_tree")

    config = load_config(args.config) if args.config else {"model": {}, "agent": {}}
    agent_config = dict(config.get("agent") or {})
    model = build_model(config.get("model"))
    if strategy == "coarse_to_fine":
        agent = CoarseToFineVideoAgent(model, config=config.get("coarse_to_fine"))
    elif strategy == "active_tree":
        agent = ActiveTreeVideoAgent(model, config=config.get("active_tree"))
    else:
        agent = Qwen3VLAgent(model, tools=build_default_registry(), **agent_config)

    query = args.query
    if args.choice and strategy not in search_strategies:
        query = build_direct_prompt(query, build_answer_adapter(args.choice))
    agent.load()
    try:
        if strategy in search_strategies:
            result = agent.generate(
                [{"role": "user", "content": query}],
                videos=args.video or None,
                images=args.image or None,
                choices=args.choice or None,
                subtitle_path=args.subtitle,
            )
        else:
            result = agent.generate(
                [{"role": "user", "content": query}],
                videos=args.video or None,
                images=args.image or None,
                use_tools=strategy == "tools",
            )
    finally:
        agent.unload()

    print(result.text)
    if args.show_metadata:
        print(json.dumps(result.metadata, ensure_ascii=False, indent=2, default=str))
    if args.trace_output:
        output_path = Path(args.trace_output).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(result.metadata, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
    if args.replay_output:
        if strategy != "active_tree":
            raise SystemExit("--replay-output currently requires --strategy active_tree")
        render_trace_html(result.metadata["active_tree"], args.replay_output)


if __name__ == "__main__":
    main()
