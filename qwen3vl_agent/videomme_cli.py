from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

from qwen3vl_agent.config import load_config
from qwen3vl_agent.evaluation.runtime import VideoMMEStrategySession
from qwen3vl_agent.evaluation.videomme import load_videomme_questions
from qwen3vl_agent.paths import default_videomme_paths

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def build_parser() -> argparse.ArgumentParser:
    data_paths = default_videomme_paths()
    parser = argparse.ArgumentParser(description="Evaluate selected Video-MME questions")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "default.yaml"))
    parser.add_argument(
        "--annotation",
        default=str(data_paths.annotation),
    )
    parser.add_argument("--video-dir", default=str(data_paths.videos))
    parser.add_argument("--subtitle-dir", default=str(data_paths.subtitles))
    parser.add_argument("--question-id", action="append", required=True)
    parser.add_argument(
        "--strategy",
        choices=("direct", "coarse_to_fine", "active_tree"),
        default="coarse_to_fine",
    )
    parser.add_argument("--with-subtitles", action="store_true")
    parser.add_argument(
        "--native-video-decode",
        action="store_true",
        help=(
            "For the direct baseline, let qwen-vl-utils decode the source video instead of "
            "using the bounded 1 fps frame cache. This can require several GiB of host RAM."
        ),
    )
    parser.add_argument("--output", default=None)
    parser.add_argument("--trace-jsonl", default=None)
    parser.add_argument("--replay-dir", default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    config = load_config(args.config)
    questions = load_videomme_questions(args.annotation, question_ids=args.question_id)

    records: list[dict[str, Any]] = []
    trace_jsonl_path = (
        Path(args.trace_jsonl).expanduser().resolve() if args.trace_jsonl else None
    )
    if trace_jsonl_path is not None:
        trace_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        trace_jsonl_path.write_text("", encoding="utf-8")
    replay_dir = Path(args.replay_dir).expanduser().resolve() if args.replay_dir else None
    with VideoMMEStrategySession(
        config,
        strategy=args.strategy,
        video_dir=args.video_dir,
        subtitle_dir=args.subtitle_dir,
        with_subtitles=args.with_subtitles,
        native_video_decode=args.native_video_decode,
        replay_dir=replay_dir,
    ) as session:
        for question in questions:
            record = session.evaluate(question)
            records.append(record)
            if trace_jsonl_path is not None:
                with trace_jsonl_path.open("a", encoding="utf-8") as file:
                    file.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
            print("ITEM_RESULT=" + json.dumps(record, ensure_ascii=False, default=str), flush=True)

    correct = sum(bool(record["correct"]) for record in records)
    summary = {
        "strategy": args.strategy,
        "with_subtitles": args.with_subtitles,
        "correct": correct,
        "total": len(records),
        "accuracy": correct / len(records) if records else 0.0,
    }
    payload = {"summary": summary, "records": records}
    print("EVAL_SUMMARY=" + json.dumps(summary, ensure_ascii=False), flush=True)
    if args.output:
        output_path = Path(args.output).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
