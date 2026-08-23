from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

from qwen3vl_agent.config import load_config
from qwen3vl_agent.evaluation.ablations import (
    PACKET_SOURCES,
    load_dev_trace_rows,
    preflight_ablation,
    run_ablation_suite,
)
from qwen3vl_agent.evaluation.evidence30 import Evidence30Dataset
from qwen3vl_agent.evaluation.videomme import load_videomme_questions
from qwen3vl_agent.paths import default_videomme_paths

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATHS = default_videomme_paths()
DEFAULT_EVIDENCE_ROOT = PROJECT_ROOT / "annotations" / "videomme_evidence30" / "0.2.0"
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "evidence30_ablation_2b.yaml"
DEFAULT_PARQUET = DATA_PATHS.annotation
DEFAULT_VIDEO_DIR = DATA_PATHS.videos
DEFAULT_SUBTITLE_DIR = DATA_PATHS.subtitles
DEFAULT_SOURCE_ITEMS = PROJECT_ROOT / "runs" / "evidence30" / "dev_v6" / "items.jsonl"


def _common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--evidence-root", default=str(DEFAULT_EVIDENCE_ROOT))
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--parquet", default=str(DEFAULT_PARQUET))
    parser.add_argument("--video-dir", default=str(DEFAULT_VIDEO_DIR))
    parser.add_argument("--subtitle-dir", default=str(DEFAULT_SUBTITLE_DIR))
    parser.add_argument("--source-dev-items", default=str(DEFAULT_SOURCE_ITEMS))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evidence30 dev-only Oracle/unified-answer-head ablations"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    preflight = subparsers.add_parser("preflight", help="Build all packets without loading a model")
    _common_arguments(preflight)
    preflight.add_argument("--output", default=None)

    run = subparsers.add_parser("run", help="Run deterministic unified-head inference on dev")
    _common_arguments(run)
    run.add_argument("--packet-source", action="append", choices=PACKET_SOURCES)
    run.add_argument("--output-dir", required=True)
    return parser


def _load_inputs(args: argparse.Namespace) -> tuple[
    Evidence30Dataset,
    list[Any],
    dict[tuple[str, str], dict[str, Any]],
    dict[str, Any],
]:
    dataset = Evidence30Dataset.load(args.evidence_root)
    question_ids = dataset.question_ids("dev")
    questions = load_videomme_questions(args.parquet, question_ids=question_ids)
    source_rows = load_dev_trace_rows(
        args.source_dev_items,
        dev_question_ids=question_ids,
    )
    config = load_config(args.config)
    return dataset, questions, source_rows, config


def _preflight(
    args: argparse.Namespace,
    dataset: Evidence30Dataset,
    questions: list[Any],
    source_rows: dict[tuple[str, str], dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, Any]:
    return preflight_ablation(
        dataset=dataset,
        questions=questions,
        source_rows=source_rows,
        config=config,
        video_dir=args.video_dir,
        subtitle_dir=args.subtitle_dir,
        source_items_path=args.source_dev_items,
    )


def _print_result(value: dict[str, Any]) -> None:
    print("ABLATION_RESULT=" + json.dumps(value, ensure_ascii=False))


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    args = build_parser().parse_args(argv)
    dataset, questions, source_rows, config = _load_inputs(args)
    report = _preflight(args, dataset, questions, source_rows, config)
    if args.command == "preflight":
        if args.output:
            output = Path(args.output).expanduser().resolve()
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(
                json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        _print_result(report)
        return 0 if report["ok"] else 2

    if not report["ok"]:
        _print_result(report)
        return 2

    packet_sources = tuple(args.packet_source or PACKET_SOURCES)

    def progress(item: dict[str, Any]) -> None:
        row = item["item"]
        visible = {
            "split": "dev",
            "packet_source": item["packet_source"],
            "question_id": item["question_id"],
            "completed": item["completed"],
            "expected": item["expected"],
            "prediction": row.get("prediction"),
            "correct": row.get("correct"),
            "grounded": row.get("relaxed_grounding", {}).get("grounded"),
            "selected_frames": row.get("packet", {}).get("selected_frame_count"),
            "input_tokens": row.get("model_metadata", {}).get("input_tokens"),
            "error": row.get("error"),
        }
        print("ABLATION_PROGRESS=" + json.dumps(visible, ensure_ascii=False), flush=True)

    summary = run_ablation_suite(
        dataset=dataset,
        questions=questions,
        source_rows=source_rows,
        config=config,
        video_dir=args.video_dir,
        subtitle_dir=args.subtitle_dir,
        source_items_path=args.source_dev_items,
        output_dir=args.output_dir,
        packet_sources=packet_sources,
        progress=progress,
    )
    _print_result(summary)
    return 0 if summary["engineering_pass"] else 3


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
