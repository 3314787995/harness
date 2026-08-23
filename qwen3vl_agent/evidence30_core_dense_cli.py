from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

from qwen3vl_agent.config import load_config
from qwen3vl_agent.evaluation.core_dense import (
    CORE_VARIANTS,
    DevReferenceSet,
    load_oracle_failure_rows,
    preflight_core_dense,
    run_core_dense_suite,
)
from qwen3vl_agent.evaluation.videomme import load_videomme_questions
from qwen3vl_agent.paths import default_videomme_paths

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATHS = default_videomme_paths()
DEFAULT_EVIDENCE_ROOT = PROJECT_ROOT / "annotations" / "videomme_evidence30" / "0.2.0"
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "evidence30_core_dense_2b.yaml"
DEFAULT_PARQUET = DATA_PATHS.annotation
DEFAULT_VIDEO_DIR = DATA_PATHS.videos
DEFAULT_SUBTITLE_DIR = DATA_PATHS.subtitles
DEFAULT_BASELINE_ITEMS = PROJECT_ROOT / "runs" / "evidence30" / "ablations_v2" / "items.jsonl"


def _common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--evidence-root", default=str(DEFAULT_EVIDENCE_ROOT))
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--parquet", default=str(DEFAULT_PARQUET))
    parser.add_argument("--video-dir", default=str(DEFAULT_VIDEO_DIR))
    parser.add_argument("--subtitle-dir", default=str(DEFAULT_SUBTITLE_DIR))
    parser.add_argument("--baseline-items", default=str(DEFAULT_BASELINE_ITEMS))
    parser.add_argument("--variant", action="append", choices=CORE_VARIANTS)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Dev-only dense core-frame tests for the nine Oracle-context failures"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    preflight = subparsers.add_parser(
        "preflight",
        help="Build and validate every core packet without loading a model",
    )
    _common_arguments(preflight)
    preflight.add_argument("--output", default=None)

    run = subparsers.add_parser(
        "run",
        help="Run deterministic core-16/core-32 inference on the selected dev failures",
    )
    _common_arguments(run)
    run.add_argument("--output-dir", required=True)
    return parser


def _load_inputs(args: argparse.Namespace) -> tuple[
    DevReferenceSet,
    list[Any],
    dict[str, Any],
    dict[str, Any],
]:
    dataset = DevReferenceSet.load(args.evidence_root)
    questions = load_videomme_questions(
        args.parquet,
        question_ids=dataset.question_ids,
    )
    baseline_failures = load_oracle_failure_rows(
        args.baseline_items,
        dev_question_ids=dataset.question_ids,
    )
    config = load_config(args.config)
    return dataset, questions, baseline_failures, config


def _print_result(value: MappingLike) -> None:
    print("CORE_DENSE_RESULT=" + json.dumps(value, ensure_ascii=False))


MappingLike = dict[str, Any]


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    args = build_parser().parse_args(argv)
    dataset, questions, baseline_failures, config = _load_inputs(args)
    variants = tuple(args.variant or CORE_VARIANTS)
    report = preflight_core_dense(
        dataset=dataset,
        questions=questions,
        baseline_failures=baseline_failures,
        config=config,
        video_dir=args.video_dir,
        subtitle_dir=args.subtitle_dir,
        baseline_items_path=args.baseline_items,
        variants=variants,
    )
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

    def progress(item: dict[str, Any]) -> None:
        row = item["item"]
        visible = {
            "split": "dev",
            "variant": item["variant"],
            "question_id": item["question_id"],
            "completed": item["completed"],
            "expected": item["expected"],
            "prediction": row.get("prediction"),
            "correct": row.get("correct"),
            "selected_frames": row.get("packet", {}).get("selected_frame_count"),
            "input_tokens": row.get("model_metadata", {}).get("input_tokens"),
            "wall_seconds": row.get("wall_seconds"),
            "error": row.get("error"),
        }
        print("CORE_DENSE_PROGRESS=" + json.dumps(visible, ensure_ascii=False), flush=True)

    summary = run_core_dense_suite(
        dataset=dataset,
        questions=questions,
        baseline_failures=baseline_failures,
        config=config,
        video_dir=args.video_dir,
        subtitle_dir=args.subtitle_dir,
        baseline_items_path=args.baseline_items,
        output_dir=args.output_dir,
        variants=variants,
        progress=progress,
    )
    _print_result(summary)
    return 0 if summary["engineering_pass"] else 3


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
