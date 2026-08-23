from __future__ import annotations

import argparse
import json
import logging
from dataclasses import fields
from pathlib import Path
from typing import Any

from qwen3vl_agent.active_tree import ActiveTreeConfig
from qwen3vl_agent.coarse_to_fine import CoarseToFineConfig
from qwen3vl_agent.config import load_config
from qwen3vl_agent.evaluation.evidence30 import (
    STRATEGIES,
    Evidence30Dataset,
    preflight_evidence30,
    sha256_file,
)
from qwen3vl_agent.evaluation.evidence30_runner import (
    mark_locked_spent,
    run_evidence30_suite,
    summarize_existing_run,
)
from qwen3vl_agent.evaluation.freeze import (
    create_freeze_manifest,
    verify_freeze_manifest,
    write_freeze_manifest,
)
from qwen3vl_agent.paths import default_videomme_paths

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATHS = default_videomme_paths()
DEFAULT_EVIDENCE_ROOT = PROJECT_ROOT / "annotations" / "videomme_evidence30" / "0.2.0"
DEFAULT_SCHEMA = PROJECT_ROOT / "schemas" / "videomme_evidence_annotation.schema.json"
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "evidence30_2b.yaml"
DEFAULT_PARQUET = DATA_PATHS.annotation
DEFAULT_VIDEO_DIR = DATA_PATHS.videos
DEFAULT_SUBTITLE_DIR = DATA_PATHS.subtitles
DEFAULT_REGISTRY = PROJECT_ROOT / "runs" / "evidence30" / "locked_registry.json"


def _common_data_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--evidence-root", default=str(DEFAULT_EVIDENCE_ROOT))
    parser.add_argument("--schema", default=str(DEFAULT_SCHEMA))
    parser.add_argument("--parquet", default=str(DEFAULT_PARQUET))
    parser.add_argument("--video-dir", default=str(DEFAULT_VIDEO_DIR))
    parser.add_argument("--subtitle-dir", default=str(DEFAULT_SUBTITLE_DIR))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evidence30 relaxed engineering evaluation suite"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    preflight = subparsers.add_parser("preflight", help="Validate all 30 references/media")
    _common_data_arguments(preflight)
    preflight.add_argument("--skip-media-hashes", action="store_true")
    preflight.add_argument("--output", default=None)

    run = subparsers.add_parser("run", help="Run dev or frozen locked suite")
    _common_data_arguments(run)
    run.add_argument("--split", choices=("dev", "locked"), required=True)
    run.add_argument("--strategy", action="append", choices=STRATEGIES)
    run.add_argument("--config", default=str(DEFAULT_CONFIG))
    run.add_argument("--output-dir", required=True)
    run.add_argument("--freeze-manifest", default=None)
    run.add_argument("--locked-registry", default=str(DEFAULT_REGISTRY))
    run.add_argument("--skip-media-hashes", action="store_true")

    freeze = subparsers.add_parser("freeze", help="Freeze a passing 18-question dev run")
    freeze.add_argument("--dev-summary", required=True)
    freeze.add_argument("--config", default=str(DEFAULT_CONFIG))
    freeze.add_argument("--evidence-root", default=str(DEFAULT_EVIDENCE_ROOT))
    freeze.add_argument("--output", required=True)

    verify = subparsers.add_parser("verify-freeze", help="Verify the current frozen state")
    verify.add_argument("--freeze-manifest", required=True)

    summarize = subparsers.add_parser("summarize", help="Recompute aggregate run metrics")
    summarize.add_argument("--items", required=True)
    summarize.add_argument("--expected-items", type=int, required=True)
    summarize.add_argument("--locked", action="store_true")
    summarize.add_argument("--output", default=None)

    unseal = subparsers.add_parser(
        "unseal", help="Mark locked data spent before inspecting per-question traces"
    )
    unseal.add_argument("--evidence-root", default=str(DEFAULT_EVIDENCE_ROOT))
    unseal.add_argument("--locked-registry", default=str(DEFAULT_REGISTRY))
    unseal.add_argument("--items", required=True)
    return parser


def _write_optional(path: str | None, value: dict[str, Any]) -> None:
    if path is None:
        return
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _preflight(args: argparse.Namespace) -> dict[str, Any]:
    dataset = Evidence30Dataset.load(args.evidence_root)
    report = preflight_evidence30(
        dataset,
        schema_path=args.schema,
        parquet_path=args.parquet,
        video_dir=args.video_dir,
        subtitle_dir=args.subtitle_dir,
        verify_media_hashes=not args.skip_media_hashes,
    )
    _write_optional(getattr(args, "output", None), report)
    return report


def _explicit_config_errors(config: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    evaluation = config.get("evaluation")
    if not isinstance(evaluation, dict):
        errors.append("missing explicit evaluation mapping")
    else:
        required_evaluation = {
            "direct_frames",
            "direct_min_pixels",
            "direct_max_pixels",
            "direct_total_pixels",
            "direct_fps",
            "direct_subtitle_max_chars",
            "direct_subtitle_segments",
        }
        missing = sorted(required_evaluation - set(evaluation))
        if missing:
            errors.append("evaluation relies on implicit defaults: " + ", ".join(missing))
    for key, config_type in (
        ("coarse_to_fine", CoarseToFineConfig),
        ("active_tree", ActiveTreeConfig),
    ):
        value = config.get(key)
        if not isinstance(value, dict):
            errors.append(f"missing explicit {key} mapping")
            continue
        expected = {field.name for field in fields(config_type)}
        missing = sorted(expected - set(value))
        if missing:
            errors.append(f"{key} relies on implicit defaults: {', '.join(missing)}")
    model = config.get("model")
    if not isinstance(model, dict):
        errors.append("missing explicit model mapping")
    else:
        for key in ("path", "device", "dtype", "attn_implementation", "generation", "video"):
            if key not in model:
                errors.append(f"model.{key} is not explicit")
    return errors


def _freeze(args: argparse.Namespace) -> dict[str, Any]:
    dev_summary_path = Path(args.dev_summary).expanduser().resolve()
    dev_summary = json.loads(dev_summary_path.read_text(encoding="utf-8"))
    if dev_summary.get("split") != "dev":
        raise RuntimeError("freeze requires a dev summary")
    if not dev_summary.get("engineering_pass"):
        raise RuntimeError("dev engineering gate has not passed")
    if int(dev_summary.get("expected_items", 0)) != 54:
        raise RuntimeError("freeze requires all 18 questions across all three strategies")
    if set(dev_summary.get("strategies", {})) != set(STRATEGIES):
        raise RuntimeError("freeze requires direct, coarse_to_fine, and active_tree")

    config = load_config(args.config)
    config_errors = _explicit_config_errors(config)
    if config_errors:
        raise RuntimeError("Config is not freeze-safe: " + "; ".join(config_errors))
    model_path = config["model"]["path"]
    manifest = create_freeze_manifest(
        project_root=PROJECT_ROOT,
        config_path=args.config,
        evidence_root=args.evidence_root,
        model_path=model_path,
    )
    target = write_freeze_manifest(args.output, manifest)
    return {
        "ok": True,
        "freeze_id": manifest["freeze_id"],
        "manifest_path": str(target),
    }


def _progress(value: dict[str, Any]) -> None:
    if value["split"] == "locked":
        public = {
            "split": "locked",
            "strategy": value["strategy"],
            "completed": value["completed"],
            "expected": value["expected"],
        }
        print("LOCKED_PROGRESS=" + json.dumps(public, ensure_ascii=False), flush=True)
        return
    item = value["item"]
    public = {
        "question_id": item["question_id"],
        "strategy": item["strategy"],
        "prediction": item["prediction"],
        "correct": item["correct"],
        "engineering": item["engineering"],
        "relaxed_grounding": {
            "grounded": item["relaxed_grounding"]["grounded"],
            "slot_coverage": item["relaxed_grounding"]["slot_coverage"],
        },
    }
    print("DEV_ITEM=" + json.dumps(public, ensure_ascii=False), flush=True)


def _run(args: argparse.Namespace) -> dict[str, Any]:
    preflight = _preflight(args)
    if not preflight["ok"]:
        raise RuntimeError("Evidence30 preflight failed; refusing to run the model")
    dataset = Evidence30Dataset.load(args.evidence_root)
    config = load_config(args.config)
    strategies = args.strategy or list(STRATEGIES)
    return run_evidence30_suite(
        dataset=dataset,
        split=args.split,
        strategies=strategies,
        config=config,
        parquet_path=args.parquet,
        video_dir=args.video_dir,
        subtitle_dir=args.subtitle_dir,
        output_dir=args.output_dir,
        project_root=PROJECT_ROOT,
        freeze_manifest=args.freeze_manifest,
        locked_registry=args.locked_registry,
        progress=_progress,
    )


def _unseal(args: argparse.Namespace) -> dict[str, Any]:
    items = Path(args.items).expanduser().resolve()
    if not items.is_file():
        raise FileNotFoundError(items)
    dataset = Evidence30Dataset.load(args.evidence_root)
    dataset_id = sha256_file(dataset.root / "manifest.json")
    mark_locked_spent(args.locked_registry, dataset_id=dataset_id)
    return {
        "status": "spent",
        "sealed_artifact": str(items),
        "sha256": sha256_file(items),
        "warning": "Per-question inspection is now allowed; this locked set cannot be reused.",
    }


def main() -> None:
    args = build_parser().parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    if args.command == "preflight":
        result = _preflight(args)
    elif args.command == "run":
        result = _run(args)
    elif args.command == "freeze":
        result = _freeze(args)
    elif args.command == "verify-freeze":
        result = verify_freeze_manifest(args.freeze_manifest)
    elif args.command == "summarize":
        result = summarize_existing_run(
            items_path=args.items,
            expected_items=args.expected_items,
            locked=args.locked,
        )
        _write_optional(args.output, result)
    elif args.command == "unseal":
        result = _unseal(args)
    else:  # pragma: no cover - argparse prevents this branch
        raise AssertionError(args.command)
    print("EVIDENCE30_RESULT=" + json.dumps(result, ensure_ascii=False), flush=True)
    if result.get("ok") is False:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
