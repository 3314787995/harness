from __future__ import annotations

import json
import statistics
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from qwen3vl_agent.evaluation.evidence30 import (
    POLICY_ID,
    STRATEGIES,
    Evidence30Dataset,
    assess_engineering_record,
    canonical_sha256,
    score_relaxed_grounding,
    sha256_file,
)
from qwen3vl_agent.evaluation.freeze import verify_freeze_manifest
from qwen3vl_agent.evaluation.runtime import VideoMMEStrategySession
from qwen3vl_agent.evaluation.videomme import load_videomme_questions

SessionFactory = Callable[..., VideoMMEStrategySession]
ProgressCallback = Callable[[dict[str, Any]], None]


def _mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def _strategy_summary(items: list[dict[str, Any]]) -> dict[str, Any]:
    engineering = [item["engineering"] for item in items]
    scores = [item["relaxed_grounding"] for item in items]
    verified_values = [
        bool(item["metadata"].get("verified"))
        for item in items
        if item["strategy"] == "active_tree" and not item.get("error")
    ]
    visual_exposures = [
        exposure
        for item in items
        for exposure in item.get("metadata", {}).get("exposures", [])
        if exposure.get("modality") == "visual"
    ]
    return {
        "items": len(items),
        "completed": sum(bool(item["completed"]) for item in engineering),
        "completion_rate": _mean([float(bool(item["completed"])) for item in engineering]),
        "valid_prediction_rate": _mean(
            [float(bool(item["valid_prediction"])) for item in engineering]
        ),
        "trace_valid_rate": _mean(
            [float(bool(item["trace_valid"])) for item in engineering]
        ),
        "fatal_rate": _mean([float(bool(item["fatal"])) for item in engineering]),
        "degraded_rate": _mean(
            [float(bool(item["degraded"])) for item in engineering]
        ),
        "budget_violation_rate": _mean(
            [float(bool(item["budget_violation"])) for item in engineering]
        ),
        "accuracy": _mean([float(bool(item.get("correct"))) for item in items]),
        "verified_rate": _mean([float(value) for value in verified_values])
        if verified_values
        else None,
        "relaxed_grounded_rate": _mean(
            [float(bool(score["grounded"])) for score in scores]
        ),
        "mean_slot_coverage": _mean([float(score["slot_coverage"]) for score in scores]),
        "mean_context_item_recall": _mean(
            [float(score["context_item_recall"]) for score in scores]
        ),
        "mean_core_item_recall": _mean(
            [float(score["core_item_recall"]) for score in scores]
        ),
        "mean_model_calls": _mean(
            [float(item["model_calls"]) for item in engineering]
        ),
        "mean_input_tokens": _mean(
            [float(item["input_tokens"]) for item in engineering]
        ),
        "mean_output_tokens": _mean(
            [float(item["output_tokens"]) for item in engineering]
        ),
        "mean_visual_exposures": len(visual_exposures) / len(items) if items else 0.0,
        "mean_wall_seconds": _mean([float(item.get("wall_seconds", 0)) for item in items]),
        "stop_reasons": dict(Counter(item["stop_reason"] for item in engineering)),
    }


def aggregate_suite(
    items: list[dict[str, Any]],
    *,
    expected_items: int,
    suppress_groups_smaller_than: int | None = None,
) -> dict[str, Any]:
    by_strategy: dict[str, dict[str, Any]] = {}
    for strategy in STRATEGIES:
        selected = [item for item in items if item.get("strategy") == strategy]
        if selected:
            by_strategy[strategy] = _strategy_summary(selected)

    groups: dict[str, dict[str, Any]] = {}
    for field in ("topology", "duration_bucket", "modality_key"):
        values = sorted({str(item.get(field, "")) for item in items})
        field_groups: dict[str, Any] = {}
        suppressed = 0
        for value in values:
            selected = [item for item in items if str(item.get(field, "")) == value]
            question_count = len({item.get("question_id") for item in selected})
            if (
                suppress_groups_smaller_than is not None
                and question_count < suppress_groups_smaller_than
            ):
                suppressed += question_count
                continue
            field_groups[value] = {
                strategy: _strategy_summary(
                    [item for item in selected if item.get("strategy") == strategy]
                )
                for strategy in STRATEGIES
                if any(item.get("strategy") == strategy for item in selected)
            }
        groups[field] = {
            "values": field_groups,
            "suppressed_question_count": suppressed,
        }

    indexed = {
        (str(item.get("question_id")), str(item.get("strategy"))): item
        for item in items
    }
    question_ids = sorted({key[0] for key in indexed})
    comparisons: dict[str, Any] = {}
    for baseline in ("direct", "coarse_to_fine"):
        wins = losses = ties = comparable = 0
        for question_id in question_ids:
            active = indexed.get((question_id, "active_tree"))
            base = indexed.get((question_id, baseline))
            if active is None or base is None:
                continue
            comparable += 1
            active_correct = bool(active.get("correct"))
            base_correct = bool(base.get("correct"))
            if active_correct and not base_correct:
                wins += 1
            elif base_correct and not active_correct:
                losses += 1
            else:
                ties += 1
        comparisons[f"active_tree_vs_{baseline}"] = {
            "comparable": comparable,
            "wins": wins,
            "losses": losses,
            "ties": ties,
        }

    engineering_pass = (
        len(items) == expected_items
        and all(bool(item["engineering"]["completed"]) for item in items)
        and all(not bool(item["engineering"]["fatal"]) for item in items)
        and all(not bool(item["engineering"]["degraded"]) for item in items)
        and all(not bool(item["engineering"]["budget_violation"]) for item in items)
    )
    return {
        "policy_id": POLICY_ID,
        "expected_items": expected_items,
        "observed_items": len(items),
        "engineering_pass": engineering_pass,
        "strategies": by_strategy,
        "paired_comparisons": comparisons,
        "groups": groups,
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _append_jsonl(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, default=str) + "\n")
        handle.flush()


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _load_registry(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"version": 1, "datasets": {}}
    return json.loads(path.read_text(encoding="utf-8"))


def _reserve_locked_run(
    registry_path: Path,
    *,
    dataset_id: str,
    freeze_id: str,
    output_dir: Path,
) -> None:
    registry = _load_registry(registry_path)
    existing = registry["datasets"].get(dataset_id)
    if existing is not None:
        if existing.get("status") == "spent":
            raise RuntimeError("This locked Evidence30 dataset has been unsealed and is spent")
        if existing.get("freeze_id") != freeze_id:
            raise RuntimeError(
                "Locked Evidence30 is already reserved for a different freeze_id"
            )
        if Path(existing.get("output_dir", "")).resolve() != output_dir.resolve():
            raise RuntimeError("Locked Evidence30 must resume in its original output directory")
        return
    registry["datasets"][dataset_id] = {
        "freeze_id": freeze_id,
        "output_dir": str(output_dir),
        "status": "started",
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    _write_json(registry_path, registry)


def mark_locked_spent(registry_path: str | Path, *, dataset_id: str) -> None:
    path = Path(registry_path).expanduser().resolve()
    registry = _load_registry(path)
    entry = registry.get("datasets", {}).get(dataset_id)
    if entry is None:
        raise KeyError(f"Unknown locked dataset registry id: {dataset_id}")
    entry["status"] = "spent"
    entry["spent_at"] = datetime.now(timezone.utc).isoformat()
    _write_json(path, registry)


def _complete_locked_run(registry_path: Path, *, dataset_id: str) -> None:
    registry = _load_registry(registry_path)
    entry = registry["datasets"][dataset_id]
    entry["status"] = "completed_aggregate_only"
    entry["completed_at"] = datetime.now(timezone.utc).isoformat()
    _write_json(registry_path, registry)


def run_evidence30_suite(
    *,
    dataset: Evidence30Dataset,
    split: str,
    strategies: Sequence[str],
    config: Mapping[str, Any],
    parquet_path: str | Path,
    video_dir: str | Path,
    subtitle_dir: str | Path,
    output_dir: str | Path,
    project_root: str | Path,
    freeze_manifest: str | Path | None = None,
    locked_registry: str | Path | None = None,
    session_factory: SessionFactory = VideoMMEStrategySession,
    progress: ProgressCallback | None = None,
) -> dict[str, Any]:
    if split not in {"dev", "locked"}:
        raise ValueError("run split must be dev or locked")
    normalized_strategies = list(dict.fromkeys(strategies))
    if not normalized_strategies or any(item not in STRATEGIES for item in normalized_strategies):
        raise ValueError(f"strategies must be selected from: {', '.join(STRATEGIES)}")

    freeze_id: str | None = None
    registry_path: Path | None = None
    dataset_id = sha256_file(dataset.root / "manifest.json")
    target = Path(output_dir).expanduser().resolve()
    if split == "locked":
        if freeze_manifest is None:
            raise ValueError("locked run requires --freeze-manifest")
        freeze_check = verify_freeze_manifest(
            freeze_manifest,
            runtime_config=config,
        )
        if not freeze_check["ok"]:
            raise RuntimeError(
                "Freeze mismatch: current source/config/model/annotations differ from freeze"
            )
        freeze_id = str(freeze_check["expected_freeze_id"])
        registry_path = Path(
            locked_registry
            or Path(project_root).expanduser().resolve()
            / "runs"
            / "evidence30"
            / "locked_registry.json"
        ).expanduser().resolve()
        _reserve_locked_run(
            registry_path,
            dataset_id=dataset_id,
            freeze_id=freeze_id,
            output_dir=target,
        )

    references = dataset.records(split)
    question_ids = [str(item["source"]["question_id"]) for item in references]
    questions = load_videomme_questions(parquet_path, question_ids=question_ids)
    question_by_id = {item.question_id: item for item in questions}
    run_signature = canonical_sha256(
        {
            "policy_id": POLICY_ID,
            "split": split,
            "strategies": normalized_strategies,
            "config": config,
            "dataset_id": dataset_id,
            "freeze_id": freeze_id,
        }
    )
    sealed_dir = target / "sealed"
    items_path = sealed_dir / "items.jsonl" if split == "locked" else target / "items.jsonl"
    run_manifest_path = (
        sealed_dir / "run_manifest.json" if split == "locked" else target / "run_manifest.json"
    )
    summary_path = target / "summary.json"
    target.mkdir(parents=True, exist_ok=True)
    existing_items = _read_jsonl(items_path)
    if run_manifest_path.is_file():
        prior = json.loads(run_manifest_path.read_text(encoding="utf-8"))
        if prior.get("run_signature") != run_signature:
            raise RuntimeError("Output directory belongs to a different Evidence30 run")
    else:
        _write_json(
            run_manifest_path,
            {
                "run_signature": run_signature,
                "freeze_id": freeze_id,
                "split": split,
                "strategies": normalized_strategies,
                "question_ids": question_ids,
                "policy_id": POLICY_ID,
                "created_at": datetime.now(timezone.utc).isoformat(),
            },
        )
    completed_keys = {
        (str(item.get("strategy")), str(item.get("question_id")))
        for item in existing_items
    }
    reference_by_id = {
        str(item["source"]["question_id"]): item for item in references
    }
    expected_items = len(references) * len(normalized_strategies)
    done = len(existing_items)

    for strategy in normalized_strategies:
        replay_dir = target / "replays" / strategy if split == "dev" else None
        session = session_factory(
            config,
            strategy=strategy,
            video_dir=video_dir,
            subtitle_dir=subtitle_dir,
            with_subtitles=True,
            native_video_decode=False,
            replay_dir=replay_dir,
        )
        with session:
            for question_id in question_ids:
                key = (strategy, question_id)
                if key in completed_keys:
                    continue
                question = question_by_id[question_id]
                reference = reference_by_id[question_id]
                try:
                    item = session.evaluate(question)
                except Exception as exc:  # noqa: BLE001 - suite records item-level failures
                    item = {
                        **question.to_dict(),
                        "strategy": strategy,
                        "with_subtitles": True,
                        "prediction": None,
                        "correct": False,
                        "wall_seconds": 0.0,
                        "model_output": "",
                        "metadata": {"exposures": []},
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                item["run_signature"] = run_signature
                item["freeze_id"] = freeze_id
                item["topology"] = reference["evidence_contract"]["primary_topology"]
                item["duration_bucket"] = reference["source"]["duration_bucket"]
                item["modality_key"] = "+".join(
                    sorted(reference["evidence_contract"]["required_modalities"])
                )
                item["relaxed_grounding"] = score_relaxed_grounding(
                    reference,
                    item.get("metadata", {}).get("exposures", []),
                )
                item["engineering"] = assess_engineering_record(item)
                _append_jsonl(items_path, item)
                existing_items.append(item)
                completed_keys.add(key)
                done += 1
                if progress is not None:
                    progress(
                        {
                            "split": split,
                            "strategy": strategy,
                            "completed": done,
                            "expected": expected_items,
                            "item": item if split == "dev" else None,
                        }
                    )

    summary = aggregate_suite(
        existing_items,
        expected_items=expected_items,
        suppress_groups_smaller_than=3 if split == "locked" else None,
    )
    summary.update(
        {
            "split": split,
            "run_signature": run_signature,
            "freeze_id": freeze_id,
            "visibility": "aggregate" if split == "locked" else "full",
        }
    )
    if split == "locked":
        summary["sealed_artifact"] = {
            "path": str(items_path),
            "sha256": sha256_file(items_path),
            "unsealed": False,
        }
    else:
        summary["items_path"] = str(items_path)
    _write_json(summary_path, summary)
    if split == "locked" and registry_path is not None:
        _complete_locked_run(registry_path, dataset_id=dataset_id)
    return summary


def summarize_existing_run(
    *,
    items_path: str | Path,
    expected_items: int,
    locked: bool,
) -> dict[str, Any]:
    items = _read_jsonl(Path(items_path).expanduser().resolve())
    return aggregate_suite(
        items,
        expected_items=expected_items,
        suppress_groups_smaller_than=3 if locked else None,
    )
