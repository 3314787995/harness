from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import re
import subprocess
import sys
import time
import traceback
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from qwen3vl_agent.config import load_config
from qwen3vl_agent.factory import build_model
from qwen3vl_agent.p01.agent import P01VideoAgent
from qwen3vl_agent.p01.types import P01Request, TimeSpan

BENCHMARKS = ("Video-MME", "MLVU", "LVBench")
RUNNER_SCHEMA_VERSION = 2


@dataclass(frozen=True)
class SmokeQuestion:
    benchmark: str
    native_question_id: str
    probe_category: str
    video_path: Path
    relative_video_path: str
    question: str
    choices: tuple[str, ...]
    choice_ids: tuple[str, ...]
    answer_label: str | None
    reference_answer: str | None
    scoring_points: tuple[str, ...]
    given_interval: TimeSpan | None
    source: dict[str, Any]

    @property
    def key(self) -> str:
        return f"{self.benchmark}::{self.native_question_id}"

    @property
    def slug(self) -> str:
        value = re.sub(r"[^A-Za-z0-9._-]+", "_", self.key)
        return value.strip("._") or "question"

    def metadata(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "benchmark": self.benchmark,
            "native_question_id": self.native_question_id,
            "probe_category": self.probe_category,
            "video_path": str(self.video_path),
            "relative_video_path": self.relative_video_path,
            "question": self.question,
            "choices": [
                {"id": option_id, "text": text}
                for option_id, text in zip(self.choice_ids, self.choices)
            ],
            "answer_label": self.answer_label,
            "reference_answer": self.reference_answer,
            "scoring_points": list(self.scoring_points),
            "given_interval": (
                self.given_interval.to_dict() if self.given_interval is not None else None
            ),
        }


def _require_text(value: Any, field_name: str, source: Path, line_number: int) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{source}:{line_number}: {field_name} must not be empty")
    return text


def _safe_media_path(benchmark_root: Path, relative: str) -> Path:
    candidate = Path(relative)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise ValueError(f"media path must stay relative to benchmark root: {relative}")
    resolved = (benchmark_root / candidate).resolve()
    if not resolved.is_relative_to(benchmark_root.resolve()):
        raise ValueError(f"media path escapes benchmark root: {relative}")
    return resolved


def _parse_choices(
    value: Any,
    source: Path,
    line_number: int,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    if value in (None, []):
        return (), ()
    if not isinstance(value, list):
        raise TypeError(f"{source}:{line_number}: choices must be a list")
    ids: list[str] = []
    texts: list[str] = []
    for index, option in enumerate(value):
        if isinstance(option, dict):
            option_id = _require_text(
                option.get("id"),
                f"choices[{index}].id",
                source,
                line_number,
            )
            text = _require_text(
                option.get("text"),
                f"choices[{index}].text",
                source,
                line_number,
            )
        else:
            option_id = chr(ord("A") + index)
            text = _require_text(option, f"choices[{index}]", source, line_number)
        ids.append(option_id)
        texts.append(text)
    if len(set(ids)) != len(ids):
        raise ValueError(f"{source}:{line_number}: duplicate choice IDs")
    return tuple(ids), tuple(texts)


def _parse_interval(value: Any, source: Path, line_number: int) -> TimeSpan | None:
    if value is None:
        return None
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError(f"{source}:{line_number}: given_interval_sec must be [START, END]")
    return TimeSpan(float(value[0]), float(value[1]), source="given_interval")


def load_smoke_questions(
    data_root: str | Path,
    *,
    benchmarks: Sequence[str] = BENCHMARKS,
    subset: str = "p01-smoke-v1",
) -> list[SmokeQuestion]:
    root = Path(data_root).expanduser().resolve()
    questions: list[SmokeQuestion] = []
    seen: set[str] = set()
    for benchmark in benchmarks:
        if benchmark not in BENCHMARKS:
            raise ValueError(f"unsupported benchmark: {benchmark}")
        benchmark_root = root / benchmark
        manifest = benchmark_root / "subsets" / subset / "questions.jsonl"
        if not manifest.is_file():
            raise FileNotFoundError(f"question manifest does not exist: {manifest}")
        with manifest.open("r", encoding="utf-8-sig") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{manifest}:{line_number}: invalid JSON: {exc}") from exc
                if record.get("benchmark") != benchmark:
                    raise ValueError(
                        f"{manifest}:{line_number}: benchmark field does not match directory"
                    )
                if record.get("media_status") != "available":
                    raise ValueError(f"{manifest}:{line_number}: media_status must be available")
                native_id = _require_text(
                    record.get("native_question_id"),
                    "native_question_id",
                    manifest,
                    line_number,
                )
                relative_video = _require_text(
                    record.get("video_path"),
                    "video_path",
                    manifest,
                    line_number,
                )
                video_path = _safe_media_path(benchmark_root, relative_video)
                if not video_path.is_file():
                    raise FileNotFoundError(
                        f"{manifest}:{line_number}: video does not exist: {video_path}"
                    )
                choice_ids, choices = _parse_choices(
                    record.get("choices"),
                    manifest,
                    line_number,
                )
                item = SmokeQuestion(
                    benchmark=benchmark,
                    native_question_id=native_id,
                    probe_category=_require_text(
                        record.get("probe_category"),
                        "probe_category",
                        manifest,
                        line_number,
                    ),
                    video_path=video_path,
                    relative_video_path=relative_video,
                    question=_require_text(
                        record.get("question"),
                        "question",
                        manifest,
                        line_number,
                    ),
                    choices=choices,
                    choice_ids=choice_ids,
                    answer_label=(
                        str(record["answer_label"]).strip()
                        if record.get("answer_label") is not None
                        else None
                    ),
                    reference_answer=(
                        str(record["reference_answer"]).strip()
                        if record.get("reference_answer") is not None
                        else None
                    ),
                    scoring_points=tuple(
                        str(point).strip()
                        for point in record.get("scoring_points", [])
                        if str(point).strip()
                    ),
                    given_interval=_parse_interval(
                        record.get("given_interval_sec"),
                        manifest,
                        line_number,
                    ),
                    source=record,
                )
                if item.key in seen:
                    raise ValueError(f"duplicate smoke question key: {item.key}")
                seen.add(item.key)
                questions.append(item)
    if not questions:
        raise ValueError("no smoke questions were loaded")
    return questions


def select_questions(
    questions: Sequence[SmokeQuestion],
    *,
    question_ids: Sequence[str] = (),
    probes: Sequence[str] = (),
    order: str = "video",
    limit: int | None = None,
) -> list[SmokeQuestion]:
    selected = list(questions)
    if question_ids:
        ordered: list[SmokeQuestion] = []
        for requested in question_ids:
            matches = [
                item
                for item in selected
                if item.key == requested or item.native_question_id == requested
            ]
            if not matches:
                raise ValueError(f"unknown question ID: {requested}")
            if len(matches) > 1:
                keys = ", ".join(item.key for item in matches)
                raise ValueError(f"ambiguous question ID {requested!r}; use one of: {keys}")
            if matches[0] not in ordered:
                ordered.append(matches[0])
        selected = ordered
    if probes:
        requested_probes = set(probes)
        selected = [item for item in selected if item.probe_category in requested_probes]
        missing = requested_probes - {item.probe_category for item in selected}
        if missing:
            raise ValueError(f"unknown or empty probe filters: {', '.join(sorted(missing))}")
    if not question_ids:
        benchmark_rank = {name: index for index, name in enumerate(BENCHMARKS)}
        if order == "video":
            selected.sort(
                key=lambda item: (
                    benchmark_rank[item.benchmark],
                    item.relative_video_path,
                    item.native_question_id,
                )
            )
        elif order == "probe":
            selected.sort(
                key=lambda item: (
                    item.probe_category,
                    benchmark_rank[item.benchmark],
                    item.native_question_id,
                )
            )
        elif order != "manifest":
            raise ValueError(f"unsupported question order: {order}")
    if limit is not None:
        if limit < 1:
            raise ValueError("limit must be positive")
        selected = selected[:limit]
    if not selected:
        raise ValueError("question filters selected no records")
    return selected


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _existing_record(output_dir: Path, question: SmokeQuestion) -> dict[str, Any] | None:
    path = output_dir / "items" / f"{question.slug}.json"
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _git_snapshot(project_root: Path) -> dict[str, Any]:
    def run(*args: str) -> str | None:
        try:
            completed = subprocess.run(
                ["git", *args],
                cwd=project_root,
                check=True,
                capture_output=True,
                text=True,
            )
        except (FileNotFoundError, subprocess.CalledProcessError):
            return None
        return completed.stdout.strip()

    status = run("status", "--short")
    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(status),
        "status": status.splitlines() if status else [],
    }


def environment_snapshot(config: dict[str, Any], project_root: Path) -> dict[str, Any]:
    packages = {
        name: _package_version(name)
        for name in (
            "torch",
            "torchvision",
            "transformers",
            "accelerate",
            "flash-attn",
            "qwen-vl-utils",
            "av",
            "Pillow",
        )
    }
    gpu: dict[str, Any] = {}
    try:
        import torch

        gpu = {
            "torch_cuda_version": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "device_count": torch.cuda.device_count(),
        }
        if torch.cuda.is_available():
            devices = []
            for device in range(torch.cuda.device_count()):
                properties = torch.cuda.get_device_properties(device)
                devices.append(
                    {
                        "device_index": device,
                        "device_name": properties.name,
                        "total_memory_bytes": properties.total_memory,
                        "compute_capability": list(torch.cuda.get_device_capability(device)),
                    }
                )
            gpu["devices"] = devices
    except Exception as exc:  # noqa: BLE001  # pragma: no cover - rental host
        gpu = {"probe_error": f"{type(exc).__name__}: {exc}"}
    model_config = dict(config.get("model") or {})
    return {
        "schema_version": RUNNER_SCHEMA_VERSION,
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "packages": packages,
        "gpu": gpu,
        "model_config": model_config,
        "p01_config": dict(config.get("p01") or {}),
        "git": _git_snapshot(project_root),
    }


def _reset_gpu_peaks() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            for device in range(torch.cuda.device_count()):
                torch.cuda.reset_peak_memory_stats(device)
    except Exception:  # noqa: BLE001  # GPU telemetry must not abort a run
        return


def _gpu_usage() -> dict[str, Any]:
    try:
        import torch

        if not torch.cuda.is_available():
            return {"cuda_available": False}
        torch.cuda.synchronize()
        devices = []
        for device in range(torch.cuda.device_count()):
            free_bytes, total_bytes = torch.cuda.mem_get_info(device)
            devices.append(
                {
                    "device_index": device,
                    "max_memory_allocated_bytes": torch.cuda.max_memory_allocated(device),
                    "max_memory_reserved_bytes": torch.cuda.max_memory_reserved(device),
                    "free_memory_bytes": free_bytes,
                    "total_memory_bytes": total_bytes,
                }
            )
        return {
            "cuda_available": True,
            "device_count": len(devices),
            "devices": devices,
            "max_memory_allocated_bytes": max(
                (item["max_memory_allocated_bytes"] for item in devices), default=0
            ),
            "max_memory_reserved_bytes": max(
                (item["max_memory_reserved_bytes"] for item in devices), default=0
            ),
        }
    except Exception as exc:  # noqa: BLE001  # GPU telemetry is best-effort
        return {"probe_error": f"{type(exc).__name__}: {exc}"}


def _release_after_error() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001  # cleanup is best-effort after an error
        return


def run_question(
    agent: P01VideoAgent,
    question: SmokeQuestion,
    *,
    force_choice: bool,
) -> dict[str, Any]:
    _reset_gpu_peaks()
    started = time.perf_counter()
    try:
        result = agent.solve(
            P01Request(
                video_path=str(question.video_path),
                question=question.question,
                choices=question.choices,
                given_interval=question.given_interval,
                force_choice=force_choice and bool(question.choices),
            )
        )
        result_payload = result.to_dict()
        runner_status = "completed"
        error = None
    except Exception as exc:  # noqa: BLE001  # pragma: no cover - runtime boundary
        result_payload = None
        runner_status = "error"
        error = {
            "type": type(exc).__name__,
            "message": str(exc),
            "is_cuda_oom": type(exc).__name__ == "OutOfMemoryError"
            or "out of memory" in str(exc).lower(),
            "traceback": traceback.format_exc(),
        }
        _release_after_error()
    elapsed = time.perf_counter() - started
    prediction = None
    if result_payload is not None:
        prediction = result_payload.get("prediction")
        if prediction is None:
            prediction = result_payload.get("verified_answer")
        if prediction is None:
            prediction = result_payload.get("forced_prediction")
    return {
        "schema_version": RUNNER_SCHEMA_VERSION,
        "runner_status": runner_status,
        "question": question.metadata(),
        "result": result_payload,
        "evaluation": {
            "prediction": prediction,
            "label_exact_match": (
                prediction == question.answer_label
                if prediction is not None and question.answer_label is not None
                else None
            ),
        },
        "runtime": {
            "elapsed_seconds": round(elapsed, 6),
            "gpu": _gpu_usage(),
        },
        "error": error,
    }


def _metric_summary(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    runner_counts = Counter(str(record.get("runner_status")) for record in records)
    p01_statuses = Counter(
        str(record["result"].get("status"))
        for record in records
        if isinstance(record.get("result"), dict)
    )
    decision_sources = Counter(
        str(record["result"].get("decision_source"))
        for record in records
        if isinstance(record.get("result"), dict)
    )
    support_levels = Counter(
        str(record["result"].get("support_level"))
        for record in records
        if isinstance(record.get("result"), dict)
    )
    evaluated = [
        record["evaluation"]["label_exact_match"]
        for record in records
        if record.get("evaluation", {}).get("label_exact_match") is not None
    ]
    mcq_records = [
        record
        for record in records
        if isinstance(record.get("question", {}).get("choices"), list)
        and bool(record["question"]["choices"])
    ]
    free_text_records = [
        record
        for record in records
        if isinstance(record.get("question", {}).get("choices"), list)
        and not record["question"]["choices"]
    ]
    predicted_mcq = sum(
        1
        for record in mcq_records
        if isinstance(record.get("result"), dict) and record["result"].get("prediction") is not None
    )
    exact_matches = sum(bool(value) for value in evaluated)
    predicted_free_text = sum(
        1
        for record in free_text_records
        if isinstance(record.get("result"), dict)
        and bool(str(record["result"].get("prediction") or "").strip())
    )
    predicted_all = predicted_mcq + predicted_free_text
    stop_reasons = Counter(
        str(record["result"].get("trace", {}).get("stop_reason"))
        for record in records
        if isinstance(record.get("result"), dict)
        and record["result"].get("trace", {}).get("stop_reason") is not None
    )
    elapsed = sum(float(record.get("runtime", {}).get("elapsed_seconds", 0)) for record in records)
    model_calls = sum(
        int(record.get("result", {}).get("resources", {}).get("model_call_count", 0))
        for record in records
        if isinstance(record.get("result"), dict)
    )
    return {
        "record_count": len(records),
        "runner_status_counts": dict(sorted(runner_counts.items())),
        "p01_status_counts": dict(sorted(p01_statuses.items())),
        "decision_source_counts": dict(sorted(decision_sources.items())),
        "support_level_counts": dict(sorted(support_levels.items())),
        "stop_reason_counts": dict(sorted(stop_reasons.items())),
        "answer_rate": (p01_statuses.get("answered", 0) / len(records) if records else None),
        "prediction_coverage": predicted_all / len(records) if records else None,
        "mcq_total": len(mcq_records),
        "mcq_predictions": predicted_mcq,
        "mcq_prediction_coverage": (predicted_mcq / len(mcq_records) if mcq_records else None),
        "mcq_predictions_scored": len(evaluated),
        "mcq_label_exact_matches": exact_matches,
        "mcq_label_accuracy": exact_matches / len(evaluated) if evaluated else None,
        "mcq_strict_label_accuracy": (exact_matches / len(mcq_records) if mcq_records else None),
        "g39_total": len(free_text_records),
        "g39_predictions": predicted_free_text,
        "g39_prediction_coverage": (
            predicted_free_text / len(free_text_records) if free_text_records else None
        ),
        "total_model_calls": model_calls,
        "total_elapsed_seconds": round(elapsed, 6),
    }


def summarize_records(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    metrics = _metric_summary(records)
    engineering_failures = [
        {
            "key": record.get("question", {}).get("key"),
            "failures": record.get("result", {})
            .get("trace", {})
            .get("engineering_acceptance", {})
            .get("failures", []),
        }
        for record in records
        if isinstance(record.get("result"), dict)
        and record["result"]
        .get("trace", {})
        .get("engineering_acceptance", {})
        .get("passed")
        is False
    ]
    peak_reserved = max(
        (
            int(record.get("runtime", {}).get("gpu", {}).get("max_memory_reserved_bytes", 0))
            for record in records
        ),
        default=0,
    )
    return {
        "schema_version": RUNNER_SCHEMA_VERSION,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        **metrics,
        "engineering_acceptance": {
            "expected_mcq": 20,
            "expected_g39": 5,
            "mcq_full_output": metrics["mcq_total"] == 20 and metrics["mcq_predictions"] == 20,
            "g39_full_output": metrics["g39_total"] == 5 and metrics["g39_predictions"] == 5,
            "failure_count": len(engineering_failures),
            "failures": engineering_failures,
            "passed": metrics["mcq_total"] == 20
            and metrics["mcq_predictions"] == 20
            and metrics["g39_total"] == 5
            and metrics["g39_predictions"] == 5
            and not engineering_failures,
        },
        "peak_gpu_memory_reserved_bytes": peak_reserved or None,
        "by_probe": {
            probe: _metric_summary(
                [
                    record
                    for record in records
                    if record.get("question", {}).get("probe_category") == probe
                ]
            )
            for probe in sorted(
                {
                    str(record.get("question", {}).get("probe_category"))
                    for record in records
                    if record.get("question", {}).get("probe_category")
                }
            )
        },
    }


def _load_item_records(output_dir: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted((output_dir / "items").glob("*.json")):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(value, dict):
            records.append(value)
    return records


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the P01 25-question smoke set in one model process."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--benchmark",
        action="append",
        choices=BENCHMARKS,
        default=[],
        help="Repeat to restrict benchmarks; defaults to all three.",
    )
    parser.add_argument("--probe", action="append", default=[])
    parser.add_argument(
        "--question-id",
        action="append",
        default=[],
        help="Native ID or BENCHMARK::ID; repeated values preserve requested order.",
    )
    parser.add_argument("--order", choices=("manifest", "video", "probe"), default="video")
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--force-choice",
        action="store_true",
        help="Deprecated compatibility flag; P01 v2 always predicts valid MCQs.",
    )
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--rerun-errors", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config_path = Path(args.config).expanduser().resolve()
    data_root = Path(args.data_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    benchmarks = tuple(args.benchmark or BENCHMARKS)
    questions = load_smoke_questions(data_root, benchmarks=benchmarks)
    selected = select_questions(
        questions,
        question_ids=args.question_id,
        probes=args.probe,
        order=args.order,
        limit=args.limit,
    )
    plan = {
        "schema_version": RUNNER_SCHEMA_VERSION,
        "config": str(config_path),
        "strategy": "p01-v2",
        "data_root": str(data_root),
        "output_dir": str(output_dir),
        "force_choice": args.force_choice,
        "resume": not args.no_resume,
        "questions": [item.metadata() for item in selected],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_json(output_dir / "run_plan.json", plan)
    if args.dry_run:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return 0

    config = load_config(config_path)
    project_root = Path(__file__).resolve().parents[2]
    _atomic_json(
        output_dir / "environment.json",
        environment_snapshot(config, project_root),
    )
    model = build_model(config.get("model"))
    agent = P01VideoAgent(model, config=config.get("p01"))
    agent.load()
    infrastructure_errors = 0
    try:
        for index, question in enumerate(selected, start=1):
            existing = _existing_record(output_dir, question)
            should_skip = (
                not args.no_resume
                and existing is not None
                and not (args.rerun_errors and existing.get("runner_status") == "error")
            )
            if should_skip:
                print(f"[{index}/{len(selected)}] skip {question.key} (resume)")
                continue
            print(f"[{index}/{len(selected)}] run {question.key} [{question.probe_category}]")
            record = run_question(agent, question, force_choice=args.force_choice)
            item_path = output_dir / "items" / f"{question.slug}.json"
            _atomic_json(item_path, record)
            _append_jsonl(output_dir / "results.jsonl", record)
            if record["runner_status"] == "error":
                infrastructure_errors += 1
                print(f"  error: {record['error']['type']}: {record['error']['message']}")
                if args.fail_fast:
                    break
            else:
                result = record["result"]
                print(
                    "  "
                    f"status={result['status']} "
                    f"prediction={result['prediction']!r} "
                    f"source={result['decision_source']} "
                    f"support={result['support_level']} "
                    f"calls={result['resources']['model_call_count']}"
                )
    finally:
        agent.unload()

    records = _load_item_records(output_dir)
    summary = summarize_records(records)
    _atomic_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 2 if infrastructure_errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
