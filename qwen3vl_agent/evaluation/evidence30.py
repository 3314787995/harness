from __future__ import annotations

import copy
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from qwen3vl_agent.evaluation.videomme import VideoMMEQuestion

STRATEGIES = ("direct", "coarse_to_fine", "active_tree")
POLICY_ID = "evidence30-relaxed-debug/1.0"

_SUBTITLE_INTERVAL = re.compile(
    r"\[(?P<start>\d+(?:\.\d+)?)s-(?P<end>\d+(?:\.\d+)?)s\]"
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class Exposure:
    """One piece of source media that was actually supplied to a model call."""

    stage: str
    modality: str
    start_seconds: float
    end_seconds: float
    frame_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "modality": self.modality,
            "start_seconds": round(self.start_seconds, 6),
            "end_seconds": round(self.end_seconds, 6),
            "frame_id": self.frame_id,
        }


def subtitle_exposures(text: str, *, stage: str) -> list[Exposure]:
    return [
        Exposure(
            stage=stage,
            modality="subtitle",
            start_seconds=float(match.group("start")),
            end_seconds=float(match.group("end")),
        )
        for match in _SUBTITLE_INTERVAL.finditer(text or "")
    ]


def _frame_exposures(frames: Any, *, stage: str) -> list[Exposure]:
    if not isinstance(frames, list):
        return []
    exposures: list[Exposure] = []
    for frame in frames:
        if not isinstance(frame, dict) or "timestamp_seconds" not in frame:
            continue
        timestamp = float(frame["timestamp_seconds"])
        exposures.append(
            Exposure(
                stage=stage,
                modality="visual",
                start_seconds=timestamp,
                end_seconds=timestamp,
                frame_id=str(frame.get("id")) if frame.get("id") is not None else None,
            )
        )
    return exposures


def normalize_exposures(strategy: str, metadata: dict[str, Any]) -> list[dict[str, Any]]:
    """Normalize existing strategy traces without consulting an Evidence30 record."""

    if strategy not in STRATEGIES:
        raise ValueError(f"Unsupported strategy: {strategy}")
    exposures: list[Exposure] = []
    if strategy == "direct":
        trace = metadata.get("direct_preprocessing", {})
        exposures.extend(_frame_exposures(trace.get("frames"), stage="direct_answer"))
        for item in trace.get("subtitle_intervals", []):
            if not isinstance(item, dict):
                continue
            exposures.append(
                Exposure(
                    "direct_answer",
                    "subtitle",
                    float(item["start_seconds"]),
                    float(item["end_seconds"]),
                )
            )
    elif strategy == "coarse_to_fine":
        trace = metadata.get("coarse_to_fine", {})
        glance = trace.get("glance", {})
        exposures.extend(_frame_exposures(glance.get("frames"), stage="video_glance"))
        for round_trace in trace.get("rounds", []):
            if not isinstance(round_trace, dict):
                continue
            round_id = round_trace.get("round", "unknown")
            exposures.extend(
                _frame_exposures(
                    round_trace.get("selection_frames"),
                    stage=f"round_{round_id}_selection",
                )
            )
            exposures.extend(
                _frame_exposures(
                    round_trace.get("frames"),
                    stage=f"round_{round_id}_reasoning",
                )
            )
            candidate_subtitles = round_trace.get("candidate_subtitles", {})
            if isinstance(candidate_subtitles, dict):
                for value in candidate_subtitles.values():
                    exposures.extend(
                        subtitle_exposures(
                            str(value),
                            stage=f"round_{round_id}_selection",
                        )
                    )
            exposures.extend(
                subtitle_exposures(
                    str(round_trace.get("subtitles", "")),
                    stage=f"round_{round_id}_reasoning",
                )
            )
        exposures.extend(
            _frame_exposures(trace.get("fallback_frames"), stage="degraded_fallback")
        )
        exposures.extend(
            subtitle_exposures(
                str(trace.get("fallback_subtitles", "")),
                stage="degraded_fallback",
            )
        )
    else:
        trace = metadata.get("active_tree", {})
        for event in trace.get("events", []):
            if not isinstance(event, dict):
                continue
            event_type = str(event.get("type", "active_tree_event"))
            exposures.extend(_frame_exposures(event.get("frames"), stage=event_type))
            exposures.extend(
                subtitle_exposures(str(event.get("subtitles", "")), stage=event_type)
            )
            subtitles_by_node = event.get("subtitles_by_node", {})
            if isinstance(subtitles_by_node, dict):
                for value in subtitles_by_node.values():
                    exposures.extend(subtitle_exposures(str(value), stage=event_type))

    unique: dict[tuple[Any, ...], Exposure] = {}
    for item in exposures:
        key = (
            item.stage,
            item.modality,
            round(item.start_seconds, 6),
            round(item.end_seconds, 6),
            item.frame_id,
        )
        unique[key] = item
    return [item.to_dict() for item in unique.values()]


def _overlaps(left_start: float, left_end: float, right_start: float, right_end: float) -> bool:
    return left_end >= right_start and left_start <= right_end


def _compatible(reference_modality: str, exposure_modality: str) -> bool:
    if reference_modality == "subtitle":
        return exposure_modality == "subtitle"
    if reference_modality in {"visual", "ocr"}:
        return exposure_modality in {"visual", "ocr"}
    return reference_modality == exposure_modality


def _interval_hit(
    interval: dict[str, Any],
    modality: str,
    exposures: list[dict[str, Any]],
) -> bool:
    start = float(interval["start_sec"])
    end = float(interval["end_sec"])
    return any(
        _compatible(modality, str(item.get("modality", "")))
        and _overlaps(
            float(item.get("start_seconds", -1)),
            float(item.get("end_seconds", -1)),
            start,
            end,
        )
        for item in exposures
    )


def score_relaxed_grounding(
    reference: dict[str, Any],
    exposures: list[dict[str, Any]],
) -> dict[str, Any]:
    """Score temporal/media exposure only; never compare generated semantic text to gold."""

    contract = reference["evidence_contract"]
    required_slots = {
        str(item["slot_id"])
        for item in contract["evidence_slots"]
        if bool(item.get("required", True))
    }
    item_results: list[dict[str, Any]] = []
    set_results: list[dict[str, Any]] = []
    for evidence_set in contract["sufficient_evidence_sets"]:
        hit_slots: set[str] = set()
        set_item_ids: list[str] = []
        context_hits = 0
        core_hits = 0
        for item in evidence_set["items"]:
            item_id = str(item["evidence_id"])
            slot_id = str(item["slot_id"])
            modality = str(item["modality"])
            context_hit = _interval_hit(item["context_interval"], modality, exposures)
            core_hit = _interval_hit(item["core_interval"], modality, exposures)
            if context_hit:
                hit_slots.add(slot_id)
                context_hits += 1
            if core_hit:
                core_hits += 1
            set_item_ids.append(item_id)
            item_results.append(
                {
                    "set_id": str(evidence_set["set_id"]),
                    "evidence_id": item_id,
                    "slot_id": slot_id,
                    "modality": modality,
                    "context_hit": context_hit,
                    "core_hit": core_hit,
                }
            )
        set_grounded = required_slots <= hit_slots
        set_results.append(
            {
                "set_id": str(evidence_set["set_id"]),
                "item_ids": set_item_ids,
                "context_hit_items": context_hits,
                "core_hit_items": core_hits,
                "covered_slot_ids": sorted(hit_slots),
                "grounded": set_grounded,
            }
        )

    covered_slots = {
        item["slot_id"] for item in item_results if bool(item["context_hit"])
    }
    required_items = [item for item in item_results]
    hard_negative_hits = 0
    for negative in reference.get("hard_negatives", []):
        if any(
            _interval_hit(negative["interval"], modality, exposures)
            for modality in negative.get("modalities", [])
        ):
            hard_negative_hits += 1
    return {
        "policy_id": POLICY_ID,
        "required_slot_ids": sorted(required_slots),
        "covered_slot_ids": sorted(covered_slots),
        "slot_coverage": (
            len(covered_slots & required_slots) / len(required_slots)
            if required_slots
            else 1.0
        ),
        "context_item_recall": (
            sum(bool(item["context_hit"]) for item in required_items) / len(required_items)
            if required_items
            else 1.0
        ),
        "core_item_recall": (
            sum(bool(item["core_hit"]) for item in required_items) / len(required_items)
            if required_items
            else 1.0
        ),
        "grounded": any(bool(item["grounded"]) for item in set_results),
        "hard_negative_exposure_count": hard_negative_hits,
        "items": item_results,
        "sets": set_results,
    }


def _model_metadata_records(value: Any) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if isinstance(value, dict):
        model_metadata = value.get("model_metadata")
        if isinstance(model_metadata, dict):
            records.append(model_metadata)
        for key, item in value.items():
            if key == "model_metadata":
                continue
            records.extend(_model_metadata_records(item))
    elif isinstance(value, list):
        for item in value:
            records.extend(_model_metadata_records(item))
    return records


def assess_engineering_record(record: dict[str, Any]) -> dict[str, Any]:
    strategy = str(record.get("strategy", ""))
    prediction = record.get("prediction")
    metadata = record.get("metadata", {})
    trace_valid = False
    degraded = False
    budget_violation = False
    stop_reason = ""
    model_calls = 0
    input_tokens = 0
    output_tokens = 0

    if strategy == "direct":
        preprocessing = metadata.get("direct_preprocessing", {})
        trace_valid = isinstance(preprocessing, dict) and bool(metadata.get("exposures"))
        stop_reason = "single_pass_complete"
        model_calls = 1
        input_tokens = int(metadata.get("input_tokens", 0))
        output_tokens = int(metadata.get("output_tokens", 0))
    elif strategy == "coarse_to_fine":
        trace = metadata.get("coarse_to_fine", {})
        stop_reason = str(trace.get("stop_reason", ""))
        degraded = bool(trace.get("degraded", False))
        budget = trace.get("budget", {})
        trace_valid = isinstance(trace, dict) and bool(stop_reason) and isinstance(budget, dict)
        if budget:
            budget_violation = (
                int(budget.get("unique_frames", 0)) > int(budget.get("unique_limit", 0))
                or int(budget.get("cumulative_views", 0))
                > int(budget.get("cumulative_limit", 0))
            )
        calls = _model_metadata_records(trace)
        model_calls = len(calls)
        input_tokens = sum(int(item.get("input_tokens", 0)) for item in calls)
        output_tokens = sum(int(item.get("output_tokens", 0)) for item in calls)
    elif strategy == "active_tree":
        trace = metadata.get("active_tree", {})
        stop_reason = str(trace.get("stop_reason", ""))
        degraded = bool(trace.get("degraded", False))
        resources = trace.get("resources", {})
        trace_valid = (
            isinstance(trace, dict)
            and bool(stop_reason)
            and isinstance(resources, dict)
            and "model_call_count" in resources
        )
        model_calls = int(resources.get("model_call_count", 0))
        input_tokens = int(resources.get("input_tokens", 0))
        output_tokens = int(resources.get("output_tokens", 0))
        budget_violation = model_calls > int(resources.get("max_model_calls", 0))

    valid_prediction = isinstance(prediction, str) and prediction in {"A", "B", "C", "D"}
    has_error = bool(record.get("error"))
    return {
        "completed": not has_error and valid_prediction and trace_valid,
        "valid_prediction": valid_prediction,
        "trace_valid": trace_valid,
        "degraded": degraded,
        "budget_violation": budget_violation,
        "fatal": has_error or not trace_valid,
        "stop_reason": stop_reason,
        "model_calls": model_calls,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
    }


class Evidence30Dataset:
    """Immutable view of a versioned Evidence30 artifact directory."""

    def __init__(
        self,
        root: str | Path,
        manifest: dict[str, Any],
        records: dict[str, dict[str, Any]],
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        self.manifest = manifest
        self._records = records

    @classmethod
    def load(cls, root: str | Path) -> Evidence30Dataset:
        resolved = Path(root).expanduser().resolve()
        manifest = json.loads((resolved / "manifest.json").read_text(encoding="utf-8"))
        records: dict[str, dict[str, Any]] = {}
        for name in ("dev.jsonl", "locked.jsonl"):
            for line in (resolved / name).read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                record = json.loads(line)
                question_id = str(record["source"]["question_id"])
                if question_id in records:
                    raise ValueError(f"Duplicate Evidence30 question_id: {question_id}")
                records[question_id] = record
        return cls(resolved, manifest, records)

    def records(self, split: str) -> list[dict[str, Any]]:
        if split not in {"dev", "locked", "all"}:
            raise ValueError("split must be dev, locked, or all")
        if split == "all":
            question_ids = list(self.manifest["selection"]["question_ids"])
        else:
            question_ids = list(self.manifest["splits"][f"{split}_question_ids"])
        missing = [question_id for question_id in question_ids if question_id not in self._records]
        if missing:
            raise ValueError(f"Manifest references missing records: {', '.join(missing)}")
        return [copy.deepcopy(self._records[question_id]) for question_id in question_ids]

    def question_ids(self, split: str) -> list[str]:
        return [str(record["source"]["question_id"]) for record in self.records(split)]

    def reference(self, question_id: str) -> dict[str, Any]:
        try:
            return copy.deepcopy(self._records[question_id])
        except KeyError as exc:
            raise KeyError(f"Unknown Evidence30 question_id: {question_id}") from exc


def _option_texts(row_options: Any) -> list[str]:
    values = row_options.tolist() if hasattr(row_options, "tolist") else list(row_options)
    return [re.sub(r"^\s*[A-D]\.\s*", "", str(item)).strip() for item in values]


def preflight_evidence30(
    dataset: Evidence30Dataset,
    *,
    schema_path: str | Path,
    parquet_path: str | Path,
    video_dir: str | Path,
    subtitle_dir: str | Path,
    verify_media_hashes: bool = True,
) -> dict[str, Any]:
    """Validate all 30 references without returning evidence content."""

    import pyarrow.parquet as pq
    from jsonschema import Draft202012Validator, FormatChecker

    errors: list[str] = []
    warnings: list[str] = []
    schema = json.loads(Path(schema_path).read_text(encoding="utf-8"))
    validator = Draft202012Validator(schema, format_checker=FormatChecker())
    records = dataset.records("all")
    for record in records:
        question_id = str(record["source"]["question_id"])
        for error in validator.iter_errors(record):
            errors.append(f"{question_id} {error.json_path}: {error.message}")

    artifact_mapping = {
        "dev": "dev.jsonl",
        "locked": "locked.jsonl",
        "excluded": "excluded.jsonl",
    }
    for key, name in artifact_mapping.items():
        expected = dataset.manifest.get("artifact_sha256", {}).get(key)
        path = dataset.root / name
        if not path.is_file():
            errors.append(f"missing artifact: {path}")
        elif expected != sha256_file(path):
            errors.append(f"artifact hash mismatch: {name}")

    parquet = Path(parquet_path).expanduser().resolve()
    expected_parquet_hash = (
        dataset.manifest.get("source_files", {})
        .get("question_parquet", {})
        .get("sha256")
    )
    if not parquet.is_file():
        errors.append(f"missing parquet: {parquet}")
        rows: dict[str, dict[str, Any]] = {}
    else:
        if expected_parquet_hash and sha256_file(parquet) != expected_parquet_hash:
            errors.append("question parquet hash mismatch")
        rows = {
            str(row["question_id"]): row
            for row in pq.read_table(parquet).to_pylist()
        }

    resolved_video_dir = Path(video_dir).expanduser().resolve()
    resolved_subtitle_dir = Path(subtitle_dir).expanduser().resolve()
    seen_videos: set[str] = set()
    media_bytes = 0
    for record in records:
        source = record["source"]
        question = record["question"]
        question_id = str(source["question_id"])
        row = rows.get(question_id)
        if row is None:
            errors.append(f"missing parquet question: {question_id}")
            continue
        expected_question = {
            "text": str(row["question"]),
            "options": _option_texts(row["options"]),
            "answer": str(row["answer"]).strip().upper(),
            "video_id": str(row["video_id"]),
        }
        actual_question = {
            "text": str(question["text"]),
            "options": [str(item["text"]) for item in question["options"]],
            "answer": str(question["official_answer"]["benchmark_label"]),
            "video_id": str(source["video_id"]),
        }
        if expected_question != actual_question:
            errors.append(f"parquet/reference mismatch: {question_id}")
        expected_question_hash = dataset.manifest["selection"]["question_hashes"].get(
            question_id
        )
        if expected_question_hash != canonical_sha256(question):
            errors.append(f"question hash mismatch: {question_id}")

        video_key = str(row["videoID"])
        if video_key in seen_videos:
            errors.append(f"duplicate source video: {video_key}")
        seen_videos.add(video_key)
        video_path = resolved_video_dir / f"{video_key}.mp4"
        if not video_path.is_file():
            errors.append(f"missing video for {question_id}")
        else:
            media_bytes += video_path.stat().st_size
            if (
                verify_media_hashes
                and source.get("video_sha256")
                and sha256_file(video_path) != source["video_sha256"]
            ):
                errors.append(f"video hash mismatch: {question_id}")

        subtitle_path = resolved_subtitle_dir / f"{video_key}.srt"
        subtitle_required = "subtitle" in record["evidence_contract"]["required_modalities"]
        if subtitle_required and not subtitle_path.is_file():
            errors.append(f"missing required subtitle: {question_id}")
        if (
            subtitle_path.is_file()
            and source.get("subtitle_sha256")
            and verify_media_hashes
            and sha256_file(subtitle_path) != source["subtitle_sha256"]
        ):
            errors.append(f"subtitle hash mismatch: {question_id}")

    contaminated = dataset.manifest.get("selection", {}).get(
        "trace_contaminated_question_ids", []
    )
    if contaminated:
        warnings.append(
            f"{len(contaminated)} trace-contaminated references are retained in dev"
        )
    warnings.append("references are AI-assisted internal debug data, not independent human gold")
    split_counts = {
        split: len(dataset.records(split)) for split in ("dev", "locked")
    }
    return {
        "policy_id": POLICY_ID,
        "ok": not errors,
        "records": len(records),
        "split_counts": split_counts,
        "unique_videos": len(seen_videos),
        "media_gib": round(media_bytes / (1024**3), 3),
        "verified_media_hashes": verify_media_hashes,
        "errors": errors,
        "warnings": warnings,
    }


def question_from_reference(record: dict[str, Any], parquet_row: dict[str, Any]) -> VideoMMEQuestion:
    """Construct model input from parquet only; reference evidence is deliberately ignored."""

    return VideoMMEQuestion.from_row(parquet_row)
