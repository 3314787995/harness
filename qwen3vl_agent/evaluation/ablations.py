from __future__ import annotations

import gc
import json
import math
import statistics
import time
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self

from qwen3vl_agent.coarse_to_fine import (
    CachedVideo,
    SubtitleTrack,
    TimeWindow,
    VideoEvidenceCache,
)
from qwen3vl_agent.coarse_to_fine.adapters import MultipleChoiceAdapter
from qwen3vl_agent.coarse_to_fine.types import FrameRef
from qwen3vl_agent.evaluation.evidence30 import (
    Evidence30Dataset,
    Exposure,
    canonical_sha256,
    score_relaxed_grounding,
    sha256_file,
    subtitle_exposures,
)
from qwen3vl_agent.evaluation.videomme import VideoMMEQuestion
from qwen3vl_agent.factory import build_model
from qwen3vl_agent.models.base import BaseVideoModel

PACKET_SOURCES = ("direct_replay", "active_tree_replay", "oracle_context")
ABLATION_POLICY_ID = "evidence30-oracle-unified-head/1.1"


@dataclass(frozen=True)
class AblationConfig:
    cache_dir: str
    sample_fps: float = 1.0
    cache_max_side: int = 768
    cache_jpeg_quality: int = 85
    cache_lru_size: int = 4
    max_frames: int = 16
    subtitle_max_chars: int = 8_000
    min_pixels: int = 4_096
    max_pixels: int = 131_072
    total_pixels: int = 2_097_152
    video_fps: float = 1.0
    max_new_tokens: int = 16

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> AblationConfig:
        data = dict(value or {})
        if "cache_dir" not in data:
            raise ValueError("ablation.cache_dir is required")
        config = cls(
            cache_dir=str(data["cache_dir"]),
            sample_fps=float(data.get("sample_fps", 1.0)),
            cache_max_side=int(data.get("cache_max_side", 768)),
            cache_jpeg_quality=int(data.get("cache_jpeg_quality", 85)),
            cache_lru_size=int(data.get("cache_lru_size", 4)),
            max_frames=int(data.get("max_frames", 16)),
            subtitle_max_chars=int(data.get("subtitle_max_chars", 8_000)),
            min_pixels=int(data.get("min_pixels", 4_096)),
            max_pixels=int(data.get("max_pixels", 131_072)),
            total_pixels=int(data.get("total_pixels", 2_097_152)),
            video_fps=float(data.get("video_fps", 1.0)),
            max_new_tokens=int(data.get("max_new_tokens", 16)),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if self.sample_fps <= 0 or self.video_fps <= 0:
            raise ValueError("ablation frame rates must be positive")
        if self.max_frames < 1:
            raise ValueError("ablation.max_frames must be positive")
        if self.subtitle_max_chars < 0:
            raise ValueError("ablation.subtitle_max_chars must be non-negative")
        if self.min_pixels < 1 or self.max_pixels < self.min_pixels:
            raise ValueError("invalid ablation per-frame pixel limits")
        if self.total_pixels < self.max_pixels:
            raise ValueError("ablation.total_pixels must be at least max_pixels")
        if self.max_new_tokens < 1:
            raise ValueError("ablation.max_new_tokens must be positive")

    def to_dict(self) -> dict[str, Any]:
        return {
            "cache_dir": self.cache_dir,
            "sample_fps": self.sample_fps,
            "cache_max_side": self.cache_max_side,
            "cache_jpeg_quality": self.cache_jpeg_quality,
            "cache_lru_size": self.cache_lru_size,
            "max_frames": self.max_frames,
            "subtitle_max_chars": self.subtitle_max_chars,
            "min_pixels": self.min_pixels,
            "max_pixels": self.max_pixels,
            "total_pixels": self.total_pixels,
            "video_fps": self.video_fps,
            "max_new_tokens": self.max_new_tokens,
        }


@dataclass(frozen=True)
class EvidencePacket:
    source: str
    frames: tuple[FrameRef, ...]
    subtitles: str
    source_frame_count: int
    subtitle_window_count: int
    priority_frame_count: int = 0
    oracle_set_id: str | None = None
    oracle_windows: tuple[TimeWindow, ...] = ()
    oracle_subtitle_windows: tuple[TimeWindow, ...] = ()

    def __post_init__(self) -> None:
        if self.source not in PACKET_SOURCES:
            raise ValueError(f"Unsupported packet source: {self.source}")
        if not self.frames:
            raise ValueError(f"{self.source} packet has no frames")

    def exposures(self) -> list[dict[str, Any]]:
        result = [
            Exposure(
                stage="unified_answer_head",
                modality="visual",
                start_seconds=frame.timestamp_seconds,
                end_seconds=frame.timestamp_seconds,
                frame_id=frame.id,
            ).to_dict()
            for frame in self.frames
        ]
        result.extend(
            item.to_dict()
            for item in subtitle_exposures(
                self.subtitles,
                stage="unified_answer_head",
            )
        )
        return result

    def metadata(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "source_frame_count": self.source_frame_count,
            "selected_frame_count": len(self.frames),
            "priority_frame_count": self.priority_frame_count,
            "frames": [frame.to_dict() for frame in self.frames],
            "subtitle_chars": len(self.subtitles),
            "subtitle_window_count": self.subtitle_window_count,
            "subtitle_intervals": [
                {
                    "start_seconds": item.start_seconds,
                    "end_seconds": item.end_seconds,
                }
                for item in subtitle_exposures(
                    self.subtitles,
                    stage="unified_answer_head",
                )
            ],
            "oracle_set_id": self.oracle_set_id,
            "oracle_windows": [window.to_dict() for window in self.oracle_windows],
            "oracle_subtitle_windows": [
                window.to_dict() for window in self.oracle_subtitle_windows
            ],
        }


def _frame_from_dict(value: Mapping[str, Any]) -> FrameRef:
    return FrameRef(
        id=str(value["id"]),
        timestamp_seconds=float(value["timestamp_seconds"]),
        path=str(Path(str(value["path"])).expanduser().resolve()),
    )


def _deduplicate_frames(frames: Iterable[FrameRef]) -> list[FrameRef]:
    by_id: dict[str, FrameRef] = {}
    for frame in frames:
        by_id.setdefault(frame.id, frame)
    return sorted(by_id.values(), key=lambda item: (item.timestamp_seconds, item.id))


def _evenly_select(values: Sequence[FrameRef], limit: int) -> list[FrameRef]:
    ordered = list(values)
    if limit < 1:
        return []
    if len(ordered) <= limit:
        return ordered
    indices = [min(len(ordered) - 1, math.floor((index + 0.5) * len(ordered) / limit)) for index in range(limit)]
    return [ordered[index] for index in indices]


def select_priority_frames(
    frames: Sequence[FrameRef],
    *,
    priority_ids: Iterable[str],
    limit: int,
) -> tuple[list[FrameRef], int]:
    """Select a deterministic matched-budget packet while retaining cited evidence first."""

    ordered = _deduplicate_frames(frames)
    priority = set(priority_ids)
    cited = [frame for frame in ordered if frame.id in priority]
    if len(cited) >= limit:
        selected = _evenly_select(cited, limit)
        return selected, len(selected)
    remaining = [frame for frame in ordered if frame.id not in priority]
    selected = cited + _evenly_select(remaining, limit - len(cited))
    return sorted(selected, key=lambda item: (item.timestamp_seconds, item.id)), len(cited)


def _merge_windows(windows: Iterable[TimeWindow], *, prefix: str) -> list[TimeWindow]:
    ordered = sorted(windows, key=lambda item: (item.start_seconds, item.end_seconds))
    merged: list[tuple[float, float]] = []
    for window in ordered:
        if not merged or window.start_seconds > merged[-1][1]:
            merged.append((window.start_seconds, window.end_seconds))
            continue
        merged[-1] = (merged[-1][0], max(merged[-1][1], window.end_seconds))
    return [
        TimeWindow(f"{prefix}-{index:02d}", start, end, depth=0)
        for index, (start, end) in enumerate(merged)
        if end > start
    ]


def _windows_from_exposures(exposures: Sequence[Mapping[str, Any]]) -> list[TimeWindow]:
    windows = [
        TimeWindow(
            f"TRACE-{index:04d}",
            float(item["start_seconds"]),
            max(
                float(item["end_seconds"]),
                float(item["start_seconds"]) + 0.001,
            ),
            depth=0,
        )
        for index, item in enumerate(exposures)
        if str(item.get("modality")) == "subtitle"
        and float(item.get("start_seconds", -1)) >= 0
        and float(item.get("end_seconds", -1)) >= float(item.get("start_seconds", -1))
    ]
    return _merge_windows(windows, prefix="TRACE")


def _interval_union_duration(items: Sequence[Mapping[str, Any]]) -> float:
    windows = [
        TimeWindow(
            f"COST-{index:02d}",
            float(item["context_interval"]["start_sec"]),
            max(
                float(item["context_interval"]["end_sec"]),
                float(item["context_interval"]["start_sec"]) + 0.001,
            ),
            depth=0,
        )
        for index, item in enumerate(items)
    ]
    return sum(window.duration_seconds for window in _merge_windows(windows, prefix="COST"))


def choose_oracle_set(reference: Mapping[str, Any]) -> Mapping[str, Any]:
    """Choose the least-duration valid sufficient set without consulting the answer."""

    contract = reference["evidence_contract"]
    required_slots = {
        str(slot["slot_id"])
        for slot in contract["evidence_slots"]
        if bool(slot.get("required", True))
    }
    candidates: list[tuple[float, int, Mapping[str, Any]]] = []
    for index, evidence_set in enumerate(contract["sufficient_evidence_sets"]):
        items = list(evidence_set["items"])
        covered = {str(item["slot_id"]) for item in items}
        if not required_slots <= covered:
            continue
        candidates.append((_interval_union_duration(items), index, evidence_set))
    if not candidates:
        raise ValueError("reference has no sufficient evidence set covering all required slots")
    return min(candidates, key=lambda item: (item[0], item[1]))[2]


def oracle_context_windows(
    reference: Mapping[str, Any],
    *,
    modalities: set[str] | None = None,
) -> tuple[str, list[TimeWindow]]:
    evidence_set = choose_oracle_set(reference)
    windows = [
        TimeWindow(
            f"ORACLE-RAW-{index:02d}",
            float(item["context_interval"]["start_sec"]),
            max(
                float(item["context_interval"]["end_sec"]),
                float(item["context_interval"]["start_sec"]) + 0.001,
            ),
            depth=0,
        )
        for index, item in enumerate(evidence_set["items"])
        if modalities is None or str(item["modality"]) in modalities
    ]
    return str(evidence_set["set_id"]), _merge_windows(windows, prefix="ORACLE")


def _allocate_window_frames(windows: Sequence[TimeWindow], total: int) -> list[int]:
    if not windows or total < 1:
        return []
    if len(windows) >= total:
        return [1 if index in set(_evenly_selected_indices(len(windows), total)) else 0 for index in range(len(windows))]
    counts = [1] * len(windows)
    remaining = total - len(windows)
    duration = sum(window.duration_seconds for window in windows)
    if remaining == 0 or duration <= 0:
        return counts
    exact = [remaining * window.duration_seconds / duration for window in windows]
    floors = [math.floor(value) for value in exact]
    counts = [count + floor for count, floor in zip(counts, floors)]
    leftover = remaining - sum(floors)
    order = sorted(
        range(len(windows)),
        key=lambda index: (-(exact[index] - floors[index]), index),
    )
    for index in order[:leftover]:
        counts[index] += 1
    return counts


def _evenly_selected_indices(length: int, limit: int) -> list[int]:
    if limit >= length:
        return list(range(length))
    return [min(length - 1, math.floor((index + 0.5) * length / limit)) for index in range(limit)]


def sample_cached_windows(
    cached: CachedVideo,
    windows: Sequence[TimeWindow],
    *,
    total: int,
) -> list[FrameRef]:
    selected: list[FrameRef] = []
    for window, count in zip(windows, _allocate_window_frames(windows, total)):
        if count:
            selected.extend(cached.uniform_frames(count, window))
    selected = _deduplicate_frames(selected)
    if len(selected) >= total:
        return _evenly_select(selected, total)
    candidates = [
        frame
        for frame in cached.frames
        if any(window.contains(frame.timestamp_seconds) for window in windows)
        and frame.id not in {item.id for item in selected}
    ]
    selected.extend(_evenly_select(candidates, total - len(selected)))
    return sorted(selected, key=lambda item: (item.timestamp_seconds, item.id))


def _trace_frames_direct(row: Mapping[str, Any]) -> list[FrameRef]:
    trace = row.get("metadata", {}).get("direct_preprocessing", {})
    return _deduplicate_frames(
        _frame_from_dict(frame)
        for frame in trace.get("frames", [])
        if isinstance(frame, Mapping)
    )


def _trace_frames_active(row: Mapping[str, Any]) -> tuple[list[FrameRef], set[str]]:
    trace = row.get("metadata", {}).get("active_tree", {})
    frames = _deduplicate_frames(
        _frame_from_dict(frame)
        for event in trace.get("events", [])
        if isinstance(event, Mapping)
        for frame in event.get("frames", [])
        if isinstance(frame, Mapping)
    )
    priority_ids = {
        str(frame_id)
        for evidence in trace.get("evidence_ledger", [])
        if isinstance(evidence, Mapping)
        for frame_id in evidence.get("source_frame_ids", [])
    }
    priority_ids.update(
        str(frame_id)
        for attempt in trace.get("verification_attempts", [])
        if isinstance(attempt, Mapping)
        for frame_id in attempt.get("frame_ids", [])
    )
    return frames, priority_ids


def _subtitles_for_windows(
    track: SubtitleTrack | None,
    windows: Sequence[TimeWindow],
    *,
    max_chars: int,
) -> str:
    if track is None or not windows or max_chars <= 0:
        return ""
    return track.text_for_windows(list(windows), max_chars=max_chars)


def build_evidence_packet(
    source: str,
    *,
    reference: Mapping[str, Any],
    direct_row: Mapping[str, Any],
    active_row: Mapping[str, Any],
    cached: CachedVideo,
    subtitle_track: SubtitleTrack | None,
    config: AblationConfig,
) -> EvidencePacket:
    if source == "direct_replay":
        candidates = _trace_frames_direct(direct_row)
        selected, priority_count = select_priority_frames(
            candidates,
            priority_ids=(),
            limit=config.max_frames,
        )
        windows = _windows_from_exposures(direct_row.get("metadata", {}).get("exposures", []))
        subtitles = _subtitles_for_windows(
            subtitle_track,
            windows,
            max_chars=config.subtitle_max_chars,
        )
        return EvidencePacket(
            source=source,
            frames=tuple(selected),
            subtitles=subtitles,
            source_frame_count=len(candidates),
            subtitle_window_count=len(windows),
            priority_frame_count=priority_count,
        )
    if source == "active_tree_replay":
        candidates, priority_ids = _trace_frames_active(active_row)
        selected, priority_count = select_priority_frames(
            candidates,
            priority_ids=priority_ids,
            limit=config.max_frames,
        )
        windows = _windows_from_exposures(active_row.get("metadata", {}).get("exposures", []))
        subtitles = _subtitles_for_windows(
            subtitle_track,
            windows,
            max_chars=config.subtitle_max_chars,
        )
        return EvidencePacket(
            source=source,
            frames=tuple(selected),
            subtitles=subtitles,
            source_frame_count=len(candidates),
            subtitle_window_count=len(windows),
            priority_frame_count=priority_count,
        )
    if source == "oracle_context":
        set_id, all_windows = oracle_context_windows(reference)
        _, visual_windows = oracle_context_windows(
            reference,
            modalities={"visual", "ocr"},
        )
        _, subtitle_windows = oracle_context_windows(
            reference,
            modalities={"subtitle"},
        )
        frame_windows = visual_windows or all_windows
        selected = sample_cached_windows(
            cached,
            frame_windows,
            total=config.max_frames,
        )
        subtitles = _subtitles_for_windows(
            subtitle_track,
            subtitle_windows,
            max_chars=config.subtitle_max_chars,
        )
        candidates = {
            frame.id
            for frame in cached.frames
            if any(window.contains(frame.timestamp_seconds) for window in frame_windows)
        }
        return EvidencePacket(
            source=source,
            frames=tuple(selected),
            subtitles=subtitles,
            source_frame_count=len(candidates),
            subtitle_window_count=len(subtitle_windows),
            oracle_set_id=set_id,
            oracle_windows=tuple(frame_windows),
            oracle_subtitle_windows=tuple(subtitle_windows),
        )
    raise ValueError(f"Unsupported packet source: {source}")


def build_unified_answer_prompt(
    question: VideoMMEQuestion,
    packet: EvidencePacket,
) -> str:
    adapter = MultipleChoiceAdapter(question.options)
    frame_map = "\n".join(
        f"- Frame {index}: {frame.timestamp_seconds:.3f}s"
        for index, frame in enumerate(packet.frames, start=1)
    )
    subtitles = packet.subtitles or "<no aligned subtitles supplied>"
    return f"""Answer the multiple-choice question from only the supplied chronological frames and aligned subtitles.

Treat this as a fresh one-pass decision. Do not assume any unseen event. Compare every option with
the concrete visual, OCR, subtitle, and temporal evidence before choosing. For NOT/EXCEPT questions,
choose the unsupported option. Return exactly one capital letter and no explanation.

Question:
{question.question.strip()}

Options:
{adapter.options_text()}

Frame timestamps:
{frame_map}

Aligned subtitles:
{subtitles}

Allowed output: {', '.join(adapter.letters)}
"""


def load_dev_trace_rows(
    path: str | Path,
    *,
    dev_question_ids: Sequence[str],
) -> dict[tuple[str, str], dict[str, Any]]:
    source = Path(path).expanduser().resolve()
    allowed = set(dev_question_ids)
    rows = [
        json.loads(line)
        for line in source.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    unexpected = sorted(
        {
            str(row.get("question_id"))
            for row in rows
            if str(row.get("question_id")) not in allowed
        }
    )
    if unexpected:
        raise ValueError("source trace contains non-dev question IDs")
    result: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        strategy = str(row.get("strategy"))
        if strategy not in {"direct", "active_tree"}:
            continue
        key = (str(row["question_id"]), strategy)
        if key in result:
            raise ValueError(f"duplicate source trace record: {key}")
        result[key] = row
    missing = [
        (question_id, strategy)
        for question_id in dev_question_ids
        for strategy in ("direct", "active_tree")
        if (question_id, strategy) not in result
    ]
    if missing:
        raise ValueError(f"source trace is missing {len(missing)} direct/active records")
    return result


class UnifiedAnswerSession:
    def __init__(
        self,
        config: Mapping[str, Any],
        *,
        model: BaseVideoModel | None = None,
        model_factory: Callable[[Mapping[str, Any] | None], BaseVideoModel] = build_model,
    ) -> None:
        self.raw_config = dict(config)
        self.config = AblationConfig.from_mapping(self.raw_config.get("ablation"))
        self.model = model or model_factory(self.raw_config.get("model"))
        self._loaded = False

    def load(self) -> None:
        if not self._loaded:
            self.model.load()
            self._loaded = True

    def close(self) -> None:
        if self._loaded:
            self.model.unload()
            self._loaded = False

    def __enter__(self) -> Self:
        self.load()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def evaluate(
        self,
        question: VideoMMEQuestion,
        packet: EvidencePacket,
        *,
        reference: Mapping[str, Any],
    ) -> dict[str, Any]:
        if not self._loaded:
            raise RuntimeError("UnifiedAnswerSession is not loaded")
        prompt = build_unified_answer_prompt(question, packet)
        video_options = {
            "min_pixels": self.config.min_pixels,
            "max_pixels": self.config.max_pixels,
            "total_pixels": self.config.total_pixels,
            "fps": self.config.video_fps,
            "max_frames": self.config.max_frames,
        }
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": [frame.path for frame in packet.frames],
                        **video_options,
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        started = time.perf_counter()
        try:
            output = self.model.generate(
                messages,
                max_new_tokens=self.config.max_new_tokens,
            )
            prediction = MultipleChoiceAdapter(question.options).normalize(output.text)
            exposures = packet.exposures()
            return {
                **question.to_dict(),
                "policy_id": ABLATION_POLICY_ID,
                "packet_source": packet.source,
                "prediction": prediction,
                "correct": prediction == question.answer,
                "completed": prediction in MultipleChoiceAdapter(question.options).letters,
                "model_output": output.text,
                "prompt_sha256": canonical_sha256(prompt),
                "wall_seconds": time.perf_counter() - started,
                "packet": packet.metadata(),
                "exposures": exposures,
                "relaxed_grounding": score_relaxed_grounding(dict(reference), exposures),
                "model_metadata": output.metadata,
            }
        finally:
            gc.collect()
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:  # pragma: no cover - inference dependency
                pass


def _mean(rows: Sequence[Mapping[str, Any]], getter: Callable[[Mapping[str, Any]], float]) -> float | None:
    values = [getter(row) for row in rows]
    return statistics.mean(values) if values else None


def _strategy_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    completed = [row for row in rows if bool(row.get("completed"))]
    return {
        "items": len(rows),
        "completed": len(completed),
        "completion_rate": len(completed) / len(rows) if rows else 0.0,
        "accuracy": sum(bool(row.get("correct")) for row in rows) / len(rows) if rows else 0.0,
        "relaxed_grounded_rate": (
            sum(bool(row.get("relaxed_grounding", {}).get("grounded")) for row in rows)
            / len(rows)
            if rows
            else 0.0
        ),
        "mean_slot_coverage": _mean(
            rows,
            lambda row: float(row.get("relaxed_grounding", {}).get("slot_coverage", 0)),
        ),
        "mean_selected_frames": _mean(
            rows,
            lambda row: float(row.get("packet", {}).get("selected_frame_count", 0)),
        ),
        "mean_source_frames": _mean(
            rows,
            lambda row: float(row.get("packet", {}).get("source_frame_count", 0)),
        ),
        "mean_subtitle_chars": _mean(
            rows,
            lambda row: float(row.get("packet", {}).get("subtitle_chars", 0)),
        ),
        "mean_input_tokens": _mean(
            rows,
            lambda row: float(row.get("model_metadata", {}).get("input_tokens", 0)),
        ),
        "mean_output_tokens": _mean(
            rows,
            lambda row: float(row.get("model_metadata", {}).get("output_tokens", 0)),
        ),
        "mean_wall_seconds": _mean(rows, lambda row: float(row.get("wall_seconds", 0))),
        "correct_grounding_cells": dict(
            Counter(
                ("correct" if row.get("correct") else "wrong")
                + "_"
                + (
                    "grounded"
                    if row.get("relaxed_grounding", {}).get("grounded")
                    else "ungrounded"
                )
                for row in rows
            )
        ),
    }


def _paired(
    left: Mapping[str, Mapping[str, Any]],
    right: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    question_ids = sorted(set(left) & set(right))
    wins = losses = ties = same_prediction = 0
    for question_id in question_ids:
        left_row = left[question_id]
        right_row = right[question_id]
        left_correct = bool(left_row.get("correct"))
        right_correct = bool(right_row.get("correct"))
        if left_correct and not right_correct:
            wins += 1
        elif right_correct and not left_correct:
            losses += 1
        else:
            ties += 1
        same_prediction += int(left_row.get("prediction") == right_row.get("prediction"))
    return {
        "comparable": len(question_ids),
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "same_prediction": same_prediction,
        "different_prediction": len(question_ids) - same_prediction,
    }


def summarize_ablation(
    rows: Sequence[Mapping[str, Any]],
    *,
    source_rows: Mapping[tuple[str, str], Mapping[str, Any]],
    expected_items: int,
    run_signature: str,
) -> dict[str, Any]:
    packet_rows = {
        source: [row for row in rows if row.get("packet_source") == source]
        for source in PACKET_SOURCES
    }
    by_source = {
        source: {str(row["question_id"]): row for row in values}
        for source, values in packet_rows.items()
    }
    original_direct = {
        question_id: row
        for (question_id, strategy), row in source_rows.items()
        if strategy == "direct"
    }
    original_active = {
        question_id: row
        for (question_id, strategy), row in source_rows.items()
        if strategy == "active_tree"
    }
    original_metrics = {
        "direct": {
            "items": len(original_direct),
            "accuracy": sum(bool(row.get("correct")) for row in original_direct.values())
            / len(original_direct),
        },
        "active_tree": {
            "items": len(original_active),
            "accuracy": sum(bool(row.get("correct")) for row in original_active.values())
            / len(original_active),
        },
    }
    return {
        "policy_id": ABLATION_POLICY_ID,
        "split": "dev",
        "expected_items": expected_items,
        "observed_items": len(rows),
        "engineering_pass": (
            len(rows) == expected_items
            and all(bool(row.get("completed")) for row in rows)
            and all(not row.get("error") for row in rows)
        ),
        "run_signature": run_signature,
        "packet_metrics": {
            source: _strategy_metrics(values) for source, values in packet_rows.items()
        },
        "unified_head_paired": {
            "active_vs_direct": _paired(
                by_source["active_tree_replay"],
                by_source["direct_replay"],
            ),
            "oracle_vs_direct": _paired(
                by_source["oracle_context"],
                by_source["direct_replay"],
            ),
            "oracle_vs_active": _paired(
                by_source["oracle_context"],
                by_source["active_tree_replay"],
            ),
        },
        "original_dev_metrics": original_metrics,
        "answer_head_effect": {
            "direct_replay_vs_original_direct": _paired(
                by_source["direct_replay"],
                original_direct,
            ),
            "active_replay_vs_original_active": _paired(
                by_source["active_tree_replay"],
                original_active,
            ),
        },
    }


def preflight_ablation(
    *,
    dataset: Evidence30Dataset,
    questions: Sequence[VideoMMEQuestion],
    source_rows: Mapping[tuple[str, str], Mapping[str, Any]],
    config: Mapping[str, Any],
    video_dir: str | Path,
    subtitle_dir: str | Path,
    source_items_path: str | Path,
) -> dict[str, Any]:
    ablation = AblationConfig.from_mapping(config.get("ablation"))
    cache = VideoEvidenceCache(
        ablation.cache_dir,
        sample_fps=ablation.sample_fps,
        max_side=ablation.cache_max_side,
        jpeg_quality=ablation.cache_jpeg_quality,
        lru_size=ablation.cache_lru_size,
    )
    errors: list[str] = []
    packets = Counter()
    frame_counts: dict[str, list[int]] = {source: [] for source in PACKET_SOURCES}
    question_ids = dataset.question_ids("dev")
    if [question.question_id for question in questions] != question_ids:
        errors.append("question order does not match Evidence30 dev manifest")
    resolved_subtitle_dir = Path(subtitle_dir).expanduser().resolve()
    for question in questions:
        try:
            cached = cache.prepare(question.video_path(video_dir))
            subtitle_path = question.subtitle_path(resolved_subtitle_dir)
            track = SubtitleTrack.from_srt(subtitle_path) if subtitle_path.is_file() else None
            direct_row = source_rows[(question.question_id, "direct")]
            active_row = source_rows[(question.question_id, "active_tree")]
            reference = dataset.reference(question.question_id)
            for source in PACKET_SOURCES:
                packet = build_evidence_packet(
                    source,
                    reference=reference,
                    direct_row=direct_row,
                    active_row=active_row,
                    cached=cached,
                    subtitle_track=track,
                    config=ablation,
                )
                prompt = build_unified_answer_prompt(question, packet)
                if any(
                    forbidden in prompt
                    for forbidden in (
                        "atomic_fact",
                        "context_interval",
                        "core_interval",
                        "official_answer",
                        "hard_negative",
                    )
                ):
                    raise ValueError("reference schema key leaked into unified prompt")
                missing_frames = [frame.path for frame in packet.frames if not Path(frame.path).is_file()]
                if missing_frames:
                    raise FileNotFoundError(f"packet has {len(missing_frames)} missing frames")
                if len(packet.frames) > ablation.max_frames:
                    raise ValueError("packet exceeds matched frame budget")
                if source == "oracle_context":
                    oracle_score = score_relaxed_grounding(
                        reference,
                        packet.exposures(),
                    )
                    if not oracle_score["grounded"]:
                        raise ValueError("oracle packet does not cover every required slot")
                packets[source] += 1
                frame_counts[source].append(len(packet.frames))
        except Exception as exc:  # noqa: BLE001 - aggregate every preflight failure
            errors.append(f"{question.question_id}: {type(exc).__name__}: {exc}")
    return {
        "policy_id": ABLATION_POLICY_ID,
        "ok": not errors,
        "split": "dev",
        "questions": len(questions),
        "expected_packets": len(questions) * len(PACKET_SOURCES),
        "packets": dict(packets),
        "matched_frame_limit": ablation.max_frames,
        "frame_count_ranges": {
            source: {
                "min": min(values) if values else None,
                "max": max(values) if values else None,
            }
            for source, values in frame_counts.items()
        },
        "source_items_sha256": sha256_file(source_items_path),
        "errors": errors,
    }


def run_ablation_suite(
    *,
    dataset: Evidence30Dataset,
    questions: Sequence[VideoMMEQuestion],
    source_rows: Mapping[tuple[str, str], Mapping[str, Any]],
    config: Mapping[str, Any],
    video_dir: str | Path,
    subtitle_dir: str | Path,
    source_items_path: str | Path,
    output_dir: str | Path,
    packet_sources: Sequence[str] = PACKET_SOURCES,
    session_factory: Callable[..., UnifiedAnswerSession] = UnifiedAnswerSession,
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    invalid = sorted(set(packet_sources) - set(PACKET_SOURCES))
    if invalid:
        raise ValueError(f"unsupported packet sources: {', '.join(invalid)}")
    ablation = AblationConfig.from_mapping(config.get("ablation"))
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    items_path = output / "items.jsonl"
    manifest_path = output / "run_manifest.json"
    run_signature = canonical_sha256(
        {
            "policy_id": ABLATION_POLICY_ID,
            "config": config,
            "dev_question_ids": dataset.question_ids("dev"),
            "source_items_sha256": sha256_file(source_items_path),
            "packet_sources": list(packet_sources),
        }
    )
    run_manifest = {
        "policy_id": ABLATION_POLICY_ID,
        "split": "dev",
        "run_signature": run_signature,
        "source_items_path": str(Path(source_items_path).expanduser().resolve()),
        "source_items_sha256": sha256_file(source_items_path),
        "packet_sources": list(packet_sources),
        "matched_frame_limit": ablation.max_frames,
    }
    if manifest_path.is_file():
        existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing_manifest != run_manifest:
            raise ValueError("existing ablation output has a different run signature")
    else:
        manifest_path.write_text(
            json.dumps(run_manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    rows: list[dict[str, Any]] = []
    if items_path.is_file():
        rows = [
            json.loads(line)
            for line in items_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    completed_keys = {
        (str(row.get("question_id")), str(row.get("packet_source"))) for row in rows
    }
    expected_keys = {
        (question.question_id, source) for source in packet_sources for question in questions
    }
    unexpected = completed_keys - expected_keys
    if unexpected:
        raise ValueError("ablation checkpoint contains unexpected records")

    cache = VideoEvidenceCache(
        ablation.cache_dir,
        sample_fps=ablation.sample_fps,
        max_side=ablation.cache_max_side,
        jpeg_quality=ablation.cache_jpeg_quality,
        lru_size=ablation.cache_lru_size,
    )
    resolved_subtitle_dir = Path(subtitle_dir).expanduser().resolve()
    with session_factory(config) as session:
        for source in packet_sources:
            for question in questions:
                key = (question.question_id, source)
                if key in completed_keys:
                    continue
                reference = dataset.reference(question.question_id)
                direct_row = source_rows[(question.question_id, "direct")]
                active_row = source_rows[(question.question_id, "active_tree")]
                started = time.perf_counter()
                try:
                    cached = cache.prepare(question.video_path(video_dir))
                    subtitle_path = question.subtitle_path(resolved_subtitle_dir)
                    track = SubtitleTrack.from_srt(subtitle_path) if subtitle_path.is_file() else None
                    packet = build_evidence_packet(
                        source,
                        reference=reference,
                        direct_row=direct_row,
                        active_row=active_row,
                        cached=cached,
                        subtitle_track=track,
                        config=ablation,
                    )
                    row = session.evaluate(question, packet, reference=reference)
                except Exception as exc:  # noqa: BLE001 - preserve resumable diagnostics
                    row = {
                        **question.to_dict(),
                        "policy_id": ABLATION_POLICY_ID,
                        "packet_source": source,
                        "prediction": None,
                        "correct": False,
                        "completed": False,
                        "wall_seconds": time.perf_counter() - started,
                        "error": {
                            "type": type(exc).__name__,
                            "message": str(exc),
                        },
                    }
                row["run_signature"] = run_signature
                with items_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                rows.append(row)
                completed_keys.add(key)
                if progress is not None:
                    progress(
                        {
                            "split": "dev",
                            "packet_source": source,
                            "question_id": question.question_id,
                            "completed": len(rows),
                            "expected": len(expected_keys),
                            "item": row,
                        }
                    )

    summary = summarize_ablation(
        rows,
        source_rows=source_rows,
        expected_items=len(expected_keys),
        run_signature=run_signature,
    )
    summary["items_path"] = str(items_path)
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary
