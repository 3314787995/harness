"""Full-scope, nonoverlapping cores and explicit downstream cost estimates."""

from __future__ import annotations

import math
from dataclasses import asdict
from typing import Any

from qwen3vl_agent.p01.types import TimeSpan
from qwen3vl_agent.r3.checkpoint import file_digest
from qwen3vl_agent.r5.config import R5Config
from qwen3vl_agent.r5.types import R5Request


def resolve_source(request: R5Request, media: Any) -> tuple[dict[str, Any], TimeSpan]:
    meta = media.probe(request.video_path)
    duration = meta.duration_seconds
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("video has no positive duration")
    allowed = request.allowed_scope or TimeSpan(0, duration)
    if allowed.end_seconds > duration + 1e-6:
        raise ValueError("allowed_scope exceeds media duration")
    end = min(allowed.end_seconds, duration)
    if request.observation_cutoff is not None:
        end = min(end, request.observation_cutoff)
    if end <= allowed.start_seconds:
        raise ValueError("observation cutoff leaves no permitted video")
    allowed = TimeSpan(allowed.start_seconds, end)
    scope = request.query_scope or allowed
    if scope.start_seconds < allowed.start_seconds or scope.end_seconds > allowed.end_seconds:
        raise ValueError("query_scope exceeds permitted video")
    fps = getattr(meta, "fps", None)
    return {
        "source_id": file_digest(request.video_path),
        "entry_id": request.video_id,
        "video_path": request.video_path,
        "source_fps": fps,
        "duration_sec": duration,
        "allowed": [[allowed.start_seconds, allowed.end_seconds]],
        "available_modalities": list(request.available_modalities),
        "external_files": [asdict(f) for f in request.external_files],
    }, scope


def make_plan(
    scope: TimeSpan,
    config: R5Config,
    source_fps: float | None,
    *,
    fps: float | None = None,
    prefix: str = "seg",
) -> list[dict[str, Any]]:
    rate = fps or config.sample_fps
    if source_fps and math.isfinite(source_fps) and source_fps > 0:
        rate = min(rate, source_fps)
    duration, padding = config.core_frames / rate, config.context_frames / rate
    count = max(1, math.ceil(scope.duration_seconds / duration - 1e-9))
    result = []
    for i in range(count):
        start, end = (
            scope.start_seconds + i * duration,
            min(scope.end_seconds, scope.start_seconds + (i + 1) * duration),
        )
        result.append(
            {
                "segment_id": f"{prefix}_{i:05d}",
                "core": [start, end],
                "context": [
                    max(scope.start_seconds, start - padding),
                    min(scope.end_seconds, end + padding),
                ],
                "fps": rate,
                "depth": 0,
                "status": "pending",
                "children": [],
            }
        )
    return result


def times(tile: dict[str, Any]) -> list[float]:
    start, end = tile["context"]
    return [
        start + i / tile["fps"]
        for i in range(max(1, math.ceil((end - start) * tile["fps"] - 1e-9)))
        if start + i / tile["fps"] < end
    ]


def split_tile(tile: dict[str, Any], config: R5Config) -> list[dict[str, Any]]:
    a, b = tile["core"]
    if tile["depth"] >= config.max_split_depth or (b - a) / 2 < config.min_split_sec:
        return []
    mid = (a + b) / 2
    padding = min(config.context_frames / tile["fps"], (b - a) / 8)
    return [
        {
            **tile,
            "segment_id": f"{tile['segment_id']}.{i}",
            "core": [x, y],
            "context": [max(tile["context"][0], x - padding), min(tile["context"][1], y + padding)],
            "depth": tile["depth"] + 1,
            "status": "pending",
            "children": [],
        }
        for i, (x, y) in enumerate(((a, mid), (mid, b)))
    ]


def merge_calls(n: int, fan_in: int = 4) -> int:
    calls = 0
    while n > 1:
        groups, remainder = divmod(n, fan_in)
        calls += groups + (remainder > 1)
        n = groups + bool(remainder)
    return calls


def post_calls(n: int, config: R5Config) -> int:
    # Terminal allowance: Composer plus at most one format repair.
    return merge_calls(n, config.merge_fan_in) + 2


def estimate(plan: list[dict[str, Any]], config: R5Config, modalities: tuple[str, ...]) -> dict:
    n = len(plan)
    return {
        "base_segments": n,
        "base_observer_calls": n,
        "merge_calls": merge_calls(n, config.merge_fan_in),
        "composer_calls_estimated": 1,
        "terminal_repair_reserve": 1,
        "planned_model_calls": 1 + n + post_calls(n, config),
        "planned_provider_calls": n * len(set(modalities) & {"asr", "subtitle"}),
        "base_frame_exposures": sum(len(times(t)) for t in plan),
        "note": "Text packing, bounded observation recovery and format repairs can add calls; no semantic coverage yet.",
    }


def union_duration(ranges: list[list[float]]) -> float:
    total, end = 0.0, float("-inf")
    for a, b in sorted(ranges):
        total += max(0.0, b - max(a, end))
        end = max(end, b)
    return total
