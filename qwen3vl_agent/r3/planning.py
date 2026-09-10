"""Deterministic coverage plans: indexes never substitute for semantic observations."""

from __future__ import annotations

import math
from dataclasses import replace

from qwen3vl_agent.p01.types import TimeSpan
from qwen3vl_agent.r3.config import R3Config
from qwen3vl_agent.r3.types import Bracket, CoverageTile, EventQuery, R3Request


def resolve_allowed(request: R3Request, duration: float) -> TimeSpan:
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("video has no positive duration")
    allowed = request.allowed_scope or TimeSpan(0, duration)
    if allowed.end_seconds > duration + 1e-6:
        raise ValueError("allowed_scope exceeds media duration")
    end = min(allowed.end_seconds, request.observation_cutoff or duration)
    if end <= allowed.start_seconds:
        raise ValueError("observation cutoff leaves no accessible media")
    return TimeSpan(allowed.start_seconds, end)


def numeric_scope(request: R3Request, query: EventQuery, allowed: TimeSpan) -> TimeSpan:
    value = (
        request.query_scope
        if isinstance(request.query_scope, TimeSpan)
        else (TimeSpan(*query.scope.interval) if query.scope.kind == "interval" else allowed)
    )
    if value.start_seconds < allowed.start_seconds or value.end_seconds > allowed.end_seconds:
        raise ValueError("query scope exceeds allowed scope/cutoff")
    return value


def make_tiles(
    scope: TimeSpan, allowed: TimeSpan, fps: float, config: R3Config, *, prefix: str = "tile"
) -> list[CoverageTile]:
    duration = config.core_frames / fps
    padding = config.context_frames / fps
    n = max(1, math.ceil(scope.duration_seconds / duration - 1e-9))
    return [
        CoverageTile(
            f"{prefix}_{i:05d}",
            (
                scope.start_seconds + i * duration,
                min(scope.end_seconds, scope.start_seconds + (i + 1) * duration),
            ),
            (
                max(allowed.start_seconds, scope.start_seconds + i * duration - padding),
                min(
                    allowed.end_seconds, scope.end_seconds, scope.start_seconds + (i + 1) * duration
                )
                + min(
                    padding,
                    max(
                        0.0,
                        allowed.end_seconds
                        - min(scope.end_seconds, scope.start_seconds + (i + 1) * duration),
                    ),
                ),
            ),
            fps,
        )
        for i in range(n)
    ]


def sample_times(tile: CoverageTile, *, shifted: bool = False) -> list[float]:
    start, end = tile.context
    step = 1.0 / tile.fps
    offset = step / 2 if shifted else 0.0
    return [
        start + offset + i * step
        for i in range(max(0, math.ceil((end - start - offset) / step - 1e-9)))
        if start + offset + i * step < end
    ]


def base_rate(query: EventQuery, config: R3Config, source_fps: float | None) -> float:
    fps = (
        config.cycle_fps
        if any(t.unit_kind in {"action_cycle", "state_transition"} for t in query.targets)
        else config.episode_fps
    )
    return min(fps, source_fps) if source_fps and source_fps > 0 else fps


def planned_calls(n: int) -> int:
    return sum(call_pools(n).values())


def call_pools(n: int) -> dict[str, int]:
    return {"preparation": 10, "base": n, "review": max(4, math.ceil(n / 2)),
            "format": max(2, math.ceil(n / 4)), "terminal": 2}


def covered(interval: tuple[float, float], spans: list[tuple[float, float]]) -> bool:
    cursor, end = interval
    for lo, hi in sorted(spans):
        if hi <= cursor:
            continue
        if lo > cursor + 1e-6:
            return False
        cursor = max(cursor, hi)
        if cursor >= end - 1e-6:
            return True
    return cursor >= end - 1e-6


def overlaps(a: tuple[float, float], b: tuple[float, float]) -> bool:
    return a[0] < b[1] and b[0] < a[1]


def relative_scope(
    start: Bracket, end: Bracket, query: EventQuery, allowed: TimeSpan
) -> tuple[TimeSpan, Bracket, Bracket]:
    if query.scope.relative_first_sec is not None:
        delta = query.scope.relative_first_sec
        end = Bracket(
            None if start.lo is None or end.lo is None else min(end.lo, start.lo + delta),
            None if start.hi is None or end.hi is None else min(end.hi, start.hi + delta),
        )
    elif query.scope.relative_last_sec is not None:
        delta = query.scope.relative_last_sec
        start = Bracket(
            None if end.lo is None or start.lo is None else max(start.lo, end.lo - delta),
            None if end.hi is None or start.hi is None else max(start.hi, end.hi - delta),
        )
    lo = max(allowed.start_seconds, start.lo if start.lo is not None else allowed.start_seconds)
    hi = min(allowed.end_seconds, end.hi if end.hi is not None else allowed.end_seconds)
    return TimeSpan(lo, hi), start, end


def with_semantic_scope(query: EventQuery, description: str) -> EventQuery:
    return replace(
        query, scope=replace(query.scope, kind="semantic", description=description, interval=None)
    )
