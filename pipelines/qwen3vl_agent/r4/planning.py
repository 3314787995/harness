"""Source permissions, absolute history cutoffs, and coverage-preserving tiles."""

from __future__ import annotations

import math
from dataclasses import asdict
from typing import Any

from qwen3vl_agent.r3.checkpoint import file_digest
from qwen3vl_agent.r3.planning import covered
from qwen3vl_agent.r4.config import R4Config
from qwen3vl_agent.r4.types import (
    CoverageTile,
    HistoryMap,
    InventorySpec,
    R4Request,
    interval,
    timestamp,
)


def resolve_sources(request: R4Request, media: Any) -> tuple[dict[str, Any], list[str]]:
    sources, issues = {}, []
    cutoff = timestamp(request.query_time) if request.query_time is not None else None
    for source in request.media_sources():
        metadata = media.probe(source.video_path)
        duration = metadata.duration_seconds
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError("source duration must be positive")
        digest = file_digest(source.video_path)
        if source.source_id and source.source_id != digest:
            raise ValueError("source_id differs from media SHA-256")
        allowed = source.allowed_spans or ((0.0, duration),)
        if any(b > duration + 1e-6 for _, b in allowed):
            raise ValueError("source permission exceeds media duration")
        maps = source.time_map or (
            (HistoryMap(0, duration, source.recorded_at),) if source.recorded_at else ()
        )
        spans = []
        for a, b in allowed:
            b = min(b, duration)
            if request.observation_cutoff is not None:
                b = min(b, request.observation_cutoff)
            if b <= a:
                continue
            if cutoff is None:
                spans.append((a, b))
                continue
            if not covered((a, b), [(m.media_start, m.media_end) for m in maps]):
                issues.append(f"history_mapping_missing:{source.entry_id}")
            for mapping in maps:
                lo, hi = max(a, mapping.media_start), min(b, mapping.media_end)
                hi = min(hi, mapping.media_start + cutoff - timestamp(mapping.history_start))
                if hi > lo:
                    spans.append((lo, hi))
        sources[source.entry_id] = {
            **asdict(source),
            "source_id": digest,
            "duration": duration,
            "source_fps": metadata.source_fps,
            "allowed": sorted(spans),
            "time_map": [asdict(m) for m in maps],
        }
    return sources, issues


def history_time(source: dict[str, Any], local_time: float) -> float | None:
    values = {
        HistoryMap(**m).at(local_time)
        for m in source["time_map"]
        if m["media_start"] <= local_time < m["media_end"]
    }
    return next(iter(values)) if len(values) == 1 else None


def query_scope(request: R4Request, compiled: dict[str, Any]) -> dict[str, Any]:
    value = request.query_scope
    if value is None:
        return compiled
    if isinstance(value, str):
        return {"kind": "semantic", "description": value}
    if isinstance(value, dict) and "kind" in value:
        return value
    return {"kind": "interval", "interval": interval(value)}


def scope_spans(
    scope: dict[str, Any], source: dict[str, Any], bindings: dict[str, Any], scope_key: str
) -> list[tuple[float, float]]:
    if scope.get("entry_ids") and source["entry_id"] not in scope["entry_ids"]:
        return []
    if scope.get("kind", "full") == "semantic":
        return [
            tuple(s["interval"])
            for s in bindings.get(scope_key, [])
            if s["entry_id"] == source["entry_id"]
        ]
    if scope.get("kind") == "frame":
        t = float(scope["timestamp_sec"])
        if not any(a <= t < b for a, b in source["allowed"]):
            raise ValueError("query frame is outside allowed scope")
        end = min(b for a, b in source["allowed"] if a <= t < b)
        return [(t, min(end, t + 1 / (source["source_fps"] or 24)))]
    if scope.get("kind") == "interval":
        a, b = interval(scope["interval"])
        if not covered((a, b), source["allowed"]):
            raise ValueError("query scope is outside permissions/cutoff")
        return [(a, b)]
    return [tuple(v) for v in source["allowed"]]


def make_plan(
    spec: InventorySpec, sources: dict[str, Any], config: R4Config, bindings: dict[str, Any]
) -> list[CoverageTile]:
    tiles: dict[tuple[Any, ...], CoverageTile] = {}
    for target in spec.sets:
        scope = target.scope or spec.scope
        key = target.set_id if target.scope else "global"
        for source in sources.values():
            spans = scope_spans(scope, source, bindings, key)
            enabled = set(target.required_modalities) & set(source["available_modalities"])
            visual = bool(enabled & {"video", "screen_text"})
            external = sorted(enabled & {"subtitle", "asr"})
            kinds = (["video"] if visual else []) + external
            rate = (
                config.motion_fps
                if target.predicate_kind in {"moving", "enters", "exits"}
                else config.static_fps
            )
            rate = min(rate, source["source_fps"]) if source["source_fps"] else rate
            if scope.get("kind") == "frame":
                rate = source["source_fps"] or 24
            for a, b in spans:
                for kind in kinds:
                    length = (
                        config.core_frames / rate if kind == "video" else config.text_window_sec
                    )
                    if scope.get("kind") == "frame":
                        length = b - a
                    count = max(1, math.ceil((b - a) / length - 1e-9))
                    for i in range(count):
                        lo, hi = a + i * length, min(b, a + (i + 1) * length)
                        permission = next(
                            (v for v in source["allowed"] if v[0] <= lo and hi <= v[1]), None
                        )
                        if permission is None:
                            raise ValueError("bound query scope exceeds source permissions")
                        pad = config.context_frames / rate if kind == "video" and scope.get("kind") != "frame" else 0
                        context = (max(permission[0], lo - pad), min(permission[1], hi + pad))
                        token = (source["entry_id"], lo, hi, rate, kind, context)
                        if token not in tiles:
                            tiles[token] = CoverageTile(
                                f"tile_{len(tiles):06d}",
                                source["entry_id"],
                                (lo, hi),
                                context,
                                rate,
                                kind,
                            )
                        tiles[token].set_ids.append(target.set_id)
    return list(tiles.values())


def sample_times(tile: CoverageTile, *, shifted: bool = False) -> list[float]:
    a, b = tile.context
    offset = (0.5 / tile.fps) if shifted else 0.0
    result = [
        a + offset + i / tile.fps
        for i in range(max(0, math.ceil((b - a - offset) * tile.fps - 1e-9)))
        if a + offset + i / tile.fps < b
    ]
    return result or [a]


def planned_calls(n: int) -> int:
    return min(192, n + 14)


def tile_closed(tile: dict[str, Any], all_tiles: dict[str, Any]) -> bool:
    if tile["children"]:
        return all(tile_closed(all_tiles[k], all_tiles) for k in tile["children"])
    return tile["status"] == "observed" and tile["audit_done"] and not tile["issues"]


def set_coverage(set_id: str, tiles: dict[str, Any]) -> bool:
    selected = [t for t in tiles.values() if set_id in t["set_ids"]]
    return bool(selected) and all(tile_closed(t, tiles) for t in selected)


def split_tile(
    tile: CoverageTile, config: R4Config, *, spatial: bool, fps: float | None = None
) -> list[CoverageTile]:
    if tile.depth >= config.max_split_depth:
        return []
    result = []
    if spatial:
        x1, y1, x2, y2 = tile.bbox or [0, 0, 1000, 1000]
        if min(x2 - x1, y2 - y1) < 32:
            return []
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        px, py = (x2 - x1) * config.crop_overlap / 2, (y2 - y1) * config.crop_overlap / 2
        boxes = [
            [a, b, c, d]
            for a, c in ((x1, mx + px), (mx - px, x2))
            for b, d in ((y1, my + py), (my - py, y2))
        ]
        for i, box in enumerate(boxes):
            result.append(
                CoverageTile(
                    f"{tile.tile_id}.s{i}",
                    tile.entry_id,
                    tile.core,
                    tile.context,
                    tile.fps,
                    tile.kind,
                    list(tile.set_ids),
                    box,
                    tile.depth + 1,
                )
            )
    else:
        a, b = tile.core
        if b - a <= config.min_split_sec:
            return []
        mid, rate = (a + b) / 2, fps or tile.fps
        for i, core in enumerate(((a, mid), (mid, b))):
            pad = min(config.context_frames / rate, (b - a) / 4) if tile.kind == "video" else 0
            context = (max(tile.context[0], core[0] - pad), min(tile.context[1], core[1] + pad))
            result.append(
                CoverageTile(
                    f"{tile.tile_id}.t{i}",
                    tile.entry_id,
                    core,
                    context,
                    rate,
                    tile.kind,
                    list(tile.set_ids),
                    tile.bbox,
                    tile.depth + 1,
                )
            )
    return result


__all__ = ["covered", "history_time", "make_plan", "resolve_sources", "sample_times"]
