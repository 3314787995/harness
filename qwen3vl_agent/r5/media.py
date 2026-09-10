"""Use the existing source decoder and preserve the actual PTS catalog."""

from __future__ import annotations

from dataclasses import replace
from itertools import pairwise
from typing import Any

from qwen3vl_agent.p01.types import TimeSpan
from qwen3vl_agent.r3.media import R3Media, R3SourceFrameStore
from qwen3vl_agent.r5.planning import times


class R5Media(R3Media):
    def observe_tile(self, source: dict, tile: dict) -> tuple[Any, dict, dict]:
        if isinstance(self.source_store, R3SourceFrameStore):
            self.source_store.bind_source(source["video_path"], source["source_id"])
        batch = self.extract(
            source["video_path"],
            TimeSpan(*tile["context"]),
            times(tile),
            TimeSpan(*source["allowed"][0]),
            fps=tile["fps"],
            ordered=True,
        )
        batch.frames = tuple(
            replace(f, id=f"frame:{source['source_id'][:12]}:{f.id}")
            for f in batch.frames
            if f.timestamp_seconds < tile["context"][1]
        )
        catalog = {
            f.id: {
                "id": f.id,
                "kind": "frame",
                "source_id": source["source_id"],
                "timestamp_sec": f.timestamp_seconds,
                "start_sec": f.timestamp_seconds,
                "end_sec": f.timestamp_seconds,
                "path": f.path,
            }
            for f in batch.frames
        }
        a, b = tile["core"]
        pts = sorted({f.timestamp_seconds for f in batch.frames if a <= f.timestamp_seconds < b})
        gap = max((y - x for x, y in pairwise([a, *pts, b])), default=b - a)
        quality = {
            "source_frame_ids": list(catalog),
            "actual_max_frame_gap_sec": gap,
            "required_resolution_met": bool(pts) and gap <= 1.6 / tile["fps"] + 1e-6,
            "errors": list(batch.errors),
        }
        return batch, catalog, quality


def public_catalog(catalog: dict) -> dict:
    return {key: {k: v for k, v in item.items() if k != "path"} for key, item in catalog.items()}
