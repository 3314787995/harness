"""Source-frame reuse and bounded, timestamp-preserving R1 model media."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from itertools import pairwise
from pathlib import Path
from typing import Any

from qwen3vl_agent.coarse_to_fine.types import FrameRef
from qwen3vl_agent.p01.media import P01IndexBuilder, P01VideoIndex, SourceFrameStore
from qwen3vl_agent.p01.types import BoundingBox, TimeSpan
from qwen3vl_agent.r1.config import R1Config
from qwen3vl_agent.r1.control import BudgetExhausted
from qwen3vl_agent.r1.types import CoverageRecord, QuerySpec


@dataclass
class MediaBatch:
    span: TimeSpan
    frames: tuple[FrameRef, ...]
    requested_fps: float | None = None
    ordered: bool = False
    crops: dict[str, dict[str, Any]] = field(default_factory=dict)
    missing_anchor_ids: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    coverage_kind: str = "base"


@dataclass
class PreparedMedia:
    parts: list[dict[str, Any]]
    frames: tuple[FrameRef, ...]
    pixels: int
    sizes: list[tuple[int, int]]
    quality_limited: bool
    kind: str


def time_batches(
    span: TimeSpan, fps: float, limit: int, overlap: float = 2.0
) -> list[tuple[TimeSpan, list[float]]]:
    # Reserve three locator anchors plus a non-grid endpoint without deleting regular samples.
    duration = (limit - 5) / fps
    if duration <= overlap:
        raise ValueError("frame limit cannot preserve sampling density and two-second overlap")
    result = []
    start = span.start_seconds
    while start < span.end_seconds - 1e-7:
        end = min(span.end_seconds, start + duration)
        count = math.floor((end - start) * fps + 1e-7)
        times = [start + i / fps for i in range(count + 1)]
        if times[-1] < end - 1e-7:
            times.append(end)
        result.append((TimeSpan(start, end), times))
        if end >= span.end_seconds - 1e-7:
            break
        start = end - overlap
    return result


class R1Media:
    def __init__(
        self,
        config: R1Config,
        index_builder: P01IndexBuilder | None = None,
        source_store: SourceFrameStore | None = None,
    ) -> None:
        self.config = config
        self.index_builder = index_builder or P01IndexBuilder(config.media)
        self.source_store = source_store or SourceFrameStore(config.media)

    def probe(self, path: str) -> Any:
        return self.index_builder.probe(path)

    def navigation(self, path: str, allowed: TimeSpan, metadata: Any) -> P01VideoIndex:
        # prepare_interval avoids full-video indexing when the input is a permitted prefix.
        return self.index_builder.prepare_interval(path, allowed, metadata=metadata)

    def plan(
        self, span: TimeSpan, query: QuerySpec, *, refine: bool = False
    ) -> list[tuple[TimeSpan, list[float], float | None]]:
        c = self.config.media
        modes = set(query.observation_modes)
        if "ordered" in modes or "caption" in modes:
            fps = (
                (c.dynamic_refine_fps if refine else c.dynamic_fps)
                if "ordered" in modes
                else (c.caption_refine_fps if refine else c.caption_fps)
            )
            limit = (
                (c.dynamic_refine_max_frames if refine else c.dynamic_max_frames)
                if ("ordered" in modes)
                else (c.caption_refine_max_frames if refine else c.caption_max_frames)
            )
            if "ocr" in modes:
                fps = max(fps, c.ocr_search_fps)
            return [
                (s, ts, fps) for s, ts in time_batches(span, fps, limit, self.config.overlap_sec)
            ]
        if query.coverage in {"full_span", "sequence", "existence"}:
            return [
                (s, ts, c.interval_medium_fps)
                for s, ts in time_batches(
                    span,
                    c.interval_medium_fps,
                    c.interval_medium_max_frames,
                    self.config.overlap_sec,
                )
            ]
        count = c.static_max_frames if refine else c.static_frames
        if "ocr" in modes:
            # Inspect all regular OCR samples; do not retain only the model's favourite frames.
            return [
                (s, ts, c.ocr_search_fps)
                for s, ts in time_batches(
                    span,
                    c.ocr_search_fps,
                    self.config.ocr_batch_max_frames,
                    self.config.overlap_sec,
                )
            ]
        times = [span.start_seconds + span.duration_seconds * i / (count - 1) for i in range(count)]
        return [(span, times, None)]

    def extract(
        self,
        path: str,
        span: TimeSpan,
        timestamps: Sequence[float],
        allowed: TimeSpan,
        *,
        fps: float | None,
        anchors: Sequence[FrameRef] = (),
        ordered: bool = False,
    ) -> MediaBatch:
        anchor_times = [
            f.timestamp_seconds
            for f in anchors
            if span.contains_evidence(f.timestamp_seconds, tolerance=1e-6)
        ]
        times = sorted({*timestamps, *anchor_times})
        times = [t for t in times if allowed.contains_evidence(t, tolerance=1e-6)]
        frames = self.source_store.extract(path, times, purpose="r1_observation", max_side=None)
        frames = tuple(
            f
            for f in frames
            if allowed.contains_evidence(f.timestamp_seconds, tolerance=1e-6)
            and span.contains_evidence(f.timestamp_seconds, tolerance=1e-6)
        )
        missing = [
            f.id
            for f in anchors
            if f.timestamp_seconds in anchor_times
            and not any(abs(f.timestamp_seconds - g.timestamp_seconds) < 0.001 for g in frames)
        ]
        errors = [] if frames else ["decoder_returned_no_permitted_frames"]
        return MediaBatch(span, frames, fps, ordered, missing_anchor_ids=missing, errors=errors)

    def crop(self, frame: FrameRef, bbox: Sequence[float]) -> tuple[FrameRef, dict[str, Any]]:
        if len(bbox) != 4 or not all(math.isfinite(float(n)) for n in bbox):
            raise ValueError("crop requires four finite normalized_1000 coordinates")
        x1, y1, x2, y2 = (float(v) for v in bbox)
        if not (0 <= x1 < x2 <= 1000 and 0 <= y1 < y2 <= 1000):
            raise ValueError("crop bbox outside source frame")
        box = BoundingBox(frame.id, x1, y1, x2, y2)
        crop = self.source_store.crop(frame, box, padding_fraction=0.0)
        from PIL import Image

        with Image.open(frame.path) as image:
            width, height = image.size
        pixels = [
            math.floor(x1 * width / 1000),
            math.floor(y1 * height / 1000),
            math.ceil(x2 * width / 1000),
            math.ceil(y2 * height / 1000),
        ]
        return crop, {
            "source_frame_id": frame.id,
            "bbox_xyxy_1000": list(bbox),
            "bbox_xyxy_pixels": pixels,
            "source_size": [width, height],
            "crop_to_source": {
                "offset_x": pixels[0],
                "offset_y": pixels[1],
                "scale_x": 1,
                "scale_y": 1,
            },
        }

    def prepare(self, batch: MediaBatch, *, safe: bool = False) -> PreparedMedia:
        from PIL import Image

        c = self.config.media
        total = c.safe_normal_total_pixels if safe else c.normal_total_pixels
        normal = c.safe_normal_max_pixels if safe else c.normal_max_pixels
        detail = c.safe_detail_max_pixels if safe else c.detail_max_pixels
        if len(batch.frames) > c.caption_refine_max_frames:
            raise BudgetExhausted("combined media exceeds the per-call frame cap")
        caps = [detail if f.id in batch.crops else normal for f in batch.frames]
        share = min(1.0, total / max(1, sum(caps)))
        root = c.resolved_cache_dir / "prepared"
        root.mkdir(parents=True, exist_ok=True)
        rendered, sizes = [], []
        for frame, cap in zip(batch.frames, caps):
            with Image.open(frame.path) as original:
                width, height = original.size
                pixel_limit = max(1024, int(cap * share))
                factor = min(1.0, math.sqrt(pixel_limit / (width * height)))
                w = max(32, math.floor(width * factor / 32) * 32)
                h = max(32, math.floor(height * factor / 32) * 32)
                while w * h > pixel_limit:
                    if w >= h and w > 32:
                        w -= 32
                    elif h > 32:
                        h -= 32
                    else:
                        raise BudgetExhausted("media cannot fit the minimum image size")
                stat = Path(frame.path).stat()
                key = hashlib.sha256(
                    f"{frame.path}:{stat.st_mtime_ns}:{w}:{h}".encode()
                ).hexdigest()
                destination = root / f"{key}.png"
                if not destination.exists():
                    original.convert("RGB").resize((w, h), Image.Resampling.LANCZOS).save(
                        destination
                    )
            rendered.append(FrameRef(frame.id, frame.timestamp_seconds, str(destination)))
            sizes.append((w, h))
        pixels = sum(w * h for w, h in sizes)
        if pixels > total:
            raise BudgetExhausted("combined media exceeds total pixel budget")
        # Irregular source PTS / anchors remain explicitly timestamped images, never a fake FPS.
        ordinary = [i for i, f in enumerate(rendered) if f.id not in batch.crops]
        gaps = [
            rendered[b].timestamp_seconds - rendered[a].timestamp_seconds
            for a, b in pairwise(ordinary)
        ]
        # Qwen's list-of-frames loader pads odd-length clips. Use explicit images for those
        # batches so no hidden repeated exposure or invented frame time escapes accounting.
        regular = (
            batch.ordered
            and len(ordinary) >= 2
            and len(ordinary) % 2 == 0
            and gaps
            and min(gaps) > 0
            and max(gaps) - min(gaps) < 0.001
            and len({sizes[i] for i in ordinary}) == 1
        )
        parts = []
        if regular:
            area = sizes[ordinary[0]][0] * sizes[ordinary[0]][1]
            parts.append(
                {
                    "type": "video",
                    "video": [rendered[i].path for i in ordinary],
                    "fps": 1 / gaps[0],
                    "sample_fps": 1 / gaps[0],
                    "raw_fps": 1 / gaps[0],
                    "min_pixels": area,
                    "max_pixels": area,
                    "total_pixels": area * len(ordinary),
                }
            )
        for i, frame in enumerate(rendered):
            if regular and i in ordinary:
                continue
            area = sizes[i][0] * sizes[i][1]
            parts.extend(
                [
                    {"type": "text", "text": f"Frame {frame.id} at {frame.timestamp_seconds:.6f}s"},
                    {"type": "image", "image": frame.path, "min_pixels": area, "max_pixels": area},
                ]
            )
        return PreparedMedia(
            parts,
            tuple(rendered),
            pixels,
            sizes,
            safe,
            "ordered_video" if regular else "timestamped_images",
        )


def coverage_record(
    batch: MediaBatch, prepared: PreparedMedia, *, completed: bool, truncated: bool
) -> CoverageRecord:
    times = sorted({f.timestamp_seconds for f in batch.frames if f.id not in batch.crops})
    points = [batch.span.start_seconds, *times, batch.span.end_seconds]
    gap = max((b - a for a, b in pairwise(points)), default=None)
    resolution = (
        bool(times)
        and not prepared.quality_limited
        and not batch.missing_anchor_ids
        and not batch.errors
        and (
            batch.requested_fps is None
            or (gap is not None and gap <= 1.6 / batch.requested_fps + 0.001)
        )
    )
    return CoverageRecord(
        batch.span,
        tuple(f.id for f in batch.frames),
        gap,
        prepared.sizes,
        resolution,
        completed,
        truncated,
        list(batch.errors),
        [f"missing_anchor:{key}" for key in batch.missing_anchor_ids],
        batch.coverage_kind,
    )
