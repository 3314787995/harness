"""Permitted source PTS, stable frame identities and explicit model timing."""

from __future__ import annotations

import hashlib
import itertools
import math
from dataclasses import dataclass, field
from pathlib import Path

from qwen3vl_agent.coarse_to_fine.types import FrameRef
from qwen3vl_agent.p01.types import TimeSpan
from qwen3vl_agent.r1.config import R1Config
from qwen3vl_agent.r1.media import MediaBatch, PreparedMedia, R1Media
def file_digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


from typing import Any

InputContract = Any


@dataclass
class TemporalPrepared(PreparedMedia):
    video_frame_metadata: list[dict] = field(default_factory=list)


class TemporalMedia:
    def __init__(self, config, *, namespace="r2"):
        self.namespace = namespace
        self.config = config
        self.shared = R1Media(R1Config(media=config.media))
        self.catalog: dict[str, dict] = {}
        self._digests = {}

    def probe(self, path):
        return self.shared.probe(path)

    def source_digest(self, path):
        source = Path(path).resolve()
        key = (str(source), source.stat().st_size, source.stat().st_mtime_ns)
        if key not in self._digests:
            self._digests[key] = file_digest(source)
        return self._digests[key]

    def extract(self, path, span, timestamps, contract: InputContract, *, fps=None, anchors=()):
        """Only permitted decoded images are materialized, including nearest-frame selection.

        A codec may decode a preceding keyframe internally. Its pixels never enter a
        cache, locator, observer or summary if its PTS is outside the permitted span.
        """
        import av

        span = tuple(span)
        if not contract.permits_span(span):
            raise ValueError("window crosses a media access boundary")
        targets = sorted({float(t) for t in timestamps if span[0] <= t <= span[1]})
        requested_anchors = [
            self.frame(self.catalog[f.id]["source_frame_id"]) if f.id in self.catalog else f
            for f in anchors
            if span[0] <= f.timestamp_seconds <= span[1]
        ]
        targets = sorted(set(targets + [f.timestamp_seconds for f in requested_anchors]))
        if not targets:
            return MediaBatch(TimeSpan(*span), (), fps, True, errors=["empty_sampling_plan"])
        digest = self.source_digest(path)
        root = self.config.media.resolved_cache_dir / (self.namespace + "_source") / digest
        refs, index, previous = [], 0, None
        previous_time, cfr_compatible = None, True
        ordinal = None
        missing_pts = 0
        known_indices = {
            m["pts"]: m["decoded_frame_index"]
            for m in self.catalog.values()
            if m["source_sha256"] == digest and m.get("decoded_frame_index") is not None
        }
        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            time_base = stream.time_base
            origin = stream.start_time or 0
            rate = stream.base_rate or stream.average_rate
            source_fps = float(rate) if rate else None
            seek = max(origin, origin + math.floor(targets[0] / float(time_base)))
            container.seek(seek, stream=stream, backward=True, any_frame=False)

            def save(item):
                seconds, pts, decoded, frame_index = item
                key = f"{digest}:{stream.index}:{pts}:{time_base.numerator}/{time_base.denominator}"
                fid = self.namespace.upper() + "F-" + hashlib.sha256(key.encode()).hexdigest()[:24]
                target = root / (fid + ".png")
                if not target.exists():
                    root.mkdir(parents=True, exist_ok=True)
                    decoded.to_image().convert("RGB").save(target)
                representable = bool(
                    source_fps
                    and frame_index is not None
                    and abs(frame_index / source_fps - seconds) < 1e-6
                )
                self.catalog[fid] = {
                    "id": fid,
                    "path": str(target),
                    "source_frame_id": fid,
                    "timestamp_seconds": seconds,
                    "pts": pts,
                    "time_base": [time_base.numerator, time_base.denominator],
                    "origin_pts": origin,
                    "source_fps": source_fps,
                    "source_frame_index": frame_index if representable else None,
                    "decoded_frame_index": frame_index,
                    "frame_index_basis": "decoded_ordinal_from_start_or_cached_anchor"
                    if frame_index is not None
                    else "unresolved_after_seek",
                    "source_size": [decoded.width, decoded.height],
                    "view_box": [0, 0, decoded.width, decoded.height],
                    "source_sha256": digest,
                    "source_total_frames": stream.frames or None,
                }
                return FrameRef(fid, seconds, str(target))

            for decoded in container.decode(stream):
                if ordinal is not None:
                    ordinal += 1
                if decoded.pts is None:
                    missing_pts += 1
                    continue
                if decoded.pts == origin:
                    ordinal = 0
                elif decoded.pts in known_indices:
                    ordinal = known_indices[decoded.pts]
                seconds = float((decoded.pts - origin) * time_base)
                if seconds < span[0]:
                    continue
                if seconds > span[1]:
                    break
                if not contract.permits(seconds):
                    continue
                if (
                    previous_time is not None
                    and source_fps
                    and abs(seconds - previous_time - 1 / source_fps) > 1e-6
                ):
                    cfr_compatible = False
                previous_time = seconds
                current = (seconds, decoded.pts, decoded, ordinal)
                while index < len(targets) and targets[index] <= seconds:
                    chosen = current
                    if previous and targets[index] - previous[0] <= seconds - targets[index]:
                        chosen = previous
                    refs.append(save(chosen))
                    index += 1
                previous = current
                if index == len(targets):
                    break
            if previous:
                while index < len(targets):
                    refs.append(save(previous))
                    index += 1
        by_id = {f.id: f for f in refs}
        # Handoffs reuse the actual original frame, never a newly approximated timestamp.
        for frame in requested_anchors:
            if contract.permits(frame.timestamp_seconds) and frame.id in self.catalog:
                by_id[frame.id] = frame
        frames = tuple(sorted(by_id.values(), key=lambda f: f.timestamp_seconds))
        if not cfr_compatible:
            for frame in frames:
                self.catalog[frame.id]["source_frame_index"] = None
        missing = [f.id for f in requested_anchors if f.id not in by_id]
        return MediaBatch(
            TimeSpan(*span),
            frames,
            fps,
            True,
            missing_anchor_ids=missing,
            errors=([f"decoded_frames_without_pts:{missing_pts}"] if missing_pts else [])
            + ([] if frames else ["decoder_returned_no_permitted_frames"]),
        )

    def crop(self, frame, bbox):
        cropped, transform = self.shared.crop(frame, bbox)
        original = self.catalog[frame.id]
        self.catalog[cropped.id] = {
            **original,
            "id": cropped.id,
            "path": cropped.path,
            "source_frame_id": original["source_frame_id"],
            "view_box": transform["bbox_xyxy_pixels"],
            "crop_transform": transform,
        }
        return cropped, transform

    def fixed_crop(self, batch, bbox):
        if any(self.catalog[f.id]["source_frame_id"] != f.id for f in batch.frames):
            raise ValueError("crop must be defined in original full-frame coordinates")
        frames, crops = [], {}
        for frame in batch.frames:
            cropped, transform = self.crop(frame, bbox)
            frames.append(cropped)
            crops[cropped.id] = transform
        return MediaBatch(
            batch.span,
            tuple(frames),
            batch.requested_fps,
            True,
            crops,
            batch.missing_anchor_ids,
            batch.errors,
            batch.coverage_kind,
        )

    def prepare(self, batch, *, safe=False):
        if len(batch.frames) > self.config.max_frames_per_call:
            raise ValueError("temporal per-call frame cap exceeded")
        # Reuse pixel allocation and rendering, then build R2's own temporal input.
        plain = MediaBatch(
            batch.span,
            batch.frames,
            batch.requested_fps,
            False,
            batch.crops,
            batch.missing_anchor_ids,
            batch.errors,
            batch.coverage_kind,
        )
        prepared = self.shared.prepare(plain, safe=safe)
        frames = prepared.frames
        metadata = [self.catalog[f.id] for f in frames]
        indices = [m["source_frame_index"] for m in metadata]
        fps = metadata[0]["source_fps"] if metadata else None
        native = (
            batch.ordered
            and len(frames) >= 2
            and len(frames) % 2 == 0
            and all(i is not None for i in indices)
            and len(set(prepared.sizes)) == 1
            and len({m["source_sha256"] for m in metadata}) == 1
            and all(m["source_fps"] == fps for m in metadata)
            and all(a < b for a, b in itertools.pairwise(indices))
        )
        selected = []
        if native:
            area = math.prod(prepared.sizes[0])
            catalog_text = "Ordered source frames (absolute source seconds): " + "; ".join(
                f"F{i + 1:02d}={f.timestamp_seconds:.6f}s" for i, f in enumerate(frames)
            )
            prepared.parts = [
                {"type": "text", "text": catalog_text},
                {
                    "type": "video",
                    "video": [f.path for f in frames],
                    "sample_fps": fps,
                    "raw_fps": fps,
                    "min_pixels": area,
                    "max_pixels": area,
                    "total_pixels": area * len(frames),
                },
            ]
            selected = [
                {
                    "fps": fps,
                    "frames_indices": indices,
                    "total_num_frames": max(
                        max(indices) + 1, metadata[0].get("source_total_frames") or 0
                    ),
                    "source_timestamps": [f.timestamp_seconds for f in frames],
                    "frame_ids": [f.id for f in frames],
                }
            ]
        else:
            # Explicit IDs are short and scoped to this call, also for odd-length clips.
            for i, part in enumerate(prepared.parts):
                if part.get("type") == "text" and part.get("text", "").startswith("Frame "):
                    frame = frames[i // 2]
                    part["text"] = f"Frame F{i // 2 + 1:02d} at {frame.timestamp_seconds:.6f}s"
        return TemporalPrepared(
            prepared.parts,
            frames,
            prepared.pixels,
            prepared.sizes,
            prepared.quality_limited,
            "ordered_video" if native else "timestamped_images",
            selected,
        )

    def frame(self, fid):
        m = self.catalog[fid]
        return FrameRef(fid, m["timestamp_seconds"], m["path"])

    def coverage(self, batch, window_id, completed):
        times = sorted({f.timestamp_seconds for f in batch.frames})
        limits = [batch.span.start_seconds, *times, batch.span.end_seconds]
        gap = max((b - a for a, b in itertools.pairwise(limits)), default=None)
        return {
            "window_id": window_id,
            "span": [batch.span.start_seconds, batch.span.end_seconds],
            "frame_ids": [f.id for f in batch.frames],
            "max_gap_sec": gap,
            "requested_fps": batch.requested_fps,
            "completed": completed,
            "resolution_met": bool(times)
            and not batch.errors
            and not batch.missing_anchor_ids
            and (not batch.requested_fps or gap <= 1.6 / batch.requested_fps + 0.001),
            "missing_anchor_ids": batch.missing_anchor_ids,
            "errors": batch.errors,
            "finite_sampling_only": True,
        }


def source_point(point, meta):
    x1, y1, x2, y2 = meta["view_box"]
    return [x1 + point[0] / 1000 * (x2 - x1), y1 + point[1] / 1000 * (y2 - y1)]
