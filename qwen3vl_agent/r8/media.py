"""R8 permission gate over the existing real-PTS decoder and pixel allocator."""

from __future__ import annotations

import itertools
import json
import math
from copy import deepcopy
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

from qwen3vl_agent.coarse_to_fine.types import FrameRef
from qwen3vl_agent.r2.media import R2Media
from qwen3vl_agent.r3.checkpoint import file_digest

from .types import InputContract, ProtocolError, digest


def windows(contract, config, query_scope=None):
    result = []
    for a, b in contract.intersect(query_scope) if query_scope else contract.allowed_time_intervals:
        core = a
        while core < b:
            end = min(b, core + config.core_seconds)
            result.append(
                {
                    "core": (core, end),
                    "context": (
                        max(a, core - config.context_seconds),
                        min(b, end + config.context_seconds),
                    ),
                }
            )
            core = end
    return result


def uniform(a, b, n):
    return [a + (b - a) * i / (n - 1) for i in range(n)] if n > 1 else [(a + b) / 2]


def sampling(window, config, *, fps=None):
    a, b = window
    if fps:
        count = max(2, math.ceil((b - a) * fps) + 1)
        if count > config.max_frames_per_call:
            raise ProtocolError(
                "narrow/split the window before requesting this FPS; density is never lowered"
            )
        return uniform(a, b, count), fps
    return uniform(a, b, config.initial_frames), None


class ScopedMedia:
    def __init__(self, request, config, model_fingerprint):
        self.request, self.config = request, config
        self.model_fingerprint = model_fingerprint
        probe = R2Media(config)
        self.metadata = probe.probe(request.video_path)
        self.source_hash = probe.source_digest(request.video_path)
        self.contract = InputContract.resolve(request, self.metadata.duration_seconds)
        identity = digest(
            {
                "scope": asdict(self.contract),
                "preprocessing": asdict(config.media),
                "model": model_fingerprint,
            }
        )
        root = config.media.resolved_cache_dir / identity
        self.root = root.resolve()
        scoped_config = replace(config, media=replace(config.media, cache_dir=str(self.root)))
        self.inner = R2Media(scoped_config)
        self.catalog, self.mapping = {}, {}
        self.subtitles = self._subtitles()

    def _subtitles(self):
        if not self.request.subtitle_path:
            return []
        from qwen3vl_agent.r4.providers import read_external_file

        path = Path(self.request.subtitle_path)
        if path.suffix.lower() == ".jsonl":
            rows = [
                json.loads(s)
                for s in path.read_text(encoding="utf-8-sig").splitlines()
                if s.strip()
            ]
        else:
            external = SimpleNamespace(path=str(path), kind="subtitle", alignment_error_sec=None)
            rows = [asdict(s) for s in read_external_file(external, self.request.video_id)]
        output = []
        for row in rows:
            start, end = row["start_sec"], row["end_sec"]
            if self.contract.permits_span((start, end)):
                value = {"start_sec": start, "end_sec": end, "text": row["text"]}
                value["id"] = "R8T-" + digest(value)[:24]
                output.append(value)
        return output

    def read_subtitles(self, window):
        if not self.contract.permits_span(window):
            raise ProtocolError("subtitle window outside permission")
        return [
            deepcopy(s)
            for s in self.subtitles
            if window[0] <= s["start_sec"] <= s["end_sec"] <= window[1]
        ]

    def _register(self, frame):
        meta = self.inner.catalog[frame.id]
        # Model-visible IDs depend only on permitted pixels and timestamps, not forbidden suffix bytes.
        pixel_hash = file_digest(frame.path)
        fid = (
            "R8F-"
            + digest(
                {
                    "pixels": pixel_hash,
                    "time": meta["timestamp_seconds"],
                    "scope": self.contract.fingerprint,
                    "view": meta["view_box"],
                }
            )[:24]
        )
        self.mapping[fid] = frame.id
        self.catalog[fid] = {
            "id": fid,
            "path": frame.path,
            "pixel_sha256": pixel_hash,
            "timestamp_seconds": meta["timestamp_seconds"],
            "pts": meta["pts"],
            "time_base": meta["time_base"],
            "view_box": meta["view_box"],
            "source_size": meta["source_size"],
            "modality": "video",
            "source_frame_id": self._parent_id(meta, fid),
            "scope_hash": self.contract.fingerprint,
        }
        return FrameRef(fid, frame.timestamp_seconds, frame.path)

    def _parent_id(self, meta, fid):
        return next(
            (outer for outer, inner in self.mapping.items() if inner == meta["source_frame_id"]),
            fid,
        )

    def _validate(self, frame):
        known = self.catalog.get(frame.id)
        if not known or known["scope_hash"] != self.contract.fingerprint:
            raise ProtocolError("evidence not registered in this scope")
        if (
            not self.contract.permits(known["timestamp_seconds"])
            or frame.timestamp_seconds != known["timestamp_seconds"]
        ):
            raise ProtocolError("frame timestamp outside permission or forged")
        if Path(frame.path).resolve() != Path(known["path"]).resolve():
            raise ProtocolError("frame path substitution")
        path = Path(frame.path).resolve()
        if (
            self.root not in path.parents
            or not path.is_file()
            or file_digest(path) != known["pixel_sha256"]
        ):
            raise ProtocolError("untrusted or modified frame cache")

    def frame(self, fid):
        meta = self.catalog.get(fid)
        if not meta:
            raise ProtocolError("unknown evidence ID")
        frame = FrameRef(fid, meta["timestamp_seconds"], meta["path"])
        self._validate(frame)
        return frame

    def get_evidence(self, fid):
        self.frame(fid)
        return deepcopy(self.catalog[fid])

    def restore_evidence(self, evidence):
        """Rehydrate from source pixels, never trust persisted frame paths across resumes."""
        for e in evidence.values():
            if e.get("modality") != "video" or e["id"] in self.catalog:
                continue
            t = e["timestamp_seconds"]
            allowed = next(
                (s for s in self.contract.allowed_time_intervals if s[0] <= t <= s[1]), None
            )
            if allowed is None or e["scope_hash"] != self.contract.fingerprint:
                raise ProtocolError("restored frame outside scope")
            original = self.extract(allowed, [t]).frames[0]
            width, height = e["source_size"]
            box = e["view_box"]
            if box != [0, 0, width, height]:
                original = self.crop(
                    original, [box[0] / width, box[1] / height, box[2] / width, box[3] / height]
                )
            if (
                original.id != e["id"]
                or self.catalog[original.id]["pixel_sha256"] != e["pixel_sha256"]
            ):
                raise ProtocolError("restored pixels/PTS mismatch")

    def extract(self, window, times, *, fps=None):
        if not self.contract.permits_span(window) or any(
            not window[0] <= t <= window[1] for t in times
        ):
            raise ProtocolError("sampling plan exceeds permitted window")
        if len(times) > self.config.max_frames_per_call:
            raise ProtocolError("sampling plan exceeds per-call frame cap")
        batch = self.inner.extract(self.request.video_path, window, times, self.contract, fps=fps)
        frames = tuple(self._register(f) for f in batch.frames)
        if not frames:
            raise ProtocolError("decoder returned no permitted frames")
        return replace(batch, frames=frames)

    def crop(self, frame, bbox):
        self._validate(frame)
        if (
            not isinstance(bbox, (list, tuple))
            or len(bbox) != 4
            or not 0 <= bbox[0] < bbox[2] <= 1
            or not 0 <= bbox[1] < bbox[3] <= 1
        ):
            raise ProtocolError("crop requires normalized [x1,y1,x2,y2] in [0,1]")
        original = self.inner.frame(self.mapping[frame.id])
        if self.catalog[frame.id]["source_frame_id"] != frame.id:
            raise ProtocolError("crop coordinates must refer to an original full frame")
        cropped, transform = self.inner.crop(original, [v * 1000 for v in bbox])
        result = self._register(cropped)
        self.catalog[result.id]["crop_transform"] = transform
        self.catalog[result.id]["source_frame_id"] = self.catalog[frame.id]["source_frame_id"]
        return result

    def prepare(self, batch):
        if not self.contract.permits_span((batch.span.start_seconds, batch.span.end_seconds)):
            raise ProtocolError("prepare exceeds permitted window")
        for frame in batch.frames:
            self._validate(frame)
        inner_batch = replace(
            batch,
            frames=tuple(self.inner.frame(self.mapping[f.id]) for f in batch.frames),
            crops={self.mapping[fid]: transform for fid, transform in batch.crops.items()},
        )
        prepared = self.inner.prepare(inner_batch)
        outer = {self.mapping[f.id]: f for f in batch.frames}
        # Preserve the model's encoded ordering while keeping public IDs independent of forbidden bytes.
        prepared.frames = tuple(
            FrameRef(outer[f.id].id, f.timestamp_seconds, f.path) for f in prepared.frames
        )
        for metadata in prepared.video_frame_metadata:
            metadata["frame_ids"] = [outer[fid].id for fid in metadata["frame_ids"]]
            metadata["total_num_frames"] = max(metadata["frames_indices"]) + 1
        return prepared

    def evidence(self, prepared, window):
        result = {f"F{i + 1:02d}": self.get_evidence(f.id) for i, f in enumerate(prepared.frames)}
        for i, cue in enumerate(self.read_subtitles(window)):
            result[f"T{i + 1:02d}"] = {
                "id": cue["id"],
                "timestamp_seconds": cue["start_sec"],
                "end_seconds": cue["end_sec"],
                "text": cue["text"],
                "modality": "subtitle",
                "scope_hash": self.contract.fingerprint,
            }
        return result

    def coverage(self, batch, completed):
        times = sorted({f.timestamp_seconds for f in batch.frames})
        edges = [batch.span.start_seconds, *times, batch.span.end_seconds]
        return {
            "span": [batch.span.start_seconds, batch.span.end_seconds],
            "frame_ids": [f.id for f in batch.frames],
            "times": times,
            "max_gap_seconds": max((b - a for a, b in itertools.pairwise(edges)), default=None),
            "requested_fps": batch.requested_fps,
            "completed": completed,
            "finite_sampling_only": True,
        }

    def local_batch(self, frame_id, boxes=()):
        from qwen3vl_agent.p01.types import TimeSpan
        from qwen3vl_agent.r1.media import MediaBatch

        frame = self.frame(frame_id)
        if self.catalog[frame.id]["source_frame_id"] != frame.id:
            frame = self.frame(self.catalog[frame.id]["source_frame_id"])
        crops = [self.crop(frame, box) for box in boxes[: self.config.max_crops]]
        t = frame.timestamp_seconds
        permitted = next(s for s in self.contract.allowed_time_intervals if s[0] <= t <= s[1])
        return MediaBatch(TimeSpan(*permitted), (frame, *crops), None, False)


def coverage_record(batch, core, context, discovery):
    times = sorted(
        {f.timestamp_seconds for f in batch.frames if core[0] <= f.timestamp_seconds <= core[1]}
    )
    edges = [core[0], *times, core[1]]
    gap = max((b - a for a, b in itertools.pairwise(edges)), default=core[1] - core[0])
    resolution_met = bool(times) and (
        batch.requested_fps is None or gap <= 1 / batch.requested_fps + 1e-6
    )
    errors = list(batch.errors)
    return {
        "core": list(core),
        "context": list(context),
        "timestamps": times,
        "frame_ids": [f.id for f in batch.frames],
        "max_gap_seconds": gap,
        "requested_fps": batch.requested_fps,
        "scan_completed": bool(times) and not errors,
        "resolution_met": resolution_met,
        "discovery_complete": discovery["complete"] and resolution_met and not errors,
        "open_event_boundaries": discovery["open_event_boundaries"],
        "possible_replays": discovery["possible_replays"],
        "unreadable_items": discovery["unreadable_items"],
        "discovery_rationale": discovery["rationale"],
        "finite_sampling_only": True,
        "errors": errors,
    }
