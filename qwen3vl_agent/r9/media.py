"""R9 scope wrapper over R2's real-PTS decoding and source-frame crop mapping."""

from copy import deepcopy
from dataclasses import asdict, replace
from itertools import pairwise
from pathlib import Path

from qwen3vl_agent.coarse_to_fine.types import FrameRef
from qwen3vl_agent.p01.types import TimeSpan
from qwen3vl_agent.r1.media import MediaBatch
from qwen3vl_agent.r2.media import R2Media
from qwen3vl_agent.r3.checkpoint import file_digest

from .types import InputContract, ProtocolError, digest


class ScopedMedia:
    def __init__(self, request, config):
        self.request, self.config = request, config
        probe = R2Media(config)
        self.metadata = probe.probe(request.video_path)
        self.source_hash = probe.source_digest(request.video_path)
        self.contract = InputContract.resolve(request, self.metadata.duration_seconds)
        self.root = (
            config.media.resolved_cache_dir
            / digest({"scope": asdict(self.contract), "preprocessing": asdict(config.media)})
        ).resolve()
        self.inner = R2Media(replace(config, media=replace(config.media, cache_dir=str(self.root))))
        self.catalog, self.mapping = {}, {}

    def _register(self, frame):
        m = self.inner.catalog[frame.id]
        pixels = file_digest(frame.path)
        fid = (
            "R9F-"
            + digest(
                {
                    "pixels": pixels,
                    "time": m["timestamp_seconds"],
                    "view": m["view_box"],
                    "scope": self.contract.fingerprint,
                }
            )[:24]
        )
        self.mapping[fid] = frame.id
        parent = next(
            (outer for outer, inner in self.mapping.items() if inner == m["source_frame_id"]), fid
        )
        self.catalog[fid] = {
            "id": fid,
            "path": frame.path,
            "pixel_sha256": pixels,
            "source_frame_id": parent,
            "timestamp_seconds": m["timestamp_seconds"],
            "pts": m["pts"],
            "time_base": m["time_base"],
            "view_box": m["view_box"],
            "source_size": m["source_size"],
            "scope_hash": self.contract.fingerprint,
            "modality": "video",
        }
        return FrameRef(fid, frame.timestamp_seconds, frame.path)

    def frame(self, fid):
        if fid not in self.catalog:
            raise ProtocolError("unknown R9 source frame")
        m = self.catalog[fid]
        p = Path(m["path"]).resolve()
        if self.root not in p.parents or not p.is_file() or file_digest(p) != m["pixel_sha256"]:
            raise ProtocolError("modified/untrusted frame cache")
        if m["scope_hash"] != self.contract.fingerprint or not self.contract.permits(
            m["timestamp_seconds"]
        ):
            raise ProtocolError("frame outside permitted scope")
        return FrameRef(fid, m["timestamp_seconds"], m["path"])

    def extract(self, window, times, *, fps=None):
        if not self.contract.permits_span(window) or any(
            not window[0] <= t <= window[1] for t in times
        ):
            raise ProtocolError("sampling outside permitted scope")
        if len(times) > self.config.max_video_frames_per_call:
            raise ProtocolError("source-frame sampling cap exceeded")
        batch = self.inner.extract(self.request.video_path, window, times, self.contract, fps=fps)
        frames = tuple(self._register(f) for f in batch.frames)
        if not frames:
            raise RuntimeError("decoder returned no permitted source frames")
        return replace(batch, frames=frames)

    def crop(self, frame, box):
        self.frame(frame.id)
        if self.catalog[frame.id]["source_frame_id"] != frame.id:
            raise ProtocolError("crop requires original-frame coordinates")
        if len(box) != 4 or not 0 <= box[0] < box[2] <= 1 or not 0 <= box[1] < box[3] <= 1:
            raise ProtocolError("crop must use normalized original coordinates")
        inner, _ = self.inner.crop(
            self.inner.frame(self.mapping[frame.id]), [v * 1000 for v in box]
        )
        return self._register(inner)

    def local_batch(self, fid, boxes=()):
        original = self.catalog[fid]["source_frame_id"]
        frame = self.frame(original)
        crops = tuple(
            self.crop(frame, box) for box in boxes[: self.config.max_detail_images_per_call]
        )
        window = next(
            s
            for s in self.contract.allowed_time_intervals
            if s[0] <= frame.timestamp_seconds <= s[1]
        )
        return MediaBatch(
            TimeSpan(*window),
            (frame, *crops),
            ordered=False,
            crops={f.id: self.catalog[f.id]["view_box"] for f in crops},
        )

    def prepare(self, batch):
        for f in batch.frames:
            self.frame(f.id)
        crop_ids = [f.id for f in batch.frames if self.catalog[f.id]["source_frame_id"] != f.id]
        if (
            len(crop_ids) > self.config.max_detail_images_per_call
            or len(batch.frames) - len(crop_ids) > self.config.max_video_frames_per_call
        ):
            raise ProtocolError("R9 video/detail frame cap exceeded")
        inner_batch = replace(
            batch,
            frames=tuple(self.inner.frame(self.mapping[f.id]) for f in batch.frames),
            crops={self.mapping[i]: self.catalog[i]["view_box"] for i in crop_ids},
        )
        prepared = self.inner.prepare(inner_batch)
        outer = {self.mapping[f.id]: f.id for f in batch.frames}
        prepared.frames = tuple(
            FrameRef(outer[f.id], f.timestamp_seconds, f.path) for f in prepared.frames
        )
        for m in prepared.video_frame_metadata:
            m["frame_ids"] = [outer[i] for i in m["frame_ids"]]
            m["total_num_frames"] = max(m["frames_indices"]) + 1
        return prepared

    def evidence(self, prepared):
        return {
            f"F{i + 1:02d}": deepcopy(self.catalog[f.id]) for i, f in enumerate(prepared.frames)
        }

    def restore(self, catalog):
        for fid, e in catalog.items():
            if fid in self.catalog:
                continue
            t = e["timestamp_seconds"]
            allowed = next(
                (s for s in self.contract.allowed_time_intervals if s[0] <= t <= s[1]), None
            )
            if allowed is None or e["scope_hash"] != self.contract.fingerprint:
                raise ProtocolError("checkpoint source outside permission")
            frame = self.extract(allowed, [t]).frames[0]
            w, h = e["source_size"]
            box = e["view_box"]
            if box != [0, 0, w, h]:
                frame = self.crop(frame, [box[0] / w, box[1] / h, box[2] / w, box[3] / h])
            if frame.id != fid or self.catalog[fid]["pixel_sha256"] != e["pixel_sha256"]:
                raise ProtocolError("restored source pixels/timestamps changed")

    @staticmethod
    def coverage(batch, *, sequential=False):
        times = sorted({f.timestamp_seconds for f in batch.frames})
        a, b = batch.span.start_seconds, batch.span.end_seconds
        edges = [a, *times, b]
        return {
            "span": [a, b],
            "times": times,
            "frame_ids": [f.id for f in batch.frames],
            "max_gap_seconds": max(y - x for x, y in pairwise(edges)),
            "completed": True,
            "sequential": sequential,
            "event_discovery_complete": False,
            "finite_sampling_only": True,
        }
