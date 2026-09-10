"""Protocol-scoped access to real PTS media; raw pixels remain independently auditable."""

import json
from copy import deepcopy
from dataclasses import asdict, replace
from pathlib import Path

from qwen3vl_agent.coarse_to_fine.types import FrameRef
from qwen3vl_agent.p01.types import TimeSpan
from qwen3vl_agent.r1.media import MediaBatch
from qwen3vl_agent.r3.checkpoint import file_digest
from qwen3vl_agent.temporal_media import TemporalMedia

from .types import InputContract, ProtocolError, digest


class ScopedMedia:
    def __init__(self, request, config):
        self.request, self.config = request, config
        if not Path(request.video_path).is_file():
            raise FileNotFoundError("R6 requires an existing local video file; URL decoding is disabled")
        probe = TemporalMedia(config, namespace="r6")
        self.metadata = probe.probe(request.video_path)
        self.source_hash = probe.source_digest(request.video_path)
        self.contract = InputContract.resolve(
            request, self.metadata.duration_seconds, self.source_hash
        )
        namespace = {"protocol": asdict(self.contract), "preprocess": asdict(config.media)}
        if request.cross_question_cache_policy == "isolated":
            namespace["request"] = digest(
                {
                    "id": request.request_id,
                    "question": request.question,
                    "choices": [asdict(c) for c in request.choices],
                }
            )
        self.root = (config.media.resolved_cache_dir / digest(namespace)).resolve()
        self.inner = TemporalMedia(
            replace(config, media=replace(config.media, cache_dir=str(self.root))), namespace="r6"
        )
        self.catalog, self.mapping = {}, {}
        self.integrity_path = self.root / "pixel_integrity.json"
        self.integrity = (
            json.loads(self.integrity_path.read_text(encoding="utf-8"))
            if self.integrity_path.exists()
            else {}
        )

    def _register(self, frame):
        meta = self.inner.catalog[frame.id]
        sha = file_digest(frame.path)
        if frame.id in self.integrity and self.integrity[frame.id] != sha:
            raise ProtocolError("frame cache changed since source registration")
        if frame.id not in self.integrity:
            self.integrity[frame.id] = sha
            self.root.mkdir(parents=True, exist_ok=True)
            temporary = self.integrity_path.with_suffix(".tmp")
            temporary.write_text(json.dumps(self.integrity, sort_keys=True), encoding="utf-8")
            temporary.replace(self.integrity_path)
        identity = {
            "source": self.source_hash,
            "pts": meta["pts"],
            "view": meta["view_box"],
            "pixels": sha,
            "scope": self.contract.fingerprint,
        }
        fid = "R6S-" + digest(identity)[:24]
        self.mapping[fid] = frame.id
        parent = next((k for k, v in self.mapping.items() if v == meta["source_frame_id"]), fid)
        self.catalog[fid] = {
            **deepcopy(meta),
            "id": fid,
            "pixel_sha256": sha,
            "source_frame_id": parent,
            "modality": "video",
            "scope_hash": self.contract.fingerprint,
            "source_time": [frame.timestamp_seconds, frame.timestamp_seconds],
        }
        return FrameRef(fid, frame.timestamp_seconds, frame.path)

    def frame(self, fid):
        if fid not in self.catalog:
            raise ProtocolError("unknown source frame")
        source = self.catalog[fid]
        path = Path(source["path"]).resolve()
        if (
            self.root not in path.parents
            or not path.is_file()
            or file_digest(path) != source["pixel_sha256"]
        ):
            raise ProtocolError("frame cache changed or lies outside its namespace")
        if source["scope_hash"] != self.contract.fingerprint or not self.contract.permits_span(
            source["source_time"]
        ):
            raise ProtocolError("cached source violates current protocol")
        return FrameRef(fid, source["timestamp_seconds"], str(path))

    def extract(self, window, times, *, fps=None):
        if "video" not in self.contract.allowed_modalities:
            raise ProtocolError("video observation is not permitted")
        if not self.contract.permits_span(window) or any(
            not window[0] <= t <= window[1] for t in times
        ):
            raise ProtocolError("sampling crosses an access boundary")
        if len(times) > self.config.max_frames_per_call:
            raise ProtocolError("frame cap exceeded; split window first")
        batch = self.inner.extract(self.request.video_path, window, times, self.contract, fps=fps)
        frames = tuple(self._register(f) for f in batch.frames)
        if not frames:
            raise RuntimeError("no permitted decoded frames")
        return replace(batch, frames=frames)

    def crop(self, fid, bbox):
        frame = self.frame(fid)
        if self.catalog[fid]["source_frame_id"] != fid:
            raise ProtocolError("crop must reference original-frame coordinates")
        if len(bbox) != 4 or not 0 <= bbox[0] < bbox[2] <= 1 or not 0 <= bbox[1] < bbox[3] <= 1:
            raise ProtocolError("invalid normalized crop")
        cropped, _ = self.inner.crop(
            self.inner.frame(self.mapping[frame.id]), [v * 1000 for v in bbox]
        )
        return self._register(cropped)

    def batch_for_sources(self, ids):
        frames = tuple(self.frame(i) for i in ids)
        if not frames:
            return None
        times = [f.timestamp_seconds for f in frames]
        # Nonadjacent legal evidence is rendered as timestamped images, never a fictitious clip.
        return MediaBatch(TimeSpan(min(times), max(times) + 1e-9), frames, ordered=False)

    def prepare(self, batch, *, safe=False):
        for frame in batch.frames:
            self.frame(frame.id)
        inner_batch = replace(
            batch, frames=tuple(self.inner.frame(self.mapping[f.id]) for f in batch.frames)
        )
        prepared = self.inner.prepare(inner_batch, safe=safe)
        outer = {self.mapping[f.id]: f.id for f in batch.frames}
        prepared.frames = tuple(
            FrameRef(outer[f.id], f.timestamp_seconds, f.path) for f in prepared.frames
        )
        for meta in prepared.video_frame_metadata:
            meta["frame_ids"] = [outer[i] for i in meta["frame_ids"]]
        return prepared

    def evidence(self, prepared):
        return {
            f"F{i + 1:02d}": deepcopy(self.catalog[f.id]) for i, f in enumerate(prepared.frames)
        }

    def restore(self, sources):
        for fid, source in sources.items():
            if source["modality"] != "video" or fid in self.catalog:
                continue
            t = source["timestamp_seconds"]
            window = next((s for s in self.contract.allowed_intervals if s[0] <= t <= s[1]), None)
            if window is None or source["scope_hash"] != self.contract.fingerprint:
                raise ProtocolError("checkpoint contains out-of-scope media")
            frame = self.extract(window, [t]).frames[0]
            w, h = source["source_size"]
            box = source["view_box"]
            if box != [0, 0, w, h]:
                frame = self.crop(frame.id, [box[0] / w, box[1] / h, box[2] / w, box[3] / h])
            if frame.id != fid or self.catalog[fid]["pixel_sha256"] != source["pixel_sha256"]:
                raise ProtocolError("checkpoint pixels or source PTS changed")

    def coverage(self, batch, *, overview=False):
        info = self.inner.coverage(batch, "r6-window", True)
        return {**info, "overview": overview, "event_discovery_complete": False}
