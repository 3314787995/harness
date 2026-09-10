"""Per-entry reference namespaces over the existing PTS-preserving decoder."""

from __future__ import annotations

import hashlib
from dataclasses import replace
from typing import Any

from qwen3vl_agent.coarse_to_fine.types import FrameRef
from qwen3vl_agent.p01.types import TimeSpan
from qwen3vl_agent.r1.media import MediaBatch
from qwen3vl_agent.r3.media import R3Media, R3SourceFrameStore
from qwen3vl_agent.r4.planning import history_time, sample_times
from qwen3vl_agent.r4.types import CoverageTile


def ref_prefix(entry: str) -> str:
    return hashlib.sha256(entry.encode()).hexdigest()[:12]


class R4Media(R3Media):
    def prepare(self, batch, *, safe=False):
        # Explicit images retain every selected source frame; the host labels temporal order.
        return super().prepare(replace(batch, ordered=False), safe=safe)

    def representatives(self, cards, archive, limit=6):
        frames, crops, catalog = [], {}, {}
        contexts = []
        for card in cards[:limit]:
            detection = next(iter(card.get("representative_detections", card.get("detections", []))), None)
            if detection:
                meta = archive[detection["ref"]]
                original = archive.get(meta.get("parent_ref"), meta)
                base = FrameRef(original["id"], original["start_sec"], original["path"])
                crop, transform = self.crop(base, detection["bbox"])
                crop = replace(crop, id=card["candidate_id"] + "-representative")
                frames.append(crop)
                crops[crop.id] = transform
                catalog[crop.id] = {**original, "id": crop.id, "path": crop.path, "parent_ref": original["id"],
                                    "crop_transform": transform, "candidate_id": card["candidate_id"]}
                contexts.append((base, original))
            else:
                ref = next((r for r in card["evidence_refs"] if archive[r]["kind"] == "frame"), None)
                if ref:
                    meta = archive[ref]
                    if ref not in catalog:
                        frames.append(FrameRef(ref, meta["start_sec"], meta["path"]))
                    catalog[ref] = {**meta, "candidate_id": card["candidate_id"]}
                for ref in card["evidence_refs"]:
                    if archive[ref]["kind"] != "frame":
                        catalog[ref] = archive[ref]
        for frame, meta in contexts:
            if sum(not bool(catalog[f.id].get("crop_transform")) for f in frames) >= 2:
                break
            if frame.id not in catalog:
                frames.append(frame)
                catalog[frame.id] = meta
        if not frames:
            return None, catalog
        times = [f.timestamp_seconds for f in frames]
        return MediaBatch(TimeSpan(min(times), max(times) + 1e-6), tuple(frames), ordered=False, crops=crops), catalog

    def observe_tile(
        self, source: dict[str, Any], tile: CoverageTile, *, shifted: bool = False
    ) -> tuple[MediaBatch, dict[str, Any]]:
        # At native cadence a half-frame shift cannot reveal additional frames. Nearest-PTS
        # ties can instead skip alternating frames, so recheck the original full cadence.
        if source["source_fps"] and tile.fps >= source["source_fps"] - 1e-6:
            shifted = False
        if isinstance(self.source_store, R3SourceFrameStore):
            self.source_store.bind_source(source["video_path"], source["source_id"])
        allowed = next(
            s for s in source["allowed"] if s[0] <= tile.context[0] and tile.context[1] <= s[1]
        )
        batch = self.extract(
            source["video_path"],
            TimeSpan(*tile.context),
            sample_times(tile, shifted=shifted),
            TimeSpan(*allowed),
            fps=tile.fps,
            ordered=True,
        )
        # PTS at the upper access boundary is never fed to the model.
        batch.frames = tuple(f for f in batch.frames if f.timestamp_seconds < allowed[1])
        batch.frames = tuple(
            replace(f, id=f"frame-{ref_prefix(source['entry_id'])}-{f.id}") for f in batch.frames
        )
        catalog = {}
        for frame in batch.frames:
            catalog[frame.id] = {
                "id": frame.id,
                "kind": "frame",
                "entry_id": source["entry_id"],
                "source_id": source["source_id"],
                "source_frame_id": frame.id,
                "start_sec": frame.timestamp_seconds,
                "end_sec": frame.timestamp_seconds,
                "timestamp_sec": frame.timestamp_seconds,
                "path": frame.path,
                "history_start": history_time(source, frame.timestamp_seconds),
            }
        if tile.bbox is not None:
            crops, transforms = [], {}
            for frame in batch.frames:
                crop, transform = self.crop(frame, tile.bbox)
                crop = replace(
                    crop,
                    id=f"{frame.id}-crop-{hashlib.sha256(str(tile.bbox).encode()).hexdigest()[:10]}",
                )
                crops.append(crop)
                transforms[crop.id] = transform
                catalog[crop.id] = {
                    **catalog[frame.id],
                    "id": crop.id,
                    "path": crop.path,
                    "parent_ref": frame.id,
                    "crop_transform": transform,
                }
            batch.frames, batch.crops = tuple(crops), transforms
        return batch, catalog

    def comparison(
        self, records: list[dict[str, Any]], catalog: dict[str, Any]
    ) -> tuple[MediaBatch | None, dict[str, Any]]:
        frames, crops, call_catalog = [], {}, {}
        for record in records:
            detection = next(iter(record.get("detections", [])), None)
            if detection is None:
                continue
            evidence = catalog[detection["ref"]]
            original = catalog.get(evidence.get("parent_ref"), evidence)
            frame = FrameRef(original["id"], original["start_sec"], original["path"])
            crop, transform = self.crop(frame, detection["bbox"])
            crop = replace(crop, id=f"{record['observation_id']}-comparison-crop")
            for f, meta in (
                (frame, original),
                (
                    crop,
                    {
                        **original,
                        "id": crop.id,
                        "path": crop.path,
                        "parent_ref": frame.id,
                        "crop_transform": transform,
                    },
                ),
            ):
                if f.id not in call_catalog:
                    frames.append(f)
                    call_catalog[f.id] = meta
            crops[crop.id] = transform
            # Previously read observation refs are explicitly included, even if a crop
            # has now been rendered from the original source for comparison.
            for ref in record["evidence_refs"]:
                call_catalog[ref] = catalog[ref]
        if not frames:
            return None, {}
        times = [f.timestamp_seconds for f in frames]
        return MediaBatch(
            TimeSpan(min(times), max(times) + 1e-6), tuple(frames), ordered=False, crops=crops
        ), call_catalog


def public_catalog(catalog: dict[str, Any]) -> dict[str, Any]:
    return {key: {k: v for k, v in value.items() if k != "path"} for key, value in catalog.items()}
