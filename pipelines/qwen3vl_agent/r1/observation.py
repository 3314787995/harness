"""Observation protocol validation and source-grounded packet updates."""

from __future__ import annotations

from dataclasses import asdict, replace
from typing import Any

from qwen3vl_agent.r1.control import ProtocolError, parse_fact, strings
from qwen3vl_agent.r1.media import MediaBatch, coverage_record
from qwen3vl_agent.r1.providers import ExternalSegment
from qwen3vl_agent.r1.runtime import CallResult
from qwen3vl_agent.r1.types import EvidencePacket, QuerySpec


def observation_parser(
    packet: EvidencePacket,
    batch: MediaBatch,
    query: QuerySpec,
    source_id: str,
    segments: tuple[ExternalSegment, ...],
    available_modalities: tuple[str, ...] = ("video", "screen_text"),
):
    frames = {f.id: f for f in batch.frames}
    text = {s.segment_id: s for s in segments}
    shown = set(frames) | set(text)

    def parse(data: dict[str, Any], limited: bool) -> dict[str, Any]:
        if "prediction" in data or "option_id" in data:
            raise ProtocolError("observers must not select answers")
        anchor = data.get("anchor_match")
        target = data.get("target_binding")
        if anchor not in {"matched", "mismatched", "unresolved"}:
            raise ProtocolError("invalid anchor_match")
        if target not in {"confirmed", "unresolved"}:
            raise ProtocolError("invalid target_binding")
        anchor_refs = strings(data.get("anchor_source_ids", []), "anchor_source_ids")
        target_refs = strings(data.get("target_source_ids", []), "target_source_ids")
        if (set(anchor_refs) | set(target_refs)) - shown:
            raise ProtocolError("binding references unseen material")
        if anchor == "matched" and not anchor_refs:
            anchor = "unresolved"
        if target == "confirmed" and not target_refs:
            target = "unresolved"
        speech = data.get("speech_binding", {})
        if not isinstance(speech, dict):
            raise ProtocolError("speech_binding must be an object")
        speech_frames = strings(speech.get("source_frame_ids", []), "speech frame IDs")
        speech_segments = strings(speech.get("source_segment_ids", []), "speech segment IDs")
        if set(speech_frames) - set(frames) or set(speech_segments) - set(text):
            raise ProtocolError("speech binding references unseen material")
        speech_valid = (
            speech.get("status") == "confirmed"
            and bool(speech_frames)
            and bool(speech_segments)
            and bool(speech.get("basis"))
            and speech.get("basis_kind") in {"onscreen_speaker_label", "explicit_identification"}
            and all(text[r].alignment_status == "aligned" for r in speech_segments)
            and not limited
        )
        raw_facts = data.get("facts")
        if not isinstance(raw_facts, list) or len(raw_facts) > 48:
            raise ProtocolError("facts must contain at most 48 atomic records")
        facts = []
        for i, item in enumerate(raw_facts):
            if not isinstance(item, dict):
                raise ProtocolError("invalid fact record")
            fact = parse_fact(
                item,
                fact_id=f"{packet.packet_id}.f{len(packet.facts) + i + 1}",
                source_id=source_id,
                view_id="pending",
                frames=frames,
                segments=text,
                query=query,
                quality_limited=limited,
            )
            if fact.source_kind == "screen_text" and "screen_text" not in available_modalities:
                raise ProtocolError("screen text is not an allowed evidence modality")
            if fact.source_kind == "reported_speech" and query.requires_reference:
                # A transcript alone cannot prove cross-scene visual identity.
                fact = replace(fact, observation_status="partial")
            if (
                fact.source_kind == "reported_speech"
                and query.requires_speaker_binding
                and (not speech_valid or set(fact.source_segment_ids) - set(speech_segments))
            ):
                fact = replace(fact, observation_status="partial")
            original_ids = tuple(
                dict.fromkeys(
                    batch.crops.get(r, {}).get("source_frame_id", r) for r in fact.source_frame_ids
                )
            )
            fact = replace(
                fact,
                original_frame_ids=original_ids,
                crop_transforms={
                    r: batch.crops[r] for r in fact.source_frame_ids if r in batch.crops
                },
            )
            facts.append(fact)
        unresolved = strings(data.get("unresolved", []), "unresolved")
        coverage_gaps = strings(data.get("coverage_gaps", []), "coverage_gaps")
        truncated = data.get("truncated", False)
        if not isinstance(truncated, bool):
            raise ProtocolError("truncated must be boolean")
        if truncated:
            unresolved = (*unresolved, "observation_truncated")
            facts = [replace(f, observation_status="partial") for f in facts]
        if (
            query.requires_speaker_binding
            and any(f.source_segment_ids for f in facts)
            and not speech_valid
        ):
            unresolved = (*unresolved, "speaker_binding_unresolved")
        crops = data.get("crop_requests", [])
        if not isinstance(crops, list) or len(crops) > 6:
            raise ProtocolError("invalid crop_requests")
        for crop in crops:
            if not isinstance(crop, dict) or crop.get("frame_id") not in frames:
                raise ProtocolError("crop references an unseen frame")
            bbox = crop.get("bbox_xyxy_1000")
            if not isinstance(bbox, list) or len(bbox) != 4:
                raise ProtocolError("crop requires normalized_1000 bbox")
            if not all(isinstance(n, (float, int)) for n in bbox):
                raise ProtocolError("crop coordinates must be numeric")
            if not (0 <= bbox[0] < bbox[2] <= 1000 and 0 <= bbox[1] < bbox[3] <= 1000):
                raise ProtocolError("crop coordinates outside the source frame")
            if crop["frame_id"] in batch.crops:
                raise ProtocolError("crop-of-crop requests must be mapped to the source frame")
        existence = data.get("existence", "unknown")
        if existence not in {"present", "absent", "unknown"}:
            raise ProtocolError("invalid existence value")
        if existence != "unknown" and not facts:
            existence = "unknown"
        reviews = data.get("fact_reviews", [])
        if not isinstance(reviews, list) or len(reviews) > 48:
            raise ProtocolError("invalid fact_reviews")
        prior = {f.fact_id: f for f in packet.facts}
        shown_originals = {batch.crops.get(r, {}).get("source_frame_id", r) for r in frames}
        checked_reviews = []
        for review in reviews:
            if not isinstance(review, dict) or review.get("fact_id") not in prior:
                raise ProtocolError("fact review requires a prior fact in this packet")
            original = prior[review["fact_id"]]
            refs = strings(review.get("source_ids", []), "review source IDs")
            judgment = review.get("judgment")
            if set(refs) - shown or judgment not in {"verified", "refuted", "unresolved"}:
                raise ProtocolError("invalid fact review sources or judgment")
            cited_originals = {
                batch.crops.get(r, {}).get("source_frame_id", r) for r in refs if r in frames
            }
            covered_sources = set(
                original.original_frame_ids
            ) <= shown_originals & cited_originals and set(original.source_segment_ids) <= set(refs)
            if limited or not refs or not covered_sources or not review.get("basis"):
                judgment = "unresolved"
            checked_reviews.append({**review, "judgment": judgment})
        review_request = data.get("review_request", {})
        if not isinstance(review_request, dict) or review_request.get("kind", "denser") not in {
            "before",
            "after",
            "denser",
            "conflict",
            "identity",
        }:
            raise ProtocolError("invalid review_request")
        return {
            "anchor_match": anchor,
            "target_binding": target,
            "anchor_source_ids": list(anchor_refs),
            "target_source_ids": list(target_refs),
            "facts": facts,
            "unresolved": list(unresolved),
            "coverage_gaps": list(coverage_gaps),
            "truncated": truncated,
            "crop_requests": crops,
            "existence": existence,
            "speech_binding": {**speech, "status": "confirmed" if speech_valid else "unresolved"},
            "fact_reviews": checked_reviews,
            "review_request": review_request,
            "absence_basis": str(data.get("absence_basis", "")),
        }

    return parse


def apply_observation(packet: EvidencePacket, batch: MediaBatch, result: CallResult) -> None:
    data = result.value
    packet.facts.extend(replace(f, view_id=result.call_id) for f in data["facts"])
    packet.anchor_match = data["anchor_match"]
    packet.target_binding = data["target_binding"]
    packet.anchor_source_ids = data["anchor_source_ids"]
    packet.target_source_ids = data["target_source_ids"]
    packet.unresolved = data["unresolved"]
    packet.crop_requests = data["crop_requests"]
    packet.review_request = data["review_request"]
    packet.fact_reviews.extend({**r, "view_id": result.call_id} for r in data["fact_reviews"])
    if data["speech_binding"]:
        packet.speech_bindings.append({**data["speech_binding"], "view_id": result.call_id})
    if data["existence"] == "present" or packet.existence != "present":
        packet.existence = data["existence"]
        packet.absence_basis = data["absence_basis"]
    prepared = result.prepared
    if prepared is None:
        raise ProtocolError("visual observation has no submitted media")
    record = coverage_record(batch, prepared, completed=True, truncated=data["truncated"])
    record.unresolved.extend(data["coverage_gaps"])
    packet.coverage.append(record)
    packet.source_views.append(
        {
            "view_id": result.call_id,
            "span": asdict(batch.span),
            "frames": [f.to_dict() for f in batch.frames],
            "crop_transforms": batch.crops,
            "media_kind": prepared.kind,
            "quality_limited": prepared.quality_limited,
        }
    )
