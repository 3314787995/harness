"""Parse local proposals using only the source catalog displayed in this call."""

from __future__ import annotations

import math
from dataclasses import asdict
from typing import Any

from qwen3vl_agent.r1.media import MediaBatch
from qwen3vl_agent.r3.providers import ReadEvidence
from qwen3vl_agent.r3.types import Bracket, EventQuery, EventRecord, ProtocolError


def refs(value: Any, catalog: dict[str, Any], *, required: bool = False) -> list[str]:
    if not isinstance(value, list) or any(not isinstance(x, str) for x in value):
        raise ProtocolError("source references must be a list of IDs")
    if set(value) - catalog.keys() or (required and not value):
        raise ProtocolError("fact cites absent or unshown source evidence")
    return list(dict.fromkeys(value))


def strings(value: Any) -> list[str]:
    if not isinstance(value, list) or any(not isinstance(x, str) for x in value):
        raise ProtocolError("expected a string array")
    return value


def source_catalog(
    batch: MediaBatch, external: ReadEvidence, source_id: str
) -> dict[str, dict[str, Any]]:
    result = {
        f.id: {
            "source_id": source_id,
            "kind": "frame",
            "frame_id": f.id,
            "timestamp_sec": f.timestamp_seconds,
            "start_sec": f.timestamp_seconds,
            "end_sec": f.timestamp_seconds,
            "path": f.path,
            "crop_transform": batch.crops.get(f.id),
        }
        for f in batch.frames
    }
    for item in external.items:
        if item.segment_id in result:
            raise ProtocolError("frame and external segment IDs collide")
        result[item.segment_id] = {
            **asdict(item),
            "alignment_error_sec": external.alignment_error_sec,
        }
    return result


def _frame_boundary(
    before: list[str], active: list[str], catalog: dict[str, Any], *, onset: bool
) -> Bracket:
    values_before = [catalog[r]["timestamp_sec"] for r in before]
    values_active = [catalog[r]["timestamp_sec"] for r in active]
    if onset:
        lo = max(values_before) if values_before else None
        hi = min(values_active) if values_active else None
    else:
        lo = max(values_active) if values_active else None
        hi = min(values_before) if values_before else None
    try:
        return Bracket(lo, hi)
    except ValueError as exc:
        raise ProtocolError("phase evidence violates temporal order") from exc


def parse_batch(
    data: dict[str, Any], query: EventQuery, catalog: dict[str, Any], *, limit: int = 32
) -> dict[str, Any]:
    if {"prediction", "answer", "count", "total_count"} & data.keys():
        raise ProtocolError("observers must not answer or return a total count")
    if data.get("observation_status") != "valid":
        raise ProtocolError("observation is not valid")
    if not isinstance(data.get("truncated"), bool):
        raise ProtocolError("truncated must be explicit boolean")
    rows = data.get("events")
    if not isinstance(rows, list) or len(rows) > limit:
        raise ProtocolError("event output overflows the bounded observer schema")
    targets = {t.target_id: t for t in query.targets}
    local_ids, events = set(), []
    phase_keys = (
        "before_start_refs",
        "start_refs",
        "last_active_refs",
        "completion_refs",
        "after_end_refs",
        "reset_refs",
    )
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("local_id"), str):
            raise ProtocolError("event requires local_id")
        if row["local_id"] in local_ids:
            raise ProtocolError("duplicate local_id")
        local_ids.add(row["local_id"])
        target = targets.get(row.get("target_id"))
        if target is None:
            raise ProtocolError("unknown target_id")
        if row.get("fact_kind") != target.fact_kind:
            raise ProtocolError("event fact kind does not match the query (mention is not action)")
        evidence = refs(row.get("evidence_refs", []), catalog, required=True)
        phases = {k: refs(row.get(k, []), catalog) for k in phase_keys}
        all_refs = list(dict.fromkeys(evidence + [r for values in phases.values() for r in values]))
        if target.fact_kind in {"visual_event", "screen_text_event"}:
            if not any(catalog[r]["kind"] == "frame" for r in evidence):
                raise ProtocolError("visual event requires visual evidence")
            if any(catalog[r]["kind"] != "frame" for values in phases.values() for r in values):
                raise ProtocolError("visual phase cannot use a speech timestamp")
            onset = _frame_boundary(
                phases["before_start_refs"], phases["start_refs"], catalog, onset=True
            )
            end_refs = phases["completion_refs"] or phases["after_end_refs"]
            offset = _frame_boundary(end_refs, phases["last_active_refs"], catalog, onset=False)
            visible = [
                catalog[r]["timestamp_sec"] for r in evidence if catalog[r]["kind"] == "frame"
            ]
        else:
            segments = [catalog[r] for r in evidence if catalog[r]["kind"] in {"subtitle", "asr"}]
            if not segments:
                raise ProtocolError("utterance requires read external segments")
            start, end = min(s["start_sec"] for s in segments), max(s["end_sec"] for s in segments)
            if all(s.get("alignment_status") == "aligned" for s in segments):
                onset, offset = Bracket(start, start), Bracket(end, end)
            elif all(
                isinstance(s.get("alignment_error_sec"), (float, int))
                and math.isfinite(s["alignment_error_sec"])
                and s["alignment_error_sec"] >= 0
                for s in segments
            ):
                delta = max(s["alignment_error_sec"] for s in segments)
                onset, offset = (
                    Bracket(max(0, start - delta), start + delta),
                    Bracket(max(0, end - delta), end + delta),
                )
            else:
                onset, offset = Bracket(), Bracket()
            visible = [start, end]
        if onset.lo is not None and offset.hi is not None and onset.lo > offset.hi:
            raise ProtocolError("event end precedes its start")
        if row.get("match") not in {"clear", "uncertain", "rejected"}:
            raise ProtocolError("match must be clear/uncertain/rejected")
        if not isinstance(row.get("completed"), bool):
            raise ProtocolError("completed must be boolean")
        status = {"clear": "accepted", "uncertain": "proposed", "rejected": "rejected"}[
            row["match"]
        ]
        issues = strings(row.get("unresolved_reasons", []))
        if target.requires_actor_binding:
            binding_refs = refs(row.get("actor_binding_refs", []), catalog)
            all_refs.extend(binding_refs)
            kinds = {catalog[r]["kind"] for r in binding_refs}
            if "frame" not in kinds or not kinds & {"asr", "subtitle"} or not row.get("actor_ref"):
                issues = [*issues, "speaker_visual_identity_unbound"]
                if status != "rejected":
                    status = "proposed"
        if target.unit_kind in {"action_cycle", "state_transition"} and (
            not row["completed"] or not phases["completion_refs"]
        ):
            status = "boundary_pending" if status != "rejected" else status
        if target.fact_kind == "reported_event":
            issues = [*issues, "reported_event_has_no_observed_event_time"]
            status = "proposed"
        attrs = {}
        if not isinstance(row.get("attributes", {}), dict) or not isinstance(
            row.get("cooccurrence", {}), dict
        ):
            raise ProtocolError("attributes and cooccurrence must be objects")
        for name, item in row.get("attributes", {}).items():
            if not isinstance(item, dict) or "value" not in item:
                raise ProtocolError("attributes require value and evidence")
            all_refs.extend(refs(item.get("evidence_refs", []), catalog, required=True))
            attrs[name] = item["value"]
        cooccurrence = {}
        for name, item in row.get("cooccurrence", {}).items():
            if not isinstance(item, dict) or item.get("status") not in {
                "present",
                "absent",
                "unknown",
            }:
                raise ProtocolError("invalid cooccurrence status")
            co_refs = refs(
                item.get("evidence_refs", []), catalog, required=item["status"] != "unknown"
            )
            if (
                item["status"] != "unknown"
                and target.fact_kind == "visual_event"
                and not any(catalog[r]["kind"] == "frame" for r in co_refs)
            ):
                raise ProtocolError("visual cooccurrence cannot rely on speech alone")
            if item["status"] != "unknown" and not any(
                catalog[r]["start_sec"] <= max(visible) and catalog[r]["end_sec"] >= min(visible)
                for r in co_refs
            ):
                raise ProtocolError("cooccurrence evidence lies outside the observed main episode")
            all_refs.extend(co_refs)
            cooccurrence[name] = item["status"]
        replay = row.get("replay_status", "unknown")
        if replay not in {"original", "replay", "unknown"}:
            raise ProtocolError("invalid replay status")
        event = EventRecord(
            row["local_id"],
            target.target_id,
            str(row.get("actor_ref", "")),
            str(row.get("object_ref", "")),
            str(row.get("description", target.description)),
            str(row.get("category", target.target_id)),
            target.unit_kind,
            target.fact_kind,
            onset,
            offset,
            (min(visible), max(visible)),
            list(dict.fromkeys(all_refs)),
            phases["start_refs"],
            phases["completion_refs"],
            phases["reset_refs"],
            attributes=attrs,
            cooccurrence=cooccurrence,
            left_censored=onset.lo is None,
            right_censored=offset.hi is None,
            status=status,
            completed=row["completed"],
            replay_status=replay,
            unresolved_reasons=issues,
        )
        events.append(asdict(event))
    crops = data.get("crop_requests", [])
    if not isinstance(crops, list) or len(crops) > 2:
        raise ProtocolError("at most two detail crop requests")
    for crop in crops:
        if not isinstance(crop, dict):
            raise ProtocolError("crop request must be an object")
        frame_id = crop.get("frame_id")
        if (
            frame_id not in catalog
            or catalog[frame_id]["kind"] != "frame"
            or catalog[frame_id].get("crop_transform")
        ):
            raise ProtocolError("crop must reference a displayed original frame")
        box = crop.get("bbox_xyxy_1000")
        if (
            not isinstance(box, list)
            or len(box) != 4
            or not all(
                isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
                for v in box
            )
        ):
            raise ProtocolError("invalid normalized crop")
        if not (0 <= box[0] < box[2] <= 1000 and 0 <= box[1] < box[3] <= 1000):
            raise ProtocolError("crop outside frame")
    return {
        "events": events,
        "unresolved": strings(data.get("unresolved", [])),
        "truncated": data["truncated"],
        "crop_requests": crops,
    }


def parse_relation(data: dict[str, Any], catalog: dict[str, Any]) -> dict[str, Any]:
    relation = data.get("relation")
    if relation not in {"same_occurrence", "distinct_occurrences", "unresolved"}:
        raise ProtocolError("invalid event relationship")
    return {
        "relation": relation,
        "evidence_refs": refs(
            data.get("evidence_refs", []), catalog, required=relation != "unresolved"
        ),
        "reason": str(data.get("reason", "")),
    }
