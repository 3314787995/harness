"""Transactional observation parsing and reference/coordinate validation."""

from __future__ import annotations

from typing import Any

from qwen3vl_agent.r4.contracts import ContractError, issue

from qwen3vl_agent.r4.types import (
    UPDATE_KINDS,
    InventorySpec,
    ObservationRecord,
    ProtocolError,
    finite,
    timestamp,
)


def invalid(path: str, message: str, expected: Any, actual: Any) -> ContractError:
    return ContractError([issue(path, "invalid_observation", message, expected=expected, actual=actual)])


def refs(value: Any, catalog: dict[str, Any], *, required: bool = True, path: str = "evidence_refs") -> list[str]:
    if not isinstance(value, list) or any(
        not isinstance(v, str) or v not in catalog for v in value
    ):
        raise invalid(path, "invalid evidence references", "array of supplied references", value)
    if required and not value:
        raise invalid(path, "evidence references required", "nonempty reference array", value)
    return list(dict.fromkeys(value))


def bbox(value: Any) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ProtocolError("bbox requires four coordinates")
    a, b, c, d = [finite(v, minimum=0) for v in value]
    if not (a < c <= 1000 and b < d <= 1000):
        raise ProtocolError("invalid normalized bbox")
    return [a, b, c, d]


def source_bbox(box: list[float], evidence: dict[str, Any]) -> list[float]:
    transform = evidence.get("crop_transform")
    if not transform:
        return box
    x1, y1, x2, y2 = transform["bbox_xyxy_1000"]
    if transform.get("source_size") and transform.get("bbox_xyxy_pixels"):
        width, height = transform["source_size"]
        left, top, right, bottom = transform["bbox_xyxy_pixels"]
        x1, y1, x2, y2 = left * 1000 / width, top * 1000 / height, right * 1000 / width, bottom * 1000 / height
    return [
        x1 + box[0] * (x2 - x1) / 1000,
        y1 + box[1] * (y2 - y1) / 1000,
        x1 + box[2] * (x2 - x1) / 1000,
        y1 + box[3] * (y2 - y1) / 1000,
    ]


def parse_batch(
    data: dict[str, Any],
    spec: InventorySpec,
    catalog: dict[str, Any],
    tile: dict[str, Any],
    source: dict[str, Any],
    limit: int,
) -> dict[str, Any]:
    if data.get("observation_status") not in {"valid", "unreadable", "partial"}:
        raise ProtocolError("invalid observation status")
    by_set = {s.set_id: s for s in spec.sets}
    observations, local_ids = [], set()
    for index, item in enumerate(data.get("observations", [])):
        field = f"observations[{index}]"
        local_id, set_id = item["local_id"], item["set_id"]
        if not isinstance(local_id, str) or not local_id or local_id in local_ids:
            raise invalid(field + ".local_id", "invalid/duplicate local observation ID", "unique ID", local_id)
        local_ids.add(local_id)
        if set_id not in tile["set_ids"] or set_id not in by_set:
            raise invalid(field + ".set_id", "observation belongs to an unrequested set", tile["set_ids"], set_id)
        target = by_set[set_id]
        evidence = refs(item["evidence_refs"], catalog, path=field + ".evidence_refs")
        status = item.get("predicate_status", "unknown")
        if status not in {"satisfied", "refuted", "unknown"}:
            raise invalid(field + ".predicate_status", "invalid predicate status", "satisfied/refuted/unknown", status)
        relation = item.get("evidence_relation", target.evidence_relation)
        if relation != target.evidence_relation:
            raise invalid(field + ".evidence_relation", "wrong evidence relation for set", target.evidence_relation, relation)
        kinds = {catalog[r]["kind"] for r in evidence}
        if relation == "visually_present" and "frame" not in kinds:
            raise invalid(field + ".evidence_refs", "visual presence requires a frame", "frame", sorted(kinds))
        if (
            relation == "mentioned"
            and not (kinds & {"subtitle", "asr"})
            and ("screen_text" not in target.required_modalities or not item.get("raw_text"))
        ):
            raise invalid(field + ".evidence_refs", "mention requires permitted language evidence", "permitted language", sorted(kinds))
        if relation == "text_present" and "frame" not in kinds:
            raise invalid(field + ".evidence_refs", "screen text requires visual evidence", "frame", sorted(kinds))
        issues = []
        # Context helps identify a member, but does not establish query membership on its own.
        in_core = any(
            tile["core"][0] <= catalog[r]["start_sec"] < tile["core"][1]
            or (
                catalog[r]["kind"] != "frame"
                and catalog[r]["start_sec"] < tile["core"][1]
                and catalog[r]["end_sec"] > tile["core"][0]
            )
            for r in evidence
        )
        if not in_core:
            status = "unknown"
            issues.append("context_only_member")
        if any("membership_sets" in catalog[r] for r in evidence) and not any(
            set_id in catalog[r].get("membership_sets", []) for r in evidence
        ):
            status = "unknown"
            issues.append("query_membership_unproven")
        detections = []
        for detection_index, detection in enumerate(item.get("detections", [])):
            ref = detection["ref"]
            if ref not in evidence or catalog[ref]["kind"] != "frame":
                raise invalid(f"{field}.detections[{detection_index}].ref", "detection must reference an observed frame", evidence, ref)
            box = bbox(detection["bbox"])
            detections.append(
                {
                    "ref": ref,
                    "bbox": source_bbox(box, catalog[ref]),
                    "source_frame_id": catalog[ref]["source_frame_id"],
                }
            )
        if target.namespace == "physical_instance" and not detections:
            raise invalid(field + ".detections", "physical instances require source detections", "nonempty detections", [])
        predicate_evidence = {}
        if target.predicate_kind in {"moving", "exits", "enters"} and status == "satisfied":
            motion = item.get("predicate_evidence", {})
            motion_refs = refs(motion.get("refs", []), catalog, required=False, path=field + ".predicate_evidence.refs")
            witnesses = refs(motion.get("witness_refs", []), catalog, required=False, path=field + ".predicate_evidence.witness_refs")
            times = {catalog[r]["start_sec"] for r in motion_refs if catalog[r]["kind"] == "frame"}
            predicate_evidence = {**motion, "refs": motion_refs, "witness_refs": witnesses}
            tracked_times = {catalog[d["ref"]]["start_sec"] for d in detections}
            justified = (
                len(times) >= 2
                and len(tracked_times) >= 2
                and motion.get("object_motion") is True
                and motion.get("camera_motion_accounted") is True
                and any(
                    tile["core"][0] <= catalog[r]["start_sec"] < tile["core"][1]
                    for r in witnesses
                    if catalog[r]["kind"] == "frame"
                )
            )
            if target.predicate_kind in {"exits", "enters"}:
                justified = (
                    justified
                    and motion.get("boundary_crossing") is True
                    and motion.get("identity_continuity") is True
                )
            if not justified:
                status = "unknown"
                issues.append("motion_predicate_unproven")
            evidence = list(dict.fromkeys(evidence + motion_refs + witnesses))
        visibility = item.get("visibility", "clear")
        population = item.get("population_status", "included")
        if visibility not in {"clear", "partial", "unreadable", "occluded"} or population not in {
            "included",
            "excluded",
            "unknown",
        }:
            raise ProtocolError("invalid visibility/population state")
        if visibility in {"unreadable", "occluded"} or population == "unknown":
            status = "unknown"
            issues.append("unreadable_or_occluded")
        value = item.get("value", item.get("raw_text", ""))
        if not isinstance(value, str) or not value.strip():
            raise invalid(field + ".value", "member value required", "nonempty text", value)
        attributes = item.get("attributes", {})
        if not isinstance(attributes, dict) or not isinstance(
            item.get("candidate_values", []), list
        ):
            raise ProtocolError("invalid member attributes/candidate values")
        if any(not isinstance(v, str) or not v for v in item.get("candidate_values", [])):
            raise ProtocolError("candidate values must be nonempty strings")
        for name, attribute in attributes.items():
            if isinstance(attribute, dict):
                refs(attribute.get("evidence_refs", []), catalog, path=f"{field}.attributes.{name}.evidence_refs")
        observations.append(
            ObservationRecord(
                "",
                set_id,
                source["entry_id"],
                source["source_id"],
                local_id,
                value,
                str(item.get("category", value)),
                status,
                relation,
                evidence,
                detections,
                attributes,
                item.get("raw_text"),
                item.get("candidate_values", []),
                visibility,
                population,
                item.get("owner"),
                item.get("task_id"),
                issues,
                predicate_evidence=predicate_evidence,
            )
        )
    relations = []
    by_local = {o.local_id: o for o in observations}
    for index, item in enumerate(data.get("local_identity_relations", [])):
        field = f"local_identity_relations[{index}]"
        if item["left"] not in local_ids or item["right"] not in local_ids:
            raise invalid(field, "identity references unknown local observations", sorted(local_ids), [item["left"], item["right"]])
        if item["left"] == item["right"]:
            raise invalid(field, "self identity relation", "two distinct observations", item["left"])
        parsed = parse_relation(item, catalog, path=field)
        if parsed["supersedes"]:
            raise invalid(field + ".supersedes", "local identity observations cannot supersede global decisions", [], parsed["supersedes"])
        if parsed["relation"] != "unknown" and any(
            not set(parsed["evidence_refs"]).intersection(by_local[k].evidence_refs)
            for k in (item["left"], item["right"])
        ):
            raise invalid(field + ".evidence_refs", "local relation must cite both members", "evidence for both members", parsed["evidence_refs"])
        if any(
            by_set[by_local[k].set_id].namespace != "physical_instance"
            for k in (item["left"], item["right"])
        ):
            raise invalid(field, "local identity links require physical instances", "physical_instance", "nonphysical member")
        relations.append({**item, **parsed})
    updates, update_ids = [], set()
    for index, item in enumerate(data.get("task_updates", [])):
        field = f"task_updates[{index}]"
        kind = item["kind"]
        if kind not in UPDATE_KINDS or item["set_id"] not in tile["set_ids"]:
            raise ProtocolError("invalid task update")
        target = by_set[item["set_id"]]
        if target.namespace != "task_item":
            raise ProtocolError("task update outside task namespace")
        evidence = refs(item["evidence_refs"], catalog, path=field + ".evidence_refs")
        if not any(catalog[r]["kind"] in {"subtitle", "asr"} for r in evidence) and (
            "screen_text" not in target.required_modalities or not item.get("raw_text")
        ):
            raise invalid(field + ".evidence_refs", "task update requires explicit language evidence", "permitted language", evidence)
        local_id = item["local_id"]
        if not isinstance(local_id, str) or not local_id or local_id in update_ids:
            raise invalid(field + ".local_id", "duplicate local update ID", "unique ID", local_id)
        update_ids.add(local_id)
        for field in ("quantity", "replacement_quantity"):
            if item.get(field) is not None:
                finite(item[field], minimum=0)
        if item.get("effective_time") is not None:
            try:
                timestamp(item["effective_time"])
            except (ValueError, TypeError) as exc:
                raise invalid(field + ".effective_time", str(exc),
                              "timezone-qualified timestamp or finite numeric time", item["effective_time"]) from exc
        if not item.get("item_key"):
            raise ProtocolError("task update item required")
        times = [catalog[r].get("history_start") for r in evidence]
        occurred = min((t for t in times if t is not None), default=None)
        if occurred is None and len({catalog[r]["entry_id"] for r in evidence}) == 1:
            occurred = min(catalog[r]["start_sec"] for r in evidence)
        owner = item.get("owner")
        speaker_ids = {catalog[r].get("speaker_id") for r in evidence} - {None}
        bound_owner = source["actor_bindings"].get(owner, owner)
        if owner in speaker_ids and owner not in source["actor_bindings"]:
            bound_owner = source["entry_id"] + ":" + owner
        in_query = not any("membership_sets" in catalog[r] for r in evidence) or any(
            item["set_id"] in catalog[r].get("membership_sets", []) for r in evidence
        )
        updates.append(
            {
                "update_id": "",
                "local_id": local_id,
                "kind": kind,
                "set_id": item["set_id"],
                "entry_id": source["entry_id"],
                "source_id": source["source_id"],
                "owner": bound_owner,
                "task_id": item.get("task_id") or source["task_context"],
                "item_key": item["item_key"],
                "quantity": item.get("quantity"),
                "unit": item.get("unit", "item"),
                "evidence_refs": evidence,
                "observed_time": occurred,
                "has_history_time": all(t is not None for t in times),
                "effective_time": timestamp(item["effective_time"])
                if item.get("effective_time")
                else occurred,
                "explicit_effective_time": item.get("effective_time") is not None,
                "refers_to": item.get("refers_to"),
                "replacement_item": item.get("replacement_item"),
                "replacement_quantity": item.get("replacement_quantity"),
                "completion_predicate": item.get("completion_predicate"),
                "binding_supported": item.get("binding_supported") is True and in_query,
            }
        )
    regions = data.get("unresolved_regions", [])
    if not isinstance(regions, list):
        raise ProtocolError("unresolved_regions must be a list")
    return {
        "observations": observations,
        "relations": relations,
        "updates": updates,
        "unresolved": regions,
        "status": data["observation_status"],
        "truncated": data.get("truncated", False) or len(observations) > limit or len(updates) > limit,
    }


def parse_relation(data: dict[str, Any], catalog: dict[str, Any], *, path: str = "relation") -> dict[str, Any]:
    relation = str(data["relation"]).lower()
    if relation not in {"same", "different", "unknown"}:
        raise ProtocolError("invalid identity relation")
    evidence = refs(data.get("evidence_refs", []), catalog, required=relation != "unknown", path=path + ".evidence_refs")
    reason = data.get("reason", "")
    if relation != "unknown" and not reason:
        raise ProtocolError("identity evidence needs a reason")
    return {
        "relation": relation,
        "evidence_refs": evidence,
        "reason": str(reason),
        "supersedes": data.get("supersedes", []),
    }
