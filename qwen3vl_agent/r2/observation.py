"""Program-owned measurement requirements shared by observation and reduction."""

from __future__ import annotations

import copy
from dataclasses import dataclass

from .types import ProtocolError


def identity_requirements(query, handoff, frames):
    """Concrete prior nodes to account for; frame aliases remain program-owned."""
    requirements = []
    for entity in handoff.get("entities", []):
        prior = [
            r
            for r in handoff.get("previous_observations", [])
            if r["entity_node"] == entity["node_id"]
        ]
        source_ids = {r["source_frame_id"] for r in prior}
        requirements.append(
            {
                "key": f"K{len(requirements) + 1}",
                "from_node": entity["node_id"],
                "target_id": entity["target_id"],
                "slot_ids": [
                    s["id"] for s in query["slots"] if s["target_id"] == entity["target_id"]
                ],
                "shared_frame_ids": [
                    fid for fid, meta in frames.items() if meta["source_frame_id"] in source_ids
                ],
            }
        )
    return requirements


def normalize_identity_handoff(value, requirements):
    """Translate explicit short matches; missing identity is a program gap, never a match."""
    value = copy.deepcopy(value)
    by_key = {r["key"]: r for r in requirements}
    seen = set()
    for match in value.pop("identity_matches", []):
        key = match["key"]
        if key not in by_key or key in seen:
            raise ProtocolError("Unknown or duplicate identity match key")
        seen.add(key)
        requirement = by_key[key]
        if any(
            link["from_node"] == requirement["from_node"]
            for g in value["associations"]
            for a in g["alternatives"]
            for link in a["links"]
        ):
            raise ProtocolError(
                "Do not combine short identity match with association alternatives for the same node"
            )
        if match["status"] == "same_entity":
            if not match.get("entity_id") or not match.get("evidence_frames"):
                raise ProtocolError(
                    "same_entity requires a current entity and shown evidence frames"
                )
            value["associations"].append(
                {
                    "group_id": "short_" + key,
                    "supersedes": [],
                    "unresolved_extra": False,
                    "relation_preserved": False,
                    "alternatives": [
                        {
                            "links": [
                                {
                                    "from_node": requirement["from_node"],
                                    "to_entity": match["entity_id"],
                                    "kind": "same_entity",
                                }
                            ],
                            "evidence_frames": match["evidence_frames"],
                        }
                    ],
                }
            )
        else:
            for sid in requirement["slot_ids"]:
                value["gaps"].append(
                    {
                        "kind": "identity",
                        "slot_id": sid,
                        "description": match["reason"],
                        "source": "observer_identity_match",
                    }
                )
    targets = {e["id"]: e["target_id"] for e in value["entities"]}
    for requirement in requirements:
        linked = any(
            link["from_node"] == requirement["from_node"]
            for g in value["associations"]
            for a in g["alternatives"]
            for link in a["links"]
        )
        if linked:
            continue
        for sid in {
            r["slot_id"]
            for r in value["records"]
            if targets.get(r["entity_id"]) == requirement["target_id"]
        }:
            if not any(g["kind"] == "identity" and g.get("slot_id") == sid for g in value["gaps"]):
                value["gaps"].append(
                    {
                        "kind": "identity",
                        "slot_id": sid,
                        "description": "Identity correspondence not supplied for "
                        + requirement["from_node"],
                        "source": "program_identity_contract",
                    }
                )
    return value


def validate_identity_handoff(value, requirements):
    targets = {e["id"]: e["target_id"] for e in value["entities"]}
    for required in requirements:
        rows = [r for r in value["records"] if targets[r["entity_id"]] == required["target_id"]]
        if not rows:
            continue
        linked = False
        for group in value["associations"]:
            for alternative in group["alternatives"]:
                links = [
                    link
                    for link in alternative["links"]
                    if link["from_node"] == required["from_node"]
                ]
                if any(targets[link["to_entity"]] != required["target_id"] for link in links):
                    raise ProtocolError("Identity link crosses query targets")
                if links:
                    shared = required["shared_frame_ids"]
                    if shared and not set(shared).intersection(alternative["evidence_frames"]):
                        raise ProtocolError(
                            "Identity link must cite a shown shared frame for "
                            + required["from_node"]
                        )
                    linked = True
        explained = all(
            any(g["kind"] == "identity" and g.get("slot_id") == sid for g in value["gaps"])
            for sid in {r["slot_id"] for r in rows}
        )
        if not linked and not explained:
            raise ProtocolError(
                "Missing identity handoff for "
                + required["from_node"]
                + ": return associations or a slot-specific identity gap"
            )


@dataclass(frozen=True)
class ObservationSpec:
    tasks: tuple[dict, ...]

    @classmethod
    def from_query(cls, query):
        slots = {s["id"]: s for s in query["slots"]}
        tasks = []
        for op in query["operations"]:
            kind, params = op["op"], op["parameters"]
            for sid in op["slot_ids"]:
                slot = slots[sid]
                fields, alternatives = [], []
                task = {
                    "operation_id": op["id"],
                    "operation": kind,
                    "slot_id": sid,
                    "target_id": slot["target_id"],
                    "reference_frame": slot["reference_frame"],
                    "required_fields": fields,
                    "one_of_fields": alternatives,
                }
                if kind in {"endpoint_delta", "state_sequence", "relation_transition"}:
                    fields.append("value")
                    task["non_null_value"] = True
                elif kind == "motion_condition_filter":
                    alternatives.extend([["point"], ["motion"]])
                    if params.get("motion_condition"):
                        fields.append("condition_satisfied")
                        task["motion_condition"] = params["motion_condition"]
                elif kind == "rotation_pattern":
                    fields.extend(["rotation_type", "feature_identifiable", "adjacency_resolved"])
                    alternatives.extend([["orientation_angle"], ["point", "reference_point"]])
                    task["rotation_type_hint"] = params.get("rotation_type", "unknown")
                elif kind == "identity_at_time":
                    fields.append("rank")
                elif kind == "periodic_continuation":
                    fields.append("adjacency_resolved")
                    alternatives.extend([["phase"], ["value"]])
                elif kind == "motion_property_trend" and params.get("metric") == "frequency":
                    fields.extend(["cycle_marker", "adjacency_resolved"])
                    alternatives.extend([["phase"], ["value"]])
                    task["phase_unit"] = params.get("phase_unit", "unknown")
                else:
                    fields.append("point")
                    reference = slot["reference_frame"]
                    if reference in {"scene", "body", "object"}:
                        fields.append("reference_point")
                    if reference in {"body", "object"}:
                        fields.append("scale")
                    if params.get("metric") == "amplitude":
                        fields.append("phase")
                    task["axis"] = params.get("axis", "x")
                if "metric" in params:
                    task["metric"] = params["metric"]
                tasks.append(task)
        return cls(tuple(tasks))

    def to_dict(self):
        return {"basis": "program_requirements", "tasks": copy.deepcopy(list(self.tasks))}

    @property
    def operations(self):
        return {t["operation"] for t in self.tasks}

    @property
    def record_fields(self):
        fields = {f for t in self.tasks for f in t["required_fields"]}
        fields.update(f for t in self.tasks for a in t["one_of_fields"] for f in a)
        if "identity_at_time" in self.operations:
            fields.add("point")
        return fields

    def record_missing(self, record, *, stored=False):
        """Missing wire measurements, not a conclusion about visual truth."""
        issues = []
        if record["visibility"] != "visible" or record["basis"] != "visual_observation":
            return issues

        def present(field):
            key = (
                "source_" + field
                if stored and field in {"point", "reference_point", "scale"}
                else field
            )
            return key in record and record[key] is not None and record[key] != ""

        for task in self.tasks:
            if task["slot_id"] != record["slot_id"]:
                continue
            missing = [f for f in task["required_fields"] if not present(f)]
            if task["operation"] == "rotation_pattern":
                mode = record.get("rotation_type")
                if mode == "orbit" and not all(present(f) for f in ["point", "reference_point"]):
                    missing.append("point + reference_point for orbit")
                if mode in {"self_spin", "heading"} and not present("orientation_angle"):
                    missing.append("orientation_angle for self_spin/heading")
            alternatives = task["one_of_fields"]
            if alternatives and not any(all(present(f) for f in group) for group in alternatives):
                missing.append(" or ".join(" + ".join(group) for group in alternatives))
            if missing:
                issues.append(
                    f"{record['slot_id']} at {record['frame_id']}: missing {', '.join(missing)}"
                )
        return issues

    def validate(self, value, frame_map):
        slots = {t["slot_id"] for t in self.tasks}
        for issue in value["gaps"]:
            if issue.get("slot_id") not in slots:
                raise ProtocolError("Observation gaps must name a requested slot_id")
            if not issue["description"].strip():
                raise ProtocolError("Observation gaps require a concrete explanation")
        issues = []
        if not value["complete"] and not value["gaps"]:
            issues.append(
                "complete=false requires a concrete slot-specific gap explaining unfinished work"
            )
        for sid in sorted(slots):
            rows = [r for r in value["records"] if r["slot_id"] == sid]
            gaps = [g for g in value["gaps"] if g.get("slot_id") == sid]
            if not rows and not gaps:
                issues.append(
                    f"slot {sid}: provide observed records or a slot-specific reason it cannot be read"
                )
            for row in rows:
                timestamp = frame_map[row["frame_id"]]["timestamp_seconds"]
                explained = any(
                    "span" not in g or g["span"][0] <= timestamp <= g["span"][1] for g in gaps
                )
                unreadable = row["visibility"] != "visible" or row["basis"] != "visual_observation"
                if unreadable:
                    if not explained and not row.get("description", "").strip():
                        issues.append(
                            f"{sid} at {row['frame_id']}: explain {row['visibility']} in description or gaps"
                        )
                elif not explained:
                    issues.extend(self.record_missing(row))
        if issues:
            raise ProtocolError("Observation contract: " + "; ".join(issues[:12]))

    def normalize(self, value, span, frame_map=None):
        """Fill structural omissions and derive explicit gaps; never synthesize measurements."""
        value = copy.deepcopy(value)
        added = []
        for gap in value["gaps"]:
            if gap.get("bbox", "absent") is None:
                gap.pop("bbox")
                added.append("gaps:unavailable_bbox")
        for field, default in {
            "associations": [],
            "containments": [],
            "anchor_states": [],
            "superseded_observation_ids": [],
            "candidate_coverage_complete": False,
            "reference_status": "unknown",
        }.items():
            if field not in value:
                value[field] = copy.deepcopy(default)
                added.append(field)
        for row in value["records"]:
            timestamp = (frame_map or {}).get(row["frame_id"], {}).get("timestamp_seconds")
            row_span = list(span) if timestamp is None else [timestamp, timestamp]

            def explained(slot_id=row["slot_id"], interval=row_span):
                return any(
                    g.get("slot_id") == slot_id
                    and (
                        "span" not in g
                        or g["span"][0] <= interval[0] <= interval[1] <= g["span"][1]
                    )
                    for g in value["gaps"]
                )

            nullable = {"point", "reference_point", "scale", "orientation_angle"}
            nulls = [k for k in nullable if k in row and row[k] is None]
            for key in nulls:
                row.pop(key)
                added.append("unavailable_measurement:" + key)
            rotation_unknown = "rotation_pattern" in self.operations and (
                row.get("rotation_type") == "unknown"
                or row.get("feature_identifiable") is False
                or row.get("adjacency_resolved") is False
            )
            if (
                (nulls and row.get("description", "").strip()) or rotation_unknown
            ) and not explained():
                value["gaps"].append(
                    {
                        "kind": "detail",
                        "slot_id": row["slot_id"],
                        "span": row_span,
                        "description": row.get("description")
                        or "Rotation type, feature or temporal connection explicitly unresolved",
                        "source": "program_measurement_contract",
                    }
                )
                added.append("gaps:unavailable_measurement")
            if row["visibility"] != "visible" or row["basis"] != "visual_observation":
                if (
                    row["visibility"] == "absent"
                    and row["basis"] == "visual_observation"
                    and value["candidate_coverage_complete"]
                    and "motion_condition_filter" in self.operations
                ):
                    continue
                if row.get("description", "").strip() and not explained():
                    value["gaps"].append(
                        {
                            "kind": "localization" if row["visibility"] == "absent" else "detail",
                            "slot_id": row["slot_id"],
                            "span": row_span,
                            "description": row["description"],
                            "source": "program_measurement_contract",
                        }
                    )
                    added.append("gaps:unreadable_record")
        return value, added

    def schema(self, payload):
        from .contracts import OBSERVE

        result = copy.deepcopy(OBSERVE)
        props = result["properties"]
        record = props["records"]["items"]
        allowed = set(record["required"]) | self.record_fields | {"description"}
        record["properties"] = {k: v for k, v in record["properties"].items() if k in allowed}
        for field in ("point", "reference_point", "scale", "orientation_angle"):
            if field in record["properties"]:
                record["properties"][field] = {
                    "anyOf": [record["properties"][field], {"type": "null"}]
                }
        slot_ids = sorted({t["slot_id"] for t in self.tasks})
        record["properties"]["slot_id"] = {"enum": slot_ids}
        record["properties"]["frame_id"] = {"enum": [f["frame_id"] for f in payload["frames"]]}
        props["gaps"]["items"]["required"] = ["kind", "description", "slot_id"]
        props["gaps"]["items"]["properties"]["slot_id"] = {"enum": slot_ids}
        props["entities"]["items"]["required"] = ["id", "target_id", "description"]
        props["entities"]["items"]["properties"]["target_id"] = {
            "enum": [t["id"] for t in payload["targets"]]
        }
        identity = "identity_at_time" in self.operations
        handoff = payload.get("handoff", {})
        if payload.get("identity_requirements"):
            from .contracts import S, arr, obj

            props["identity_matches"] = arr(
                obj(
                    {
                        "key": {"enum": [r["key"] for r in payload["identity_requirements"]]},
                        "status": {"enum": ["same_entity", "unknown"]},
                        "entity_id": S,
                        "evidence_frames": arr(
                            {"enum": [f["frame_id"] for f in payload["frames"]]}
                        ),
                        "reason": S,
                    },
                    ["key", "status", "reason"],
                )
            )
        if not identity and not handoff.get("entities"):
            props.pop("associations")
        if not identity:
            props.pop("containments")
        if not identity and "motion_condition_filter" not in self.operations:
            props.pop("candidate_coverage_complete")
        if not payload.get("anchors"):
            props.pop("anchor_states")
        if not handoff.get("previous_observations"):
            props.pop("superseded_observation_ids")
        result["required"] = [k for k in result["required"] if k in props]
        if payload.get("identity_requirements"):
            result["required"] = [k for k in result["required"] if k != "associations"]
        return result
