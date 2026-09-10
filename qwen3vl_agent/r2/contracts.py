"""One schema source for prompts, validation and small structured observations."""

from __future__ import annotations

import copy
import json
import re

from .types import OPERATIONS, ProtocolError


def obj(properties, required=None):
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties) if required is None else required,
        "additionalProperties": False,
    }


def arr(items, **kwargs):
    return {"type": "array", "items": items, **kwargs}


S = {"type": "string", "minLength": 1}
TEXT = {"type": "string"}
N = {"type": "number"}
B = {"type": "boolean"}
POINT = arr({"type": "number", "minimum": 0, "maximum": 1000}, minItems=2, maxItems=2)
SPAN = arr({"type": "number", "minimum": 0}, minItems=2, maxItems=2)
BOX = arr({"type": "number", "minimum": 0, "maximum": 1000}, minItems=4, maxItems=4)
REFERENCE = {"enum": ["screen", "scene", "body", "object", "unknown"]}
TARGET = obj({"id": S, "description": S})
SLOT = obj({"id": S, "target_id": S, "property": S, "description": S, "reference_frame": REFERENCE})
SLOT["properties"]["reference_evidence"] = TEXT
SCOPE = obj(
    {
        "kind": {"enum": ["full", "start", "end", "interval", "semantic"]},
        "interval": {"anyOf": [SPAN, {"type": "null"}]},
        "description": TEXT,
    }
)
ANCHOR = obj(
    {
        "id": S,
        "kind": {"enum": ["start", "end", "time", "semantic"]},
        "time": {"type": ["number", "null"], "minimum": 0},
        "description": TEXT,
    }
)
PARAMS = obj(
    {
        "metric": {"enum": ["speed", "frequency", "amplitude"]},
        "axis": {"enum": ["x", "y"]},
        "projection": S,
        "motion_condition": TEXT,
        "attribute_equals": {"type": ["string", "number", "boolean", "null"]},
        "query_anchor_id": S,
        "rotation_type": {"enum": ["orbit", "heading", "self_spin", "unknown"]},
        "phase_unit": S,
        "allow_role_correspondence": B,
    },
    [],
)
OP = obj(
    {
        "id": S,
        "op": {"enum": list(OPERATIONS)},
        "target_ids": arr(S, minItems=1),
        "slot_ids": arr(S, minItems=1),
        "parameters": PARAMS,
    }
)
QUERY = obj(
    {
        "targets": arr(TARGET, minItems=1, maxItems=12),
        "slots": arr(SLOT, minItems=1, maxItems=12),
        "operations": arr(OP, minItems=1, maxItems=10),
        "scope": SCOPE,
        "anchors": arr(ANCHOR, maxItems=6),
        "fast_motion": B,
        "unresolved": arr(S),
    }
)
GAP_KINDS = [
    "localization",
    "coverage",
    "identity",
    "stage_order",
    "temporal_resolution",
    "detail",
    "reference",
    "phase_alias",
    "boundary",
    "conflict",
    "protocol",
]
GAP = obj(
    {
        "kind": {"enum": GAP_KINDS},
        "description": S,
        "span": SPAN,
        "slot_id": S,
        "bbox": {"anyOf": [BOX, {"type": "null"}]},
    },
    ["kind", "description"],
)
RECORD = obj(
    {
        "slot_id": S,
        "entity_id": S,
        "frame_id": S,
        "visibility": {"enum": ["visible", "occluded", "absent", "unknown"]},
        "value": {"type": ["string", "number", "boolean", "null"]},
        "point": POINT,
        "reference_point": POINT,
        "scale": {"type": "number", "exclusiveMinimum": 0},
        "phase": S,
        "cycle_marker": B,
        "rank": {"type": "integer", "minimum": 1},
        "orientation_angle": N,
        "rotation_type": {"enum": ["orbit", "heading", "self_spin", "unknown"]},
        "feature_identifiable": B,
        "adjacency_resolved": B,
        "motion": {"enum": ["moving", "not_observed", "unknown"]},
        "condition_satisfied": {"type": ["boolean", "null"]},
        "basis": {"enum": ["visual_observation", "propagated_inference", "unresolved"]},
        "description": TEXT,
    },
    ["slot_id", "entity_id", "frame_id", "visibility", "value", "basis"],
)
LINK = obj({"from_node": S, "to_entity": S, "kind": {"enum": ["same_entity", "same_role"]}})
ALTERNATIVE = obj({"links": arr(LINK, minItems=1), "evidence_frames": arr(S, minItems=1)})
ASSOCIATION = obj(
    {
        "group_id": S,
        "alternatives": arr(ALTERNATIVE, minItems=1, maxItems=16),
        "supersedes": arr(S),
        "unresolved_extra": B,
        "relation_preserved": B,
    }
)
OBSERVE = obj(
    {
        "entities": arr(
            obj({"id": S, "target_id": S, "description": S, "part_of": TEXT}), maxItems=16
        ),
        "records": arr(RECORD, maxItems=48),
        "associations": arr(ASSOCIATION, maxItems=12),
        "containments": arr(
            obj(
                {
                    "hidden_target_id": S,
                    "carrier_entity_id": S,
                    "frame_id": S,
                    "status": {"enum": ["visible_reveal", "possible_transfer", "release"]},
                }
            )
        ),
        "gaps": arr(GAP),
        "reference_status": {"enum": ["stable", "moving", "unknown"]},
        "candidate_coverage_complete": B,
        "complete": B,
    }
)
OBSERVE["properties"]["anchor_states"] = arr(
    obj(
        {
            "anchor_id": S,
            "frame_id": S,
            "status": {"enum": ["before", "at", "after"]},
            "description": S,
        }
    )
)
OBSERVE["properties"]["superseded_observation_ids"] = arr(S, uniqueItems=True)
LOCATE = obj(
    {
        "candidates": arr(obj({"start_frame": S, "end_frame": S, "reason": S}), maxItems=4),
        "unresolved": arr(S),
    }
)
FINAL_GAP = copy.deepcopy(GAP)
FINAL_GAP["properties"]["evidence_ids"] = arr(S, uniqueItems=True)
FINAL = obj(
    {
        "prediction": S,
        "evidence_ids": arr(S),
        "assessments": arr(
            obj(
                {
                    "label": S,
                    "status": {"enum": ["supported", "contradicted", "unknown"]},
                    "evidence_ids": arr(S),
                }
            )
        ),
        "weakest_premise": TEXT,
        "recheck": {"anyOf": [FINAL_GAP, {"type": "null"}]},
        "unresolved": arr(S),
    }
)
SCHEMAS = {
    "compile_intent": QUERY,
    "compile_discriminants": QUERY,
    "locate": LOCATE,
    "observe": OBSERVE,
    "final": FINAL,
}


def stage_schema(role, payload):
    """The exact same per-call contract is displayed and validated."""
    if role == "observe" and payload.get("observation_spec"):
        from .observation import ObservationSpec

        return ObservationSpec(tuple(payload["observation_spec"]["tasks"])).schema(payload)
    result = copy.deepcopy(SCHEMAS[role])
    if role == "final":
        labels = [c["label"] for c in payload.get("options", [])]
        assessments = result["properties"]["assessments"]
        item = assessments.pop("items")
        assessments.update(minItems=len(labels), maxItems=len(labels), items=False)
        if labels:
            result["properties"]["prediction"] = {"enum": labels}
            assessments["prefixItems"] = []
            for label in labels:
                entry = copy.deepcopy(item)
                entry["properties"]["label"] = {"const": label}
                if payload.get("assessment_policy"):
                    entry["properties"]["operation_ids"] = arr(S, uniqueItems=True)
                assessments["prefixItems"].append(entry)
    return result


def repair_context(role, payload, raw):
    """Actionable diagnostics, containing no hidden options in visual stages."""
    result = {}
    if role == "observe":
        result["required_measurements"] = payload.get("observation_spec", {})
        result["identity_requirements"] = payload.get("identity_requirements", [])
        result["allowed_frame_ids"] = [f["frame_id"] for f in payload.get("frames", [])]
        result["instruction"] = (
            "Fill missing records/fields, or give a concrete gap for each unreadable slot. Do not invent evidence."
        )
    if role == "final":
        labels = [c["label"] for c in payload.get("options", [])]
        result["required_assessment_order"] = labels
        result["allowed_evidence_ids"] = [r["id"] for r in payload.get("observations", [])]
        result["assessment_policy"] = payload.get("assessment_policy", {})
        result["instruction"] = (
            "Return exactly one assessment per original label, in order. With no evidence use status=unknown and evidence_ids=[]."
        )
        try:
            value = json.loads(raw)
            actual = [a.get("label") for a in value.get("assessments", [])]
            result.update(
                missing_labels=[label for label in labels if label not in actual],
                duplicate_labels=list(
                    dict.fromkeys(label for label in actual if actual.count(label) > 1)
                ),
                unexpected_labels=[label for label in actual if label not in labels],
                actual_assessment_order=actual,
            )
        except (ValueError, TypeError, AttributeError):
            pass
    return result


def parse(raw, role, schema=None):
    import jsonschema

    text = raw.strip()
    if text.startswith("```json\n") and text.endswith("```"):
        text = text[8:-3]

    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ProtocolError("duplicate key: " + key)
            result[key] = value
        return result

    try:
        value = json.loads(
            text,
            object_pairs_hook=pairs,
            parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite JSON")),
        )
        jsonschema.Draft202012Validator(schema if schema is not None else SCHEMAS[role]).validate(
            value
        )
        return value
    except (ValueError, jsonschema.ValidationError) as exc:
        raise ProtocolError(str(exc)[:2400]) from exc


def validate_query(query, previous=None, hint=None, question=None):
    q = copy.deepcopy(query)
    if question is not None:
        geometric = {
            sid
            for op in q["operations"]
            if op["op"]
            in {"direction_sequence", "path_shape", "rotation_pattern", "motion_property_trend"}
            for sid in op["slot_ids"]
        }
        for slot in q["slots"]:
            if slot["id"] not in geometric or slot["reference_frame"] not in {"body", "object"}:
                continue
            quote = slot.get("reference_evidence", "")
            relation = r"relative|with respect|around|toward|away from|left of|right of|in front of|behind|viewpoint|相对|参照|参考|绕|朝|远离|以.+为|左侧|右侧|视角"
            explicit = (
                quote
                and quote in question
                and re.search(
                    relation,
                    quote,
                    re.IGNORECASE,
                )
            )
            if not explicit:
                if re.search(relation, question, re.IGNORECASE):
                    raise ProtocolError(
                        "Body/object reference requires reference_evidence quoting the question's reference relation"
                    )
                slot["reference_frame"] = "screen"
                slot.pop("reference_evidence", None)
    for name in ("targets", "slots", "operations", "anchors"):
        ids = [v["id"] for v in q[name]]
        if len(ids) != len(set(ids)):
            raise ProtocolError("duplicate " + name + " IDs")
    targets = {v["id"] for v in q["targets"]}
    slots = {v["id"] for v in q["slots"]}
    if any(s["target_id"] not in targets for s in q["slots"]):
        raise ProtocolError("slot references unknown target")
    for op in q["operations"]:
        if set(op["target_ids"]) - targets or set(op["slot_ids"]) - slots:
            raise ProtocolError("operation references unknown target/slot")
        if op["op"] == "motion_property_trend" and "metric" not in op["parameters"]:
            raise ProtocolError("motion trend requires speed/frequency/amplitude metric")
    if hint and not any(op["op"] == hint for op in q["operations"]):
        raise ProtocolError("compiler ignored the explicit operation hint")
    if q["scope"]["kind"] == "interval":
        span = q["scope"]["interval"]
        if span is None or not span[0] < span[1]:
            raise ProtocolError("interval scope requires increasing boundaries")
    for anchor in q["anchors"]:
        if anchor["kind"] == "time" and anchor["time"] is None:
            raise ProtocolError("numeric anchor requires a time")
    if previous:
        # Options add discriminating slots; they cannot rewrite the user's intent/permissions.
        if (
            q["targets"] != previous["targets"]
            or q["scope"] != previous["scope"]
            or q["anchors"] != previous["anchors"]
        ):
            raise ProtocolError("discriminants must preserve targets, scope and anchors")
        old = {s["id"]: s for s in previous["slots"]}
        if any(
            next((s for s in q["slots"] if s["id"] == key), None) != value
            for key, value in old.items()
        ):
            raise ProtocolError("discriminants must preserve existing slots")
        new_ops = {o["id"]: o for o in q["operations"]}
        for old_op in previous["operations"]:
            new_op = new_ops.get(old_op["id"])
            if (
                new_op is None
                or any(new_op[key] != old_op[key] for key in ("op", "target_ids", "parameters"))
                or not set(old_op["slot_ids"]) <= set(new_op["slot_ids"])
            ):
                raise ProtocolError("discriminants must preserve original operations")
    return q


def validate_observation(
    value, query, frame_map, known_nodes, max_records=48, previous_observations=(), spec=None
):
    ids = {v["id"] for v in value["entities"]}
    targets = {t["id"] for t in query["targets"]}
    slots = {s["id"] for s in query["slots"]}
    if len(ids) != len(value["entities"]) or len(value["records"]) > max_records:
        raise ProtocolError("duplicate entities or too many observations")
    keys = [(r["slot_id"], r["entity_id"], r["frame_id"]) for r in value["records"]]
    if len(keys) != len(set(keys)):
        raise ProtocolError("duplicate frame/entity/slot observation")
    known_observations = {r["id"] for r in previous_observations}
    if set(value.get("superseded_observation_ids", [])) - known_observations:
        raise ProtocolError("cannot correct an observation absent from this call's context")
    if any(e["target_id"] not in targets for e in value["entities"]):
        raise ProtocolError("unknown query target")

    def refs(values):
        if any(v not in frame_map for v in values):
            raise ProtocolError("reference to a frame not shown in this call")

    for r in value["records"]:
        if r["slot_id"] not in slots or r["entity_id"] not in ids:
            raise ProtocolError("observation references unknown slot/entity")
        refs([r["frame_id"]])
        if (
            r["basis"] == "visual_observation"
            and r["visibility"] != "visible"
            and ("point" in r or "orientation_angle" in r)
        ):
            raise ProtocolError("unseen point cannot be a visual observation")
    for anchor in value.get("anchor_states", []):
        refs([anchor["frame_id"]])
        if anchor["anchor_id"] not in {a["id"] for a in query["anchors"]}:
            raise ProtocolError("unknown temporal anchor")
    for group in value["associations"]:
        for alternative in group["alternatives"]:
            refs(alternative["evidence_frames"])
            for link in alternative["links"]:
                if link["from_node"] not in known_nodes or link["to_entity"] not in ids:
                    raise ProtocolError("association references an unknown node")
    group_ids = [g["group_id"] for g in value["associations"]]
    if len(set(group_ids)) != len(group_ids):
        raise ProtocolError("duplicate association group IDs")
    for item in value["containments"]:
        refs([item["frame_id"]])
        if item["hidden_target_id"] not in targets or item["carrier_entity_id"] not in ids:
            raise ProtocolError("containment references unknown target/entity")
    if spec is None:
        from .observation import ObservationSpec

        spec = ObservationSpec.from_query(query)
    spec.validate(value, frame_map)
    return value


def validate_final_references(value, payload):
    """Structural identity and references stay strict, independently of certification."""
    labels = [c["label"] for c in payload["options"]]
    actual = [a["label"] for a in value["assessments"]]
    if actual != labels or (labels and value["prediction"] not in labels):
        raise ProtocolError(
            "Final option contract: "
            + json.dumps(repair_context("final", payload, json.dumps(value)))
        )
    allowed = {r["id"] for r in payload["observations"]}
    cited = (
        set(value["evidence_ids"])
        | {r for a in value["assessments"] for r in a["evidence_ids"]}
        | set((value.get("recheck") or {}).get("evidence_ids", []))
    )
    if cited - allowed:
        raise ProtocolError(
            "Final cited unavailable observations: " + ", ".join(sorted(cited - allowed))
        )
    policy = payload.get("assessment_policy")
    if policy:
        operation_ids = {oid for a in value["assessments"] for oid in a.get("operation_ids", [])}
        if operation_ids - set(policy["operations"]):
            raise ProtocolError("Final cites unknown operation IDs")
    return value


def validate_final(value, payload):
    validate_final_references(value, payload)
    missing = [
        a["label"]
        for a in value["assessments"]
        if a["status"] != "unknown" and not a["evidence_ids"]
    ]
    if missing:
        raise ProtocolError(
            "Options " + ", ".join(missing) + " need supporting observations; otherwise use unknown"
        )
    from .evidence import validate_assessment_evidence

    validate_assessment_evidence(value, payload)
    return value


def certify_final(value, payload):
    """Retain a valid prediction without certifying unsupported model claims.

    Runtime keeps the raw response and records this program transformation separately.
    Malformed options and nonexistent references still enter bounded format repair.
    """
    value = copy.deepcopy(value)
    if value.get("recheck") and value["recheck"].get("bbox", "absent") is None:
        value["recheck"].pop("bbox")
    validate_final_references(value, payload)
    try:
        return validate_final(value, payload)
    except ProtocolError as exc:
        value = copy.deepcopy(value)
        for assessment in value["assessments"]:
            assessment.update(status="unknown", evidence_ids=[])
            if "operation_ids" in assessment:
                assessment["operation_ids"] = []
        value["unresolved"].append("program_evidence_certification: " + str(exc))
        return validate_final(value, payload)
