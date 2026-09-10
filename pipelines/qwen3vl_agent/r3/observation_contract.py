"""One declaration for observation wire schemas, validation and reference boundaries."""
from __future__ import annotations

import copy
import json
from typing import Any

from .types import EventQuery, ProtocolError

OBSERVATION_VERSION = 3


class ObservationContractError(ProtocolError):
    def __init__(self, errors: list[dict]):
        self.errors = errors
        super().__init__("; ".join(f"{e['path']}: {e['message']}" for e in errors))


def fail(path: str, message: str) -> None:
    raise ObservationContractError([{"path": path, "code": "contract", "message": message}])


def string(**kw):
    return {"type": "string", "minLength": 1, **kw}


def array(items, **kw):
    return {"type": "array", "items": items, **kw}


def obj(properties, required=None):
    return {"type": "object", "properties": properties,
            "required": list(properties) if required is None else required, "additionalProperties": False}


REFS = array(string(), maxItems=4)
FACT = obj({"fact_id": string(pattern=r"V[0-9]+"),
            "kind": string(enum=["state", "action", "change"]), "description": string(),
            "evidence_refs": {**REFS, "minItems": 1}})
CROP = obj({"frame_id": string(), "bbox_xyxy_1000": array({"type": "number"}, minItems=4, maxItems=4)})
SUMMARY = obj({"description": string(), "evidence_refs": {**REFS, "minItems": 1}})
PHASES = ("before_start_refs", "start_refs", "last_active_refs", "completion_refs", "after_end_refs", "reset_refs")
EVENT_FIELDS = {
    "local_id": string(), "target_id": string(), "fact_refs": array(string(), uniqueItems=True),
    "description": string(), "category": string(), "fact_kind": string(enum=["visual_event", "screen_text_event", "utterance", "reported_event"]),
    "evidence_refs": array(string(), minItems=1), "completed": {"type": "boolean"},
    "match": string(enum=["clear", "uncertain", "rejected"]),
    "actor_ref": {"type": "string", "default": ""}, "object_ref": {"type": "string", "default": ""},
    "actor_binding_refs": {**array(string()), "default": []},
    **{p: {**array(string()), "default": []} for p in PHASES},
    "replay_status": string(enum=["original", "replay", "unknown"], default="unknown"),
    "attributes": {"type": "object", "default": {}, "additionalProperties": obj({"value": {}, "evidence_refs": array(string(), minItems=1)})},
    "cooccurrence": {"type": "object", "default": {}, "additionalProperties": obj({"status": string(enum=["present", "absent", "unknown"]), "evidence_refs": array(string())})},
    "unresolved_reasons": {**array(string()), "default": []},
}
EVENT = obj(EVENT_FIELDS, [k for k, v in EVENT_FIELDS.items() if "default" not in v])
DISPOSITION = obj({"fact_id": string(), "status": string(enum=["event", "unrelated", "uncertain"]),
                   "event_ids": array(string(), uniqueItems=True), "reason": string()},
                  ["fact_id", "status", "reason"])
ASSESSMENT = obj({"target_id": string(), "status": string(enum=["observed", "absent", "uncertain"]),
                  "event_ids": array(string(), uniqueItems=True), "reason": string(),
                  # This summarizes several facts/events, not one atomic O1 fact.
                  "evidence_refs": array(string(), minItems=1)},
                 ["target_id", "status", "reason"])
SCHEMAS = {
    "observe_visual": obj({"version": {"type": "integer", "const": OBSERVATION_VERSION},
        "summary": SUMMARY, "facts": array(FACT, maxItems=32), "unresolved": array(string()),
        "crop_requests": array(CROP, maxItems=2), "truncated": {"type": "boolean"}}),
    "observe_events": obj({"version": {"type": "integer", "const": OBSERVATION_VERSION},
        "fact_dispositions": array(DISPOSITION, maxItems=32), "events": array(EVENT, maxItems=32),
        "target_assessments": array(ASSESSMENT), "unresolved": array(string()), "truncated": {"type": "boolean"}}),
    "final": obj({"prediction": string(), "evidence_refs": array(string(), maxItems=8, uniqueItems=True)}),
}


def contract(role: str) -> dict:
    return copy.deepcopy(SCHEMAS[role])


def contract_text(role: str) -> str:
    return json.dumps(contract(role), ensure_ascii=False, separators=(",", ":"))


def validate(value: Any, schema: dict, path="$", changes=None):
    """Small schema interpreter; defaults are shared with the displayed protocol."""
    import math
    import re
    if "type" not in schema:
        return copy.deepcopy(value)
    kind = schema["type"]
    good = {"object": isinstance(value, dict), "array": isinstance(value, list),
            "string": isinstance(value, str), "boolean": isinstance(value, bool),
            "integer": isinstance(value, int) and not isinstance(value, bool),
            "number": isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)}[kind]
    if not good:
        fail(path, f"expected {kind}; values are not coerced")
    if "const" in schema and value != schema["const"]:
        fail(path, f"expected {schema['const']}")
    if "enum" in schema and value not in schema["enum"]:
        fail(path, "invalid enum value")
    if kind == "string":
        if len(value.strip()) < schema.get("minLength", 0):
            fail(path, "must not be blank")
        if "pattern" in schema and not re.fullmatch(schema["pattern"], value):
            fail(path, "invalid identifier")
    if kind == "array":
        if not schema.get("minItems", 0) <= len(value) <= schema.get("maxItems", 100000):
            fail(path, f"array size {len(value)} outside contract "
                       f"(minimum {schema.get('minItems', 0)}, maximum {schema.get('maxItems', 'unbounded')})")
        if schema.get("uniqueItems") and len({json.dumps(v, sort_keys=True) for v in value}) != len(value):
            fail(path, "duplicate entries")
        return [validate(v, schema["items"], f"{path}[{i}]", changes) for i, v in enumerate(value)]
    if kind == "object":
        props = schema.get("properties", {})
        extra = schema.get("additionalProperties", False)
        for key in value:
            if key not in props and extra is False:
                fail(f"{path}.{key}", "undeclared field")
        result = {}
        for key, rule in props.items():
            if key not in value or value[key] is None:
                if key in schema.get("required", []):
                    fail(f"{path}.{key}", "required field is missing or null")
                if "default" in rule:
                    result[key] = copy.deepcopy(rule["default"])
                    if changes is not None:
                        changes.append({"path": f"{path}.{key}", "action": "default", "value": result[key]})
                continue
            result[key] = validate(value[key], rule, f"{path}.{key}", changes)
        if isinstance(extra, dict):
            result.update({k: validate(v, extra, f"{path}.{k}", changes) for k, v in value.items() if k not in props})
        return result
    return copy.deepcopy(value)


def checked_refs(values, allowed, path):
    for i, ref in enumerate(values):
        if ref not in allowed:
            fail(f"{path}[{i}]", "reference was not supplied in this evidence context")


def derived_field(row, key, value, path, changes):
    """Fill redundant links only when omitted; explicit contradictions are still errors."""
    if key not in row:
        row[key] = copy.deepcopy(value)
        if changes is not None:
            changes.append({"path": path + "." + key, "action": "derive_from_event_links", "value": row[key]})


def normalize_visual(data, frame_catalog, changes=None):
    data = validate(data, SCHEMAS["observe_visual"], changes=changes)
    checked_refs(data["summary"]["evidence_refs"], frame_catalog, "$.summary.evidence_refs")
    ids = set()
    for i, fact in enumerate(data["facts"]):
        if fact["fact_id"] in ids:
            fail(f"$.facts[{i}].fact_id", "duplicate fact ID")
        ids.add(fact["fact_id"])
        checked_refs(fact["evidence_refs"], frame_catalog, f"$.facts[{i}].evidence_refs")
    for i, crop in enumerate(data["crop_requests"]):
        ref, box = crop["frame_id"], crop["bbox_xyxy_1000"]
        if ref not in frame_catalog or frame_catalog[ref].get("crop_transform"):
            fail(f"$.crop_requests[{i}].frame_id", "crop must cite a displayed original frame")
        if not (0 <= box[0] < box[2] <= 1000 and 0 <= box[1] < box[3] <= 1000):
            fail(f"$.crop_requests[{i}].bbox_xyxy_1000", "crop outside frame")
    return data


def normalize_events(data, visual, query: EventQuery, catalog, changes=None):
    from .observation import parse_batch
    data = validate(data, SCHEMAS["observe_events"], changes=changes)
    facts = {f["fact_id"]: f for f in visual["facts"]}
    summary_refs = set(visual["summary"]["evidence_refs"])
    external = {r for r, s in catalog.items() if s["kind"] in {"asr", "subtitle"}}
    available = summary_refs | external | {r for f in facts.values() for r in f["evidence_refs"]}
    events = {e["local_id"]: e for e in data["events"]}
    if len(events) != len(data["events"]):
        fail("$.events", "duplicate event ID")
    for i, event in enumerate(data["events"]):
        path = f"$.events[{i}]"
        if event["target_id"] not in {t.target_id for t in query.targets}:
            fail(path + ".target_id", "unknown query target")
        checked_refs(event["fact_refs"], facts, path + ".fact_refs")
        if event["fact_kind"] in {"visual_event", "screen_text_event"} and not event["fact_refs"]:
            fail(path + ".fact_refs", "visual events must cite O1 facts")
        allowed = external | {r for f in event["fact_refs"] for r in facts[f]["evidence_refs"]}
        for key in ("evidence_refs", "actor_binding_refs", *PHASES):
            checked_refs(event[key], allowed, path + "." + key)
        for key in ("attributes", "cooccurrence"):
            for name, item in event[key].items():
                checked_refs(item["evidence_refs"], allowed, f"{path}.{key}.{name}.evidence_refs")
        if event["completed"] and event["fact_kind"] in {"visual_event", "screen_text_event"} and not event["completion_refs"]:
            fail(path + ".completion_refs", "completed requires cited completion evidence")
    dispositions = {}
    for i, disposition in enumerate(data["fact_dispositions"]):
        fid = disposition["fact_id"]
        if fid not in facts:
            fail(f"$.fact_dispositions[{i}].fact_id", "unknown O1 fact ID")
        if fid in dispositions:
            fail(f"$.fact_dispositions[{i}].fact_id", "duplicate O1 fact disposition")
        dispositions[fid] = disposition
    missing = [fact for fid, fact in facts.items() if fid not in dispositions]
    if missing:
        # The repair must review the supplied facts, not infer their absence from an
        # empty event list. Keep concrete IDs/descriptions in the durable error record.
        raise ObservationContractError([{
            "path": "$.fact_dispositions", "code": "missing_fact_disposition",
            "message": f"O1 fact {f['fact_id']} has no disposition: {f['description']}",
            "fact_id": f["fact_id"], "kind": f["kind"], "evidence_refs": f["evidence_refs"],
        } for f in missing])
    unresolved = list(dict.fromkeys([*visual["unresolved"], *data["unresolved"]]))
    for i, d in enumerate(data["fact_dispositions"]):
        expected = [e["local_id"] for e in events.values() if d["fact_id"] in e["fact_refs"]]
        derived_field(d, "event_ids", expected, f"$.fact_dispositions[{i}]", changes)
        if d["status"] == "event":
            if not d["event_ids"] or set(d["event_ids"]) != set(expected):
                fail(f"$.fact_dispositions[{i}].event_ids", "event links must match event fact references")
        elif d["event_ids"] or expected:
            fail(f"$.fact_dispositions[{i}]", "non-event disposition conflicts with event links")
        if d["status"] == "uncertain":
            unresolved.append("unexplained_fact:" + d["fact_id"])
    targets = {t.target_id for t in query.targets}
    assessments = {x["target_id"]: x for x in data["target_assessments"]}
    if len(assessments) != len(data["target_assessments"]) or set(assessments) != targets:
        fail("$.target_assessments", "each query target must be assessed exactly once")
    for i, a in enumerate(data["target_assessments"]):
        matching = [e for e in events.values() if e["target_id"] == a["target_id"] and e["match"] != "rejected"]
        expected = [e["local_id"] for e in matching]
        path = f"$.target_assessments[{i}]"
        derived_field(a, "event_ids", expected, path, changes)
        if a["status"] == "observed" and matching:
            derived_field(a, "evidence_refs", list(dict.fromkeys(r for e in matching for r in e["evidence_refs"])), path, changes)
        if "evidence_refs" not in a:
            fail(path + ".evidence_refs", "an assessment without observed events requires explicit cited evidence")
        checked_refs(a["evidence_refs"], available, f"$.target_assessments[{i}].evidence_refs")
        if set(a["event_ids"]) != set(expected):
            fail(f"$.target_assessments[{i}].event_ids", "must name all non-rejected events for this target")
        if a["status"] == "observed" and not expected:
            fail(f"$.target_assessments[{i}].status", "observed requires an event, including partial events")
        if a["status"] == "absent" and expected:
            fail(f"$.target_assessments[{i}].status", "observed events contradict an absent assessment")
        if a["status"] == "uncertain":
            unresolved.append("target_uncertain:" + a["target_id"])
    # Reuse temporal/type/provider checks before admitting the normalized wire object.
    for i, event in enumerate(data["events"]):
        try:
            parse_batch({"events": [event], "observation_status": "valid", "unresolved": unresolved,
                         "truncated": data["truncated"], "crop_requests": []}, query, catalog)
        except ProtocolError as exc:
            fail(f"$.events[{i}]", str(exc))
    data["unresolved"] = list(dict.fromkeys(unresolved))
    return data
