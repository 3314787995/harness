"""R3 v4 wire declarations, normalization and source-bound event projection."""
from __future__ import annotations

import copy
import json
import math
import re
from typing import TypedDict

from .observation import parse_batch
from .observation_contract import ObservationContractError, fail

TIMELINE_VERSION = 4


class TimelinePatch(TypedDict):
    replaces: list[dict]
    events: list[dict]
    reason: str
    evidence_refs: list[str]


def obj(properties, required=()):
    return {"type": "object", "properties": properties, "required": list(required), "additionalProperties": False}


def arr(items, *, limit=32, default=True):
    value = {"type": "array", "items": items, "maxItems": limit}
    if default:
        value["default"] = []
    return value


STR = {"type": "string", "minLength": 1}
TEXT = {"type": "string", "default": ""}
REFS = arr(STR, limit=4)
FACT_REFS = arr(STR)
PROOF = obj({k: REFS for k in (
    "before_start", "start", "last_active", "completion", "after_end", "reset", "actor_binding"
)})
FACT = obj({"fact_id": STR, "kind": {"enum": ["state", "action", "change"]},
    "phase": {"enum": ["preparation", "action", "result", "state", "unknown"], "default": "unknown"},
    "description": STR, "target_ids": arr(STR),
    "relevance": {"enum": ["target", "unrelated", "uncertain"]},
    "evidence_refs": {**REFS, "minItems": 1}},
    ("fact_id", "kind", "description", "relevance", "evidence_refs"))
ATTRIBUTE = obj({"value": {"type": ["string", "number", "boolean", "array", "object"],
                           "items": True, "additionalProperties": True},
                 "fact_refs": {**FACT_REFS, "minItems": 1}}, ("value", "fact_refs"))
COOCCURRENCE = obj({"status": {"enum": ["present", "absent", "unknown"]},
                    "fact_refs": FACT_REFS}, ("status",))
EVENT = obj({"local_id": STR, "target_id": STR, "description": STR,
    "category": TEXT, "actor_ref": TEXT, "object_ref": TEXT,
    "fact_refs": {**FACT_REFS, "minItems": 1}, "proof": {**PROOF, "default": {}},
    "completed": {"type": "boolean", "default": False},
    "attributes": {"type": "object", "additionalProperties": ATTRIBUTE, "default": {}},
    "cooccurrence": {"type": "object", "additionalProperties": COOCCURRENCE, "default": {}},
    "checks": arr(STR), "unresolved": arr(STR),
    "replay_status": {"enum": ["original", "replay", "unknown"], "default": "unknown"}},
    ("local_id", "target_id", "description", "fact_refs"))
PATCH = obj({"replaces": arr(obj({"event_id": STR, "revision": {"type": "integer", "minimum": 1}},
                                ("event_id", "revision"))),
    "events": arr(EVENT), "reason": STR, "evidence_refs": {**REFS, "minItems": 1}},
    ("events", "reason", "evidence_refs"))
REPORT = obj({"target_id": STR, "status": {"enum": ["observed", "absent", "uncertain"]},
              "reason": STR, "evidence_refs": {**REFS, "minItems": 1}},
             ("target_id", "status", "reason", "evidence_refs"))
SCHEMA = obj({"version": {"type": "integer", "enum": [TIMELINE_VERSION]},
    "summary": obj({"description": STR, "evidence_refs": {**REFS, "minItems": 1}},
                   ("description", "evidence_refs")),
    "facts": arr(FACT), "updates": arr(PATCH, limit=16), "targets": arr(REPORT),
    "unresolved": arr(STR), "truncated": {"type": "boolean"},
    "crop_requests": arr(obj({"frame_id": STR,
        "bbox_xyxy_1000": {"type": "array", "items": {"type": "number"}, "minItems": 4, "maxItems": 4}},
        ("frame_id", "bbox_xyxy_1000")), limit=2)},
    ("version", "summary", "facts", "updates", "targets", "truncated"))


def contract():
    return copy.deepcopy(SCHEMA)


def _normalize(value, schema, path="$", changes=None):
    if schema is True:
        return copy.deepcopy(value)
    expected = schema.get("type")
    types = expected if isinstance(expected, list) else [expected] if expected else []
    matches = {"object": isinstance(value, dict), "array": isinstance(value, list),
        "string": isinstance(value, str), "boolean": isinstance(value, bool),
        "integer": type(value) is int,
        "number": type(value) in (int, float) and math.isfinite(value)}
    if types and not any(matches[t] for t in types):
        fail(path, "expected " + "/".join(types))
    if "enum" in schema and value not in schema["enum"]:
        fail(path, "not an allowed value")
    if isinstance(value, str) and len(value.strip()) < schema.get("minLength", 0):
        fail(path, "nonempty text required")
    if type(value) in (int, float) and value < schema.get("minimum", -math.inf):
        fail(path, "below minimum")
    if isinstance(value, list):
        if not schema.get("minItems", 0) <= len(value) <= schema.get("maxItems", math.inf):
            fail(path, "array length outside allowed bounds")
        return [_normalize(v, schema.get("items", {}), f"{path}[{i}]", changes) for i, v in enumerate(value)]
    if isinstance(value, dict):
        props, result = schema.get("properties", {}), {}
        extra = schema.get("additionalProperties", False)
        for key in value:
            if key not in props and extra is False:
                fail(path + "." + key, "undeclared field")
        for key in schema.get("required", []):
            if key not in value or value[key] is None:
                fail(path + "." + key, "required field missing or null")
        for key in dict.fromkeys([*props, *value]):
            rule = props.get(key, extra)
            if rule is True:
                result[key] = copy.deepcopy(value[key])
                continue
            if key not in value or value[key] is None:
                if "default" not in rule:
                    continue
                raw = copy.deepcopy(rule["default"])
                if changes is not None:
                    changes.append({"path": path + "." + key, "action": "default"})
            else:
                raw = value[key]
            result[key] = _normalize(raw, rule, path + "." + key, changes)
        return result
    return value


def normalize_update(data, query, catalog, current, *, review=False, changes=None, limit=32):
    """Validate only structure, provenance and dependencies; no question/query semantic checker."""
    result = _normalize(data, SCHEMA, changes=changes)
    specs = {t.target_id: t for t in query.targets}

    def sources(values, path, *, frame_only=False):
        for ref in values:
            if ref not in catalog:
                fail(path, "unknown or unshown source reference: " + ref)
            if frame_only and catalog[ref]["kind"] != "frame":
                fail(path, "visual fact/phase requires an image source")
        return list(dict.fromkeys(values))

    sources(result["summary"]["evidence_refs"], "$.summary.evidence_refs")
    facts, local_ids, replaced, used_facts = {}, set(), set(), set()
    for i, fact in enumerate(result["facts"]):
        path = f"$.facts[{i}]"
        if fact["fact_id"] in facts:
            fail(path + ".fact_id", "duplicate fact ID")
        if set(fact["target_ids"]) - specs.keys():
            fail(path + ".target_ids", "unknown target")
        if fact["relevance"] == "target" and not fact["target_ids"]:
            fail(path + ".target_ids", "target-relevant facts require target references")
        sources(fact["evidence_refs"], path + ".evidence_refs")
        facts[fact["fact_id"]] = fact

    def fact_sources(values, path, target=None):
        evidence = []
        for fid in values:
            if fid not in facts:
                fail(path, "unknown fact: " + fid)
            if target and (facts[fid]["relevance"] == "unrelated" or target not in facts[fid]["target_ids"]):
                fail(path, "event uses a fact not assigned to this target")
            evidence.extend(facts[fid]["evidence_refs"])
        return list(dict.fromkeys(evidence))

    projected = []
    for i, patch in enumerate(result["updates"]):
        path = f"$.updates[{i}]"
        sources(patch["evidence_refs"], path + ".evidence_refs")
        for j, old in enumerate(patch["replaces"]):
            if old["event_id"] not in current or current[old["event_id"]]["revision"] != old["revision"]:
                fail(f"{path}.replaces[{j}]", "unknown event or stale revision")
            if old["event_id"] in replaced:
                fail(f"{path}.replaces[{j}]", "an event may be replaced only once per transaction")
            replaced.add(old["event_id"])
        if not patch["events"] and not patch["replaces"]:
            fail(path, "empty patch has no effect")
        rows = []
        for j, event in enumerate(patch["events"]):
            ep = f"{path}.events[{j}]"
            target = specs.get(event["target_id"])
            if target is None:
                fail(ep + ".target_id", "unknown target")
            if event["local_id"] in local_ids:
                fail(ep + ".local_id", "duplicate output event ID")
            local_ids.add(event["local_id"])
            evidence = fact_sources(event["fact_refs"], ep + ".fact_refs", target.target_id)
            used_facts.update(event["fact_refs"])
            visual = target.fact_kind in {"visual_event", "screen_text_event"}
            sources(evidence, ep + ".fact_refs", frame_only=visual)
            for key, values in event["proof"].items():
                sources(values, ep + ".proof." + key, frame_only=visual and key != "actor_binding")
                if key != "actor_binding" and set(values) - set(evidence):
                    fail(ep + ".proof." + key, "phase evidence must belong to this event's cited facts")
            allowed_checks = {"target", "unit", "identity", "completion", "onset", "offset", "replay",
                "attribute:description", "attribute:category", "attribute:actor_ref", "attribute:object_ref"}
            allowed_checks.update("attribute:" + k for k in event["attributes"])
            allowed_checks.update("cooccurrence:" + k for k in event["cooccurrence"])
            if set(event["checks"]) - allowed_checks or (event["checks"] and not review):
                fail(ep + ".checks", "only review may verify declared event facets")
            for check, keys in {"onset": ("before_start", "start"),
                                "offset": ("last_active", "after_end")}.items():
                if check in event["checks"] and visual:
                    second = event["proof"][keys[1]]
                    if check == "offset":
                        second = second or event["proof"]["completion"]
                    if not event["proof"][keys[0]] or not second:
                        fail(ep + ".checks", check + " verification requires both bracket sides")
            if "completion" in event["checks"] and (not event["completed"] or (visual and not event["proof"]["completion"])):
                fail(ep + ".checks", "completion verification requires completion evidence")
            if "completion" in event["checks"] and visual and target.unit_kind in {"action_cycle", "state_transition"} and not event["proof"]["last_active"]:
                fail(ep + ".checks", "a completed action/transition needs active and completion evidence")
            if "replay" in event["checks"] and event["replay_status"] == "unknown":
                fail(ep + ".checks", "unknown replay identity cannot be verified")
            attributes = {}
            for name, attr in event["attributes"].items():
                ev = fact_sources(attr["fact_refs"], ep + ".attributes." + name + ".fact_refs", target.target_id)
                sources(ev, ep + ".attributes." + name, frame_only=visual)
                used_facts.update(attr["fact_refs"])
                attributes[name] = {"value": attr["value"], "evidence_refs": ev}
            cooccurrence = {}
            for name, item in event["cooccurrence"].items():
                ev = fact_sources(item["fact_refs"], ep + ".cooccurrence." + name + ".fact_refs")
                if item["status"] != "unknown" and not ev:
                    fail(ep + ".cooccurrence." + name, "present/absent requires evidence")
                if item["status"] == "unknown" and "cooccurrence:" + name in event["checks"]:
                    fail(ep + ".checks", "unknown cooccurrence cannot be verified")
                sources(ev, ep + ".cooccurrence." + name, frame_only=visual)
                used_facts.update(item["fact_refs"])
                cooccurrence[name] = {"status": item["status"], "evidence_refs": ev}
            row = {"local_id": event["local_id"], "target_id": target.target_id,
                "description": event["description"], "category": event["category"],
                "actor_ref": event["actor_ref"], "object_ref": event["object_ref"],
                "fact_kind": target.fact_kind, "evidence_refs": evidence,
                "completed": event["completed"], "match": "uncertain",
                "attributes": attributes, "cooccurrence": cooccurrence,
                "replay_status": event["replay_status"], "unresolved_reasons": event["unresolved"]}
            row.update({k + "_refs": v for k, v in event["proof"].items()})
            try:
                internal = parse_batch({"observation_status": "valid", "events": [row],
                                        "truncated": result["truncated"]}, query, catalog)["events"][0]
            except (ValueError, TypeError) as exc:
                fail(ep, str(exc))
            internal["verification"] = {k: True for k in event["checks"]} if review and not result["truncated"] else {}
            internal["status"] = "proposed"  # The ledger, never the wire, controls acceptance.
            rows.append(internal)
        projected.append({"replaces": patch["replaces"], "events": rows,
                          "reason": patch["reason"], "evidence_refs": patch["evidence_refs"]})
    if len(local_ids) > limit:
        fail("$.updates", "too many output events")
    reports = {}
    for i, report in enumerate(result["targets"]):
        if report["target_id"] not in specs or report["target_id"] in reports:
            fail(f"$.targets[{i}].target_id", "unknown or repeated target")
        visual = specs[report["target_id"]].fact_kind in {"visual_event", "screen_text_event"}
        sources(report["evidence_refs"], f"$.targets[{i}].evidence_refs", frame_only=visual)
        reports[report["target_id"]] = report
    if set(reports) != set(specs):
        fail("$.targets", "every supplied target requires one assessment")
    gaps = []
    for fid, fact in facts.items():
        if fact["relevance"] == "uncertain" or (fact["relevance"] == "target" and fid not in used_facts):
            gaps.append({"kind": "unexplained_fact", "fact_id": fid,
                         "target_ids": fact["target_ids"] or list(specs)})
    for target, report in reports.items():
        if report["status"] == "absent" and any(target in f["target_ids"] and f["relevance"] != "unrelated" for f in facts.values()):
            gaps.append({"kind": "contradictory_absence", "target_ids": [target]})
        if report["status"] == "uncertain":
            gaps.append({"kind": "uncertain_target", "target_ids": [target]})
    # Reuse the existing normalized crop geometry/source checks.
    parse_batch({"events": [], "truncated": False, "observation_status": "valid",
                 "crop_requests": result["crop_requests"]}, query, catalog)
    for crop in result["crop_requests"]:
        targets = sorted({tid for f in facts.values() if crop["frame_id"] in f["evidence_refs"]
                          and f["relevance"] != "unrelated" for tid in f["target_ids"]})
        gaps.append({"kind": "unread_detail", "target_ids": targets or list(specs)})
    result["projected_patches"], result["gaps"] = projected, gaps
    return result


def repair_guard(raw, repaired, aliases=None):
    """Formatting may change reference spelling, never event content or partitioning."""
    from .runtime import json_object
    try:
        original = json_object(raw, strict=True)
    except ValueError:
        fail("$", "unparseable visual content requires a visual follow-up, not invented text reconstruction")

    aliases = aliases or {}

    def reference(value):
        match = re.fullmatch(r"([FSVEL])0*(\d+)", value) if isinstance(value, str) else None
        if match:
            padded = f"{match[1]}{int(match[2]):02d}"
            return aliases.get(padded, padded)
        return aliases.get(value, value) if isinstance(value, str) else value

    def content(value, schema, key=""):
        if schema is True:
            return copy.deepcopy(value)
        if value is None and "default" in schema:
            value = copy.deepcopy(schema["default"])
        if isinstance(value, list):
            items = [content(v, schema.get("items", {}), key) for v in value]
            return sorted(set(items)) if key.endswith("refs") or key in PROOF["properties"] else items
        if not isinstance(value, dict):
            return reference(value) if key.endswith("refs") or key in {*PROOF["properties"], "fact_id", "local_id", "event_id", "frame_id"} else value
        props, extra = schema.get("properties", {}), schema.get("additionalProperties", False)
        result = {}
        for name in dict.fromkeys([*props, *value]):
            if name in {"version", "revision", "truncated"} or name not in props and extra is False:
                continue
            rule = props.get(name, extra)
            if name in value:
                val = value[name]
            elif isinstance(rule, dict) and "default" in rule:
                val = copy.deepcopy(rule["default"])
            else:
                continue
            result[name] = content(val, rule, name)
        return result
    # Reference spelling/default corrections are allowed; adding V04 to a mixed event
    # is a semantic reassignment and must return to a visual review, not a text repair.
    if content(original, SCHEMA) != content(repaired, SCHEMA):
        fail("$", "format repair changed visual claims, event partition, or verification facets")


def canonicalize(value, aliases, key=""):
    """Translate references only. OCR/category/attribute text such as 'F01' is data."""
    if key in {"value", "attributes", "cooccurrence", "verification"}:
        return copy.deepcopy(value)
    if isinstance(value, dict):
        return {k: canonicalize(v, aliases, k) for k, v in value.items()}
    reference_field = key.endswith("_refs") or key in {*PROOF["properties"], "event_id", "frame_id"}
    if isinstance(value, list):
        return [canonicalize(v, aliases, key) for v in value]
    return aliases.get(value, value) if reference_field and isinstance(value, str) else value
