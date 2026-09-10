"""Compact per-call observation references; canonical evidence stays in the ledger."""

from __future__ import annotations

from typing import Any

from qwen3vl_agent.r4.contracts import ContractError, issue, validate_observation_shape
from qwen3vl_agent.r4.observation import parse_batch


def compact_catalog(catalog: dict, core: list | tuple) -> tuple[dict, dict]:
    public, aliases, counters = {}, {}, {"F": 0, "T": 0}
    for ref, item in sorted(catalog.items(), key=lambda kv: (kv[1]["start_sec"], kv[0])):
        prefix = "F" if item["kind"] == "frame" else "T"
        counters[prefix] += 1
        alias = f"{prefix}{counters[prefix]}"
        aliases[alias] = ref
        in_core = core[0] <= item["start_sec"] < core[1] or (
            item["kind"] != "frame" and item["start_sec"] < core[1] and item["end_sec"] > core[0])
        public[alias] = {
            "id": alias, "kind": item["kind"], "start_sec": item["start_sec"],
            "end_sec": item["end_sec"], "region": "core" if in_core else "context",
            **{k: item[k] for k in ("text", "speaker_id", "membership_sets", "history_start",
                                    "alignment_status", "alignment_error_sec") if k in item},
        }
    return public, aliases


def observation_payload(payload: dict, aliases: dict, public: dict) -> dict:
    # Operations/count alternatives and canonical source metadata are not observation facts.
    sets = [s for s in payload["spec"]["sets"] if s["set_id"] in payload["set_ids"]]
    result = {**payload, "spec": {"sets": sets}, "catalog": public}
    registry = []
    inverse = {ref: alias for alias, ref in aliases.items()}
    for row in payload.get("task_registry", []):
        registry.append({k: ([inverse[r] for r in v if r in inverse] if k == "evidence_refs" else v)
                         for k, v in row.items()})
    if any(s["namespace"] == "task_item" for s in sets):
        result["task_registry"] = registry
    else:
        result.pop("task_registry", None)
    return result


def static_visual_payload(payload: dict) -> bool:
    return bool(payload.get("catalog")) and all(
        item["kind"] == "frame" for item in payload["catalog"].values()
    ) and all(s["namespace"] != "task_item" and s.get("predicate_kind", "static") == "static"
              for s in payload["spec"]["sets"])


def model_observation_payload(payload: dict) -> dict:
    """A display projection only. Durable payload/catalog retain exact source metadata."""
    static = static_visual_payload(payload)
    result = {k: v for k, v in payload.items() if k in {
        "set_ids", "audit", "member_limit", "task_registry", "attempt_phase", "previous_call_id", "validation_errors"}}
    fields = {"set_id", "namespace", "target", "predicate", "predicate_kind", "required_modalities",
              "evidence_relation", "normalization", "population", "candidates"}
    if not static:
        fields |= {"owner", "task_id", "task_projection", "scope"}
        result.update(core=payload["core"], context=payload["context"])
    result["spec"] = {"sets": [{k: v for k, v in target.items() if k in fields and v not in (None, [], {})}
                               for target in payload["spec"]["sets"]]}
    result["catalog"] = {}
    for alias, item in payload["catalog"].items():
        keys = {"region", "membership_sets"}
        if not static:
            keys |= {"kind", "start_sec", "end_sec", "text", "speaker_id", "history_start",
                     "alignment_status", "alignment_error_sec"}
        result["catalog"][alias] = {k: v for k, v in item.items() if k in keys and v is not None}
        if static and "membership_sets" in result["catalog"][alias]:
            result["catalog"][alias]["sets"] = result["catalog"][alias].pop("membership_sets")
    return result


def observation_parts(prepared: Any, aliases: dict, payload: dict | None = None) -> list:
    if not prepared:
        return []
    inverse = {ref: alias for alias, ref in aliases.items()}
    static = payload is not None and static_visual_payload(payload)
    def label(frame):
        alias = inverse[frame.id]
        region = payload["catalog"][alias]["region"] if payload else ""
        if static:
            return f"{alias} · {region}"
        return f"Frame {alias} at {frame.timestamp_seconds:.6f}s" + (f" · {region}" if region else "")
    headers = {f"Frame {f.id} at {f.timestamp_seconds:.6f}s": label(f) for f in prepared.frames}
    parts = [dict(part) for part in prepared.parts]
    for part in parts:
        if part.get("type") == "text" and part.get("text") in headers:
            part["text"] = headers[part["text"]]
    if prepared.kind == "ordered_video":
        parts.insert(0, {"type": "text", "text": "Supplied frames in order: " +
                        ", ".join(label(f) for f in prepared.frames)})
    return parts


def decode_observation(data: dict, spec: Any, catalog: dict, aliases: dict,
                       tile: dict, source: dict, limit: int) -> dict:
    selected = {"sets": [s for s in spec.to_dict()["sets"] if s["set_id"] in tile["set_ids"]]}
    data = validate_observation_shape(data, selected, limit)
    normalizations = []

    def fail(path, code, message, expected, actual):
        raise ContractError([issue(path, code, message, expected=expected, actual=actual)])

    def reference(value, path):
        if value not in aliases:
            fail(path, "unknown_evidence_alias", "Reference was not supplied in this call",
                 "a supplied F/T alias", value)
        return aliases[value]

    def restore(value, path=""):
        if isinstance(value, dict):
            for key, item in value.items():
                p = f"{path}.{key}" if path else key
                if key == "ref":
                    value[key] = reference(item, p)
                elif key in {"evidence_refs", "refs", "witness_refs"}:
                    value[key] = [reference(r, f"{p}[{i}]") for i, r in enumerate(item)]
                else:
                    restore(item, p)
        elif isinstance(value, list):
            for i, item in enumerate(value):
                restore(item, f"{path}[{i}]")

    restore(data)
    targets = {s.set_id: s for s in spec.sets}
    ids = set()
    for i, row in enumerate(data["observations"]):
        path = f"observations[{i}]"
        if row["local_id"] in ids:
            fail(path + ".local_id", "duplicate_observation_id", "Local IDs must be unique",
                 "unique local ID", row["local_id"])
        ids.add(row["local_id"])
        if targets[row["set_id"]].namespace == "physical_instance":
            derived = list(dict.fromkeys(d["ref"] for d in row["detections"]))
            explicit = row.get("evidence_refs", [])
            if explicit and set(explicit) != set(derived):
                fail(path + ".evidence_refs", "conflicting_entity_evidence",
                     "Entity evidence must agree with its source detections", derived, explicit)
            row["evidence_refs"] = derived
            normalizations.append({"path": path + ".evidence_refs", "kind": "derived_from_detections",
                                   "evidence_refs": derived})
        row.setdefault("category", row["value"])
        row.setdefault("evidence_relation", targets[row["set_id"]].evidence_relation)
        for j, detection in enumerate(row.get("detections", [])):
            p = f"{path}.detections[{j}]"
            if detection["ref"] not in row["evidence_refs"] or catalog[detection["ref"]]["kind"] != "frame":
                fail(p + ".ref", "invalid_detection_reference", "Detection must cite this member's frame",
                     row["evidence_refs"], detection["ref"])
            a, b, c, d = detection["bbox"]
            if not (a < c and b < d):
                fail(p + ".bbox", "invalid_bbox", "Box must have positive width and height",
                     "x1 < x2 and y1 < y2", detection["bbox"])
    for collection in ("observations", "task_updates"):
        if len(data.get(collection, [])) > limit:
            fail(collection, "member_limit_exceeded", "Report overflow and unresolved regions",
                 f"at most {limit} records", len(data[collection]))
    if data.get("truncated") and not data.get("unresolved_regions"):
        fail("unresolved_regions", "overflow_region_required", "Explicit overflow needs a coverage gap",
             "nonempty array", [])
    result = parse_batch(data, spec, catalog, tile, source, limit)
    gaps = []
    for row in result["observations"]:
        if row.population_status != "excluded" and row.predicate_status == "unknown":
            gaps.append({"set_id": row.set_id, "local_id": row.local_id,
                         "interval": list(tile["core"]),
                         "reason": ",".join(row.issues) or "membership_unproven"})
    for row in result["updates"]:
        if not row["binding_supported"]:
            gaps.append({"set_id": row["set_id"], "interval": list(tile["core"]),
                         "reason": "task_binding_unproven"})
    result["unresolved"] = [*result["unresolved"], *gaps]
    result["normalizations"] = normalizations
    return result
