"""Small v5 wire contracts. Mechanical provenance never comes from the model."""
from __future__ import annotations

import copy
import json
from typing import Any

from jsonschema import Draft202012Validator

from .contracts import ContractError, issue, response_schema, value_type
from .types import InventorySpec, UPDATE_KINDS

STR = {"type": "string", "minLength": 1}
STRS = {"type": "array", "items": STR}
REFS = {"type": "array", "items": {"type": "string", "pattern": r"^[FT][1-9][0-9]*$"}, "uniqueItems": True}
TRI = {"enum": ["yes", "no", "unknown"]}
UNIT_EQUIVALENCE = {"physical_instance": "entity", "semantic_category": "category",
                    "text_value": "literal", "task_item": "task"}


def obj(properties, required=()):
    return {"type": "object", "properties": properties, "required": list(required), "additionalProperties": False}


def array(item, limit=None):
    return {"type": "array", "items": item, **({"maxItems": limit} if limit is not None else {})}


def validate(data, schema, path="$"):
    errors = []
    for err in sorted(Draft202012Validator(schema).iter_errors(data), key=lambda e: str(list(e.path))):
        p = path
        for part in err.absolute_path:
            p += f"[{part}]" if isinstance(part, int) else "." + part
        # Keep the raw diagnostic, but model feedback is generated separately.
        errors.append({**issue(p, "schema_" + err.validator, err.message,
                            expected=err.validator_value, actual=value_type(err.instance)),
                       "actual_value": _brief(err.instance)})
    if errors:
        raise ContractError(errors)
    return copy.deepcopy(data)


def conditions(target):
    return {"target": "is a " + target.target, "predicate": target.predicate, **target.membership}


def compile_schema():
    schema = response_schema("compile")
    member = schema["properties"]["sets"]["items"]
    member["required"] += ["count_unit", "predicate"]
    member["properties"]["count_unit"] = STR
    schema["properties"]["version"] = {"const": 5, "default": 5}
    return schema


def _brief(value):
    if isinstance(value, dict):
        return {"type": "object", "keys": list(value)[:12]}
    if isinstance(value, list):
        return {"type": "array", "length": len(value)}
    return value[:160] if isinstance(value, str) else value


def compile_findings(data, path="$", *, inspect_wrapper=True):
    """Read-only diagnosis. Invalid containers never become an accepted task."""
    errors = []
    for err in sorted(Draft202012Validator(compile_schema()).iter_errors(data), key=lambda e: str(list(e.path))):
        field = path + "".join(f"[{p}]" if isinstance(p, int) else "." + p for p in err.absolute_path)
        row = issue(field, "schema_" + err.validator, err.message,
                    expected=err.validator_value, actual=value_type(err.instance))
        row["actual_value"] = _brief(err.instance)
        if err.validator == "required" and isinstance(err.instance, dict):
            row["missing_fields"] = [k for k in err.validator_value if k not in err.instance]
        if err.validator == "additionalProperties" and isinstance(err.instance, dict):
            allowed = list(err.schema.get("properties", {}))
            row.update(unexpected_fields=sorted(set(err.instance) - set(allowed)), allowed_fields=allowed)
        errors.append(row)
    if not isinstance(data, dict):
        return errors
    if inspect_wrapper and isinstance(data.get("r4"), dict) and isinstance(data["r4"].get("compile"), dict):
        errors.insert(0, issue(path + ".r4.compile", "task_wrapper", "Task fields are inside an unsupported wrapper",
                              expected="sets and operations at root", actual="r4.compile object"))
        errors.extend(compile_findings(data["r4"]["compile"], path + ".r4.compile", inspect_wrapper=False))
    # These checks are safe even when another field has a structural error.
    for i, member in enumerate(data.get("sets", []) if isinstance(data.get("sets"), list) else []):
        if not isinstance(member, dict):
            continue
        ns, eq = member.get("namespace"), member.get("equivalence", "auto")
        expected = UNIT_EQUIVALENCE.get(ns) if isinstance(ns, str) else None
        if expected and isinstance(eq, str) and eq not in {"auto", expected, "combination"}:
            errors.append(issue(f"{path}.sets[{i}].equivalence", "unit_equivalence_conflict",
                f"{ns} uses {expected}; select the namespace from the question, then omit ordinary equivalence",
                expected=expected, actual=eq))
        if eq == "combination" and not member.get("attribute_keys"):
            errors.append(issue(f"{path}.sets[{i}].attribute_keys", "missing_combination_keys",
                                "A combination needs explicit attribute dimensions", expected="nonempty string array", actual="missing/empty"))
    return errors


def compile_diagnostics(errors):
    """Keep the subject of each compiler error; never deduplicate by path alone."""
    result, seen = [], set()
    for err in errors:
        path, code = err.get("path", "$"), err.get("code", "compile_error")
        fields = err.get("unexpected_fields") or err.get("missing_fields") or []
        key = (path, code, tuple(fields), str(err.get("actual")))
        if key in seen:
            continue
        seen.add(key)
        instruction = "Regenerate the complete task from the original question. "
        if code == "task_wrapper":
            instruction += "Remove the r4.compile wrapper; put sets, operations and task fields directly at the JSON root."
        elif code == "duplicate_json_field":
            instruction += f"Keep exactly one {err.get('field_name')!r} field at this path; do not repeat JSON keys."
        elif code == "schema_additionalProperties":
            instruction += f"Remove unsupported fields {fields!r}. Legal fields here: {err.get('allowed_fields', [])!r}."
            if "optional_membership" in fields:
                instruction += " The optional condition object is named membership; omit it if no extra conditions are needed."
            if "result_kind" in fields:
                instruction += " result_kind belongs inside scope, only when a semantic range needs it."
            if "r4" in fields:
                instruction += " Remove r4.compile and move its task to the root."
            if "query_scope" in fields or "execution_subtype" in fields:
                instruction += " query_scope/execution_subtype are input metadata, not output fields."
        elif code == "schema_required":
            instruction += f"Supply {fields!r} directly at {path}, following the valid task examples."
        elif code == "unit_equivalence_conflict":
            instruction += err["message"] + "."
        elif code == "schema_type":
            instruction += f"Use type {err.get('expected')!r}, not {err.get('actual')!r}; do not replace arrays/objects with null."
        elif code == "schema_enum":
            instruction += f"Use one of {err.get('expected')!r}, not {err.get('actual_value', err.get('actual'))!r}."
        elif code == "response_truncated":
            instruction += "Return shorter complete JSON: required task fields and necessary exceptions only; omit host defaults."
        else:
            instruction += str(err.get("message", "Follow the task contract."))[:320]
        if path.endswith(".scope") and code in {"schema_type", "schema_additionalProperties"}:
            instruction += ' scope is an object using kind, e.g. {"kind":"full"}; never "full" or {"type":"full"}.'
        row = {k: err[k] for k in ("call_id", "stage", "path", "code", "expected", "actual", "actual_value",
                                  "missing_fields", "unexpected_fields", "field_name") if k in err}
        row["instruction"] = instruction
        result.append(row)
        if len(result) == 8:
            break
    return result


def parse_compile_json(text):
    """Reject duplicates with paths; any decoded projection is diagnostic-only."""
    value = text.strip()
    if value.startswith("```"):
        lines = value.splitlines()
        if len(lines) < 3 or lines[-1].strip() != "```":
            raise ContractError([issue("$", "invalid_json", "Incomplete code fence", expected="complete JSON object", actual="unparseable")])
        value = "\n".join(lines[1:-1])
    class Pairs(list):
        pass
    try:
        tree = json.loads(value, object_pairs_hook=Pairs,
                          parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))
    except (ValueError, TypeError) as exc:
        raise ContractError([issue("$", "invalid_json", str(exc), expected="complete JSON object", actual="unparseable")]) from exc
    errors = []
    def visit(node, path):
        if isinstance(node, Pairs):
            result = {}
            for k, v in node:
                field = path + "." + k
                child = visit(v, field)
                if k in result:
                    errors.append({**issue(field, "duplicate_json_field", "duplicate JSON field: " + k,
                        expected="one occurrence", actual="duplicate"), "field_name": k, "actual_value": _brief(child)})
                else:
                    result[k] = child
            return result
        if isinstance(node, list):
            return [visit(v, f"{path}[{i}]") for i, v in enumerate(node)]
        return node
    data = visit(tree, "$")
    if errors:
        # No projection with duplicate fields can escape this function as accepted data.
        raise ContractError(errors + compile_findings(data))
    return data


def _complete_compile(data, assignments):
    schema = compile_schema()
    def assign(obj, key, value, path, reason):
        old = obj.get(key)
        assignments.append({"path": path + "." + key, "was_present": key in obj,
                            "previous": copy.deepcopy(old), "value": copy.deepcopy(value), "reason": reason})
        obj[key] = copy.deepcopy(value)
    for key, field in schema["properties"].items():
        if key not in data and "default" in field:
            if key == "output_id":
                assign(data, key, data["operations"][-1]["operation_id"], "$", "last_operation")
            else:
                assign(data, key, field["default"], "$", "contract_default")
    for i, member in enumerate(data["sets"]):
        path = f"$.sets[{i}]"
        for key, field in schema["properties"]["sets"]["items"]["properties"].items():
            if key != "equivalence" and key not in member and "default" in field:
                assign(member, key, field["default"], path, "inherit_query_scope" if key == "scope" else "contract_default")
        if member.get("equivalence", "auto") == "auto":
            assign(member, "equivalence", UNIT_EQUIVALENCE[member["namespace"]], path, "namespace_equivalence")
    for i, operation in enumerate(data["operations"]):
        for key, field in schema["properties"]["operations"]["items"]["properties"].items():
            if key not in operation and "default" in field:
                assign(operation, key, field["default"], f"$.operations[{i}]", "contract_default")
    return data


def box_schema():
    return obj({"ref": REFS["items"], "xyxy": {"type": "array", "items": {
        "type": "integer", "minimum": 0, "maximum": 1000}, "minItems": 4, "maxItems": 4}}, ["ref", "xyxy"])


def record_schema(target, *, inspection=False):
    props = {"id": {"type": "string", "pattern": r"^O[1-9][0-9]*$"}, "set": {"const": target.set_id},
             "name": STR, "class": STR, "facts": STR,
             "conditions": obj({k: TRI for k in conditions(target)}, conditions(target)),
             "refs": REFS, "boxes": array(box_schema(), 24 if target.predicate_kind == "static" else 3),
             "query_value": {"type": ["string", "null"]}, "mapping_evidence": STR,
             "attributes": {"type": "object", "additionalProperties": {"type": ["string", "number", "boolean", "null"]}},
             "visibility": {"enum": ["clear", "occluded", "unreadable", "unknown"]},
             "uncertainties": STRS, "raw_text": {"type": ["string", "null"]},
             "motion": obj({"refs": REFS, "witness_refs": REFS, "object_motion": {"type": "boolean"},
                            "camera_motion_accounted": {"type": "boolean"}, "boundary_crossing": {"type": "boolean"},
                            "identity_continuity": {"type": "boolean"}},
                           ["refs", "witness_refs", "object_motion", "camera_motion_accounted"])}
    required = ["id", "set", "name", "class", "facts", "conditions"]
    if target.namespace == "physical_instance":
        required.append("boxes")
        props["boxes"]["minItems"] = 1
    else:
        required.append("refs")
        props["refs"] = {**REFS, "minItems": 1}
    if target.predicate_kind in {"moving", "enters", "exits"}:
        required.append("motion")
    if inspection:
        props.pop("id")
        required.remove("id")
        props["candidate_id"] = STR
        required.append("candidate_id")
    return obj(props, required)


def check_schema(target):
    schema = obj({"set": {"const": target.set_id}, "candidate": {"enum": list(target.candidates)},
                "state": {"enum": ["seen", "not_seen", "unreadable"]}, "refs": REFS,
                "support": {"enum": ["direct", "related", "uncertain"]}, "uncertainties": STRS,
                "facts": STR, "raw_text": {"type": ["string", "null"]}},
               ["set", "candidate", "state", "refs", "facts"])
    schema["allOf"] = [{"if": {"properties": {"state": {"const": "seen"}}, "required": ["state"]},
                        "then": {"required": ["support"]}}]
    return schema


def task_schema(target):
    props = {"local_id": STR, "set_id": {"const": target.set_id}, "kind": {"enum": sorted(UPDATE_KINDS)},
             "item_key": STR, "evidence_refs": {**REFS, "minItems": 1}, "owner": {"type": ["string", "null"]},
             "task_id": {"type": ["string", "null"]}, "quantity": {"type": ["number", "null"], "minimum": 0},
             "unit": STR, "binding_supported": {"type": "boolean"}, "raw_text": {"type": ["string", "null"]},
             "refers_to": {"type": ["string", "null"]}, "effective_time": {"type": ["number", "string", "null"]},
             "replacement_item": {"type": ["string", "null"]},
             "replacement_quantity": {"type": ["number", "null"], "minimum": 0},
             "completion_predicate": {"type": ["string", "null"]}}
    return obj(props, ["local_id", "set_id", "kind", "item_key", "evidence_refs", "binding_supported"])


RELATION_SCHEMA = obj({"left": STR, "right": STR, "relation": {"enum": ["SAME", "DIFFERENT", "UNKNOWN"]},
    "basis": {"enum": ["shared_observation", "continuous_track", "reidentification", "coexistence",
                       "distinct_tracks", "stable_difference", "uncertain"]},
    "refs": REFS, "facts": STR, "features": STRS, "independent_objects": {"type": "boolean"},
    "continuous_identity": {"type": "boolean"}, "supersedes": STRS},
    ["left", "right", "relation", "basis", "refs", "facts"])

DISTINCT_SCHEMA = obj({"left": STR, "right": STR, "refs": REFS, "independent_objects": {"const": True}},
                      ["left", "right", "refs", "independent_objects"])
COEXISTING_SCHEMA = obj({"ids": {**STRS, "minItems": 2, "maxItems": 12, "uniqueItems": True},
                        "ref": REFS["items"], "independent_objects": {"const": True}},
                       ["ids", "ref", "independent_objects"])


def snapshot_schema(target):
    groups = {"type":"object","maxProperties":12,"additionalProperties":{**STRS,"maxItems":12,"uniqueItems":True}}
    return obj({"set":{"const":target.set_id},"frames":{"type":"object","maxProperties":24,
        "propertyNames":{"pattern":r"^F[1-9][0-9]*$"},"additionalProperties":groups}},["set","frames"])


def envelope_schema(role, *, candidate_only=False):
    # Row schemas are checked independently, to quarantine invalid rows without losing valid ones.
    props = {"coverage": {"enum": ["complete", "partial", "unreadable"]}, "overflow": {"type": "boolean"},
             "gaps": STRS}
    if role in {"discover_candidates", "inspect_existing"}:
        props.update(records={"type": "array", "maxItems": 12}, checks={"type": "array", "maxItems": 12},
                     input_gaps={"type":"array","maxItems":12,"items":obj({
                         "set":STR,"candidate":STR,"reason":{"enum":["unreadable","occluded","uninspected"]}},
                         ["set","candidate","reason"])},
                     snapshots={"type":"array","maxItems":12},
                     task_updates={"type": "array", "maxItems": 12},
                     distinct_pairs={"type":"array", "maxItems":66})
        props["coexisting"] = {"type":"array", "maxItems":12}
        if role == "inspect_existing":
            props.update(updates={"type": "array", "maxItems": 12}, new_candidates={"type": "array", "maxItems": 12})
        return obj(props, ["checks"] if candidate_only else ["coverage"])
    if role == "identity":
        return obj({"relations": array(RELATION_SCHEMA, 9)}, ["relations"])
    if role == "scope":
        binding = obj({"scope_id": STR, "refs": {**REFS, "minItems": 1},
                       "interval": {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2},
                       "facts": STR}, ["scope_id", "refs", "interval", "facts"])
        return obj({"bindings": array(binding, 12), "coverage": props["coverage"], "gaps": STRS}, ["bindings", "coverage"])
    raise ValueError("unknown v5 role: " + role)


def diagnostics(errors, raw=None):
    """Bounded model-facing feedback; never echo a schema's serialized bad object."""
    seen, result = set(), []
    for err in sorted(errors, key=lambda e: (-len(e.get("path", "")), e.get("path", ""))):
        path = err.get("path", "$")
        key = (path, err.get("code"), str(err.get("expected")))
        if key in seen:
            continue
        seen.add(key)
        expected = err.get("expected")
        actual = err.get("actual")
        if isinstance(expected, dict):
            expected = {"keys": list(expected)[:6]}
        if isinstance(actual, dict):
            actual = {"type": "object", "keys": list(actual)[:6]}
        if isinstance(actual, list) and len(actual) > 4:
            actual = {"type": "array", "length": len(actual)}
        instruction = "Correct this field using the supplied media; preserve the record and its array position."
        if ".conditions." in path:
            instruction = "Output a judgment: yes, no, or unknown. Do NOT copy the text from input requirements. If uncertain retain the object with unknown."
        elif path.endswith(".boxes") and err.get("code") == "schema_maxItems":
            instruction = "Keep this object; select only 1–3 representative boxes from the media. Do not enumerate frames or erase the object."
        elif path.startswith("checks") and expected == []:
            instruction = "This set has no requested candidate checks. Replace this invalid checks slot with null during recovery; do not invent a candidate."
        elif err.get("code") == "unrequested_check":
            instruction = "This check is outside the requested candidate list. In recovery replace only this checks slot with null."
        elif err.get("code") == "recovery_slot_missing":
            instruction = "Do not return empty arrays to erase invalid records. Return all original positions and repair each reported record; use unknown when unsure."
        elif err.get("code") == "ambiguous_candidate_coverage":
            instruction = "Do not use global coverage/gaps to mean the action is absent. Return each requested check: not_seen when checked but absent; unreadable when unjudgeable. Only input_gaps with set,candidate,reason:unreadable/occluded/uninspected represent actual input gaps."
        elif err.get("code") == "candidate_gap_conflict":
            instruction = "This check conflicts with an explicit input gap. Use unreadable for an unjudgeable input, or remove the input gap only if the candidate was actually checked."
        elif err.get("code") == "identity_reference_is_object_id":
            instruction = "left/right are object IDs C...; refs are evidence IDs F.../T... from evidence_by_object. Do not copy an object ID into refs. If identity evidence is insufficient return UNKNOWN with basis:uncertain."
        elif err.get("code") in {"independence_unproven", "distinction_unproven", "insufficient_identity_basis"}:
            instruction = "DIFFERENT requires independent_objects:true supported by evidence, and distinct_tracks/stable_difference require nonempty features describing that evidence. Different scenes, contents or contexts alone are not independent tracks. Do not add true merely to pass validation: if the supplied evidence cannot establish independence, return UNKNOWN, basis:uncertain, refs:[], and explain the uncertainty."
        elif err.get("code") == "comparison_pair_outside_shortlist":
            instruction = "Return only one left-list ID paired with one right-list ID. Do not compare two right-list objects."
        elif path.endswith(".support") or "support" in err.get("message", ""):
            instruction = "state and support have different meanings. For state:not_seen, OMIT support and retain the check, its candidate, refs and nonempty inspection facts. For state:seen, support is required: direct/related/uncertain. For unreadable, support may be omitted; never put not_seen in support. Do not change state to seen merely to fix support."
        result.append({"path": path, "code": err.get("code"), "expected": expected,
                       "actual": actual, **({"actual_value":err["actual_value"]} if "actual_value" in err else {}),
                       "instruction": instruction})
        if len(result) == 8:
            break
    return result


def parse_json(text):
    value = text.strip()
    if value.startswith("```"):
        lines = value.splitlines()
        if len(lines) < 3 or lines[-1].strip() != "```":
            raise ContractError([issue("$", "invalid_json", "Incomplete code fence")])
        value = "\n".join(lines[1:-1])
    def unique(pairs):
        result = {}
        for k, v in pairs:
            if k in result:
                raise ValueError("duplicate JSON field: " + k)
            result[k] = v
        return result
    try:
        return json.loads(value, object_pairs_hook=unique, parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))
    except (ValueError, TypeError) as exc:
        raise ContractError([issue("$", "invalid_json", str(exc), expected="complete JSON object", actual="unparseable")]) from exc


def parse_compile(data, request, *, assignments=None):
    errors = compile_findings(data)
    if errors:
        raise ContractError(errors)
    data = _complete_compile(copy.deepcopy(data), assignments if assignments is not None else [])
    spec = InventorySpec.from_dict(data)
    errors = []
    for i, s in enumerate(spec.sets):
        policy = request.benchmark_policy
        if policy.get("count_unit") and s.count_unit != policy["count_unit"] and s.namespace != policy["count_unit"]:
            errors.append(issue(f"sets[{i}].count_unit", "public_unit_conflict", "Respect the public count unit", expected=policy["count_unit"], actual=s.count_unit))
        if s.normalization and s.normalization != policy.get("normalization", {}):
            errors.append(issue(f"sets[{i}].normalization", "nonpublic_normalization", "Semantic mappings belong to evidence cards, not public string policy"))
        expected = UNIT_EQUIVALENCE[s.namespace]
        if s.equivalence not in {expected, "combination"}:
            errors.append(issue(f"sets[{i}].equivalence", "unit_equivalence_conflict", "Equivalence must match the member unit", expected=expected, actual=s.equivalence))
        if s.equivalence == "combination" and not s.attribute_keys:
            errors.append(issue(f"sets[{i}].attribute_keys", "missing_combination_keys", "An attribute combination needs explicit dimensions"))
    labels = {c.label for c in request.choices}
    if set(spec.choice_values) - labels:
        errors.append(issue("choice_values", "unknown_choice_label", "Only original labels may be mapped"))
    for label, value in spec.choice_values.items():
        if isinstance(value, dict) and "kind" in value:
            try:
                validate(value, choice_value_schema(), "choice_values." + label)
                if value["kind"] == "interval" and value.get("high") is not None and value["low"] > value["high"]:
                    raise ContractError([issue("choice_values." + label, "invalid_interval", "low must not exceed high")])
            except ContractError as exc:
                errors.extend(exc.errors)
    if request.execution_subtype and spec.operations[-1].op != request.execution_subtype:
        from .types import OP_ALIASES
        if spec.operations[-1].op != OP_ALIASES.get(request.execution_subtype, request.execution_subtype):
            errors.append(issue("operations", "operation_hint_conflict", "Respect the requested operation"))
    if errors:
        raise ContractError(errors)
    return spec


def choice_value_schema():
    scalar = {"type":["number", "string", "boolean"]}
    return {"oneOf":[
        obj({"kind":{"const":"interval"},"low":{"type":"number"},"high":{"type":["number","null"]}},["kind","low","high"]),
        obj({"kind":{"const":"one_of"},"values":{**array(scalar),"minItems":1}},["kind","values"]),
        obj({"kind":{"enum":["set","ordered_counts"]},"value":array(scalar)},["kind","value"])]}
