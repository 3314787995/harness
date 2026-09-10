"""Shared model wire contracts, defaults, and actionable compile validation."""

from __future__ import annotations

import copy
import json
import re
from dataclasses import MISSING, fields
from functools import lru_cache
from typing import Any

from jsonschema import Draft202012Validator

from qwen3vl_agent.r4.types import (
    MODALITIES,
    NAMESPACES,
    OPERATIONS,
    PREDICATE_KINDS,
    RELATIONS,
    TASK_PROJECTIONS,
    InventorySpec,
    ProtocolError,
    SetOperation,
    SetSpec,
    UPDATE_KINDS,
    OP_ALIASES,
)


class ContractError(ProtocolError):
    def __init__(self, errors: list[dict[str, Any]]) -> None:
        self.errors = errors
        super().__init__("; ".join(f"{e['path']}: {e['message']}" for e in errors))


def issue(path: str, code: str, message: str, *, expected=None, actual=None) -> dict:
    return {"path": path, "code": code, "message": message, "expected": expected, "actual": actual}


def json_value(value: Any) -> Any:
    # Internal dataclass snapshots contain tuples. No coercion of null/bool/string values.
    if isinstance(value, dict):
        return {k: json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(v) for v in value]
    return value


def value_type(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, dict):
        return "object"
    if isinstance(value, (list, tuple)):
        return "array"
    if isinstance(value, str):
        return "string"
    return "number" if isinstance(value, (int, float)) else type(value).__name__


def _object(properties, required=()):
    return {
        "type": "object",
        "properties": properties,
        "required": list(required),
        "additionalProperties": False,
    }


def _defaults(cls, properties):
    result = copy.deepcopy(properties)
    for f in fields(cls):
        if f.name not in result:
            continue
        if f.default is not MISSING:
            result[f.name]["default"] = json_value(f.default)
        elif f.default_factory is not MISSING:
            result[f.name]["default"] = json_value(f.default_factory())
    return result


@lru_cache(maxsize=2)
def _schema(role: str) -> dict:
    string = {"type": "string", "minLength": 1}
    strings = {"type": "array", "items": string}
    if role == "candidate_union":
        return _object({"targets": strings}, ["targets"])
    if role != "compile":
        raise ValueError(f"No structural contract for {role}")
    scope = _object(
        {
            "kind": {"enum": ["full", "interval", "frame", "semantic", "history"]},
            "interval": {
                "type": "array",
                "items": {"type": "number"},
                "minItems": 2,
                "maxItems": 2,
            },
            "timestamp_sec": {"type": "number", "minimum": 0},
            "description": string,
            "entry_ids": strings,
            "selection": {"enum": ["all", "first", "last"]},
            "result_kind": {"enum": ["frame", "interval"]},
        }
    )
    scope["allOf"] = [
        {
            "if": {"properties": {"kind": {"const": kind}}, "required": ["kind"]},
            "then": {"required": [name]},
        }
        for kind, name in (
            ("interval", "interval"),
            ("frame", "timestamp_sec"),
            ("semantic", "description"),
        )
    ]
    normalization = _object(
        {
            "casefold": {"type": "boolean"},
            "strip_punctuation": {"type": "boolean"},
            "aliases": {"type": "object", "additionalProperties": string},
            "policy_id": string,
        }
    )
    members = _object(
        _defaults(
            SetSpec,
            {
                "set_id": string,
                "namespace": {"enum": sorted(NAMESPACES)},
                "target": string,
                "predicate": string,
                "predicate_kind": {"enum": sorted(PREDICATE_KINDS)},
                "evidence_relation": {"enum": sorted(RELATIONS)},
                "required_modalities": {
                    "type": "array",
                    "minItems": 1,
                    "items": {"enum": sorted(MODALITIES)},
                },
                "candidates": strings,
                "scope": scope,
                "normalization": normalization,
                "population": string,
                "owner": {"type": ["string", "null"]},
                "task_id": {"type": ["string", "null"]},
                "task_projection": {"enum": sorted(TASK_PROJECTIONS)},
                "count_unit": {"type": "string"},
                "membership": {"type": "object", "additionalProperties": string},
                "equivalence": {"enum": ["auto", "entity", "category", "literal", "combination", "task"]},
                "attribute_keys": strings,
            },
        ),
        ["set_id", "namespace", "target"],
    )
    operation = _object(
        _defaults(
            SetOperation,
            {
                "operation_id": string,
                "op": {"enum": sorted(OPERATIONS | set(OP_ALIASES))},
                "inputs": {**strings, "minItems": 1},
                "candidates": strings,
                "group_by": string,
                "compare": {"enum": ["greater", "less", "equal"]},
            },
        ),
        ["operation_id", "op", "inputs"],
    )
    return _object(
        _defaults(
            InventorySpec,
            {
                "sets": {"type": "array", "minItems": 1, "items": members},
                "operations": {"type": "array", "minItems": 1, "items": operation},
                "scope": scope,
                "unresolved": strings,
                "version": {"enum": [1, 5]},
                "choice_values": {"type": "object", "additionalProperties": {"type": ["number", "string", "array", "object", "boolean", "null"]}},
                "output_id": {"type": ["string", "null"]},
            },
        ),
        ["sets", "operations"],
    )


def response_schema(role: str, spec: dict | None = None) -> dict | None:
    if role == "observe" and spec is not None:
        return observation_schema(spec)
    return copy.deepcopy(_schema(role)) if role in {"compile", "candidate_union"} else None


@lru_cache(maxsize=2)
def _validator(role):
    return Draft202012Validator(_schema(role))


def validate_shape(role: str, value: Any) -> dict:
    data = json_value(value)
    errors = []
    for error in sorted(_validator(role).iter_errors(data), key=lambda e: str(list(e.path))):
        path = ""
        for part in error.absolute_path:
            path += f"[{part}]" if isinstance(part, int) else ("." if path else "") + part
        errors.append(
            issue(
                path or "$",
                "schema_" + error.validator,
                error.message,
                expected=error.validator_value,
                actual=value_type(error.instance),
            )
        )
    if errors:
        raise ContractError(errors)
    return data


def parse_candidates(data: dict) -> list[str]:
    data = validate_shape("candidate_union", data)
    return sorted(set(data["targets"]))


def validate_semantics(spec: InventorySpec) -> None:
    errors = []
    by_id = {s.set_id: (i, s) for i, s in enumerate(spec.sets)}
    for op in spec.operations:
        if op.inputs and op.inputs[0] in by_id:
            by_id[op.operation_id] = by_id[op.inputs[0]]
    for i, target in enumerate(spec.sets):
        if target.namespace == "task_item" and target.evidence_relation not in {
            "planned",
            "completed",
        }:
            errors.append(
                issue(
                    f"sets[{i}].namespace",
                    "task_lifecycle_required",
                    "task_item requires planned/completed text evidence; visually observed activities "
                    "belong to semantic_category, not a plan/completion lifecycle",
                    expected="planned or completed",
                    actual=target.evidence_relation,
                )
            )
    for op in spec.operations:
        if op.op != "missing_members":
            continue
        for key in op.inputs:
            if key not in by_id:
                continue  # The existing reference validator reports this separately.
            i, target = by_id[key]
            predicate = re.sub(r"[_\s-]+", " ", target.predicate.strip().casefold())
            if re.match(
                r"^(?:not (?:shown|present|visible|seen)|absent|missing)(?:$| in\b| from\b)",
                predicate,
            ) or re.match(r"^(?:未出现|没有出现|未展示|不在视频中)", predicate):
                errors.append(
                    issue(
                        f"sets[{i}].predicate",
                        "absence_in_observation_predicate",
                        "Collect observed members satisfying the local condition; missing_members "
                        "computes absence after coverage. Do not use global absence as that condition.",
                        expected="positive observation condition",
                        actual=target.predicate,
                    )
                )
    if errors:
        raise ContractError(errors)


def compile_examples() -> list[dict]:
    def example(question, name, namespace, op, candidates=()):
        output = {
            "sets": [{"set_id": "members", "namespace": namespace, "target": name}],
            "operations": [{"operation_id": "answer", "op": op, "inputs": ["members"]}],
            "scope": {"kind": "full"},
            "unresolved": [],
            "version": 1,
        }
        if candidates:
            output["sets"][0]["candidates"] = list(candidates)
            output["operations"][0]["candidates"] = list(candidates)
        return {"question": question, "candidate_union": list(candidates), "output": output}

    return [
        example(
            "How many distinct chairs appear in the video?",
            "chair",
            "physical_instance",
            "count_unique",
        ),
        example(
            "How many types of geometric shapes are displayed?",
            "geometric shape type",
            "semantic_category",
            "count_unique",
        ),
        example(
            "Which listed sport is not shown?",
            "observed sport",
            "semantic_category",
            "missing_members",
            ["cycling", "swimming", "tennis"],
        ),
    ]


def error_details(exc: Exception) -> list[dict]:
    return getattr(exc, "errors", None) or [issue("$", "invalid_response", str(exc))]


class ResponseFailure(ProtocolError):
    """A model response remained invalid after its one permitted correction."""

    def __init__(self, role: str, attempts: list[dict]) -> None:
        self.failure = {
            "stage": role,
            "code": "response_contract_failed",
            "message": f"{role} response invalid after bounded correction",
            "attempts": attempts,
            "call_ids": [a["call_id"] for a in attempts],
        }
        super().__init__(self.failure["message"])


def observation_schema(spec: dict, member_limit: int = 32) -> dict:
    """The same task-specific wire contract is shown to the observer and validated."""
    string = {"type": "string", "minLength": 1}
    nullable = {"type": ["string", "null"]}
    strings = {"type": "array", "items": string}
    evidence = {**strings, "minItems": 1}
    detection = _object({"ref": string, "bbox": {
        "type": "array", "minItems": 4, "maxItems": 4,
        "items": {"type": "integer", "minimum": 0, "maximum": 1000},
    }}, ["ref", "bbox"])
    attributes = {"type": "object", "additionalProperties": {"anyOf": [
        {"type": ["string", "number", "boolean", "null", "array"]},
        {"type": "object", "properties": {"evidence_refs": evidence},
         "required": ["evidence_refs"]},
    ]}}
    sets = spec["sets"]
    branches = []
    for target in sets:
        props = {
            "local_id": {**string, "pattern": r"^O[1-9][0-9]*$"}, "set_id": {"const": target["set_id"]},
            "value": string, "category": string, "raw_text": nullable,
            "candidate_values": strings, "attributes": attributes,
            "predicate_status": {"enum": ["satisfied", "refuted", "unknown"]},
            "evidence_relation": {"const": target.get("evidence_relation", "visually_present")},
            "visibility": {"enum": ["clear", "partial", "unreadable", "occluded"]},
            "population_status": {"enum": ["included", "excluded", "unknown"]},
        }
        required = ["local_id", "set_id", "value", "predicate_status", "visibility"]
        if target["namespace"] == "physical_instance":
            props["detections"] = {"type": "array", "items": detection, "minItems": 1}
            if target.get("predicate_kind", "static") == "static":
                props["detections"]["maxItems"] = 3
            # Compatibility with explicit references is strict; missing/empty refs alone
            # are harmless because source detections are the authoritative entity evidence.
            props["evidence_refs"] = strings
            required.append("detections")
        else:
            props["evidence_refs"] = evidence
            required.append("evidence_refs")
            props["detections"] = {"type": "array", "items": detection}
        if target.get("predicate_kind") in {"moving", "enters", "exits"}:
            flags = {name: {"type": ["boolean", "null"]} for name in (
                "object_motion", "camera_motion_accounted", "boundary_crossing", "identity_continuity")}
            props["predicate_evidence"] = _object({"refs": strings, "witness_refs": strings, **flags},
                                                  ["refs", "witness_refs", *flags])
            required.append("predicate_evidence")
        if target["namespace"] == "task_item":
            props.update(owner=nullable, task_id=nullable)
        branches.append({"if": {"properties": {"set_id": {"const": target["set_id"]}}},
                         "then": _object(props, required)})
    member = {"type": "object", "properties": {"set_id": {"enum": [s["set_id"] for s in sets]}},
              "required": ["set_id"], "allOf": branches}
    relation = _object({"left": string, "right": string, "relation": {"enum": ["same", "different", "unknown"]},
                        "evidence_refs": strings, "reason": {"type": "string"}, "supersedes": strings},
                       ["left", "right", "relation", "evidence_refs", "reason"])
    props = {
        "observations": {"type": "array", "items": member, "maxItems": member_limit},
        "observation_status": {"enum": ["valid", "unreadable", "partial"]},
        "truncated": {"type": "boolean", "default": False},
        "unresolved_regions": {"type": "array", "items": {"anyOf": [
            string, _object({"set_id": string, "reason": string, "evidence_refs": strings,
                            "interval": {"type": "array", "minItems": 2, "maxItems": 2,
                                         "items": {"type": "number"}}}, ["reason"])]}, "default": []},
        "local_identity_relations": {"type": "array", "items": relation, "default": []},
        "task_updates": {"type": "array", "maxItems": 0, "default": []},
    }
    if not any(s["namespace"] == "physical_instance" for s in sets):
        props["local_identity_relations"]["maxItems"] = 0
    tasks = [s["set_id"] for s in sets if s["namespace"] == "task_item"]
    if tasks:
        number = {"type": ["number", "null"], "minimum": 0}
        update = _object({
            "local_id": string, "set_id": {"enum": tasks}, "kind": {"enum": sorted(UPDATE_KINDS)},
            "owner": nullable, "task_id": nullable, "item_key": string, "quantity": number,
            "unit": string, "evidence_refs": evidence, "raw_text": nullable,
            "effective_time": {"type": ["string", "number", "null"]},
            "refers_to": nullable, "replacement_item": nullable, "replacement_quantity": number,
            "completion_predicate": nullable, "binding_supported": {"type": "boolean"},
        }, ["local_id", "set_id", "kind", "item_key", "evidence_refs", "binding_supported"])
        props["task_updates"] = {"type": "array", "items": update, "default": []}
    return _object(props, ["observations", "observation_status"])


def validate_observation_shape(data: dict, spec: dict, member_limit: int = 32) -> dict:
    errors = []
    for error in Draft202012Validator(observation_schema(spec, member_limit)).iter_errors(data):
        parts = list(error.absolute_path)
        if error.validator == "required":
            parts.append(next(k for k in error.validator_value if k not in error.instance))
        elif error.validator == "additionalProperties" and isinstance(error.instance, dict):
            extra = sorted(set(error.instance) - set(error.schema.get("properties", {})))
            if extra:
                parts.append(extra[0])
        path = "".join(f"[{p}]" if isinstance(p, int) else ("." if i else "") + p
                       for i, p in enumerate(parts)) or "$"
        errors.append(issue(path, "schema_" + error.validator, error.message,
                            expected=error.validator_value, actual=value_type(error.instance)))
    if errors:
        raise ContractError(errors)
    return copy.deepcopy(data)


class PromptContractError(ContractError):
    """An invalid program-generated example must never reach the observation model."""


def observation_examples(spec: dict | None = None, member_limit: int = 32) -> list[dict]:
    """Complete fictional responses. The explanation is never serialized as a wrapper."""
    spec = spec or {"sets": [{"set_id": "example_tools", "namespace": "physical_instance", "target": "wrench"}]}
    examples = []
    for target in spec["sets"]:
        sid, namespace = target["set_id"], target["namespace"]
        motion = target.get("predicate_kind") in {"moving", "enters", "exits"}
        visual = target.get("evidence_relation", "visually_present") in {"visually_present", "text_present"} or (
            "screen_text" in target.get("required_modalities", []))
        ref = "F2" if visual else "T1"
        row = {"local_id": "O1", "set_id": sid, "value": "wrench" if namespace == "physical_instance" else "triangle",
               "predicate_status": "satisfied", "visibility": "clear"}
        response = {"observations": [row], "observation_status": "valid"}
        if namespace == "task_item":
            kind = "complete_item" if target.get("evidence_relation") == "completed" else "create_plan"
            sentence = "Alex bought two envelopes." if kind == "complete_item" else "Alex plans to buy two envelopes."
            response.update(observations=[], task_updates=[{
                "local_id": "U1", "set_id": sid, "kind": kind, "item_key": "envelopes",
                "quantity": 2, "unit": "item", "owner": "Alex", "task_id": "errand",
                "evidence_refs": [ref], "raw_text": sentence, "binding_supported": True,
            }])
            meaning = "Fictional explicit task statement with supported owner/task binding."
        elif namespace == "physical_instance":
            row["detections"] = [{"ref": "F2", "bbox": [100, 200, 300, 700]},
                                 {"ref": "F4", "bbox": [130, 200, 330, 700]}]
            meaning = "One continuously visible wrench across two frames: ONE member record."
        else:
            row["evidence_refs"] = [ref]
            meaning = "One fictional geometric category with a supporting reference."
            if namespace == "text_value":
                row.update(value="EXIT", raw_text="EXIT")
                meaning = "One fictional text value, copied exactly from its evidence."
            elif not visual or target.get("evidence_relation") == "mentioned":
                row["raw_text"] = "The shape is a triangle."
        if motion:
            if namespace != "physical_instance":
                row["evidence_refs"] = ["F2", "F4"]
            row["predicate_evidence"] = {
                "refs": ["F2", "F4"], "witness_refs": ["F2", "F4"],
                "object_motion": True, "camera_motion_accounted": True,
                "boundary_crossing": target["predicate_kind"] in {"enters", "exits"}, "identity_continuity": True,
            }
            # Nonphysical motion-conditioned members need the same explicit visual witnesses.
            row.setdefault("detections", [{"ref": "F2", "bbox": [100, 200, 300, 700]},
                                          {"ref": "F4", "bbox": [130, 200, 330, 700]}])
            meaning += " Here the fictional motion condition is supported by ordered witnesses."
        examples.append({"meaning": meaning, "output": response})
        if namespace == "physical_instance" and member_limit >= 2:
            separate = copy.deepcopy(response)
            first = separate["observations"][0]
            if not motion:
                first["detections"] = first["detections"][:1]
            second = copy.deepcopy(first)
            second["local_id"] = "O2"
            for detection in second["detections"]:
                detection["bbox"] = [600, 200, 800, 700]
            separate["observations"].append(second)
            examples.append({"meaning": "Two similar wrenches simultaneously in F2: TWO separate member records.",
                             "output": separate})
    return examples


def _check_example_geometry(value: Any, path: str = "$") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = path + "." + key
            if key == "bbox" and not (child[0] < child[2] and child[1] < child[3]):
                raise ContractError([issue(child_path, "invalid_bbox", "Example box has no positive area")])
            _check_example_geometry(child, child_path)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _check_example_geometry(child, f"{path}[{index}]")


def observation_instructions(spec: dict, member_limit: int) -> str:
    try:
        schema = observation_schema(spec, member_limit)
        examples = observation_examples(spec, member_limit)
        if not examples:
            raise ValueError("Observation format has no applicable example")
        rules = [
            "RESPONSE FORMAT: observations is a JSON array of member objects, not a schema object.",
            f"At most {schema['properties']['observations']['maxItems']} members; omit unused optional fields.",
            "observation_status: " + ", ".join(schema["properties"]["observation_status"]["enum"]) + ".",
            "Optional: truncated (boolean; real member overflow only), unresolved_regions (array of reason strings).",
            "Empty observations is allowed only for no observed members; unknown/occluded needs unresolved_regions.",
        ]
        for target, branch in zip(spec["sets"], schema["properties"]["observations"]["items"]["allOf"]):
            member = branch["then"]
            fields = member["properties"]
            rule = f"Set {target['set_id']}: member fields " + ", ".join(member["required"]) + ". "
            rule += "predicate_status: " + ", ".join(fields["predicate_status"]["enum"]) + "; "
            rule += "visibility: " + ", ".join(fields["visibility"]["enum"]) + "."
            if target["namespace"] == "physical_instance":
                det = fields["detections"]
                rule += f" detections is an array, at least {det['minItems']}"
                if "maxItems" in det:
                    rule += f", at most {det['maxItems']}"
                rule += "; bbox is [left,top,right,bottom], four integers 0..1000, positive area."
            else:
                rule += " evidence_refs is a nonempty array of supplied aliases."
            if target["namespace"] == "task_item":
                update = schema["properties"]["task_updates"]["items"]
                rule += " Explicit lifecycle events go in task_updates (array), fields: " + ", ".join(update["required"]) + "."
            rules.append(rule)
        rules.append("The following COMPLETE JSON responses illustrate format ONLY. Replace fictional names, "
                     "references, coordinates and states with actual evidence; never copy them as video facts. "
                     "Return the response itself, with no schema or explanatory wrapper.")
        for example in examples:
            # Validate the exact serialized response that is subsequently put into the prompt.
            encoded = json.dumps(example["output"], ensure_ascii=False, allow_nan=False, separators=(",", ":"))
            value = json.loads(encoded)
            validate_observation_shape(value, spec, member_limit)
            _check_example_geometry(value)
            rules.append(example["meaning"] + "\n```json\n" + encoded + "\n```")
        return "\n".join(rules)
    except (ContractError, ValueError, TypeError, KeyError, IndexError) as exc:
        raise PromptContractError(error_details(exc)) from exc
