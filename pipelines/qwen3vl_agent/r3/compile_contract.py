"""Model-boundary contracts for B1/B2. No question-to-query semantic validator.

The declarations below drive both normalization/validation and prompt documentation.
Internal EventQuery constructors and checkpoint deserialization remain independent.
"""

from __future__ import annotations

import copy
import json
import math
from dataclasses import dataclass
from typing import Any, TypedDict

from qwen3vl_agent.r3.types import FACT_KINDS, UNITS, EventQuery, ProtocolError

CONTRACT_VERSION = "r3-compile-2.0"
WIRE_VERSION = 2
_MISSING = object()


class IntentSpec(TypedDict):
    version: int
    targets: list[dict[str, Any]]
    tasks: list[dict[str, Any]]
    scope: dict[str, Any]
    needs_candidate_union: bool
    unresolved: list[str]


class CompileContractError(ProtocolError):
    def __init__(self, errors: list[dict[str, Any]]) -> None:
        self.errors = errors
        super().__init__("; ".join(f"{e['path']}: {e['message']}" for e in errors))


@dataclass(frozen=True)
class Field:
    kind: str
    default: Any = _MISSING
    values: tuple[Any, ...] = ()
    minimum: float | None = None

    @property
    def required(self) -> bool:
        return self.default is _MISSING

    def describe(self) -> dict[str, Any]:
        result: dict[str, Any] = {"type": self.kind, "required": self.required}
        if not self.required:
            result["default"] = self.default
        if self.values:
            result["enum"] = list(self.values)
        if self.minimum is not None:
            result["minimum"] = self.minimum
        return result


COMMON_OPERATION = {
    "operation_id": Field("nonempty_string"),
    "op": Field("nonempty_string"),
    "target_ids": Field("nonempty_strings"),
    "basis": Field("string", "onset", ("onset", "offset")),
    "selection": Field("string", "all", (
        "all", "unique", "first", "last", "first_per_category", "last_per_category",
    )),
    "group_by": Field("string", "occurrence", ("occurrence", "category")),
}
PROJECT = Field("nonempty_string", "description")
K = Field("integer", minimum=1)
ANCHOR = {
    "anchor_target_id": Field("nonempty_string"),
    "anchor_selection": Field("string", "unique", ("unique", "first", "last")),
    "project": PROJECT,
}
OPERATION_FIELDS = {
    "count_occurrences": {},
    "localize_event": {"project": PROJECT},
    "first_occurrence": {"project": PROJECT},
    "last_occurrence": {"project": PROJECT},
    "first_k": {"k": K, "project": PROJECT},
    "last_k": {"k": K, "project": PROJECT},
    "nth_occurrence": {"k": K, "project": PROJECT},
    "order_events": {"project": PROJECT},
    "next_after_anchor": ANCHOR,
    "previous_before_anchor": ANCHOR,
    "event_duration": {
        "duration_aggregation": Field("string", "single", ("single", "union", "sum", "compare")),
    },
    "cooccurrence_frequency": {"cooccurrence_targets": Field("strings", [])},
}
DURATION_COMPARISON = Field("string", values=("longest", "shortest"))
INTENT_TARGET = {
    "target_id": Field("nonempty_string"),
    "description": Field("nonempty_string"),
    "unit_kind": Field("string", values=tuple(sorted(UNITS))),
}
EVENT_TARGET = {
    **INTENT_TARGET,
    **{key: Field("string", "") for key in (
        "actor_constraint", "object_constraint", "start_criterion", "completion_criterion",
        "reset_criterion", "binding_description",
    )},
    "inclusion_rule": Field("string", "intersects", ("starts_inside", "completes_inside", "intersects")),
    "repeat_policy": Field("string", "presentation", ("presentation", "world")),
    "fact_kind": Field("string", "visual_event", tuple(sorted(FACT_KINDS))),
    "required_modalities": Field("strings", [], ("video", "screen_text", "subtitle", "asr")),
    "requires_actor_binding": Field("boolean", False),
}
SCOPE_FIELDS = {
    "full": {},
    "interval": {"interval": Field("interval")},
    "semantic": {
        "description": Field("nonempty_string"),
        "relative_first_sec": Field("positive_number", None),
        "relative_last_sec": Field("positive_number", None),
        "ordinal": Field("integer", None, minimum=1),
    },
}
_ALL_OPERATION = set(COMMON_OPERATION) | {"duration_comparison"}
for _fields in OPERATION_FIELDS.values():
    _ALL_OPERATION.update(_fields)
_ALL_SCOPE = {"kind"} | {key for fields in SCOPE_FIELDS.values() for key in fields}


def output_contract(stage: str) -> dict[str, Any]:
    if stage not in {"compile_intent", "compile"}:
        raise ValueError("not a B compilation stage")
    intent = stage == "compile_intent"
    return {
        "contract_version": CONTRACT_VERSION,
        "stage": stage,
        "output": {
            "version": WIRE_VERSION,
            "targets": "1..32 target objects; unique target_id",
            "tasks" if intent else "operations": "1..12 operation objects; unique operation_id",
            "scope": "one scope object",
            "unresolved": "array of strings (default [])",
            **({"needs_candidate_union": "boolean; normally false"} if intent else {}),
        },
        "candidate_request_only": (
            {"version": WIRE_VERSION, "needs_candidate_union": True, "unresolved": []}
            if intent else "not allowed in B2"
        ),
        "target_fields": {k: v.describe() for k, v in (INTENT_TARGET if intent else EVENT_TARGET).items()},
        "target_dependencies": (
            [] if intent else ["action_cycle/state_transition require nonempty completion_criterion"]
        ),
        "operation_common_fields": {k: v.describe() for k, v in COMMON_OPERATION.items()},
        "operation_fields_by_op": {
            op: {k: v.describe() for k, v in fields.items()} for op, fields in OPERATION_FIELDS.items()
        },
        "operation_dependencies": [
            "event_duration with duration_aggregation=compare additionally requires duration_comparison: longest/shortest",
            "duration_comparison is not applicable otherwise",
            "anchor_target_id must reference a separately declared target outside target_ids",
            "first_k/last_k/nth_occurrence cannot use selection=first/last/unique (which preselects one event)",
        ],
        "scope_fields_by_kind": {
            kind: {k: v.describe() for k, v in fields.items()} for kind, fields in SCOPE_FIELDS.items()
        },
        "scope_dependencies": ["relative_first_sec and relative_last_sec are mutually exclusive"],
        "field_policy": (
            "Unknown keys are errors. Omit optional/inapplicable keys. Optional null uses its declared default; "
            "inapplicable null is removed; empty cooccurrence_targets=[] is also omitted outside cooccurrence. "
            "Other inapplicable values are errors. Required null/missing is an error. "
            "Never coerce strings or booleans to numbers. All target references must exist. "
            "interval=[start,end] requires finite numbers and 0<=start<end."
        ),
    }


def contract_text(stage: str) -> str:
    return json.dumps(output_contract(stage), ensure_ascii=False, separators=(",", ":"))


def _matches(value: Any, rule: Field) -> bool:
    number = type(value) in (int, float) and math.isfinite(value)
    kinds = {
        "string": isinstance(value, str),
        "nonempty_string": isinstance(value, str) and bool(value.strip()),
        "integer": type(value) is int,
        "positive_number": number and value > 0,
        "boolean": type(value) is bool,
        "strings": isinstance(value, list) and all(isinstance(v, str) and v.strip() for v in value),
        "nonempty_strings": isinstance(value, list) and bool(value) and all(
            isinstance(v, str) and v.strip() for v in value
        ),
        "interval": isinstance(value, list) and len(value) == 2 and all(
            type(v) in (int, float) and math.isfinite(v) for v in value
        ) and 0 <= value[0] < value[1],
    }
    if not kinds[rule.kind]:
        return False
    if rule.values:
        if rule.kind in {"strings", "nonempty_strings"}:
            if any(v not in rule.values for v in value):
                return False
        elif value not in rule.values:
            return False
    return rule.minimum is None or value >= rule.minimum


class _Normalizer:
    def __init__(self, changes: list[dict[str, Any]] | None) -> None:
        self.errors: list[dict[str, Any]] = []
        self.changes = changes if changes is not None else []

    def error(self, path: str, code: str, message: str) -> None:
        self.errors.append({"path": path, "code": code, "message": message})

    def object(self, data: Any, rules: dict[str, Field], path: str, known: set[str] | None = None) -> dict:
        if not isinstance(data, dict):
            self.error(path, "type", "expected object")
            return {}
        result = {}
        for key, value in data.items():
            if key in rules:
                continue
            where = f"{path}.{key}"
            if known is not None and key in known:
                if value is None:
                    self.changes.append({"path": where, "action": "drop_inapplicable_null"})
                elif key == "cooccurrence_targets" and isinstance(value, list) and not value:
                    self.changes.append({"path": where, "action": "drop_inapplicable_empty"})
                else:
                    self.error(where, "inapplicable", "field is not applicable to this variant; omit it")
            else:
                self.error(where, "unknown", "undeclared field")
        for key, rule in rules.items():
            where = f"{path}.{key}"
            value = data.get(key)
            if value is None:
                if rule.required:
                    self.error(where, "required", "required non-null " + rule.kind)
                elif rule.default is not None:
                    result[key] = copy.deepcopy(rule.default)
                    self.changes.append({"path": where, "action": "default", "value": result[key]})
                elif key in data:
                    self.changes.append({"path": where, "action": "omit_optional_null"})
            elif not _matches(value, rule):
                self.error(where, "value", "expected " + json.dumps(rule.describe(), ensure_ascii=False))
            else:
                result[key] = copy.deepcopy(value)
        return result

    def finish(self) -> None:
        if self.errors:
            raise CompileContractError(self.errors)


def _normalize(data: Any, stage: str, changes: list[dict[str, Any]] | None, legacy: bool) -> dict:
    n = _Normalizer(changes)
    if not isinstance(data, dict):
        n.error("$", "type", "expected object")
        n.finish()
    data = copy.deepcopy(data)
    if legacy:
        if data.get("version") == 1:
            data["version"] = WIRE_VERSION
            n.changes.append({"path": "$.version", "action": "legacy_version", "value": WIRE_VERSION})
        if data.get("needs_candidate_union") is False and stage == "compile":
            data.pop("needs_candidate_union")
    intent = stage == "compile_intent"
    operation_key = "tasks" if intent else "operations"
    candidate = intent and data.get("needs_candidate_union") is True
    top = {"version": Field("integer", values=(WIRE_VERSION,)), "unresolved": Field("strings", [])}
    if intent:
        top["needs_candidate_union"] = Field("boolean", False)
    nested = set() if candidate else {"targets", operation_key, "scope"}
    for key in data.keys() - (set(top) | nested):
        n.error("$." + key, "unknown", "undeclared field for this output variant")
    result = n.object({k: v for k, v in data.items() if k in top}, top, "$")
    if candidate:
        n.finish()
        return result
    for key, limit in (("targets", 32), (operation_key, 12)):
        items = data.get(key)
        if not isinstance(items, list) or not 1 <= len(items) <= limit:
            n.error("$." + key, "items", f"expected 1..{limit} objects")
            items = []
        normalized = []
        for i, item in enumerate(items):
            path = f"$.{key}[{i}]"
            if key == "targets":
                rules = dict(INTENT_TARGET if intent else EVENT_TARGET)
                if not intent and isinstance(item, dict) and item.get("unit_kind") in {"action_cycle", "state_transition"}:
                    rules["completion_criterion"] = Field("nonempty_string")
                normalized.append(n.object(item, rules, path))
            else:
                op = item.get("op") if isinstance(item, dict) else None
                if not isinstance(op, str) or op not in OPERATION_FIELDS:
                    n.error(path + ".op", "enum", "expected one of " + ", ".join(OPERATION_FIELDS))
                    continue
                rules = {**COMMON_OPERATION, **OPERATION_FIELDS[op]}
                if op == "event_duration" and item.get("duration_aggregation") == "compare":
                    rules["duration_comparison"] = DURATION_COMPARISON
                entry = n.object(item, rules, path, _ALL_OPERATION)
                if op in {"first_k", "last_k", "nth_occurrence"} and entry.get("selection") in {"first", "last", "unique"}:
                    n.error(path + ".selection", "dependency", "K/ordinal operations require a non-singleton selector")
                normalized.append(entry)
        result[key] = normalized
    scope = data.get("scope")
    kind = scope.get("kind") if isinstance(scope, dict) else None
    if not isinstance(kind, str) or kind not in SCOPE_FIELDS:
        n.error("$.scope.kind", "enum", "expected full/interval/semantic")
        result["scope"] = {}
    else:
        result["scope"] = n.object(scope, {"kind": Field("string"), **SCOPE_FIELDS[kind]}, "$.scope", _ALL_SCOPE)
        if all(result["scope"].get(k) is not None for k in ("relative_first_sec", "relative_last_sec")):
            n.error("$.scope", "dependency", "first and last relative durations are mutually exclusive")
    for key, id_key in (("targets", "target_id"), (operation_key, "operation_id")):
        seen = set()
        for i, entry in enumerate(result[key]):
            identity = entry.get(id_key)
            if identity in seen:
                n.error(f"$.{key}[{i}].{id_key}", "duplicate", "ID must be unique")
            seen.add(identity)
    ids = {t.get("target_id") for t in result["targets"]} - {None}
    for i, op in enumerate(result[operation_key]):
        for field in ("target_ids", "cooccurrence_targets", "anchor_target_id"):
            values = [op[field]] if field == "anchor_target_id" and field in op else op.get(field, [])
            if any(value not in ids for value in values):
                n.error(f"$.{operation_key}[{i}].{field}", "reference", "reference must name a declared target")
        if op.get("anchor_target_id") in op.get("target_ids", []):
            n.error(f"$.{operation_key}[{i}].anchor_target_id", "dependency", "anchor must be a separate target")
    n.finish()
    return result


def normalize_intent(data: Any, changes: list[dict[str, Any]] | None = None) -> IntentSpec:
    return _normalize(data, "compile_intent", changes, False)


def normalize_query(
    data: Any, changes: list[dict[str, Any]] | None = None, *, allow_legacy: bool = False,
) -> dict[str, Any]:
    result = _normalize(data, "compile", changes, allow_legacy)
    # Existing internal invariants remain authoritative after boundary normalization.
    EventQuery.from_dict(result)
    return result
