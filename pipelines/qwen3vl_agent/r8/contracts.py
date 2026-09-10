"""Bounded JSON contracts for every learned role and executable graph."""

import json

import jsonschema

from .operators import OPS
from .types import ProtocolError


def obj(**properties):
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }


def arr(item, maximum=128):
    return {"type": "array", "items": item, "maxItems": maximum}


def enum(*values):
    return {"enum": list(values)}


S = {"type": "string", "maxLength": 4000}
ID = {"type": "string", "minLength": 1, "maxLength": 160}
STRINGS = arr(S)
BOOL = {"type": "boolean"}
N = {"type": ["number", "null"]}
SPAN = {"type": ["array", "null"], "items": {"type": "number"}, "minItems": 2, "maxItems": 2}
BOX = {"type": ["array", "null"], "items": {"type": "number"}, "minItems": 4, "maxItems": 4}
MAP = {"type": "object", "additionalProperties": ID, "maxProperties": 32}
VALUE = {"type": ["string", "integer", "boolean", "null", "array", "object"]}
SLOT = obj(id=ID, entity_id=ID, attribute=S, role=S, scope=ID, snapshot=ID, description=S)
GIVEN = obj(
    id=ID,
    entity_id=ID,
    attribute=S,
    role=S,
    scope=ID,
    snapshot=ID,
    value=VALUE,
    unit=S,
    unit_basis=S,
    question_span=S,
    origin=enum("given", "hypothetical"),
)
TASK = obj(
    target=ID,
    target_entity=ID,
    answer_type=enum(
        "number", "count", "percentage", "percentage_point", "date", "clock", "tuple", "object"
    ),
    output_unit=S,
    coverage_need=enum("local", "all_items", "all_attempts", "all_transactions"),
    plan=enum("local", "multi", "range"),
    slots=arr(SLOT),
    givens=arr(GIVEN),
    anchors=arr(obj(description=S, time=N)),
    semantic_constraints=STRINGS,
    precision=obj(
        kind=enum("exact", "closest", "rounded"),
        places={"type": ["integer", "null"], "minimum": 0, "maximum": 12},
        question_span=S,
    ),
    scope=ID,
    snapshot=ID,
)
OBS_VAR = obj(
    id=ID,
    entity_id=ID,
    attribute=S,
    role=S,
    scope=ID,
    snapshot=ID,
    raw_text=S,
    value=VALUE,
    unit=S,
    unit_basis=S,
    evidence_refs=STRINGS,
    event_time=N,
    content_time={"type": ["string", "null"]},
    valid_interval=SPAN,
    alternatives=arr(VALUE, 8),
    alternatives_exhaustive=BOOL,
    unresolved=STRINGS,
)
RELATION = obj(
    id=ID,
    kind=enum(
        "equation",
        "rectangle",
        "square",
        "on_segment",
        "collinear",
        "perpendicular",
        "parallel",
        "equal_length",
        "triangle",
        "right_triangle",
        "shared_base",
        "shared_height",
        "similar_triangles",
        "rectangle_partition",
        "nondegenerate",
        "positive",
        "adjacent",
        "measurement",
    ),
    objects=MAP,
    raw_text=S,
    source_kind=enum("explicit_marker", "structure", "given"),
    evidence_refs=STRINGS,
    question_span=S,
    scope=ID,
    snapshot=ID,
)
ENTITY = obj(id=ID, description=S, scope=ID, snapshot=ID)
CONTEXT = obj(reason=S, frame_id={"type": ["string", "null"]}, bbox=BOX, window=SPAN)
ATTEMPT = obj(
    id=ID,
    actor_id=ID,
    scope=ID,
    round_id=ID,
    start_time={"type": "number"},
    end_time=N,
    outcome=enum("success", "failure", "unknown"),
    attributes={
        "type": "object",
        "additionalProperties": {"type": ["boolean", "null"]},
        "maxProperties": 32,
    },
    evidence_refs=STRINGS,
    replay_of={"type": ["string", "null"]},
    duplicate_of={"type": ["string", "null"]},
)
ITEM = obj(
    id=ID,
    entity_id=ID,
    scope=ID,
    snapshot=ID,
    attributes=MAP,
    evidence_refs=STRINGS,
    duplicate_of={"type": ["string", "null"]},
    readable=BOOL,
)
TRANSACTION = obj(
    id=ID,
    payer=ID,
    payee=ID,
    amount_ref=ID,
    scope=ID,
    snapshot=ID,
    nature=enum("payment", "refund", "loan", "repayment", "purchase", "sale"),
    stage=ID,
    time={"type": "number"},
    evidence_refs=STRINGS,
    duplicate_of={"type": ["string", "null"]},
)
OBSERVATION = obj(
    entities=arr(ENTITY),
    observations=arr(OBS_VAR),
    relations=arr(RELATION),
    attempts=arr(ATTEMPT),
    items=arr(ITEM),
    transactions=arr(TRANSACTION),
    unresolved=STRINGS,
    requested_context=arr(CONTEXT, 8),
    discovery=obj(
        complete=BOOL,
        rationale=S,
        open_event_boundaries=STRINGS,
        possible_replays=STRINGS,
        unreadable_items=STRINGS,
    ),
)
ARG = {"oneOf": [ID, obj(refs=arr(ID, 128)), obj(value=VALUE, unit=S, source=ID)]}
PARAMETERS = {
    "type": "object",
    "properties": {
        "attribute": ID,
        "relation": enum("eq", "ne", "lt", "le", "gt", "ge"),
        "key": ID,
        "index_base": enum(0, 1),
        "count_unit": S,
        "kind": S,
        "wrap_24h": BOOL,
        "format": S,
        "target": ID,
    },
    "additionalProperties": False,
}
NODE = obj(id=ID, op=enum(*OPS), args=arr(ARG, 3), params=PARAMETERS)
QUERY = obj(nodes=arr(NODE, 64), target_node={"oneOf": [ID, obj(refs=arr(ID, 128))]}, output_unit=S)
TERM = {
    "oneOf": [
        obj(ref=ID),
        obj(constant=ID),
        obj(
            op=enum("add", "subtract", "multiply", "divide", "square"),
            args=arr({"$ref": "#/$defs/term"}, 2),
        ),
    ]
}
CONSTRAINT = obj(
    id=ID,
    relation=enum("equal", "less_equal", "greater_equal", "less", "greater", "not_equal"),
    args=arr({"$ref": "#/$defs/term"}, 2),
    sources=STRINGS,
    snapshot=ID,
)
SYMBOL = obj(
    id=ID,
    entity_id=ID,
    attribute=ID,
    unit=S,
    domain=enum("real", "integer", "positive", "nonnegative"),
    sources=STRINGS,
    snapshot=ID,
)
RULE = obj(
    id=ID,
    rule=enum(
        "rectangle",
        "square",
        "segment_addition",
        "triangle_angles",
        "shared_base",
        "shared_height",
        "pythagoras",
        "similar_triangles",
        "rectangle_partition",
    ),
    premises=STRINGS,
    mapping=MAP,
    snapshot=ID,
)
GEOMETRY = obj(
    symbols=arr(SYMBOL, 64),
    constraints=arr(CONSTRAINT, 128),
    rules=arr(RULE, 50),
    target={"$ref": "#/$defs/term"},
    output_unit=S,
)
ADAPTER_QUERY = obj(
    id=ID,
    kind=enum("attempts", "inventory", "transactions", "replace_rule"),
    scope=ID,
    actor=ID,
    round_id=S,
    boundary=enum("start", "end", "contained"),
    window=SPAN,
    attribute=S,
    operation=S,
    rule_span=S,
    complete_claim=BOOL,
    initial_ref={"type": ["string", "null"]},
)
FORMAL = obj(
    backend=enum("direct", "constraints"),
    query=QUERY,
    geometry=GEOMETRY,
    adapters=arr(ADAPTER_QUERY, 16),
    checks=STRINGS,
    unresolved=STRINGS,
)
DEFECT = obj(
    kind=enum(
        "unreadable_value",
        "binding_ambiguity",
        "snapshot_conflict",
        "missing_denominator",
        "incomplete_extrema_set",
        "unsupported_relation",
        "bad_ir",
        "solver_unknown",
        "option_sensitive_uncertainty",
    ),
    detail=S,
    variable_refs=STRINGS,
    evidence_refs=STRINGS,
    window=SPAN,
    bbox=BOX,
    blocking=BOOL,
    answer_sensitive=BOOL,
    dependent_nodes={"type": "integer", "minimum": 0, "maximum": 64},
)
AUDIT = obj(
    semantics_ok=BOOL, sources_ok=BOOL, coverage_ok=BOOL, defects=arr(DEFECT, 16), explanation=S
)
DIRECT = obj(
    prediction={"type": ["string", "null"]},
    value={"type": ["string", "null"]},
    unit=S,
    explanation=S,
)
SCHEMAS = {
    "compile": TASK,
    "observe": OBSERVATION,
    "reread": OBSERVATION,
    "formalize": {**FORMAL, "$defs": {"term": TERM}},
    "audit": AUDIT,
    "direct": DIRECT,
    "model_math": DIRECT,
    "fallback": DIRECT,
}


def validate(value, role):
    jsonschema.Draft202012Validator(SCHEMAS[role]).validate(value)

    # Bound even flexible record values, without accepting arbitrary deeply nested payloads.
    def visit(v, depth=0):
        if depth > 24:
            raise ProtocolError("JSON depth limit")
        if isinstance(v, dict):
            if len(v) > 128:
                raise ProtocolError("JSON object limit")
            for x in v.values():
                visit(x, depth + 1)
        elif isinstance(v, list):
            if len(v) > 4096:
                raise ProtocolError("JSON array limit")
            for x in v:
                visit(x, depth + 1)
        elif isinstance(v, int) and len(str(abs(v))) > 100:
            raise ProtocolError("JSON integer digit limit")

    visit(value)
    return value


def parse(text, role, validator=None):
    if len(text) > 160000:
        raise ProtocolError("model output length limit")
    raw = text.strip()
    if raw.startswith("```") and raw.endswith("```"):
        raw = raw.split("\n", 1)[1].rsplit("```", 1)[0]

    def unique(pairs):
        out = {}
        for key, value in pairs:
            if key in out:
                raise ProtocolError("duplicate JSON key")
            out[key] = value
        return out

    value = json.loads(
        raw,
        object_pairs_hook=unique,
        parse_constant=lambda s: (_ for _ in ()).throw(ProtocolError(s)),
    )
    validate(value, role)
    if validator:
        validator(value)
    return value
