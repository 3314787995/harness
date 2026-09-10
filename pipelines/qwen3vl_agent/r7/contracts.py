"""Small role schemas plus semantic checks; model JSON never grants media access."""

import json

import jsonschema

from .types import MECHANISMS, ProtocolError, plain


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
N = {"type": ["number", "null"]}
ANY = {}
STRINGS = arr(S)
EXPR = {"oneOf": [obj(ref=S), obj(literal=ANY, source=S)]}
INTERVENTION = obj(
    id=S,
    op=enum("set", "remove", "swap", "delay", "replace_event", "continue_trend"),
    target=S,
    value=ANY,
    read_world=enum("factual", "current"),
    sequential={"type": "boolean"},
    source_span=S,
)
INTERVENTION["properties"]["at_time"] = {"type": ["number", "null"], "minimum": 0}
INTERVENTION["properties"]["scenario_id"] = S
SLOT = obj(id=S, entity_id=S, predicate=S, description=S)
TASK = obj(
    mechanisms=arr(enum(*MECHANISMS), 5),
    target_visibility=enum("unobserved_future", "counterfactual", "observed", "unresolved"),
    query_operator=enum("most_likely", "least_likely", "will", "will_not", "value", "ordering"),
    targets=arr(obj(id=S, description=S)),
    slots=arr(SLOT),
    anchors=arr(
        obj(
            description=S,
            time_span={
                "type": ["array", "null"],
                "items": {"type": "number"},
                "minItems": 2,
                "maxItems": 2,
            },
        )
    ),
    target_time=N,
    interventions=arr(INTERVENTION),
    invariants=STRINGS,
    stipulations=arr(obj(id=S, key=S, value=ANY, source_span=S)),
    unresolved=STRINGS,
)
ATOM = obj(
    id=S,
    text_span=S,
    scenario_id=S,
    key=S,
    relation=enum("eq", "ne", "lt", "le", "gt", "ge", "contains", "semantic"),
    expected=ANY,
    polarity=enum("positive", "negative"),
    modality=enum("will", "may", "likely", "least_likely", "fact"),
)
CANDIDATES = obj(
    candidates=arr(obj(label=S, text=S, atoms=arr(ATOM), interventions=arr(INTERVENTION)))
)
OBSERVATION = obj(
    entities=arr(obj(id=S, description=S)),
    facts=arr(
        obj(
            key=S,
            entity_id=S,
            predicate=S,
            value=ANY,
            kind=enum("observed", "unknown", "reported_claim"),
            evidence_ids=STRINGS,
            unit=S,
            time=N,
            complete={"type": "boolean"},
        )
    ),
    gaps=STRINGS,
)
STEP = obj(
    id=S,
    op=enum(
        "COUNT_EVENTS",
        "READ_FACT",
        "SET",
        "ADD",
        "SWAP",
        "TABLE_LOOKUP",
        "RULE_TRANSITION",
        "SORT",
        "SELECT",
        "COMPARE",
    ),
    out=S,
    args=arr(EXPR, 32),
    params={"type": "object"},
    rule_id=S,
)
RULE = obj(id=S, basis=enum("stipulated", "hypothesis"), source_span=S, description=S)
HYPOTHESIS = obj(
    id=S,
    mechanism=enum(*MECHANISMS),
    output=S,
    value=ANY,
    conditions=arr(
        obj(key=S, relation=enum("eq", "ne", "lt", "le", "gt", "ge", "contains"), expected=ANY)
    ),
    rule_id=S,
)
PHYSICS = obj(
    objects=arr(
        obj(
            entity_id=S,
            exists_key=S,
            position_key=S,
            velocity_key=S,
            radius_key=S,
            reference_key=S,
            valid_until_key=S,
            delay_key=S,
        )
    ),
    horizon=EXPR,
    supports=arr(obj(supporter=S, supported=S, key=S)),
    rule_id=S,
)
TREND = obj(
    output=S,
    entities=STRINGS,
    entity_binding_time=EXPR,
    trend_start=EXPR,
    trend_end=EXPR,
    base_time=EXPR,
    target_time=EXPR,
    method=enum("absolute", "ratio"),
    series=arr(obj(entity_id=S, start_key=S, end_key=S, base_key=S)),
    binding_key=S,
    universe_key=S,
    rule_id=S,
)
SCENARIO = obj(
    id=S,
    candidate_label={"type": ["string", "null"]},
    programs=arr(STEP, 64),
    hypotheses=arr(HYPOTHESIS),
    physics=arr(PHYSICS, 8),
    trends=arr(TREND, 16),
)
REASON = obj(
    rules=arr(RULE), factual_program=arr(STEP, 64), scenarios=arr(SCENARIO), unresolved=STRINGS
)
GAP = obj(
    kind=enum("perceptual", "binding", "mechanism", "future_ambiguity", "modality", "protocol"),
    affects_candidates=STRINGS,
    neutral_query=S,
    window={"type": ["array", "null"], "items": {"type": "number"}, "minItems": 2, "maxItems": 2},
    action=enum("observe", "crop", "none"),
    frame_id={"type": ["string", "null"]},
    bbox={"type": ["array", "null"], "items": {"type": "number"}, "minItems": 4, "maxItems": 4},
    impact=enum(1, 2, 3),
    resolvability=enum(0, 1, 2, 3),
)
VERIFY = obj(
    prediction=S,
    assessments=arr(
        obj(
            label=S,
            atoms=arr(
                obj(
                    id=S,
                    status=enum("entailed_by_execution", "supported", "contradicted", "unknown"),
                    evidence_ids=STRINGS,
                    reason=S,
                )
            ),
        )
    ),
    gaps=arr(GAP, 8),
    unresolved=STRINGS,
)
DIRECT = obj(prediction=S, reason=S, unresolved=STRINGS)
SCHEMAS = {
    "compile": TASK,
    "candidates": CANDIDATES,
    "observe": OBSERVATION,
    "reason": REASON,
    "verify": VERIFY,
    "final": VERIFY,
    "direct": DIRECT,
}


def parse(text, role, validator=None):
    raw = text.strip()
    if raw.startswith("```") and raw.endswith("```"):
        raw = raw.split("\n", 1)[1].rsplit("```", 1)[0]

    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ProtocolError("duplicate JSON key")
            result[key] = value
        return result

    value = json.loads(
        raw,
        object_pairs_hook=unique,
        parse_constant=lambda s: (_ for _ in ()).throw(ProtocolError(s)),
    )
    jsonschema.Draft202012Validator(SCHEMAS[role]).validate(value)
    plain(value)
    if validator:
        validator(value)
    return value


def unique_ids(items, field="id"):
    values = [x[field] for x in items]
    if any(not str(v).strip() for v in values) or len(values) != len(set(values)):
        raise ProtocolError(f"empty/duplicate {field}")


def validate_interventions(items, text):
    unique_ids(items)
    for item in items:
        if not item["source_span"] or item["source_span"] not in text:
            raise ProtocolError("intervention lacks an exact source span")
        if item["read_world"] == "current" and not item["sequential"]:
            raise ProtocolError("current-world RHS requires explicit sequential intervention")


def validate_task(value, request):
    for name in ("targets", "slots", "stipulations"):
        unique_ids(value[name])
    if not value["mechanisms"]:
        raise ProtocolError("at least one mechanism is required")
    if request.execution_subtype and request.execution_subtype not in value["mechanisms"]:
        raise ProtocolError("compiled mechanism conflicts with supplied hint")
    for item in value["stipulations"]:
        if not item["source_span"] or item["source_span"] not in request.question:
            raise ProtocolError("stipulation lacks exact question source")
    validate_interventions(value["interventions"], request.question)
    targets = {x["id"] for x in value["targets"]}
    if any(s["entity_id"] not in targets for s in value["slots"]):
        raise ProtocolError("slot refers to unknown entity")


def validate_candidates(value, request):
    candidates = value["candidates"]
    if [(c["label"], c["text"]) for c in candidates] != [
        (c.label, c.text) for c in request.choices
    ]:
        raise ProtocolError("candidate compiler changed original labels/order/text")
    for candidate in candidates:
        unique_ids(candidate["atoms"])
        if not candidate["atoms"]:
            raise ProtocolError("every candidate needs its atomic commitments")
        for atom in candidate["atoms"]:
            if not atom["text_span"] or atom["text_span"] not in candidate["text"]:
                raise ProtocolError("atom lacks original option span")
            if candidate["interventions"] and atom["scenario_id"] == "factual":
                raise ProtocolError("candidate intervention cannot be evaluated in factual world")
        validate_interventions(candidate["interventions"], candidate["text"])


def validate_observation(value, spec, evidence, known_entities):
    unique_ids(value["entities"])
    known = {e["id"]: e["description"] for e in [*spec["targets"], *known_entities]}
    for entity in value["entities"]:
        if entity["id"] in known and entity["description"] != known[entity["id"]]:
            raise ProtocolError("stable entity ID rebound to a different description")
        known[entity["id"]] = entity["description"]
    for fact in value["facts"]:
        if fact["entity_id"] not in known:
            raise ProtocolError("unknown observation entity")
        if any(e not in evidence for e in fact["evidence_ids"]):
            raise ProtocolError("observation cites unavailable evidence")
        if fact["kind"] != "unknown" and not fact["evidence_ids"]:
            raise ProtocolError("observation requires source evidence")
        if fact["kind"] == "unknown" and fact["value"] is not None:
            raise ProtocolError("unknown observations must carry null values")
        if fact["time"] is not None and fact["evidence_ids"]:
            times = [evidence[e]["timestamp_seconds"] for e in fact["evidence_ids"]]
            if not min(times) - 1e-6 <= fact["time"] <= max(times) + 1e-6:
                raise ProtocolError("fact time outside its evidence span")
        if (
            any(evidence[e].get("modality") == "subtitle" for e in fact["evidence_ids"])
            and fact["kind"] == "observed"
        ):
            raise ProtocolError("subtitle claims must remain reported_claim")


def validate_reason(value, spec, candidates, max_steps, max_branches):
    unique_ids(value["rules"])
    unique_ids(value["scenarios"])
    sources = [s["source_span"] for s in spec["stipulations"]] + [
        s["source_span"] for s in spec["interventions"]
    ]
    for rule in value["rules"]:
        if rule["basis"] == "stipulated" and (
            not rule["source_span"] or not any(rule["source_span"] in s for s in sources)
        ):
            raise ProtocolError("stipulated rule not backed by compiled question")
    labels = {c["label"] for c in candidates}
    scenarios = {s["id"]: s for s in value["scenarios"]}
    if "factual" in scenarios:
        raise ProtocolError("factual is reserved for the immutable factual world")
    for intervention in spec["interventions"]:
        if (
            intervention.get("scenario_id", "*") != "*"
            and intervention["scenario_id"] not in scenarios
        ):
            raise ProtocolError("explicit question scenario is missing")
    if any(
        s["candidate_label"] is not None and s["candidate_label"] not in labels
        for s in scenarios.values()
    ):
        raise ProtocolError("scenario refers to unknown candidate")
    for candidate in candidates:
        for atom in candidate["atoms"]:
            if atom["scenario_id"] == "factual":
                if spec["interventions"]:
                    raise ProtocolError("intervened result cannot cite factual world directly")
                continue
            scenario = scenarios.get(atom["scenario_id"])
            if scenario is None:
                raise ProtocolError("explicit candidate scenario missing")
            if candidate["interventions"] and scenario["candidate_label"] != candidate["label"]:
                raise ProtocolError("candidate-specific intervention needs its own world")
    for steps in [value["factual_program"], *(s["programs"] for s in scenarios.values())]:
        unique_ids(steps)
        if len(steps) > max_steps:
            raise ProtocolError("program step budget exceeded")
    for scenario in scenarios.values():
        counts = {}
        for h in scenario["hypotheses"]:
            counts[h["output"]] = counts.get(h["output"], 0) + 1
        if any(n > max_branches for n in counts.values()):
            raise ProtocolError("too many latent alternatives for an explicit scenario")


def validate_verdict(value, candidates, valid_ids):
    if value["prediction"] not in {c["label"] for c in candidates}:
        raise ProtocolError("invalid prediction label")
    expected = {c["label"]: {a["id"] for a in c["atoms"]} for c in candidates}
    if len(value["assessments"]) != len(expected):
        raise ProtocolError("all candidates must be audited")
    unique_ids(value["assessments"], "label")
    for entry in value["assessments"]:
        if entry["label"] not in expected:
            raise ProtocolError("unknown candidate")
        atoms = entry["atoms"]
        unique_ids(atoms)
        if {a["id"] for a in atoms} != expected[entry["label"]]:
            raise ProtocolError("every compound atom must be audited")
        for atom in atoms:
            if set(atom["evidence_ids"]) - valid_ids:
                raise ProtocolError("verifier cites unavailable evidence/state")


def validate_execution_choice(value, assessments, query_operator="value"):
    if query_operator in {"will_not", "least_likely"}:
        return  # A false proposition can answer a negative query correctly.
    eliminated = {
        a["label"] for a in assessments if any(x["status"] == "contradicted" for x in a["atoms"])
    }
    remaining = {a["label"] for a in assessments} - eliminated
    if remaining and value["prediction"] in eliminated:
        raise ProtocolError("chosen candidate conflicts with a grounded executed atom")
