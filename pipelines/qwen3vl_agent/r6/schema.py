"""Strict role protocols. Syntactic validation is not semantic ground truth."""

import json

import jsonschema

from .logic import expression_atoms
from .types import MODALITIES, RELATIONS, ProtocolError, interval


def obj(properties):
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }


def arr(items, minimum=0, maximum=64):
    return {"type": "array", "items": items, "minItems": minimum, "maxItems": maximum}


def enum(*values):
    return {"enum": list(values)}


def nullable(value):
    return {"anyOf": [value, {"type": "null"}]}


TEXT = {"type": "string", "maxLength": 2000}
NAME = {"type": "string", "minLength": 1, "maxLength": 160}
BOOL = {"type": "boolean"}
NAMES = {**arr(NAME), "uniqueItems": True}
TIME = nullable(arr({"type": "number", "minimum": 0}, 2, 2))
STATUS = enum("supported", "contradicted", "unknown")
ACTION_NAMES = (
    "search_allowed_text",
    "observe_clip",
    "expand_context",
    "inspect_source_frame_or_crop",
    "resolve_entity",
    "analyze_audio_if_available",
    "reduce_by_code",
)
ACTION = obj(
    {
        "kind": enum(*ACTION_NAMES),
        "gap_index": nullable({"type": "integer", "minimum": 0}),
        "span": TIME,
        "source_ids": NAMES,
        "bbox": nullable(arr({"type": "number", "minimum": 0, "maximum": 1}, 4, 4)),
        "query": TEXT,
        "fps": enum(2, 4),
    }
)
GAP = obj(
    {
        "kind": enum(
            "location",
            "identity",
            "local_fact",
            "time",
            "relation_bridge",
            "competing_explanation",
            "coverage",
            "modality_missing",
            "input_ambiguity",
        ),
        "predicate": NAME,
        "candidate_labels": NAMES,
        "entity_ids": NAMES,
        "span": TIME,
        "modality": enum(*MODALITIES),
        "desired_observation": TEXT,
        "blocks_answer": BOOL,
    }
)
ATOM = obj(
    {
        "id": NAME,
        "claim": NAME,
        "entity_ids": NAMES,
        "story_time": TIME,
        "required_modalities": arr(enum(*MODALITIES), 1, 4),
        "relation_type": enum("direct", *RELATIONS),
    }
)
EXPRESSION = {"type": "object"}  # recursive shape and identifiers checked below
OPTION = obj(
    {
        "label": NAME,
        "logic": EXPRESSION,
        "selection_polarity": enum("positive", "negative"),
        "selection_set": nullable(arr({"type": "integer", "minimum": 1})),
        "none_of_context": TEXT,
    }
)
QUERY = obj(
    {
        "answer_operator": enum(
            "best_explanation", "true_statement", "false_statement", "exact_set"
        ),
        "target_description": NAME,
        "reference_scope": TEXT,
        "reference_interval": TIME,
        "subtype": enum("S1", "S2", "S3", "S4", "S5", "S6"),
        "coverage": enum("local", "global", "occasions"),
        "entities": arr(obj({"id": NAME, "description": NAME})),
        "atoms": arr(ATOM, 1),
        "option_claims": arr(OPTION, 2),
        "discriminators": arr(NAME),
        "initial_actions": arr(ACTION, 0, 3),
        "ambiguities": arr(NAME),
    }
)
FACT = obj(
    {
        "source_ids": {**NAMES, "minItems": 1},
        "entity_ids": NAMES,
        "story_time": TIME,
        "kind": enum("observation", "attributed_statement"),
        "predicate": NAME,
        "speaker": nullable(NAME),
        "referred_entity": nullable(NAME),
        "quote_or_paraphrase": enum("quote", "paraphrase"),
        "quality": enum("clear", "partial", "unreadable", "unresolved"),
        "coverage_notes": TEXT,
    }
)
OBSERVATION = obj({"records": arr(FACT, 0, 10), "gaps": arr(GAP), "overflow": BOOL})
RELATION = obj(
    {
        "key": NAME,
        "claim": NAME,
        "relation_type": enum(*RELATIONS),
        "entity_ids": NAMES,
        "story_time": TIME,
        "premise_fact_ids": NAMES,
        "premise_relation_ids": NAMES,
        "bridge": NAME,
        "support_state": STATUS,
        "missing_premises": arr(NAME),
        "strong_alternatives": arr(NAME),
        "conflict_flag": BOOL,
    }
)
ATOM_ASSESSMENT = obj(
    {
        "atom_id": NAME,
        "status": STATUS,
        "fact_ids": NAMES,
        "relation_ids": NAMES,
    }
)
CANDIDATE = obj(
    {
        "label": NAME,
        "atom_assessments": arr(ATOM_ASSESSMENT, 1),
        "answer_target_fit": enum("complete", "partial", "off_target", "unresolved"),
        "fit_reason": TEXT,
        "fit_fact_ids": NAMES,
        "fit_relation_ids": NAMES,
        "missing_premises": arr(NAME),
        "direct": BOOL,
    }
)
OCCASION = obj(
    {
        "index": {"type": "integer", "minimum": 1},
        "status": STATUS,
        "fact_ids": {**NAMES, "minItems": 1},
        "relation_ids": NAMES,
    }
)
ASSESSMENT = obj(
    {
        "relations": arr(RELATION, 0, 12),
        "candidates": arr(CANDIDATE, 2),
        "preferred_label": NAME,
        "competitors": arr(
            obj(
                {
                    "label": NAME,
                    "addressed": BOOL,
                    "reason": TEXT,
                    "fact_ids": NAMES,
                    "relation_ids": NAMES,
                }
            )
        ),
        "gaps": arr(GAP),
        "actions": arr(ACTION, 0, 3),
        "occasions": arr(OCCASION),
        "universe_complete": BOOL,
        "coverage_fact_ids": NAMES,
    }
)
VERIFICATION = obj(
    {
        "checks": arr(
            obj(
                {
                    "check_id": NAME,
                    "state": STATUS,
                    "source_ids": NAMES,
                    "missing": arr(NAME),
                }
            ),
            1,
            3,
        ),
        "gaps": arr(GAP),
    }
)
ANSWER = obj({"preferred_label": NAME, "source_ids": NAMES, "limitations": arr(NAME)})
SCHEMAS = {
    "compiler": QUERY,
    "observer": OBSERVATION,
    "relation_checker": ASSESSMENT,
    "verifier": VERIFICATION,
    "answer": ANSWER,
}


def parse(raw, role):
    text = raw.strip()
    if text.startswith("```") and text.endswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0]
    try:
        value = json.loads(text)
        jsonschema.validate(value, SCHEMAS[role])
    except (ValueError, jsonschema.ValidationError) as exc:
        raise ProtocolError(str(exc)[:600]) from exc
    return value


def unique(rows, key):
    ids = [r[key] for r in rows]
    if len(ids) != len(set(ids)):
        raise ProtocolError(f"duplicate {key}")
    return set(ids)


def validate_query(value, request, contract):
    labels = [c.label for c in request.choices]
    if request.subtype != "auto" and value["subtype"] != request.subtype:
        raise ProtocolError("compiler must preserve the supplied discussion subtype")
    if [c["label"] for c in value["option_claims"]] != labels:
        raise ProtocolError("compiler must preserve all labels and their order")
    entities = unique(value["entities"], "id")
    atoms = unique(value["atoms"], "id")
    if request.reference_scope is not None and value["reference_interval"] != list(
        request.reference_scope
    ):
        raise ProtocolError("compiler cannot replace the supplied reference period")
    for atom in value["atoms"]:
        if set(atom["entity_ids"]) - entities:
            raise ProtocolError("unknown query entity")
        if atom["story_time"] is not None:
            interval(atom["story_time"], point=True)
    for option in value["option_claims"]:
        if expression_atoms(option["logic"]) - atoms:
            raise ProtocolError("expression references unknown atom")
        if '"none_of"' in json.dumps(option["logic"]) and not option["none_of_context"].strip():
            raise ProtocolError("NONE_OF requires its question-relative selection meaning")
        if value["answer_operator"] == "exact_set" and option["selection_set"] is None:
            raise ProtocolError("exact-set option needs literal selected occasion numbers")
    for action in value["initial_actions"]:
        if action["span"] is not None and not contract.permits_span(action["span"]):
            raise ProtocolError("compiler proposed an illegal initial observation")


def validate_observation(value, sources, query, maximum=10):
    if len(value["records"]) > maximum:
        raise ProtocolError("fact count exceeds configured limit; split the observation")
    if not value["records"] and not value["gaps"]:
        raise ProtocolError("empty observation must explain an evidence gap")
    entities = {e["id"] for e in query["entities"]}
    for fact in value["records"]:
        if set(fact["source_ids"]) - sources.keys():
            raise ProtocolError("observer cited a source not shown in this call")
        if set(fact["entity_ids"]) - entities:
            raise ProtocolError("unbound entity; use a gap rather than inventing an identity")
        if fact["story_time"] is not None:
            interval(fact["story_time"], point=True)
        if (
            all(sources[s]["modality"] in {"subtitle", "asr"} for s in fact["source_ids"])
            and fact["kind"] != "attributed_statement"
        ):
            raise ProtocolError("text-only claims must retain statement attribution")
        if fact["kind"] == "attributed_statement" and not fact["speaker"]:
            raise ProtocolError("attributed statement requires speaker or unresolved descriptor")
        if fact["quote_or_paraphrase"] == "quote" and not any(
            sources[s]["modality"] in {"subtitle", "asr"} for s in fact["source_ids"]
        ):
            raise ProtocolError("verbatim quotes require actual transcript text in v1")
        if fact["quote_or_paraphrase"] == "quote" and not any(
            fact["predicate"] in sources[s].get("text", "") for s in fact["source_ids"]
        ):
            raise ProtocolError("quote text is absent from the supplied transcript; use paraphrase")
