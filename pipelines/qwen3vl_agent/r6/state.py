"""Versioned F/R/C memory; conservative program-side evidence obligations."""

from copy import deepcopy

from .logic import conjunction, evaluate_expression, exact_set_status, expression_atoms, negate
from .schema import unique
from .types import ProtocolError, digest


def new_state():
    return {
        "facts": {},
        "relations": {},
        "sources": {},
        "coverage": [],
        "actions": [],
        "assessments": [],
        "verifications": [],
        "gaps": [],
        "refinements": 0,
        "events": [],
        "phase": "compile",
        "stop_reason": None,
    }


def add_observation(state, value, sources, *, job_key):
    """Exactly-once application, separate from exactly-once successful model replay."""
    if job_key in state.setdefault("applied_observations", []):
        return 0
    state["sources"].update({s["id"]: deepcopy(s) for s in sources.values()})
    before = {
        digest({k: v for k, v in f.items() if k not in {"id", "job_key"}})
        for f in state["facts"].values()
    }
    added = 0
    for raw in value["records"]:
        fact = deepcopy(raw)
        fact["source_ids"] = sorted({sources[s]["id"] for s in raw["source_ids"]})
        times = [state["sources"][s]["source_time"] for s in fact["source_ids"]]
        fact["source_time"] = [min(t[0] for t in times), max(t[1] for t in times)]
        signature = digest(fact)
        if signature in before:
            continue
        fid = f"F{len(state['facts']) + 1:06d}"
        state["facts"][fid] = {**fact, "id": fid, "job_key": job_key}
        before.add(signature)
        added += 1
    state["gaps"].extend(deepcopy(value["gaps"]))
    state["applied_observations"].append(job_key)
    return added


def relation_sources(rid, facts, relations, trail=()):
    if rid in trail or rid not in relations:
        raise ProtocolError("cyclic or unknown relation dependency")
    relation = relations[rid]
    result = set()
    for fid in relation["premise_fact_ids"]:
        if fid not in facts:
            raise ProtocolError("relation cites unknown/unshown fact")
        result.update(facts[fid]["source_ids"])
    for parent in relation["premise_relation_ids"]:
        result.update(relation_sources(parent, facts, relations, (*trail, rid)))
    if not result:
        raise ProtocolError("relation must reach actual source evidence")
    return result


def _usable_relation(rid, facts, relations):
    relation = relations[rid]
    return (
        relation["support_state"] != "unknown"
        and not relation["missing_premises"]
        and not relation["conflict_flag"]
        and all(facts[f]["quality"] == "clear" for f in relation["premise_fact_ids"])
        and all(_usable_relation(r, facts, relations) for r in relation["premise_relation_ids"])
    )


def _matching_time(record, atom):
    wanted, actual = atom["story_time"], record["story_time"]
    return wanted is None or (
        actual is not None and max(wanted[0], actual[0]) <= min(wanted[1], actual[1])
    )


def _references(row, facts, relations, fact_key="fact_ids", relation_key="relation_ids"):
    if set(row[fact_key]) - facts.keys() or set(row[relation_key]) - relations.keys():
        raise ProtocolError("assessment references unshown/unknown evidence")
    result = {s for fid in row[fact_key] for s in facts[fid]["source_ids"]}
    for rid in row[relation_key]:
        result.update(relation_sources(rid, facts, relations))
    return result


def evaluate_assessment(
    value, query, facts, old_relations, sources, *, coverage_complete=False, relation_offset=None
):
    """Pure validation/normalization; commit only after the entire response is accepted."""
    data = deepcopy(value)
    labels = [o["label"] for o in query["option_claims"]]
    if [c["label"] for c in data["candidates"]] != labels or data["preferred_label"] not in labels:
        raise ProtocolError("assessment must include every original candidate in order")
    unique(data["relations"], "key")
    old_count = len(old_relations) if relation_offset is None else relation_offset
    mapping = {r["key"]: f"R{old_count + i + 1:06d}" for i, r in enumerate(data["relations"])}
    if mapping.keys() & old_relations.keys():
        raise ProtocolError("local relation key collides with prior version")
    relations = deepcopy(old_relations)
    for r in data["relations"]:
        r["id"] = mapping[r.pop("key")]
        r["premise_relation_ids"] = [mapping.get(i, i) for i in r["premise_relation_ids"]]
        relations[r["id"]] = r
    for rid in relations:
        relation_sources(rid, facts, relations)
    atoms = {a["id"]: a for a in query["atoms"]}
    warnings = []
    for candidate, option in zip(data["candidates"], query["option_claims"], strict=True):
        expected = expression_atoms(option["logic"])
        if unique(candidate["atom_assessments"], "atom_id") != expected:
            raise ProtocolError("candidate must assess exactly its necessary atoms")
        states = {}
        cited = set()
        for a in candidate["atom_assessments"]:
            a["relation_ids"] = [mapping.get(i, i) for i in a["relation_ids"]]
            refs = _references(a, facts, relations)
            cited.update(refs)
            atom = atoms[a["atom_id"]]
            support_records = [facts[f] for f in a["fact_ids"] if facts[f]["quality"] == "clear"]
            support_records += [
                relations[r] for r in a["relation_ids"] if _usable_relation(r, facts, relations)
            ]
            modalities = {sources[s]["modality"] for s in refs}
            entity_ok = any(
                set(atom["entity_ids"]) <= set(r["entity_ids"]) and _matching_time(r, atom)
                for r in support_records
            )
            relation_ok = atom["relation_type"] == "direct" or any(
                relations[r]["relation_type"] == atom["relation_type"]
                and _usable_relation(r, facts, relations)
                and relations[r]["support_state"] == a["status"]
                for r in a["relation_ids"]
            )
            missing_modalities = sorted(set(atom["required_modalities"]) - modalities)
            if a["status"] != "unknown" and (
                not refs
                or not entity_ok
                or not relation_ok
                or missing_modalities
                or any(facts[f]["quality"] != "clear" for f in a["fact_ids"])
            ):
                warnings.append(
                    {
                        "atom_id": a["atom_id"],
                        "reason": "grounding_obligation_missing",
                        "missing_modalities": missing_modalities,
                    }
                )
                a["status"] = "unknown"
            states[a["atom_id"]] = a["status"]
        candidate["fit_relation_ids"] = [mapping.get(i, i) for i in candidate["fit_relation_ids"]]
        cited.update(_references(candidate, facts, relations, "fit_fact_ids", "fit_relation_ids"))
        if candidate["answer_target_fit"] in {"complete", "off_target"} and not (
            candidate["fit_fact_ids"] or candidate["fit_relation_ids"]
        ):
            candidate["answer_target_fit"] = "unresolved"
        candidate["factual_status"] = evaluate_expression(option["logic"], states)
        candidate["selection_status"] = (
            negate(candidate["factual_status"])
            if option["selection_polarity"] == "negative"
            else candidate["factual_status"]
        )
        candidate["source_ids"] = sorted(cited)
        if candidate["direct"] and any(atoms[i]["relation_type"] != "direct" for i in expected):
            candidate["direct"] = False
    shared_states = {}
    for candidate in data["candidates"]:
        for a in candidate["atom_assessments"]:
            shared_states.setdefault(a["atom_id"], set()).add(a["status"])
    conflicts = {aid for aid, values in shared_states.items() if len(values) > 1}
    for candidate, option in zip(data["candidates"], query["option_claims"], strict=True):
        for a in candidate["atom_assessments"]:
            if a["atom_id"] in conflicts:
                a["status"] = "unknown"
        if conflicts:
            candidate["factual_status"] = evaluate_expression(
                option["logic"], {a["atom_id"]: a["status"] for a in candidate["atom_assessments"]}
            )
            candidate["selection_status"] = (
                negate(candidate["factual_status"])
                if option["selection_polarity"] == "negative"
                else candidate["factual_status"]
            )
    warnings.extend(
        {"atom_id": aid, "reason": "conflicting_shared_atom"} for aid in sorted(conflicts)
    )
    if unique(data["competitors"], "label") != set(labels) - {data["preferred_label"]}:
        raise ProtocolError("every nonpreferred candidate needs a comparison entry")
    for competitor in data["competitors"]:
        competitor["model_addressed"] = competitor["addressed"]
        competitor["relation_ids"] = [mapping.get(i, i) for i in competitor["relation_ids"]]
        refs = _references(competitor, facts, relations)
        candidate = next(c for c in data["candidates"] if c["label"] == competitor["label"])
        distinguished = (
            candidate["selection_status"] == "contradicted"
            or candidate["answer_target_fit"] == "off_target"
            or (
                candidate["selection_status"] == "supported"
                and candidate["answer_target_fit"] == "partial"
            )
        )
        competitor["addressed"] = bool(
            competitor["addressed"] and refs and competitor["reason"] and distinguished
        )
    if set(data["coverage_fact_ids"]) - facts.keys():
        raise ProtocolError("coverage references unknown evidence")
    if query["answer_operator"] == "exact_set":
        unique(data["occasions"], "index")
        numbered = sorted(data["occasions"], key=lambda o: o["index"])
        if [o["index"] for o in numbered] != list(range(1, len(numbered) + 1)):
            raise ProtocolError("actual occasions must be numbered consecutively")
        last_time = -1
        for occasion in numbered:
            occasion["relation_ids"] = [mapping.get(i, i) for i in occasion["relation_ids"]]
            _references(occasion, facts, relations)
            times = [facts[f]["story_time"] for f in occasion["fact_ids"]]
            if any(t is None for t in times):
                occasion["status"] = "unknown"
            elif min(t[0] for t in times) <= last_time:
                raise ProtocolError("occasion numbering is not chronological")
            else:
                last_time = min(t[0] for t in times)
            if (
                not occasion["relation_ids"]
                or any(not _usable_relation(r, facts, relations) for r in occasion["relation_ids"])
                or any(facts[f]["quality"] != "clear" for f in occasion["fact_ids"])
            ):
                occasion["status"] = "unknown"
        complete = bool(
            data["universe_complete"]
            and coverage_complete
            and data["coverage_fact_ids"]
            and numbered
        )
        data["universe_complete"] = complete
        states = {o["index"]: o["status"] for o in numbered}
        for c, o in zip(data["candidates"], query["option_claims"], strict=True):
            c["selection_status"] = conjunction(
                [
                    c["selection_status"],
                    exact_set_status(o["selection_set"], states, universe_complete=complete),
                ]
            )
    for competitor in data["competitors"]:
        candidate = next(c for c in data["candidates"] if c["label"] == competitor["label"])
        refs = _references(competitor, facts, relations)
        distinguished = (
            candidate["selection_status"] == "contradicted"
            or candidate["answer_target_fit"] == "off_target"
            or (
                candidate["selection_status"] == "supported"
                and candidate["answer_target_fit"] == "partial"
            )
        )
        competitor["addressed"] = bool(
            competitor["model_addressed"] and refs and competitor["reason"] and distinguished
        )
    data["normalizations"] = warnings
    data["relations"] = [r for rid, r in relations.items() if rid not in old_relations]
    return data


def obligations_met(assessment, query):
    candidate = next(
        c for c in assessment["candidates"] if c["label"] == assessment["preferred_label"]
    )
    return (
        not query["ambiguities"]
        and candidate["selection_status"] == "supported"
        and candidate["answer_target_fit"] == "complete"
        and not candidate["missing_premises"]
        and all(c["addressed"] for c in assessment["competitors"])
        and not any(g["blocks_answer"] for g in assessment["gaps"])
    )


def verification_checks(assessment, query, facts, relations):
    candidate = next(
        c for c in assessment["candidates"] if c["label"] == assessment["preferred_label"]
    )
    # One complete-proposition check plus critical bridges, all without candidate labels.
    atoms = {a["id"]: a for a in query["atoms"]}
    option = next(o for o in query["option_claims"] if o["label"] == candidate["label"])
    claims = [atoms[a["atom_id"]]["claim"] for a in candidate["atom_assessments"]]
    checks = [
        {
            "check_id": "C1",
            "question": query["target_description"],
            "claims_to_check": claims,
            "atom_claims": {
                a["atom_id"]: atoms[a["atom_id"]]["claim"] for a in candidate["atom_assessments"]
            },
            "literal_proposition": option["text"],
            "expression": option["logic"],
            "selection_polarity": option["selection_polarity"],
            "none_of_context": option["none_of_context"],
            "selected_occasions": option["selection_set"],
            "source_ids": candidate["source_ids"],
        }
    ]
    relation_ids = list(
        dict.fromkeys(r for a in candidate["atom_assessments"] for r in a["relation_ids"])
    )
    for rid in relation_ids[:2]:
        relation = relations[rid]
        checks.append(
            {
                "check_id": f"C{len(checks) + 1}",
                "question": relation["claim"],
                "claims_to_check": ["Check attribution, story period and alternatives."],
                "source_ids": sorted(relation_sources(rid, facts, relations)),
            }
        )
    return checks
