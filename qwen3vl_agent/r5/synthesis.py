"""Bounded hierarchical synthesis followed by a directly retained Composer answer."""

from __future__ import annotations

import math
from functools import partial
from typing import Any

from qwen3vl_agent.r5.ledger import FactCardStore, identity, statement, strings
from qwen3vl_agent.r5.planning import merge_calls
from qwen3vl_agent.r5.runtime import ModelSession, NeedsSplit
from qwen3vl_agent.r5.types import ProtocolError


def parse_merge(data: dict, units: list[dict], config: Any) -> dict:
    allowed = {u["id"] for u in units}
    rows = data.get("claims")
    if not isinstance(rows, list) or not rows or len(rows) > config.max_claims:
        raise ProtocolError("invalid merged claims")
    claims, used = [], set()
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ProtocolError(f"claims[{index}] must be an object")
        try:
            refs = strings(row.get("support_refs", []))
        except ProtocolError as exc:
            raise ProtocolError(f"claims[{index}].support_refs: {exc}") from exc
        if not refs:
            raise ProtocolError(f"claims[{index}].support_refs needs an input reference")
        unknown = set(refs) - allowed
        if unknown:
            raise ProtocolError(f"claims[{index}].support_refs contains unknown input IDs: {sorted(unknown)}")
        try:
            text = statement(row.get("statement"), config)
        except ProtocolError as exc:
            raise ProtocolError(f"claims[{index}].statement: {exc}") from exc
        used.update(refs)
        claims.append({"statement": text, "support_refs": refs})
    omitted = data.get("omitted_refs", [])
    if not isinstance(omitted, list):
        raise ProtocolError("invalid omission index")
    for index, row in enumerate(omitted):
        if (
            not isinstance(row, dict)
            or not isinstance(row.get("ref_id"), str)
            or row["ref_id"] not in allowed
            or not isinstance(row.get("reason"), str)
            or not row["reason"].strip()
        ):
            raise ProtocolError(f"omitted_refs[{index}] needs a known input ref_id and a nonempty reason")
        used.add(row["ref_id"])
    # Account for omissions without inventing a semantic link to any summary claim.
    passthrough = list(dict.fromkeys(u["id"] for u in units if u["id"] not in used))
    return {
        "claims": claims,
        "omitted_refs": omitted,
        "passthrough_refs": passthrough,
        "conflicts": strings(data.get("conflicts", [])),
    }


def build_tree(
    store: FactCardStore, session: ModelSession, work: dict, request: Any, *, extra_reserve: int = 0
) -> dict:
    cards = store.pages(session.config.max_facts_per_card)
    if not cards:
        return {"id": "empty", "units": [], "protected": [], "ranges": [], "conflicts": []}
    nodes = [store.leaf_unit(c, boundary=i in {0, len(cards) - 1}) for i, c in enumerate(cards)]
    config, ctx = session.config, session.context
    level = 0
    while len(nodes) > 1:
        parents, pos = [], 0
        while pos < len(nodes):
            group = nodes[pos : pos + config.merge_fan_in]

            def payload_for(children: list[dict]) -> dict:
                return {
                    "units": [u for c in children for u in c["units"]],
                    "child_ids": [c["id"] for c in children],
                    "conflicts": [x for c in children for x in c["conflicts"]],
                    "claim_limit": config.max_claims,
                    "output_language": request.output_language,
                    "question": request.question,
                }

            while len(group) > 1 and not session.fits("merge", payload_for(group)):
                group = group[:-1]
            if len(group) == 1:
                parents.extend(group)
                pos += 1
                continue
            key = identity([c["id"] for c in group])
            if key in store.state["nodes"]:
                parent = store.state["nodes"][key]
            else:
                pending_parents = len(parents) + math.ceil((len(nodes) - pos) / config.merge_fan_in)
                remaining_this_level = math.ceil(
                    (len(nodes) - pos - len(group)) / config.merge_fan_in
                )
                ctx.required_reserve = (
                    remaining_this_level
                    + merge_calls(pending_parents, config.merge_fan_in)
                    + 2
                    + extra_reserve
                )
                payload = payload_for(group)
                if not payload["units"]:
                    parent = {
                        "id": f"empty:{key}",
                        "units": [],
                        "protected": [],
                        "ranges": [r for c in group for r in c["ranges"]],
                        "conflicts": [x for c in group for x in c["conflicts"]],
                    }
                else:
                    result = session.call(
                        "merge",
                        payload,
                        parser=partial(parse_merge, units=payload["units"], config=config),
                    )
                    parent = store.commit_node(key, group, result.value, result.call_id)
                    ctx.changed()
            parents.append(parent)
            pos += len(group)
        if len(parents) >= len(nodes):
            raise NeedsSplit("cannot_pack_two_summary_nodes")
        nodes = parents
        level += 1
    work["tree_depth"] = max(work.get("tree_depth", 0), level)
    return nodes[0]


def fallback_root(store: FactCardStore, config: Any) -> dict:
    # Explicit best-effort material, never a claim that the full hierarchy was completed.
    cards = store.cards()
    facts = [f for c in cards for f in store.card_facts(c)]
    if len(facts) > config.max_claims:
        half = config.max_claims // 2
        facts = facts[:half] + facts[-(config.max_claims - half) :]
    return {
        "id": "incomplete-tree",
        "units": [store.public_fact(f) for f in facts],
        "protected": facts,
        "ranges": [c["core_interval"] for c in cards],
        "conflicts": ["hierarchy_incomplete"],
    }


def parse_draft(
    data: dict, units: list[dict], facts: list[dict], request: Any, config: Any
) -> dict:
    """Validate the answer separately from optional explanatory material."""
    if not isinstance(data, dict):
        raise ProtocolError("composer must return a JSON object")
    prediction = data.get("prediction", "")
    if request.choices:
        if not isinstance(prediction, str) or prediction.strip() not in {c.label for c in request.choices}:
            raise ProtocolError("prediction must be an original option label")
        prediction = prediction.strip()
    else:
        prediction = ""
    allowed = {u["id"] for u in [*units, *facts]}
    claims, rejected, rejected_refs, unresolved = [], [], [], []
    rows = data.get("claims", [])
    if not isinstance(rows, list):
        rejected.append({"field": "claims", "reason": "claims must be an array", "raw": rows})
        rows = []
    if len(rows) > config.max_claims:
        rejected.append({"field": "claims", "reason": "claim_limit_exceeded",
                         "omitted_count": len(rows) - config.max_claims})
    for index, row in enumerate(rows[:config.max_claims]):
        try:
            if not isinstance(row, dict):
                raise ProtocolError("claim must be an object")
            text = statement(row.get("statement"), config)
        except (ValueError, TypeError, KeyError, ProtocolError) as exc:
            rejected.append({"index": index, "reason": str(exc), "raw": row})
            continue
        refs = row.get("support_refs", [])
        if isinstance(refs, str):
            refs = [refs]
        if not isinstance(refs, list):
            rejected_refs.append({"index": index, "reason": "support_refs must be an array", "raw": refs})
            refs = []
        kept = []
        for ref in refs:
            if not isinstance(ref, str) or ref not in allowed:
                rejected_refs.append({"index": index, "reason": "reference was not supplied as evidence", "raw": ref})
            elif ref not in kept:
                kept.append(ref)
        claims.append({"id": f"answer_claim_{index + 1}", "statement": text, "support_refs": kept})
    if not request.choices and not claims:
        raise ProtocolError("free-text answer needs at least one valid statement")
    if rejected:
        unresolved.append("isolated_answer_claims")
    if rejected_refs:
        unresolved.append("isolated_answer_references")
    if not claims:
        unresolved.append("no_answer_claims")
    elif any(not c["support_refs"] for c in claims):
        unresolved.append("answer_claims_without_references")
    return {"prediction": prediction, "claims": claims, "rejected": rejected,
            "rejected_references": rejected_refs, "unresolved": unresolved}


def compose(
    store: FactCardStore, root: dict, session: ModelSession, work: dict, request: Any
) -> dict:
    if work.get("draft") is not None:
        return work["draft"]
    cards, critical = store.cards(), []
    for card in [cards[0], cards[-1]] if cards else []:
        critical.extend(store.card_facts(card))
    active = {f for c in cards for f in store.card_facts(c)}
    facts = [store.public_fact(f) for f in dict.fromkeys(critical) if f in active]
    payload = {
        "question": request.question, "spec": work["spec"], "units": root["units"], "facts": facts,
        "choices": [{"label": c.label, "text": c.text} for c in request.choices],
        "output_language": request.output_language, "length_instruction": request.length_instruction,
        "claim_limit": session.config.max_claims, "candidate_explanations": store.candidate_explanations(),
    }
    while facts and not session.fits("compose", payload):
        facts.pop(len(facts) // 2)
    if not session.fits("compose", payload):
        raise NeedsSplit("composer_context_budget")
    # Consume the terminal allowance now. Repair may use the last call if necessary.
    session.context.required_reserve = 0
    result = session.call(
        "compose", payload,
        parser=lambda d: parse_draft(d, payload["units"], facts, request, session.config),
    )
    draft = {**result.value, "call_id": result.call_id}
    work["draft"] = draft
    session.context.changed()
    return draft
