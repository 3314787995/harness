"""Query-local, versioned visual confirmations. Completeness is not verification."""
import hashlib
import json
from .candidates import gap, overlap


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False).encode()).hexdigest()[:24]


def subjects(events, state):
    rows={r["id"]:r for r in state.get("candidates",[]) if r["active"]}
    out={}
    for e in events:
        versions=[(r,rows[r]["version"]) for r in e["rows"] if r in rows]
        lineage=sorted({x for r in e["rows"] if r in rows and rows[r].get("decision")!="attribute" for x in rows[r].get("lineage",[r])})
        out[e["id"]]={"signature":digest({"versions":versions,"event":{k:e.get(k) for k in
                        ("target","onset","offset","extent","value","clock","companions")}}),
                       "lineage":lineage,"rows":[r for r,v in versions]}
    return out


def verified(state, subject, aspects):
    proved=set()
    for proof in state.get("confirmations",[]):
        if proof.get("signature")==subject["signature"]:
            proved.update(proof["aspects"])
    return set(aspects)<=proved


def structural_aspects(event, requested, query, *, conflicts=()):
    """Necessary structural conditions, never a substitute for visual truth."""
    supported=set(requested)
    for edge in ("onset","offset"):
        bounds=event.get(edge)
        if not bounds or len(bounds)!=2 or bounds[0]>=bounds[1]: supported.discard(edge)
    if not event.get("complete"): supported.discard("completion")
    if event.get("value") is None: supported.discard("attribute")
    clock=event.get("clock")
    if not clock or any(x is None for x in clock): supported.discard("clock")
    companions=event.get("companions",{})
    if any(companions.get(t) is None for t in query.targets): supported.discard("companions")
    for issue in conflicts:
        if event["id"] in issue.get("events",[]):
            if issue["kind"]=="identity": supported.discard("identity")
            if issue["kind"]=="event_unit": supported.difference_update({"unit","identity","completion"})
    return sorted(supported)


def required_aspects(q, e, selected):
    aspects={"target","unit","identity"}
    if q.template in {"COUNT","SELECT","COOCCUR"}: aspects.add("completion")
    if q.template in {"SELECT","ORDER"}:
        aspects.update({"onset","offset"} if q.basis=="ambiguous" else {q.basis})
    if q.template=="NEIGHBOR":
        aspects.add("offset" if (e["target"]==q.anchor)==(q.op=="next_after_anchor") else "onset")
        if e["target"]==q.target: aspects.add("attribute")
    if q.template=="SELECT" and q.project!="target":
        if q.op!="nth_occurrence" or selected and e["id"]==selected[-1]: aspects.add("attribute")
    if q.group_by=="category": aspects.add("attribute")
    if q.template=="MEASURE":
        if q.time_basis!="media": aspects.add("clock")
        elif q.aggregation=="difference": aspects.add("offset" if e["target"]==q.targets[0] else "onset")
        else: aspects.update({"onset","offset"})
    if q.template=="COOCCUR": aspects.add("companions")
    if e.get("purpose")=="scope": aspects.update({"onset","offset"})
    return aspects


def identity_resolved(issue, state, mapping):
    ids=issue.get("events",[])
    signatures=sorted(mapping[x]["signature"] for x in ids if x in mapping)
    return bool(ids and len(signatures)==len(ids) and any(
        p.get("kind")=="identity" and p.get("signatures")==signatures
        for p in state.get("relation_confirmations",[])))


def gate(q, result, state):
    """Keep the mathematical estimate for scheduling, but expose no unproven value."""
    mapping=subjects(result["events"],state)
    result["candidate_estimate"]=result["value"]
    result["observation_ready"]=not any(g["kind"] not in {"attribute","zero_confirmation"} for g in result["gaps"])
    selected=set(result["selected"])
    needed=[e for e in result["events"] if e["id"] in selected or e.get("purpose")=="scope"]
    # Category count depends on each classification, including additional occurrences.
    if q.template=="COUNT" and q.group_by=="category":
        needed=[e for e in result["events"] if e["target"]==q.target and overlap(e["extent"],result["required_scope"])]
    for e in needed:
        subject=mapping[e["id"]]
        aspects=required_aspects(q,e,result["selected"])
        missing=sorted(a for a in aspects if not verified(state,subject,[a]))
        e["verification"]={"signature":subject["signature"],"required":sorted(aspects),"missing":missing}
        if missing:
            item=gap("confirmation",e["extent"],"visual confirmation required: "+", ".join(missing),
                     [e["id"]],target=e["target"],operation=q.op)
            item.update(aspects=missing,lineage=subject["lineage"])
            item["key"]=digest([q.op,"confirmation",subject["lineage"],missing])
            result["gaps"].append(item)
    if q.template=="NEIGHBOR" and q.relation=="adjacent" and len(result["selected"])==2:
        pair=[next(e for e in result["events"] if e["id"]==i) for i in result["selected"]]
        signatures=sorted(mapping[e["id"]]["signature"] for e in pair)
        if not any(p.get("kind")=="adjacency_confirmation" and p.get("signatures")==signatures for p in state.get("relation_confirmations",[])):
            a,b=pair
            edges=[a["offset" if q.op=="next_after_anchor" else "onset"],b["onset" if q.op=="next_after_anchor" else "offset"]]
            span=[min(x[0] for x in edges),max(x[1] for x in edges)] if all(edges) else result["required_scope"]
            result["gaps"].append(gap("adjacency_confirmation",span,"confirm the anchor transition, neighboring activity and intervening content",result["selected"],target=q.target,operation=q.op))
    if q.template=="COUNT":
        relevant=[e for e in needed if e["target"]==q.target]
        competing={x for g in result["gaps"] if g["kind"] in {"identity","replay"} for x in g["events"]}
        sound=[e for e in relevant if e["complete"] and e["id"] not in competing and verified(state,mapping[e["id"]],required_aspects(q,e,result["selected"]))]
        lower=len({str(e["value"]).casefold() for e in sound}) if q.group_by=="category" else len(sound)
        result["count_bounds"]={"lower":lower,"upper":result["candidate_estimate"] if not result["gaps"] else None}
    result["closed"]=not result["gaps"] and result["value"] is not None
    if not result["closed"]: result["value"]=None
    return result
