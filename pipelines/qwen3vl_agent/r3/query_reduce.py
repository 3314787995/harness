"""Six deterministic reducers, twelve operations, query-local proof obligations."""
from itertools import pairwise
from .candidates import gap, overlap, events_from_rows

def covers(scans, span):
    cursor = span[0]
    for s in sorted((s["span"] for s in scans if s["complete"]),key=lambda x:x[0]):
        if s[0] > cursor+1e-6: break
        if s[1] >= cursor: cursor = max(cursor,s[1])
    return cursor >= span[1]-1e-6


def union_length(intervals):
    length, end = 0.0, None
    for a,b in sorted(intervals):
        if end is None or a>end: length += max(0,b-a)
        elif b>end: length += b-end
        end = max(end if end is not None else b,b)
    return length


def ordered(events, basis):
    if any(e.get(basis) is None for e in events): return None
    seq = sorted(events,key=lambda e:e[basis][0])
    if any(a[basis][1] >= b[basis][0] for a,b in pairwise(seq)): return None
    return seq


def reduce_query(query, state, scope):
    events, problems = events_from_rows(state.get("candidates",[]),query)
    from .evidence import subjects, identity_resolved, gate
    mapping=subjects(events,state)
    for conflict in state.get("local_conflicts",[]):
        active={r["id"] for r in state.get("candidates",[]) if r["active"]}
        ids=[e["id"] for e in events if set(e["rows"])&set(conflict["rows"])]
        if ids and any(x in active for x in conflict["rows"]):
            for target in {e["target"] for e in events if e["id"] in ids}:
                problems.append(gap("identity",conflict["span"],conflict["reason"],ids,target=target))
    problems=[p for p in problems if p["kind"]!="identity" or not identity_resolved(p,state,mapping)]
    result=reduce_events(query, events, state.get("coverage",[]), scope, problems,
                         zero_reviewed=state.get("zero_reviewed",False))
    return gate(query,result,state)


def semantic_range(q, events, scope):
    """Possible and certain membership, without inventing an exact stage boundary."""
    if q.scope["kind"]!="semantic": return list(scope),list(scope)
    stages=[e for e in events if e["target"]==q.scope["description"] and e.get("purpose")!="query"]
    if len(stages)!=1 or not stages[0]["onset"] or not stages[0]["offset"]: return None
    start,end=stages[0]["onset"],stages[0]["offset"]
    if start[1]>=end[0]: return None
    possible=[max(scope[0],start[0]),min(scope[1],end[1])]
    certain=[max(scope[0],start[1]),min(scope[1],end[0])]
    if "first_sec" in q.scope:
        possible[1]=min(possible[1],start[1]+q.scope["first_sec"])
        certain[1]=min(certain[1],start[0]+q.scope["first_sec"])
    if "last_sec" in q.scope:
        possible[0]=max(possible[0],end[0]-q.scope["last_sec"])
        certain[0]=max(certain[0],end[1]-q.scope["last_sec"])
    if possible[0]>=possible[1] or certain[0]>certain[1]: return None
    return possible,certain


def reduce_events(q, events, scans, scope, problems=(), *, zero_reviewed=False):
    gaps, selected, value = [], [], None
    required = list(scope)
    certain = list(scope)
    def add(kind, reason, es=(), span=None, target=None):
        gaps.append(gap(kind,span or required,reason,[e["id"] for e in es],target=target or q.target,operation=q.op))
    if q.scope["kind"] == "semantic":
        stages = [e for e in events if e["target"]==q.scope["description"]]
        ranges=semantic_range(q,events,scope)
        if ranges is None:
            add("scope","semantic range has no unique supported boundaries",stages)
        else:
            required,certain=ranges
    relevant_targets = set((q.target,*q.targets)) | ({q.anchor} if q.anchor else set())
    scan_targets = {q.target} if q.template=="COOCCUR" else relevant_targets
    scans=[s for s in scans if s.get("purpose")!="scope" and (s.get("targets") is None or scan_targets<=set(s["targets"]))]
    active = [e for e in events if e.get("purpose")!="scope" and e["target"] in relevant_targets and overlap(e["extent"],required)]
    if q.scope["kind"]=="semantic":
        for e in active:
            membership=e["extent"]
            if not (certain[0]<=membership[0]<=membership[1]<=certain[1]):
                add("scope_membership","candidate lies near the uncertain task-scope boundary",[e],e["extent"])
    main = [e for e in active if e["target"]==q.target]
    if q.group_by=="category":
        buckets={}
        for e in main:
            if e.get("value") is None:
                add("attribute","category grouping requires the visible category",[e],e["extent"])
                continue
            buckets.setdefault(str(e["value"]).casefold(),[]).append(e)
        grouped=[]
        for candidates in buckets.values():
            if q.template=="COUNT":
                grouped.append(candidates[0])
                continue
            axis="offset" if q.basis=="offset" else "onset"
            sequence=ordered(candidates,axis)
            if sequence is None:
                add("order","cannot select an occurrence within category",candidates)
                grouped.extend(candidates)
            else:
                grouped.append(sequence[-1] if q.selection=="last_per_category" else sequence[0])
        main=grouped
    for issue in q.unresolved: add("query",issue)
    if q.time_basis=="narrative" and not any(e.get("clock") for e in main):
        add("unavailable_input","narrative time requires legitimate temporal evidence")
    if q.unit=="utterance_or_mention":
        add("unavailable_input","visual candidates cannot establish speech events; permitted aligned provider evidence required")
    def seq_for(es):
        if q.basis == "ambiguous":
            a,b = ordered(es,"onset"),ordered(es,"offset")
            if a is None or b is None or [x["id"] for x in a]!=[x["id"] for x in b]:
                add("order","start/completion orders are uncertain or disagree",es)
                return None
            return a
        out = ordered(es,q.basis)
        if out is None: add("boundary","required ordering boundaries overlap or are unknown",es)
        return out
    def projection(e):
        if q.project=="target": return e["target"]
        if q.project=="description" and q.template!="NEIGHBOR" and q.unit!="production_instance":
            return e.get("value") or e["target"]
        if e.get("value") is None:
            add("attribute","selected instance lacks the requested attribute",[e],e["extent"])
        return e.get("value")
    def complete_units(es):
        for e in es:
            if not e["complete"]:
                add("event_unit","visible fragment lacks a complete independent event unit",[e],e["extent"])
    if q.template == "COUNT":
        selected = main
        complete_units(main)
        value = len(main)
        if not main and not zero_reviewed and covers(scans,required):
            add("zero_confirmation","zero/absence needs one dedicated visual reread after the required scan")
    elif q.template == "SELECT":
        k = q.k or 1
        suffix = q.op in {"last_occurrence","last_k"}
        axis = "offset" if q.basis=="ambiguous" else q.basis
        ready = [e for e in main if e.get(axis) is not None and (q.basis!="ambiguous" or e.get("onset") is not None)]
        ready.sort(key=lambda e:e[axis][0])
        relevant = main
        if len(ready)>=k:
            tentative=ready[-k:] if suffix else ready[:k]
            cut=tentative[0][axis][0] if suffix else tentative[-1][axis][1]
            relevant=[e for e in main if (e["extent"][1]>=cut if suffix else e["extent"][0]<=cut)]
        seq = seq_for(relevant)
        if seq is not None:
            if len(seq)<k:
                add("insufficient_events","fewer events than the requested ordinal",seq)
            else:
                selected = seq[-k:] if q.op in {"last_occurrence","last_k"} else seq[:k]
                # Preserve predecessor evidence for nth; select the attribute in code.
                complete_units(selected)
                basis = "offset" if q.basis=="ambiguous" else q.basis
                if q.op in {"last_occurrence","last_k"}:
                    required[0] = selected[0][basis][0]
                else: required[1] = selected[-1][basis][1]
                values = [projection(e) for e in (selected[-1:] if q.op=="nth_occurrence" else selected)]
                value = values[-1] if q.op=="nth_occurrence" else (values if q.op in {"first_k","last_k"} else values[0])
    elif q.template == "ORDER":
        wanted = [e for e in active if e["target"] in q.targets]
        for target in q.targets:
            instances = [e for e in wanted if e["target"]==target]
            if not instances: add("missing_target","specified ordering target not located",target=target)
            if len(instances)>1 and q.selection=="unique": add("identity","ordering target occurs repeatedly",instances,target=target)
        seq = seq_for(wanted)
        if seq is not None:
            if q.selection in {"first_per_category","last_per_category"}:
                choose = {}
                for e in seq if q.selection=="last_per_category" else reversed(seq): choose[e["target"]]=e
                seq = [e for e in seq if choose[e["target"]] is e]
            selected, value = seq, [e["target"] for e in seq]
    elif q.template == "NEIGHBOR":
        anchors = [e for e in active if e["target"]==q.anchor]
        if q.anchor_selection != "unique" and anchors:
            order = ordered(anchors,"onset")
            anchors = ([order[0] if q.anchor_selection=="first" else order[-1]] if order else anchors)
        if len(anchors)!=1:
            add("anchor","anchor identity/selection remains ambiguous",anchors,target=q.anchor)
        else:
            anchor = anchors[0]
            forward = q.op=="next_after_anchor"
            edge, other = ("offset","onset") if forward else ("onset","offset")
            if anchor[edge] is None:
                add("boundary","anchor's relevant boundary is unknown",[anchor],anchor["extent"],q.anchor)
            else:
                competitors = [e for e in main if e[other] is not None and
                               (e[other][0]>anchor[edge][1] if forward else e[other][1]<anchor[edge][0])]
                unknown = [e for e in main if e[other] is None or overlap(e[other],anchor[edge])]
                if unknown: add("boundary","candidate can compete at the anchor boundary",unknown)
                order = ordered(competitors,other)
                if not order: add("neighbor","no ordered neighboring activity located",competitors)
                else:
                    picked = order[0] if forward else order[-1]
                    selected = [anchor,picked]
                    if q.relation=="adjacent":
                        required = [scope[0] if q.anchor_selection in {"unique","first"} else anchor[edge][0],
                                    scope[1] if q.anchor_selection in {"unique","last"} else picked[other][1]]
                        # Full scan establishes anchor uniqueness; the gap gets dense 8fps checking.
                        between = sorted([anchor[edge][0],picked[other][1]])
                        dense = [s for s in scans if (s.get("sampling",{}).get("requested_fps") or 0)>=8]
                        if not covers(dense,between): add("adjacency","intervening range needs dense visual coverage",selected,between)
                        value = projection(picked)
                    else:
                        selected = [anchor,*competitors]
                        value = [projection(e) for e in competitors]
    elif q.template == "MEASURE":
        selected = [e for e in active if e["target"] in (set(q.targets) or {q.target})]
        if q.selection in {"first","last"}:
            seq = seq_for(selected)
            selected = ([seq[0] if q.selection=="first" else seq[-1]] if seq else [])
        if not selected: add("missing_target","no measurable event located")
        if q.selection=="unique" and len(selected)!=1: add("identity","measurement target not unique",selected)
        durations, intervals, measured = [], [], {}
        for e in selected:
            start,end = e.get("onset"),e.get("offset")
            if q.time_basis != "media":
                c = e.get("clock")
                start,end = ([c[0],c[0]],[c[1],c[1]]) if c and None not in c else (None,None)
            if q.aggregation=="difference":
                measured[e["id"]]=(start,end)
                continue
            if not start or not end or start[1]>end[0]:
                add("boundary","duration/localization boundaries missing or conflicting",[e],e["extent"])
                continue
            intervals.append((start,end))
            measured[e["id"]]=(start,end)
            durations.append({"target":e["target"],"min_sec":max(0,end[0]-start[1]),"max_sec":end[1]-start[0]})
        if q.aggregation=="difference":
            a,b = ([e for e in selected if e["target"]==t] for t in q.targets)
            if len(a)==len(b)==1:
                end,start=measured[a[0]["id"]][1],measured[b[0]["id"]][0]
                if start and end:
                    value={"min_sec":start[0]-end[1],"max_sec":start[1]-end[0]}
                else: add("boundary","time difference needs the earlier end and later start",selected)
            else: add("identity","time difference needs two uniquely selected targets",selected)
        elif q.op=="localize_event": value = [{"onset":a,"offset":b,"time_basis":q.time_basis} for a,b in intervals]
        elif durations:
            if q.aggregation=="single":
                if len(durations)!=1: add("identity","single duration has multiple events",selected)
                else: value = durations[0]
            elif q.aggregation=="sum": value = {"min_sec":sum(d["min_sec"] for d in durations),"max_sec":sum(d["max_sec"] for d in durations)}
            elif q.aggregation=="union": value = {"min_sec":union_length([(a[1],b[0]) for a,b in intervals]),"max_sec":union_length([(a[0],b[1]) for a,b in intervals])}
            elif q.aggregation=="compare":
                candidates = [d for d in durations if all(x is d or (d["min_sec"]>x["max_sec"] if q.comparison=="longest" else d["max_sec"]<x["min_sec"]) for x in durations)]
                if len(candidates)==1: value = candidates[0]["target"]
                else: add("duration_conflict","duration intervals do not determine the comparison",selected)
    elif q.template == "COOCCUR":
        selected = main
        complete_units(main)
        stats = {t:{"present":0,"absent":0,"unknown":0,"denominator":len(main)} for t in q.targets}
        for e in main:
            for t in q.targets:
                flag = e.get("companions",{}).get(t)
                stats[t]["present" if flag is True else "absent" if flag is False else "unknown"] += 1
        value = stats
        if not main: add("missing_target","no independent primary activities")
        if any(x["unknown"] for x in stats.values()): add("cooccurrence_unknown","unknown activities remain in the denominator",main)
        if q.comparison:
            winners = [t for t,d in stats.items() if all(t==u or (d["present"]>v["present"]+v["unknown"] if q.comparison=="most" else d["present"]+d["unknown"]<v["present"]) for u,v in stats.items())]
            if len(winners)==1: value = winners[0]
            else: add("cooccurrence_unknown","frequency winner is not uniquely supported",main)
    for p in problems:
        if p.get("target") in relevant_targets and overlap(p["span"],required):
            gaps.append({**p,"operation":q.op})
    for i,scan in enumerate(scans):
        if scan.get("errors") and not scan.get("attribute_only") and overlap(scan["span"],required):
            affected=[max(required[0],scan["span"][0]),min(required[1],scan["span"][1])]
            if not covers(scans[i+1:],affected):
                add("protocol","unparsed observation rows may hide competing evidence",span=affected)
    if not covers(scans,required): add("coverage","necessary range was not completely and readably sampled")
    # Candidate count is a supported lower estimate only when its unit and identity are sound.
    closed = not gaps and value is not None
    return {"op":q.op,"template":q.template,"value":value,"closed":closed,"required_scope":required,
            "gaps":gaps,"events":events,"selected":[e["id"] for e in selected],
            "evidence_refs":list(dict.fromkeys(ref for e in selected for ref in e["source_refs"])),
            "count_bounds":({"lower":len(main) if not any(g["kind"] in {"identity","event_unit"} for g in gaps) else 0,
                             "upper":len(main) if closed else None} if q.template=="COUNT" else None)}
