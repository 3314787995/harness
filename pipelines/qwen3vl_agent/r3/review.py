"""Bounded, local visual checks; model aliases never address arbitrary internal records."""
import json
import math
import re
from collections import Counter
from .query import VERSION, unique_keys, reject_constant
from .lines import parse_lines, marker, line_rules, frame_markers
from .candidates import ingest, overlap, events_from_rows
from .evidence import digest, subjects, required_aspects, structural_aspects

REVIEW_FIELDS={"check":"the supplied C1 alias","verdict":"confirmed|corrected|rejected|uncertain",
               "at":"actual supplied Fxx markers: prefer 2-4 representative markers; all must be supplied; cite every required segment","note":"short visible reason",
               "rows":"corrected only: complete replacement ordinary observation rows for this group"}
REVIEW_PROMPT="""R3:confirm_candidates r3-5.4
Inspect the attached numbered images. Frame labels are outside the video.
Return one JSON line for the supplied C1. The candidate text may be wrong.
The task covers ONLY the supplied local interval. Other interval descriptions are not
part of this task. Cross-edge descriptions are read-only context, not replacement rows.
Read T1/T2 literally from targets. A similar household action is not the anchor.
A new visible object/activity is a new local instance, not necessarily a visible begin.

Choose one verdict:
- confirmed: these pictures support ALL listed matters for the local candidate.
- rejected: the entire owned local group is unrelated/preparation. No replacement rows.
- corrected: replace ONLY owned local facts with up to 8 short observation rows.
  Retain their valid parts. Do not reconstruct the whole video. Do not copy substeps.
- uncertain: the local pictures cannot decide; keep the old facts and explain briefly.

Use check, verdict, at and note. rows is needed only for corrected; empty rows on other
verdicts has no meaning. Prefer 2-4 representative citations; never enumerate a video.
For confirmed, cite every supplied segment; scan coverage requires its first/last core
pictures. Only shown images can support facts. Missing boundaries remain unknown.
For a target cycle distinguish preparation, actual target motion, and reset; generic
hand/object movement does not confirm the requested action. Never invent a boundary.
For target absence, use rejected/not_target; never put the absent target on a positive
row just because it was requested. No choices, final answer, internal IDs or wrappers.
"""

REVIEW_EXAMPLES = {
    "cycles": {"input":"C1: verify one lever-pull cycle. F01 hand reaches; F02-F03 lever moves down; F04 lever returns.",
        "output":{"check":"C1","verdict":"corrected","at":["F01","F04"],"note":"The reach is preparation; the lever then moves down and returns.",
                  "rows":[{"at":["F01"],"decision":"not_target","note":"Hand reaches for stationary lever."},
                          {"at":["F02","F03","F04"],"decision":"occurrence","note":"Lever moves down and returns once."}]}},
    "instances": {"input":"C1: T1 making a clay vessel. F01-F02 shaping; F03-F04 decoration; F05-F06 finished vessel removed. F07-F08 show a different vessel already being shaped; its beginning is unseen.",
        "output":{"check":"C1","verdict":"corrected","at":["F01","F06","F08"],"note":"One vessel finishes; a different vessel is in progress.",
                  "rows":[{"at":["F01","F04"],"decision":"continue","target":"T1","instance":"I1","note":"Shaping and decorating the same vessel."},
                          {"at":["F05","F06"],"decision":"complete","target":"T1","instance":"I1","value":"Tall vase","note":"The decorated vessel is finished and removed."},
                          {"at":["F07","F08"],"decision":"continue","target":"T1","instance":"I2","note":"A different vessel is being shaped; its beginning is not shown."}]}},
    "anchors": {"input":"C1: correct this local group. T2 closing a suitcase is the anchor; T1 subsequent activity. F01-F02 closure; F03-F04 jacket starts; F05-F06 jacket ends.",
        "output":{"check":"C1","verdict":"corrected","at":["F01","F06"],"note":"Suitcase closure precedes putting on a jacket.",
                  "rows":[{"at":["F01","F02"],"decision":"complete","target":"T2","note":"Suitcase lid shuts and hands release it."},
                          {"at":["F03","F04"],"decision":"begin","target":"T1","value":"Putting on a jacket","note":"An arm enters the sleeve."},
                          {"at":["F05","F06"],"decision":"complete","target":"T1","value":"Putting on a jacket","note":"Both arms are in the jacket."}]}},
    "time": {"input":"C1: T1 treadmill activity, media-time boundaries. F01-F02 belt starts; F05-F06 belt stops.",
        "output":{"check":"C1","verdict":"corrected","at":["F01","F06"],"note":"Supplied images bracket the start and stop.",
                  "rows":[{"at":["F01","F02"],"decision":"begin","target":"T1","note":"Belt starts moving."},
                          {"at":["F05","F06"],"decision":"complete","target":"T1","note":"Belt stops moving."}]}}
}


def review_prompt(recipe):
    example=REVIEW_EXAMPLES[recipe]
    return (REVIEW_PROMPT + "\nReplacement object protocol for this recipe only:\n" +
            line_rules(recipe, require_end=False) + "\nComplete neutral example input: " +
            example["input"] + "\nOutput:\n" + json.dumps(example["output"], separators=(",",":")))



def checks_for(q, reduction, state):
    """Group actual candidates first; no replacement of a broad gap by its midpoint."""
    events={e["id"]:e for e in reduction["events"]}
    rows={r["id"]:r for r in state.get("candidates",[]) if r["active"]}
    mapping=subjects(reduction["events"],state)
    checks=[]
    priority={"anchor":0,"recognition":0,"event_unit":0,"identity":1,"order":1,"boundary":1,
              "adjacency":1,"adjacency_confirmation":1,"attribute":2,"confirmation":2,"protocol":3,"coverage":4}
    for issue in sorted(reduction["gaps"],key=lambda g:priority.get(g["kind"],2)):
        if issue["kind"] in {"query","unavailable_input"}: continue
        es=[events[x] for x in issue["events"] if x in events]
        row_ids=list(dict.fromkeys([r for e in es for r in e["rows"]]+[x for x in issue["events"] if x in rows]))
        local=[rows[x] for x in row_ids]
        if not local and issue["kind"]=="anchor":
            # A missing alias is not a missing picture. Search the actual base cores;
            # never ask a text-only/empty-context check to invent the anchor.
            scans=[s for s in state.get("coverage",[]) if not s.get("refinement") and not s.get("attribute_only")
                   and s.get("purpose","query")=="query" and overlap(s["span"],issue["span"])]
            for scan in scans:
                search_key=digest([q.op,"anchor_search",scan["span"],q.anchor])
                if search_key in state.get("review_attempts",[]): continue
                search_rows=[r for r in rows.values() if overlap(r["bounds"],scan["span"])]
                checks.append({"key":search_key,"kind":"anchor_search","reason":"Locate the reference anchor separately from other activities; correct wrong target aliases only from images.",
                    "span":scan["span"],"purpose":"query","events":[],"rows":[r["id"] for r in search_rows],
                    "lineage":sorted({x for r in search_rows for x in r.get("lineage",[r["id"]])}),
                    "subjects":{},"aspects":[],"source_refs":list(dict.fromkeys(x for r in search_rows for x in r["source_refs"])),
                    "descriptions":[{k:r.get(k) for k in ("target","decision","note","value","bounds","version","purpose")} for r in search_rows],
                    "versions":{r["id"]:r["version"] for r in search_rows}})
            continue
        if not local and issue["kind"]=="coverage":
            # A coverage defect belongs to the actual unverified core, not the entire video.
            scans=[s for s in state.get("coverage",[]) if not s["complete"] and overlap(s["span"],issue["span"]) and not s.get("attribute_only")]
            if scans: issue={**issue,"span":scans[0]["span"]}
        if not local and issue["kind"] in {"coverage","protocol"}:
            local=[r for r in rows.values() if issue["span"][0]<=r["bounds"][0]<=r["bounds"][1]<=issue["span"][1] and r.get("purpose","query")=="query"]
            row_ids=[r["id"] for r in local]
        lineage=sorted({x for r in local for x in r.get("lineage",[r["id"]])})
        key=digest([q.op,issue["kind"],lineage or issue["span"],issue["target"]])
        if issue["kind"] in {"protocol","coverage","zero_confirmation"}:
            key=digest([q.op,issue["kind"],issue["span"],issue["target"]])
        if key in state.get("review_attempts",[]): continue
        # A blocking reconstruction also asks the facts needed by the current operation.
        aspects=set(issue.get("aspects",[]))
        for e in es: aspects.update(required_aspects(q,e,reduction["selected"]))
        if issue["kind"]=="identity": aspects.add("identity")
        if issue["kind"] in {"order","boundary"} and len(es)>1:
            # Ordering is computed from each event's verified boundaries. Different
            # events do not need the same verdict or a shared replacement transaction.
            for e in es:
                erows=[rows[x] for x in e["rows"] if x in rows]
                lineage_e=sorted({x for r in erows for x in r.get("lineage",[r["id"]])})
                ekey=digest([q.op,issue["kind"],lineage_e,e["target"]])
                if ekey in state.get("review_attempts",[]) or any(set(e["rows"])&set(c["rows"]) for c in checks): continue
                checks.append({"key":ekey,"kind":issue["kind"],"reason":issue["reason"]+"; verify this event's boundaries and unit; the program computes order",
                    "span":e["extent"],"purpose":e.get("purpose","query"),"events":[e["id"]],"rows":[r["id"] for r in erows],
                    "lineage":lineage_e,"subjects":{e["id"]:mapping[e["id"]]},"aspects":sorted(required_aspects(q,e,reduction["selected"])),
                    "source_refs":list(dict.fromkeys(x for r in erows for x in r["source_refs"])),
                    "descriptions":[{k:r.get(k) for k in ("target","decision","note","value","bounds","version","purpose")} for r in erows],
                    "versions":{r["id"]:r["version"] for r in erows}})
            continue
        candidate={"key":key,"kind":issue["kind"],"reason":issue["reason"],"span":issue["span"],
                   "purpose":local[0].get("purpose","query") if local else "query",
                   "events":[e["id"] for e in es],"rows":row_ids,"lineage":lineage,
                   "subjects":{e["id"]:mapping[e["id"]] for e in es},"aspects":sorted(aspects),
                   "source_refs":list(dict.fromkeys(x for r in local for x in r["source_refs"])),
                   "descriptions":[{k:r.get(k) for k in ("target","decision","note","value","bounds","version","purpose")} for r in local],
                   "versions":{r["id"]:r["version"] for r in local}}
        if any(set(row_ids)&set(c["rows"]) for c in checks): continue
        checks.append(candidate)
    return checks


SCAN_KINDS={"coverage","protocol","anchor_search","zero_confirmation"}


def segments_for(check, media, q, scope, fps):
    points=sorted({media.catalog[x]["timestamp_seconds"] for x in check["source_refs"] if x in media.catalog})
    if not points or check["kind"] in SCAN_KINDS:
        # Exact faulty core or search range; retain ALL of it through dependent pieces.
        a,b=check["span"]
        return [{"span":[max(scope[0],a),min(scope[1],b)],"core":[max(scope[0],a),min(scope[1],b)],"required_refs":[]}]
    intervals=[]
    if check["kind"] in {"adjacency","adjacency_confirmation"}:
        points=sorted(set(points+[check["span"][0],check["span"][1]]))
    for t in points:
        a,b=max(scope[0],t-.5),min(scope[1],t+.5)
        if intervals and a<=intervals[-1][1]: intervals[-1][1]=max(intervals[-1][1],b)
        else: intervals.append([a,b])
    if check["kind"] in {"adjacency","adjacency_confirmation"}:
        intervals.append(list(check["span"]))
        merged=[]
        for a,b in sorted(intervals):
            if merged and a<=merged[-1][1]: merged[-1][1]=max(merged[-1][1],b)
            else: merged.append([a,b])
        intervals=merged
    # Real references at BOTH ends of a long event stay in separate, labelled segments.
    return [{"span":p,"core":p,"required_refs":[x for x in check["source_refs"] if x in media.catalog and p[0]<=media.catalog[x]["timestamp_seconds"]<=p[1]]} for p in intervals]


from .local_review import plan_checks


def parse_checks(raw, task, refs, q, aliases, *, truncated=False):
    valid,errors,seen=[],[],set()
    retained_facts=[]
    checks={f"C{i}":c for i,c in enumerate(task["checks"],1)}
    # Inspect the entire response before accepting any destructive replacement.
    # A later duplicate (including a malformed tail naming C1) invalidates C1.
    named=[]
    for text in raw.splitlines():
        try:
            obj=json.loads(text,object_pairs_hook=unique_keys,parse_constant=reject_constant)
            alias=obj.get("check") if isinstance(obj,dict) else None
        except (ValueError,TypeError):
            match=re.search(r'"check"\s*:\s*"(C\d+)"',text)
            alias=match.group(1) if match else None
        if isinstance(alias,str): named.append(alias)
    duplicates={k for k,n in Counter(named).items() if n>1}
    unsafe_mutation=truncated or any(k not in checks for k in named)
    for line,text in enumerate(raw.splitlines(),1):
        if not text.strip() or text.strip() in {"```","```json","```jsonl"}: continue
        try:
            r=json.loads(text,object_pairs_hook=unique_keys,parse_constant=reject_constant)
            if not isinstance(r,dict): raise ValueError("check object required")
            alias=r.get("check")
            if not isinstance(alias,str) or alias not in checks: raise ValueError("check: unknown alias")
            if alias in duplicates: raise ValueError("check: duplicate reply invalidates the entire check; old group preserved")
            if alias in seen: raise ValueError("check: duplicate reply")
            seen.add(alias)
            if r.get("verdict") not in {"confirmed","corrected","rejected","uncertain"}: raise ValueError("verdict: invalid")
            if r["verdict"]!="corrected" and "rows" in r and (r["rows"] is None or r["rows"]==[]):
                r.pop("rows")  # Empty payload has no mutation semantics.
            fields={"check","verdict","at","note"}|({"rows"} if r["verdict"]=="corrected" else set())
            if set(r)!=fields: raise ValueError("unexpected or missing check fields")
            if not isinstance(r["note"],str) or not r["note"].strip(): raise ValueError("note required")
            r["at"]=frame_markers(r.get("at"),refs)
            if len(set(r["at"]))!=len(r["at"]): raise ValueError("duplicate frame")
            if [refs[x]["timestamp_seconds"] for x in r["at"]]!=sorted(refs[x]["timestamp_seconds"] for x in r["at"]): raise ValueError("at: chronological references required")
            check=checks[alias]
            allowed={x for x,v in refs.items() if any(s["span"][0]-1e-6<=v["timestamp_seconds"]<=s["span"][1]+1e-6 for s in check["segments"])}
            if not set(r["at"])<=allowed: raise ValueError("at: reference belongs to another check")
            if r["verdict"]=="confirmed":
                for s in check["segments"]:
                    local=sorted((x for x in allowed if s["span"][0]-1e-6<=refs[x]["timestamp_seconds"]<=s["span"][1]+1e-6),key=lambda x:refs[x]["timestamp_seconds"])
                    if not local or not set(local)&set(r["at"]): raise ValueError("at: every required segment needs cited evidence")
                    if check["kind"] in SCAN_KINDS and not {local[0],local[-1]}<=set(r["at"]): raise ValueError("at: coverage confirmation requires actual first and last core frames")
            if r["verdict"]=="confirmed" and q.recipe=="cycles" and check["rows"] and len({refs[x]["source_frame_id"] for x in r["at"]})<2:
                raise ValueError("a single static frame cannot confirm an action cycle")
            if r["verdict"]=="corrected":
                if not isinstance(r["rows"],list) or not 1<=len(r["rows"])<=8: raise ValueError("rows: 1-8 complete replacement rows required")
                for index,x in enumerate(r["rows"],1):
                    if isinstance(x,dict) and x.get("decision")=="end":
                        raise ValueError(f"rows[{index}].decision: end is forbidden in replacement rows")
                local_refs={k:v for k,v in refs.items() if k in allowed}
                parsed=parse_lines("\n".join(json.dumps(x) for x in r["rows"]),q.recipe,local_refs,aliases,
                                   unit=q.unit,time_basis=q.time_basis,companions=q.targets if q.template=="COOCCUR" else (),require_end=False)
                # These are observations, not an authorized complete replacement.
                # A bad sibling row must not erase valid visual descriptions from
                # the final evidence presentation, nor retire the previous group.
                retained_facts.extend({"check":alias,"row":x,"refs":local_refs} for x in parsed["rows"])
                if parsed["errors"] or not parsed["rows"] or parsed["end"]: raise ValueError("rows: incomplete correction: "+str(parsed["errors"]))
                if any(x["decision"]=="uncertain" for x in parsed["rows"]): raise ValueError("uncertain replacement must retain the old group")
                if q.unit=="production_instance":
                    closed=set()
                    for x in parsed["rows"]:
                        target=(x["target"],x.get("instance"))
                        if x["decision"]=="begin": closed.discard(target)
                        elif x["decision"] in {"continue","complete"} and target in closed:
                            raise ValueError(f"rows[{x['line']}].decision: production continues after completion without a new begin; old group preserved")
                        elif x["decision"]=="complete": closed.add(target)
                # Local candidates may have been assigned the wrong target. All aliases
                # were validated against the frozen query; never restrict correction to
                # the very target label being checked. Unrelated internal rows stay out.
                r["parsed"]=parsed
            valid.append({**r,"line":line,"spec":check})
        except (ValueError,TypeError) as exc:
            errors.append({"path":f"line[{line}]","error":str(exc),"raw":text})
            # Unattributable malformed content could be the rest of a replacement.
            if not re.search(r'"check"\s*:\s*"C\d+"',text): unsafe_mutation=True
    if unsafe_mutation:
        kept=[]
        for item in valid:
            if item["verdict"] in {"corrected","rejected"}:
                errors.append({"path":item["check"],"error":"incomplete or unassigned response content; destructive update withheld"})
            else: kept.append(item)
        valid=kept
    for alias in checks:
        if alias not in {x["check"] for x in valid}: errors.append({"path":alias,"error":"no valid check reply"})
    if truncated: errors.append({"path":"output","error":"truncated output; complete independent checks retained"})
    accepted={x["check"] for x in valid}
    return {"checks":valid,"errors":errors,
            "retained_facts":[x for x in retained_facts if x["check"] not in accepted]}


def commit_checks(state, record, parsed, q, *, commit=None):
    facts=state.setdefault("uncommitted_facts",{})
    for f in parsed.get("retained_facts",[]):
        r=f["row"];sources=[f["refs"][x] for x in r["at"]]
        key=f"{record['id']}:{f['check']}:{r['line']}"
        facts.setdefault(key,{**r,"id":key,"call_id":record["id"],"active":False,"uncommitted":True,
                              "purpose":"query","bounds":[min(x["timestamp_seconds"] for x in sources),max(x["timestamp_seconds"] for x in sources)],
                              "source_refs":[x["id"] for x in sources],"reason":"replacement incomplete; not applied to event state"})
    commits=state.setdefault("check_commits",[])
    for item in parsed["checks"]:
        key=f"{record['id']}:{item['line']}"
        if key in commits: continue
        c=item["spec"]
        active={r["id"]:r for r in state.get("candidates",[]) if r["active"]}
        if any(x not in active or active[x]["version"]!=v for x,v in c["versions"].items()):
            state.setdefault("check_errors",[]).append({"call_id":record["id"],"check":c["key"],"reason":"candidate version changed"})
            commits.append(key)
            if commit: commit()
            continue
        parts=state.setdefault("check_parts",{}).setdefault(c["key"],{})
        parts[str(c["part"])]=dict(item,call_id=record["id"],refs=record["refs"],sampling=record["sampling"])
        commits.append(key)
        if len(parts)!=c["parts"] or c["key"] in state.setdefault("resolved_checks",[]):
            if commit: commit()
            continue
        values=[parts[str(i)] for i in range(c["parts"])]
        verdicts={v["verdict"] for v in values}
        if len(verdicts)!=1 or verdicts=={"uncertain"}:
            state.setdefault("check_errors",[]).append({"check":c["key"],"reason":"joint evidence unresolved: inconsistent or uncertain verdicts; local scan slices commit separately"})
            if commit: commit()
            continue
        if not all(v["sampling"].get("resolution_met") for v in values):
            state.setdefault("check_errors",[]).append({"check":c["key"],"reason":"required review sampling or source anchors incomplete"})
            if commit: commit()
            continue
        verdict=values[0]["verdict"]
        new_ids=[]
        if verdict in {"corrected","rejected"}:
            incoming=[]
            if verdict=="corrected":
                # Validate every piece before retiring anything. No partial replacement.
                for v in values:
                    tmp={}
                    core=[min(s["span"][0] for s in v["spec"]["segments"]),max(s["span"][1] for s in v["spec"]["segments"])]
                    call={"id":v["call_id"]+":"+c["key"],"task":{"key":"revision","recipe":q.recipe,"core":core,"last":True},"refs":v["refs"],"sampling":v["sampling"]}
                    ingest(tmp,call,v["parsed"])
                    incoming+=tmp["candidates"]
                if not incoming: continue
            roots=c["lineage"] or [c["key"]]
            for r in incoming:
                r.update(lineage=roots,review_origin=c["key"],call_id="review-group:"+c["key"])
                if c["descriptions"]: r["purpose"]=active[c["rows"][0]].get("purpose","query")
            for x in c["rows"]:
                active[x].update(active=False,version=active[x]["version"]+1,replaced_by=record["id"])
            state.setdefault("candidates",[]).extend(incoming)
            new_ids=[r["id"] for r in incoming]
            state.setdefault("revisions",[]).append({"call_id":record["id"],"check":c["key"],"retired":c["rows"],"added":new_ids,"verdict":verdict,"lineage":roots})
            if c.get("context_rows"):
                state.setdefault("local_conflicts",[]).append({"rows":c["context_rows"]+new_ids,"span":c["span"],
                    "reason":"local revision overlaps a preserved cross-slice candidate; reconcile its full identity", "call_id":record["id"]})
            state["zero_reviewed"]=False
        events,event_conflicts=events_from_rows(state.get("candidates",[]),q)
        mapping=subjects(events,state)
        touched=[e for e in events if e["id"] in c["events"]] if verdict=="confirmed" else [e for e in events if set(e["rows"])&set(new_ids)]
        evidence=list(dict.fromkeys(v["refs"][x]["id"] for v in values for x in v["at"]))
        for e in touched:
            # A local slice may update facts, but cannot certify the whole event
            # assembled using facts outside that slice or retained cross-slice rows.
            event_rows={r["id"]:r for r in state.get("candidates",[]) if r["active"]}
            full_local=all(any(s["span"][0]<=event_rows[r]["bounds"][0]<=event_rows[r]["bounds"][1]<=s["span"][1]
                               for s in c["segments"]) for r in e["rows"] if r in event_rows)
            if c.get("local_only") and not full_local:
                state.setdefault("check_errors",[]).append({"call_id":record["id"],"check":c["key"],"event":e["id"],
                    "reason":"local facts committed; whole-event verification needs its remaining evidence"})
                continue
            aspects=structural_aspects(e,c["aspects"],q,conflicts=event_conflicts)
            withheld=sorted(set(c["aspects"])-set(aspects))
            if withheld:
                state.setdefault("check_errors",[]).append({"call_id":record["id"],"check":c["key"],
                    "event":e["id"],"reason":"confirmation withheld: missing or conflicting event structure","aspects":withheld})
            # A correction with single-frame cycle rows remains a candidate.
            if q.recipe=="cycles" and len(e["source_refs"])<2 and verdict=="corrected": aspects=[]
            if not aspects: continue
            state.setdefault("confirmations",[]).append({"signature":mapping[e["id"]]["signature"],"event":e["id"],
                "aspects":aspects,"check":c["key"],"source_refs":evidence,"call_ids":[v["call_id"] for v in values]})
        if c["kind"] in {"identity","adjacency_confirmation"} and verdict in {"confirmed","corrected"}:
            state.setdefault("relation_confirmations",[]).append({"kind":c["kind"],"signatures":sorted(mapping[e["id"]]["signature"] for e in touched),"source_refs":evidence})
        qualified=all(v["sampling"].get("resolution_met") for v in values)
        if c["kind"] in SCAN_KINDS|{"adjacency","missing_target","neighbor"} and verdict in {"confirmed","corrected"} and qualified:
            for v in values:
                for s in v["spec"]["segments"]:
                    marks=sorted((x for x,f in v["refs"].items() if s["core"][0]<=f["timestamp_seconds"]<=s["core"][1]),
                                 key=lambda x:v["refs"][x]["timestamp_seconds"])
                    if not marks or not {marks[0],marks[-1]}<=set(v["at"]):
                        state.setdefault("check_errors",[]).append({"call_id":v["call_id"],"check":c["key"],
                            "reason":"local facts committed; scan boundary citations incomplete"})
                        continue
                    state.setdefault("coverage",[]).append({"call_id":v["call_id"],"span":s["core"],"complete":True,
                        "negative":c["kind"]=="zero_confirmation" and verdict=="confirmed","errors":[],"sampling":v["sampling"],"purpose":c.get("purpose","query"),
                        "targets":[q.target,*q.targets,*([q.anchor] if q.anchor else [])],"refinement":True})
            if c["kind"]=="zero_confirmation" and verdict=="confirmed":
                from .query_reduce import covers
                zero=state.setdefault("zero_scan_parts",{}).setdefault(c.get("parent_key",c["key"]),[])
                zero.extend({"span":s["core"],"complete":True} for s in c["segments"])
                state["zero_reviewed"]=covers(zero,c.get("parent_span",c["span"]))
        state["resolved_checks"].append(c["key"])
        if commit: commit()
    state.setdefault("check_errors",[]).extend({**e,"call_id":record["id"]} for e in parsed["errors"])
    state["reduction"]=None
