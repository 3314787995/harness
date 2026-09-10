"""Carry query-dependent facts and their images to the one visual final.

Ranking never reads options or gold answers. A packet is a presentation of fallible
evidence, not a new reduction, and cannot upgrade the program's support level.
"""
from .candidates import overlap


def build_packet(q,reduction,state,catalog,access,*,max_frames=32,max_facts=32):
    scope=reduction.get("required_scope",access.allowed if hasattr(access,"allowed") else (-float("inf"),float("inf")))
    selected=set(reduction.get("selected",[]))
    selected_rows={r for e in reduction.get("events",[]) if e["id"] in selected for r in e["rows"]}
    targets=set([q.target,*q.targets,*([q.anchor] if q.anchor else [])]) if q else set()
    raw_rows=[*state.get("candidates",[]),*state.get("uncommitted_facts",{}).values()]
    rows=[r for r in raw_rows if r.get("purpose","query")=="query"
          and (not targets or r.get("target") in targets) and overlap(r["bounds"],scope)]
    def priority(r):
        # First present active requested facts, then historical ones explicitly
        # marked superseded. Text values rank before generic action descriptions.
        tier=0 if r.get("active") else 1 if r.get("uncommitted") else 2
        focus=0 if r["id"] in selected_rows else 1
        role=0 if q and q.template=="NEIGHBOR" and r.get("target")==q.anchor else 1
        attribute=0 if r.get("value") is not None else 1
        return tier,focus,role,attribute,r["bounds"][0],r["id"]
    rows.sort(key=priority)
    # Keep one entry per fact; repeated citations do not create extra votes.
    entries=[]
    for r in rows:
        ids=list(dict.fromkeys(x for x in r["source_refs"] if x in catalog and access.permits(catalog[x]["timestamp_seconds"])))
        if not ids: continue
        entries.append({"source_row":r["id"],"target":r["target"],"decision":r["decision"],
                        "note":r["note"][:240],"value":r.get("value"),"bounds":r["bounds"],
                        "status":"unverified_candidate" if r.get("active") else "uncommitted_fact_not_event_evidence" if r.get("uncommitted") else "superseded_not_current_evidence",
                        "source_refs":ids})
    chosen=[]
    # Fair allocation across facts FIRST; a long early event cannot crowd out a
    # later anchor neighbor or selected attribute. Then add each fact's other end.
    pool=entries[:max_facts]
    for phase in (0,1):
        for e in pool:
            fid=e["source_refs"][-1 if phase==0 else 0]
            if fid not in chosen and len(chosen)<max_frames: chosen.append(fid)
    # With no facts, use chronological media coverage, never catalog insertion order.
    if not chosen:
        legal=sorted((k for k,v in catalog.items() if access.permits(v["timestamp_seconds"])),key=lambda k:catalog[k]["timestamp_seconds"])
        if legal:
            n=min(max_frames,len(legal))
            chosen=[legal[round(i*(len(legal)-1)/max(1,n-1))] for i in range(n)]
    chosen=sorted(set(chosen),key=lambda k:catalog[k]["timestamp_seconds"])
    pictured=set(chosen)
    visible=[];omitted=[]
    for e in entries:
        attached=[x for x in e["source_refs"] if x in pictured]
        if len(visible)<max_facts and attached:
            visible.append({**e,"source_refs":attached,"all_fact_frames_attached":len(attached)==len(e["source_refs"])})
        else:
            omitted.append({"source_row":e["source_row"],"reason":"final frame/fact cap; not silently treated as absent"})
    return {"frame_ids":chosen,"facts":visible,"omitted":omitted,
            "selection_policy":"query-related facts first, per-fact fair allocation; no option text used",
            "warning":"Descriptions are fallible. Superseded rows cannot establish the program result. Gaps between images were not watched."}


def packet_payload(packet,refs):
    aliases={v["id"]:k for k,v in refs.items()}
    return {"facts":[{**{k:v for k,v in f.items() if k not in {"source_row","source_refs"}},
                       "at":[aliases[x] for x in f["source_refs"] if x in aliases]}
                      for f in packet["facts"]],"omitted_fact_count":len(packet["omitted"]),"warning":packet["warning"]}
