"""Three state layers: calls, local rows, derived query results. No global accepted ledger."""
import hashlib
import json

def overlap(a, b):
    return a[0] <= b[1] and b[0] <= a[1]

def ingest(state, call, parsed):
    if call["id"] in state.setdefault("committed", []):
        return
    rows = state.setdefault("candidates", [])
    task, refs = call["task"], call["refs"]
    purpose="scope" if task.get("key","").startswith("scope:") else "query"
    replace_span = task.get("replace_span")
    incoming = []
    for row in parsed["rows"]:
        sources = [refs[x] for x in row["at"]]
        bounds = [min(x["timestamp_seconds"] for x in sources), max(x["timestamp_seconds"] for x in sources)]
        # Count ownership is by the decisive (last) frame; context never votes twice.
        if task["recipe"] == "cycles" and not (task["core"][0] <= bounds[1] < task["core"][1] or task.get("last") and bounds[1] == task["core"][1]):
            continue
        incoming.append({**row,"id":f"{call['id']}:{row['line']}","call_id":call["id"],
                         "bounds":bounds,"source_refs":[x["id"] for x in sources],
                         "core":task["core"],"active":True,"version":1,"lineage":[f"{call['id']}:{row['line']}"],
                         "bind_to":task.get("selected_candidates",[]),
                         "purpose":purpose,
                         "tail_refs":task.get("tail_refs",[]),"recipe":task["recipe"]})
    retired = []
    for old in rows:
        if not old["active"] or not replace_span or not overlap(old["bounds"],replace_span):
            continue
        if old.get("purpose","query")!=purpose or (task.get("observed_targets") and old["target"] not in task["observed_targets"]):
            continue
        # A fully described local reread may replace its local interval. In partial output,
        # only an explicit negative covering the old evidence can retract that row.
        explicit = any(r["decision"] == "not_target" and r["target"]==old["target"]
                       and r["bounds"][0] <= old["bounds"][0] <= old["bounds"][1] <= r["bounds"][1] for r in incoming)
        inside = replace_span[0] <= old["bounds"][0] <= old["bounds"][1] <= replace_span[1]
        if (parsed["complete"] and inside and not task.get("attribute_only")) or explicit:
            old["active"] = False
            old["replaced_by"] = call["id"]
            old["version"] += 1
            retired.append(old["id"])
    rows.extend(incoming)
    scan = {"call_id":call["id"],"span":task["core"],"complete":parsed["complete"] and call["sampling"]["resolution_met"],
            "negative":parsed["negative"],"sampling":call["sampling"],"errors":parsed["errors"],
            "status":parsed["end"],"refinement":task.get("kind")=="review"}
    scan["attribute_only"]=bool(task.get("attribute_only"))
    scan["targets"]=task.get("observed_targets")
    scan["purpose"]=purpose
    if task.get("attribute_only"):
        scan["complete"] = False
    state.setdefault("coverage", []).append(scan)
    state.setdefault("revisions", []).append({"call_id":call["id"],"retired":retired,"added":[r["id"] for r in incoming]})
    state["committed"].append(call["id"])
    state["reduction"] = None


def gap(kind, span, reason, events=(), *, target="", operation=""):
    # No mutable IDs in the retry key. Substitution/re-ID cannot renew the allowance.
    semantic = [operation, kind, target, [round(float(x),3) for x in span]]
    return {"key":hashlib.sha256(json.dumps(semantic).encode()).hexdigest()[:20],
            "kind":kind,"span":list(span),"reason":reason,"events":list(events),
            "target":target,"operation":operation}


def events_from_rows(rows, query):
    rows = sorted((r for r in rows if r["active"]), key=lambda r:(r["bounds"][0],r["call_id"],r["line"]))
    events, pending, problems, attributes = [], {}, [], []
    local_instances = {}
    last_completed={}
    for r in rows:
        target, decision = r["target"], r["decision"]
        if decision == "not_target": continue
        if decision == "replay":
            if query.count_replays is False: continue
            if query.count_replays is None:
                problems.append(gap("replay",r["bounds"],"replayed occurrence has no public counting rule",[r["id"]],target=target))
                continue
            decision = "occurrence"
        if decision == "attribute":
            attributes.append(r)
            continue
        if decision == "uncertain":
            problems.append(gap("recognition",r["bounds"],r["note"],[r["id"]],target=target))
            continue
        def new():
            return {"id":r["id"],"target":target,"onset":None,"offset":None,
                    "purpose":r.get("purpose","query"),
                    "extent":list(r["bounds"]),"value":r.get("value"),"source_refs":list(r["source_refs"]),
                    "rows":[r["id"]],"last_call":r["call_id"],"complete":False,
                    "clock":r.get("clock"),"companions":r.get("companions",{}),
                    "instance":r.get("instance")}
        if decision in {"occurrence","appearance"}:
            e = new()
            e.update(onset=list(r["bounds"]),offset=list(r["bounds"]),complete=True)
            if decision == "appearance" and query.unit == "production_instance" and target == query.target:
                e["complete"] = False
                problems.append(gap("event_unit",r["bounds"],"appearance does not prove a production instance",[e["id"]],target=target))
            events.append(e)
            continue
        opened = pending.setdefault((target,r.get("purpose","query")), [])
        local_key=(r["call_id"],target,r.get("instance")) if r.get("instance") else None
        if decision == "begin":
            if local_key and local_key in local_instances:
                problems.append(gap("event_unit",r["bounds"],"repeated beginning for the same local instance",
                                    [local_instances[local_key]["id"]],target=target))
                # Keep the fact, but do not manufacture a second instance.
                local_instances[local_key]["rows"].append(r["id"])
                continue
            last_completed.pop((target,r.get("purpose","query")),None)
            e = new()
            e["onset"] = list(r["bounds"]) if len(r["source_refs"])>1 else None
            events.append(e)
            opened.append(e)
            if local_key: local_instances[local_key]=e
            continue
        viable = [e for e in opened if e["last_call"]==r["call_id"] or set(e["source_refs"]) & set(r["tail_refs"])]
        if local_key:
            if local_key in local_instances:
                viable=[local_instances[local_key]]
            else:
                # Different local names never merge just because a target matches.
                # Across calls only an actually re-supplied visual tail can link them.
                viable=[e for e in opened if e["last_call"]!=r["call_id"] and set(e["source_refs"]) & set(r["tail_refs"])]
        if len(viable) != 1:
            e = new()
            events.append(e)
            previous=last_completed.get((target,r.get("purpose","query")))
            if not local_key and query.unit=="production_instance" and previous and (
                    previous["last_call"]==r["call_id"] or set(previous["source_refs"]) & set(r["tail_refs"])):
                problems.append(gap("event_unit",[previous["extent"][0],r["bounds"][1]],
                    "production continues after completion without evidence of a new instance",[previous["id"],e["id"]],target=target))
            if decision == "continue": opened.append(e)
            if len(opened)>1:
                problems.append(gap("identity",r["bounds"],"multiple possible unfinished instances",[x["id"] for x in opened],target=target))
        else:
            e = viable[0]
            if e["complete"]:
                problems.append(gap("event_unit",r["bounds"],"finished local instance receives more production facts",[e["id"]],target=target))
            e["extent"] = [min(e["extent"][0],r["bounds"][0]),max(e["extent"][1],r["bounds"][1])]
            e["rows"].append(r["id"])
            e["source_refs"] = list(dict.fromkeys(e["source_refs"]+r["source_refs"]))
            e["last_call"] = r["call_id"]
        if local_key: local_instances[local_key]=e
        for key in ("value","clock","companions"):
            if key in r:
                if key == "companions":
                    for name, value in r[key].items():
                        old = e[key].get(name)
                        e[key][name] = True if old is True or value is True else value
                else: e[key] = r[key]
        if decision == "complete":
            e["offset"], e["complete"] = (list(r["bounds"]) if len(r["source_refs"])>1 else None), True
            last_completed[(target,r.get("purpose","query"))]=e
            if e in opened: opened.remove(e)
    # Attribute reads can sort before a beginning at the same PTS; attach only after
    # reconstructing the local instances, using the program's selection when provided.
    for r in attributes:
        matches=[e for e in events if e["target"]==r["target"] and
                 (e["id"] in r["bind_to"] if r.get("bind_to") else overlap(e["extent"],r["bounds"]))]
        if len(matches)==1 and "value" in r:
            matches[0]["value"]=r["value"]
            matches[0]["rows"].append(r["id"])
            matches[0]["source_refs"]=list(dict.fromkeys(matches[0]["source_refs"]+r["source_refs"]))
        else:
            problems.append(gap("attribute",r["bounds"],"attribute has no unique current local instance",target=r["target"]))
    # Actual competition only: overlapping occurrence evidence or concurrent unfinished instances.
    atomic = [e for e in events if len(e["rows"])==1 and e["complete"]]
    for i, a in enumerate(atomic):
        for b in atomic[i+1:]:
            if a["target"]==b["target"] and a.get("purpose")==b.get("purpose") and overlap(a["extent"],b["extent"]):
                problems.append(gap("identity",[min(a["extent"][0],b["extent"][0]),max(a["extent"][1],b["extent"][1])],
                                    "overlapping candidate evidence may represent the same occurrence",[a["id"],b["id"]],target=a["target"]))
    return events, problems
