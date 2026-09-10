"""Adapt the existing synthetic temporal oracle to the two-stage wire protocol."""
import copy

from qwen3vl_agent.r3.observation_contract import OBSERVATION_VERSION


def visual_report(payload):
    frames = payload["frames"]
    return {"version": OBSERVATION_VERSION,
            "summary": {"description": "Synthetic local scene supplied by the test frame store.",
                        "evidence_refs": [frames[0]["id"], frames[-1]["id"]]},
            "facts": [{"fact_id": f"V{i//4+1:02d}", "kind": "state",
                       "description": f"Synthetic visible state samples beginning at {frames[i]['timestamp_sec']}.",
                       "evidence_refs": [f["id"] for f in frames[i:i+4]]} for i in range(0, len(frames), 4)],
            "unresolved": [], "truncated": False, "crop_requests": []}


def event_report(payload, old):
    old = copy.deepcopy(old)
    events = old["events"]
    facts = payload["visual"]["facts"]
    for e in events:
        target = next(t for t in payload["query"]["targets"] if t["target_id"] == e["target_id"])
        e.setdefault("description", target["description"])
        e.setdefault("category", target["target_id"])
        refs = set()
        def collect(v):
            if isinstance(v, dict):
                for k,x in v.items():
                    if k.endswith("refs") and isinstance(x, list):
                        refs.update(r for r in x if isinstance(r,str))
                    elif isinstance(x, dict):
                        collect(x)
        collect(e)
        e["fact_refs"] = [f["fact_id"] for f in facts if refs & set(f["evidence_refs"])]
    unresolved = old.get("unresolved", [])
    return {"version": OBSERVATION_VERSION, "events": events, "truncated": old.get("truncated", False), "unresolved": unresolved,
            "fact_dispositions": [{"fact_id": f["fact_id"],
                "status": "event" if any(f["fact_id"] in e["fact_refs"] for e in events) else "unrelated",
                "event_ids": [e["local_id"] for e in events if f["fact_id"] in e["fact_refs"]],
                "reason": "The test oracle identifies the associated temporal event or a static unrelated scene."} for f in facts],
            "target_assessments": [{"target_id": t["target_id"],
                "status": "uncertain" if unresolved else "observed" if any(e["target_id"]==t["target_id"] and e["match"]!="rejected" for e in events) else "absent",
                "event_ids": [e["local_id"] for e in events if e["target_id"] == t["target_id"] and e["match"] != "rejected"],
                "reason": "Independent synthetic event oracle for this sampled window.",
                "evidence_refs": payload["visual"]["summary"]["evidence_refs"]} for t in payload["query"]["targets"]]}
