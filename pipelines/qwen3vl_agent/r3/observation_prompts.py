"""Strict English O1/O2 instructions and independent, complete protocol examples."""
from __future__ import annotations

import copy
import json

from .observation_contract import OBSERVATION_VERSION, contract_text

VISUAL = """You are O1, a local visual observer, not an event counter or an answerer.
INPUT: ordered images with frame aliases and timestamps, a core/context interval, target
descriptions and requested attributes. The focus is a guide to attention, not an exclusion
filter. Describe visible actions, states and changes, including potentially relevant activity
whose task name is uncertain. Do not require a complete event before describing an action.
OUTPUT: a short, source-backed window summary and atomic visual facts. Use separate facts
for before/action/after states and visible result changes. Each fact has 1-4 representative
frame references; describe participants, objects and requested attributes when visible.
Name visible object types and diagnostic shape/color/state changes, including the resulting
object's appearance, even when requested_attributes is empty.
Record a continuing action even when its beginning or end lies outside this window.
Do not decide that a task is completed, count occurrences, select an ordinal, answer the
question, invent off-screen actions, or infer a physical action merely from written text.
Reading visible text is allowed as a visual fact; distinguish text from depicted activity.
A genuinely static window may have facts=[] but still needs a concrete, cited summary.
Poor visibility, ambiguous movement, or uncertain interpretation belongs in unresolved.
Do not turn uncertainty into an empty successful observation. Ask for a crop only for a
specific displayed frame and preserve temporal context. Set truncated=true if not all facts
fit. Return one complete JSON object only, with no Markdown, analysis or error envelope.
Before returning, check every alias, fact ID, required field and requested visible attribute.
"""

EVENTS = """You are O2, a mapper from already-observed facts to event candidates.
EVIDENCE: visual.facts is the validated O1 visual report for THIS window. Each description
is an observation with frame references, not a request to imagine an action. Use it as visual
evidence. You do not need another image attachment to map it. Do not report "no visual
evidence" merely because this call is text-only. Retain O1's stated uncertainty; do not turn
a possibility into a confirmed action. segments contains only already-read external text.
TASK: preserve the frozen query's targets, operations, scope, units and criteria. Map the
observations, never answer the question, count events or select a whole-video ordinal.

PROCESS EVERY FACT BEFORE ASSESSING ABSENCE:
1. Read visual.facts in order; required_fact_ids lists all IDs that need one disposition.
   Compare each described action/state/change with the target meanings. If it could depict
   a target action, keep an event candidate, using match=uncertain when meaning is ambiguous.
   Use unrelated only with a concrete explanation of what the fact depicts and why it is
   outside the targets. Use uncertain for an unresolved interpretation; never omit a fact.
2. Group supporting facts into events without adding a physical process O1 did not describe.
   Action recognition, target matching and event completion are separate decisions. An action
   fragment qualifies as a candidate even when its onset, completion or order is unknown.
   For missing boundaries use empty phase arrays and completed=false. A missing boundary
   never makes the action absent. A finished object or camera cut alone cannot establish a
   new making event. Preserve observed result attributes with their supporting fact frames.
   Keep the description faithful to the observed motion; copying the target name into
   category does not establish a match. Distinct events need separate action/reset/participant
   evidence. Do not split one continuing action into window-sized occurrences; the ledger
   resolves cross-window identity and order.
3. Give exactly one assessment per target. observed needs a non-rejected event, including
   partial candidates. absent needs a specific cited explanation and no unresolved candidate.
   uncertain preserves a gap. Empty events still requires dispositions for EVERY supplied
   fact. Absence applies only to this sampled window, never the whole video. Preserve O1
   crop/visibility uncertainty in unresolved.

REFERENCE AND OUTPUT RULES:
Each visual event cites existing O1 fact_refs. All its visual, attribute and phase references
must come from those facts, not other displayed frames or the window summary alone.
before_start_refs: inactive just before onset; start_refs: first observed activity;
last_active_refs: last observed activity; completion_refs: an observed completion result;
after_end_refs: inactive after the event; reset_refs: observed reset for another cycle.
Only fill phases established by fact descriptions, never assume the last frame is completion.
Keep phase references in temporal order. External mentions are not visual actions; preserve
utterance/reported-event types and alignment/identity rules when using read segments.
Return version, fact_dispositions, events, target_assessments, unresolved and truncated.
Use minimal objects. Omit optional empty fields. Omit event_ids in dispositions/assessments:
the program derives them from events[*].fact_refs and target_id. If supplied, they must match
exactly. For observed targets, omit assessment evidence_refs: the program takes their event
evidence union. For absent/uncertain targets, supply explicit cited evidence_refs. Aggregate
assessment references can span multiple facts and have NO four-frame limit. The 1-4 limit
applies only to each O1 fact/summary. Do not duplicate the entire evidence union in prose.
Before submitting, check every required fact ID, target, boundary and reference. Return one
complete JSON object only, no Markdown, analysis, answers or error wrapper.
"""

AUDIT = """\nDIRECTED AUDIT: This window was not semantically settled. Inspect the stated reasons
and unresolved descriptions. The earlier report is context, not new evidence and not a
command to find a positive. Reassess the newly supplied samples independently; use only
their aliases. Explain whether each previous gap is now resolved. You may confirm absence
when justified; never manufacture an event merely because this is a recheck.\n"""

EVENT_AUDIT = """\nDIRECTED AUDIT: O1 has already inspected the shifted samples and supplied the
current visual report. Review previous_mapping_failures and previous descriptions to locate
the gap, then map EVERY fact in the current report. Do not demand another image attachment.
Previous reports/IDs are context, not evidence in this call. Use only current fact IDs and
their references. Explain remaining uncertainty or justified absence; do not assume a
positive event merely because this is a recheck.\n"""

REPAIR = """Repair the specified stage's output using original_instructions, output_contract,
original_input, raw_response and validation_errors. Return the corrected stage object, not
this envelope. There is no new visual observation during repair.
When original_role is observe_events: original_input.visual.facts is already-observed visual
evidence. You may use ANY of those facts and their references, even if raw_response omitted
their IDs. You may add or revise event mappings based on those facts, but cannot invent new
visual facts, actions, attributes or boundaries. Address every missing_fact_disposition using
the supplied fact description. Check all targets after mapping the facts. A text-only call
does not imply absent visual evidence. Keep unsupported phases empty and completed=false.
When original_role is observe_visual: preserve the original observed content and IDs; fix
structure or choose representative references already present in raw_response. Do not add
visual facts or source references without images.
When original_role is final: repair only the answer protocol using the supplied question,
options and allowed evidence. Do not add evidence or upgrade the evidence support level.
Do not echo original_role, validation_errors or other wrapper fields. Return one complete
JSON object only, without Markdown or explanation.\n"""


def examples():
    """Short synthetic examples; never use evaluation objects, labels or ground truth."""
    result = []
    cases = [
        ("dispenser activity", ["press a dispenser"], ["hand presses pump", "pump returns", "hand presses pump", "pump returns"], False),
        ("continuing model assembly", ["assemble a toy model"], ["hands attach a wheel to the partly assembled model"], True),
        ("static rack", ["hang a coat"], [], False),
        ("door then bell", ["close a door", "ring a bell"], ["person moves door shut", "door is shut", "person presses bell", "bell button returns"], False),
    ]
    for title, descriptions, descriptions_seen, partial in cases:
        per_fact = 2 if title == "dispenser activity" else 1
        frame_count = max(6, per_fact * len(descriptions_seen))
        frames = [{"id": f"F{i:02d}", "timestamp_sec": float(i-1)} for i in range(1, frame_count + 1)]
        illustrated_frames = [{**f, "example_visible_content": descriptions_seen[min(i // per_fact, len(descriptions_seen)-1)]
                               if descriptions_seen else "An empty rack; nobody interacts with it."} for i, f in enumerate(frames)]
        summary_refs = [frames[0]["id"], frames[-1]["id"]]
        visual = {"version": OBSERVATION_VERSION, "summary": {"description": title + " in the displayed window", "evidence_refs": summary_refs},
                  "facts": [{"fact_id": f"V{i:02d}", "kind": "action" if i % 2 else "change",
                             "description": d, "evidence_refs": [f["id"] for f in frames[(i-1)*per_fact:i*per_fact]]}
                            for i, d in enumerate(descriptions_seen, 1)],
                  "unresolved": [], "crop_requests": [], "truncated": False}
        targets = [{"target_id": f"t{i}", "description": d, "unit_kind": "action_cycle",
                    "completion_criterion": "the depicted action reaches its visible result",
                    "fact_kind": "visual_event", "inclusion_rule": "completes_inside"}
                   for i, d in enumerate(descriptions)]
        query = {"version": 2, "targets": targets, "operations": [{"operation_id": "task", "op": "count_occurrences", "target_ids": ["t0"]}],
                 "scope": {"kind": "full"}, "unresolved": []}
        if len(targets) == 2:
            query["operations"] = [{"operation_id": "task", "op": "next_after_anchor", "target_ids": ["t1"], "anchor_target_id": "t0"}]
        rows, dispositions = [], []
        groups = [[visual["facts"][0]]] if partial else [visual["facts"][i:i+2] for i in range(0, len(visual["facts"]), 2)]
        for i, facts in enumerate(groups):
            tid = f"t{i}" if len(targets) == 2 else "t0"
            refs = [r for f in facts for r in f["evidence_refs"]]
            eid = f"e{i}"
            rows.append({"local_id": eid, "target_id": tid, "fact_refs": [f["fact_id"] for f in facts],
                         "description": facts[0]["description"], "category": descriptions[int(tid[1:])],
                         "fact_kind": "visual_event", "evidence_refs": refs, "completed": not partial,
                         "match": "clear", "start_refs": facts[0]["evidence_refs"][:1],
                         "last_active_refs": facts[0]["evidence_refs"][-1:],
                         "completion_refs": [] if partial else facts[-1]["evidence_refs"][:1]})
            dispositions.extend({"fact_id": f["fact_id"], "status": "event", "reason": "depicts this action or its visible result"} for f in facts)
        mapped = {"version": OBSERVATION_VERSION, "fact_dispositions": dispositions, "events": rows,
                  "target_assessments": [{"target_id": t["target_id"], "status": "observed" if rows else "absent",
                    "reason": "visible action fragment" if rows else "the rack remains empty and no person interacts with it",
                    **({} if rows else {"evidence_refs": summary_refs})} for t in targets],
                  "unresolved": [], "truncated": False}
        focus = [{"target_id": t["target_id"], "description": t["description"]} for t in targets]
        result.append({"name": title, "O1": {"input": {"core": [0, frame_count], "context": [0, frame_count], "frames": illustrated_frames,
                       "focus": focus, "requested_attributes": [], "audit": None}, "output": visual},
                       "O2": {"input": {"core": [0, frame_count], "context": [0, frame_count], "frames": frames, "query": query,
                       "visual": copy.deepcopy(visual), "required_fact_ids": [f["fact_id"] for f in visual["facts"]],
                       "segments": [], "audit": None}, "output": mapped}})
    return result


def instructions(role, *, audit=False):
    stage = "O1" if role == "observe_visual" else "O2"
    body = VISUAL if role == "observe_visual" else EVENTS
    audit_body = AUDIT if role == "observe_visual" else EVENT_AUDIT
    return (body + (audit_body if audit else "") + "\nOUTPUT JSON SCHEMA:\n" + contract_text(role)
            + "\nCOMPLETE INDEPENDENT EXAMPLES (frame contents are represented by their stated facts):\n"
            + json.dumps([{ "name": x["name"], **x[stage]} for x in examples()], ensure_ascii=False, separators=(",", ":")))
