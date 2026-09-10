"""Independent observation lines; generated timestamps/IDs are never accepted."""
import json
import math
import re
from .query import VERSION, unique_keys, reject_constant

DECISIONS = {
    "cycles": {"occurrence","not_target","uncertain","replay"},
    "instances": {"begin","continue","complete","appearance","not_target","uncertain","attribute"},
    "anchors": {"begin","continue","complete","appearance","not_target","uncertain","attribute"},
    "time": {"begin","continue","complete","appearance","not_target","uncertain","attribute"},
}

def line_rules(recipe, *, require_end=True):
    extra = "" if recipe == "cycles" else (
        " Optional value is the requested visible attribute (string or finite number). "
        "Use instance=I1/I2/... to distinguish visible objects or activities WITHIN THIS CALL. "
        "Keep the same instance across its steps; use a new instance for another object/activity, "
        "even when its actual beginning was not visible. Local instance names do not link different calls. "
        "Every affirmative non-count line MUST include target using the supplied T1/T2 alias, even with one target. "
        "begin/complete use two frames bracketing the transition, or one if only one is visible; "
        "continue says these frames show the SAME still-unfinished instance as the supplied visual tail. "
        "appearance is legal ONLY for unit=appearance_episode, never for making or general activity. "
        "attribute reads a selected instance without proposing another event.")
    if recipe == "time":
        extra += (" For screen/narrative time only, optional clock=[start,end] records actually visible "
                  "readings in seconds; null denotes an unknown endpoint. Media timing comes only from frames. "
                  "For cooccurrence only, companions maps EVERY supplied companion to true/false/null. "
                  "false requires seeing that companion absent throughout this independent primary activity; "
                  "partial visibility is null. One primary activity contributes at most once per companion.")
    fields=("Required: at (1-4 supplied local frame markers), decision (" +
            ", ".join(sorted(DECISIONS[recipe])) + "), note (concrete visible reason)." + extra)
    if recipe == "cycles":
        fields += " Only at, decision and note are legal. No target, begin, continue or complete fields/states. An occurrence must show the requested action cycle, not arbitrary object motion."
    if recipe == "instances":
        fields += " complete means the whole production instance is finished, not one construction step. A new instance and its beginning are different facts: use a new instance name with continue when the beginning is unseen. Never invent begin to make the structure pass."
    limits=" No internal IDs, seconds as references, transactions, confidence, counts, or answers."
    if not require_end:
        return ("Each rows array element is one observation object. " + fields +
                " At most 8 rows. No end row or status field in replacement rows." + limits)
    return ("Each line is a separate JSON object. " + fields +
            ' Finish with exactly one line {"at":["F01","F02"],"decision":"end",'
            '"status":"complete|absent|unclear","note":"visible window description"}. '
            "complete means the supplied sample was readable and all relevant facts were reported, "
            "not that any event or the whole video is complete. absent requires an explicit visible "
            "reason that the requested target is absent. unclear must be used for insufficient visibility. "
            "The end line must cite BOTH actual first and last CORE markers and describe readability. "
            "At most 8 event lines plus end." + limits)


COMMON = """Observe only the actual images in this call. Frame labels are local to this call.
Each numbered image has a label outside the original picture. This label is NOT video content.
Check the cited picture itself: understanding the general video does not mean that Fxx
supports your description. Do not refer to a nearby or remembered frame instead.
Titles, background objects and finished-item displays belong in the end description.
For making, use begin/continue/complete; uncertain if actual production is not visible.
Describe only a necessary beginning, the last still-running state and a completion.
Do not repeat a similar folding, smoothing or adjusting step for every sampled frame.
Never infer an unseen start, finish or identity from a cut. Gaps between supplied segments
were NOT watched continuously. Text mentioning an action is not visual evidence of that action.
Separate preparation, the target action, and its result. Respect the requested event unit.
Keep an uncertain fragment instead of dropping it. Do not predict the final answer.
Candidate vocabulary is optional recognition help, never a list of activities that must appear.
An unrelated action stays not_target; never attach its description to an absent anchor.
Context and visual tails help interpretation; report count occurrences in the CORE only.
For a count, the LAST cited frame is the decisive occurrence/completion moment; earlier
references may establish context. A visible replay of earlier footage uses replay, never
another occurrence by assumption. State the concrete repeated-footage evidence in note.
For a still-running long instance, report continue at the final visible frames as well as
any observed beginning. Reuse an existing instance only when the supplied visual tail
actually shows its continuity. Preparation and substeps do not begin new production units.
No Markdown, explanation paragraphs or wrapper objects. Keep notes short and specific.
"""

EXAMPLES = {
"cycles": """Complete neutral input: unit=action_cycle, target=one lever pull, core F03-F14.
Numbered pictures: F03 hand approaches; F05-F06 lever descends; F08 it returns.
F10 shows a pause; F12-F14 show a second movement obscured before its end.
Output:
{"at":["F03"],"decision":"not_target","note":"Hand approaches the stationary lever."}
{"at":["F05","F06","F08"],"decision":"occurrence","note":"Lever descends and returns."}
{"at":["F12","F14"],"decision":"uncertain","note":"Movement begins but completion is hidden."}
{"at":["F03","F14"],"decision":"end","status":"unclear","note":"Lever visible; last movement is obscured."}
""",
"instances": """Complete neutral input: unit=production_instance, T1=making a clay vessel,
attribute=shape, core F03-F08, visual tail F01. The SAME unfinished clay body in F01
is smoothed in F03, then F05-F06 show it finished. F07-F08 show work on a DIFFERENT clay body whose beginning is not shown.
Output:
{"at":["F01","F03"],"decision":"continue","target":"T1","instance":"I1","note":"Same unfinished clay body is smoothed."}
{"at":["F05","F06"],"decision":"complete","target":"T1","instance":"I1","value":"Tall cylindrical vase","note":"Finished vessel is lifted off the work surface."}
{"at":["F07","F08"],"decision":"continue","target":"T1","instance":"I2","note":"A different clay body is being shaped; its beginning was not shown."}
{"at":["F03","F08"],"decision":"end","status":"complete","note":"Readable finishing of one vessel and work on another; their unseen beginnings remain unknown."}
""",
"anchors": """Complete neutral input: unit=activity, T2=closing a suitcase,
T1=subsequent activity, core F01-F08. F01-F03 show the suitcase closing;
F04-F05 show a pause; F06-F08 show the person putting on a jacket.
Output:
{"at":["F01","F02"],"decision":"begin","target":"T2","note":"Lid begins moving down."}
{"at":["F02","F03"],"decision":"complete","target":"T2","note":"Lid shuts; hands release it."}
{"at":["F05","F06"],"decision":"begin","target":"T1","value":"Putting on a jacket","note":"After a visible pause, an arm enters the sleeve."}
{"at":["F07","F08"],"decision":"complete","target":"T1","value":"Putting on a jacket","note":"Both arms are inside the jacket."}
{"at":["F01","F08"],"decision":"end","status":"complete","note":"Closure, pause and jacket activity are readable."}
""",
"time": """Complete neutral input: unit=activity, T1=using a treadmill, core F01-F08.
Every numbered picture shows a still, empty gym with no concealed target areas.
Output:
{"at":["F01","F04","F08"],"decision":"not_target","note":"Empty treadmill remains still; no person is present."}
{"at":["F01","F08"],"decision":"end","status":"absent","note":"The sampled gym is clear and contains no people."}
Second complete neutral input: T1=a reading session, unit=activity, time_basis=screen,
companions=music/conversation, core F01-F05. F01-F02 show the book opening; F04-F05
show it closing, with legible timer readings 10 and 18 seconds; audio is not supplied.
Output:
{"at":["F01","F02"],"decision":"begin","target":"T1","note":"Reader opens the book and follows lines."}
{"at":["F04","F05"],"decision":"complete","target":"T1","clock":[10,18],"companions":{"music":null,"conversation":false},"note":"Book closes; no conversation seen; music unknown."}
{"at":["F01","F05"],"decision":"end","status":"complete","note":"Readable session and timer; audio unavailable."}
"""}

PROMPTS = {r:f"R3:observe_{r} {VERSION}\n"+COMMON+line_rules(r)+"\n"+EXAMPLES[r] for r in DECISIONS}


def marker(value, refs):
    # Only equivalent spellings of the SAME supplied label; never nearest-frame matching.
    if not isinstance(value, str) or not re.fullmatch(r"F\d+", value, re.I):
        raise ValueError("expected local frame marker F01")
    key = f"F{int(value[1:]):02d}"
    if key not in refs:
        raise ValueError(f"unknown local frame {value}")
    return key


def frame_markers(values, refs):
    """Preserve every valid citation; only normalize equivalent spellings/duplicates.

    The prompt asks for representative frames, but verbosity is not invalid evidence.
    No nearest-frame guessing, interval expansion, sorting or citation truncation.
    """
    if not isinstance(values,list) or not 1<=len(values)<=64:
        raise ValueError("at: 1-64 supplied markers required")
    at=list(dict.fromkeys(marker(v,refs) for v in values))
    times=[refs[v]["timestamp_seconds"] for v in at]
    if times!=sorted(times): raise ValueError("at: markers must be chronological")
    return at


def parse_lines(raw, recipe, refs, targets=(), *, truncated=False, time_basis="media", companions=(),
                unit=None, core_markers=None, require_end=True):
    aliases = targets if isinstance(targets,dict) else {x:x for x in targets}
    rows, errors, ending = [], [], None
    for number, line in enumerate(raw.splitlines(), 1):
        if not line.strip() or re.fullmatch(r"```(?:jsonl?|ndjson)?", line.strip()):
            continue
        try:
            obj = json.loads(line, object_pairs_hook=unique_keys, parse_constant=reject_constant)
            if not isinstance(obj, dict): raise ValueError("object required")
            decision = obj.get("decision")
            allowed = {"at","decision","note"}
            if decision == "end": allowed.add("status")
            elif recipe != "cycles": allowed |= {"target","value","instance"}
            if recipe == "time" and decision != "end":
                if time_basis != "media": allowed.add("clock")
                if companions: allowed.add("companions")
            if set(obj) - allowed: raise ValueError(f"unknown fields {sorted(set(obj)-allowed)}")
            if not isinstance(obj.get("note"), str) or not obj["note"].strip(): raise ValueError("note: visible reason required")
            at = frame_markers(obj.get("at"), refs)
            times=[refs[v]["timestamp_seconds"] for v in at]
            if times!=sorted(times): raise ValueError("at: markers must be chronological")
            if ending is not None: raise ValueError("line follows the end status")
            if decision == "end":
                if obj.get("status") not in {"complete","absent","unclear"}: raise ValueError("invalid end status")
                if core_markers is not None and (not core_markers or not {core_markers[0],core_markers[-1]} <= set(at)):
                    raise ValueError("end.at: actual first and last CORE markers required")
                ending = {**obj,"at":at,"line":number}
                continue
            if decision not in DECISIONS[recipe]: raise ValueError("decision not allowed in this recipe")
            if "target" in obj and (not isinstance(obj["target"],str) or obj["target"] not in aliases): raise ValueError("target not in supplied aliases")
            if recipe != "cycles" and decision != "not_target" and "target" not in obj: raise ValueError("target: explicit supplied alias required")
            if decision == "appearance" and unit != "appearance_episode": raise ValueError("appearance: only applicable to appearance_episode")
            if "instance" in obj and (not isinstance(obj["instance"],str) or not re.fullmatch(r"I[1-9]\d?",obj["instance"])):
                raise ValueError("instance: local I1..I99 name required")
            if "value" in obj and (isinstance(obj["value"], bool) or not isinstance(obj["value"], (str,int,float)) or (isinstance(obj["value"],(int,float)) and not math.isfinite(obj["value"]))): raise ValueError("invalid attribute value")
            if "clock" in obj:
                c = obj["clock"]
                if not isinstance(c,list) or len(c)!=2 or any(x is not None and (isinstance(x,bool) or not isinstance(x,(int,float)) or not math.isfinite(x) or x<0) for x in c): raise ValueError("clock: two visible readings or null required")
                if all(x is not None for x in c) and c[0]>c[1]: raise ValueError("clock: reversed readings")
            if "companions" in obj and (not isinstance(obj["companions"],dict) or set(obj["companions"])!=set(companions) or any(v is not None and type(v) is not bool for v in obj["companions"].values())): raise ValueError("companions: complete boolean/null vocabulary required")
            target=aliases.get(obj.get("target"),next(iter(aliases.values()),"target") if recipe=="cycles" else "")
            rows.append({**obj,"at":at,"line":number,"target":target})
        except (ValueError, TypeError) as exc:
            errors.append({"line":number,"path":f"line[{number}]","error":str(exc),"raw":line})
    if not ending and require_end: errors.append({"path":"end","error":"missing end status"})
    if len(rows)>8: errors.append({"path":"rows","error":"more than 8 event lines; rows retained but coverage incomplete"})
    if truncated: errors.append({"path":"output","error":"truncated output; complete lines retained"})
    unresolved = any(r["decision"] == "uncertain" for r in rows)
    positive = any(r["decision"] not in {"not_target","uncertain"} for r in rows)
    if ending and ending["status"] == "absent" and positive:
        errors.append({"path":"end.status","error":"absent conflicts with positive rows"})
    valid = bool(((ending and ending["status"] in {"complete","absent"}) or not require_end) and not errors and not unresolved)
    # A status-complete empty result is not an explained negative.
    negative = valid and not positive and ((ending and ending["status"] == "absent") or any(r["decision"]=="not_target" for r in rows))
    return {"rows":rows,"errors":errors,"end":ending,"complete":valid and (positive or negative),"negative":negative}
