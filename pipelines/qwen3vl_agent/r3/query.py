"""Small, public query contract. No gold data and no generated execution code."""
from dataclasses import dataclass, asdict, field
import json
import re
from jsonschema import Draft202012Validator
from .types import OPERATIONS, ProtocolError

VERSION = "r3-5.4"
TEMPLATES = {"count_occurrences":"COUNT", "order_events":"ORDER",
             "next_after_anchor":"NEIGHBOR", "previous_before_anchor":"NEIGHBOR",
             "localize_event":"MEASURE", "event_duration":"MEASURE",
             "cooccurrence_frequency":"COOCCUR"}
TEMPLATES.update({x:"SELECT" for x in OPERATIONS - TEMPLATES.keys()})
S = {"type":"string", "minLength":1}
POS = {"type":"integer", "minimum":1}
SCHEMA = {"type":"object", "additionalProperties":False, "required":["op","target","unit"], "properties":{
    "op":{"enum":sorted(OPERATIONS)}, "target":{**S,"description":"Event to find or measure, NOT the reference anchor."},
    "unit":{"enum":["action_cycle","production_instance","appearance_episode","state_transition","activity","utterance_or_mention"]},
    "basis":{"enum":["onset","offset","ambiguous"]},
    "time_basis":{"enum":["media","screen","narrative"]}, "k":POS,
    "targets":{"type":"array","items":S,"uniqueItems":True,"minItems":1},
    "anchor":{**S,"description":"Independent reference event locating the before/after relation."}, "anchor_selection":{"enum":["unique","first","last"]},
    "relation":{"enum":["adjacent","after","before"]}, "project":S,
    "selection":{"enum":["all","unique","first","last","first_per_category","last_per_category"]},
    "group_by":{"enum":["occurrence","category"]},
    "count_replays":{"type":"boolean"},
    "aggregation":{"enum":["single","sum","union","compare","difference"]},
    "comparison":{"enum":["longest","shortest","most","least"]},
    "scope":{"type":"object","additionalProperties":False,"required":["kind"],"properties":{
        "kind":{"enum":["full","interval","semantic"]},
        "interval":{"type":"array","minItems":2,"maxItems":2,"items":{"type":"number","minimum":0}},
        "description":S, "source":S, "first_sec":{"type":"number","exclusiveMinimum":0},
        "last_sec":{"type":"number","exclusiveMinimum":0}}},
    "unresolved":{"type":"array","items":S}}}

@dataclass(frozen=True)
class QuerySpec:
    op: str
    target: str
    unit: str
    basis: str = "ambiguous"
    time_basis: str = "media"
    k: int | None = None
    targets: tuple[str, ...] = ()
    anchor: str | None = None
    anchor_selection: str = "unique"
    relation: str = "adjacent"
    project: str = "description"
    selection: str = "all"
    group_by: str = "occurrence"
    count_replays: bool | None = None
    aggregation: str = "single"
    comparison: str | None = None
    scope: dict = field(default_factory=lambda:{"kind":"full"})
    unresolved: tuple[str, ...] = ()

    @property
    def template(self):
        return TEMPLATES[self.op]

    @property
    def recipe(self):
        if self.template == "COUNT" and self.group_by == "occurrence" and self.unit in {"action_cycle","state_transition"}:
            return "cycles"
        if self.template in {"NEIGHBOR","ORDER"}:
            return "anchors"
        if self.template in {"MEASURE","COOCCUR"}:
            return "time"
        return "instances"

    def to_dict(self):
        return asdict(self)

    def observer_view(self):
        # No ordinal, current reduction, count alternatives or option labels.
        return {"target":self.target, "unit":self.unit, "targets":list(self.targets),
                "anchor":self.anchor, "attribute":self.project, "time_basis":self.time_basis,
                "scope_description":self.scope.get("description"), "recipe":self.recipe}


def public_policy(request):
    return {k:v for k,v in request.benchmark_policy.items() if k in {
        "public_rules","event_unit","count_replays","temporal_basis","time_basis",
        "duration_option_intervals","temporal_option_intervals"}}


def validate_query(value, request):
    if isinstance(value, QuerySpec):
        value = value.to_dict()
        # Internal defaults are not model declarations of irrelevant parameters.
        value = {k:v for k,v in value.items() if v is not None}
        for key in ("targets","unresolved"):
            if isinstance(value.get(key),tuple): value[key]=list(value[key])
        if value["op"] not in {"order_events","cooccurrence_frequency"} and not value.get("targets"):
            value.pop("targets",None)
        for k in ("k","anchor","anchor_selection","relation","aggregation","comparison","targets"):
            if k not in applicable(value["op"]):
                value.pop(k, None)
    if not isinstance(value, dict):
        raise ProtocolError("$: expected a query object")
    value = dict(value)
    allowed = applicable(value.get("op"))
    for k in list(value):
        if value[k] is None and k not in {"op","target","unit"}:
            value.pop(k)
        elif k in {"k","anchor","anchor_selection","relation","aggregation","comparison","targets"} and k not in allowed:
            raise ProtocolError(f"$.{k}: parameter is not applicable")
    errors = sorted(Draft202012Validator(SCHEMA).iter_errors(value), key=lambda e:str(e.path))
    if errors:
        raise ProtocolError("; ".join("$."+".".join(map(str,e.path))+": "+e.message for e in errors))
    op = value["op"]
    if op in {"first_k","last_k","nth_occurrence"} and "k" not in value:
        raise ProtocolError("$.k: explicit positive integer required")
    if "k" in value and type(value["k"]) is not int:
        raise ProtocolError("$.k: integer required; coercion is forbidden")
    if TEMPLATES[op] == "NEIGHBOR" and (not value.get("anchor") or value["anchor"] == value["target"]):
        raise ProtocolError("$.anchor: independent anchor required")
    if op in {"order_events","cooccurrence_frequency"} and not value.get("targets"):
        raise ProtocolError("$.targets: complete target vocabulary required")
    if value.get("aggregation") == "compare" and value.get("comparison") not in {"longest","shortest"}:
        raise ProtocolError("$.comparison: duration comparison direction required")
    if op=="event_duration" and "comparison" in value and value.get("aggregation")!="compare":
        raise ProtocolError("$.comparison: not applicable outside duration comparison")
    if "count_replays" in value and value["count_replays"] != public_policy(request).get("count_replays"):
        raise ProtocolError("$.count_replays: public replay-counting rule required")
    if value.get("aggregation") == "difference" and len(value.get("targets", [])) != 2:
        raise ProtocolError("$.targets: time difference requires two ordered targets")
    if op == "cooccurrence_frequency" and value.get("comparison") not in {None,"most","least"}:
        raise ProtocolError("$.comparison: expected most/least")
    scope = value.get("scope", {"kind":"full"})
    fields = {"full":{"kind"},"interval":{"kind","interval","source"},
              "semantic":{"kind","description","source","first_sec","last_sec"}}[scope["kind"]]
    if set(scope) - fields:
        raise ProtocolError("$.scope: inapplicable scope parameters")
    if scope["kind"] != "full":
        needed = "interval" if scope["kind"] == "interval" else "description"
        if needed not in scope:
            raise ProtocolError(f"$.scope.{needed}: required")
        if scope["kind"] == "interval" and scope["interval"][0] >= scope["interval"][1]:
            raise ProtocolError("$.scope.interval: empty or reversed interval")
        source = scope.get("source", "")
        original = request.question + "\n" + json.dumps(public_policy(request), ensure_ascii=False)
        if not source or source.casefold() not in original.casefold():
            raise ProtocolError("$.scope.source: must quote the original question or public rule")
        if scope["kind"] == "interval" or "first_sec" in scope or "last_sec" in scope:
            mentioned = {float(x) for x in re.findall(r"\d+(?:\.\d+)?",source)}
            for n,word in enumerate("zero one two three four five six seven eight nine ten".split()):
                if re.search(r"\b"+word+r"\b",source,re.I): mentioned.add(float(n))
            if re.search(r"minutes?|分钟",source,re.I): mentioned |= {x*60 for x in mentioned}
            limits = scope.get("interval",[]) + [scope[k] for k in ("first_sec","last_sec") if k in scope]
            if any(float(v) not in mentioned | {0.0} for v in limits):
                raise ProtocolError("$.scope: seconds must be traceable to the quoted numeric restriction")
    for k in ("targets","unresolved"):
        if k in value:
            value[k] = tuple(value[k])
    return QuerySpec(**value)


def applicable(op):
    out = set()
    if op in {"first_k","last_k","nth_occurrence"}: out.add("k")
    if op in {"next_after_anchor","previous_before_anchor"}: out.update(("anchor","anchor_selection","relation"))
    if op in {"order_events","cooccurrence_frequency","event_duration"}: out.add("targets")
    if op == "event_duration": out.update(("aggregation","comparison"))
    if op == "cooccurrence_frequency": out.add("comparison")
    return out


def rule_query(request):
    """Conservative reusable grammar. Ambiguous language goes to the one parser call."""
    q = request.question.strip()
    # Full matches only; compound clauses and time restrictions use the single parser.
    ambiguous = re.search(r"\b(?:during|within|until|seconds|minutes?|stage|either|or)\b|\b(?:\d+|one)\s+second\b|;",q,re.I)
    ordinal = r"(first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|[1-9]\d*(?:st|nd|rd|th))"
    m = re.fullmatch(r"(?:What|Which) (?:is|was) the "+ordinal+r" ([\w -]+?) (?:made|produced|created)(?: in (?:the|this) video)?\?",q,re.I)
    parts=(m[1],m[2]) if m else None
    if not parts:
        m = re.fullmatch(r"(?:What|Which) ([\w -]+?) (?:is|was) (?:made|produced|created) "+ordinal+r"(?: in (?:the|this) video)?\?",q,re.I)
        if m: parts=(m[2],m[1])
    if parts and not ambiguous:
        word,obj=parts
        words="first second third fourth fifth sixth seventh eighth ninth tenth".split()
        k=words.index(word.lower())+1 if word.lower() in words else int(re.match(r"\d+",word)[0])
        return {"op":"nth_occurrence","target":"making "+obj,"unit":"production_instance",
                "basis":"ambiguous","k":k,"project":obj+" type"}
    m = re.fullmatch(r"(After|Before) ([^,?]+),?\s+what (?:did|does) (?:he|she|they|the (?:person|man|woman)) do (next|previously|before that)\?",q,re.I)
    if m and not ambiguous and not re.search(r"\b(?:and|then|after|before)\b",m[2],re.I):
        forward=m[1].lower()=="after"
        if (forward and m[3].lower()=="next") or (not forward and m[3].lower()!="next"):
            return {"op":"next_after_anchor" if forward else "previous_before_anchor",
                    "target":"subsequent activity" if forward else "preceding activity",
                    "anchor":m[2].strip(),"unit":"activity","relation":"adjacent","project":"description"}
    if not re.search(r"\b(after|before|during|first|last|seconds?|minutes?|stage)\b", q, re.I):
        m = re.fullmatch(r"How many times (?:did|does) (.+?)\?", q, re.I)
        if m:
            return {"op":"count_occurrences", "target":m[1], "unit":"action_cycle"}
    return None


def candidate_vocabulary(request, query=None):
    numeric = re.compile(r"^(?:\d+(?:\.\d+)?|zero|one|two|three|four|five|six|seven|eight|nine|ten)(?:\s+times?)?\.?$",re.I)
    texts = [c.text for c in request.choices]
    if (query and query.template == "COUNT") or any(numeric.fullmatch(t.strip()) for t in texts):
        return []
    return texts  # every option, unchanged and without labels, or no vocabulary


def parse_object(text):
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    return json.loads(text, object_pairs_hook=unique_keys, parse_constant=reject_constant)


def unique_keys(pairs):
    result={}
    for k,v in pairs:
        if k in result: raise ValueError(f"duplicate field: {k}")
        result[k]=v
    return result


def reject_constant(value):
    raise ValueError(f"non-finite JSON constant: {value}")


QUERY_PROMPT = """R3:query_spec r3-5.4
Extract the requested temporal query, without answering or claiming to have watched video.
Return one compact JSON object. Only the applicable parameters are allowed. No explanations.
Preserve event unit: an entire item being made is one production_instance, not each step
and not its first display. For 'made' without a start/completion rule use basis=ambiguous.
Total count, nth, first/last and first/last k are distinct. k is 1-based and explicit.
After/before means a relation, not necessarily adjacency; next/just after/previous means
adjacent. Use a separate anchor and preserve its selection rule; unique if unspecified.
The target is what must be found. The anchor is the independently described reference
activity, never copy it into target. Ordinary daily activities use unit=activity;
production_instance means making an identifiable product, not performing any action.
Use media time only for source-video timing; clock readings use screen, story time narrative.
Never invent a duration bin or cutoff. Every non-full scope must quote its source exactly.
If a stage's first seconds are requested use semantic scope with first_sec, not video zero.
Do not infer unseen mechanics or prefill a count. Unresolved meanings go in unresolved.
Example input: What is the fourth ceramic ornament made? No public temporal rule.
Example output: {"op":"nth_occurrence","target":"making a ceramic ornament","unit":"production_instance","basis":"ambiguous","k":4,"project":"ornament type"}
Example input: After closing the suitcase, what did the person do next?
Example output: {"op":"next_after_anchor","target":"subsequent activity","anchor":"closing the suitcase","unit":"activity","relation":"adjacent","project":"description"}
Example input: Before opening the gate, what did she do previously?
Example output: {"op":"previous_before_anchor","target":"preceding activity","anchor":"opening the gate","unit":"activity","relation":"adjacent","project":"description"}
Field schema (optional fields may be omitted):
""" + json.dumps(SCHEMA, separators=(",",":"))
