"""Role-specific prompts built from the same small contracts used on acceptance."""
from __future__ import annotations
import json
from dataclasses import replace
from .collection_contracts import (compile_schema, conditions, envelope_schema, record_schema,
                                  task_schema, check_schema, validate, COEXISTING_SCHEMA, DISTINCT_SCHEMA, snapshot_schema,
                                  UNIT_EQUIVALENCE, parse_compile, parse_compile_json, choice_value_schema)
from .types import SetSpec, R4Request
from .prompts import VERSION


def dumps(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def compile_example_cases():
    """Hypothetical task definitions, never observations or known answers."""
    def case(name, question, namespace, target, unit, predicate, *, op="count_unique", scope=None, **extra):
        response = {"sets": [{"set_id": "items", "namespace": namespace, "target": target,
                              "count_unit": unit, "predicate": predicate, **extra}],
                    "operations": [{"operation_id": "result", "op": op, "inputs": ["items"]}]}
        if scope is not None:
            response["scope"] = scope
        expected = {"namespace": namespace, "target": target, "count_unit": unit,
                    "equivalence": extra.get("equivalence", UNIT_EQUIVALENCE[namespace]),
                    "op": op, "scope": scope or {"kind": "full"}}
        return {"name": name, "question": question, "response": response, "expected": expected}
    return [
        case("distinct_tools", "How many separate wrenches appear throughout the video?",
             "physical_instance", "wrench", "individual wrench", "visually present"),
        case("fruit_categories", "How many different kinds of fruit appear?",
             "semantic_category", "fruit", "fruit kind", "visually present"),
        case("missing_fruit", "Which of apple, pear and orange is not shown?",
             "semantic_category", "fruit", "fruit kind", "visually present",
             op="missing_members", candidates=["apple", "pear", "orange"]),
        case("target_frame", "How many wrenches are visible at source time 12 seconds?",
             "physical_instance", "wrench", "individual wrench", "visually present",
             scope={"kind": "frame", "timestamp_sec": 12}),
        case("bounded_interval", "How many wrenches appear between source seconds 2 and 6?",
             "physical_instance", "wrench", "individual wrench", "visually present",
             scope={"kind": "interval", "interval": [2, 6]}),
        case("semantic_phase", "How many wrenches appear during the first packing scene?",
             "physical_instance", "wrench", "individual wrench", "visually present",
             scope={"kind": "semantic", "description": "packing scene", "selection": "first", "result_kind": "interval"}),
        case("literal_codes", "How many distinct printed product codes are displayed?",
             "text_value", "printed product code", "literal product code", "printed in the image",
             evidence_relation="text_present", required_modalities=["screen_text"]),
        case("attribute_combinations", "How many different fruit-kind and color combinations appear?",
             "semantic_category", "fruit", "fruit-kind/color pair", "visually present",
             equivalence="combination", attribute_keys=["fruit_kind", "color"]),
        case("planned_history", "Which shopping items does the speaker explicitly plan to buy in the supplied subtitle history?",
             "task_item", "shopping item", "planned shopping item", "explicitly planned to be bought",
             op="list_members", scope={"kind": "history"}, predicate_kind="task",
             evidence_relation="planned", required_modalities=["subtitle"]),
    ]


def validate_compile_example(case):
    req = R4Request(case["question"], video_path="format-demo.mp4",
                    available_modalities=("video", "screen_text", "subtitle"))
    spec = parse_compile(parse_compile_json(dumps(case["response"])), req)
    expected, target = case["expected"], spec.sets[0]
    for key in ("namespace", "target", "count_unit", "equivalence"):
        if getattr(target, key) != expected[key]:
            raise ValueError(f"Compile format example {case['name']} contradicts its task: {key}")
    if spec.operations[-1].op != expected["op"] or spec.scope != expected["scope"]:
        raise ValueError(f"Compile format example {case['name']} has the wrong operator or scope")
    return spec


def _contract_lines(schema, path="$", required=True):
    """Describe legal fields from the validator's schema, outside copyable JSON."""
    details = ["required" if required else "optional"]
    if "type" in schema:
        details.append(str(schema["type"]))
    if "enum" in schema:
        details.append("one of " + dumps(schema["enum"]))
    if "const" in schema:
        details.append("exact value " + dumps(schema["const"]))
    for k, label in (("minItems", "minimum array length"), ("minLength", "minimum string length")):
        if k in schema:
            details.append(f"{label} {schema[k]}")
    if schema.get("additionalProperties") is False:
        details.append("no other keys")
    if "default" in schema:
        details.append("host default " + dumps(schema["default"]))
    lines = [path + ": " + "; ".join(details)]
    for key, value in schema.get("properties", {}).items():
        lines.extend(_contract_lines(value, path + "." + key, key in schema.get("required", [])))
    if "items" in schema:
        lines.extend(_contract_lines(schema["items"], path + "[]"))
    if isinstance(schema.get("additionalProperties"), dict):
        lines.extend(_contract_lines(schema["additionalProperties"], path + ".<name>"))
    for index, branch in enumerate(schema.get("oneOf", [])):
        lines.extend(_contract_lines(branch, path + f" (alternative {index+1})"))
    for rule in schema.get("allOf", []):
        condition, consequence = rule.get("if", {}), rule.get("then", {})
        if condition and consequence:
            lines.append(f"{path}: when " + dumps(condition.get("properties", {})) + ", require " + dumps(consequence.get("required", [])))
    return lines


def compile_instructions():
    schema = compile_schema()
    text = ("Define the task before seeing video. Collections describe WHAT to inspect later, not observed members. "
        "No video evidence is needed for this definition. Return a compact JSON task with sets and operations DIRECTLY AT ROOT. "
        "Do not return r4/compile wrappers or repeat keys. Do not copy INPUT metadata into the output. "
        "query_scope and execution_subtype constrain your task but are not output fields.\n"
        "Choose the member unit from the question: physical_instance = separate real objects, repeated views count once; "
        "semantic_category = kinds/types/classes, including types of animals or observed activities; "
        "text_value = exact literal strings; task_item = explicit planned/completed history backed by text, not ordinary visible activities. "
        "A common-noun target does not imply category counting: 'How many tools appear?' counts individual tools; "
        "'How many kinds of tools appear?' counts tool categories. "
        "Use count_unique for counts, on the appropriate namespace. count_unit is the named unit you are counting, not its unknown quantity. "
        "For missing_members, collect categories that ARE shown and specify candidate names; never put global absence in predicate.\n"
        "Required fields for each set: " + ", ".join(schema["properties"]["sets"]["items"]["required"]) + ". "
        "Ordinary equivalence is supplied by the host: " + dumps(UNIT_EQUIVALENCE) + ". OMIT equivalence unless the task needs combination; "
        "then provide equivalence=combination and the nonempty attribute_keys. An explicit conflicting equivalence is rejected. "
        "membership is optional: its condition names map to required facts, and cannot replace target/predicate. "
        "Preserve population, motion and modality requirements when the question requires them. "
        "Preserve made/completed/performed conditions in predicate or membership; visible alone is insufficient. "
        "For categories name the classification axis in count_unit (what kind of thing is distinguished), not a generic display title.\n"
        "OMIT fields that use host defaults: version, ordinary equivalence, empty membership/normalization/unresolved, full scope, output_id. "
        "Output version, if supplied, is the integer data version 5, not the protocol string. "
        "Only change normalization using supplied public policy; class synonyms belong to later evidence mapping. "
        "Special scope is an object with kind; e.g. kind=frame needs timestamp_sec, interval needs interval, semantic needs description. "
        "result_kind belongs inside semantic scope. Preserve first/last/all and distinguish source time from clocks visible in images. "
        "An omitted/empty set scope inherits the query scope. Do not invent times for semantic scenes. "
        "owner and task_id keep nullable meaning; null is not an empty array or object.\n"
        "operations is a nonempty dependency-ordered array: operation_id, op, inputs. Inputs name sets or earlier operations; "
        "last operation is the result unless output_id is supplied. Set algebra must use consistent units and population. "
        "If options have complex meanings, provide choice_values for their original labels using the typed formats below; "
        "never select an option or infer the answer. Bare numbers and number words are parsed by the host.\n"
        "FIELD CONTRACT (descriptions, NOT output data):\n" + "\n".join(_contract_lines(schema)) + "\n"
        "TYPED CHOICE CONTRACT (only when an option needs it):\n" + "\n".join(_contract_lines(choice_value_schema(), "choice_values.<label>")) + "\n")
    for case in compile_example_cases():
        validate_compile_example(case)
        text += ("FORMAT DEMO " + case["name"] + "; hypothetical question: " + case["question"] +
                 " This is a task definition, not facts or answers for the current video.\n```json\n" + dumps(case["response"]) + "\n```\n")
    return text


def example_cases(role, targets):
    if role not in {"discover_candidates", "inspect_existing"}:
        return []
    cases = [("empty", None, {"records": [], "checks": [], "coverage": "complete", "overflow": False, "gaps": []})]
    if any(t.candidates for t in targets):
        # A finite requested list cannot be represented by an empty check list.
        cases = []
        t = SetSpec("format_presence", "semantic_category", "fruit types", candidates=("apple", "pear"))
        cases.append(("finite_candidate_checks", t, {"records": [], "checks": [
            {"set": t.set_id, "candidate": "apple", "state": "seen", "support": "direct", "refs": ["F1"], "facts": "an apple is visible"},
            {"set": t.set_id, "candidate": "pear", "state": "not_seen", "refs": [], "facts": "no pear observed in this input"}], "coverage": "complete"}))
        action = SetSpec("format_action", "semantic_category", "activity", candidates=("writing a note",))
        cases.append(("related_is_not_action", action, {"checks": [
            {"set": action.set_id, "candidate": "writing a note", "state": "unreadable",
             "support": "related", "refs": ["F1"], "facts": "A pen rests beside paper, but the actor's hands are hidden."}],
            "coverage": "partial", "input_gaps": [{"set":action.set_id,"candidate":"writing a note","reason":"occluded"}]}))
        cases.append(("checked_absence_is_not_gap", action, {"checks": [
            {"set":action.set_id,"candidate":"writing a note","state":"not_seen","refs":[],
             "facts":"The whole supplied input was inspected; nobody performs the writing action."}],
            "coverage":"complete","input_gaps":[]}))
    elif any(t.namespace == "physical_instance" for t in targets):
        t = SetSpec("format_tools", "physical_instance", "wrench")
        record = {"id": "O1", "set": t.set_id, "name": "metal wrench", "class": "wrench",
                  "conditions": {"target": "yes", "predicate": "yes"}, "facts": "a wrench with an open jaw",
                  "boxes": [{"ref": "F1", "xyxy": [100, 200, 300, 600]},
                            {"ref": "F2", "xyxy": [120, 200, 320, 600]}]}
        cases.append(("same_tool_across_frames", t, {"records": [record], "coverage": "complete"}))
        if any(v.predicate_kind in {"moving", "enters", "exits"} for v in targets):
            from dataclasses import replace
            motion_target = replace(t, predicate_kind="moving", predicate="the wrench itself moves")
            moving = {**record, "motion": {"refs": ["F1", "F2"], "witness_refs": ["F2"],
                "object_motion": True, "camera_motion_accounted": True}}
            cases.append(("moving_tool", motion_target, {"records": [moving], "coverage": "complete"}))
        second = {**record, "id": "O2", "boxes": [{"ref": "F1", "xyxy": [600, 200, 800, 600]}]}
        first = {**record, "boxes": record["boxes"][:1]}
        cases.append(("two_independent_tools_in_one_frame", t, {"records": [first, second],
            "coexisting": [{"ids": ["O1", "O2"], "ref": "F1", "independent_objects": True}], "coverage": "complete"}))
    elif any(t.namespace == "task_item" for t in targets):
        t = SetSpec("format_plan", "task_item", "shopping items", evidence_relation="planned", required_modalities=("subtitle",))
        cases.append(("text_plan", t, {"records": [], "task_updates": [{"local_id": "U1", "set_id": t.set_id,
            "kind": "create_plan", "item_key": "apples", "quantity": 2, "unit": "item", "owner": "person-A",
            "task_id": "trip-A", "evidence_refs": ["T1"], "binding_supported": True}], "coverage": "complete"}))
    elif any(t.namespace == "text_value" for t in targets):
        t = SetSpec("format_words", "text_value", "printed words", evidence_relation="text_present")
        cases.append(("literal_text", t, {"records": [{"id": "O1", "set": t.set_id, "name": "OPEN", "class": "printed word",
            "conditions": {"target": "yes", "predicate": "yes"}, "facts": "the sign reads OPEN", "raw_text": "OPEN", "refs": ["F1"]}], "coverage": "complete"}))
    else:
        t = SetSpec("format_fruit", "semantic_category", "fruit types", equivalence="category")
        cases.append(("category_mapping", t, {"records": [{"id": "O1", "set": t.set_id, "name": "red apple", "class": "apple",
            "conditions": {"target": "yes", "predicate": "yes"}, "facts": "an apple is visible", "query_value": "apple",
            "mapping_evidence": "color does not define a fruit type", "refs": ["F1"]}], "coverage": "complete"}))
    if any(t.equivalence == "combination" for t in targets):
        t = SetSpec("format_combinations", "semantic_category", "fruit and color combinations", equivalence="combination", attribute_keys=("fruit", "color"))
        cases.append(("attribute_combination", t, {"records": [{"id":"O1","set":t.set_id,"name":"red apple","class":"apple",
            "conditions":{"target":"yes","predicate":"yes"},"facts":"the apple skin is red","refs":["F1"],
            "attributes":{"fruit":"apple","color":"red"}}],"coverage":"complete"}))
    if role == "inspect_existing":
        converted = []
        for name, t, output in cases:
            output = {k: v for k, v in output.items() if k not in {"records", "distinct_pairs", "coexisting"}}
            original = next(v for n, _, v in cases if n == name)
            output["updates"] = [{**{k: v for k, v in r.items() if k != "id"}, "candidate_id": "format_candidate_" + str(i+1)}
                                 for i, r in enumerate(original.get("records", []))]
            converted.append((name, t, output))
        return converted
    return cases


def validate_example(role, target, response):
    validate(response, envelope_schema(role))
    for collection, schema in (("coexisting", COEXISTING_SCHEMA), ("distinct_pairs", DISTINCT_SCHEMA)):
        for row in response.get(collection, []):
            validate(row, schema)
    if target:
        rows = response.get("updates" if role == "inspect_existing" else "records", [])
        for r in rows:
            validate(r, record_schema(target, inspection=role == "inspect_existing"))
            for d in r.get("boxes", []):
                assert d["xyxy"][0] < d["xyxy"][2] and d["xyxy"][1] < d["xyxy"][3]
        for r in response.get("task_updates", []):
            validate(r, task_schema(target))
        for r in response.get("checks", []):
            validate(r, check_schema(target))
        for r in response.get("snapshots", []):
            validate(r, snapshot_schema(target))
        # These examples have independent hypothetical tasks, never a real question's set ID.
        assert target.set_id.startswith("format_")
        if target.namespace == "physical_instance":
            assert all(r["class"] == "wrench" for r in rows)
        if target.equivalence == "combination":
            assert all(set(target.attribute_keys) <= r["attributes"].keys() and r["class"] == "apple" for r in rows)
        elif target.namespace == "semantic_category":
            assert all(r["class"] == r["query_value"] == "apple" for r in rows)


def observation_instructions(role, payload, targets):
    present = [t for t in targets if t.set_id in payload.get("candidates", {})]
    history = [t for t in targets if t.namespace == "task_item"]
    records = [t for t in targets if t not in present and t not in history]
    header = ("Return checks and optional input_gaps; the host computes coverage. " if present and not records and not history
              else "Return coverage:complete/partial/unreadable, optional overflow:false and gaps:[]. ")
    text = ("Observe only the requested sets. Return one JSON response. " + header +
            "Unknown is not absence; report unreadable regions. "
            "Only core evidence proves current-window membership. Never guess missing facts.\n")
    if records:
        text += ("For these sets output records (or updates/new_candidates during inspection): " +
                 dumps([t.set_id for t in records]) + ". Each record: id:O1..., set, name, class, conditions, facts. "
                 "INPUT requirements contain descriptions. OUTPUT conditions contain ONLY yes/no/unknown for EACH requirement key. "
                 "Do not copy requirement sentences as judgments. Recognize actual class first; target=yes means it is the requested object, "
                 "not merely that something is visible. A non-target remains target=no, an uncertain target stays unknown.\n")
        if any(t.namespace == "physical_instance" for t in records):
            limit = 3
            text += (f"Physical entities: boxes is an array of 1–{limit} representative detections per object, "
                     "each {ref:F1,xyxy:[left,top,right,bottom]}. Four integer image coordinates 0–1000, positive area. "
                     "F1 is a frame, O1 is an object. One continuously visible object is ONE record; do not enumerate frames. "
                     "Choose at most three clear representative views even if the object appears in twenty frames. "
                     "Do not merge uncertain reappearances. Optional coexisting:[{ids:[O1,O2],ref:F1,independent_objects:true}] "
                     "requires distinct real objects simultaneously visible. No identity proof from equal color or box width.\n")
        if any(t.namespace in {"semantic_category", "text_value"} for t in records):
            text += ("Category/text records use nonempty refs:[F1/T1], no boxes. Categories require query_value at the requested "
                     "counting granularity and mapping_evidence: a generic target label is not a specific kind. "
                     "Raw aliases are not automatically distinct kinds; irrelevant shapes are not categories. "
                     "Literal text uses raw_text preserved exactly, no semantic correction. "
                     "Attribute combinations require attributes for each requested attribute_keys dimension.\n")
        if any(t.predicate_kind in {"moving", "enters", "exits"} for t in records):
            text += ("Motion records additionally require motion:{refs,witness_refs,object_motion,camera_motion_accounted,"
                     "boundary_crossing,identity_continuity}; witness frames must show object motion at distinct source times.\n")
        text += "Optional record fields: visibility, uncertainties:[], attributes:{}, raw_text. At most 12 records total.\n"
    if present:
        text += ("For sets listed in INPUT candidates output checks ONLY; copy candidate names exactly from that list, "
                 "never option letters. Each check: {set,candidate,state:seen/not_seen/unreadable,refs:[],facts,"
                 "uncertainties:[]}. support is required ONLY for seen, with direct/related/uncertain. "
                 "For not_seen OMIT support; keep the candidate, state, refs and nonempty inspection facts. "
                 "For unreadable support may be omitted. Never copy a state value into support. "
                 "seen requires support and legal core references. direct means the actual requested object/action is shown. "
                 "Preparation, associated objects, likely intention, or the eventual result do NOT prove that action happened. "
                 "Use unreadable with related/uncertain for ambiguous evidence; not_seen means this input was checked without observing it. "
                 "State facts as the observed action and distinguishing evidence, not an inference from context. "
                 "Check every requested candidate. Empty positive refs cannot establish seen. "
                 "Do not create generic category records for these sets.\n")
        text += ("Do not decide global coverage. Return checks and optional input_gaps:[{set,candidate,"
                 "reason:unreadable/occluded/uninspected}]. A checked input with no target action is not_seen and NO input gap. "
                 "A genuinely unjudgeable input is unreadable with its structured input gap. "
                 "Do not put 'the activity did not occur' into gaps or call that partial coverage. "
                 "Return one check per requested candidate; empty checks cannot prove absence. "
                 "The program combines these local checks over the required scope.\n")
    else:
        text += "No candidate checks are requested: omit checks or use checks:[]. Do not invent a candidate list.\n"
    if history:
        text += ("History sets output task_updates, not object records. Fields local_id,set_id,kind,item_key,evidence_refs,"
                 "binding_supported; optional owner,task_id,quantity,unit,raw_text,effective_time,refers_to,replacement_item,"
                 "replacement_quantity,completion_predicate. Only explicit text supports plan changes and completion.\n")
    if payload.get("check_review"):
        text += ("This is a targeted review of EXISTING candidate checks, not discovery. Output one check for each supplied "
                 "check_review entry using the same set/candidate. Re-examine whether the exact action is directly shown, "
                 "not merely preparation or related context. You may retract an earlier seen to not_seen or unreadable. "
                 "Previous claims are hypotheses, not facts. Return no records/updates/new candidates. "
                 "Do not preserve a seen judgment merely because it was previously reported.\n")
    if any("query_value_is_count_unit" in c.get("issues", []) for c in payload.get("existing", [])):
        text += ("TARGETED CATEGORY REVIEW: the earlier query_value copied the count_unit label. "
                 "Return a specific observed value at the requested granularity, with evidence and mapping_evidence. "
                 "If it cannot be identified, keep it unresolved; do not repeat the unit label or invent a category.\n")
    return text


def build_prompt(role, payload, targets=(), *, feedback=None):
    if role == "best_effort":
        from .answering import answer_prompt
        return f"Protocol {VERSION}. R4:best_effort.\n" + answer_prompt(payload)
    header = f"Protocol {VERSION}. R4:{role}. Return one complete JSON object. No final answer or option guessing.\n"
    if payload.get('pair_task'):
        from .interaction import pair_prompt
        return header + pair_prompt(payload,feedback)
    if payload.get('interaction'):
        from .interaction import interaction_prompt
        return header + interaction_prompt(payload,targets,role,feedback)
    if role == "compile":
        text = compile_instructions()
        if feedback:
            text += ("COMPILE_CORRECTION: Rebuild the COMPLETE task from the original question and the contract above. "
                     "You may add missing task definitions and correct units, scopes and operations. "
                     "Return only the corrected root task. Do not copy the previous_output/original_task/errors wrapper. "
                     "No observations have been committed at compilation; do not invent observed members or answers.\n" + dumps(feedback) + "\n")
        return header + text + "INPUT\n" + dumps(payload)
    elif role == "identity":
        text = ("Compare only specified existing candidates using the supplied crops and context. Return relations array. "
            "Each relation: {left,right,relation:SAME/DIFFERENT/UNKNOWN,basis,refs,facts}. "
            "Conditional requirements: DIFFERENT needs independent_objects:true grounded in the supplied evidence; "
            "distinct_tracks/stable_difference need nonempty features stating the distinguishing evidence. "
            "SAME continuous_track needs continuous_identity:true and multiple source times; reidentification needs features. "
            "These are evidence assertions, never boilerplate defaults. Optional supersedes revises the same pair only. "
            "SAME basis: shared_observation,continuous_track,reidentification. "
            "DIFFERENT basis: coexistence,distinct_tracks,stable_difference. UNKNOWN basis:uncertain. "
            "SAME needs both members' evidence and continuity or stable features plus context. DIFFERENT needs independently "
            "coexisting physical objects, distinct tracks, or stable distinguishing features. Box width, color alone, unmatched "
            "shortlist, different crops or failure to track do not prove DIFFERENT. Unknown is not different. "
            "Do not create objects or invent references.\n")
        text += ("left/right must use IDs from INPUT left/right. refs must use only F.../T... evidence from "
                 "INPUT evidence_by_object and catalog; C... object IDs are NEVER evidence references. "
                 "Different scenes or appearance alone do not prove independent trajectories. "
                 "If the provided evidence cannot establish SAME or DIFFERENT, return UNKNOWN with basis:uncertain, "
                 "refs:[] and a factual explanation. Never manufacture a relation to finish the task.\n")
        demo = {"relations":[{"left":"CexampleLeft","right":"CexampleRight","relation":"UNKNOWN",
                              "basis":"uncertain","refs":[],"facts":"No continuity or independent coexistence is established."}]}
        validate(demo, envelope_schema("identity"))
        text += "FORMAT DEMO only; substitute actual object IDs, never copy these placeholders.\n" + dumps(demo) + "\n"
    elif role == "scope":
        text = ("Locate only the requested semantic scopes in this input. Return {bindings:[{scope_id,refs,interval:[start,end],facts}],"
            "coverage:complete/partial/unreadable,gaps:[]}. Interval must stay in the supplied core and be supported by cited evidence. "
            "Source time and a clock/scoreboard value visible in the image are different. First means the earliest occurrence, "
            "not an arbitrary matching frame. No binding if not seen. Do not count objects.\n")
    else:
        text = observation_instructions(role, payload, targets)
        if any(t.namespace == "task_item" for t in targets):
            text += ("Use task_updates for history, not object records: local_id,set_id,kind,item_key,evidence_refs,binding_supported; "
                     "optional owner,task_id,quantity,unit,raw_text,effective_time,refers_to,replacement_item,replacement_quantity,completion_predicate. "
                     "Only explicit text establishes plans/completions; preserve cancellation/replacement/undo and task versions.\n")
        if role == "inspect_existing" and not payload.get("check_review"):
            text = text.replace("Return {records:[],checks:[],coverage:complete/partial/unreadable,overflow:false,gaps:[]}.",
                                "Return {updates:[],new_candidates:[],coverage:complete/partial/unreadable,overflow:false,gaps:[]}.")
            text += ("This is inspection: return updates:[record fields with candidate_id instead of id]. Update ONLY listed IDs. "
                     "Keep raw names; map query categories separately. Explicit discoveries go into new_candidates, never updates/records.\n")
        if payload.get("simultaneous_sets"):
            text += ("For simultaneous_sets also return snapshots:[{set,frames:{F1:{carrier_name:[O1,O2]},F2:{}}}]. "
                     "Use the compiled simultaneous_carriers unit (such as person/hand/table), never invent a different grouping; category/value defaults mean the carrier specified by the query. "
                     "Each listed CORE frame is a complete census of qualifying independent instances per carrier at that instant. "
                     "Use accepted local object IDs; a repeated ID cannot occupy two carriers at one instant. "
                     "A frame with no qualifying objects has {}. Omit unjudgeable frames and report gaps. "
                     "Representative boxes do NOT prove complete frame censuses. Never use context frames to establish the maximum.\n")
        # Candidate names alone do not route a count/list set to the presence executor.
        demo_targets = [replace(t, candidates=()) if t.set_id not in payload.get("candidates", {}) and t.namespace != "task_item" else t
                        for t in targets]
        for name, task, output in example_cases(role, demo_targets):
            if feedback and name == "empty":
                continue
            validate_example(role, task, output)
            text += (f"FORMAT DEMO {name}; independent hypothetical task, NOT facts about current media. "
                     "Never copy its names, IDs, boxes or references as observations. " +
                     (f"Hypothetical target: {task.target}; set:{task.set_id}. " if task else "") +
                     "\n```json\n" + dumps(output) + "\n```\n")
        if payload.get("simultaneous_sets"):
            _, demo_target, demo = example_cases("discover_candidates",[SetSpec("format_tools","physical_instance","wrench")])[-1]
            demo = {**demo,"snapshots":[{"set":demo_target.set_id,"frames":{"F1":{"workbench":["O1","O2"]},"F2":{}}}]}
            validate_example("discover_candidates",demo_target,demo)
            text += ("FORMAT DEMO simultaneous_census; independent hypothetical wrench task, never current video facts. "
                     "The two tools are independent in F1; F2 was checked and has no tools.\n```json\n"+dumps(demo)+"\n```\n")
    if feedback:
        if role in {"discover_candidates", "inspect_existing"}:
            text += "CORRECTION: fix each reported invalid slot at its original array position; accepted slots are ignored by the host. "
            if payload.get("candidates"):
                text += ("CHECK REPAIR: preserve each requested set/candidate and return every invalid check. "
                         "For not_seen omit support, retain nonempty facts about the checked input. "
                         "If unjudgeable use state:unreadable. Empty checks cannot repair a nonempty invalid list. "
                         "Do not emit object conditions in a check. ")
            if any(t.set_id not in payload.get("candidates", {}) and t.namespace != "task_item" for t in targets):
                text += ("RECORD REPAIR: conditions contains judgments, not requirement descriptions. "
                         "For each key return yes only when supported, no when contradicted, unknown when uncertain. "
                         "For example a judgment may be {\"target\":\"unknown\",\"predicate\":\"unknown\"}; "
                         "this does not assert a new observation. Preserve the object ID and evidence; do not erase it. ")
            if any(t.namespace == "task_item" for t in targets):
                text += "HISTORY REPAIR: return the invalid task_updates using their original local_id and explicit text evidence. "
        elif role == "identity":
            text += ("IDENTITY CORRECTION: regenerate relations for the requested left/right pairs only. "
                     "Use UNKNOWN if independence or continuity is not established; do not manufacture features or flags. ")
        else:
            text += "SCOPE CORRECTION: repair the reported bindings using the original scope IDs and evidence. "
        text += "Use the same evidence; no copied error wrappers, no guessed facts. " + dumps(feedback) + "\n"
    return header + text + "INPUT\n" + dumps(payload)
