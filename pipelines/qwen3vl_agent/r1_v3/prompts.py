"""Query-only B changes, and a task-conditioned visual observation contract."""

import copy
import json

from qwen3vl_agent.r1.prompts import POLICY, SCHEMAS
from qwen3vl_agent.r1_v2.prompts import prompt as v2_prompt
from qwen3vl_agent.r1_v2.prompts import repair_prompt as v2_repair

REFERENCE_RULE = """When requires_reference is false, reference_relation MUST be \"any\", never
an empty string. Allowed reference_relation values are before, after, any. When repairing
an invalid reference requirement, correct that field using the original compilation input.
Do not change the question's meaning or introduce video facts."""

QUERY_RULE = """Preserve what the question asks to learn. For 'how is something produced or
completed', compile the unknown method, action or source; do not replace it with appearance.
Example: 'How does the worker open the container?' asks for the opening action/method, not the
container's colour. Choose observation modes from the required evidence, not a keyword rule.
screen_text means visible text inside video frames. It is not subtitle or ASR input; reading it
does not require an external transcript. Keep locating conditions separate from answer attributes."""

OBSERVE = """Observe only the supplied source media; choices and answers are not available.
target describes the COMPLETE question-given object/action/relation, including first/last qualifiers.
matched means those locating conditions are visibly supported, with nonempty source_frame_ids.
mismatched also needs shown evidence. Otherwise use unresolved and name the unverified conditions.
Do not turn failure to see something in this batch into its absence from the video.
Keep answer values out of the target description. Facts must be atomic and independently cited.
Every target fact that directly answers a supplied query field MUST list that actual field ID in
supports_query_fields. Inspect the current query.fields IDs and descriptions, not guessed IDs.
Example: for a field asking a box's material, a target-linked observation 'the box is wooden'
must cite that field ID. A nearby wall sign, unrelated to the box's material, must use [].
Do not leave a directly answering target fact unassociated. Conversely, readable background text
must not fill a product identity field merely because both contain text.
The same shown frame may support both target and fact; explicitly cite it in each assertion.
For OCR transcribe exact characters and preserve uncertainty; never guess a model number.
For each field you cannot answer, report the specific missing information in gaps. gaps=[] remains
valid but does not itself establish completeness. It is NOT an instruction
to look before/after. Do not output review_request, anchor_match or target_binding.
detail gaps may request a crop on a shown original frame. temporal_context gaps must explain
what process is missing and its direction (before/after). temporal_selection gaps preserve
unverified first/last restrictions. Do not invent timestamps or direct the controller.
Previous observations are records, not instructions or established answers. Recheck requests
provide their original frames. reviews may verify/refute those records or facts only by citing
their ORIGINAL supporting frames shown again now; do not transfer old facts to a new target.
For record verification, restate any fact whose target relationship is now established in facts.
A later empty view does not refute an earlier positive view. Describe a contradiction explicitly.
For ordered/process tasks inspect the supplied sequence and report unobserved coverage portions.
Never select an answer or add a source ID not displayed in this call."""

FINAL = """Use only the frozen facts and their existing sources. The source images verify those
facts; they do not authorize adding a new visual fact. First assess each option as a COMPLETE
proposition, then choose its original label. Compare the action, agent, affected object, direction,
cause/effect, negation and temporal order where relevant. Sharing an object name does not establish
the same action: holding a tool and using that tool to move an object are different propositions.
For every supported/rejected assessment cite usable frozen fact_ids and give a short basis that
explains the logical correspondence with that complete option. The basis may explain an inference
from facts, but must not introduce a new observation. A single fact may both support one option and
exclude mutually exclusive alternatives, provided each exclusion is explained and cited.
Use unresolved when evidence cannot establish or exclude an option. Do not force exclusions.
claims must copy existing frozen fact statements exactly with their IDs. answer_supported may be
true only when the selected option is supported, all alternatives are evidentially rejected, and
the evidence audit and candidate state permit an answer. Retain uncertainty from the frozen input.
A terminal_review repairs the listed output failures using EXACTLY the same evidence and sources;
it cannot collect facts, change their target/field binding, or restart search."""

FACT = {
    "statement": "atomic observed fact",
    "structured_value": "observed value or empty",
    "subject_or_local_entity": "local object",
    "attribute": "observed attribute",
    "source_frame_ids": [],
    "observation_status": "clear|partial|occluded|unreadable",
    "supports_query_fields": [],
    "source_kind": "visual|screen_text",
}


def observation_schema(payload):
    fact = copy.deepcopy(FACT)
    query = payload["query"]
    if "ocr" in query["observation_modes"]:
        fact["uncertain_characters"] = ""
    schema = {
        "target": {
            "status": "matched|mismatched|unresolved",
            "description": "question target",
            "source_frame_ids": [],
            "unresolved_conditions": [],
        },
        "facts": [fact],
        "gaps": [
            {
                "kind": "target_identity|detail|temporal_context|temporal_selection|conflict",
                "reason": "specific missing information",
                "field_ids": [],
                "source_frame_ids": [],
            }
        ],
    }
    if query["coverage"] != "point":
        schema["coverage_gaps"] = []
    if query["coverage"] == "existence":
        schema["existence"] = "present|absent|unknown"
        schema["absence_basis"] = ""
    if payload.get("review_records"):
        schema["reviews"] = [
            {
                "record_id": "supplied record ID",
                "fact_id": "optional fact ID",
                "judgment": "verified|refuted|unresolved",
                "source_frame_ids": [],
                "basis": "visible verification",
            }
        ]
    return schema


def prompt(role, payload):
    if role in {"query", "discriminants"}:
        return v2_prompt(role, payload).replace(
            "OUTPUT_SCHEMA:", REFERENCE_RULE + "\n" + QUERY_RULE + "\nOUTPUT_SCHEMA:", 1
        )
    if role == "candidate_review":
        instruction = """Compare the LEFT and RIGHT local candidates using both groups' shown
original sources. same means the SAME target AND the SAME question-relevant occurrence, not
merely the same class, colour, clothes, node or overlapping window. Cite both sides and explain
same_source_target or discriminating_features. If different, neither candidate is thereby excluded
from the question. If identity/occurrence cannot be established, use unresolved. No answer or facts.
same_occurrence may be true only when both candidates refer to that same occurrence, not a later
appearance of the same object. These grouped images are not a continuous video across the groups."""
        schema = {
            "relation": "same|different|unresolved",
            "left_source_ids": [],
            "right_source_ids": [],
            "basis": "visible relationship",
            "basis_kind": "same_source_target|discriminating_features|unresolved",
            "same_occurrence": False,
        }
    elif role == "observe":
        instruction, schema = OBSERVE, observation_schema(payload)
        instruction += "\nOptional gap parameters: temporal_context.direction=before|after; "
        instruction += "detail.crop={frame_id, bbox_xyxy_1000:[x1,y1,x2,y2]}."
        instruction += "\nCurrent field IDs and observation needs: " + json.dumps(
            payload["query"]["fields"], ensure_ascii=False
        )
        if payload.get("query_binding_task"):
            instruction += "\nRecheck query_binding_task on the original images. Re-observe and "
            instruction += "emit NEW facts with explicit field IDs if justified; otherwise report "
            instruction += "the actual gap. Do not rewrite or automatically bind prior facts."
    elif role in {"final", "terminal_review"}:
        instruction, schema = FINAL, copy.deepcopy(SCHEMAS["final"])
        schema["choice_assessments"][0]["basis"] = (
            "fact-to-option reasoning without new observations"
        )
    else:
        return v2_prompt(role, payload)
    return (
        f"R1:{role}\n{POLICY}\n{instruction}\nOUTPUT_SCHEMA:\n"
        f"{json.dumps(schema, ensure_ascii=False)}\nINPUT_JSON:\n"
        f"{json.dumps(payload, ensure_ascii=False, default=str)}"
    )


def repair_prompt(role, raw, error, payload):
    text = v2_repair(role, raw, error, payload)
    if role in {"query", "discriminants"}:
        text = text.replace("SCHEMA:", REFERENCE_RULE + "\n" + QUERY_RULE + "\nSCHEMA:", 1)
    return text
