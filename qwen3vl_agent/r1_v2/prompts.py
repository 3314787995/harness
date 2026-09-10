"""V2 query improvements are instructions only; the R1 schemas/parsers are unchanged."""

import json

from qwen3vl_agent.r1.prompts import INSTRUCTIONS, POLICY, SCHEMAS
from qwen3vl_agent.r1.prompts import prompt as original_prompt
from qwen3vl_agent.r1.prompts import repair_prompt as original_repair_prompt

QUERY = """Compile the supplied QUESTION into a plan for observing UNKNOWN information.
You have not seen the video. Do not answer the question or describe imagined video content.
Copy no presumed answer, object value, cause, action or identity into fields or anchor_description.
Return at least one field: each description names the observable variable to read, not its value.
Keep identifying anchors separate from answer fields. Anchor descriptions retain all question-given
entities, relations, qualifiers, negation and temporal constraints; add no identifying detail.
Choose only observation modes needed to answer the question:
- static: colour, shape, material, appearance and other properties readable in a frame;
- ocr: exact text, digits, labels or a model/name that must be read from visible writing;
- ordered: actions or changes whose answer needs consecutive visual evidence;
- caption/full_span: a requested description of a process over its specified span.
An initially/first/last/opening/end qualifier identifies which occurrence to locate. Preserve it in
the anchor and semantic_hint; it alone does not require ordered mode or sequence coverage.
Never invent a seconds range from these words. Only supplied explicit intervals impose time bounds.
Combine modes only when each is independently necessary. Do not request OCR for a colour question.
Existence requires a positive witness or adequate coverage for a negative; a search miss is unknown.
requires_reference is true only for comparison/binding to another discrete reference scene.
Keep supplied before/after relations. Do not invent reference evidence or require continuous tracking.
required_modalities contains indispensable dependencies, not every available/helpful modality.
requires_speaker_binding applies only when attributing speech to an identified visible person.
Examples of compilation, not of answers:
Question: What pattern is on the scarf first worn by the dancer?
Field: The pattern on the scarf. Anchor: The scarf first worn by the dancer.
Modes: [static]. Coverage: point. Hint: first occurrence. Do NOT write 'striped scarf'.
Question: Which word is printed on the parcel label?
Field: The exact word printed on the parcel label. Modes: [ocr]. Do NOT invent the word.
Question: How does the worker open the container?
Field: The worker's action used to open the container. Modes: [ordered]. Do NOT invent an action.
Use concise descriptions; the controller assigns Q1, Q2, ... . Return only the existing JSON schema.
"""

DISCRIMINANTS = (
    INSTRUCTIONS["discriminants"]
    + """
These are UNKNOWN observation requirements, not facts or answers. Add a mode only when the original
question and the distinguishing attribute require it. Colour/shape alternatives do not require OCR.
First/last/opening/end alone does not require ordered observations. Preserve the question's target
and temporal qualifiers; do not infer a scene, identity, action, answer value or fixed time window.
"""
)

LOCATOR = (
    INSTRUCTIONS["locator"]
    + """
Check the complete target relation: seeing a person alone does not locate that person holding the
requested object. List any unverified required condition under unresolved_anchor_conditions.
Do not describe an opening frame as the requested first occurrence without seeing the target event.
"""
)

V2_INSTRUCTIONS = {"query": QUERY, "discriminants": DISCRIMINANTS, "locator": LOCATOR}


def prompt(role, payload):
    if role not in V2_INSTRUCTIONS:
        return original_prompt(role, payload)
    return (
        f"R1:{role}\n{POLICY}\n{V2_INSTRUCTIONS[role]}\n"
        f"OUTPUT_SCHEMA:\n{json.dumps(SCHEMAS[role], ensure_ascii=False)}\n"
        f"INPUT_JSON:\n{json.dumps(payload, ensure_ascii=False, default=str)}"
    )


def repair_prompt(role, raw, error, payload):
    if role not in {"query", "discriminants"}:
        return original_repair_prompt(role, raw, error)
    return (
        f"R1:repair\nRepair the {role} JSON using the original compilation input below. "
        "Restore missing observation requests from the question, never their answers. "
        "You have not seen the video. Add no video facts, evidence IDs or answer values.\n"
        f"{V2_INSTRUCTIONS[role]}\n"
        f"SCHEMA: {json.dumps(SCHEMAS[role], ensure_ascii=False)}\n"
        f"ERROR: {error}\nRAW_RESPONSE:\n{raw}\n"
        f"INPUT_JSON:\n{json.dumps(payload, ensure_ascii=False, default=str)}"
    )
