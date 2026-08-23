from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from qwen3vl_agent.active_tree.types import (
    ACTION_KINDS,
    MODALITIES,
    OBSERVATION_MODES,
    TOPOLOGIES,
    CanonicalOption,
    EvidenceLedger,
    EvidenceSlot,
    OptionTest,
    PlannedAction,
    ProtocolError,
    SceneNode,
    TaskContract,
)
from qwen3vl_agent.coarse_to_fine.prompts import parse_json_object


@dataclass(frozen=True)
class ObserverFact:
    node_id: str
    slot_ids: tuple[str, ...]
    start_seconds: float
    end_seconds: float
    modality: str
    fact: str
    supports_option_ids: tuple[str, ...]
    refutes_option_ids: tuple[str, ...]
    source_frame_ids: tuple[str, ...]
    subtitle_refs: tuple[str, ...]


@dataclass(frozen=True)
class ObserverDecision:
    facts: tuple[ObserverFact, ...]
    missing_evidence: str


@dataclass(frozen=True)
class VerificationDecision:
    candidate_option_id: str | None
    sufficient: bool
    citations_valid: bool
    missing_slot_ids: tuple[str, ...]
    counterevidence: str
    reason: str
    decisive_evidence: str
    strongest_alternative_option_id: str | None
    alternative_refuted: bool


def canonicalize_options(choices: list[str] | tuple[str, ...]) -> list[CanonicalOption]:
    result: list[CanonicalOption] = []
    for index, choice in enumerate(choices):
        label = chr(ord("A") + index)
        text = re.sub(r"^[A-Z][.):]\s*", "", str(choice).strip(), flags=re.IGNORECASE)
        result.append(CanonicalOption(f"O{index + 1}", label, text))
    return result


def options_text(options: list[CanonicalOption]) -> str:
    return "\n".join(f"{item.option_id}: {item.text}" for item in options)


def contract_text(contract: TaskContract) -> str:
    return json.dumps(contract.to_dict(), ensure_ascii=False, separators=(",", ":"))


def build_task_compiler_prompt(question: str, *, subtitles_available: bool) -> str:
    return f"""You compile an observable evidence contract for long-video QA.
You see the QUESTION ONLY. Do not guess an answer and do not invent timestamps.
Choose one topology: local, sequence, multi_set, global, exclusion.
Create 1-2 option-neutral evidence slots, each under 18 words. Every slot description MUST name the concrete event, entity, relation, count, or spoken topic requested by the question. Keep JSON on one line.
Forbidden generic descriptions: "observable fact to find", "relevant evidence", "answer the question".
Use subtitle for questions about what someone says, asks, discusses, argues about, names, or calls something. Modalities may be visual, subtitle, ocr.
Use visual for enumerated sequences of actions or events; do not claim those events are named in subtitles.
Subtitles available later: {str(subtitles_available).lower()}.

QUESTION:
{question}

Return one compact JSON object with keys primary_topology, required_modalities, answer_criterion, slots. Each slot has slot_id, description, required, constraint.
No markdown. No answer choice. No reasoning outside JSON."""


def parse_task_contract(text: str, *, subtitles_available: bool) -> TaskContract:
    payload = _payload(text)
    topology = str(payload.get("primary_topology", "")).strip().lower()
    if topology not in TOPOLOGIES:
        raise ProtocolError(f"invalid evidence topology: {topology!r}")
    modalities = [
        str(item).strip().lower()
        for item in payload.get("required_modalities", [])
        if str(item).strip().lower() in MODALITIES
    ]
    if not subtitles_available:
        modalities = [item for item in modalities if item != "subtitle"]
    if not modalities:
        modalities = ["visual"]
    if "slots" not in payload:
        raise ProtocolError("task compiler response has no slots field")
    raw_slots = payload.get("slots", [])
    if not isinstance(raw_slots, list):
        raise ProtocolError("task compiler slots must be a list")
    slots: list[EvidenceSlot] = []
    for index, raw in enumerate(raw_slots[:4], start=1):
        if not isinstance(raw, dict):
            continue
        description = str(raw.get("description", "")).strip()
        if not description:
            continue
        slots.append(
            EvidenceSlot(
                slot_id=f"S{index}",
                description=description,
                required=bool(raw.get("required", True)),
                constraint=str(raw.get("constraint", "")).strip(),
            )
        )
    if not slots:
        raise ProtocolError("task compiler returned no usable evidence slots")
    criterion = str(payload.get("answer_criterion", "direct_support")).strip().lower()
    if criterion not in {"direct_support", "elimination", "mixed", "coverage"}:
        criterion = "direct_support"
    return TaskContract(topology, slots, list(dict.fromkeys(modalities)), answer_criterion=criterion)


def build_breadth_prompt(
    question: str,
    contract: TaskContract,
    nodes: list[SceneNode],
    frame_lines: list[str],
    subtitles_by_node: dict[str, str],
) -> str:
    node_lines = "\n".join(
        f"{node.id}: {node.start_seconds:.3f}-{node.end_seconds:.3f}s"
        for node in nodes
    )
    subtitle_text = _subtitle_blocks(subtitles_by_node)
    return f"""You are the breadth Observer. Inspect every labelled root-child storyboard.
The answer options are intentionally hidden. Record only atomic visible/spoken facts useful for the evidence slots. Do not answer the question.
Prefer the contract's required modalities. Emit at most ONE decisive fact per node and at most 3 facts total. Each fact is at most 25 words and subtitle_refs contains at most ONE decisive line. Keep JSON compact on one line.

QUESTION:
{question}

EVIDENCE CONTRACT:
{contract_text(contract)}

NODES:
{node_lines}

STORYBOARD TILE ORDER:
{chr(10).join(frame_lines)}

ALIGNED SUBTITLES (may be empty):
{subtitle_text or '(none)'}

Return one compact JSON object with top-level keys facts (array) and missing_evidence (string).
Every facts item must contain node_id, slot_ids, start_seconds, end_seconds, modality, fact, supports_option_ids, refutes_option_ids, source_frame_ids, subtitle_refs.
Use only listed node, slot and frame IDs. supports_option_ids and refutes_option_ids must be empty because options are hidden. Empty facts are allowed. No markdown."""


def build_discriminator_prompt(
    question: str,
    options: list[CanonicalOption],
    contract: TaskContract,
    ledger: EvidenceLedger,
) -> str:
    return f"""You refine an evidence contract after one option-blind breadth pass.
Use the options only to define discriminative observable tests. Do not choose an answer.
Keep INITIAL CONTRACT slots unchanged. A support/refute test must describe evidence that could actually appear in video, subtitles, or OCR.

QUESTION:
{question}

OPTIONS:
{options_text(options)}

INITIAL CONTRACT:
{contract_text(contract)}

BREADTH FACTS:
{ledger.compact_text()}

Return one compact JSON object with option_tests (array) and answer_criterion.
Every option_tests item has option_id, support_test, refute_test. Include exactly one item for every listed option ID. No answer, markdown, or confidence."""


def parse_discriminator(
    text: str,
    *,
    initial: TaskContract,
    options: list[CanonicalOption],
) -> TaskContract:
    payload = _payload(text)
    slots: list[EvidenceSlot] = list(initial.slots)
    valid_option_ids = {item.option_id for item in options}
    option_tests: list[OptionTest] = []
    raw_tests = payload.get("option_tests", [])
    if isinstance(raw_tests, list):
        for raw in raw_tests:
            if not isinstance(raw, dict):
                continue
            option_id = str(raw.get("option_id", "")).strip().upper()
            if option_id not in valid_option_ids:
                continue
            option_tests.append(
                OptionTest(
                    option_id,
                    str(raw.get("support_test", "")).strip(),
                    str(raw.get("refute_test", "")).strip(),
                )
            )
    known = {item.option_id for item in option_tests}
    for option in options:
        if option.option_id not in known:
            option_tests.append(
                OptionTest(
                    option.option_id,
                    f"Find direct observable evidence for: {option.text}",
                    f"Find observable evidence incompatible with: {option.text}",
                )
            )
    criterion = str(payload.get("answer_criterion", initial.answer_criterion)).lower()
    if criterion not in {"direct_support", "elimination", "mixed", "coverage"}:
        criterion = initial.answer_criterion
    return TaskContract(
        initial.primary_topology,
        slots,
        initial.required_modalities,
        option_tests,
        criterion,
    )


def build_planner_prompt(
    question: str,
    options: list[CanonicalOption],
    contract: TaskContract,
    ledger: EvidenceLedger,
    *,
    current_node_id: str,
    visible_nodes: list[SceneNode],
    expanded_node_ids: set[str],
    observation_signatures: list[tuple[Any, ...]],
    repair_directive: str,
) -> str:
    node_lines = "\n".join(
        f"{node.id} {node.start_seconds:.1f}-{node.end_seconds:.1f}s "
        f"leaf={str(node.is_leaf).lower()} expanded={str(node.id in expanded_node_ids).lower()} "
        f"change={node.boundary_score:.3f}"
        for node in visible_nodes
    )
    missing = ",".join(ledger.missing_slot_ids(contract)) or "none"
    repeats = json.dumps(observation_signatures[-12:], ensure_ascii=False, default=str)
    return f"""You are a bounded long-video search Planner.
Select actions that maximize NEW DISCRIMINATIVE EVIDENCE, not mere relevance.
Return a ranked Top-3 slate. Never invent timestamps or node IDs.

Legal kinds: expand, zoom_out, shift, observe, compare, verify, answer.
Observation modes: overview, inspect, motion, event_verify, detail_ocr, subtitle.
event_verify is controller-reserved; do not propose it.
- expand: target a visible non-leaf node not yet expanded.
- zoom_out: leave the current branch.
- shift: move to a visible sibling branch.
- observe: inspect one visible node with one mode for one slot.
- compare: compare 2 visible nodes; put them in compare_node_ids.
- verify/answer: only when the evidence contract appears complete.

QUESTION:
{question}

OPTIONS:
{options_text(options)}

CONTRACT:
{contract_text(contract)}

MISSING REQUIRED SLOTS: {missing}
CURRENT NODE: {current_node_id}
VISIBLE FRONTIER:
{node_lines}

EVIDENCE LEDGER:
{ledger.compact_text()}

ALREADY OBSERVED SIGNATURES:
{repeats}

VERIFIER REPAIR DIRECTIVE:
{repair_directive or '(none)'}

Return one compact JSON object whose only top-level key is actions. actions is a ranked array of at most 3 objects; every object has kind, node_id, mode, slot_id, compare_node_ids, expected_new_evidence.
Use null for fields that do not apply. No markdown or hidden chain of thought."""


def parse_planner(text: str) -> list[PlannedAction]:
    payload = _payload(text)
    raw_actions = payload.get("actions")
    if raw_actions is None:
        raw_actions = [
            item
            for item in _complete_json_objects(text)
            if isinstance(item, dict) and str(item.get("kind", "")).lower() in ACTION_KINDS
        ]
    if not isinstance(raw_actions, list):
        raise ProtocolError("planner actions must be a list")
    actions: list[PlannedAction] = []
    for raw in raw_actions[:3]:
        if not isinstance(raw, dict):
            continue
        kind = str(raw.get("kind", "")).strip().lower()
        if kind not in ACTION_KINDS:
            continue
        mode_value = str(raw.get("mode", "")).strip().lower() or None
        if mode_value is not None and mode_value not in OBSERVATION_MODES:
            mode_value = None
        compare_ids = raw.get("compare_node_ids", [])
        if not isinstance(compare_ids, list):
            compare_ids = []
        actions.append(
            PlannedAction(
                kind=kind,
                node_id=str(raw.get("node_id", "")).strip() or None,
                mode=mode_value,
                slot_id=str(raw.get("slot_id", "")).strip() or None,
                compare_node_ids=tuple(str(item) for item in compare_ids[:2]),
                expected_new_evidence=str(raw.get("expected_new_evidence", "")).strip(),
            )
        )
    if not actions:
        raise ProtocolError("planner returned no usable actions")
    return actions


def build_observer_prompt(
    question: str,
    contract: TaskContract,
    *,
    action: PlannedAction,
    nodes: list[SceneNode],
    frame_lines: list[str],
    subtitles: str,
) -> str:
    node_text = "\n".join(
        f"{node.id}: {node.start_seconds:.3f}-{node.end_seconds:.3f}s"
        for node in nodes
    )
    if contract.primary_topology == "sequence" and action.mode == "event_verify":
        slot = next(
            (item for item in contract.slots if item.slot_id == action.slot_id),
            None,
        )
        target = slot.description if slot is not None else "(unknown event)"
        return f"""You are a fine-grained visual event validator.
The full question and answer options are hidden. Test only the target event against this already-localized short shot.
Emit ONE fact only when the exact visible transition occurs; otherwise return empty facts. Do not accept a merely related static pose or object state.
Cite exactly TWO distinct supplied frames in temporal order: one before the transition and one after it. Describe what visibly changed rather than copying the target.
Use modality=visual, slot_ids=["{action.slot_id or ''}"], and keep supports_option_ids/refutes_option_ids/subtitle_refs empty.

TARGET EVENT TO TEST:
{target}

LOCALIZED NODE:
{node_text}

FRAME ORDER:
{chr(10).join(frame_lines) or '(none)'}

The supplied contact sheet shows these labelled frames left-to-right in time.

Return one compact JSON object with top-level keys facts and missing_evidence.
Every fact contains node_id, slot_ids, start_seconds, end_seconds, modality, fact, supports_option_ids, refutes_option_ids, source_frame_ids, subtitle_refs.
Only use supplied node/frame/slot IDs. No markdown, answer choice, or confidence."""
    if contract.primary_topology == "sequence" and action.mode in {
        "overview",
        "inspect",
        "motion",
    }:
        return f"""You are a target-blind visual transcriber.
The question, target event, slot descriptions, subtitles, and answer options are hidden to prevent prompt copying.
Describe at most ONE concrete action or visible state transition that is directly shown in the supplied frames. If no clear transition is visible, return empty facts.
A sequence event needs before/after proof: cite exactly TWO distinct supplied frames that show the transition. A static pose or object state does not prove an action (for example, holding something does not prove picking it up).
Use modality=visual, keep slot_ids/supports_option_ids/refutes_option_ids/subtitle_refs empty. The fact is at most 25 words.

OBSERVED NODES:
{node_text}

FRAME ORDER:
{chr(10).join(frame_lines) or '(none)'}

Return one compact JSON object with top-level keys facts and missing_evidence.
Every fact contains node_id, slot_ids, start_seconds, end_seconds, modality, fact, supports_option_ids, refutes_option_ids, source_frame_ids, subtitle_refs.
Only use supplied node/frame IDs. No markdown, target guesses, or confidence."""
    return f"""You are an evidence Observer, not the final answerer.
Inspect only the supplied raw frames and subtitles. Emit atomic facts with grounded time ranges.
Answer options are hidden to prevent confirmation bias. Do not infer an option or paraphrase a hypothesis. Unknown is allowed.
If ACTION mode is subtitle, emit at most ONE fact, at most 25 words, use modality=subtitle, and cite at most TWO decisive supplied subtitle lines in subtitle_refs. Do not copy the full subtitle block.
If ACTION mode is visual/overview/inspect/motion, emit at most ONE fact, at most 25 words, use modality=visual, cite at most TWO decisive supplied source_frame_ids, and leave subtitle_refs empty.
If ACTION mode is detail_ocr, emit at most ONE fact, at most 25 words, use modality=ocr, cite at most TWO decisive supplied source_frame_ids, and leave subtitle_refs empty.

QUESTION:
{question}

CONTRACT:
{contract_text(contract)}

ACTION:
{json.dumps(action.to_dict(), ensure_ascii=False)}

OBSERVED NODES:
{node_text}

FRAME ORDER:
{chr(10).join(frame_lines) or '(no frames for subtitle-only observation)'}

SUBTITLES:
{subtitles or '(none)'}

Return one compact JSON object with top-level keys facts and missing_evidence.
Every facts item must contain node_id, slot_ids, start_seconds, end_seconds, modality, fact, supports_option_ids, refutes_option_ids, source_frame_ids, subtitle_refs.
supports_option_ids and refutes_option_ids must be empty because options are hidden. Only use supplied IDs. Empty facts are allowed. No markdown or confidence."""


def parse_observer(
    text: str,
    *,
    valid_node_ids: set[str],
    valid_slot_ids: set[str],
    valid_option_ids: set[str],
    valid_frame_ids: set[str],
    prefer_frame_endpoints: bool = False,
) -> ObserverDecision:
    payload = _payload(text)
    raw_facts = payload.get("facts")
    if raw_facts is None:
        required_fact_keys = {
            "node_id",
            "slot_ids",
            "start_seconds",
            "end_seconds",
            "modality",
            "fact",
        }
        raw_facts = [
            item
            for item in _complete_json_objects(text)
            if isinstance(item, dict) and required_fact_keys <= item.keys()
        ]
        if not raw_facts:
            raise ProtocolError("observer response has no recoverable fact objects")
    if not isinstance(raw_facts, list):
        raise ProtocolError("observer facts must be a list")
    facts: list[ObserverFact] = []
    for raw in raw_facts:
        if not isinstance(raw, dict):
            continue
        node_id = str(raw.get("node_id", "")).strip()
        fact = str(raw.get("fact", "")).strip()
        modality = str(raw.get("modality", "")).strip().lower()
        modality = {
            "overview": "visual",
            "inspect": "visual",
            "motion": "visual",
            "detail_ocr": "ocr",
        }.get(modality, modality)
        if node_id not in valid_node_ids or not fact or modality not in MODALITIES:
            continue
        try:
            start = float(raw.get("start_seconds"))
            end = float(raw.get("end_seconds"))
        except (TypeError, ValueError):
            continue
        if end < start:
            start, end = end, start
        slot_ids = _valid_ids(raw.get("slot_ids"), valid_slot_ids)
        support = _valid_ids(raw.get("supports_option_ids"), valid_option_ids)
        refute = _valid_ids(raw.get("refutes_option_ids"), valid_option_ids)
        frame_ids = _valid_ids(raw.get("source_frame_ids"), valid_frame_ids)
        if prefer_frame_endpoints and len(frame_ids) > 2:
            frame_ids = [frame_ids[0], frame_ids[-1]]
        else:
            frame_ids = frame_ids[:2]
        subtitle_refs = raw.get("subtitle_refs", [])
        if not isinstance(subtitle_refs, list):
            subtitle_refs = []
        if modality != "subtitle":
            subtitle_refs = []
        facts.append(
            ObserverFact(
                node_id,
                tuple(slot_ids),
                start,
                end,
                modality,
                fact,
                tuple(support),
                tuple(refute),
                tuple(frame_ids),
                tuple(str(item) for item in subtitle_refs[:16]),
            )
        )
    return ObserverDecision(
        tuple(facts),
        str(payload.get("missing_evidence", "")).strip(),
    )


def build_completeness_prompt(
    question: str,
    options: list[CanonicalOption],
    contract: TaskContract,
    ledger: EvidenceLedger,
    *,
    frame_lines: list[str],
    controller_hint: str,
) -> str:
    slot_lines = "\n".join(
        f"{slot.slot_id}: {slot.description}" for slot in contract.slots if slot.required
    )
    return f"""You are the completeness Verifier. Re-check the cited RAW evidence, not the planner's confidence.
Decide whether every required evidence slot is grounded and whether one option is uniquely supported.
Prefer the most specific option directly stated by evidence. Mere topical relation is not support.
Your candidate must agree with your reason: never choose an option your reason negates.

QUESTION:
{question}

OPTIONS:
{options_text(options)}

REQUIRED EVIDENCE SLOTS:
{slot_lines or '(none)'}

ACTIVE GROUNDED EVIDENCE:
{ledger.compact_text(max_items=8, include_breadth=False)}

DETERMINISTIC CONTROLLER PROPOSAL (untrusted; audit it):
{controller_hint or '(none)'}

RAW FRAME ORDER:
{chr(10).join(frame_lines) or '(none)'}

Return one single-line JSON object with exactly these keys: candidate_option_text, decisive_evidence_id, strongest_alternative_text, alternative_refuted, missing_slot_ids, counterevidence, reason.
candidate_option_text and strongest_alternative_text must copy the complete text of a listed option or be null; never output an O-number. decisive_evidence_id must be one listed EV ID or null.
alternative_refuted must be a JSON boolean. Keep reason and counterevidence under 12 words each. Use null when unresolved. Never copy evidence text. No markdown or confidence."""


def build_skeptic_prompt(
    question: str,
    shuffled_options: list[CanonicalOption],
    contract: TaskContract,
    ledger: EvidenceLedger,
    *,
    frame_lines: list[str],
) -> str:
    slot_lines = "\n".join(
        f"{slot.slot_id}: {slot.description}" for slot in contract.slots if slot.required
    )
    return f"""You are a blinded skeptical Verifier.
You do not know another model's favored answer. Options are deliberately shuffled.
Independently choose the uniquely supported option, search for the strongest counterexample, and reject insufficient evidence.
Prefer the most specific option directly stated by evidence. Mere topical relation is not support.
Your candidate must agree with your reason: never choose an option your reason negates.

QUESTION:
{question}

SHUFFLED OPTIONS:
{options_text(shuffled_options)}

REQUIRED EVIDENCE SLOTS:
{slot_lines or '(none)'}

ACTIVE GROUNDED EVIDENCE:
{ledger.compact_text(max_items=8, include_breadth=False)}

RAW FRAME ORDER:
{chr(10).join(frame_lines) or '(none)'}

Return one single-line JSON object with exactly these keys: candidate_option_text, decisive_evidence_id, strongest_alternative_text, alternative_refuted, missing_slot_ids, counterevidence, reason.
candidate_option_text and strongest_alternative_text must copy the complete text of a listed option or be null; never output an O-number. decisive_evidence_id must be one listed EV ID or null.
alternative_refuted must be a JSON boolean. Keep reason and counterevidence under 12 words each. Use null when unresolved. Never copy evidence text. No markdown or confidence."""


def parse_verification(
    text: str,
    *,
    options: list[CanonicalOption],
    valid_evidence_ids: set[str],
) -> VerificationDecision:
    payload = _payload(text)
    required_keys = {
        "candidate_option_text",
        "decisive_evidence_id",
        "strongest_alternative_text",
        "alternative_refuted",
        "missing_slot_ids",
        "reason",
    }
    if not required_keys <= payload.keys():
        raise ProtocolError("verification response is missing required top-level fields")
    def resolve_option(raw: Any) -> str | None:
        if raw is None:
            return None
        normalized = " ".join(str(raw).casefold().split()).rstrip(". ")
        normalized = re.sub(r"^(?:o\d+|[a-z])[.:)]\s*", "", normalized)
        for option in options:
            if normalized == " ".join(option.text.casefold().split()).rstrip(". "):
                return option.option_id
        return None

    candidate = resolve_option(payload.get("candidate_option_text"))
    raw_missing = payload.get("missing_slot_ids", [])
    if isinstance(raw_missing, str):
        raw_missing = [raw_missing] if raw_missing.strip() else []
    elif not isinstance(raw_missing, list):
        raw_missing = []
    decisive_raw = payload.get("decisive_evidence_id")
    decisive = str(decisive_raw).strip().upper() if decisive_raw is not None else ""
    if decisive not in valid_evidence_ids:
        decisive = ""
    alternative = resolve_option(payload.get("strongest_alternative_text"))
    alternative_refuted = payload.get("alternative_refuted") is True
    citations_valid = bool(decisive)
    sufficient = bool(
        candidate
        and citations_valid
        and not raw_missing
        and alternative
        and alternative != candidate
        and alternative_refuted
    )
    return VerificationDecision(
        candidate_option_id=candidate,
        sufficient=sufficient,
        citations_valid=citations_valid,
        missing_slot_ids=tuple(str(item) for item in raw_missing),
        counterevidence=str(payload.get("counterevidence", "")).strip(),
        reason=str(payload.get("reason", "")).strip(),
        decisive_evidence=decisive,
        strongest_alternative_option_id=alternative,
        alternative_refuted=alternative_refuted,
    )


def repair_prompt(original_prompt: str, raw_response: str, error: Exception) -> str:
    return f"""Repair a malformed protocol response. Do not reconsider the task or add facts.
Return only a corrected JSON object matching the schema stated in ORIGINAL PROMPT.

ERROR: {type(error).__name__}: {error}

MALFORMED RESPONSE:
{raw_response}

ORIGINAL PROMPT:
{original_prompt}"""


def _payload(text: str) -> dict[str, Any]:
    try:
        return parse_json_object(text)
    except Exception as exc:
        raise ProtocolError(str(exc)) from exc


def _valid_ids(raw: Any, valid: set[str]) -> list[str]:
    if not isinstance(raw, list):
        return []
    return list(dict.fromkeys(str(item).strip() for item in raw if str(item).strip() in valid))


def _subtitle_blocks(values: dict[str, str]) -> str:
    blocks = [f"[{node_id}]\n{text}" for node_id, text in values.items() if text]
    return "\n\n".join(blocks)


def _complete_json_objects(text: str) -> list[dict[str, Any]]:
    """Recover only individually complete objects from a possibly truncated response."""

    decoder = json.JSONDecoder()
    values: list[dict[str, Any]] = []
    for index, character in enumerate(text):
        if character != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            values.append(value)
    return values
