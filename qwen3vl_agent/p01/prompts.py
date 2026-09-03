from __future__ import annotations

import json
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from qwen3vl_agent.coarse_to_fine.prompts import parse_json_object
from qwen3vl_agent.coarse_to_fine.types import DecisionError, FrameRef
from qwen3vl_agent.p01.types import (
    ANSWER_MODES,
    COVERAGE_REQUIREMENTS,
    PRIMARY_MODES,
    VISIBILITIES,
    BoundingBox,
    CanonicalOption,
    ChoiceDecision,
    ClaimTest,
    ClaimVerdict,
    DecisionSpec,
    EventFact,
    EvidencePacket,
    Fact,
    ObservationSlot,
    ObservationSpec,
    OptionAssessment,
    OptionRule,
    ProtocolError,
    StaticFact,
    TextFact,
    TimeSpan,
)


@dataclass(frozen=True)
class LocatorDecision:
    node_ids: tuple[str, ...]
    visible_anchors: tuple[str, ...]


@dataclass(frozen=True)
class ScoutDecision:
    target_visible: bool
    supporting_span: TimeSpan
    visible_anchor: str
    facts: tuple[Fact, ...]
    missing_slot_ids: tuple[str, ...]
    conflicts: tuple[str, ...]


@dataclass(frozen=True)
class RescueLocatorDecision:
    frame_ids: tuple[str, ...]
    visible_anchors: tuple[str, ...]


def build_observation_compiler_prompt(question: str, *, answer_mode: str) -> str:
    return f"""You compile a question-only observation contract for a bounded local-video task.

You see the QUESTION but never the answer choices. Do not answer it and do not invent timestamps.
The question is already asserted to be answerable from one continuous local span.

Choose one primary_mode from the visual evidence needed, not merely from the wording:
- static_visual: one or several still views establish an object, attribute, pose, relation, or state
- dynamic_action: temporal evolution, an action, a cause, a transition, or an outcome is necessary
- ocr: in-frame text, a number, a brand, or an exact sentence must be located or transcribed
- subscene_caption: a free-text answer must describe a short event or ordered local situation

Use dynamic_action for questions asking what somebody does, how a visible result is produced, or
what happens at a moment. Use subscene_caption for free-text requests to describe a situation,
actions, or what happens; an event phrase in such a question is a localization anchor, not proof
that one still frame is sufficient. Do not choose static_visual merely because an action or result
can be named as a noun phrase. OCR normally requires text_consensus; ordered events require
sequence unless the question explicitly requires full-span coverage.

Create 1-4 concrete required_slots, but include only facts directly required by the question. Do
not make setting, spatial context, motive, or background detail mandatory unless it is explicitly
requested. For a free-text event description, participants, visible actions, objects, and order are
normally sufficient. Use coverage_requirement point, sequence, full_span, or text_consensus.
detail_requests may include spatial_detail, temporal_boundary, or ocr_detail. The output language
records the language expected for a later free-text answer.

ANSWER MODE: {answer_mode}
QUESTION:
{question.strip()}

Return one compact JSON object with exactly these keys:
answer_mode, primary_mode, required_slots, target_entities, target_actions,
target_attributes, target_relations, detail_requests, temporal_hint,
coverage_requirement, output_language.
Each required_slots item has description, required, value_type. Return JSON only."""


def parse_observation_spec(text: str, *, expected_answer_mode: str) -> ObservationSpec:
    if expected_answer_mode not in ANSWER_MODES:
        raise ValueError(f"unsupported answer mode: {expected_answer_mode}")
    payload = _payload(text)
    answer_mode = str(payload.get("answer_mode", expected_answer_mode)).strip().lower()
    if answer_mode != expected_answer_mode:
        raise ProtocolError(
            f"observation compiler changed answer mode from {expected_answer_mode} to {answer_mode}"
        )
    mode = str(payload.get("primary_mode", "")).strip().lower()
    if mode not in PRIMARY_MODES:
        raise ProtocolError(f"invalid primary_mode: {mode!r}")
    raw_slots = payload.get("required_slots")
    if not isinstance(raw_slots, list):
        raise ProtocolError("required_slots must be a list")
    slots: list[ObservationSlot] = []
    for index, raw in enumerate(raw_slots[:4], start=1):
        if not isinstance(raw, dict):
            continue
        description = str(raw.get("description", "")).strip()
        if not description:
            continue
        slots.append(
            ObservationSlot(
                slot_id=f"S{index}",
                description=description,
                required=bool(raw.get("required", True)),
                value_type=str(raw.get("value_type", "fact")).strip() or "fact",
            )
        )
    if not slots:
        raise ProtocolError("observation compiler returned no usable required slots")
    coverage = str(payload.get("coverage_requirement", "point")).strip().lower()
    if coverage not in COVERAGE_REQUIREMENTS:
        raise ProtocolError(f"invalid coverage_requirement: {coverage!r}")
    temporal_hint = payload.get("temporal_hint")
    return ObservationSpec(
        answer_mode=answer_mode,
        primary_mode=mode,
        required_slots=tuple(slots),
        target_entities=_string_tuple(payload.get("target_entities")),
        target_actions=_string_tuple(payload.get("target_actions")),
        target_attributes=_string_tuple(payload.get("target_attributes")),
        target_relations=_string_tuple(payload.get("target_relations")),
        detail_requests=_string_tuple(payload.get("detail_requests")),
        temporal_hint=(str(temporal_hint).strip() if temporal_hint is not None else None),
        coverage_requirement=coverage,
        output_language=str(payload.get("output_language", "same_as_question")).strip()
        or "same_as_question",
    )


def build_locator_prompt(
    question: str,
    spec: ObservationSpec,
    nodes: Sequence[tuple[str, TimeSpan]],
    *,
    max_candidates: int,
) -> str:
    node_lines = "\n".join(
        f"- {node_id}: {span.start_seconds:.3f}s to {span.end_seconds:.3f}s"
        for node_id, span in nodes
    )
    target = {
        "target_entities": list(spec.target_entities),
        "target_actions": list(spec.target_actions),
        "target_attributes": list(spec.target_attributes),
        "target_relations": list(spec.target_relations),
        "primary_mode": spec.primary_mode,
    }
    return f"""You locate visible evidence in a temporal contact sheet.

Do not answer the question. You never see answer choices. Every cell ID and absolute time range is
listed below. This is a coarse routing step, not verification: do not test whether the full event or
answer is visible in a sparse thumbnail.
Select up to {max_candidates} best cells whenever a question-relevant person, object, action,
setting, or text is visibly present. The questioned subject is a valid anchor even when its exact
attribute, identity, action, or text cannot yet be resolved. Use an empty candidates list only when
every supplied cell lacks any usable visual anchor related to the question. Irrelevant visual
salience is not an anchor. For a question that names an event, the event itself does not need to be
captured in this coarse sheet: a named participant or distinctive object is enough to select a cell
for finer inspection. When people recur, prefer cells showing both named participant types, close
interaction, or the clearest relevant person. When several cells qualify, prefer the clearest and
most direct views. It is a protocol error to return empty merely because the exact action, handoff,
or small object is absent when a named participant type is visible.

QUESTION:
{question.strip()}

QUESTION-ONLY OBSERVATION TARGET:
{json.dumps(target, ensure_ascii=False, separators=(",", ":"))}

CELLS:
{node_lines}

Return one compact JSON object:
{{"candidates":[{{"node_id":"...","visible_anchor":"brief visible anchor"}}]}}
node_id must be exactly one listed ID such as S0001, with no time range or commentary appended.
JSON only."""


def parse_locator(
    text: str,
    *,
    valid_node_ids: set[str],
    max_candidates: int,
) -> LocatorDecision:
    payload = _payload(text)
    raw_candidates = payload.get("candidates")
    if not isinstance(raw_candidates, list):
        raise ProtocolError("locator candidates must be a list")
    node_ids: list[str] = []
    anchors: list[str] = []
    for raw in raw_candidates:
        if not isinstance(raw, dict):
            continue
        node_id = _resolve_node_id(raw.get("node_id"), valid_node_ids)
        anchor = str(raw.get("visible_anchor", "")).strip()
        if node_id not in valid_node_ids or node_id in node_ids or not anchor:
            continue
        node_ids.append(node_id)
        anchors.append(anchor)
        if len(node_ids) >= max_candidates:
            break
    return LocatorDecision(tuple(node_ids), tuple(anchors))


def build_scout_prompt(
    question: str,
    spec: ObservationSpec,
    candidate_id: str,
    span: TimeSpan,
    frames: Sequence[FrameRef],
    *,
    purpose: str = "candidate_scout",
    discriminants: Sequence[ClaimTest] = (),
) -> str:
    frame_lines = _frame_lines(frames, span)
    fact_protocol = _fact_protocol(spec.primary_mode)
    rescue_targets = (
        json.dumps(
            [claim.statement for claim in discriminants],
            ensure_ascii=False,
            separators=(",", ":"),
        )
        if discriminants
        else "(hidden during initial scouting)"
    )
    role_description = (
        "label-free discriminator-aware rescue observer"
        if discriminants
        else "question-only local evidence observer"
    )
    return f"""You are a {role_description} ({purpose}).

Answer labels and option mappings are hidden. Inspect only the supplied local media. target_visible means that any
question-relevant observable evidence is present; it does not mean that the final answer is already
certain. Set target_visible=true when the supplied view shows the questioned subject, a relevant
action, or a potentially relevant text region. For first-person footage, do not require the actor's
face or body: visible hands and their object interactions are the camera wearer's or vlogger's
observable action. In a timestamp-bounded first-person interval, if the hands manipulate or pick up
an object, set target_visible=true and record that visible action even if the exact object or action
name remains uncertain. Set target_visible=false only when the relevant subject, action, and text
are all absent or wholly unobservable across the supplied media.

When visible, tightly bound supporting_start/supporting_end to the first and last cited evidence
frames instead of copying the whole candidate range, and record atomic neutral facts for the
required slots. If an exact identity, object name, action boundary, or text is uncertain, preserve
the visible portion as partial evidence and list what remains missing. Do not guess identities,
motives, emotions, unreadable characters, or an answer.

QUESTION:
{question.strip()}

OBSERVATION SPEC:
{json.dumps(spec.to_dict(), ensure_ascii=False, separators=(",", ":"))}

LABEL-FREE RESCUE DISCRIMINANTS:
{rescue_targets}

CANDIDATE: {candidate_id}, {span.start_seconds:.3f}-{span.end_seconds:.3f}s
FRAME MAP:
{frame_lines}

MODE-SPECIFIC FACT PROTOCOL:
{fact_protocol}

Final check before returning JSON: in first-person footage, inspect the visible hands and any object
they touch. Such an interaction means target_visible=true; record the most literal action and use
partial visibility for an uncertain object name. Do not require a visible face or body.

Return one compact JSON object with keys target_visible, supporting_start, supporting_end,
visible_anchor, facts, missing_slot_ids, conflicts. JSON only."""


def parse_scout(
    text: str,
    *,
    candidate_span: TimeSpan,
    valid_slot_ids: set[str],
    frames: Sequence[FrameRef],
    view_id: str,
    fact_prefix: str,
) -> ScoutDecision:
    payload = _payload(text)
    target_visible = bool(payload.get("target_visible", False))
    start = _frame_time_or(payload.get("supporting_start"), frames, candidate_span.start_seconds)
    end = _frame_time_or(payload.get("supporting_end"), frames, candidate_span.end_seconds)
    start = min(max(start, candidate_span.start_seconds), candidate_span.end_seconds)
    end = min(max(end, candidate_span.start_seconds), candidate_span.end_seconds)
    if end <= start:
        start, end = candidate_span.start_seconds, candidate_span.end_seconds
    supporting = TimeSpan(start, end, source="scout_support")
    facts = parse_facts(
        payload.get("facts"),
        valid_slot_ids=valid_slot_ids,
        frames=frames,
        span=candidate_span,
        view_id=view_id,
        fact_prefix=fact_prefix,
    )
    missing = tuple(
        item for item in _string_tuple(payload.get("missing_slot_ids")) if item in valid_slot_ids
    )
    conflicts = _string_tuple(payload.get("conflicts"))
    anchor = str(payload.get("visible_anchor", "")).strip()
    if target_visible and not anchor:
        anchor = next((fact.statement for fact in facts if fact.visibility == "clear"), "")
    return ScoutDecision(
        target_visible=target_visible,
        supporting_span=supporting,
        visible_anchor=anchor,
        facts=facts,
        missing_slot_ids=missing,
        conflicts=conflicts,
    )


def build_hypothesis_compiler_prompt(
    question: str,
    options: Sequence[CanonicalOption],
    spec: ObservationSpec,
) -> str:
    option_lines = "\n".join(f"{option.option_id}: {option.text}" for option in options)
    slot_lines = "\n".join(f"{slot.slot_id}: {slot.description}" for slot in spec.required_slots)
    return f"""You compile label-free visual discriminants for a multiple-choice local-video task.

Do not answer the question. Compare all options and identify the smallest concrete observations
that distinguish them. Split overlapping options along decisive visual axes such as actor, direct
action, manipulated object, causal source, target, result, order, exact text, or exact number. A
claim is atomic and independently observable in one continuous local span. Inside claim statements
and expected values, never use A/B/C/D, option IDs, placeholders, generic answer language, or
several alternative values. Use internal IDs such as O1 only in option_rules to map those neutral
claims back to the supplied options.
Share a claim only when the same observable fact genuinely belongs to multiple options. Use all_of
for facts an option requires and none_of for facts that exclude it. A "Cannot be determined" option
is an ordinary semantic option; pipeline uncertainty is never evidence for it. Prefer at most
{max(4, min(8, len(options) * 2))} discriminants and keep every statement under 30 words.

QUESTION:
{question.strip()}

QUESTION-ONLY SLOTS:
{slot_lines}

INTERNAL OPTIONS:
{option_lines}

Return one compact JSON object with claim_tests, option_rules, cannot_determine_option_id.
Each claim_test has claim_id, statement, slot_ids, predicate, expected_value. Each option_rule has
option_id, all_of, none_of. expected_value is one concrete value, never a placeholder or list.
Include one distinguishable rule for every non-identical listed option. claim_tests must never be
empty. all_of and none_of contain claim-ID strings only: never claim objects or prose.

Exact shape example for two hypothetical distinct options:
{{"claim_tests":[{{"claim_id":"C1","statement":"The queried object is red.","slot_ids":["S1"],"predicate":"equals","expected_value":"red"}},{{"claim_id":"C2","statement":"The queried object is blue.","slot_ids":["S1"],"predicate":"equals","expected_value":"blue"}}],"option_rules":[{{"option_id":"O1","all_of":["C1"],"none_of":[]}},{{"option_id":"O2","all_of":["C2"],"none_of":[]}}],"cannot_determine_option_id":null}}

JSON only."""


def build_rescue_locator_prompt(
    question: str,
    spec: ObservationSpec,
    discriminants: Sequence[ClaimTest],
    frames: Sequence[FrameRef],
    *,
    max_candidates: int,
) -> str:
    targets = {
        "primary_mode": spec.primary_mode,
        "target_entities": list(spec.target_entities),
        "target_actions": list(spec.target_actions),
        "target_attributes": list(spec.target_attributes),
        "target_relations": list(spec.target_relations),
        "label_free_discriminants": [item.statement for item in discriminants],
    }
    return f"""You are a bounded global rescue locator, not an answerer.

The initial question-only locator failed or found unusable candidates. Inspect the supplied ordered
overview and rank the {max_candidates} most useful frame anchors for a later local inspection. You
must return at least one candidate even when every anchor is weak; this rescue is relative ranking,
not an evidence-certification step. Do not answer the question, mention an option label, or infer an
answer from the discriminants. Prefer distinct temporal neighborhoods and visible actors, objects,
actions, text regions, transitions, or outcomes that could resolve the targets.

QUESTION:
{question.strip()}

LABEL-FREE TARGETS:
{json.dumps(targets, ensure_ascii=False, separators=(",", ":"))}

FRAME MAP:
{_frame_lines(frames, TimeSpan(0.0, max(0.001, max((f.timestamp_seconds for f in frames), default=0.001)), source="rescue_overview"))}

Return one compact JSON object:
{{"candidates":[{{"frame_id":"exact supplied frame ID","visible_anchor":"brief relative reason"}}]}}
Return one to {max_candidates} candidates. JSON only."""


def parse_rescue_locator(
    text: str,
    *,
    frames: Sequence[FrameRef],
    max_candidates: int,
) -> RescueLocatorDecision:
    payload = _payload(text)
    raw_candidates = payload.get("candidates")
    if not isinstance(raw_candidates, list):
        raise ProtocolError("rescue locator candidates must be a list")
    valid_ids = {frame.id for frame in frames}
    frame_ids: list[str] = []
    anchors: list[str] = []
    for raw in raw_candidates:
        if not isinstance(raw, dict):
            continue
        references = _frame_references([raw.get("frame_id")], frames)
        frame_id = references[0] if references else ""
        anchor = str(raw.get("visible_anchor", "")).strip() or "weak relative anchor"
        if frame_id not in valid_ids or frame_id in frame_ids:
            continue
        frame_ids.append(frame_id)
        anchors.append(anchor)
        if len(frame_ids) >= max_candidates:
            break
    if not frame_ids:
        raise ProtocolError("rescue locator must rank at least one supplied frame")
    return RescueLocatorDecision(tuple(frame_ids), tuple(anchors))


def build_choice_decision_prompt(
    question: str,
    options: Sequence[CanonicalOption],
    decision_spec: DecisionSpec,
    packet: EvidencePacket,
    frames: Sequence[FrameRef],
    *,
    stage: str,
) -> str:
    option_lines = "\n".join(
        f"- {option.option_id} ({option.benchmark_label}): {option.text}" for option in options
    )
    return f"""You are the {stage} local multiple-choice decision pass.

Inspect only the supplied continuous local media. The EvidencePacket is an observation aid, not a
gate and not unquestionable truth: correct it when the media visibly disagrees. Compare options on
the label-free discriminants, especially actor/action/object/target/result/order and exact text or
number. You must select exactly one listed option even if evidence is weak. Do not treat pipeline
warnings, missing evidence, or localization failure as support for a "Cannot be determined"
option. Score every option from 0 to 3 for visible support and contradiction. Cite only supplied
fact IDs and source frame IDs. List discriminants that remain unresolved; never omit a final choice.

QUESTION:
{question.strip()}

OPTIONS:
{option_lines}

LABEL-FREE DISCRIMINANT SPEC:
{json.dumps(decision_spec.to_dict(), ensure_ascii=False, separators=(",", ":"))}

EVIDENCE PACKET:
{json.dumps(packet.to_dict(), ensure_ascii=False, separators=(",", ":"))}

FRAME MAP:
{_frame_lines(frames, packet.canonical_span)}

Put selected_option_id first. Return one compact JSON object with selected_option_id, option_assessments,
resolved_discriminant_ids, unresolved_discriminant_ids, reason. option_assessments must contain
every option exactly once and each item has option_id, support_score, contradiction_score,
discriminant_ids, evidence_fact_ids, source_frame_ids, reason. Cite at most two decisive source
frame IDs per option assessment and keep every reason under 20 words. JSON only."""


def parse_choice_decision(
    text: str,
    *,
    options: Sequence[CanonicalOption],
    decision_spec: DecisionSpec,
    packet: EvidencePacket,
    frames: Sequence[FrameRef],
) -> ChoiceDecision:
    payload = _payload(text)
    valid_options = {option.option_id for option in options}
    selected = str(payload.get("selected_option_id", "")).strip()
    if selected not in valid_options:
        raise ProtocolError(f"decision selected invalid option: {selected!r}")
    valid_discriminants = {claim.claim_id for claim in decision_spec.claim_tests}
    valid_facts = {fact.fact_id for fact in packet.facts}
    raw_assessments = payload.get("option_assessments")
    if not isinstance(raw_assessments, list):
        raise ProtocolError("option_assessments must be a list")
    assessments: list[OptionAssessment] = []
    seen: set[str] = set()
    for raw in raw_assessments:
        if not isinstance(raw, dict):
            continue
        option_id = str(raw.get("option_id", "")).strip()
        if option_id not in valid_options or option_id in seen:
            continue
        try:
            support = max(0, min(3, int(raw.get("support_score", 0))))
            contradiction = max(0, min(3, int(raw.get("contradiction_score", 0))))
        except (TypeError, ValueError) as exc:
            raise ProtocolError(f"invalid assessment scores for {option_id}") from exc
        assessments.append(
            OptionAssessment(
                option_id=option_id,
                support_score=support,
                contradiction_score=contradiction,
                discriminant_ids=tuple(
                    item
                    for item in _string_tuple(raw.get("discriminant_ids"))
                    if item in valid_discriminants
                ),
                evidence_fact_ids=tuple(
                    item
                    for item in _string_tuple(raw.get("evidence_fact_ids"))
                    if item in valid_facts
                ),
                source_frame_ids=_frame_references(raw.get("source_frame_ids"), frames),
                reason=str(raw.get("reason", "")).strip(),
            )
        )
        seen.add(option_id)
    if seen != valid_options:
        raise ProtocolError(
            "decision omitted option assessments: " + ", ".join(sorted(valid_options - seen))
        )
    resolved = tuple(
        item
        for item in _string_tuple(payload.get("resolved_discriminant_ids"))
        if item in valid_discriminants
    )
    unresolved = tuple(
        item
        for item in _string_tuple(payload.get("unresolved_discriminant_ids"))
        if item in valid_discriminants and item not in resolved
    )
    return ChoiceDecision(
        selected_option_id=selected,
        option_assessments=tuple(assessments),
        resolved_discriminant_ids=resolved,
        unresolved_discriminant_ids=unresolved,
        reason=str(payload.get("reason", "")).strip(),
    )


def extract_option_id_from_text(
    text: str,
    options: Sequence[CanonicalOption],
) -> str | None:
    by_label = {option.benchmark_label.upper(): option.option_id for option in options}
    valid_ids = {option.option_id for option in options}
    id_match = re.search(r"\bO(?:PTION\s*)?([0-9]{1,2})\b", text, flags=re.IGNORECASE)
    if id_match is not None:
        option_id = f"O{int(id_match.group(1))}"
        if option_id in valid_ids:
            return option_id
    label_match = re.search(
        r"(?:ANSWER|OPTION|CHOICE|SELECTED_OPTION_ID)?\s*[:=\-]?\s*\b([A-Z])\b",
        text.upper(),
    )
    if label_match is not None:
        return by_label.get(label_match.group(1))
    return None


def compile_decision_spec(
    question: str,
    options: Sequence[CanonicalOption],
    spec: ObservationSpec,
) -> DecisionSpec:
    """Deterministically wrap option content as unlabeled observable hypotheses."""
    if not options:
        raise ValueError("decision compilation requires at least one option")
    question_text = " ".join(question.strip().split())
    claim_by_text: dict[str, ClaimTest] = {}
    claims: list[ClaimTest] = []
    rules: list[OptionRule] = []
    cannot_candidates: list[str] = []
    cannot_phrases = (
        "cannot be determined",
        "cannot determine",
        "not enough information",
        "insufficient information",
        "not possible to determine",
        "unknown from the video",
    )
    for option in options:
        normalized = " ".join(option.text.casefold().split())
        claim = claim_by_text.get(normalized)
        if claim is None:
            claim_id = f"C{len(claims) + 1}"
            claim = ClaimTest(
                claim_id=claim_id,
                statement=(
                    f'For the visual question "{question_text}", the complete observable '
                    f'answer is: "{option.text.strip()}"'
                ),
                slot_ids=tuple(sorted(spec.required_slot_ids)),
                predicate="matches_complete_option",
                expected_value=option.text.strip(),
            )
            claim_by_text[normalized] = claim
            claims.append(claim)
        rules.append(OptionRule(option.option_id, (claim.claim_id,), ()))
        if any(phrase in normalized for phrase in cannot_phrases):
            cannot_candidates.append(option.option_id)
    cannot_id = cannot_candidates[0] if len(cannot_candidates) == 1 else None
    return DecisionSpec(tuple(claims), tuple(rules), cannot_id)


def parse_decision_spec(
    text: str,
    *,
    options: Sequence[CanonicalOption],
    valid_slot_ids: set[str],
) -> DecisionSpec:
    payload = _payload(text)
    raw_claims = payload.get("claim_tests")
    raw_rules = payload.get("option_rules")
    inline_claims: list[dict[str, Any]] = []
    normalized_rules: list[Any] = []
    if isinstance(raw_rules, list):
        for raw_rule in raw_rules:
            if not isinstance(raw_rule, dict):
                normalized_rules.append(raw_rule)
                continue
            normalized_rule = dict(raw_rule)
            for field_name in ("all_of", "none_of"):
                references = raw_rule.get(field_name)
                if not isinstance(references, list):
                    continue
                normalized_references: list[Any] = []
                for reference in references:
                    if not isinstance(reference, dict):
                        normalized_references.append(reference)
                        continue
                    inline = dict(reference)
                    claim_id = str(
                        inline.get("claim_id") or f"C_INLINE_{len(inline_claims) + 1}"
                    ).strip()
                    inline["claim_id"] = claim_id
                    inline_claims.append(inline)
                    normalized_references.append(claim_id)
                normalized_rule[field_name] = normalized_references
            normalized_rules.append(normalized_rule)
    if inline_claims:
        raw_claims = [
            *(raw_claims if isinstance(raw_claims, list) else []),
            *inline_claims,
        ]
        raw_rules = normalized_rules
    if not isinstance(raw_claims, list):
        raise ProtocolError("claim_tests must be a list")
    claims: list[ClaimTest] = []
    seen_claim_ids: set[str] = set()
    for index, raw in enumerate(raw_claims, start=1):
        if not isinstance(raw, dict):
            continue
        claim_id = str(raw.get("claim_id", f"C{index}")).strip()
        statement = str(raw.get("statement", "")).strip()
        if not claim_id or claim_id in seen_claim_ids or not statement:
            continue
        seen_claim_ids.add(claim_id)
        claims.append(
            ClaimTest(
                claim_id=claim_id,
                statement=statement,
                slot_ids=tuple(
                    slot_id
                    for slot_id in _string_tuple(raw.get("slot_ids"))
                    if slot_id in valid_slot_ids
                ),
                predicate=str(raw.get("predicate", "freeform")).strip() or "freeform",
                expected_value=str(raw.get("expected_value", "")).strip(),
            )
        )
    if not claims:
        raise ProtocolError("hypothesis compiler returned no usable claims")
    max_claims = max(4, min(8, len(options) * 2))
    if len(claims) > max_claims:
        raise ProtocolError(
            f"hypothesis compiler exceeded the {max_claims}-claim bound"
        )
    claim_ids = {claim.claim_id for claim in claims}
    valid_options = {option.option_id for option in options}
    if not isinstance(raw_rules, list):
        raise ProtocolError("option_rules must be a list")
    rules: list[OptionRule] = []
    seen_options: set[str] = set()
    for raw in raw_rules:
        if not isinstance(raw, dict):
            continue
        option_id = str(raw.get("option_id", "")).strip()
        if option_id not in valid_options or option_id in seen_options:
            continue
        all_of = tuple(
            claim_id for claim_id in _string_tuple(raw.get("all_of")) if claim_id in claim_ids
        )
        none_of = tuple(
            claim_id for claim_id in _string_tuple(raw.get("none_of")) if claim_id in claim_ids
        )
        if not all_of:
            raise ProtocolError(f"option rule {option_id} has no valid all_of claims")
        seen_options.add(option_id)
        rules.append(OptionRule(option_id, all_of, none_of))
    if seen_options != valid_options:
        missing = sorted(valid_options - seen_options)
        raise ProtocolError(f"hypothesis compiler omitted option rules: {', '.join(missing)}")
    by_id = {option.option_id: option for option in options}
    signatures: dict[tuple[tuple[str, ...], tuple[str, ...]], str] = {}
    for rule in rules:
        signature = (tuple(sorted(rule.all_of)), tuple(sorted(rule.none_of)))
        previous_id = signatures.get(signature)
        if previous_id is not None:
            previous_text = " ".join(by_id[previous_id].text.casefold().split())
            current_text = " ".join(by_id[rule.option_id].text.casefold().split())
            if previous_text != current_text:
                raise ProtocolError(
                    f"distinct options {previous_id} and {rule.option_id} have identical rules"
                )
        signatures[signature] = rule.option_id
    referenced_claim_ids = {
        claim_id for rule in rules for claim_id in (*rule.all_of, *rule.none_of)
    }
    claims = [claim for claim in claims if claim.claim_id in referenced_claim_ids]
    placeholder_tokens = ("[color]", "[text]", "[value]", "<color>", "<text>", "<value>")
    for claim in claims:
        normalized_statement = claim.statement.casefold()
        normalized_value = claim.expected_value.casefold().strip()
        if any(
            token in normalized_statement for token in placeholder_tokens
        ) or normalized_value in {
            "color",
            "text",
            "value",
            "answer",
            "option",
        }:
            raise ProtocolError(f"claim {claim.claim_id} contains a non-concrete placeholder")
    cannot = payload.get("cannot_determine_option_id")
    cannot_id = str(cannot).strip() if cannot is not None else None
    if cannot_id not in valid_options:
        cannot_id = None
    return DecisionSpec(tuple(claims), tuple(rules), cannot_id)


def build_refinement_prompt(
    question: str,
    spec: ObservationSpec,
    span: TimeSpan,
    frames: Sequence[FrameRef],
    existing_facts: Sequence[Fact],
    *,
    target_slot_ids: Sequence[str],
    target_claims: Sequence[ClaimTest],
) -> str:
    facts = [fact.to_dict() for fact in existing_facts]
    fact_protocol = _fact_protocol(spec.primary_mode)
    claims = [{"claim_id": claim.claim_id, "statement": claim.statement} for claim in target_claims]
    return f"""You perform the one allowed targeted re-observation of a bounded local span.

Do not answer the question. Answer-choice labels and mappings are hidden. Inspect the denser,
extended, or cropped media only to fill missing slots and test the unlabeled statements. Preserve
conflicts and mark unreadable or occluded details instead of guessing. Return newly observed facts
in the CandidateScout schema; do not copy internal fact_id or view_id fields from EXISTING FACTS.
Every new fact must repeat kind and all kind-specific fields. source_frame_ids and
consensus_frame_ids must be exact quoted IDs from the current FRAME MAP, never timestamps or visual
numbers. OCR bbox.frame_id must be an original non-CROP frame ID. target_visible means relevant
observable evidence is present, not that the final answer is certain. In first-person footage,
visible hands manipulating an object count as the vlogger's action even when the person is off
camera. A text fact must explicitly include exact_text, uncertain_characters, bbox, and
consensus_frame_ids.

QUESTION:
{question.strip()}

OBSERVATION SPEC:
{json.dumps(spec.to_dict(), ensure_ascii=False, separators=(",", ":"))}
TARGET SLOTS: {json.dumps(list(target_slot_ids), ensure_ascii=False)}
UNLABELED CLAIMS: {json.dumps(claims, ensure_ascii=False)}
EXISTING FACTS: {json.dumps(facts, ensure_ascii=False, separators=(",", ":"))}
SPAN: {span.start_seconds:.3f}-{span.end_seconds:.3f}s
FRAME MAP:
{_frame_lines(frames, span)}

MODE-SPECIFIC FACT PROTOCOL:
{fact_protocol}

Final first-person check: visible hands touching, picking up, placing, or using an object are the
vlogger's observable action. If present, target_visible must be true and the literal interaction
must be recorded, with partial visibility when the object name is uncertain.

Return the same compact JSON schema as a CandidateScout: target_visible, supporting_start,
supporting_end, visible_anchor, facts, missing_slot_ids, conflicts. JSON only."""


def build_verifier_prompt(
    question: str,
    claims: Sequence[ClaimTest],
    facts: Sequence[Fact],
    frames: Sequence[FrameRef],
    span: TimeSpan,
) -> str:
    claim_lines = "\n".join(f"- {claim.claim_id}: {claim.statement}" for claim in claims)
    required_claim_ids = ", ".join(claim.claim_id for claim in claims)
    fact_lines = "\n".join(
        f"- {fact.fact_id}: {fact.statement} [{fact.start_seconds:.3f}-{fact.end_seconds:.3f}s]"
        for fact in facts
    )
    text_consensus_policy = (
        "For every verdict about in-frame text, cite exactly two distinct adjacent verification "
        "frame IDs that independently show the same text. One readable frame is insufficient; "
        "return not_established if two such frames are unavailable."
        if any(isinstance(fact, TextFact) for fact in facts)
        else ""
    )
    return f"""You are a blinded claim verifier using an independent visual view.

You do not see answer choices, option IDs, or a mapping from claims to answers. For every supplied
claim, return entailed, contradicted, or not_established. Re-check the independent media rather than
trusting the existing fact text. Every non-not_established verdict must cite exactly one decisive
supplied verification frame ID, or at most two adjacent IDs when text consensus is necessary. Never
copy the whole frame map. Keep each reason under 15 words. Do not introduce new claims or answer the
question.
{text_consensus_policy}

QUESTION:
{question.strip()}

UNLABELED CLAIMS:
{claim_lines}

EXISTING QUESTION-NEUTRAL FACTS:
{fact_lines or "(none)"}

CANONICAL SPAN: {span.start_seconds:.3f}-{span.end_seconds:.3f}s
VERIFICATION FRAME MAP:
{_frame_lines(frames, span)}

Return one compact object whose top-level shape is {{"verdicts":[...]}}. The verdicts list must
contain exactly {len(claims)} items, one for every claim in this order: {required_claim_ids}. Do not
stop after the first claim. Every item has exactly claim_id, verdict, evidence_fact_ids,
verification_frame_ids, reason. JSON only."""


def parse_verifier(
    text: str,
    *,
    valid_claim_ids: set[str],
    valid_fact_ids: set[str],
    frames: Sequence[FrameRef],
) -> tuple[ClaimVerdict, ...]:
    payload = _payload(text)
    raw_verdicts = payload.get("verdicts")
    if not isinstance(raw_verdicts, list):
        raise ProtocolError("verifier verdicts must be a list")
    results: list[ClaimVerdict] = []
    seen: set[str] = set()
    for raw in raw_verdicts:
        if not isinstance(raw, dict):
            continue
        claim_id = str(raw.get("claim_id", "")).strip()
        verdict = str(raw.get("verdict", "")).strip().lower()
        if claim_id not in valid_claim_ids or claim_id in seen:
            continue
        if verdict not in {"entailed", "contradicted", "not_established"}:
            raise ProtocolError(f"invalid verdict for {claim_id}: {verdict!r}")
        fact_ids = tuple(
            item for item in _string_tuple(raw.get("evidence_fact_ids")) if item in valid_fact_ids
        )
        frame_ids = _frame_references(raw.get("verification_frame_ids"), frames)
        reason = str(raw.get("reason", "")).strip()
        if verdict != "not_established" and not frame_ids:
            verdict = "not_established"
            reason = f"{reason} [downgraded: no valid verification citation]".strip()
        seen.add(claim_id)
        results.append(
            ClaimVerdict(
                claim_id=claim_id,
                verdict=verdict,
                evidence_fact_ids=fact_ids,
                verification_frame_ids=frame_ids,
                reason=reason,
            )
        )
    missing = valid_claim_ids - seen
    if missing:
        raise ProtocolError(f"verifier omitted claims: {', '.join(sorted(missing))}")
    return tuple(results)


def build_answer_composer_prompt(
    question: str,
    verified_facts: Sequence[Fact],
    *,
    output_language: str,
    missing_slot_ids: Sequence[str] = (),
    frames: Sequence[FrameRef] = (),
    span: TimeSpan | None = None,
) -> str:
    facts = [fact.to_dict() for fact in verified_facts]
    frame_map = _frame_lines(frames, span) if frames and span is not None else "(no media)"
    return f"""Write a concise best-effort answer for a local video event.

Inspect the supplied local media when present and use the neutral evidence facts as an aid. Mention
required people or objects, the key action, its object or mechanism, order, and visible result when
available. Try to complete the event rather than stopping at its first action. Do not add motives,
emotions, relationships, or identities that are absent from the media. Missing slots are diagnostic
only and must not cause an empty answer. Use output language: {output_language}.

QUESTION:
{question.strip()}

OBSERVED FACTS:
{json.dumps(facts, ensure_ascii=False, separators=(",", ":"))}

MISSING SLOT IDS:
{json.dumps(list(missing_slot_ids), ensure_ascii=False)}

FRAME MAP:
{frame_map}

Return only the final answer, without JSON or explanation."""


def build_protocol_repair_prompt(protocol: str, raw_response: str, schema_hint: str) -> str:
    return f"""Repair a malformed structured response without solving the video task.

Do not add evidence, change an existing decision, infer missing visual facts, or answer the
question. Reformat only the SOURCE RESPONSE into one JSON object matching the target schema.

PROTOCOL: {protocol}
TARGET SCHEMA: {schema_hint}
SOURCE RESPONSE:
{raw_response}

Return JSON only."""


def parse_facts(
    raw_facts: Any,
    *,
    valid_slot_ids: set[str],
    frames: Sequence[FrameRef],
    span: TimeSpan,
    view_id: str,
    fact_prefix: str,
) -> tuple[Fact, ...]:
    if not isinstance(raw_facts, list):
        raise ProtocolError("facts must be a list")
    frame_times = {frame.id: frame.timestamp_seconds for frame in frames}
    results: list[Fact] = []
    for index, raw in enumerate(raw_facts, start=1):
        if not isinstance(raw, dict):
            continue
        kind = str(raw.get("kind", "")).strip().lower()
        slot_ids = tuple(
            slot_id for slot_id in _string_tuple(raw.get("slot_ids")) if slot_id in valid_slot_ids
        )
        statement = str(raw.get("statement", "")).strip()
        visibility = str(raw.get("visibility", "")).strip().lower()
        if visibility not in VISIBILITIES or not statement:
            continue
        source_ids = _frame_references(raw.get("source_frame_ids"), frames)
        default_time = frame_times[source_ids[0]] if source_ids else span.midpoint_seconds
        start = _float_or(raw.get("start_seconds"), default_time)
        end = _float_or(raw.get("end_seconds"), start)
        start = min(max(start, span.decode_start_seconds), span.decode_end_seconds)
        end = min(max(end, start), span.decode_end_seconds)
        common = {
            "fact_id": f"{fact_prefix}-F{index}",
            "slot_ids": slot_ids,
            "start_seconds": start,
            "end_seconds": end,
            "visibility": visibility,
            "statement": statement,
            "source_frame_ids": source_ids,
            "view_id": view_id,
        }
        if kind == "static":
            results.append(
                StaticFact(
                    **common,
                    entity=str(raw.get("entity", "")).strip(),
                    attribute=str(raw.get("attribute", "")).strip(),
                    relation=str(raw.get("relation", "")).strip(),
                    value=str(raw.get("value", "")).strip(),
                )
            )
        elif kind == "event":
            order = raw.get("order")
            try:
                parsed_order = int(order) if order is not None else None
            except (TypeError, ValueError):
                parsed_order = None
            results.append(
                EventFact(
                    **common,
                    subject=str(raw.get("subject", "")).strip(),
                    initial_state=str(raw.get("initial_state", "")).strip(),
                    action=str(raw.get("action", "")).strip(),
                    object=str(raw.get("object", "")).strip(),
                    target=str(raw.get("target", "")).strip(),
                    result=str(raw.get("result", "")).strip(),
                    order=parsed_order,
                )
            )
        elif kind == "text":
            bbox = _parse_bbox(raw.get("bbox"), frames=frames)
            consensus = _frame_references(raw.get("consensus_frame_ids"), frames)
            exact_text = str(raw.get("exact_text", "")).strip()
            if not exact_text and bbox is not None and consensus:
                exact_text = statement
            results.append(
                TextFact(
                    **common,
                    exact_text=exact_text,
                    uncertain_characters=str(raw.get("uncertain_characters", "")).strip(),
                    bbox=bbox,
                    consensus_frame_ids=consensus,
                )
            )
    return tuple(results)


def _parse_bbox(raw: Any, *, frames: Sequence[FrameRef]) -> BoundingBox | None:
    if not isinstance(raw, dict):
        return None
    resolved = _frame_references([raw.get("frame_id")], frames)
    frame_id = resolved[0] if resolved else ""
    if not frame_id or frame_id.startswith("CROP-"):
        return None
    try:
        return BoundingBox(
            frame_id=frame_id,
            x1=int(raw["x1"]),
            y1=int(raw["y1"]),
            x2=int(raw["x2"]),
            y2=int(raw["y2"]),
        )
    except (KeyError, TypeError, ValueError):
        return None


def _payload(text: str) -> dict[str, Any]:
    try:
        return parse_json_object(text)
    except DecisionError as exc:
        raise ProtocolError(str(exc)) from exc


def _string_tuple(value: Any) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    return tuple(dict.fromkeys(str(item).strip() for item in value if str(item).strip()))


def _frame_references(value: Any, frames: Sequence[FrameRef]) -> tuple[str, ...]:
    """Resolve exact/decorated IDs and unambiguous sampled timestamps."""
    if not isinstance(value, list):
        return ()
    valid_ids = {frame.id for frame in frames}
    results: list[str] = []
    for item in value:
        token = str(item).strip()
        resolved: str | None = token if token in valid_ids else None
        if resolved is None:
            embedded_ids = [
                frame_id
                for frame_id in valid_ids
                if re.search(
                    rf"(?<![A-Za-z0-9_-]){re.escape(frame_id)}(?![A-Za-z0-9_-])",
                    token,
                )
            ]
            if len(embedded_ids) == 1:
                resolved = embedded_ids[0]
        if resolved is None:
            visual_match = re.fullmatch(r"(?:visual|frame)\s*#?(\d+)", token.casefold())
            if visual_match is not None:
                visual_index = int(visual_match.group(1)) - 1
                if 0 <= visual_index < len(frames):
                    resolved = frames[visual_index].id
        if resolved is None:
            try:
                timestamp = float(item)
            except (TypeError, ValueError):
                timestamp = None
            if timestamp is not None:
                nearest = min(
                    frames,
                    key=lambda frame: abs(frame.timestamp_seconds - timestamp),
                    default=None,
                )
                if nearest is not None and abs(nearest.timestamp_seconds - timestamp) <= 0.02:
                    resolved = nearest.id
        if resolved is not None and resolved not in results:
            results.append(resolved)
    return tuple(results)


def _resolve_node_id(value: Any, valid_node_ids: set[str]) -> str:
    token = str(value).strip()
    if token in valid_node_ids:
        return token
    normalized = re.sub(r"[^A-Z0-9]", "", token.upper())
    matches = [
        node_id
        for node_id in valid_node_ids
        if normalized.startswith(re.sub(r"[^A-Z0-9]", "", node_id.upper()))
    ]
    return matches[0] if len(matches) == 1 else ""


def _fact_protocol(mode: str) -> str:
    common = (
        "Every fact must include kind, slot_ids, start_seconds, end_seconds, visibility, "
        "statement, and source_frame_ids. source_frame_ids and consensus_frame_ids must use "
        "exact quoted IDs copied from FRAME MAP, never timestamps, visual numbers, or invented "
        "IDs. Cite only 1-4 decisive frame IDs per fact, including event boundary or result views "
        "when needed; never copy the whole frame map. Return at most 4 facts. visibility is clear, "
        "partial, occluded, not_visible, or conflicting."
    )
    if mode == "static_visual":
        return f"Use only kind=static with entity, attribute, relation, and value. {common}"
    if mode in {"dynamic_action", "subscene_caption"}:
        return (
            "Use only kind=event with subject, initial_state, action, object, target, result, "
            f"and order. {common}"
        )
    if mode == "ocr":
        return (
            "Use only kind=text with exact_text, uncertain_characters, "
            "bbox={frame_id,x1,y1,x2,y2}, and consensus_frame_ids. A clear fact needs a bbox "
            "on an original non-CROP frame and at least two agreeing adjacent frames; otherwise "
            f"mark it partial. Coordinates are normalized integers from 0 to 1000. {common}"
        )
    raise ValueError(f"unsupported observation mode: {mode}")


def _float_or(value: Any, default: float) -> float:
    try:
        token = value.strip().removesuffix("s") if isinstance(value, str) else value
        return float(token)
    except (TypeError, ValueError):
        return default


def _frame_time_or(
    value: Any,
    frames: Sequence[FrameRef],
    default: float,
) -> float:
    token = str(value).strip()
    for frame in frames:
        if token == frame.id:
            return frame.timestamp_seconds
    return _float_or(value, default)


def _frame_lines(frames: Sequence[FrameRef], span: TimeSpan) -> str:
    lines: list[str] = []
    for index, frame in enumerate(frames, start=1):
        context = " context_only" if not span.contains_evidence(frame.timestamp_seconds) else ""
        lines.append(f"- visual {index}: {frame.id} @ {frame.timestamp_seconds:.3f}s{context}")
    return "\n".join(lines) or "(no frames)"


def claim_tests_from_facts(facts: Iterable[Fact]) -> tuple[ClaimTest, ...]:
    return tuple(
        ClaimTest(
            claim_id=f"VF{index}",
            statement=fact.statement,
            slot_ids=fact.slot_ids,
        )
        for index, fact in enumerate(facts, start=1)
        if fact.visibility in {"clear", "partial", "conflicting"}
    )
