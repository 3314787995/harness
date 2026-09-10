"""Deterministic scope, provenance, coverage and stopping rules for R1."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Iterable
from dataclasses import replace
from itertools import combinations
from typing import Any

from qwen3vl_agent.p01.types import TimeSpan
from qwen3vl_agent.r1.types import (
    DiscriminantSpec,
    EvidenceAudit,
    EvidenceBundle,
    EvidenceFact,
    QueryField,
    QuerySpec,
    R1Request,
    span_from,
)

MODES = {"static", "ordered", "ocr", "caption"}


class ProtocolError(ValueError):
    pass


class BudgetExhausted(RuntimeError):
    pass


def json_object(raw: str) -> dict[str, Any]:
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    try:
        value = json.loads(text)
    except (ValueError, TypeError) as exc:
        raise ProtocolError("expected one complete JSON object") from exc
    if not isinstance(value, dict):
        raise ProtocolError("expected a JSON object")
    return value


def strings(value: Any, name: str, *, limit: int = 32, deduplicate: bool = True) -> tuple[str, ...]:
    if not isinstance(value, list) or len(value) > limit:
        raise ProtocolError(f"{name} must be an array of at most {limit} strings")
    if any(not isinstance(item, str) or not item.strip() for item in value):
        raise ProtocolError(f"{name} contains invalid strings")
    result = tuple(item.strip() for item in value)
    return tuple(dict.fromkeys(result)) if deduplicate else result


def explicit_interval(question: str) -> TimeSpan | None:
    token = r"(?:\d{1,2}:)?\d{1,2}:\d{2}(?:\.\d+)?"
    match = re.search(rf"({token})\s*(?:-|–|—|to|至|到)\s*({token})", question, re.IGNORECASE)
    if match:

        def seconds(text: str) -> float:
            result = 0.0
            for part in text.split(":"):
                result = result * 60 + float(part)
            return result

        return TimeSpan(seconds(match[1]), seconds(match[2]), source="question_interval")
    match = re.search(
        r"(?:from|between)\s+(\d+(?:\.\d+)?)\s*(?:s|seconds?)?\s*"
        r"(?:to|and|-)\s*(\d+(?:\.\d+)?)\s*(?:s\b|seconds?\b)",
        question,
        re.IGNORECASE,
    )
    if match:
        return TimeSpan(float(match[1]), float(match[2]), source="question_interval")
    match = re.search(
        r"(?:first\s+(\d+(?:\.\d+)?)\s*seconds?|前\s*(\d+(?:\.\d+)?)\s*秒)", question, re.IGNORECASE
    )
    if match:
        return TimeSpan(0, float(match[1] or match[2]), source="question_interval")
    return None


def contains(outer: TimeSpan, inner: TimeSpan) -> bool:
    return (
        outer.start_seconds <= inner.start_seconds + 1e-6
        and inner.end_seconds <= outer.end_seconds + 1e-6
    )


def intersection(a: TimeSpan, b: TimeSpan) -> TimeSpan | None:
    start, end = max(a.start_seconds, b.start_seconds), min(a.end_seconds, b.end_seconds)
    return TimeSpan(start, end) if end > start else None


def resolve_scopes(request: R1Request, duration: float) -> tuple[TimeSpan, TimeSpan | None]:
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("video duration must be finite and positive")
    full = TimeSpan(0, duration)
    allowed = request.allowed_scope or full
    if not contains(full, allowed):
        raise ValueError("allowed_scope exceeds the source video")
    parsed = explicit_interval(request.question)
    query = request.query_scope or parsed
    if (
        request.query_scope
        and parsed
        and (
            abs(parsed.start_seconds - query.start_seconds) > 0.001
            or abs(parsed.end_seconds - query.end_seconds) > 0.001
        )
    ):
        raise ValueError("structured and question query scopes disagree")
    if query and not contains(allowed, query):
        raise ValueError("query_scope exceeds allowed_scope")
    return allowed, query


def parse_query(data: dict[str, Any]) -> QuerySpec:
    raw_fields = data.get("fields")
    if not isinstance(raw_fields, list) or not 1 <= len(raw_fields) <= 12:
        raise ProtocolError("query requires 1 to 12 fields")
    result = []
    for i, item in enumerate(raw_fields):
        if not isinstance(item, dict) or not str(item.get("description", "")).strip():
            raise ProtocolError("invalid query field")
        result.append(QueryField(f"Q{i + 1}", str(item["description"]).strip()))
    modes = strings(data.get("observation_modes"), "observation_modes", limit=4)
    if not modes or set(modes) - MODES:
        raise ProtocolError("invalid observation_modes")
    coverage = data.get("coverage")
    if coverage not in {"point", "sequence", "full_span", "existence"}:
        raise ProtocolError("invalid coverage requirement")
    if "caption" in modes and coverage != "existence":
        coverage = "full_span"
    elif "ordered" in modes and coverage == "point":
        coverage = "sequence"
    requires_reference = data.get("requires_reference", False)
    if not isinstance(requires_reference, bool):
        raise ProtocolError("requires_reference must be boolean")
    reference = str(data.get("reference_description", "")).strip()
    relation = data.get("reference_relation", "any")
    if relation not in {"before", "after", "any"} or (requires_reference and not reference):
        raise ProtocolError("invalid reference requirement")
    modalities = strings(data.get("required_modalities", []), "required_modalities", limit=4)
    if set(modalities) - {"video", "screen_text", "subtitle", "asr"}:
        raise ProtocolError("unknown required modality")
    speaker = data.get("requires_speaker_binding", False)
    if not isinstance(speaker, bool):
        raise ProtocolError("requires_speaker_binding must be boolean")
    return QuerySpec(
        tuple(result),
        str(data.get("anchor_description", "")),
        modes,
        coverage,
        requires_reference,
        reference,
        relation,
        modalities,
        str(data.get("semantic_hint", "")),
        speaker,
    )


def fallback_query(question: str) -> QuerySpec:
    text = question.casefold()
    modes = ["static"]
    if re.search(r"text|written|sentence|phone|clock|year|文字|写|手机|年份|几点", text):
        modes.append("ocr")
    if re.search(r"doing|happen|action|before|after|动作|发生|之前|之后", text):
        modes.append("ordered")
    caption = bool(re.search(r"describe|summari[sz]e|caption|描述|概述", text))
    if caption:
        modes.append("caption")
    return QuerySpec(
        (QueryField("Q1", question),), question, tuple(modes), "full_span" if caption else "point"
    )


def parse_discriminants(data: dict[str, Any], *, ocr: bool) -> DiscriminantSpec:
    needs = strings(data.get("inspection_needs"), "inspection_needs", limit=16)
    targets = strings(data.get("target_union", []), "target_union", limit=32)
    modes = strings(data.get("observation_modes", []), "observation_modes", limit=4)
    if set(modes) - MODES:
        raise ProtocolError("unknown discriminant observation mode")
    # Inspection prompts must not contain answer labels, option combinations or OCR candidates.
    forbidden = re.compile(r"\b(?:option|choice|answer)\s+[A-Z0-9]\b|\b[A-Z][.):]\s", re.IGNORECASE)
    if any(forbidden.search(s) for s in (*needs, *targets)):
        raise ProtocolError("answer mapping leaked into inspection requirements")
    if ocr or "ocr" in modes:
        targets = tuple(s for s in targets if not re.search(r"\d", s))
        needs = tuple(s for s in needs if not re.search(r"\d", s))
        needs = (*needs, "Read the exact visible characters, including uncertain characters.")
    return DiscriminantSpec(
        tuple(sorted(set(needs))), tuple(sorted(set(targets))), tuple(sorted(set(modes)))
    )


def combined_query(query: QuerySpec, discriminants: DiscriminantSpec) -> QuerySpec:
    modes = tuple(sorted(set(query.observation_modes) | set(discriminants.observation_modes)))
    coverage = query.coverage
    if "caption" in modes and coverage != "existence":
        coverage = "full_span"
    elif "ordered" in modes and coverage == "point":
        coverage = "sequence"
    return replace(query, observation_modes=modes, coverage=coverage)


def covered(span: TimeSpan, intervals: Iterable[TimeSpan]) -> bool:
    cursor = span.start_seconds
    for item in sorted(intervals, key=lambda s: s.start_seconds):
        if item.end_seconds <= cursor:
            continue
        if item.start_seconds > cursor + 1e-6:
            return False
        cursor = item.end_seconds
        if cursor >= span.end_seconds - 1e-6:
            return True
    return False


def find_conflicts(facts: Iterable[EvidenceFact]) -> list[dict[str, Any]]:
    clear = [
        f
        for f in facts
        if f.observation_status == "clear"
        and f.structured_value
        and f.subject_or_local_entity
        and f.attribute
    ]
    conflicts = []
    for a, b in combinations(clear, 2):
        key_a = (a.source_id, a.subject_or_local_entity, a.attribute, a.source_kind)
        key_b = (b.source_id, b.subject_or_local_entity, b.attribute, b.source_kind)
        same_visual_source = bool(
            set(a.original_frame_ids or a.source_frame_ids)
            & set(b.original_frame_ids or b.source_frame_ids)
        )
        same_text_source = bool(set(a.source_segment_ids) & set(b.source_segment_ids))
        if (
            key_a == key_b
            and (same_visual_source or same_text_source)
            and max(a.start_sec, b.start_sec) <= min(a.end_sec, b.end_sec)
            and a.structured_value.casefold() != b.structured_value.casefold()
        ):
            conflicts.append(
                {
                    "fact_ids": [a.fact_id, b.fact_id],
                    "reason": "incompatible values for the same subject/time/attribute",
                }
            )
    return conflicts


def refuted_fact_ids(bundle: EvidenceBundle) -> set[str]:
    latest = {r["fact_id"]: r for p in bundle.packets for r in p.fact_reviews}
    return {key for key, review in latest.items() if review["judgment"] == "refuted"}


def audit_bundle(bundle: EvidenceBundle, query: QuerySpec) -> EvidenceAudit:
    answers = [p for p in bundle.packets if p.role != "reference"]
    refuted = refuted_fact_ids(bundle)
    facts = [
        f
        for p in answers
        for f in p.facts
        if f.observation_status == "clear"
        and not f.uncertain_characters
        and f.fact_id not in refuted
    ]
    filled = {key for fact in facts for key in fact.supports_query_fields}
    missing = [f.field_id for f in query.fields if f.field_id not in filled]
    reasons = []
    if not answers or any(p.anchor_match != "matched" for p in answers):
        reasons.append("anchor_not_verified")
    if any(p.target_binding != "confirmed" for p in answers):
        reasons.append("target_binding_unresolved")
    for packet in bundle.packets:
        shown = {f["id"] for view in packet.source_views for f in view["frames"]}
        if set(packet.locator_anchor_ids) - shown:
            reasons.append("original_locator_anchor_not_observed")
    bundle.conflicts = find_conflicts(f for f in bundle.facts if f.fact_id not in refuted)
    if bundle.conflicts:
        reasons.append("conflicting_facts")
    records = [r for p in answers for r in p.coverage]
    valid = [
        r.planned_span
        for r in records
        if r.observation_completed
        and r.coverage_kind == "base"
        and r.required_resolution_met
        and not r.truncated
        and not r.unresolved
    ]
    scope_complete = bool(valid) and all(covered(s, valid) for s in bundle.required_spans)
    if query.coverage in {"full_span", "sequence"} and not scope_complete:
        reasons.append("required_observation_incomplete")
    if query.requires_reference:
        refs = [p for p in bundle.packets if p.role == "reference"]
        latest = {(b.left_packet_id, b.right_packet_id): b for b in bundle.bindings}
        linked = {
            b.right_packet_id
            for b in latest.values()
            if b.relation == "same"
            and b.discriminating_features
            and b.feature_kinds
            and set(b.feature_kinds) <= {"marking", "distinctive_geometry", "unique_configuration"}
        }
        if not refs or any(
            p.anchor_match != "matched" or p.target_binding != "confirmed" for p in refs
        ):
            reasons.append("reference_anchor_unresolved")
        if any(p.packet_id not in linked for p in answers):
            reasons.append("cross_packet_binding_unresolved")
    if query.coverage == "existence":
        positive = any(p.existence == "present" and p.facts for p in answers)
        negative = (
            scope_complete
            and answers
            and all(p.existence == "absent" and p.absence_basis for p in answers)
        )
        if not positive and not negative:
            reasons.append("absence_not_established")
    for p in answers:
        if p.unresolved:
            reasons.extend(p.unresolved)
    return EvidenceAudit(
        not missing and not reasons and bool(facts),
        missing,
        list(dict.fromkeys(reasons)),
        scope_complete,
        [f.fact_id for f in facts],
    )


def parse_fact(
    data: dict[str, Any],
    *,
    fact_id: str,
    source_id: str,
    view_id: str,
    frames: dict[str, Any],
    segments: dict[str, Any],
    query: QuerySpec,
    quality_limited: bool = False,
) -> EvidenceFact:
    refs = strings(data.get("source_frame_ids", []), "source_frame_ids", limit=112)
    segs = strings(data.get("source_segment_ids", []), "source_segment_ids")
    if not refs and not segs:
        raise ProtocolError("a fact requires displayed source references")
    if any(ref not in frames for ref in refs) or any(ref not in segments for ref in segs):
        raise ProtocolError("fact references material not displayed in this call")
    statement = str(data.get("statement", "")).strip()
    if not statement:
        raise ProtocolError("fact statement is empty")
    status = data.get("observation_status")
    if status not in {"clear", "partial", "occluded", "unreadable"}:
        raise ProtocolError("unknown fact observation status")
    if quality_limited and status == "clear":
        status = "partial"
    fields = strings(data.get("supports_query_fields", []), "supports_query_fields", limit=12)
    if set(fields) - {f.field_id for f in query.fields}:
        raise ProtocolError("unknown query field reference")
    times = [frames[r].timestamp_seconds for r in refs]
    times += [t for s in segs for t in (segments[s].start_sec, segments[s].end_sec)]
    if any(segments[s].alignment_status != "aligned" for s in segs):
        status = "partial"
    kind = "reported_speech" if segs else str(data.get("source_kind", "visual"))
    if kind not in {"visual", "screen_text", "reported_speech"}:
        raise ProtocolError("invalid fact source kind")
    if kind == "reported_speech" and not segs:
        raise ProtocolError("reported speech requires a read source segment")
    return EvidenceFact(
        fact_id,
        statement,
        str(data.get("structured_value", "")),
        str(data.get("subject_or_local_entity", "")),
        str(data.get("attribute", "")),
        source_id,
        refs,
        segs,
        min(times),
        max(times),
        view_id,
        status,
        fields,
        kind,
        str(data.get("uncertain_characters", "")),
    )


def parse_span(data: Any, allowed: TimeSpan) -> TimeSpan:
    try:
        span = span_from(data)
    except (ValueError, TypeError, KeyError) as exc:
        raise ProtocolError("invalid span") from exc
    if span is None or not contains(allowed, span):
        raise ProtocolError("span exceeds allowed scope")
    return span
