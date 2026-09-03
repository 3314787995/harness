from __future__ import annotations

import math
import re
from collections.abc import Iterable, Sequence
from dataclasses import replace
from itertools import pairwise

from qwen3vl_agent.coarse_to_fine.types import FrameRef
from qwen3vl_agent.p01.config import P01Config
from qwen3vl_agent.p01.types import (
    CandidateObservation,
    CanonicalOption,
    ClaimVerdict,
    ContractViolation,
    CoverageManifest,
    DecisionSpec,
    EventFact,
    EvidencePacket,
    Fact,
    ObservationSpec,
    OptionVerdict,
    RefinementPlan,
    StaticFact,
    TextFact,
    TimeSpan,
)

_TIME_TOKEN = r"(?:\d{1,2}:)?\d{1,2}:\d{2}(?:\.\d+)?"
_RANGE_PATTERN = re.compile(
    rf"(?:from\s+)?(?P<start>{_TIME_TOKEN})\s*(?:-|–|—|to)\s*(?P<end>{_TIME_TOKEN})",
    flags=re.IGNORECASE,
)
_AROUND_PATTERN = re.compile(
    rf"\b(?:around|about|near|at approximately)\s+(?P<time>{_TIME_TOKEN})",
    flags=re.IGNORECASE,
)


def canonicalize_options(choices: Sequence[str]) -> tuple[CanonicalOption, ...]:
    result: list[CanonicalOption] = []
    for index, raw in enumerate(choices):
        label = chr(ord("A") + index)
        text = re.sub(r"^[A-Z][.):]\s*", "", str(raw).strip(), flags=re.IGNORECASE)
        result.append(CanonicalOption(f"O{index + 1}", label, text))
    return tuple(result)


def parse_timestamp(value: str) -> float:
    parts = value.strip().split(":")
    if len(parts) == 2:
        minutes, seconds = parts
        hours = "0"
    elif len(parts) == 3:
        hours, minutes, seconds = parts
    else:
        raise ValueError(f"unsupported timestamp: {value!r}")
    result = int(hours) * 3600 + int(minutes) * 60 + float(seconds)
    if result < 0:
        raise ValueError("timestamp must be non-negative")
    return result


def parse_question_interval(question: str, duration_seconds: float) -> TimeSpan | None:
    range_match = _RANGE_PATTERN.search(question)
    if range_match is not None:
        start = parse_timestamp(range_match.group("start"))
        end = parse_timestamp(range_match.group("end"))
        if end <= start:
            raise ContractViolation("question interval end must follow start")
        return _validated_span(
            start,
            end,
            duration_seconds,
            source="question_interval",
        )

    around_match = _AROUND_PATTERN.search(question)
    if around_match is not None:
        center = parse_timestamp(around_match.group("time"))
        return _validated_span(
            max(0.0, center - 5.0),
            min(duration_seconds, center + 5.0),
            duration_seconds,
            source="around_timestamp",
        )

    normalized = " ".join(question.casefold().split())
    if any(
        phrase in normalized
        for phrase in (
            "opening caption",
            "at the beginning",
            "in the beginning",
            "intro",
            "title card",
        )
    ):
        return _validated_span(
            0.0,
            min(15.0, duration_seconds),
            duration_seconds,
            source="opening_hint",
        )
    if "credits" in normalized:
        return _validated_span(
            max(0.0, duration_seconds - 20.0),
            duration_seconds,
            duration_seconds,
            source="credits_hint",
        )
    if any(
        phrase in normalized
        for phrase in ("closing caption", "at the end", "near the end", "in the end")
    ):
        return _validated_span(
            max(0.0, duration_seconds - 15.0),
            duration_seconds,
            duration_seconds,
            source="ending_hint",
        )
    return None


def reconcile_interval(
    given: TimeSpan | None,
    parsed: TimeSpan | None,
    duration_seconds: float,
    *,
    tolerance_seconds: float = 0.5,
) -> TimeSpan | None:
    normalized_given = None
    if given is not None:
        normalized_given = _validated_span(
            given.start_seconds,
            given.end_seconds,
            duration_seconds,
            source="given_interval",
        )
    if (
        normalized_given is not None
        and parsed is not None
        and (
            abs(normalized_given.start_seconds - parsed.start_seconds) > tolerance_seconds
            or abs(normalized_given.end_seconds - parsed.end_seconds) > tolerance_seconds
        )
    ):
        raise ContractViolation(
            "structured interval conflicts with the interval stated in the question"
        )
    return normalized_given or parsed


def _validated_span(
    start: float,
    end: float,
    duration_seconds: float,
    *,
    source: str,
    context_seconds: float = 0.0,
) -> TimeSpan:
    if duration_seconds <= 0:
        raise ContractViolation("video duration must be positive")
    if start < -0.5 or end > duration_seconds + 0.5 or end <= start:
        raise ContractViolation(
            f"interval {start:.3f}-{end:.3f}s falls outside video duration {duration_seconds:.3f}s"
        )
    start = max(0.0, start)
    end = min(duration_seconds, end)
    return TimeSpan(
        start,
        end,
        source=source,
        context_start_seconds=max(0.0, start - context_seconds),
        context_end_seconds=min(duration_seconds, end + context_seconds),
    )


def bounded_automatic_span(
    supporting: TimeSpan,
    *,
    mode: str,
    duration_seconds: float,
    config: P01Config,
) -> TimeSpan:
    padding = config.padding(mode)
    limit = config.span_limit(mode)
    start = max(0.0, supporting.start_seconds - padding)
    end = min(duration_seconds, supporting.end_seconds + padding)
    if end - start > limit:
        center = supporting.midpoint_seconds
        start = max(0.0, center - limit / 2)
        end = min(duration_seconds, start + limit)
        start = max(0.0, end - limit)
    return TimeSpan(start, end, source="automatic_local_span")


def interval_chunks(span: TimeSpan, config: P01Config) -> tuple[TimeSpan, ...]:
    if span.duration_seconds <= 60:
        return (span,)
    step = config.interval_chunk_sec - config.interval_overlap_sec
    count = math.ceil((span.duration_seconds - config.interval_overlap_sec) / step)
    adaptive = count > config.max_interval_chunks
    if adaptive:
        count = config.max_interval_chunks
        chunk_size = (span.duration_seconds + config.interval_overlap_sec * (count - 1)) / count
        step = chunk_size - config.interval_overlap_sec
    else:
        chunk_size = config.interval_chunk_sec
    result: list[TimeSpan] = []
    start = span.start_seconds
    for _index in range(count):
        end = min(span.end_seconds, start + chunk_size)
        result.append(
            TimeSpan(
                start,
                end,
                source=("interval_chunk_adaptive" if adaptive else "interval_chunk"),
            )
        )
        if end >= span.end_seconds:
            break
        start = min(span.end_seconds, start + step)
    if result and result[-1].end_seconds < span.end_seconds - 1e-6:
        final_start = max(span.start_seconds, span.end_seconds - chunk_size)
        result[-1] = TimeSpan(
            final_start,
            span.end_seconds,
            source="interval_chunk_adaptive",
        )
    return tuple(result)


def uniform_timestamps(
    span: TimeSpan,
    *,
    fps: float,
    max_frames: int,
    offset_fraction: float = 0.0,
) -> tuple[float, ...]:
    if fps <= 0 or max_frames < 1:
        return ()
    step = 1.0 / fps
    start = span.decode_start_seconds + offset_fraction * step
    end = span.decode_end_seconds
    if start > end:
        start = span.decode_start_seconds
    desired_count = math.floor(max(0.0, end - start) / step) + 1
    if desired_count > max_frames:
        if max_frames == 1:
            return (span.midpoint_seconds,)
        return tuple(
            start + index * (end - start) / (max_frames - 1) for index in range(max_frames)
        )
    values: list[float] = []
    cursor = start
    while cursor <= end + 1e-6 and len(values) < max_frames:
        values.append(min(cursor, end))
        cursor += step
    if not values:
        values.append(span.midpoint_seconds)
    if values[-1] < end - step * 0.5 and len(values) < max_frames:
        values.append(end)
    return tuple(_deduplicate_numbers(values))


def evenly_spaced_timestamps(span: TimeSpan, count: int) -> tuple[float, ...]:
    if count < 1:
        return ()
    if count == 1:
        return (span.midpoint_seconds,)
    return tuple(
        span.decode_start_seconds
        + index * (span.decode_end_seconds - span.decode_start_seconds) / (count - 1)
        for index in range(count)
    )


def build_coverage_manifest(
    mode: str,
    span: TimeSpan,
    frames: Sequence[FrameRef],
    *,
    sample_fps: float | None,
    shot_ids: Sequence[str] = (),
    roi_ids: Sequence[str] = (),
) -> CoverageManifest:
    ordered = sorted(frames, key=lambda item: item.timestamp_seconds)
    evidence_frames = [
        frame for frame in ordered if span.contains_evidence(frame.timestamp_seconds)
    ]
    context_only = [
        frame.id for frame in ordered if not span.contains_evidence(frame.timestamp_seconds)
    ]
    if evidence_frames:
        points = [span.start_seconds]
        points.extend(frame.timestamp_seconds for frame in evidence_frames)
        points.append(span.end_seconds)
        gaps = [right - left for left, right in pairwise(points)]
        max_gap = max(gaps, default=0.0)
        observed_start = evidence_frames[0].timestamp_seconds
        observed_end = evidence_frames[-1].timestamp_seconds
    else:
        max_gap = None
        observed_start = span.start_seconds
        observed_end = span.start_seconds

    full_span = False
    if evidence_frames and sample_fps is not None and sample_fps > 0 and max_gap is not None:
        allowed_gap = max(0.5, 1.6 / sample_fps)
        full_span = max_gap <= allowed_gap + 1e-6

    return CoverageManifest(
        mode=mode,
        observed_start_seconds=observed_start,
        observed_end_seconds=observed_end,
        sample_fps=sample_fps,
        max_temporal_gap_seconds=max_gap,
        frame_ids=tuple(frame.id for frame in ordered),
        shot_ids=tuple(dict.fromkeys(shot_ids)),
        roi_ids=tuple(dict.fromkeys(roi_ids)),
        context_only_frame_ids=tuple(context_only),
        full_span_coverage=full_span,
    )


def covered_slot_ids(facts: Iterable[Fact], *, clear_only: bool = True) -> set[str]:
    allowed = {"clear"} if clear_only else {"clear", "partial"}
    return {slot_id for fact in facts if fact.visibility in allowed for slot_id in fact.slot_ids}


def missing_required_slots(spec: ObservationSpec, facts: Iterable[Fact]) -> tuple[str, ...]:
    return tuple(sorted(spec.required_slot_ids - covered_slot_ids(facts)))


def choose_candidate(
    observations: Sequence[CandidateObservation],
) -> CandidateObservation | None:
    visible = [item for item in observations if item.target_visible]
    if not visible:
        return None

    def score(item: CandidateObservation) -> tuple[int, int, int, int, int, float]:
        facts = item.evidence.facts
        clear = len(covered_slot_ids(facts, clear_only=True))
        partial = len(covered_slot_ids(facts, clear_only=False)) - clear
        conflicts = len(item.evidence.conflicts)
        missing = len(item.evidence.missing_slot_ids)
        return (
            -clear,
            -partial,
            conflicts,
            missing,
            item.locator_rank,
            item.supporting_span.duration_seconds,
        )

    return min(visible, key=score)


def build_refinement_plan(
    spec: ObservationSpec,
    packet: EvidencePacket,
    decision_spec: DecisionSpec | None,
    config: P01Config,
) -> RefinementPlan:
    missing = missing_required_slots(spec, packet.facts)
    partial_slots = sorted(
        {
            slot_id
            for fact in packet.facts
            if fact.visibility in {"partial", "occluded", "conflicting"}
            for slot_id in fact.slot_ids
        }
    )
    crop_boxes = tuple(
        fact.bbox for fact in packet.facts if isinstance(fact, TextFact) and fact.bbox is not None
    )[: config.max_detail_images]

    densify = None
    if spec.primary_mode == "dynamic_action" and (missing or partial_slots):
        densify = config.dynamic_refine_fps
    elif spec.primary_mode == "ocr" and (missing or partial_slots or crop_boxes):
        densify = config.ocr_search_fps
    elif spec.primary_mode == "subscene_caption" and (missing or partial_slots):
        densify = config.caption_refine_fps

    facts_near_start = any(
        fact.start_seconds - packet.canonical_span.start_seconds <= 0.75 for fact in packet.facts
    )
    facts_near_end = any(
        packet.canonical_span.end_seconds - fact.end_seconds <= 0.75 for fact in packet.facts
    )
    extend_before = 3.0 if facts_near_start and (missing or partial_slots) else 0.0
    extend_after = 3.0 if facts_near_end and (missing or partial_slots) else 0.0

    target_claim_ids: tuple[str, ...] = ()
    if decision_spec is not None:
        target_claim_ids = tuple(claim.claim_id for claim in decision_spec.claim_tests)
    reasons: list[str] = []
    if missing:
        reasons.append("missing required slots")
    if partial_slots:
        reasons.append("partial or conflicting visibility")
    if crop_boxes:
        reasons.append("available detail regions")
    return RefinementPlan(
        densify_fps=densify,
        extend_before_seconds=extend_before,
        extend_after_seconds=extend_after,
        crop_boxes=tuple(box for box in crop_boxes if box is not None),
        target_slot_ids=tuple(sorted(set(missing) | set(partial_slots))),
        target_claim_ids=target_claim_ids,
        reason="; ".join(reasons),
    )


def merge_evidence(
    first: EvidencePacket,
    second: EvidencePacket,
    *,
    canonical_span: TimeSpan,
    spec: ObservationSpec,
) -> EvidencePacket:
    facts: list[Fact] = []
    seen: set[tuple[object, ...]] = set()
    for fact in (*first.facts, *second.facts):
        key = (
            fact.kind,
            tuple(sorted(fact.slot_ids)),
            " ".join(fact.statement.casefold().split()),
            round(fact.start_seconds, 2),
            round(fact.end_seconds, 2),
            fact.visibility,
            fact.source_frame_ids,
        )
        if key in seen:
            continue
        seen.add(key)
        facts.append(fact)

    conflicts = list(dict.fromkeys((*first.conflicts, *second.conflicts)))
    by_slot: dict[str, set[str]] = {}
    for fact in facts:
        if fact.visibility != "clear":
            continue
        value = _fact_value(fact)
        if not value:
            continue
        for slot_id in fact.slot_ids:
            by_slot.setdefault(slot_id, set()).add(value.casefold().strip())
    for slot_id, values in by_slot.items():
        if len(values) > 1:
            conflicts.append(f"conflicting clear values for {slot_id}")

    combined_frames = tuple(
        dict.fromkeys((*first.coverage_manifest.frame_ids, *second.coverage_manifest.frame_ids))
    )
    coverage = replace(
        second.coverage_manifest,
        frame_ids=combined_frames,
        shot_ids=tuple(
            dict.fromkeys((*first.coverage_manifest.shot_ids, *second.coverage_manifest.shot_ids))
        ),
        roi_ids=tuple(
            dict.fromkeys((*first.coverage_manifest.roi_ids, *second.coverage_manifest.roi_ids))
        ),
        context_only_frame_ids=tuple(
            dict.fromkeys(
                (
                    *first.coverage_manifest.context_only_frame_ids,
                    *second.coverage_manifest.context_only_frame_ids,
                )
            )
        ),
        full_span_coverage=(
            first.coverage_manifest.full_span_coverage
            or second.coverage_manifest.full_span_coverage
        ),
    )
    return EvidencePacket(
        canonical_span=canonical_span,
        facts=tuple(facts),
        coverage_manifest=coverage,
        missing_slot_ids=missing_required_slots(spec, facts),
        conflicts=tuple(dict.fromkeys(conflicts)),
        source_views=(*first.source_views, *second.source_views),
    )


def resolve_options(
    decision_spec: DecisionSpec,
    verdicts: Sequence[ClaimVerdict],
) -> tuple[OptionVerdict, ...]:
    by_claim = {item.claim_id: item.verdict for item in verdicts}
    results: list[OptionVerdict] = []
    for rule in decision_spec.option_rules:
        support = [by_claim.get(claim_id, "not_established") for claim_id in rule.all_of]
        exclusion = [by_claim.get(claim_id, "not_established") for claim_id in rule.none_of]
        entailed_support = support.count("entailed")
        contradicted_support = support.count("contradicted")
        contradicted_exclusion = exclusion.count("contradicted")
        entailed_exclusion = exclusion.count("entailed")
        if support and all(item == "entailed" for item in support) and not entailed_exclusion:
            verdict = "entailed"
        elif contradicted_support or entailed_exclusion:
            verdict = "contradicted"
        else:
            verdict = "not_established"
        results.append(
            OptionVerdict(
                option_id=rule.option_id,
                verdict=verdict,
                entailed_support_count=entailed_support,
                contradicted_support_count=contradicted_support,
                contradicted_exclusion_count=contradicted_exclusion,
                entailed_exclusion_count=entailed_exclusion,
            )
        )
    return tuple(results)


def unique_entailed_option(verdicts: Sequence[OptionVerdict]) -> str | None:
    entailed = [item.option_id for item in verdicts if item.verdict == "entailed"]
    return entailed[0] if len(entailed) == 1 else None


def forced_option(verdicts: Sequence[OptionVerdict]) -> str | None:
    if not verdicts:
        return None
    state_rank = {"entailed": 2, "not_established": 1, "contradicted": 0}

    def score(item: OptionVerdict) -> tuple[int, int, int, int, int]:
        option_index = int(item.option_id.removeprefix("O"))
        return (
            state_rank[item.verdict],
            item.entailed_support_count,
            -item.contradicted_support_count,
            item.contradicted_exclusion_count,
            -option_index,
        )

    return max(verdicts, key=score).option_id


def option_label(options: Sequence[CanonicalOption], option_id: str | None) -> str | None:
    if option_id is None:
        return None
    for option in options:
        if option.option_id == option_id:
            return option.benchmark_label
    return None


def unresolved_status(packet: EvidencePacket, spec: ObservationSpec) -> str:
    represented_slots = {slot_id for fact in packet.facts for slot_id in fact.slot_ids}
    if (
        spec.required_slot_ids.issubset(represented_slots)
        and packet.coverage_manifest.full_span_coverage
        and (
            packet.conflicts
            or any(
                fact.visibility in {"occluded", "not_visible", "conflicting"}
                for fact in packet.facts
            )
        )
    ):
        return "video_underdetermined"
    return "pipeline_insufficient"


def validate_fact_provenance(packet: EvidencePacket) -> tuple[str, ...]:
    context_only = set(packet.coverage_manifest.context_only_frame_ids)
    errors: list[str] = []
    for fact in packet.facts:
        if fact.visibility == "not_visible":
            continue
        if not fact.source_frame_ids:
            errors.append(f"{fact.fact_id} has no source frames")
            continue
        decisive_context = context_only.intersection(fact.source_frame_ids)
        if decisive_context:
            errors.append(
                f"{fact.fact_id} cites context-only frames: {','.join(sorted(decisive_context))}"
            )
        if not packet.canonical_span.contains_evidence(fact.start_seconds):
            errors.append(f"{fact.fact_id} starts outside canonical span")
        if not packet.canonical_span.contains_evidence(fact.end_seconds):
            errors.append(f"{fact.fact_id} ends outside canonical span")
    return tuple(errors)


def _fact_value(fact: Fact) -> str:
    if isinstance(fact, StaticFact):
        return fact.value or fact.statement
    if isinstance(fact, TextFact):
        return fact.exact_text or fact.statement
    if isinstance(fact, EventFact):
        return "|".join(
            item
            for item in (fact.subject, fact.action, fact.object, fact.target, fact.result)
            if item
        )
    return fact.statement


def _deduplicate_numbers(values: Iterable[float]) -> list[float]:
    result: list[float] = []
    seen: set[int] = set()
    for value in values:
        key = round(value * 1_000_000)
        if key in seen:
            continue
        seen.add(key)
        result.append(float(value))
    return result
