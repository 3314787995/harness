"""Validate independent units and project immutable observations into the public R1 packet."""

import copy
from dataclasses import asdict, replace

from qwen3vl_agent.r1.control import ProtocolError, contains, parse_fact, strings
from qwen3vl_agent.r1_v3.types import FactEligibility


def originals(batch, refs):
    return {batch.crops.get(r, {}).get("source_frame_id", r) for r in refs}


def parse_observation(data, limited, *, packet, batch, query, source_id, record_id, review_ids=()):
    frames = {f.id: f for f in batch.frames}
    errors, facts, gaps, reviews = [], [], [], []

    def error(unit, exc):
        errors.append({"unit": unit, "reason": str(exc)})

    def refs(value):
        result = strings(value, "source_frame_ids", limit=112)
        if set(result) - set(frames):
            raise ProtocolError("references_unshown_source")
        return list(result)

    target = {
        "status": "unresolved",
        "description": "",
        "source_frame_ids": [],
        "unresolved_conditions": [],
    }
    try:
        raw = data["target"]
        if not isinstance(raw, dict):
            raise ProtocolError("target_must_be_object")
        status = raw.get("status")
        if status not in {"matched", "mismatched", "unresolved"}:
            raise ProtocolError("invalid_target_status")
        cited = refs(raw.get("source_frame_ids", []))
        description = raw.get("description", "")
        if not isinstance(description, str) or not description.strip():
            raise ProtocolError("missing_target_description")
        conditions = list(strings(raw.get("unresolved_conditions", []), "conditions"))
        target = {
            "status": status,
            "description": description,
            "source_frame_ids": cited,
            "unresolved_conditions": conditions,
        }
        if status != "unresolved" and not cited:
            raise ProtocolError("missing_target_references")
        if status == "matched" and conditions:
            raise ProtocolError("matched_target_has_unresolved_conditions")
        if limited:
            target["status"] = "unresolved"
    except (ProtocolError, KeyError, TypeError, ValueError) as exc:
        target["status"] = "unresolved"
        error("target", exc)

    raw_facts = data.get("facts")
    if not isinstance(raw_facts, list) or len(raw_facts) > 48:
        error("facts", "facts_must_be_array_of_at_most_48")
        raw_facts = []
    truncated = data.get("truncated", False) is True
    for index, raw in enumerate(raw_facts):
        try:
            if not isinstance(raw, dict) or raw.get("source_segment_ids"):
                raise ProtocolError("visual_fact_must_have_only_frame_sources")
            fact = parse_fact(
                raw,
                fact_id=f"{record_id}.f{index + 1}",
                source_id=source_id,
                view_id=record_id,
                frames=frames,
                segments={},
                query=query,
                quality_limited=limited or truncated,
            )
            if fact.source_kind not in {"visual", "screen_text"}:
                raise ProtocolError("nonvisual_fact")
            facts.append(
                replace(
                    fact,
                    original_frame_ids=tuple(sorted(originals(batch, fact.source_frame_ids))),
                    crop_transforms={
                        r: batch.crops[r] for r in fact.source_frame_ids if r in batch.crops
                    },
                )
            )
        except (ProtocolError, KeyError, TypeError, ValueError) as exc:
            error(f"facts[{index}]", exc)

    raw_gaps = data.get("gaps")
    if not isinstance(raw_gaps, list) or len(raw_gaps) > 32:
        error("gaps", "gaps_must_be_array_of_at_most_32")
        raw_gaps = []
    for index, raw in enumerate(raw_gaps):
        try:
            if not isinstance(raw, dict) or raw.get("kind") not in {
                "target_identity",
                "detail",
                "temporal_context",
                "temporal_selection",
                "conflict",
            }:
                raise ProtocolError("invalid_gap_kind")
            if not isinstance(raw.get("reason"), str) or not raw["reason"].strip():
                raise ProtocolError("gap_requires_reason")
            gap = copy.deepcopy(raw)
            gap["source_frame_ids"] = refs(raw.get("source_frame_ids", []))
            gap["field_ids"] = list(strings(raw.get("field_ids", []), "field_ids", limit=12))
            if set(gap["field_ids"]) - {f.field_id for f in query.fields}:
                raise ProtocolError("gap_references_unknown_query_field")
            if gap["kind"] == "temporal_context" and gap.get("direction") not in {
                "before",
                "after",
            }:
                raise ProtocolError("context_requires_direction")
            if "crop" in gap:
                crop = gap["crop"]
                if (
                    gap["kind"] != "detail"
                    or not isinstance(crop, dict)
                    or crop.get("frame_id") not in frames
                    or crop["frame_id"] in batch.crops
                ):
                    raise ProtocolError("invalid_crop_source")
                bbox = crop.get("bbox_xyxy_1000")
                if (
                    not isinstance(bbox, list)
                    or len(bbox) != 4
                    or any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in bbox)
                    or not (0 <= bbox[0] < bbox[2] <= 1000 and 0 <= bbox[1] < bbox[3] <= 1000)
                ):
                    raise ProtocolError("invalid_crop_coordinates")
            gaps.append(gap)
        except (ProtocolError, KeyError, TypeError, ValueError) as exc:
            error(f"gaps[{index}]", exc)

    if "review_request" in data or "prediction" in data or "option_id" in data:
        error("protocol", "observer_must_not_output_actions_or_answers")
    records = {r.record_id: r for r in packet.observations if r.record_id in review_ids}
    raw_reviews = data.get("reviews", [])
    if not isinstance(raw_reviews, list) or len(raw_reviews) > 48:
        error("reviews", "invalid_reviews_array")
        raw_reviews = []
    for index, raw in enumerate(raw_reviews):
        try:
            if not isinstance(raw, dict) or raw.get("record_id") not in records:
                raise ProtocolError("review_requires_supplied_record")
            prior = records[raw["record_id"]]
            cited = refs(raw.get("source_frame_ids", []))
            judgment = raw.get("judgment")
            if judgment not in {"verified", "refuted", "unresolved"}:
                raise ProtocolError("invalid_review_judgment")
            fact_id = raw.get("fact_id", "")
            if fact_id:
                fact = next((f for f in prior.facts if f.fact_id == fact_id), None)
                if fact is None:
                    raise ProtocolError("review_unknown_fact")
                required = set(fact.original_frame_ids)
            else:
                required = originals(prior.batch, prior.target["source_frame_ids"])
                required = required or originals(prior.batch, [f.id for f in prior.batch.frames])
            if (
                limited
                or truncated
                or not required
                or not raw.get("basis")
                or not required <= originals(batch, cited)
            ):
                judgment = "unresolved"
            reviews.append({**raw, "source_frame_ids": cited, "judgment": judgment})
        except (ProtocolError, KeyError, TypeError, ValueError) as exc:
            error(f"reviews[{index}]", exc)

    try:
        coverage_gaps = list(strings(data.get("coverage_gaps", []), "coverage_gaps"))
    except ProtocolError as exc:
        coverage_gaps = ["invalid_coverage_output"]
        error("coverage", exc)
    if query.coverage != "point" and "coverage_gaps" not in data:
        coverage_gaps.append("missing_coverage_output")
        error("coverage", "missing_coverage_output")
    if truncated:
        error("protocol", "observation_truncated")
    existence = data.get("existence", "unknown")
    if existence not in {"present", "absent", "unknown"}:
        error("existence", "invalid_existence")
        existence = "unknown"
    return {
        "target": target,
        "facts": tuple(facts),
        "gaps": tuple(gaps),
        "errors": tuple(errors),
        "reviews": tuple(reviews),
        "truncated": truncated,
        "coverage_gaps": coverage_gaps,
        "existence": existence,
        "absence_basis": str(data.get("absence_basis", "")),
    }


def rebuild(packet):
    """Derive effective state. No observation or original fact is mutated."""
    target_reviews, fact_reviews = {}, {}
    for record in packet.observations:
        for review in record.reviews:
            # An incomplete recheck cannot undo a previously source-verified ruling.
            # Its uncertainty stays in the append-only observation, not in effective rulings.
            if review["judgment"] == "unresolved":
                continue
            if review.get("fact_id"):
                fact_reviews[review["fact_id"]] = {**review, "view_id": record.call_id}
            else:
                target_reviews[review["record_id"]] = review["judgment"]
    packet.fact_reviews = list(fact_reviews.values())
    positives = [
        r
        for r in packet.observations
        if r.target["status"] == "matched" and target_reviews.get(r.record_id) != "refuted"
    ]
    negatives = [
        r
        for r in packet.observations
        if r.target["status"] == "mismatched" and target_reviews.get(r.record_id) != "refuted"
    ]
    packet.target_record_ids = [r.record_id for r in positives]
    packet.anchor_source_ids = list(
        dict.fromkeys(x for r in positives for x in r.target["source_frame_ids"])
    )
    packet.target_source_ids = list(packet.anchor_source_ids)
    packet.anchor_match = "matched" if positives else "mismatched" if negatives else "unresolved"
    packet.target_binding = "confirmed" if positives else "unresolved"
    packet.facts, packet.bound_fact_ids, packet.fact_eligibility = [], [], []
    for record in packet.observations:
        bound = record in positives
        for fact in record.facts:
            supported = bound and bool(fact.supports_query_fields)
            clear = fact.observation_status == "clear" and not fact.uncertain_characters
            refuted = fact_reviews.get(fact.fact_id, {}).get("judgment") == "refuted"
            reasons = tuple(
                reason
                for condition, reason in (
                    (not clear, "observation_quality_insufficient"),
                    (not bound, "target_relation_unconfirmed"),
                    (not fact.supports_query_fields, "query_fields_unassociated"),
                    (refuted, "fact_refuted"),
                    (bool(positives and negatives), "target_observation_conflict"),
                )
                if condition
            )
            packet.fact_eligibility.append(
                FactEligibility(
                    fact.fact_id,
                    record.record_id,
                    True,
                    fact.observation_status,
                    clear,
                    bound,
                    fact.supports_query_fields,
                    refuted,
                    not reasons,
                    reasons,
                )
            )
            if supported:
                packet.bound_fact_ids.append(fact.fact_id)
            packet.facts.append(fact if supported else replace(fact, observation_status="partial"))
    packet.coverage = [
        copy.deepcopy(r.coverage) for r in packet.observations if r.coverage is not None
    ]
    packet.source_views = [
        {
            "view_id": r.call_id,
            "span": asdict(r.batch.span),
            "frames": [f.to_dict() for f in r.batch.frames],
            "crop_transforms": r.batch.crops,
            "media_kind": "observation",
            "quality_limited": bool(r.coverage and not r.coverage.required_resolution_met),
        }
        for r in packet.observations
        if r.call_id
    ]
    packet.active_errors, packet.active_gaps, packet.resolutions = [], [], []
    for index, record in enumerate(packet.observations):
        later = packet.observations[index + 1 :]
        needed = originals(record.batch, [f.id for f in record.batch.frames])
        repair = next(
            (
                r
                for r in later
                if not r.errors
                and r.target["status"] == "matched"
                and needed <= originals(r.batch, [f.id for f in r.batch.frames])
            ),
            None,
        )
        for unit in record.errors:
            item = {**unit, "record_id": record.record_id}
            if repair:
                packet.resolutions.append({**item, "resolved_by": repair.record_id})
            else:
                packet.active_errors.append(item)
        for gap in record.gaps:
            resolved = None
            for current in later:
                fields = {
                    q
                    for f in current.facts
                    if current in positives and f.observation_status == "clear"
                    for q in f.supports_query_fields
                }
                required_fields = set(gap.get("field_ids", []))
                explicit = any(
                    v["record_id"] == record.record_id and v["judgment"] in {"verified", "refuted"}
                    for v in current.reviews
                )
                if gap["kind"] == "conflict":
                    ok = explicit
                elif gap["kind"] == "detail":
                    ok = bool(fields) and (not required_fields or required_fields <= fields)
                elif gap["kind"] in {"temporal_context", "temporal_selection"}:
                    ok = (
                        explicit
                        or current.target["status"] == "matched"
                        and current.coverage is not None
                        and current.coverage.coverage_kind == "base"
                        and current.coverage.required_resolution_met
                        and not current.coverage.unresolved
                        and contains(current.batch.span, record.batch.span)
                        and not any(g["kind"] == gap["kind"] for g in current.gaps)
                    )
                else:
                    ok = explicit or current is repair
                if ok:
                    resolved = current.record_id
                    break
            item = {**gap, "record_id": record.record_id}
            if resolved:
                packet.resolutions.append({**item, "resolved_by": resolved})
            else:
                packet.active_gaps.append(item)
    packet.unresolved = [
        f"output:{e['record_id']}:{e['unit']}:{e['reason']}" for e in packet.active_errors
    ]
    packet.unresolved.extend(f"gap:{g['record_id']}:{g['kind']}" for g in packet.active_gaps)
    if positives and negatives:
        packet.unresolved.append("target_observation_conflict")
    # A target refutation is explicit; an empty later view cannot change it.
    packet.existence = "unknown"
    packet.absence_basis = ""
    for record in packet.observations:
        if record.existence != "unknown" and record.facts:
            packet.existence, packet.absence_basis = record.existence, record.absence_basis
    packet.review_request = {"kind": "none", "reason": "V3 controller schedules from evidence gaps"}
    packet.crop_requests = []
