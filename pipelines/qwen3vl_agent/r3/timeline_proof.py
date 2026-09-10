"""Operation-specific proof obligations. Scheduling consumes data, never issue strings."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from copy import deepcopy
from itertools import combinations

from .planning import overlaps
from .types import Bracket, Operation


@dataclass
class ReviewTask:
    kind: str
    operation_id: str
    target_ids: list[str]
    event_ids: list[str]
    intervals: list[list[float]]
    reason: str
    missing_facets: list[str] = field(default_factory=list)
    tile_id: str | None = None


def temporal_proof(ledger, scope, *, scope_start=None, scope_end=None, additional_issues=None):
    from .reduce import _one_operation, before, boundary, inclusion
    start, end = scope_start or Bracket(scope[0], scope[0]), scope_end or Bracket(scope[1], scope[1])
    events = ledger.canonical_events()
    specs = {t.target_id: t for t in ledger.query.targets}
    query_issues = list(ledger.query.unresolved) + list(additional_issues or [])
    results, all_gaps = [], []
    for op in ledger.query.operations:
        operation_events = deepcopy(events)
        if op.op in {"next_after_anchor", "previous_before_anchor"}:
            for e in operation_events:
                # Identifying the activity that starts next does not require observing
                # that activity's eventual end. The anchor still needs its own boundary.
                if e.target_id in op.target_ids and not e.unresolved_reasons and all(
                    e.verification.get(k) for k in ("target", "unit", "identity")
                ) and (specs[e.target_id].repeat_policy != "world" or e.replay_status == "original"):
                    e.status = "accepted"
        result = _one_operation(op, operation_events, ledger, scope, start, end)
        intervals = [list(s) for s in result["required_intervals"]]
        target_ids = list(dict.fromkeys([*op.target_ids, *([op.anchor_target_id] if op.anchor_target_id else [])]))
        relevant = [e for e in events if e.target_id in target_ids
                    and inclusion(e, specs[e.target_id].inclusion_rule, start, end) != "outside"
                    and any(e.visible_span[0] <= hi and e.visible_span[1] >= lo for lo, hi in intervals)]
        selected_ids = set(result["event_ids"])
        gaps = []

        def gap(kind, rows, reason, *, facets=(), spans=None, targets=None, tile=None):
            gaps.append(asdict(ReviewTask(kind, op.operation_id, targets or sorted({e.target_id for e in rows}) or target_ids,
                [e.event_id for e in rows], spans or intervals, reason, list(facets), tile)))

        for tile in ledger.tiles:
            if not any(overlaps(tile.core, tuple(s)) for s in intervals):
                continue
            missing = [tid for tid in target_ids if tile.target_coverage.get(tid) not in {"observed", "absent_scanned"}]
            text_targets = [tid for tid in target_ids if specs[tid].fact_kind in {"utterance", "reported_event"}
                            or set(specs[tid].required_modalities) & {"asr", "subtitle"}]
            if not tile.observed or missing or (text_targets and tile.external_coverage == "unknown"):
                gap("coverage", [], "Required sampled interval or target assessment is incomplete", targets=missing or target_ids,
                    spans=[list(tile.core)], tile=tile.tile_id)
        for e in relevant:
            needed = {"target", "unit", "identity"}
            needs_completion = not (op.op in {"next_after_anchor", "previous_before_anchor"} and e.target_id in op.target_ids)
            if needs_completion and e.unit_kind in {"action_cycle", "state_transition"}:
                needed.add("completion")
            if specs[e.target_id].repeat_policy == "world":
                needed.add("replay")
            if op.group_by == "category" or op.selection.endswith("per_category"):
                needed.add("attribute:category")
            membership = inclusion(e, specs[e.target_id].inclusion_rule, start, end)
            if membership == "unknown":
                needed.add("onset" if specs[e.target_id].inclusion_rule == "starts_inside" else "offset")
            if e.event_id in selected_ids:
                if op.op in {"first_occurrence", "last_occurrence", "first_k", "last_k", "nth_occurrence", "order_events"}:
                    needed.add(op.basis)
                if op.op in {"localize_event", "event_duration"}:
                    needed.update(["onset", "offset"])
                if op.op in {"next_after_anchor", "previous_before_anchor"}:
                    anchor = e.target_id == op.anchor_target_id
                    needed.add("offset" if (op.op == "next_after_anchor") == anchor else "onset")
                projected = op.op not in {"count_occurrences", "localize_event", "event_duration", "cooccurrence_frequency"}
                if op.op == "event_duration" and op.duration_aggregation == "compare":
                    projected = True
                if op.op == "nth_occurrence":
                    projected = e.event_id == (result["event_ids"][-1] if result["event_ids"] else None)
                if projected and e.target_id in op.target_ids:
                    needed.add("attribute:" + op.project)
                if op.op == "cooccurrence_frequency":
                    for name in (op.cooccurrence_targets or tuple(e.cooccurrence)):
                        status = e.cooccurrence.get(name, "unknown")
                        if status != "unknown":
                            needed.add("cooccurrence:" + name)
                        if status == "absent":
                            needed.update(["onset", "offset"])
                            if name in specs and not ledger.scan_closed([name], [e.visible_span]):
                                gap("cooccurrence", [e], "Episode-wide absence needs scan coverage of the cooccurring target", targets=[e.target_id, name])
            missing_facets = sorted(k for k in needed if not e.verification.get(k))
            if missing_facets or e.unresolved_reasons:
                gap("event", [e], "Verify the required event facets; revise, split or retract unsupported claims", facets=missing_facets)
        for relation in ledger.pending_relations(relevant):
            rows = [e for e in relevant if e.event_id in {relation.left_id, relation.right_id}]
            gap("identity", rows, "Resolve this actual continuation/duplicate/replay ambiguity", facets=["identity"])
        ordered_ops = {"first_occurrence", "last_occurrence", "first_k", "last_k", "nth_occurrence", "order_events"}
        if op.op in ordered_ops:
            for a, b in combinations(relevant, 2):
                if not before(boundary(a, op.basis), boundary(b, op.basis)) and not before(boundary(b, op.basis), boundary(a, op.basis)):
                    gap("order", [a, b], "The ordering brackets overlap or remain unknown", facets=[op.basis])
        if op.op in {"next_after_anchor", "previous_before_anchor"}:
            anchors = [e for e in relevant if e.target_id == op.anchor_target_id]
            for anchor in anchors:
                for event in relevant:
                    if event.target_id not in op.target_ids:
                        continue
                    a, b = (anchor, event) if op.op == "next_after_anchor" else (event, anchor)
                    if not before(a.offset_bracket, b.onset_bracket) and not before(b.onset_bracket, a.offset_bracket):
                        gap("adjacency", [anchor, event], "Verify the separating boundary and intervening interval", facets=["onset", "offset"])
        if op.op == "event_duration" and (result["value"] is None or any(e.onset_bracket.lo is None or e.offset_bracket.hi is None for e in relevant)):
            if relevant:
                gap("duration", relevant, "Duration bounds do not yet resolve this operation", facets=["onset", "offset"])
        if op.op == "cooccurrence_frequency":
            unknown = [e for e in relevant if any(e.cooccurrence.get(k, "unknown") == "unknown" for k in op.cooccurrence_targets)]
            if unknown and not (result["value"] or {}).get("winner"):
                gap("cooccurrence", unknown, "Unknown cooccurrence may change the comparison")
        for tid in target_ids:
            if not any(e.target_id == tid for e in relevant) and not ledger.absence_closed(tid, intervals):
                gap("absence", [], "An empty candidate set needs a dedicated absence review after scanning",
                    targets=[tid])
        result["coverage_closed"] = ledger.scan_closed(target_ids, intervals)
        result["evidence_gaps"] = gaps
        result["issues"] = list(dict.fromkeys([*result["issues"], *("proof_gap:" + g["kind"] for g in gaps)]))
        result["closed_under_policy"] = not result["issues"] and not query_issues
        if result["evidence_conditional_bounds"] and not result["coverage_closed"]:
            bounds = result["evidence_conditional_bounds"]
            bounds["candidate_upper_bound"] = bounds["upper_bound"]
            bounds["upper_bound"] = None
        if not result["closed_under_policy"] and op.op == "nth_occurrence":
            result["value"] = None
        absence_evidence = [r for review in ledger.absence_reviews if review["target_id"] in target_ids for r in review["evidence_refs"]]
        result["evidence_refs"] = list(dict.fromkeys(result["evidence_refs"] + absence_evidence))
        results.append(result)
        all_gaps.extend(gaps)
    return {"results": results, "closed_under_policy": bool(results) and all(r["closed_under_policy"] for r in results),
        "issues": list(dict.fromkeys([*query_issues, *(i for r in results for i in r["issues"])])),
        "required_intervals": [s for r in results for s in r["required_intervals"]],
        "evidence_refs": list(dict.fromkeys(ref for r in results for ref in r["evidence_refs"])),
        "evidence_gaps": all_gaps}


def scope_gaps(ledger):
    """Semantic scope discovery uses the same timeline/review mechanism."""
    events = [e for e in ledger.canonical_events() if e.target_id == "__r3_scope__"]
    interval = [[ledger.tiles[0].core[0], ledger.tiles[-1].core[1]]]
    return [asdict(ReviewTask("scope", "__scope__", ["__r3_scope__"], [e.event_id for e in events], interval,
        "Bind the semantic stage using confirmed identity, order and both boundaries",
        ["target", "unit", "identity", "onset", "offset"]))]
