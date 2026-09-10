"""Executable temporal queries; no model-generated arithmetic or fabricated total order."""

from __future__ import annotations

from itertools import combinations
from typing import Any

from qwen3vl_agent.r3.ledger import EventLedger
from qwen3vl_agent.r3.planning import overlaps
from qwen3vl_agent.r3.types import Bracket, EventRecord, Operation


def boundary(event: EventRecord, basis: str) -> Bracket:
    return event.onset_bracket if basis == "onset" else event.offset_bracket


def before(a: Bracket, b: Bracket) -> bool:
    return a.hi is not None and b.lo is not None and a.hi < b.lo


def inclusion(event: EventRecord, rule: str, start: Bracket, end: Bracket) -> str:
    if rule == "intersects":
        visible = (
            event.visible_span
            if event.fact_kind in {"visual_event", "screen_text_event"}
            else (event.onset_bracket.hi, event.offset_bracket.lo)
        )
        if (
            start.hi is not None
            and end.lo is not None
            and visible[0] is not None
            and visible[1] is not None
            and visible[1] >= start.hi
            and visible[0] < end.lo
        ):
            return "inside"
        if (
            event.offset_bracket.hi is not None
            and start.lo is not None
            and event.offset_bracket.hi <= start.lo
        ) or (
            event.onset_bracket.lo is not None
            and end.hi is not None
            and event.onset_bracket.lo >= end.hi
        ):
            return "outside"
        return "unknown"
    anchor = event.onset_bracket if rule == "starts_inside" else event.offset_bracket
    if (
        anchor.lo is not None
        and anchor.hi is not None
        and start.hi is not None
        and end.lo is not None
        and anchor.lo >= start.hi
        and anchor.hi < end.lo
    ):
        return "inside"
    if (anchor.hi is not None and start.lo is not None and anchor.hi < start.lo) or (
        anchor.lo is not None and end.hi is not None and anchor.lo >= end.hi
    ):
        return "outside"
    return "unknown"


def ordered_select(
    events: list[EventRecord], *, basis: str = "onset", reverse: bool = False, k: int | None = None
) -> tuple[list[EventRecord], list[str]]:
    remaining, selected, issues = list(events), [], []
    while remaining and (k is None or len(selected) < k):
        candidates = [
            a
            for a in remaining
            if not any(
                before(boundary(a, basis), boundary(b, basis))
                if reverse
                else before(boundary(b, basis), boundary(a, basis))
                for b in remaining
                if a is not b
            )
        ]
        if not candidates:
            issues.append("inconsistent_temporal_order")
            break
        if len(candidates) != 1:
            issues.append("overlapping_or_unknown_order_brackets")
        # Stable serialization only; the ambiguity is explicitly propagated, never a proof.
        pick = min(candidates, key=lambda e: e.event_id)
        selected.append(pick)
        remaining.remove(pick)
    return selected, list(dict.fromkeys(issues))


def _selector(
    events: list[EventRecord], operation: Operation
) -> tuple[list[EventRecord], list[str]]:
    issues: list[str] = []
    mode = operation.selection
    per_category = operation.group_by == "category" or mode.endswith("per_category")
    if per_category:
        groups: dict[str, list[EventRecord]] = {}
        for event in events:
            groups.setdefault(event.category, []).append(event)
        reduced = []
        reverse = mode == "last_per_category" or (
            mode == "all" and operation.op in {"last_k", "last_occurrence"}
        )
        for values in groups.values():
            chosen, pending = ordered_select(values, basis=operation.basis, reverse=reverse, k=1)
            reduced.extend(chosen)
            issues.extend(pending)
        events = reduced
    if mode == "unique" and len(events) != 1:
        issues.append("target_occurrence_not_unique")
    if mode in {"first", "last"}:
        events, pending = ordered_select(events, basis=operation.basis, reverse=mode == "last", k=1)
        issues.extend(pending)
    return events, issues


def duration_bounds(event: EventRecord) -> tuple[float, float | None]:
    s, e = event.onset_bracket, event.offset_bracket
    lo = max(0.0, e.lo - s.hi) if e.lo is not None and s.hi is not None else 0.0
    hi = max(0.0, e.hi - s.lo) if e.hi is not None and s.lo is not None else None
    return lo, hi


def union_length(intervals: list[tuple[float, float]]) -> float:
    total, current = 0.0, None
    for lo, hi in sorted(intervals):
        if hi <= lo:
            continue
        if current is None:
            current = [lo, hi]
        elif lo <= current[1]:
            current[1] = max(current[1], hi)
        else:
            total += current[1] - current[0]
            current = [lo, hi]
    return total + (current[1] - current[0] if current else 0.0)


def _minimum_distinct(events: list[EventRecord], ledger: EventLedger) -> tuple[int, bool]:
    """Minimum compatible event partition (graph colouring), bounded to 12 vertices."""
    if not events:
        return 0, True
    if not ledger.pending_relations(events):
        return len(events), True
    owners = {m: i for i, event in enumerate(events) for m in event.member_ids}
    edges = {i: set() for i in range(len(events))}
    for rel in ledger.relations.values():
        if (
            rel.relation == "distinct_occurrences"
            and rel.left_id in owners
            and rel.right_id in owners
        ):
            a, b = owners[rel.left_id], owners[rel.right_id]
            if a != b:
                edges[a].add(b)
                edges[b].add(a)
    for a, b in combinations(range(len(events)), 2):
        if events[a].target_id != events[b].target_id or ledger.certainly_distinct(
            events[a], events[b]
        ):
            edges[a].add(b)
            edges[b].add(a)
    order = sorted(edges, key=lambda i: -len(edges[i]))
    if len(events) > 12:
        clique: list[int] = []
        for v in order:
            if all(u in edges[v] for u in clique):
                clique.append(v)
        return len(clique), False
    colours: dict[int, int] = {}

    def colour(pos: int, count: int) -> bool:
        if pos == len(order):
            return True
        vertex = order[pos]
        forbidden = {colours[n] for n in edges[vertex] if n in colours}
        max_used = max(colours.values(), default=-1)
        for c in range(min(count, max_used + 2)):
            if c not in forbidden:
                colours[vertex] = c
                if colour(pos + 1, count):
                    return True
                del colours[vertex]
        return False

    for n in range(1, len(events) + 1):
        if colour(0, n):
            return n, True
    return len(events), True


def project(event: EventRecord, key: str) -> Any:
    if key in {"description", "category", "actor_ref", "object_ref"}:
        return getattr(event, key)
    return event.attributes.get(key)


def _one_operation(
    operation: Operation,
    all_events: list[EventRecord],
    ledger: EventLedger,
    scope: tuple[float, float],
    start: Bracket,
    end: Bracket,
) -> dict[str, Any]:
    specs = {s.target_id: s for s in ledger.query.targets}
    candidates, pending_inclusion = [], []
    for event in all_events:
        if event.target_id not in operation.target_ids:
            continue
        membership = inclusion(event, specs[event.target_id].inclusion_rule, start, end)
        if membership != "outside":
            candidates.append(event)
            if membership == "unknown":
                pending_inclusion.append(event.event_id)
    selected, issues = _selector(candidates, operation)
    required = [scope]
    value: Any = None
    conditional: dict[str, Any] | None = None
    supporting_events = []
    missing = [
        target
        for target in operation.target_ids
        if not any(e.target_id == target for e in candidates)
    ]
    if operation.op == "count_occurrences":
        definite = [
            e for e in selected if e.status == "accepted" and e.event_id not in pending_inclusion
        ]
        lower, exact_bound = _minimum_distinct(definite, ledger)
        upper = len(selected)
        conditional = {
            "lower_bound": lower,
            "upper_bound": upper,
            "bounds_exact": exact_bound,
            "observed_accepted_count": lower,
        }
        value = lower if lower == upper else None
    elif operation.op in {"first_occurrence", "last_occurrence", "first_k", "last_k", "nth_occurrence"}:
        reverse = operation.op.startswith("last")
        k = operation.k if operation.op.endswith("_k") or operation.op == "nth_occurrence" else 1
        selected, pending = ordered_select(selected, basis=operation.basis, reverse=reverse, k=k)
        issues.extend(pending)
        if len(selected) < k:
            issues.append("fewer_than_requested_events")
        if len(selected) >= k and not pending:
            if reverse:
                cut = min(
                    boundary(e, operation.basis).lo
                    if boundary(e, operation.basis).lo is not None
                    else scope[0]
                    for e in selected
                )
                required = [(max(scope[0], cut), scope[1])]
            else:
                cut = max(
                    boundary(e, operation.basis).hi
                    if boundary(e, operation.basis).hi is not None
                    else scope[1]
                    for e in selected
                )
                required = [
                    (scope[0], min(scope[1], max(cut, max(e.visible_span[1] for e in selected))))
                ]
        if reverse:
            selected.reverse()  # Tail-K is presented in chronological order.
        values = [project(e, operation.project) for e in selected]
        if operation.op == "nth_occurrence":
            value = values[k - 1] if len(values) >= k and not pending else None
        else:
            value = values if operation.op.endswith("_k") else (values[0] if values else None)
    elif operation.op in {"order_events", "localize_event"}:
        selected, pending = ordered_select(selected, basis=operation.basis)
        issues.extend(pending)
        if missing:
            issues.append("missing_targets:" + ",".join(missing))
        if operation.op == "order_events":
            value = [project(e, operation.project) for e in selected]
            if pending:
                value = {
                    "kind": "partial_order",
                    "events": {e.event_id: project(e, operation.project) for e in selected},
                    "confirmed_before": [
                        [a.event_id, b.event_id]
                        for a in selected
                        for b in selected
                        if a is not b
                        and before(boundary(a, operation.basis), boundary(b, operation.basis))
                    ],
                    "unresolved_pairs": [
                        [a.event_id, b.event_id]
                        for a, b in combinations(selected, 2)
                        if not before(boundary(a, operation.basis), boundary(b, operation.basis))
                        and not before(boundary(b, operation.basis), boundary(a, operation.basis))
                    ],
                }
        else:
            value = [
                {
                    "event_id": e.event_id,
                    "onset": e.onset_bracket.to_list(),
                    "offset": e.offset_bracket.to_list(),
                }
                for e in selected
            ]
            if any(e.onset_bracket.lo is None or e.offset_bracket.hi is None for e in selected):
                issues.append("localization_boundaries_missing")
            if operation.selection == "unique" and len(selected) != 1:
                issues.append("localization_not_unique")
    elif operation.op in {"next_after_anchor", "previous_before_anchor"}:
        anchors = [
            e
            for e in all_events
            if e.target_id == operation.anchor_target_id
            and inclusion(e, specs[e.target_id].inclusion_rule, start, end) != "outside"
        ]
        if ledger.pending_relations(anchors):
            issues.append("anchor_identity_unresolved")
        if operation.anchor_selection == "unique":
            if len(anchors) != 1:
                issues.append("anchor_not_unique")
        else:
            anchors, pending = ordered_select(
                anchors, reverse=operation.anchor_selection == "last", k=1
            )
            issues.extend(pending)
        if len(anchors) == 1:
            anchor = anchors[0]
            supporting_events.append(anchor)
            if anchor.status != "accepted":
                issues.append("anchor_unconfirmed")
            forward = operation.op == "next_after_anchor"
            eligible = []
            for event in selected:
                a = anchor.offset_bracket if forward else event.offset_bracket
                b = event.onset_bracket if forward else anchor.onset_bracket
                if before(a, b):
                    eligible.append(event)
                elif not before(b, a):
                    issues.append("adjacency_boundary_unresolved")
            selected, pending = ordered_select(
                eligible, basis="onset" if forward else "offset", reverse=not forward, k=1
            )
            issues.extend(pending)
            if selected:
                event = selected[0]
                value = project(event, operation.project)
                if (
                    forward
                    and anchor.offset_bracket.lo is not None
                    and event.onset_bracket.hi is not None
                ):
                    required = [(anchor.offset_bracket.lo, event.onset_bracket.hi)]
                elif (
                    not forward
                    and event.offset_bracket.lo is not None
                    and anchor.onset_bracket.hi is not None
                ):
                    required = [(event.offset_bracket.lo, anchor.onset_bracket.hi)]
                # Unique-anchor exclusion requires discovery over the allowed query scope.
                if operation.anchor_selection == "unique":
                    required = [scope]
                elif operation.anchor_selection == "first":
                    required.append((scope[0], anchor.visible_span[1]))
                else:
                    required.append((anchor.visible_span[0], scope[1]))
            else:
                issues.append("no_adjacent_event")
        else:
            selected = []
    elif operation.op == "event_duration":
        if not selected:
            issues.append("missing_duration_event")
        bounds = [duration_bounds(e) for e in selected]
        if any(b[1] is None for b in bounds):
            issues.append("duration_boundaries_missing")
        if operation.duration_aggregation == "single":
            if len(bounds) != 1:
                issues.append("duration_event_not_unique")
            value = {"min_sec": bounds[0][0], "max_sec": bounds[0][1]} if len(bounds) == 1 else None
        elif operation.duration_aggregation == "compare":
            winners = []
            for i, own in enumerate(bounds):
                if all(
                    i == j
                    or (
                        other[1] is not None and own[0] > other[1]
                        if operation.duration_comparison == "longest"
                        else own[1] is not None and own[1] < other[0]
                    )
                    for j, other in enumerate(bounds)
                ):
                    winners.append(selected[i])
            value = project(winners[0], operation.project) if len(winners) == 1 else None
            if len(winners) != 1:
                issues.append("duration_comparison_unresolved")
        elif operation.duration_aggregation == "sum":
            value = {
                "min_sec": sum(b[0] for b in bounds),
                "max_sec": sum(b[1] for b in bounds)
                if all(b[1] is not None for b in bounds)
                else None,
            }
        else:
            certain = [
                (e.onset_bracket.hi, e.offset_bracket.lo)
                for e in selected
                if e.onset_bracket.hi is not None and e.offset_bracket.lo is not None
            ]
            possible = [
                (e.onset_bracket.lo, e.offset_bracket.hi)
                for e in selected
                if e.onset_bracket.lo is not None and e.offset_bracket.hi is not None
            ]
            value = {
                "min_sec": union_length(certain),
                "max_sec": union_length(possible) if len(possible) == len(selected) else None,
            }
    elif operation.op == "cooccurrence_frequency":
        names = set(operation.cooccurrence_targets) or {
            name for e in selected for name in e.cooccurrence
        }
        counts = {
            name: {
                "present": sum(e.cooccurrence.get(name) == "present" for e in selected),
                "absent": sum(e.cooccurrence.get(name) == "absent" for e in selected),
                "unknown": sum(e.cooccurrence.get(name, "unknown") == "unknown" for e in selected),
            }
            for name in sorted(names)
        }
        winners = [
            name
            for name in counts
            if all(
                name == other
                or counts[name]["present"] > counts[other]["present"] + counts[other]["unknown"]
                for other in counts
            )
        ]
        value = {
            "episode_count": len(selected),
            "counts": counts,
            "winner": winners[0] if len(winners) == 1 else None,
        }
        if not counts or len(winners) != 1:
            issues.append("cooccurrence_ranking_unresolved")
    selected_ids = {e.event_id for e in selected}
    for event in selected:
        if event.status != "accepted":
            issues.append("unconfirmed_event:" + event.event_id)
        if event.event_id in pending_inclusion:
            issues.append("query_boundary_inclusion_unknown:" + event.event_id)
    if operation.op not in {
        "count_occurrences",
        "localize_event",
        "event_duration",
        "cooccurrence_frequency",
    } and any(
        project(e, operation.project) in (None, "")
        for e in (selected[-1:] if operation.op == "nth_occurrence" else selected)
    ):
        issues.append("requested_attribute_not_observed:" + operation.project)
    possible_relevant = [
        e
        for e in all_events
        if e.target_id in operation.target_ids
        and inclusion(e, specs[e.target_id].inclusion_rule, start, end) != "outside"
        and any(e.visible_span[0] <= hi and e.visible_span[1] >= lo for lo, hi in required)
    ]
    pending_relations = ledger.pending_relations(possible_relevant)
    if pending_relations:
        issues.append("unresolved_event_identity")
    # Unselected uncertain candidates can still change first/last/K/adjacency.
    for event in possible_relevant:
        if event.event_id not in selected_ids and event.status != "accepted":
            issues.append("competing_unconfirmed_event:" + event.event_id)
    if ledger.conflicts:
        issues.append("ledger_conflict")
    return {
        "operation_id": operation.operation_id,
        "op": operation.op,
        "value": value,
        "event_ids": [e.event_id for e in (*selected, *supporting_events)],
        "required_intervals": required,
        "issues": list(dict.fromkeys(issues)),
        "evidence_conditional_bounds": conditional,
        "evidence_refs": list(
            dict.fromkeys(r for e in (*selected, *supporting_events) for r in e.evidence_refs)
        ),
    }


def temporal_reduce(
    ledger: EventLedger,
    scope: tuple[float, float],
    *,
    scope_start: Bracket | None = None,
    scope_end: Bracket | None = None,
    additional_issues: list[str] | None = None,
) -> dict[str, Any]:
    if getattr(ledger, "protocol_version", None) == 4:
        from .timeline_proof import temporal_proof
        return temporal_proof(ledger, scope, scope_start=scope_start, scope_end=scope_end,
                              additional_issues=additional_issues)
    start = scope_start or Bracket(scope[0], scope[0])
    end = scope_end or Bracket(scope[1], scope[1])
    events = ledger.canonical_events()
    results = [
        _one_operation(o, events, ledger, scope, start, end) for o in ledger.query.operations
    ]
    issues = list(ledger.query.unresolved) + list(additional_issues or ())
    for result in results:
        coverage_issues = []
        relevant_tiles = [
            t
            for t in ledger.tiles
            if any(overlaps(t.core, tuple(span)) for span in result["required_intervals"])
        ]
        for tile in relevant_tiles:
            if not tile.observed:
                coverage_issues.append("required_observation_incomplete:" + tile.tile_id)
            elif not tile.resolution_met:
                coverage_issues.append("resolution_limited:" + tile.tile_id)
            if tile.audit_needed and not tile.audit_done:
                marker = "negative_window_audit_unresolved:" if tile.audit_attempts else "negative_window_audit_pending:"
                coverage_issues.append(marker + tile.tile_id)
            if tile.certificates and (not tile.target_coverage or any(
                status not in {"observed", "absent_confirmed"} for status in tile.target_coverage.values()
            )):
                coverage_issues.append("target_verification_incomplete:" + tile.tile_id)
            if tile.unresolved:
                coverage_issues.append("window_unresolved:" + tile.tile_id)
            if tile.external_coverage == "unknown":
                coverage_issues.append("external_coverage_unknown:" + tile.tile_id)
        result["coverage_closed"] = bool(relevant_tiles) and not coverage_issues
        result["issues"].extend(coverage_issues)
        if not relevant_tiles:
            result["issues"].append("required_observation_incomplete")
        conditional = result["evidence_conditional_bounds"]
        if conditional and not result["coverage_closed"]:
            conditional["candidate_upper_bound"] = conditional["upper_bound"]
            conditional["upper_bound"] = None
        result["closed_under_policy"] = not result["issues"] and not issues
        if result["op"] == "nth_occurrence" and not result["closed_under_policy"]:
            # Keep the whole prefix as evidence, but do not expose a definite rank when
            # identity, ordering, inclusion or coverage can still change the selected event.
            result["value"] = None
        issues.extend(result["issues"])
    return {
        "results": results,
        "closed_under_policy": bool(results) and not issues,
        "issues": list(dict.fromkeys(issues)),
        "required_intervals": [span for r in results for span in r["required_intervals"]],
        "evidence_refs": list(dict.fromkeys(ref for r in results for ref in r["evidence_refs"])),
    }
