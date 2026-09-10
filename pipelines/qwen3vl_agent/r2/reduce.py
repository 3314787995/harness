"""Deterministic R2 operators. Inputs remain fallible visual estimates."""

from __future__ import annotations

import itertools
import math
import statistics

from .observation import ObservationSpec


def gap(kind, description, records=(), **extra):
    result = {"kind": kind, "description": description, **extra}
    times = [r["timestamp"] for r in records]
    if times and min(times) < max(times):
        result.setdefault("span", [min(times), max(times)])
    elif times:
        result.setdefault("span", [max(0, times[0] - 0.25), times[0] + 0.25])
    return result


def covered(span, windows):
    cursor = span[0]
    for a, b in sorted(windows):
        if b <= cursor:
            continue
        if a > cursor + 1e-6:
            return False
        cursor = max(cursor, b)
        if cursor >= span[1] - 1e-6:
            return True
    return cursor >= span[1] - 1e-6


def coverage_gaps(required, coverage):
    spans = [
        c["core"] if "core" in c else c["span"]
        for c in coverage
        if c.get("processed", c["completed"])
        and c["resolution_met"]
        and c.get("kind", "observation") != "navigation"
    ]
    return [
        gap(
            "coverage",
            "Required temporal interval lacks completed observation coverage",
            span=list(s),
        )
        for s in required
        if not covered(s, spans)
    ]


def distinct_times(records):
    values = {}
    for r in records:
        values[r["timestamp"]] = r
    return [values[t] for t in sorted(values)]


def positions(records):
    values, gaps = [], []
    for r in distinct_times(records):
        if "source_point" not in r or r["visibility"] != "visible":
            continue
        x, y = r["source_point"]
        reference = r["slot"]["reference_frame"]
        if reference == "unknown":
            gaps.append(gap("reference", "Coordinate reference is unknown", [r]))
            continue
        scale = math.hypot(*r["source_size"])
        if reference != "screen":
            if r["reference_status"] != "stable" or "source_reference_point" not in r:
                gaps.append(gap("reference", "Scene/body reference is not visibly comparable", [r]))
                continue
            x -= r["source_reference_point"][0]
            y -= r["source_reference_point"][1]
            if reference in {"body", "object"}:
                if not r.get("source_scale"):
                    gaps.append(
                        gap("reference", "Body/object-relative motion needs a stable scale", [r])
                    )
                    continue
                scale = r["source_scale"]
        values.append((r["timestamp"], (x / scale, y / scale), r))
    return values, gaps


def compress_runs(values):
    result = []
    for value in values:
        if not result or value != result[-1]:
            result.append(value)
    return result


def trend(values, tolerance):
    if len(values) < 2:
        return "unknown"
    # Equal-duration stages are constructed by the caller; retain all raw measurements.
    baseline = max(abs(statistics.median(values)), 1e-9)
    changes = [(b - a) / baseline for a, b in itertools.pairwise(values)]
    signs = compress_runs(
        [
            "increase" if d > tolerance else "decrease" if d < -tolerance else "stable"
            for d in changes
        ]
    )
    return signs


def stages(timed_values):
    if len(timed_values) < 3:
        return [v for _, v in timed_values]
    start, end = timed_values[0][0], timed_values[-1][0]
    bins = [[], [], []]
    for t, v in timed_values:
        bins[min(2, int(3 * (t - start) / max(end - start, 1e-9)))].append(v)
    return [statistics.median(v) for v in bins if v]


def simplify_path(points, epsilon):
    if len(points) <= 2:
        return points
    a, b = points[0], points[-1]
    dx, dy = b[0] - a[0], b[1] - a[1]
    length = math.hypot(dx, dy)
    distances = [
        abs(dx * (a[1] - p[1]) - (a[0] - p[0]) * dy) / length if length else math.dist(a, p)
        for p in points[1:-1]
    ]
    index = max(range(len(distances)), key=distances.__getitem__) + 1
    if distances[index - 1] <= epsilon:
        return [a, b]
    return simplify_path(points[: index + 1], epsilon)[:-1] + simplify_path(points[index:], epsilon)


def state_sequence(records, slot_ids):
    sequences, brackets, gaps = {}, {}, []
    for sid in slot_ids:
        rows = distinct_times(
            [r for r in records if r["slot_id"] == sid and r["value"] is not None]
        )
        sequence = []
        for i, row in enumerate(rows):
            if not sequence or sequence[-1]["value"] != row["value"]:
                item = {
                    "value": row["value"],
                    "first_seen": row["timestamp"],
                    "evidence_id": row["id"],
                    "change_bracket": [rows[i - 1]["timestamp"] if i else None, row["timestamp"]],
                }
                sequence.append(item)
                if row["value"] is True and i and rows[i - 1]["value"] is False:
                    brackets[sid] = item["change_bracket"]
        sequences[sid] = sequence
        if len(rows) < 2:
            gaps.append(
                gap("coverage", "State sequence needs ordered observations", rows, slot_id=sid)
            )
    order = []
    keys = list(brackets)
    for i, left in enumerate(keys):
        for right in keys[i + 1 :]:
            a, b = brackets[left], brackets[right]
            if a[1] <= b[0]:
                order.append([left, "before", right])
            elif b[1] <= a[0]:
                order.append([right, "before", left])
            else:
                gaps.append(
                    gap(
                        "stage_order",
                        "Overlapping onset brackets do not establish simultaneity/order",
                        span=[min(a[0], b[0]), max(a[1], b[1])],
                    )
                )
    return {"sequences": sequences, "onset_brackets": brackets, "established_order": order}, gaps


def identity_at_time(op, records, store, query, query_time, limit):
    target = op["target_ids"][0]
    anchors = [
        c
        for c in store.containments
        if c["hidden_target_id"] == target and c["status"] == "visible_reveal"
    ]
    if query_time is None:
        aid = op["parameters"].get("query_anchor_id")
        observations = [
            a
            for a in store.anchor_states
            if a["anchor_id"] == aid and a["status"] in {"at", "after"}
        ]
        if observations:
            query_time = min(a["timestamp"] for a in observations)
    if query_time is None or not anchors:
        return {}, [
            gap("identity", "Need an observed reveal and a verified query-time anchor", records)
        ]
    ranks = [r for r in records if r.get("rank") and abs(r["timestamp"] - query_time) <= 1e-6]
    if not ranks:
        return {"query_time": query_time}, [
            gap(
                "boundary",
                "No carrier ranks at the actual query frame",
                span=[max(0, query_time - 0.5), query_time + 0.5],
            )
        ]
    # Choose the closest directly observed reveal, in either direction, never the earliest by fiat.
    anchor = min(anchors, key=lambda c: abs(c["timestamp"] - query_time))
    carrier = anchor["carrier_node"]
    hypotheses, overflow, deps = store.hypotheses({carrier}, limit, preserve_relation=True)
    lo, hi = sorted((anchor["timestamp"], query_time))
    transfers = [
        c
        for c in store.containments
        if c["hidden_target_id"] == target
        and c["status"] in {"possible_transfer", "release"}
        and lo <= c["timestamp"] <= hi
    ]
    outcomes = []
    for h in hypotheses:
        possible = {r["rank"] for r in ranks if h[r["entity_node"]] == h[carrier]}
        outcomes.append(next(iter(possible)) if len(possible) == 1 else None)
    gaps = []
    if overflow or not outcomes or None in outcomes or len(set(outcomes)) != 1 or transfers:
        gaps.append(
            gap("identity", "Answer-relevant carrier identity/transfer remains unresolved", records)
        )
    return {
        "query_time": query_time,
        "reveal": anchor,
        "possible_ranks": sorted({v for v in outcomes if v is not None}),
        "unexpanded_ambiguity": overflow,
        "association_dependencies": deps,
        "propagation": "backward" if anchor["timestamp"] > query_time else "forward",
        "basis": "propagated_inference",
    }, gaps


def reduce_operation(op, query, store, config, query_time=None, query_spans=None):
    all_records = store.records(op["slot_ids"])
    if query_spans is not None and op["op"] != "identity_at_time":
        all_records = [
            r for r in all_records if any(a <= r["timestamp"] <= b for a, b in query_spans)
        ]
    records = [
        r
        for r in all_records
        if r["basis"] == "visual_observation" and r["visibility"] in {"visible", "absent"}
    ]
    kind, params = op["op"], op["parameters"]
    value, gaps = {}, []
    if kind == "identity_at_time":
        value, gaps = identity_at_time(op, records, store, query, query_time, config.max_hypotheses)
    elif not records and kind != "motion_condition_filter":
        gaps = [
            gap(
                "detail", "No directly observed state for this operation", slot_id=op["slot_ids"][0]
            )
        ]
    elif kind == "motion_condition_filter":
        selected, candidates = [], []
        for node in dict.fromkeys(r["entity_node"] for r in all_records):
            rows = [r for r in records if r["entity_node"] == node]
            ps, ref_gaps = positions(rows)
            moving = len(ps) >= 2 and any(
                math.dist(a[1], b[1]) > config.position_tolerance for a, b in itertools.pairwise(ps)
            )
            if len(distinct_times(rows)) >= 2 and any(r.get("motion") == "moving" for r in rows):
                moving = True
            values = [r["value"] for r in rows if r["value"] is not None]
            condition = True
            if params.get("motion_condition"):
                checks = [
                    r["condition_satisfied"]
                    for r in distinct_times(rows)
                    if r.get("condition_satisfied") is not None
                ]
                condition = any(checks) if len(checks) >= 2 else None
                if condition is None:
                    gaps.append(
                        gap("detail", "The specific dynamic condition has not been observed", rows)
                    )
            matches = (
                moving
                and condition is True
                and ("attribute_equals" not in params or params["attribute_equals"] in values)
            )
            candidates.append(
                {
                    "entity_node": node,
                    "motion_observed": moving,
                    "condition_satisfied": condition,
                    "values": values,
                }
            )
            if matches:
                selected.append({"entity_node": node, "attributes": values})
            if not moving and (len(ps) < 2 or ref_gaps):
                gaps.append(
                    gap(
                        "temporal_resolution", "Candidate motion has not been reliably tested", rows
                    )
                )
        if not selected and (
            any(r["visibility"] in {"occluded", "unknown"} for r in all_records)
            or not store.windows
            or not all(
                w["candidate_coverage_complete"] and w["complete"] for w in store.windows.values()
            )
        ):
            gaps.append(
                gap(
                    "coverage",
                    "Negative motion-filter result requires inspected visible candidates",
                    all_records,
                )
            )
        if selected:
            # A positive existential witness does not require resolving unrelated candidates.
            gaps = []
        value = {"selected": selected, "candidates": candidates, "exists": bool(selected)}
    else:
        if any(
            not store.bindable(
                [r for r in records if r["slot"]["target_id"] == target],
                config.max_hypotheses,
                params.get("allow_role_correspondence", False),
            )
            for target in op["target_ids"]
        ):
            gaps.append(
                gap("identity", "Cross-window target correspondence is unresolved", records)
            )
        if kind == "endpoint_delta":
            for sid in op["slot_ids"]:
                rows = distinct_times(
                    [r for r in records if r["slot_id"] == sid and r["value"] is not None]
                )
                if len(rows) < 2:
                    gaps.append(
                        gap(
                            "coverage",
                            "Two aligned endpoint states are required",
                            rows,
                            slot_id=sid,
                        )
                    )
                else:
                    value[sid] = {
                        "before": rows[0]["value"],
                        "after": rows[-1]["value"],
                        "changed": rows[0]["value"] != rows[-1]["value"],
                        "times": [rows[0]["timestamp"], rows[-1]["timestamp"]],
                    }
        elif kind in {"state_sequence", "relation_transition"}:
            value, extra = state_sequence(records, op["slot_ids"])
            gaps.extend(extra)
        elif kind == "periodic_continuation":
            rows = distinct_times(records)
            phases = compress_runs(
                [r.get("phase", r["value"]) for r in rows if r.get("phase", r["value"]) is not None]
            )
            period = next(
                (
                    p
                    for p in range(2, len(phases) // 2 + 1)
                    if all(v == phases[i % p] for i, v in enumerate(phases))
                ),
                None,
            )
            value = {
                "observed_sequence": phases,
                "period": period,
                "current_phase": phases[-1] if phases else None,
                "next_phase": phases[len(phases) % period] if period else None,
                "conditional_on_pattern_continuing": True,
            }
            if not period:
                gaps.append(
                    gap(
                        "phase_alias",
                        "Need two consistent repeated units and a resolved phase; possible R7 dependency",
                        rows,
                    )
                )
            if any(not r.get("adjacency_resolved") for r in rows[1:]):
                gaps.append(
                    gap(
                        "phase_alias",
                        "Unobserved intermediate phases may change the repeated unit",
                        rows,
                    )
                )
        elif kind == "rotation_pattern":
            angles, times = [], []
            rows = distinct_times(records)
            rotation_types = {r.get("rotation_type", "unknown") for r in rows}
            type_resolved = len(rotation_types) == 1 and "unknown" not in rotation_types
            declared_type = params.get("rotation_type", "unknown")
            if not type_resolved or (
                declared_type != "unknown" and rotation_types != {declared_type}
            ):
                type_resolved = False
                gaps.append(
                    gap(
                        "detail",
                        "Rotation type is unknown or conflicts across observations/query",
                        rows,
                        slot_id=op["slot_ids"][0],
                    )
                )
            for r in rows:
                if not r.get("feature_identifiable") or (
                    angles and not r.get("adjacency_resolved")
                ):
                    gaps.append(
                        gap(
                            "phase_alias",
                            "Rotation feature/winding between frames is unresolved",
                            rows,
                        )
                    )
                if not type_resolved:
                    continue
                if r["rotation_type"] in {"self_spin", "heading"}:
                    angle = r.get("orientation_angle")
                elif "source_point" in r and "source_reference_point" in r:
                    x, y = r["source_point"]
                    cx, cy = r["source_reference_point"]
                    angle = math.degrees(math.atan2(-(y - cy), x - cx))
                else:
                    angle = None
                if angle is not None:
                    angles.append(angle)
                    times.append(r["timestamp"])
            deltas = [(b - a + 180) % 360 - 180 for a, b in itertools.pairwise(angles)]
            if len(angles) < 3 or any(abs(d) >= 170 for d in deltas):
                gaps.append(
                    gap("phase_alias", "Rotation samples cannot safely resolve winding", rows)
                )
            value = {
                "angles_deg": angles,
                "times": times,
                "signed_deltas_deg": deltas,
                "directions": compress_runs(
                    [
                        "counterclockwise" if d > 2 else "clockwise" if d < -2 else "stationary"
                        for d in deltas
                    ]
                ),
                "measurement_kind": "estimated_feature_orientation",
                "rotation_type": next(iter(rotation_types)) if type_resolved else "unknown",
            }
        elif kind == "motion_property_trend" and params.get("metric") == "frequency":
            markers = distinct_times([r for r in records if r.get("cycle_marker")])
            phases = {r.get("phase", r["value"]) for r in markers}
            timed = [
                (b["timestamp"], 1 / (b["timestamp"] - a["timestamp"]))
                for a, b in itertools.pairwise(markers)
            ]
            if (
                not params.get("phase_unit")
                or len(phases) != 1
                or len(timed) < 2
                or any(not r.get("adjacency_resolved") for r in markers[1:])
            ):
                gaps.append(
                    gap(
                        "phase_alias",
                        "Frequency requires at least three returns of the same named full-cycle phase",
                        markers,
                    )
                )
            value = {
                "metric": "frequency",
                "measurements": timed,
                "unit": params.get("phase_unit"),
                "trend": trend(stages(timed), config.trend_tolerance),
            }
        else:
            ps, extra = positions(records)
            gaps.extend(extra)
            points = [p for _, p, _ in ps]
            if len(ps) < 3:
                gaps.append(
                    gap(
                        "temporal_resolution",
                        "At least three comparable ordered point observations are required",
                        records,
                    )
                )
            if kind == "direction_sequence":
                axis = 0 if params.get("axis", "x") == "x" else 1
                positive, negative = ("right", "left") if axis == 0 else ("down", "up")
                deltas = [b[1][axis] - a[1][axis] for a, b in itertools.pairwise(ps)]
                signs = [
                    positive
                    if d > config.position_tolerance
                    else negative
                    if d < -config.position_tolerance
                    else "stationary"
                    for d in deltas
                ]
                value = {
                    "points": [(t, p) for t, p, _ in ps],
                    "raw_deltas": deltas,
                    "tolerance": config.position_tolerance,
                    "direction_segments": compress_runs(signs),
                    "filtered_changes": [d for d in deltas if abs(d) <= config.position_tolerance],
                }
            elif kind == "path_shape":
                simplified = simplify_path(points, config.position_tolerance) if points else []
                turns = []
                for a, b, c in zip(simplified, simplified[1:], simplified[2:]):
                    u, v = (b[0] - a[0], b[1] - a[1]), (c[0] - b[0], c[1] - b[1])
                    turns.append(
                        math.degrees(
                            math.atan2(u[0] * v[1] - u[1] * v[0], u[0] * v[0] + u[1] * v[1])
                        )
                    )
                extent = math.hypot(
                    max((p[0] for p in points), default=0) - min((p[0] for p in points), default=0),
                    max((p[1] for p in points), default=0) - min((p[1] for p in points), default=0),
                )
                value = {
                    "ordered_path": points,
                    "simplified_path": simplified,
                    "turns_deg": turns,
                    "closed": len(points) > 2
                    and math.dist(points[0], points[-1])
                    <= max(config.position_tolerance, extent * 0.1),
                    "aspect_preserved": True,
                }
            elif kind == "motion_property_trend":
                metric = params.get("metric")
                if metric == "speed":
                    timed = [
                        (b[0], math.dist(a[1], b[1]) / (b[0] - a[0]))
                        for a, b in itertools.pairwise(ps)
                    ]
                    value = {
                        "metric": metric,
                        "local_displacement_rates": timed,
                        "stage_medians": stages(timed),
                        "trend": trend(stages(timed), config.trend_tolerance),
                        "path_length_not_measured": True,
                    }
                else:
                    peaks = [
                        (t, math.dist(p, (0, 0))) for t, p, r in ps if r.get("phase") == "peak"
                    ]
                    if (
                        any(r["slot"]["reference_frame"] not in {"body", "object"} for r in records)
                        or len(peaks) < 3
                    ):
                        gaps.append(
                            gap(
                                "reference",
                                "Amplitude needs three observed extrema relative to a stable body/object scale",
                                records,
                            )
                        )
                    value = {
                        "metric": "amplitude",
                        "observed_extrema": peaks,
                        "trend": trend([v for _, v in peaks], config.trend_tolerance),
                    }
    spec = ObservationSpec.from_query({**query, "operations": [op]})
    for row in records:
        for reason in spec.record_missing(row, stored=True):
            gaps.append(gap("detail", reason, [row], slot_id=row["slot_id"]))
    if kind not in {"endpoint_delta", "motion_condition_filter", "identity_at_time"}:
        unreadable = [r for r in all_records if r["visibility"] in {"unknown", "occluded"}]
        if unreadable:
            gaps.append(gap("detail", "Required intermediate states remain unreadable", unreadable))
    gaps.extend(
        store.observation_conflicts(records, config.position_tolerance, config.max_hypotheses)
    )
    return {
        "operation_id": op["id"],
        "op": kind,
        "value": value,
        "status": "unresolved" if gaps else "supported",
        "evidence_ids": [r["id"] for r in records],
        "gaps": gaps,
        "basis": "program_reduction_of_visual_estimates",
    }


def reduce_query(query, store, config, required, coverage, query_time=None):
    operations = [
        reduce_operation(op, query, store, config, query_time, required)
        for op in query["operations"]
    ]
    gaps = coverage_gaps(required, coverage)
    for operation in operations:
        if operation["op"] == "identity_at_time" and operation["value"].get("reveal"):
            value = operation["value"]
            path = sorted((value["query_time"], value["reveal"]["timestamp"]))
            gaps.extend(coverage_gaps([path], coverage))
    gaps += [g for operation in operations for g in operation["gaps"]]
    gaps += [
        g
        for g in store.gaps
        if not g["resolved"]
        and store.windows.get(g.get("window_id"), {}).get("query_relevant", True)
    ]
    gaps += [gap("protocol", reason) for reason in query.get("unresolved", [])]
    result = {
        "operations": operations,
        "gaps": gaps,
        "sufficient": not gaps and all(o["status"] == "supported" for o in operations),
        "finite_sampling_only": True,
        "revision": store.revision,
    }
    store.save_derived(result)
    return result
