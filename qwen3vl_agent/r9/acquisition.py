"""Deterministic, gap-driven evidence allocation."""

import math
from itertools import pairwise

from .types import ProtocolError, digest

PRIORITY = {
    k: i
    for i, k in enumerate(
        (
            "task",
            "identity",
            "time",
            "reference_frame",
            "relation",
            "boundary",
            "scale",
            "coverage",
            "route",
            "viewpoint",
            "protocol",
        )
    )
}


def uniform(a, b, n):
    return [a + (b - a) * i / (n - 1) for i in range(n)] if n > 1 else [(a + b) / 2]


def overview(request, contract, config):
    if request.fixed_frames:
        groups = [
            (s, [t for t in request.fixed_frames if s[0] <= t <= s[1]])
            for s in contract.allowed_time_intervals
        ]
    else:
        intervals = contract.allowed_time_intervals
        if len(intervals) > config.initial_overview_frames:
            raise ProtocolError("overview budget cannot represent every permitted interval")
        total = sum(b - a for a, b in intervals)
        counts = [
            max(1, int(config.initial_overview_frames * (b - a) / total)) for a, b in intervals
        ]
        while sum(counts) > config.initial_overview_frames:
            i = max(range(len(counts)), key=lambda i: counts[i])
            counts[i] -= 1
        while sum(counts) < config.initial_overview_frames:
            i = max(
                range(len(counts)), key=lambda i: (intervals[i][1] - intervals[i][0]) / counts[i]
            )
            counts[i] += 1
        groups = [(s, uniform(*s, n)) for s, n in zip(intervals, counts)]
    actions = []
    for window, times in groups:
        for i in range(0, len(times), config.preferred_frames_per_call):
            if not times[i : i + config.preferred_frames_per_call]:
                continue
            actions.append(
                {
                    "kind": "sample",
                    "time_interval": list(window),
                    "times": times[i : i + config.preferred_frames_per_call],
                    "named_gap": "initial_overview",
                    "target": [],
                    "expected_evidence": "objects, facing cues, transitions and bridge landmarks",
                    "sequential": False,
                }
            )
    return actions


def choose(gaps, state, media, config, history, request):
    if request.comparison == "fixed_evidence":
        return None
    used = {digest(a) for a in history}
    seen = {m["timestamp_seconds"] for m in media.catalog.values()}
    for gap in sorted(gaps, key=lambda g: PRIORITY.get(g.kind, 99)):
        if gap.frame_id in media.catalog and gap.bbox:
            action = {
                "kind": "crop",
                "frame_id": gap.frame_id,
                "bbox": gap.bbox,
                "named_gap": gap.kind,
                "target": gap.entities,
                "expected_evidence": gap.detail,
            }
            if digest(action) not in used:
                return action
        intervals = (
            media.contract.intersect(gap.time_interval)
            if gap.time_interval
            else media.contract.allowed_time_intervals
        )
        if gap.kind == "time":
            # Prefix event scans advance monotonically and never infer absence from the overview.
            for a, b in intervals:
                cursor = a
                for c in sorted(state.data["coverage"], key=lambda c: c["span"]):
                    if c.get("sequential") and c["completed"] and c["span"][0] <= cursor + 1e-6:
                        cursor = max(cursor, c["span"][1])
                if cursor >= b:
                    continue
                end = min(b, cursor + (config.preferred_frames_per_call - 1) / config.scan_fps)
                count = min(
                    config.preferred_frames_per_call,
                    max(2, math.ceil((end - cursor) * config.scan_fps) + 1),
                )
                action = {
                    "kind": "sample",
                    "time_interval": [cursor, end],
                    "times": uniform(cursor, end, count),
                    "named_gap": gap.kind,
                    "target": gap.entities,
                    "expected_evidence": gap.detail,
                    "sequential": True,
                }
                if digest(action) not in used:
                    return action
        # Sample largest unobserved temporal gaps, including bridge views with no target.
        windows = []
        for a, b in intervals:
            edges = [a, *sorted(t for t in seen if a < t < b), b]
            windows += [(x, y) for x, y in pairwise(edges) if y - x > 0.02]
        for a, b in sorted(windows, key=lambda s: s[1] - s[0], reverse=True):
            times = uniform(a, b, config.preferred_frames_per_call + 2)[1:-1]
            action = {
                "kind": "sample",
                "time_interval": [a, b],
                "times": times,
                "named_gap": gap.kind,
                "target": gap.entities,
                "expected_evidence": gap.detail,
                "sequential": False,
            }
            if digest(action) not in used:
                return action
    return None
