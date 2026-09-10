"""Coverage plans and deterministic answer-relevant gap actions."""

import math

from .types import ProtocolError


def make_windows(
    spans, contract, config, *, fast=False, prefix="base", refined=False, shifted=False
):
    label = "refine_" if refined else "fast_" if fast else ""
    core = getattr(config, label + "core_sec")
    pad = getattr(config, label + "context_sec")
    fps = getattr(config, label + "fps")
    result = []
    for span in spans:
        for permitted in contract.intersect(span):
            for j in range(max(1, math.ceil((permitted[1] - permitted[0]) / core - 1e-9))):
                lo, hi = permitted[0] + j * core, min(permitted[1], permitted[0] + (j + 1) * core)
                access = next(s for s in contract.allowed_time_intervals if s[0] <= lo < s[1])
                context = [max(access[0], lo - pad), min(access[1], hi + pad)]
                result.append(
                    {
                        "id": f"{prefix}_{len(result):05d}",
                        "core": [lo, hi],
                        "span": context,
                        "fps": fps,
                        "shifted": shifted,
                        "kind": "observation",
                    }
                )
    return result


def sample_times(window, source_fps=None):
    fps = min(window["fps"], source_fps) if source_fps else window["fps"]
    start, end = window["span"]
    if window.get("endpoint"):
        count = window["endpoint"]
        return [start + (end - start) * i / (count - 1) for i in range(count)], None
    start += 0.5 / fps if window.get("shifted") else 0
    times = [
        start + i / fps
        for i in range(max(1, math.ceil((end - start) * fps - 1e-9)))
        if start + i / fps < end
    ]
    return times, fps


def initial_spans(query, request, contract, resolved):
    if request.query_scope is not None and not isinstance(request.query_scope, str):
        spans = contract.intersect(request.query_scope)
        if not spans:
            raise ProtocolError("query scope contains no observable evidence")
        return spans
    scope = query["scope"]
    permitted = contract.allowed_time_intervals
    if scope["kind"] == "interval":
        spans = contract.intersect(scope["interval"])
        if sum(b - a for a, b in spans) < scope["interval"][1] - scope["interval"][0] - 1e-6:
            raise ProtocolError("compiled scope would expand or bridge media permissions")
        return spans
    if scope["kind"] == "start":
        return [(permitted[0][0], min(permitted[0][1], permitted[0][0] + 4))]
    if scope["kind"] == "end":
        return [(max(permitted[-1][0], permitted[-1][1] - 4), permitted[-1][1])]
    if scope["kind"] == "semantic":
        return resolved.get("scope", [])
    return list(permitted)


PRIORITY = {
    "localization": 0,
    "coverage": 1,
    "identity": 2,
    "conflict": 2,
    "stage_order": 2,
    "boundary": 3,
    "reference": 4,
    "temporal_resolution": 4,
    "phase_alias": 4,
    "detail": 5,
    "protocol": 9,
}


def select_action(gaps, required, contract, attempted, revision, config):
    for issue in sorted(gaps, key=lambda g: PRIORITY.get(g["kind"], 8)):
        kind = issue["kind"]
        if kind == "protocol":
            continue
        span = issue.get("span") or (
            required[0] if required else contract.allowed_time_intervals[0]
        )
        a, b = span
        # A frame-bound uncertainty is a point in time, but observing it needs
        # a nonempty permitted window. Keep the evidence gap itself unchanged.
        if a == b:
            a, b = max(0, a - config.refine_core_sec / 2), b + config.refine_core_sec / 2
        if kind in {"boundary", "identity", "stage_order"}:
            a, b = max(0, a - 0.5), b + 0.5
        spans = contract.intersect((a, b))
        if not spans:
            continue
        action = (
            "relocate"
            if kind == "localization"
            else "crop"
            if kind == "detail" and issue.get("bbox")
            else "full_view"
            if kind == "reference"
            else "bridge_identity"
            if kind == "identity"
            else "phase_shift"
            if kind == "phase_alias"
            else "densify"
        )
        # The same raw evidence/action may become useful after a substantive state revision.
        signature = [
            action,
            [list(s) for s in spans],
            issue.get("slot_id"),
            issue.get("bbox"),
            None if kind == "identity" else revision,
        ]
        if signature in attempted:
            continue
        return {
            "action": action,
            "spans": spans,
            "bbox": issue.get("bbox") if action == "crop" else None,
            "gap": issue,
            "signature": signature,
            "windows": make_windows(
                spans,
                contract,
                config,
                prefix=f"repair_{len(attempted)}",
                refined=kind not in {"coverage", "identity", "reference"},
                shifted=action == "phase_shift",
            ),
        }
    return None
