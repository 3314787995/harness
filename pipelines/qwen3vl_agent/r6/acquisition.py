"""Bounded legal windows and deterministic, gap-conditioned action ordering."""

import math

from .schema import ACTION_NAMES
from .types import ProtocolError, digest


def uniform_times(span, count):
    a, b = span
    return [a + (b - a) * (i + 0.5) / count for i in range(count)]


def action(kind, *, span=None, query="", gap_index=None, source_ids=(), bbox=None, fps=2):
    return {
        "kind": kind,
        "gap_index": gap_index,
        "span": list(span) if span else None,
        "query": query,
        "source_ids": list(source_ids),
        "bbox": bbox,
        "fps": fps,
    }


def windows(contract, config):
    result = []
    for a, b in contract.allowed_intervals:
        t = a
        while t < b - 1e-8:
            result.append(
                (
                    max(a, t - config.context_seconds_each_side),
                    min(b, t + config.core_window_seconds + config.context_seconds_each_side),
                )
            )
            t += config.core_window_seconds
    return result


def clip_batches(span, fps, cap):
    a, b = span
    count = max(1, math.ceil((b - a) * fps))
    times = [a + (i + 0.5) * (b - a) / count for i in range(count)]
    for index in range(0, count, cap):
        chosen = times[index : index + cap]
        lo = a if index == 0 else (times[index - 1] + times[index]) / 2
        hi = b if index + cap >= count else (times[index + cap - 1] + times[index + cap]) / 2
        yield (lo, hi), chosen


def initial_actions(request, query, media, texts, config):
    result = []
    for proposed in query["initial_actions"]:
        if (
            proposed["kind"] == "search_allowed_text"
            and texts.sources
            or proposed["kind"] in {"observe_clip", "expand_context"}
            and proposed["span"]
        ):
            result.append(proposed)
    if result:
        return result
    if texts.sources:
        return [action("search_allowed_text", query=" ".join(query["discriminators"]))]
    if "video" not in request.allowed_modalities:
        return []
    if request.reference_scope:
        for a, b in media.contract.allowed_intervals:
            lo, hi = max(a, request.reference_scope[0]), min(b, request.reference_scope[1])
            if lo < hi:
                return [
                    action(
                        "observe_clip",
                        span=(lo, min(hi, lo + config.core_window_seconds)),
                        query=query["target_description"],
                    )
                ]
    spans = media.contract.allowed_intervals
    duration = sum(b - a for a, b in spans)
    counts = [max(1, int(config.initial_overview_frames * (b - a) / duration)) for a, b in spans]
    while sum(counts) > config.initial_overview_frames and max(counts) > 1:
        counts[counts.index(max(counts))] -= 1
    for span, count in zip(spans, counts, strict=True):
        result.append(
            {
                **action("observe_clip", span=span, query=query["target_description"]),
                "overview": True,
                "overview_count": count,
            }
        )
    return result


def action_identity(proposed):
    return digest({k: v for k, v in proposed.items() if k not in {"gap_index"}})


def validate_action(proposed, state, contract):
    if proposed["kind"] not in ACTION_NAMES:
        raise ProtocolError("action is not whitelisted")
    if proposed["span"] is not None and not contract.permits_span(proposed["span"]):
        raise ProtocolError("action crosses evidence boundary")
    if set(proposed["source_ids"]) - state["sources"].keys():
        raise ProtocolError("action references undisplayed source")
    if (
        proposed["kind"]
        in {
            "observe_clip",
            "expand_context",
            "resolve_entity",
            "inspect_source_frame_or_crop",
        }
        and "video" not in contract.allowed_modalities
    ):
        raise ProtocolError("visual tool is disallowed")
    if (
        proposed["kind"] in {"observe_clip", "expand_context", "resolve_entity"}
        and proposed["span"] is None
    ):
        raise ProtocolError("local visual action needs a legal span")
    if proposed["kind"] == "inspect_source_frame_or_crop":
        if not proposed["source_ids"]:
            raise ProtocolError("frame inspection requires existing source IDs")
        if any(state["sources"][i]["modality"] != "video" for i in proposed["source_ids"]):
            raise ProtocolError("crop source must be video")
    gap = proposed["gap_index"]
    if gap is not None and not 0 <= gap < len(state["gaps"]):
        raise ProtocolError("unknown gap index")


def select_action(proposals, state, contract, config):
    previous = {r["identity"] for r in state["actions"]}
    legal = []
    for index, proposed in enumerate(proposals):
        try:
            validate_action(proposed, state, contract)
        except ProtocolError as exc:
            state["events"].append(
                {"event": "action_rejected", "reason": str(exc), "action": proposed}
            )
            continue
        if action_identity(proposed) in previous:
            state["events"].append({"event": "identical_observation_blocked", "action": proposed})
            continue
        gap = state["gaps"][proposed["gap_index"]] if proposed["gap_index"] is not None else {}
        kind = proposed["kind"]
        duration = proposed["span"][1] - proposed["span"][0] if proposed["span"] else 0
        key = (
            not gap.get("blocks_answer", False),
            -len(gap.get("candidate_labels", [])),
            kind == "reduce_by_code",
            duration * proposed["fps"],
            index,
        )
        legal.append((key, proposed))
    return min(legal, key=lambda p: p[0])[1] if legal else None


def coverage_complete(state, contract):
    spans = sorted(
        c["span"]
        for c in state["coverage"]
        if not c["overview"] and c["completed"] and c["resolution_met"]
    )
    for lo, hi in contract.allowed_intervals:
        cursor = lo
        for a, b in spans:
            if a <= cursor + 1e-6:
                cursor = max(cursor, b)
        if cursor < hi - 1e-6:
            return False
    return True


def fallback_actions(state, query, contract, config):
    proposals = []
    gaps = state["gaps"] if config.refinement_policy == "targeted" else []
    for index, gap in enumerate(gaps):
        if gap["modality"] == "audio":
            continue
        if gap["span"] is not None and contract.permits_span(gap["span"]):
            kind = (
                "search_allowed_text" if gap["modality"] in {"asr", "subtitle"} else "observe_clip"
            )
            proposals.append(
                action(
                    kind,
                    span=gap["span"],
                    query=gap["desired_observation"],
                    gap_index=index,
                    fps=config.fps_for_fast_events if gap["kind"] == "time" else config.fps_default,
                )
            )
    covered = {tuple(c["span"]) for c in state["coverage"] if not c["overview"]}
    for span in windows(contract, config):
        if span not in covered:
            proposals.append(action("observe_clip", span=span, query=query["target_description"]))
    return proposals
