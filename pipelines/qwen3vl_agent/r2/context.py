"""Budgeted terminal views. Full observations remain in the append-only store."""

import copy
import json

from .prompts import prompt
from .types import BudgetExhausted


def spread(items, count):
    if len(items) <= count:
        return list(items)
    if count <= 0:
        return []
    if count == 1:
        return items[:1]
    return [items[round(i * (len(items) - 1) / (count - 1))] for i in range(count)]


def summarize_gaps(gaps):
    """A bounded display view, never a replacement for the original gap store.

    Example spans are not merged into a continuous interval. Unshown groups
    remain explicitly unresolved instead of disappearing from certification.
    """
    groups = {}
    for gap in gaps:
        key = (gap.get("kind", "unknown"), gap.get("slot_id"))
        group = groups.setdefault(
            key, {"kind": key[0], "count": 0, "examples": [], "example_spans": []}
        )
        if key[1] is not None:
            group["slot_id"] = key[1]
        group["count"] += 1
        description = gap.get("description", "")[:400]
        if description not in group["examples"] and len(group["examples"]) < 2:
            group["examples"].append(description)
        span = gap.get("span")
        if span and span not in group["example_spans"] and len(group["example_spans"]) < 3:
            group["example_spans"].append(span)
    return list(groups.values())[:16], {
        "records": len(gaps),
        "groups": len(groups),
        "groups_shown": min(16, len(groups)),
        "basis": "program_display_summary; all original gaps remain unresolved",
    }


def fit_final_payload(payload, limit):
    """Fit the *rendered* prompt and leave room for a bounded format-repair message.

    Never shorten the question/options or change raw media/time mappings. If even
    this protected context cannot fit, fail with diagnostics instead of guessing.
    """
    result = copy.deepcopy(payload)
    before = len(prompt("final", result))
    gap_counts = {}
    derived = result.setdefault("derived", {})
    for name, owner in [
        ("query", derived),
        *[(f"operation:{i}", op) for i, op in enumerate(derived.get("operations", []))],
    ]:
        if owner.get("gaps"):
            owner["gaps"], gap_counts[name] = summarize_gaps(owner["gaps"])
    # This also protects callers replaying pre-1.5 state with repeated snippets.
    windows = result.get("raw_windows", [])
    result["raw_windows"] = list({json.dumps(w, sort_keys=True): w for w in windows}.values())
    window_omissions = len(windows) - len(result["raw_windows"])
    # A repair includes up to 12k prior output plus labels and the allowed IDs.
    reserve = min(20000, limit // 3)
    target = min(48000, limit - reserve)
    protected = copy.deepcopy(result)
    protected["observations"] = []
    protected["derived"] = {"operations": [], "gaps": result.get("derived", {}).get("gaps", [])}
    # 48k is a compactness target, not a second hard cap on a long original question.
    target = max(target, min(limit - reserve, len(prompt("final", protected)) + 16000))
    fields = {
        "id",
        "entity_node",
        "slot_id",
        "timestamp",
        "source_frame_id",
        "visibility",
        "basis",
        "value",
        "description",
        "source_point",
        "source_reference_point",
        "source_scale",
        "reference_status",
        "rotation_type",
        "feature_identifiable",
        "adjacency_resolved",
        "orientation_angle",
        "phase",
        "cycle_marker",
        "rank",
        "motion",
        "condition_satisfied",
        "slot",
        "source_size",
    }
    original = result.get("observations", [])
    compact = [{k: v for k, v in r.items() if k in fields} for r in original]
    result["observations"] = compact
    omitted_values = []

    def bounded(value, path):
        if isinstance(value, list) and len(value) > 16:
            omitted_values.append({"path": path, "original_count": len(value), "shown": 16})
            value = spread(value, 16)
        if isinstance(value, list):
            return [bounded(v, path + "[]") for v in value]
        if isinstance(value, dict):
            return {k: bounded(v, path + "." + k) for k, v in value.items()}
        return value

    for op in result.get("derived", {}).get("operations", []):
        op["value"] = bounded(op.get("value", {}), op.get("operation_id", "operation"))
    result["context_omissions"] = {
        "derived_arrays": omitted_values,
        "observation_count": 0,
        "gap_summaries": gap_counts,
        "duplicate_raw_windows": window_omissions,
    }

    def align_references():
        allowed = {r["id"] for r in result["observations"]}

        def visit(item):
            if isinstance(item, dict):
                for key, value in list(item.items()):
                    if key == "evidence_ids":
                        item[key] = [ref for ref in value if ref in allowed]
                    else:
                        visit(value)
            elif isinstance(item, list):
                for value in item:
                    visit(value)

        visit(result["derived"])
        visit(result.get("assessment_policy", {}))

    align_references()
    while len(prompt("final", result)) > target and result["observations"]:
        result["observations"] = spread(result["observations"], len(result["observations"]) // 2)
        result["context_omissions"]["observation_count"] = len(original) - len(
            result["observations"]
        )
        align_references()
    if len(prompt("final", result)) > target:
        # Preserve operation identities and unresolved gaps, but explicitly withhold
        # overlong derived values; raw visual evidence is still supplied unchanged.
        for op in result.get("derived", {}).get("operations", []):
            if op.get("value"):
                op["value"] = {"withheld_due_to_context_budget": True}
                op["status"] = "unresolved"
        result["derived"]["sufficient"] = False
        if "assessment_policy" in result:
            result["assessment_policy"]["complete_support_allowed"] = False
        result["context_omissions"]["derived_values_withheld"] = True
    mode = "standard"
    if len(prompt("final", result)) > target:
        # Internal diagnostics must not prevent an otherwise affordable final
        # choice. Rebuild from protected input plus a small evidence view.
        mode = "best_effort"
        target = limit - reserve
        operations = result.get("assessment_policy", {}).get("operations", {})
        result["assessment_policy"] = {
            "complete_support_allowed": False,
            "operations": {oid: {"status": "unresolved", "evidence_ids": []} for oid in operations},
            "instruction": "Best-effort prediction only; all assessments must be unknown. Internal details were omitted, not resolved.",
        }
        result["derived"] = {
            "sufficient": False,
            "operations": [],
            "gaps": [
                {
                    "kind": "detail",
                    "description": "Evidence certification unavailable; details remain in the original trace.",
                }
            ],
        }
        result["unresolved"] = [
            "Program best-effort terminal view: evidence/diagnostic details omitted; uncertainty remains."
        ]
        result["context_omissions"] = {
            "best_effort": True,
            "gap_records": sum(g["records"] for g in gap_counts.values()),
            "observation_count": len(original),
            "full_trace_preserved": True,
        }
        # Keep exact identities, times and geometry; limit descriptive text only.
        essential = {
            "id",
            "entity_node",
            "slot_id",
            "timestamp",
            "source_frame_id",
            "visibility",
            "basis",
            "source_point",
            "source_reference_point",
            "source_scale",
            "source_size",
            "reference_status",
            "orientation_angle",
            "rotation_type",
            "feature_identifiable",
            "adjacency_resolved",
            "value",
            "description",
        }
        candidates = [
            {
                k: (v[:240] if isinstance(v, str) and k in {"description", "value"} else v)
                for k, v in row.items()
                if k in essential
            }
            for row in compact
        ]
        for count in (8, 4, 2, 1, 0):
            result["observations"] = spread(candidates, count)
            result["context_omissions"]["observation_count"] = len(original) - len(
                result["observations"]
            )
            if len(prompt("final", result)) <= target:
                break
    after = len(prompt("final", result))
    diagnostic = {
        "original_chars": before,
        "final_chars": after,
        "target_chars": target,
        "hard_limit": limit,
        "repair_reserve_chars": reserve,
        "observations_original": len(original),
        "observations_shown": len(result["observations"]),
        "omissions": copy.deepcopy(result["context_omissions"]),
        "mode": mode,
    }
    if after > target:
        raise BudgetExhausted("protected_terminal_context: " + json.dumps(diagnostic))
    return result, diagnostic
