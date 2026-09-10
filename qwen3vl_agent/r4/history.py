"""Explicit, source-backed task updates evaluated as of the permitted query time."""

from __future__ import annotations

import itertools
from typing import Any

from qwen3vl_agent.r4.ledger import normalize


def reduce_tasks(
    ledger: Any,
    target: Any,
    *,
    coverage_closed: bool,
    multi_source: bool,
    query_time: float | None,
    completion_predicate: str | None,
    set_ids: set[str] | None = None,
) -> dict[str, Any]:
    graph = ledger.graph(updates=True)
    updates = ledger.state["task_updates"]
    issues, applicable = [], []
    if graph["conflicts"]:
        issues.append("task_update_identity_conflict")
    for representative, members in graph["groups"].items():
        records = [
            updates[m] for m in members if updates[m]["set_id"] in (set_ids or {target.set_id})
        ]
        if not records:
            continue
        update = dict(records[0])
        update["update_id"] = representative
        update["evidence_refs"] = sorted({r for v in records for r in v["evidence_refs"]})
        if any(
            any(
                v.get(k) != update.get(k)
                for k in (
                    "kind",
                    "owner",
                    "task_id",
                    "item_key",
                    "quantity",
                    "unit",
                    "replacement_item",
                    "replacement_quantity",
                    "completion_predicate",
                )
            )
            for v in records
        ):
            issues.append("task_assertion_conflict:" + representative)
            continue
        update["observed_time"] = min(
            (v["observed_time"] for v in records if v["observed_time"] is not None), default=None
        )
        explicit_times = {v["effective_time"] for v in records if v.get("explicit_effective_time")}
        if len(explicit_times) > 1:
            issues.append("task_effective_time_conflict:" + representative)
            continue
        update["effective_time"] = (
            next(iter(explicit_times))
            if explicit_times
            else min(
                (v["effective_time"] for v in records if v["effective_time"] is not None),
                default=None,
            )
        )
        if target.owner and update["owner"] != target.owner:
            continue
        if target.task_id and update["task_id"] != target.task_id:
            continue
        if not update["binding_supported"] or not update["owner"] or not update["task_id"]:
            issues.append("task_binding_missing:" + representative)
            continue
        if (multi_source or query_time is not None) and not update["has_history_time"]:
            issues.append("history_mapping_missing:" + representative)
            continue
        if update["observed_time"] is None or update["effective_time"] is None:
            issues.append("task_update_time_missing:" + representative)
            continue
        if query_time is not None and (
            update["observed_time"] >= query_time or update["effective_time"] > query_time
        ):
            continue
        if (
            update["kind"] == "complete_item"
            and completion_predicate
            and update["completion_predicate"] != completion_predicate
        ):
            issues.append("completion_predicate_unproven:" + representative)
            continue
        applicable.append(update)
    # Repeated assertions/completions require a same/different decision; do not sum UNKNOWNs.
    uncertain = set()
    by_id = {u["update_id"]: u for u in applicable}
    dependencies: dict[str, str] = {}
    for update in applicable:
        ref = graph["roots"].get(update.get("refers_to"), update.get("refers_to"))
        if ref:
            original = by_id.get(ref)
            if (
                not original
                or ref == update["update_id"]
                or any(original[k] != update[k] for k in ("owner", "task_id", "item_key", "unit"))
            ):
                uncertain.add(update["update_id"])
                issues.append("task_update_reference_invalid")
            else:
                dependencies[update["update_id"]] = ref

    def precedes(a: str, b: str) -> bool:
        visited = set()
        while b in dependencies and b not in visited:
            visited.add(b)
            b = dependencies[b]
            if a == b:
                return True
        return False

    for a, b in itertools.combinations(applicable, 2):
        if all(a[k] == b[k] for k in ("owner", "task_id", "item_key", "kind", "unit")):
            pair = tuple(sorted((a["update_id"], b["update_id"])))
            if (
                pair not in graph["different"]
                and not precedes(*pair)
                and not precedes(*reversed(pair))
            ):
                uncertain.update(pair)
                issues.append("task_update_identity_unknown:" + ":".join(pair))
        if (
            a["effective_time"] == b["effective_time"]
            and a["owner"] == b["owner"]
            and a["task_id"] == b["task_id"]
            and a["item_key"] == b["item_key"]
            and (a["kind"] not in {"complete_item", "add_item"} or b["kind"] != a["kind"])
            and not precedes(a["update_id"], b["update_id"])
            and not precedes(b["update_id"], a["update_id"])
        ):
            uncertain.update((a["update_id"], b["update_id"]))
            issues.append("task_update_order_unknown")
    states: dict[str, Any] = {}
    completion_records = {}
    known_updates = {u["update_id"] for u in applicable}
    ordered, pending = [], dict(by_id)
    while pending:
        ready = [u for key, u in pending.items() if dependencies.get(key) not in pending]
        if not ready:
            issues.append("task_update_reference_cycle")
            uncertain.update(pending)
            ordered.extend(pending.values())
            break
        update = min(ready, key=lambda u: (u["effective_time"], u["update_id"]))
        ordered.append(update)
        del pending[update["update_id"]]
    for update in ordered:
        item = normalize(update["item_key"], target.normalization)
        key = "::".join((update["owner"], update["task_id"], item, update["unit"]))
        state = states.setdefault(
            key,
            {
                "member": item,
                "owner": update["owner"],
                "task_id": update["task_id"],
                "unit": update["unit"],
                "planned": 0.0,
                "completed": 0.0,
                "active": False,
                "issues": [],
                "evidence_refs": [],
                "updates": [],
            },
        )
        state["evidence_refs"] = sorted(set(state["evidence_refs"] + update["evidence_refs"]))
        state["updates"].append(update["update_id"])
        if update["update_id"] in uncertain:
            state["issues"].append("ambiguous_update")
            continue
        kind, quantity = update["kind"], update["quantity"]
        if (
            update.get("refers_to")
            and graph["roots"].get(update["refers_to"], update["refers_to"]) not in known_updates
        ):
            state["issues"].append("unknown_update_reference")
            continue
        if kind in {"create_plan", "set_quantity"}:
            if kind == "create_plan" and state["updates"][:-1]:
                state["issues"].append("new_plan_requires_distinct_task_id")
                continue
            state["planned"], state["active"] = quantity, True
        elif kind == "add_item":
            state["planned"] = (
                state["planned"] + quantity
                if state["planned"] is not None and quantity is not None
                else None
            )
            state["active"] = True
        elif kind == "cancel_item":
            state["active"], state["planned"] = False, 0.0
        elif kind == "replace_item":
            replacement = update.get("replacement_item")
            if not replacement:
                state["issues"].append("replacement_item_missing")
                continue
            state["active"], state["planned"] = False, 0.0
            replacement = normalize(replacement, target.normalization)
            new_key = "::".join((update["owner"], update["task_id"], replacement, update["unit"]))
            if new_key in states and states[new_key]["active"]:
                states[new_key]["issues"].append("replacement_quantity_conflict")
            else:
                states[new_key] = {
                    **state,
                    "member": replacement,
                    "active": True,
                    "planned": update.get("replacement_quantity"),
                    "completed": 0.0,
                    "issues": [],
                    "updates": [update["update_id"]],
                }
        elif kind == "complete_item":
            if not state["active"]:
                state["issues"].append("completion_without_active_plan")
                continue
            state["completed"] = (
                state["completed"] + quantity
                if state["completed"] is not None and quantity is not None
                else None
            )
            state["completion_seen"] = True
            completion_records[update["update_id"]] = (key, quantity)
        elif kind == "undo_completion":
            ref = graph["roots"].get(update.get("refers_to"), update.get("refers_to"))
            original = completion_records.pop(ref, None)
            if not original or original[0] != key:
                state["issues"].append("undo_completion_reference_missing")
            elif state["completed"] is not None and original[1] is not None:
                state["completed"] = max(0, state["completed"] - original[1])
            else:
                state["completed"] = None
    for state in states.values():
        known = state["planned"] is not None and state["completed"] is not None
        state["remaining"] = (
            max(state["planned"] - state["completed"], 0) if known and not state["issues"] else None
        )
        if not state["active"] and not state["issues"]:
            state["remaining"] = 0
        if state["remaining"] is None:
            state["issues"].append("remaining_quantity_unknown")
        # Quantity uncertainty does not erase the existence of a planned/completed member.
        state["member_conflicted"] = bool(set(state["issues"]) - {"remaining_quantity_unknown"})
        issues.extend(state["issues"])
    closed = coverage_closed and not issues
    if not coverage_closed:
        issues.append("history_coverage_missing")
    return {
        "items": states,
        "closed_under_policy": closed,
        "membership_closed": coverage_closed and not (set(issues) - {"remaining_quantity_unknown"}),
        "issues": sorted(set(issues)),
        "evidence_refs": sorted({r for s in states.values() for r in s["evidence_refs"]}),
    }
