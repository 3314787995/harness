"""Deterministic three-valued set operations and conservative inventory bounds."""

from __future__ import annotations

import itertools
from typing import Any

from qwen3vl_agent.r4.history import reduce_tasks
from qwen3vl_agent.r4.ledger import clique_lower, normalize
from qwen3vl_agent.r4.planning import set_coverage
from qwen3vl_agent.r4.types import InventorySpec


def membership_state(present: bool, uncertain: bool, coverage: bool) -> str:
    if present:
        return "observed_present"
    if uncertain:
        return "unreadable_or_occluded"
    return "checked_not_observed" if coverage else "coverage_missing"


def _set_state(target: Any, ledger: Any, coverage: bool, graph: dict[str, Any]) -> dict[str, Any]:
    raw = ledger.state["observations"]
    records = [
        v
        for v in raw.values()
        if v["set_id"] == target.set_id
        and v["population_status"] != "excluded"
        and "context_only_member" not in v["issues"]
    ]
    confirmed, possible, names, evidence, issues = set(), set(), {}, set(), []
    member_groups: dict[str, set[str]] = {}
    candidate_states = {}
    for record in records:
        if record["predicate_status"] == "refuted":
            continue
        value = normalize(record["value"], target.normalization)
        key = (
            graph["roots"][record["observation_id"]]
            if target.namespace == "physical_instance"
            else value
        )
        names[key] = value
        member_groups.setdefault(key, set()).add(record["category"])
        possible.add(key)
        if record["predicate_status"] == "satisfied":
            confirmed.add(key)
        evidence.update(record["evidence_refs"])
        issues.extend(record["issues"])
    lower = (
        clique_lower(sorted(confirmed), graph["different"])
        if target.namespace == "physical_instance"
        else len(confirmed)
    )
    upper = len(possible)
    relevant_conflict = any(
        r["relation_id"] in graph["conflicts"]
        and (
            raw[r["left"]]["set_id"] == target.set_id or raw[r["right"]]["set_id"] == target.set_id
        )
        for r in ledger.state["relations"]
    )
    if relevant_conflict:
        issues.append("identity_conflict")
    if target.namespace == "physical_instance" and lower != upper:
        issues.append("identity_or_predicate_unresolved")
    unreadable = bool(possible - confirmed)
    for candidate in target.candidates:
        name = normalize(candidate, target.normalization)
        matched = [
            r
            for r in records
            if normalize(r["value"], target.normalization) == name
            or name in [normalize(v, target.normalization) for v in r["candidate_values"]]
        ]
        present = any(
            r["predicate_status"] == "satisfied"
            and normalize(r["value"], target.normalization) == name
            for r in matched
        )
        uncertain = any(r["predicate_status"] == "unknown" for r in matched) or unreadable
        candidate_states[candidate] = membership_state(present, uncertain, coverage)
    exact = coverage and lower == upper and not unreadable and not relevant_conflict
    return {
        "namespace": target.namespace,
        "confirmed_members": sorted(confirmed),
        "possible_members": sorted(possible),
        "member_names": names,
        "member_groups": {k: sorted(v) for k, v in member_groups.items()},
        "observed_groups": len(possible),
        "observed_count_bounds": [lower, upper],
        "population_count_bounds": [lower, upper if coverage and not unreadable else None],
        "exact_count": lower if exact else None,
        "coverage_closed": coverage,
        "closed_under_policy": exact,
        "candidate_states": candidate_states,
        "identity_conflict": relevant_conflict,
        "issues": sorted(set(issues)),
        "normalization": target.normalization,
        "evidence_refs": sorted(evidence),
    }


def _membership(
    key: str, state: dict[str, Any], graph: dict[str, Any] | None = None
) -> bool | None:
    if key in state["confirmed_members"]:
        return True
    if key in state["possible_members"] or not state["coverage_closed"]:
        return None
    if state["namespace"] == "physical_instance" and any(
        tuple(sorted((key, member))) not in (graph or {}).get("different", set())
        for member in state["possible_members"]
    ):
        return None
    return False


def set_operation(
    op: Any, states: dict[str, Any], graph: dict[str, Any] | None = None
) -> dict[str, Any]:
    selected = [states[key] for key in op.inputs]
    first = selected[0]
    result: dict[str, Any] = {
        "operation_id": op.operation_id,
        "op": op.op,
        "value": None,
        "closed_under_policy": False,
        "issues": [],
    }
    evidence = sorted({r for s in selected for r in s["evidence_refs"]})
    result["evidence_refs"] = evidence
    if op.op == "count_unique":
        result.update(
            value=first["exact_count"],
            bounds=first["population_count_bounds"],
            observed_bounds=first["observed_count_bounds"],
            closed_under_policy=first["closed_under_policy"],
        )
    elif op.op in {"membership", "missing_members"}:
        statuses = dict(first["candidate_states"])
        for candidate in op.candidates:
            if candidate not in statuses:
                present = any(
                    normalize(candidate, first.get("normalization", {})) == value
                    for key, value in first["member_names"].items()
                    if key in first["confirmed_members"]
                )
                statuses[candidate] = membership_state(
                    present,
                    bool(set(first["possible_members"]) - set(first["confirmed_members"])),
                    first["coverage_closed"],
                )
        result["candidate_states"] = statuses
        closed = bool(statuses) and all(
            v in {"observed_present", "checked_not_observed"} for v in statuses.values()
        )
        result["closed_under_policy"] = closed
        if op.op == "missing_members":
            result["value"] = (
                [k for k, v in statuses.items() if v == "checked_not_observed"] if closed else None
            )
        else:
            result["value"] = (
                {k: v == "observed_present" for k, v in statuses.items()} if closed else None
            )
            if len(statuses) == 1 and next(iter(statuses.values())) == "observed_present":
                result.update(value={next(iter(statuses)): True}, closed_under_policy=True)
    elif op.op in {"union", "intersection", "difference", "list_members"}:
        universe = sorted({k for s in selected for k in s["possible_members"]})
        confirmed, possible = [], []
        for key in universe:
            flags = [_membership(key, s, graph) for s in selected]
            if op.op in {"union", "list_members"}:
                value = True if True in flags else None if None in flags else False
            elif op.op == "intersection":
                value = False if False in flags else None if None in flags else True
            else:
                a, b = flags
                value = (
                    False if a is False or b is True else True if a is True and b is False else None
                )
            if value is not False:
                possible.append(key)
            if value is True:
                confirmed.append(key)
        names = {k: v for s in selected for k, v in s["member_names"].items()}
        # UNKNOWN physical identities between sets also leave set subtraction/intersection open.
        closed = all(s["closed_under_policy"] for s in selected) and confirmed == possible
        if first["namespace"] == "physical_instance" and len(selected) > 1:
            closed = (
                closed
                and bool(graph is not None and not graph["conflicts"])
                and all(
                    tuple(sorted(pair)) in (graph or {}).get("different", set())
                    for pair in itertools.combinations(universe, 2)
                )
            )
        result.update(
            value=[names[k] for k in confirmed] if closed else None,
            confirmed_members=[names[k] for k in confirmed],
            possible_members=[names[k] for k in possible],
            closed_under_policy=closed,
        )
    elif op.op in {"group_count", "argmax_count", "argmin_count", "compare_count"}:
        groups = {}
        if len(selected) > 1:
            groups = {
                key: state["population_count_bounds"] for key, state in zip(op.inputs, selected)
            }
        else:
            groups = first.get("groupings", {}).get(op.group_by, first.get("group_bounds", {}))
        result["group_bounds"] = groups
        all_covered = all(
            s["coverage_closed"] and not s.get("identity_conflict", False) for s in selected
        )
        if len(selected) == 1 and op.group_by in first.get("unresolved_groupings", []):
            all_covered = False
        if op.op == "group_count":
            closed = all_covered and bool(groups) and all(lo == hi for lo, hi in groups.values())
            result.update(
                value={k: v[0] for k, v in groups.items()} if closed else None,
                closed_under_policy=closed,
            )
        elif op.op == "compare_count":
            (la, ua), (lb, ub) = groups.values()
            value = (
                (la > ub if ub is not None else None)
                if op.compare == "greater"
                else (ua < lb if ua is not None else None)
                if op.compare == "less"
                else (la == ua == lb == ub if ua is not None and ub is not None else None)
            )
            if op.compare == "greater" and ua is not None and ua <= lb:
                value = False
            if op.compare == "less" and ub is not None and la >= ub:
                value = False
            if op.compare == "equal" and not (
                ua is not None and ub is not None and la == ua and lb == ub
            ):
                value = (
                    False if (ua is not None and ua < lb) or (ub is not None and ub < la) else None
                )
            result.update(
                value=value if all_covered else None,
                closed_under_policy=all_covered and value is not None,
            )
        else:
            winners = []
            for name, (lo, hi) in groups.items():
                other = [v for k, v in groups.items() if k != name]
                if op.op == "argmax_count":
                    wins = all(u is not None and lo > u for _, u in other)
                else:
                    wins = hi is not None and all(hi < l for l, _ in other)
                if wins:
                    winners.append(name)
            exact = bool(groups) and all(lo == hi for lo, hi in groups.values())
            if exact:
                best = (max if op.op == "argmax_count" else min)(v[0] for v in groups.values())
                winners = [k for k, v in groups.items() if v[0] == best]
            result.update(
                value=winners if all_covered and winners else None,
                closed_under_policy=bool(all_covered and winners),
            )
    elif op.op == "remaining_quantity":
        task = first["task_state"]
        quantities = {
            k: v["remaining"] for k, v in task["items"].items() if v["active"] or v["issues"]
        }
        result.update(
            value=quantities if task["closed_under_policy"] else None,
            partial_quantities=quantities,
            closed_under_policy=task["closed_under_policy"],
            issues=task["issues"],
        )
    return result


def _task_projection(target: Any, task: dict[str, Any], coverage: bool) -> dict[str, Any]:
    projection = target.task_projection
    if projection == "auto":
        projection = "completed" if target.evidence_relation == "completed" else "active_plan"
    confirmed, possible = [], []
    closed = task["membership_closed"] if projection != "remaining" else task["closed_under_policy"]
    for key, item in task["items"].items():
        if item["member_conflicted"]:
            possible.append(key)
            continue
        if projection == "active_plan":
            present = item["active"] and (item["planned"] is None or item["planned"] > 0)
        elif projection == "completed":
            present = (
                item["completed"] > 0
                if item["completed"] is not None
                else item.get("completion_seen", False)
            )
        else:
            present = item["active"] and item["remaining"] is not None and item["remaining"] > 0
            if item["active"] and item["remaining"] is None:
                possible.append(key)
        if present:
            possible.append(key)
            # Open history can contain a later cancellation/undo/completion.
            if closed:
                confirmed.append(key)
    candidates = {
        name: membership_state(
            any(
                task["items"][k]["member"] == normalize(name, target.normalization)
                for k in confirmed
            ),
            not closed,
            coverage,
        )
        for name in target.candidates
    }
    return {
        "task_state": task,
        "task_projection": projection,
        "confirmed_members": confirmed,
        "possible_members": sorted(set(possible)),
        "member_names": {k: v["member"] for k, v in task["items"].items()},
        "observed_groups": len(possible),
        "observed_count_bounds": [len(confirmed), len(possible)],
        "population_count_bounds": [len(confirmed), len(possible) if closed else None],
        "exact_count": len(confirmed) if closed else None,
        "closed_under_policy": closed,
        "coverage_closed": coverage and closed,
        "candidate_states": candidates,
        "evidence_refs": task["evidence_refs"],
        "issues": task["issues"],
    }


def _group_bounds(
    target: Any, state: dict[str, Any], ledger: Any, graph: dict[str, Any], group_by: str
) -> tuple[dict[str, Any], bool]:
    members = {key: set() for key in state["possible_members"]}
    for record in ledger.state["observations"].values():
        if record["set_id"] != target.set_id:
            continue
        key = (
            graph["roots"].get(record["observation_id"])
            if target.namespace == "physical_instance"
            else normalize(record["value"], target.normalization)
        )
        if key not in members:
            continue
        if group_by in {"category", "value"}:
            value = record[group_by]
        else:
            value = record["attributes"].get(group_by.removeprefix("attributes."))
            if isinstance(value, dict):
                value = value.get("value")
        if isinstance(value, (str, int, float)) and str(value):
            members[key].add(str(value))
    unknown = {key for key, values in members.items() if len(values) != 1}
    categories = set(target.candidates) | {v for values in members.values() for v in values}
    groups = {}
    for category in sorted(categories):
        possibles = {k for k, values in members.items() if category in values} | unknown
        confirms = sorted(
            k for k in possibles if k in state["confirmed_members"] and members[k] == {category}
        )
        lower = (
            clique_lower(confirms, graph["different"])
            if target.namespace == "physical_instance"
            else len(confirms)
        )
        groups[category] = [
            lower,
            len(possibles) if state["population_count_bounds"][1] is not None else None,
        ]
    return groups, bool(unknown)


def set_reduce(
    spec: InventorySpec,
    ledger: Any,
    tiles: dict[str, Any],
    *,
    global_issues: list[str] | None = None,
    multi_source: bool = False,
    query_time: float | None = None,
    completion_predicate: str | None = None,
) -> dict[str, Any]:
    global_issues = global_issues or []
    graph, states = ledger.graph(), {}
    for target in spec.sets:
        coverage = set_coverage(target.set_id, tiles) and not global_issues and not spec.unresolved
        state = _set_state(target, ledger, coverage, graph)
        if target.namespace == "task_item":
            # Equivalent lifecycle scopes can supply P and D separately; one task history
            # is reconstructed before projecting active-plan/completed/remaining members.
            compatible = {
                s.set_id
                for s in spec.sets
                if s.namespace == "task_item"
                and all(
                    getattr(s, k) == getattr(target, k)
                    for k in ("scope", "owner", "task_id", "normalization", "target")
                )
            }
            coverage = coverage and all(set_coverage(k, tiles) for k in compatible)
            task = reduce_tasks(
                ledger,
                target,
                coverage_closed=coverage,
                multi_source=multi_source,
                query_time=query_time,
                completion_predicate=completion_predicate,
                set_ids=compatible,
            )
            if state["possible_members"] and not any(
                u["set_id"] in compatible for u in ledger.state["task_updates"].values()
            ):
                task["issues"].append("task_observation_without_update")
                task["closed_under_policy"] = task["membership_closed"] = False
                task["evidence_refs"] = sorted(
                    set(task["evidence_refs"]) | set(state["evidence_refs"])
                )
            state.update(_task_projection(target, task, coverage))
        state["groupings"], state["unresolved_groupings"] = {}, []
        for group_by in {
            o.group_by
            for o in spec.operations
            if len(o.inputs) == 1
            and target.set_id in o.inputs
            and o.op in {"group_count", "argmax_count", "argmin_count"}
        }:
            groups, unknown = _group_bounds(target, state, ledger, graph, group_by)
            state["groupings"][group_by] = groups
            if unknown:
                state["unresolved_groupings"].append(group_by)
                state["issues"].append("grouping_unresolved:" + group_by)
        state["group_bounds"] = state["groupings"].get("category", {})
        states[target.set_id] = state
    results = [set_operation(op, states, graph) for op in spec.operations]
    issues = sorted(
        {*global_issues, *spec.unresolved, *(v for s in states.values() for v in s["issues"])}
    )
    return {
        "sets": states,
        "results": results,
        "contract_unresolved": bool(global_issues or spec.unresolved),
        "closed_under_policy": bool(results) and all(r["closed_under_policy"] for r in results),
        "evidence_refs": sorted({r for s in states.values() for r in s["evidence_refs"]}),
        "issues": issues,
    }
