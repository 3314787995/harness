"""Bounded typed data operations. No eval, code generation, imports or effects in the DSL."""

from __future__ import annotations

import itertools
import math
from copy import deepcopy

from .state import Cell
from .types import ProtocolError, finite

OPS = {
    "COUNT_EVENTS",
    "READ_FACT",
    "SET",
    "ADD",
    "SWAP",
    "TABLE_LOOKUP",
    "RULE_TRANSITION",
    "SORT",
    "SELECT",
    "COMPARE",
}
PARAMS = {
    "COUNT_EVENTS": {"entity_id", "predicate", "time_window"},
    "READ_FACT": set(),
    "SET": {"pack", "keys"},
    "ADD": set(),
    "SWAP": set(),
    "TABLE_LOOKUP": set(),
    "RULE_TRANSITION": set(),
    "SORT": {"keys", "descending"},
    "SELECT": {"top_k"},
    "COMPARE": {"relation"},
}


def bounds(value):
    if finite(value):
        return (float(value), float(value))
    if (
        isinstance(value, dict)
        and set(value) == {"lo", "hi"}
        and all(finite(x) for x in value.values())
        and value["lo"] <= value["hi"]
    ):
        return (float(value["lo"]), float(value["hi"]))
    return None


def number(lo, hi):
    if not all(math.isfinite(v) for v in (lo, hi)):
        return None
    return lo if lo == hi else {"lo": lo, "hi": hi}


def compare(value, relation, expected):
    if value is None or expected is None:
        return None
    a, b = bounds(value), bounds(expected)
    if a is not None or b is not None:
        if a is None or b is None:
            return None
        if relation == "eq":
            return (
                True
                if a[0] == a[1] == b[0] == b[1]
                else (False if a[1] < b[0] or b[1] < a[0] else None)
            )
        if relation == "ne":
            result = compare(value, "eq", expected)
            return None if result is None else not result
        if relation == "gt":
            return compare(expected, "lt", value)
        if relation == "ge":
            return compare(expected, "le", value)
        if relation == "lt":
            return True if a[1] < b[0] else (False if a[0] >= b[1] else None)
        if relation == "le":
            return True if a[1] <= b[0] else (False if a[0] > b[1] else None)
        return None
    if relation == "eq":
        return value == expected
    if relation == "ne":
        return value != expected
    if relation == "contains" and isinstance(value, (list, dict, str)):
        return expected in value
    return None


def resolve(expr, world, sources):
    if not isinstance(expr, dict):
        raise ProtocolError("DSL argument requires ref or source-backed literal")
    if set(expr) == {"ref"}:
        return world.get(expr["ref"])
    if set(expr) == {"literal", "source"} and expr["source"] in sources:
        source = sources[expr["source"]]
        # Question constants must match their compiled value, not just borrow its ID.
        if "value" in source and source["value"] != expr["literal"]:
            raise ProtocolError("literal differs from its stipulated source")
        return Cell(
            "literal:" + expr["source"],
            deepcopy(expr["literal"]),
            "stipulated",
            [expr["source"]],
            grounded=source.get("basis", "stipulated") == "stipulated",
        )
    raise ProtocolError("unbacked DSL literal")


def count_events(events):
    """Merge overlapping instances of the same entity/predicate; preserve disjoint repetitions."""
    if not isinstance(events, list):
        return None
    parsed = []
    for event in events:
        if (
            not isinstance(event, dict)
            or not {"entity_id", "predicate", "start", "end", "occurrence_id"} <= event.keys()
        ):
            return None
        if not finite(event["start"]) or not finite(event["end"]) or event["start"] > event["end"]:
            return None
        parsed.append(event)
    groups = []
    for event in sorted(parsed, key=lambda e: (e["entity_id"], e["predicate"], e["start"])):
        if any(
            g["entity_id"] == event["entity_id"]
            and g["predicate"] == event["predicate"]
            and g["occurrence_id"] == event["occurrence_id"]
            and (event["start"] > g["end"] or event["end"] < g["start"])
            for g in groups
        ):
            return None  # Reused local IDs without temporal overlap are ambiguous.
        match = next(
            (
                g
                for g in groups
                if g["entity_id"] == event["entity_id"]
                and g["predicate"] == event["predicate"]
                and (
                    g["occurrence_id"] == event["occurrence_id"]
                    or event["start"] <= g["end"]
                    and event["end"] >= g["start"]
                )
            ),
            None,
        )
        if match:
            match["start"], match["end"] = (
                min(match["start"], event["start"]),
                max(match["end"], event["end"]),
            )
        else:
            groups.append(deepcopy(event))
    return len(groups)


def execute(steps, world, sources, rules, max_steps=64):
    if len(steps) > max_steps:
        raise ProtocolError("program step limit exceeded")
    for step in steps:
        op = step["op"]
        if op not in OPS or step["rule_id"] not in rules:
            raise ProtocolError("unknown operator or mechanism source")
        args = [resolve(a, world, sources) for a in step["args"]]
        rule = rules[step["rule_id"]]
        out, params = step["out"], step["params"]
        if set(params) - PARAMS[op]:
            raise ProtocolError("unknown operator parameters")
        time_window = (
            resolve(params["time_window"], world, sources) if "time_window" in params else None
        )
        extra = [time_window] if time_window is not None else []
        dependencies = [a.key for a in args + extra if not a.key.startswith("literal:")]
        source_ids = sorted({s for a in args + extra for s in a.sources} | {step["rule_id"]})
        grounded = all(a.grounded for a in args + extra) and rule["basis"] == "stipulated"
        value, reason = None, "unknown_input"
        unit = args[0].unit if args else ""
        known = all(a.known for a in args + extra)
        if op == "SWAP":
            if len(args) != 2 or any(set(e) != {"ref"} for e in step["args"]):
                raise ProtocolError("SWAP requires two variable references")
            first, second = deepcopy(args)
            # Program swaps also read a single snapshot; invalidate once, commit both.
            world.invalidate({first.key, second.key})
            for dest, source in ((first, second), (second, first)):
                copied = Cell(
                    dest.key,
                    source.value if source.known else None,
                    "derived",
                    source_ids,
                    ["snapshot:" + source.key],
                    grounded=grounded,
                    unit=source.unit,
                    reason="simultaneous_swap",
                )
                if dest.key in world.invariants:
                    raise ProtocolError("swap violates invariant")
                world.cells[dest.key] = copied
            world.execution.append({"id": step["id"], "op": op, "outputs": [first.key, second.key]})
            continue
        if known:
            vals = [a.value for a in args]
            reason = "type_or_precondition_unresolved"
            if op == "SET" and params.get("pack") == "list":
                value = deepcopy(vals)
            elif op == "SET" and params.get("pack") == "map":
                keys = params.get("keys", [])
                if (
                    len(keys) != len(vals)
                    or len(set(keys)) != len(keys)
                    or not all(isinstance(k, str) for k in keys)
                ):
                    raise ProtocolError("SET map requires one unique key per sourced argument")
                value = dict(zip(keys, deepcopy(vals)))
            elif op in {"READ_FACT", "SET"} and len(args) == 1:
                value = deepcopy(vals[0])
            elif op == "ADD" and len(args) >= 2 and all(bounds(v) is not None for v in vals):
                if len({a.unit for a in args if a.unit}) <= 1:
                    value = number(sum(bounds(v)[0] for v in vals), sum(bounds(v)[1] for v in vals))
            elif op == "COUNT_EVENTS" and args and all(isinstance(v, list) for v in vals):
                events = [event for v in vals for event in v]
                if any(not isinstance(e, dict) for e in events):
                    raise ProtocolError("COUNT_EVENTS expects event records")
                for name in ("entity_id", "predicate"):
                    if name in params:
                        events = [e for e in events if e.get(name) == params[name]]
                if time_window is not None:
                    interval = time_window.value
                    if (
                        not isinstance(interval, list)
                        or len(interval) != 2
                        or not all(finite(v) for v in interval)
                        or interval[0] > interval[1]
                    ):
                        raise ProtocolError("COUNT_EVENTS time_window must resolve to [start,end]")
                    # A partial occurrence at an edge cannot silently count as a complete event.
                    boundary = any(
                        e.get("start", -math.inf) < interval[0] < e.get("end", math.inf)
                        or e.get("start", -math.inf) < interval[1] < e.get("end", math.inf)
                        for e in events
                    )
                    events = [
                        e
                        for e in events
                        if interval[0]
                        <= e.get("start", -math.inf)
                        <= e.get("end", math.inf)
                        <= interval[1]
                    ]
                else:
                    boundary = False
                count = count_events(events)
                value = count if all(a.complete for a in args) and not boundary else None
                unit = "events"
                reason = (
                    "incomplete_event_coverage" if value is None else "counted_unique_occurrences"
                )
            elif op == "TABLE_LOOKUP" and len(args) == 2 and isinstance(vals[0], dict):
                value = deepcopy(vals[0].get(str(vals[1])))
            elif op == "RULE_TRANSITION" and len(args) == 3 and isinstance(vals[0], dict):
                transitions = vals[0].get(str(vals[1]), {})
                value = (
                    deepcopy(transitions.get(str(vals[2])))
                    if isinstance(transitions, dict)
                    else None
                )
            elif op == "COMPARE" and len(args) == 2:
                value = (
                    compare(vals[0], params.get("relation"), vals[1])
                    if len({a.unit for a in args if a.unit}) <= 1
                    else None
                )
                unit = ""
            elif op == "SORT" and args:
                keys = params.get("keys")
                if keys is not None:
                    if len(keys) != len(vals) or len(set(keys)) != len(keys):
                        raise ProtocolError("SORT keys must match argument count uniquely")
                    entries = dict(zip(keys, vals))
                else:
                    entries = vals[0] if len(vals) == 1 and isinstance(vals[0], dict) else {}
                if (
                    entries
                    and all(bounds(v) is not None for v in entries.values())
                    and len({a.unit for a in args if a.unit}) <= 1
                ):
                    ordered = sorted(
                        entries,
                        key=lambda k: bounds(entries[k])[0],
                        reverse=params.get("descending", True),
                    )
                    uncertain = any(
                        compare(
                            entries[a], "gt" if params.get("descending", True) else "lt", entries[b]
                        )
                        is not True
                        for a, b in itertools.pairwise(ordered)
                    )
                    value = {
                        "order": None if uncertain else ordered,
                        "intervals": deepcopy(entries),
                        "ambiguous": uncertain,
                    }
                    unit = "ordering"
            elif op == "SELECT" and len(args) == 2:
                collection, key = vals
                if isinstance(collection, dict) and "ambiguous" in collection:
                    collection = collection["order"]
                if isinstance(collection, list) and finite(key) and key == int(key):
                    key = int(key)
                    if params.get("top_k") and 0 < key <= len(collection):
                        value = deepcopy(collection[:key])
                    elif not params.get("top_k") and 0 <= key < len(collection):
                        value = deepcopy(collection[key])
                elif isinstance(collection, dict):
                    value = deepcopy(collection.get(str(key)))
        cell = Cell(
            out,
            value,
            "derived" if value is not None else "unknown",
            source_ids,
            dependencies,
            grounded=grounded,
            unit=unit,
            complete=all(a.complete for a in args),
            reason="executed" if value is not None else reason,
        )
        world.write(cell)
        world.execution.append(
            {
                "id": step["id"],
                "op": op,
                "output": out,
                "status": "entailed_by_execution"
                if cell.known and grounded
                else "supported"
                if cell.known
                else "unknown",
            }
        )
