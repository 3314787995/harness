"""Typed set algebra and bounded uncertainty; no answer generation model."""
from __future__ import annotations
import itertools
import json
import re
from dataclasses import dataclass, field

from .history import reduce_tasks
from .ledger import clique_lower, normalize
from .reduce import _task_projection
from .types import timestamp


@dataclass
class View:
    namespace: str
    unit: str
    definite: set = field(default_factory=set)
    possible: set = field(default_factory=set)
    closed: bool = False
    names: dict = field(default_factory=dict)
    cards: list = field(default_factory=list)
    unknown_category_members: int = 0
    candidates: dict = field(default_factory=dict)
    task: dict | None = None
    snapshots: list = field(default_factory=list)
    snapshot_coverage: bool = False


def key_for(card, graph):
    if card["namespace"] == "physical_instance":
        return graph["roots"].get(card["candidate_id"], card["candidate_id"])
    if card.get("query_value") is None:
        return "?" + card["candidate_id"]
    value = card["query_value"]
    return value.casefold() if card["namespace"] == "semantic_category" and card.get("equivalence") != "combination" else value


def window_complete(window):
    children = window.get("children", [])
    return window.get("status") == "complete" and not children


def coverage_for(set_id, windows):
    rows = [w for w in windows.values() if set_id in w.get("set_ids", []) and not w.get("children") and not w.get("scope_only")]
    return bool(rows) and all(window_complete(w) for w in rows)


def bounds(view, graph):
    if view.namespace == "physical_instance":
        lo = clique_lower(sorted(view.definite), graph["different"])
    else:
        lo = len(view.definite)
        if view.unknown_category_members:
            lo = max(1, lo)
    return [lo, len(view.possible) if view.closed else None]


def mapping_conflicts(store):
    """A recognized class with incompatible query names needs review, not two votes."""
    groups = {}
    for card in store.cards.values():
        if card["namespace"] == "semantic_category" and card["membership"] != "excluded" and card.get("equivalence") != "combination":
            groups.setdefault((card["set_id"], card["actual_class"].casefold()), []).append(card)
    return {c["candidate_id"] for cards in groups.values()
            if len({c["query_value"].casefold() for c in cards if c.get("query_value")}) > 1 for c in cards}


def build_views(spec, store, windows, request, sources, *, scope_issues=()):
    graph, views = store.graph(), {}
    conflicts = mapping_conflicts(store)
    for s in spec.sets:
        v = View(s.namespace, s.count_unit or s.namespace)
        v.closed = coverage_for(s.set_id, windows) and not scope_issues
        v.cards = [c for c in store.cards.values() if c["set_id"] == s.set_id and c["membership"] != "excluded"]
        active_windows={w["tile_id"] for w in windows.values() if s.set_id in w.get("set_ids",[]) and not w.get("children")}
        v.snapshots=[r for r in store.state["snapshots"].values() if r["set_id"]==s.set_id and r["window_id"] in active_windows]
        required={ref for w in windows.values() if w["tile_id"] in active_windows for ref in w.get("snapshot_required",{}).get(s.set_id,[])}
        checked={ref for row in v.snapshots for ref,frame in row["frames"].items() if frame["full_frame"]}
        v.snapshot_coverage=v.closed and bool(required) and required <= checked
        for c in v.cards:
            k = "?" + c["candidate_id"] if c["candidate_id"] in conflicts else key_for(c, graph)
            v.possible.add(k)
            v.names[k] = c.get("query_value") or c["raw_value"]
            if c["membership"] == "accepted":
                if not k.startswith("?"):
                    v.definite.add(k)
                else:
                    v.unknown_category_members += 1
        if s.namespace == "task_item":
            task = reduce_tasks(store.history, s, coverage_closed=v.closed and request.benchmark_policy.get("history_complete", False),
                multi_source=len(sources) > 1, query_time=timestamp(request.query_time) if request.query_time is not None else None,
                completion_predicate=request.benchmark_policy.get("completion_predicate"),
                set_ids={t.set_id for t in spec.sets if t.namespace == "task_item" and t.owner == s.owner and t.task_id == s.task_id})
            p = _task_projection(s, task, v.closed)
            v.definite, v.possible = set(p["confirmed_members"]), set(p["possible_members"])
            v.names, v.closed, v.task = p["member_names"], p["closed_under_policy"], task
        if s.candidates and s.namespace == "task_item":
            v.candidates = {c: "present" if normalize(c, s.normalization) in v.definite else
                           "absent" if v.closed and normalize(c, s.normalization) not in v.possible else "unknown" for c in s.candidates}
        elif s.candidates:
            relevant = [w for w in windows.values() if s.set_id in w.get("set_ids", []) and not w.get("children") and not w.get("scope_only")]
            for candidate in s.candidates:
                rows = [r for r in store.state["checks"].values() if r["set"] == s.set_id and r["candidate"] == candidate]
                seen = any(r["state"] == "seen" and not r.get("needs_review") for r in rows)
                done = bool(relevant) and all(any(r["window_id"] == w["tile_id"] and r["state"] == "not_seen" for r in rows) and window_complete(w) for w in relevant)
                v.candidates[candidate] = "present" if seen else "absent" if done and not scope_issues else "unknown"
        views[s.set_id] = v
    return views, graph


def set_view(op, operands, graph):
    a = operands[0]
    v = View(a.namespace, a.unit, closed=all(x.closed for x in operands),
             names={k: n for x in operands for k, n in x.names.items()}, cards=[c for x in operands for c in x.cards])
    v.snapshots=[row for x in operands for row in x.snapshots]
    v.snapshot_coverage=all(x.snapshot_coverage for x in operands)
    if op == "union":
        v.definite = set.union(*(x.definite for x in operands))
        v.possible = set.union(*(x.possible for x in operands))
    elif op == "intersection":
        v.definite = set.intersection(*(x.definite for x in operands))
        if a.namespace == "physical_instance":
            v.possible = set.union(*(x.possible for x in operands))
            # A missing operand can contain future/unknown matches. Only a closed empty set annihilates.
            if any(x.closed and not x.possible for x in operands):
                v.possible.clear()
        else:
            v.possible = set.intersection(*(x.possible for x in operands))
            if any(any(k.startswith("?") for k in x.possible) for x in operands):
                v.possible = set.union(*(x.possible for x in operands))
    elif op == "difference":
        b = operands[1]
        v.definite, v.possible = a.definite - b.possible, a.possible - b.definite
        if a.namespace == "physical_instance":
            v.definite = {k for k in v.definite if b.closed and all(tuple(sorted((k, j))) in graph["different"] for j in b.possible)}
        elif not b.closed or any(k.startswith("?") for k in b.possible):
            v.definite.clear()
    return v


def count_result(view, graph):
    lo, hi = bounds(view, graph)
    return {"value": lo if hi == lo else None, "bounds": [lo, hi], "supported": hi == lo,
            "possible_values": None, "view": view}


def group_counts(view, group_by, graph, candidates=()):
    associations = {}
    for card in view.cards:
        key = key_for(card, graph)
        if key not in view.possible:
            continue
        val = (card.get("query_value") or card.get("actual_class")) if group_by in {"category", "value"} else card["attributes"].get(group_by.removeprefix("attributes."))
        associations.setdefault(key, set())
        if val is not None:
            associations[key].add(str(val))
    groups = set(candidates) | {v for values in associations.values() for v in values}
    unknown = {k for k in view.possible if len(associations.get(k, set())) != 1}
    result = {}
    for g in sorted(groups):
        possible = {k for k, vals in associations.items() if g in vals} | unknown
        definite = {k for k in view.definite if associations.get(k) == {g}}
        group = View(view.namespace, view.unit, definite, possible, view.closed)
        result[g] = bounds(group, graph)
    return result, len(unknown)


def evaluate_ops(spec, views, graph, *, exact_world=False):
    values, results = dict(views), []
    for op in spec.operations:
        args = [values[k] for k in op.inputs]
        row = {"operation_id": op.operation_id, "op": op.op, "value": None, "supported": False}
        if op.op in {"union", "intersection", "difference"}:
            if exact_world:
                a = args[0]
                members = (set.union(*(x.definite for x in args)) if op.op == "union" else
                           set.intersection(*(x.definite for x in args)) if op.op == "intersection" else args[0].definite - args[1].definite)
                view = View(a.namespace, a.unit, members, members, all(x.closed for x in args),
                            {k: n for x in args for k, n in x.names.items()}, [c for x in args for c in x.cards])
            else:
                view = set_view(op.op, args, graph)
            values[op.operation_id] = view
            row["bounds"] = bounds(view, graph)
            if view.closed and view.definite == view.possible and (view.namespace != "physical_instance" or exact_world or row["bounds"][0] == row["bounds"][1]):
                row.update(value=sorted(view.names.get(k, k) for k in view.definite), supported=True)
        elif op.op == "count_unique":
            row.update(count_result(args[0], graph))
            values[op.operation_id] = row
        elif op.op == "ordered_counts":
            counts = [count_result(x, graph) if isinstance(x, View) else x for x in args]
            row.update(bounds=[x.get("bounds") for x in counts], supported=all(x["supported"] for x in counts))
            if row["supported"]:
                row["value"] = [x["value"] for x in counts]
            values[op.operation_id] = row
        elif op.op in {"membership", "missing_members"}:
            view = args[0]
            states = dict(view.candidates)
            for name in op.candidates:
                if name not in states:
                    key = name.casefold() if view.namespace == "semantic_category" else name
                    states[name] = "present" if key in view.definite else "absent" if view.closed and key not in view.possible else "unknown"
            row["candidate_states"] = states
            if op.op == "membership":
                row.update(value={k: v == "present" for k, v in states.items()} if states and "unknown" not in states.values() else None,
                           supported=bool(states) and "unknown" not in states.values())
            elif states and "unknown" not in states.values():
                row.update(value=[k for k, v in states.items() if v == "absent"], supported=True)
            values[op.operation_id] = row
        elif op.op in {"group_count", "argmax_count", "argmin_count", "compare_count"}:
            if len(args) == 1 and isinstance(args[0], dict) and "groups" in args[0]:
                groups, unknown = args[0]["groups"], False
            elif len(args) > 1:
                groups = {key: bounds(v, graph) if isinstance(v, View) else v["bounds"] for key, v in zip(op.inputs, args)}
                unknown = False
            else:
                groups, unknown = group_counts(args[0], op.group_by, graph, op.candidates)
                if unknown and not op.candidates:
                    groups["?unclassified"] = [0, unknown if args[0].closed else None]
                if not args[0].closed and not op.candidates:
                    groups["?uninspected_groups"] = [0, None]
            row["groups"] = groups
            exact = bool(groups) and all(lo == hi for lo, hi in groups.values())
            if op.op == "group_count":
                row.update(value={k: v[0] for k, v in groups.items()} if exact and not unknown else None,
                           supported=exact and not unknown)
            elif op.op == "compare_count":
                (la, ua), (lb, ub) = list(groups.values())[:2]
                value = None
                if op.compare == "greater":
                    value = True if ub is not None and la > ub else False if ua is not None and ua <= lb else None
                elif op.compare == "less":
                    value = True if ua is not None and ua < lb else False if ub is not None and la >= ub else None
                else:
                    value = True if la == ua == lb == ub else False if (ua is not None and ua < lb) or (ub is not None and ub < la) else None
                row.update(value=value, supported=value is not None)
            else:
                winners = []
                for k, (lo, hi) in groups.items():
                    if k in {"?unclassified", "?uninspected_groups"}:
                        continue
                    other = [b for n, b in groups.items() if n != k]
                    if all(u is not None and lo > u for l, u in other) if op.op == "argmax_count" else hi is not None and all(hi < l for l, u in other):
                        winners.append(k)
                if exact:
                    best = (max if op.op == "argmax_count" else min)(v[0] for v in groups.values())
                    winners = [k for k, v in groups.items() if v[0] == best]
                row.update(value=winners or None, supported=bool(winners))
            values[op.operation_id] = row
        elif op.op == "max_simultaneous_count":
            view, instances = args[0], {}
            for row_snapshot in view.snapshots:
                for ref,frame in row_snapshot["frames"].items():
                    for carrier,ids in frame["groups"].items():
                        bucket=(frame["source_frame_id"],carrier)
                        values_at_time={graph["roots"].get(k,k) for k in ids}
                        item=instances.setdefault(bucket,{"definite":set(),"possible":set()})
                        item["possible"] |= values_at_time & view.possible
                        item["definite"] |= values_at_time & view.definite
            for c in view.cards:
                key=key_for(c,graph)
                if key not in view.possible:
                    continue
                carrier = c["attributes"].get(op.group_by) if op.group_by not in {"category", "value"} else c["attributes"].get("carrier")
                if carrier is None:
                    continue
                for d in c["detections"]:
                    if not d.get("in_core"):
                        continue
                    bucket = (d["source_frame_id"], str(carrier))
                    instances.setdefault(bucket, {"definite": set(), "possible": set()})
                    instances[bucket]["possible"].add(key)
                    if key in view.definite:
                        instances[bucket]["definite"].add(key)
            bs = [bounds(View(view.namespace, view.unit, **item, closed=True), graph) for item in instances.values()]
            lo = max((b[0] for b in bs), default=0)
            hi = max((b[1] for b in bs), default=0) if view.snapshot_coverage else (len(view.possible) if view.closed else None)
            row.update(value=lo if lo == hi else None, bounds=[lo, hi], supported=lo == hi, per_instant=instances,
                       per_instant_inventory_complete=view.snapshot_coverage)
            values[op.operation_id] = row
        elif op.op == "remaining_quantity":
            task = args[0].task or {}
            row.update(value={k: v["remaining"] for k, v in task.get("items", {}).items() if v["active"] or v["issues"]}
                       if task.get("closed_under_policy") else None, supported=bool(task.get("closed_under_policy")))
            values[op.operation_id] = row
        elif op.op == "list_members":
            v = args[0]
            row.update(value=sorted(v.names.get(k, k) for k in v.definite) if v.closed and v.definite == v.possible else None,
                       supported=v.closed and v.definite == v.possible and (v.namespace != "physical_instance" or bounds(v, graph)[0] == bounds(v, graph)[1]))
            values[op.operation_id] = row
        results.append(row)
    return results


def physical_worlds(views, graph, max_nodes=8, limit=4096):
    physical = {k: v for k, v in views.items() if v.namespace == "physical_instance"}
    nodes = sorted(set().union(*(v.possible for v in physical.values()))) if physical else []
    if not nodes or len(nodes) > max_nodes or any(v.possible != v.definite for v in views.values() if v.namespace != "physical_instance"):
        return None
    optional = [(k, n) for k, v in physical.items() for n in v.possible - v.definite]
    if len(optional) > 12:
        return None
    worlds, blocks, over = [], [], False
    def visit(i):
        nonlocal over
        if over:
            return
        if i == len(nodes):
            root = {n: min(b) for b in blocks for n in b}
            for bits in itertools.product((False, True), repeat=len(optional)):
                if len(worlds) >= limit:
                    over = True
                    return
                selected = {p for p, bit in zip(optional, bits) if bit}
                world = dict(views)
                for k, v in physical.items():
                    chosen = {root[n] for n in v.definite | {n for name, n in selected if name == k}}
                    world[k] = View(v.namespace, v.unit, chosen, chosen, v.closed,
                                    {root[n]: v.names.get(n, n) for n in v.possible}, v.cards)
                all_roots = sorted(set(root.values()))
                wg = {**graph, "different": set(itertools.combinations(all_roots, 2))}
                worlds.append((world, wg))
            return
        n = nodes[i]
        for b in blocks:
            if all(tuple(sorted((n, m))) not in graph["different"] for m in b):
                b.append(n); visit(i + 1); b.pop()
        blocks.append([n]); visit(i + 1); blocks.pop()
    visit(0)
    return None if over else worlds


def serializable(value):
    if isinstance(value, View):
        return {"confirmed_members": sorted(value.definite), "possible_members": sorted(value.possible),
                "coverage_closed": value.closed, "member_names": value.names}
    if isinstance(value, set):
        return sorted(value)
    if isinstance(value, dict):
        return {str(k): serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [serializable(v) for v in value]
    return value


def reduce_inventory(spec, store, windows, request, sources, scope_issues=()):
    views, graph = build_views(spec, store, windows, request, sources, scope_issues=scope_issues)
    rows = evaluate_ops(spec, views, graph)
    worlds = physical_worlds(views, graph)
    if worlds is not None and all(v.closed for v in views.values()):
        all_rows = [evaluate_ops(spec, v, g, exact_world=True) for v, g in worlds]
        for i, row in enumerate(rows):
            # Physical simultaneous counts use source detections and are handled conservatively above.
            if row["op"] in {"max_simultaneous_count", "group_count", "argmax_count", "argmin_count"}:
                continue
            possible = {json.dumps(serializable(r[i]["value"]), ensure_ascii=False, sort_keys=True)
                        for r in all_rows if r[i]["supported"]}
            if len(possible) and all(r[i]["supported"] for r in all_rows):
                row["possible_values"] = [json.loads(p) for p in sorted(possible)]
                row["supported"] = len(possible) == 1
                row["value"] = row["possible_values"][0] if len(possible) == 1 else None
                if all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in row["possible_values"]):
                    row["bounds"] = [min(row["possible_values"]), max(row["possible_values"])]
    final = next(r for r in rows if r["operation_id"] == (spec.output_id or spec.operations[-1].operation_id))
    if final["op"] == "missing_members" and request.benchmark_policy.get("unique_missing") is True:
        states = final.get("candidate_states", {})
        unknown = [k for k, v in states.items() if v != "present"]
        if len(unknown) == 1 and len(states) > 1:
            final.update(value=unknown, supported=True, constraint_basis="explicit_unique_missing")
    conflicts = sorted(mapping_conflicts(store))
    return serializable({"results": rows, "final": final, "sets": {k: {**serializable(v), "bounds": bounds(v, graph),
            "candidate_states": v.candidates} for k, v in views.items()}, "identity": graph,
            "enumerated_worlds": len(worlds) if worlds is not None else None,
            "closed_under_policy": final["supported"] and not spec.unresolved,
            "sampling_schedule_completed": all(v.closed for v in views.values()), "mapping_conflicts": conflicts,
            "issues": list(spec.unresolved) + list(scope_issues) + (["query_category_mapping_conflict"] if conflicts else [])})


WORDS = {n: i for i, n in enumerate("zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty".split())}


def simple_choice(text):
    original = text.strip()
    text = original.rstrip(".")
    if text.casefold() in {"yes", "no", "true", "false"}:
        return text.casefold() in {"yes", "true"}
    if text.casefold() in WORDS:
        return WORDS[text.casefold()]
    if re.fullmatch(r"\d+(?:\.\d+)?", text):
        return float(text) if "." in text else int(text)
    # Noun phrases, ranges and ordered clauses go through the same compile call's
    # typed mapping. A leading number does not make the whole option that number.
    return original


def comparable(value):
    if isinstance(value, str):
        return value.strip().casefold().rstrip(".")
    if isinstance(value, list):
        return [comparable(v) for v in value]
    if isinstance(value, dict):
        return {comparable(k): comparable(v) for k, v in value.items()}
    return value


def typed_equal(left, right, *, literal=False):
    # JSON booleans must never match numeric one/zero; literal values retain case.
    if isinstance(left, bool) or isinstance(right, bool):
        return type(left) is type(right) and left == right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return left == right
    if type(left) is not type(right):
        return False
    if isinstance(left, list):
        return len(left) == len(right) and all(typed_equal(a, b, literal=literal) for a, b in zip(left, right))
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(typed_equal(left[k], right[k], literal=literal) for k in left)
    return left == right if literal else comparable(left) == comparable(right)


def option_accepts(value, option, literal=False):
    if isinstance(option, dict) and "kind" in option:
        if option["kind"] == "interval":
            return type(value) in {int,float} and option["low"] <= value and (option["high"] is None or value <= option["high"])
        if option["kind"] == "one_of":
            return any(typed_equal(value, x, literal=literal) for x in option["values"])
        if option["kind"] == "set":
            return isinstance(value,list) and all(any(typed_equal(x,y,literal=literal) for y in option["value"]) for x in value) and all(any(typed_equal(x,y,literal=literal) for y in value) for x in option["value"])
        option=option["value"]
    return typed_equal(value,option,literal=literal) or (isinstance(value,list) and len(value)==1 and typed_equal(value[0],option,literal=literal))


def option_domain(option):
    if type(option) in {int,float}:
        return [(option,option)]
    if isinstance(option,dict):
        if option.get("kind")=="interval":
            return [(option["low"],option["high"])]
        if option.get("kind")=="one_of" and all(type(x) in {int,float} for x in option["values"]):
            return [(x,x) for x in option["values"]]
    return None


def map_answer(state, spec, request):
    row = state.get("final", {})
    mapping = {c.label: spec.choice_values.get(c.label, simple_choice(c.text)) for c in request.choices}
    # Unambiguous program parsing takes precedence over model-transcribed numeric options.
    for c in request.choices:
        parsed = simple_choice(c.text)
        if not isinstance(parsed, str):
            mapping[c.label] = parsed
    if row.get("supported") and not spec.unresolved:
        value = row["value"]
        if isinstance(value, dict) and len(value) == 1 and row.get("op") in {"membership", "remaining_quantity"}:
            scalar = next(iter(value.values()))
            if request.output_protocol == "numeric" or (mapping and all(isinstance(v, bool) or option_domain(v) is not None for v in mapping.values())):
                value = scalar
        if not request.choices:
            return (json.dumps(value, ensure_ascii=False) if not isinstance(value, str) else value), "deterministic_reduce"
        literal = any(s.namespace == "text_value" or s.equivalence == "combination" for s in spec.sets)
        if type(value) in {int,float} and any(option_domain(x) is None for x in mapping.values()):
            return None, "option_mapping_unresolved"
        matches = [label for label,option in mapping.items() if option_accepts(value,option,literal)]
        if len(matches) == 1:
            return matches[0], "explicit_unique_missing" if row.get("constraint_basis") else "deterministic_reduce"
    if not spec.unresolved and row.get("possible_values"):
        assignments = [[k for k,v in mapping.items() if option_accepts(value,v)] for value in row["possible_values"]]
        if all(len(a)==1 and a==assignments[0] for a in assignments):
            return assignments[0][0], "deterministic_all_interpretations"
    # A coarse public option may be invariant even with unknown identity or open coverage.
    # Open coverage has no fabricated upper bound; only an unbounded option can contain it.
    b = row.get("bounds")
    domains = {k:option_domain(v) for k,v in mapping.items()}
    if b and len(b)==2 and type(b[0]) in {int,float} and (b[1] is None or type(b[1]) in {int,float}) and domains and all(v is not None for v in domains.values()) and not spec.unresolved:
        lo,hi=b[0],float("inf") if b[1] is None else b[1]
        covering,overlapping=[],[]
        for k,spans in domains.items():
            if any(a<=lo and (float("inf") if z is None else z)>=hi for a,z in spans):
                covering.append(k)
            if any(a<=hi and (float("inf") if z is None else z)>=lo for a,z in spans):
                overlapping.append(k)
        if len(covering)==len(overlapping)==1:
            return covering[0], "deterministic_bound_mapping"
    intervals = request.benchmark_policy.get("numeric_option_intervals", {})
    b = row.get("bounds")
    if b and len(b) == 2 and all(isinstance(x, (int, float)) for x in b) and state.get("sampling_schedule_completed") and not spec.unresolved:
        matches = [k for k, (lo, hi) in intervals.items() if lo <= b[0] <= b[1] <= hi]
        overlaps = [k for k, (lo, hi) in intervals.items() if lo <= b[1] and b[0] <= hi]
        if len(matches) == len(overlaps) == 1:
            return matches[0], "deterministic_bound_mapping"
    return None, "option_mapping_conflict" if row.get("supported") else "unresolved"
