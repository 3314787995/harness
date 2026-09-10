"""S1/S2 conditional support, conservative S3 propagation, and interval-valued S5."""

from __future__ import annotations

import itertools
import math
from copy import deepcopy

from .operators import bounds, compare, execute, number, resolve
from .state import Cell, World
from .types import ProtocolError, finite

BUILTIN_RULE = {
    "id": "builtin",
    "basis": "stipulated",
    "source_span": "",
    "description": "typed mathematical/collection semantics; grants no physical or perceptual assumption",
}


def conditional_hypotheses(items, world, rules, max_branches):
    grouped = {}
    for item in items:
        if item["mechanism"] not in {"S1", "S2"}:
            raise ProtocolError("conditional hypotheses are S1/S2 only")
        if item["rule_id"] not in rules or item["rule_id"] == "builtin":
            raise ProtocolError("future support requires a named contextual mechanism")
        checks, deps, sources = [], [], [item["rule_id"]]
        for condition in item["conditions"]:
            cell = world.get(condition["key"])
            result = (
                compare(cell.value, condition["relation"], condition["expected"])
                if cell.known
                else None
            )
            # An uncertain inferred condition cannot become a hard prerequisite.
            checks.append(result if cell.grounded else None)
            deps.append(cell.key)
            sources.extend(cell.sources)
        status = (
            "supported"
            if checks and all(x is True for x in checks)
            else "contradicted"
            if False in checks
            else "unknown"
        )
        branch = {
            "id": item["id"],
            "value": deepcopy(item["value"]),
            "status": status,
            "rule_id": item["rule_id"],
            "dependencies": deps,
            "sources": sorted(set(sources)),
        }
        grouped.setdefault(item["output"], []).append(branch)
    for output, branches in grouped.items():
        if len(branches) > max_branches:
            raise ProtocolError("latent branch cap exceeded")
        supported = [b for b in branches if b["status"] == "supported"]
        values = [b["value"] for b in supported]
        value = values[0] if values and all(v == values[0] for v in values) else None
        world.write(
            Cell(
                output,
                value,
                "derived" if value is not None else "unknown",
                sorted({s for b in branches for s in b["sources"]}),
                sorted({d for b in branches for d in b["dependencies"]}),
                grounded=False,
                reason="conditional_future_support",
            )
        )
        world.execution.append({"mechanism": "S1/S2", "output": output, "branches": branches})


def _scalar(cell):
    value = bounds(cell.value) if cell.known else None
    return value[0] if value and value[0] == value[1] else None


def _contact_time(a, b, horizon):
    """Piecewise constant motion under independent actor delay; shared global time."""
    r = a["radius"] + b["radius"]
    cuts = sorted({0.0, horizon, min(horizon, a["delay"]), min(horizon, b["delay"])})
    for start, end in itertools.pairwise(cuts):
        positions = [
            [p + v * max(0.0, start - o["delay"]) for p, v in zip(o["position"], o["velocity"])]
            for o in (a, b)
        ]
        relative = [x - y for x, y in zip(*positions)]
        velocity = [
            va * (start >= a["delay"]) - vb * (start >= b["delay"])
            for va, vb in zip(a["velocity"], b["velocity"])
        ]
        aa = sum(v * v for v in velocity)
        bb = 2 * sum(p * v for p, v in zip(relative, velocity))
        cc = sum(p * p for p in relative) - r * r
        if not all(math.isfinite(v) for v in (aa, bb, cc)):
            raise ArithmeticError("unrepresentable physical scale")
        if cc <= 0:
            return start
        disc = bb * bb - 4 * aa * cc
        if aa > 0 and disc >= 0:
            delta = (-bb - math.sqrt(disc)) / (2 * aa)
            if 0 <= delta <= end - start:
                return start + delta
    return None


def physics(model, world, sources, rules):
    rule_id = model["rule_id"]
    if rule_id not in rules or rule_id == "builtin":
        raise ProtocolError("physics requires a named applicability rule")
    horizon_cell = resolve(model["horizon"], world, sources)
    horizon = _scalar(horizon_cell)
    objects, inputs = {}, {}
    for item in model["objects"]:
        keys = [v for k, v in item.items() if k.endswith("_key") and v]
        cells = [world.get(k) for k in keys]
        inputs[item["entity_id"]] = cells
        exists = world.get(item["exists_key"])
        position, velocity = world.get(item["position_key"]), world.get(item["velocity_key"])
        radius, reference = world.get(item["radius_key"]), world.get(item["reference_key"])
        valid_until = _scalar(world.get(item["valid_until_key"]))
        delay_cell = world.get(item["delay_key"]) if item["delay_key"] else None
        delay = _scalar(delay_cell) if delay_cell else 0.0
        valid = (
            exists.known
            and exists.value is True
            and position.known
            and velocity.known
            and isinstance(position.value, list)
            and isinstance(velocity.value, list)
            and len(position.value) == len(velocity.value)
            and len(position.value) in {2, 3}
            and all(finite(v) for v in position.value + velocity.value)
            and _scalar(radius) is not None
            and _scalar(radius) >= 0
            and reference.known
            and reference.value in {"world_2d_metric", "world_3d_metric"}
            and horizon is not None
            and horizon > 0
            and valid_until is not None
            and position.time is not None
            and valid_until >= position.time + horizon
            and delay is not None
            and delay >= 0
            and position.unit
            and radius.unit == position.unit
            and velocity.unit == position.unit + "/s"
        )
        if (
            world.intervention_time is not None
            and position.time is not None
            and abs(position.time - world.intervention_time) > 1e-6
        ):
            valid = False  # An observed post-intervention trajectory is not an initial condition.
        objects[item["entity_id"]] = {
            "exists": exists.value if exists.known else None,
            "valid": bool(valid),
            "position": position.value,
            "velocity": velocity.value,
            "radius": _scalar(radius),
            "reference": reference.value,
            "time": position.time,
            "unit": position.unit,
            "delay": delay,
        }
    contacts = []
    # All pairs, including relationships absent from the factual event graph.
    for a, b in itertools.combinations(objects, 2):
        aa, bb = objects[a], objects[b]
        time, value = None, None
        if aa["exists"] is False or bb["exists"] is False:
            value = False
        elif (
            aa["valid"]
            and bb["valid"]
            and aa["reference"] == bb["reference"]
            and aa["time"] == bb["time"]
            and aa["unit"] == bb["unit"]
            and len(aa["position"]) == len(bb["position"])
        ):
            try:
                time = _contact_time(aa, bb, horizon)
                value = time is not None
            except ArithmeticError:
                value = None
        contacts.append((a, b, value, time))
    first = min((t for _, _, _, t in contacts if t is not None), default=None)
    unknown_live = any(o["exists"] is not False and not o["valid"] for o in objects.values())
    for a, b, value, time in contacts:
        cells = inputs[a] + inputs[b] + [horizon_cell]
        # After the first contact or in presence of unmodelled live actors, linear replay ends.
        if (
            objects[a]["exists"] is not False
            and objects[b]["exists"] is not False
            and (unknown_live or (first is not None and (time is None or time > first + 1e-8)))
        ):
            value = None
        key = f"collision:{a}:{b}"
        cell = Cell(
            key,
            value,
            "derived" if value is not None else "unknown",
            sorted({s for c in cells for s in c.sources} | {rule_id}),
            [c.key for c in cells],
            grounded=all(c.grounded for c in cells) and rules[rule_id]["basis"] == "stipulated",
            reason="first_contact_in_valid_metric_window"
            if value is not None
            else "physical_state_or_postcontact_unresolved",
        )
        world.write(cell)
        world.execution.append(
            {
                "mechanism": "S3",
                "pair": [a, b],
                "output": key,
                "value": value,
                "contact_time": time,
                "new_interaction_checked": True,
            }
        )
    for support in model["supports"]:
        supporter = objects.get(support["supporter"], {})
        relation = world.get(support["key"])
        if supporter.get("exists") is False:
            world.write(
                Cell(
                    support["key"],
                    False,
                    "derived",
                    [rule_id],
                    [support["supporter"] + ".exists"],
                    grounded=True,
                    reason="removed_support_contact",
                )
            )
            # Losing this contact does not prove falling: other supports may exist.
            world.write(
                Cell(
                    support["supported"] + ".support_outcome",
                    None,
                    "unknown",
                    relation.sources,
                    [support["key"]],
                    reason="recheck_other_supports_and_gravity",
                )
            )


def trend(model, world, sources, rules):
    rule_id = model["rule_id"]
    if rule_id not in rules or rule_id == "builtin":
        raise ProtocolError("trend needs a named continuation assumption")
    names = ("entity_binding_time", "trend_start", "trend_end", "base_time", "target_time")
    clocks = {k: resolve(model[k], world, sources) for k in names}
    times = {k: _scalar(v) for k, v in clocks.items()}
    binding, universe = world.get(model["binding_key"]), world.get(model["universe_key"])
    entities = model["entities"]
    valid = all(v is not None for v in times.values())
    valid = (
        valid
        and times["trend_end"] > times["trend_start"]
        and times["target_time"] >= times["base_time"]
    )
    valid = (
        valid
        and binding.known
        and universe.known
        and isinstance(binding.value, dict)
        and isinstance(universe.value, list)
    )
    valid = (
        valid
        and binding.value.get("time") == times["entity_binding_time"]
        and binding.value.get("entities") == entities
    )
    valid = valid and set(entities) == set(universe.value) and len(set(entities)) == len(entities)
    valid = (
        valid
        and {s["entity_id"] for s in model["series"]} == set(entities)
        and len(model["series"]) == len(entities)
    )
    result, used, all_units = {}, [*clocks.values(), binding, universe], set()
    for series in model["series"]:
        cells = [world.get(series[k]) for k in ("start_key", "end_key", "base_key")]
        used.extend(cells)
        values = [bounds(c.value) if c.known else None for c in cells]
        units = {c.unit for c in cells}
        all_units.update(units)
        # Semantic chart years can be stored in keys/binding; cell.time always remains video seconds.
        if (
            not valid
            or None in values
            or len(units) != 1
            or not next(iter(units), "")
            or any("rank" in u.lower() for u in units)
        ):
            result[series["entity_id"]] = None
            continue
        start, end, base = values
        dt = times["trend_end"] - times["trend_start"]
        steps = (times["target_time"] - times["base_time"]) / dt
        if model["method"] == "absolute":
            value = number(
                base[0] + (end[0] - start[1]) * steps, base[1] + (end[1] - start[0]) * steps
            )
        elif min(*start, *end, *base) > 0:
            try:
                value = number(
                    base[0] * (end[0] / start[1]) ** steps, base[1] * (end[1] / start[0]) ** steps
                )
            except OverflowError:
                value = None
        else:
            value = None
        result[series["entity_id"]] = value
    known = bool(result) and all(v is not None for v in result.values()) and len(all_units) == 1
    value = result if known else None
    grounded = all(c.grounded for c in used) and rules[rule_id]["basis"] == "stipulated"
    world.write(
        Cell(
            model["output"],
            value,
            "derived" if known else "unknown",
            sorted({s for c in used for s in c.sources} | {rule_id}),
            [c.key for c in used],
            grounded=grounded,
            unit=next(iter(all_units), ""),
            reason="trend_execution" if known else "binding_universe_units_or_trend_unresolved",
        )
    )
    world.execution.append(
        {
            "mechanism": "S5",
            "output": model["output"],
            "times": times,
            "method": model["method"],
            "entities": entities,
            "partial_intervals": result,
            "status": "known" if known else "unknown",
        }
    )


def execute_worlds(spec, candidates, proposal, store, config, *, operators=True):
    rules = {r["id"]: deepcopy(r) for r in proposal["rules"]}
    if "builtin" in rules:
        raise ProtocolError("builtin mathematical rule cannot be overridden")
    rules["builtin"] = BUILTIN_RULE
    sources = {**rules, **{s["id"]: s for s in spec["stipulations"]}}
    factual = World("factual", store.version, store.current, spec["invariants"])
    for s in spec["stipulations"]:
        if s["key"] in factual.cells:
            raise ProtocolError("stipulation must not overwrite measured factual variable")
        factual.write(Cell(s["key"], s["value"], "stipulated", [s["id"]], grounded=True))
    if operators:
        execute(proposal["factual_program"], factual, sources, rules, config.max_program_steps)
        # Check physical replay only where a comparable, measured factual outcome exists.
        for scenario in proposal["scenarios"]:
            for model in scenario["physics"]:
                probe = factual.fork("replay:" + scenario["id"])
                physics(model, probe, sources, rules)
                comparisons = []
                for event in probe.execution:
                    key = event.get("output")
                    actual = factual.get(key) if key else Cell("")
                    predicted = probe.get(key) if key else Cell("")
                    if actual.kind == "observed" and actual.known and predicted.known:
                        comparisons.append(
                            {
                                "key": key,
                                "actual": actual.value,
                                "predicted": predicted.value,
                                "matches": actual.value == predicted.value,
                            }
                        )
                failed = any(not c["matches"] for c in comparisons)
                if failed:
                    rules[model["rule_id"]]["basis"] = "hypothesis"
                factual.execution.append(
                    {
                        "kind": "no_intervention_replay",
                        "rule_id": model["rule_id"],
                        "status": "failed_downgraded"
                        if failed
                        else "passed"
                        if comparisons
                        else "not_comparable",
                        "comparisons": comparisons,
                    }
                )
    worlds = {"factual": factual}
    by_label = {c["label"]: c for c in candidates}
    for scenario in proposal["scenarios"]:
        world = factual.fork(scenario["id"])
        interventions = deepcopy(
            [i for i in spec["interventions"] if i.get("scenario_id", "*") in {"*", scenario["id"]}]
        )
        if scenario["candidate_label"]:
            interventions += deepcopy(by_label[scenario["candidate_label"]]["interventions"])
        world.transact(interventions, allowed_sources=sources)
        if operators:
            # Recompute aggregate dependencies on the fork after invalidation.
            rerun = [s for s in proposal["factual_program"] if not world.get(s["out"]).valid]
            execute(rerun, world, sources, rules, config.max_program_steps)
            for model in scenario["physics"]:
                physics(model, world, sources, rules)
            for model in scenario["trends"]:
                trend(model, world, sources, rules)
            execute(scenario["programs"], world, sources, rules, config.max_program_steps)
        conditional_hypotheses(scenario["hypotheses"], world, rules, config.max_latent_branches)
        worlds[world.id] = world
    return worlds


def audit_candidates(candidates, worlds):
    assessments = []
    for candidate in candidates:
        atoms = []
        for atom in candidate["atoms"]:
            world = worlds.get(atom["scenario_id"])
            cell = world.get(atom["key"]) if world else Cell(atom["key"])
            comparison = (
                compare(cell.value, atom["relation"], atom["expected"]) if cell.known else None
            )
            if atom["polarity"] == "negative" and comparison is not None:
                comparison = not comparison
            if comparison is True:
                status = "entailed_by_execution" if cell.grounded else "supported"
            elif comparison is False and cell.grounded:
                status = "contradicted"
            else:
                status = "unknown"
            # Possibility and likelihood require comparison, not Boolean certainty about one branch.
            if atom["modality"] in {"may", "likely", "least_likely"} and status in {
                "entailed_by_execution",
                "contradicted",
            }:
                status = "supported" if comparison else "unknown"
            atoms.append(
                {
                    "id": atom["id"],
                    "status": status,
                    "evidence_ids": cell.sources,
                    "state_id": f"{atom['scenario_id']}:{atom['key']}",
                    "reason": cell.reason,
                }
            )
        assessments.append({"label": candidate["label"], "atoms": atoms})
    return assessments


def unique_entailed(assessments, query_operator="value"):
    if query_operator in {"will_not", "least_likely"}:
        return None  # Selection is not the same as positive proposition truth.
    supported = [
        a["label"]
        for a in assessments
        if a["atoms"] and all(x["status"] == "entailed_by_execution" for x in a["atoms"])
    ]
    if len(supported) != 1:
        return None
    return (
        supported[0]
        if all(
            a["label"] == supported[0] or any(x["status"] == "contradicted" for x in a["atoms"])
            for a in assessments
        )
        else None
    )
