import copy

import pytest
from r7_fakes import candidates, intervention, reason, scenario, task

from qwen3vl_agent.r7.config import R7Config
from qwen3vl_agent.r7.mechanisms import (
    BUILTIN_RULE,
    audit_candidates,
    conditional_hypotheses,
    execute_worlds,
    physics,
    trend,
    unique_entailed,
)
from qwen3vl_agent.r7.operators import compare, count_events, execute, resolve
from qwen3vl_agent.r7.state import Cell, FactStore, World
from qwen3vl_agent.r7.types import ProtocolError


def world(values):
    return World(
        "factual",
        0,
        {
            k: Cell(k, v, "observed", ["e:" + k], grounded=True, complete=True)
            for k, v in values.items()
        },
    )


def step(op, out, args, params=None):
    return {
        "id": out,
        "op": op,
        "out": out,
        "args": [{"ref": a} for a in args],
        "params": params or {},
        "rule_id": "builtin",
    }


def run(steps, w):
    execute(steps, w, {"builtin": BUILTIN_RULE}, {"builtin": BUILTIN_RULE})


def test_atomic_snapshot_assignment_and_swap():
    w = world({"L.count": 2, "R.count": 9})
    w.transact([intervention(), intervention("L.count", 5, identifier="set-left")])
    assert w.get("R.count").value == 2
    assert w.get("L.count").value == 5
    assert "e:L.count" in w.get("R.count").sources
    swapped = world({"L.count": 2, "R.count": 9})
    swapped.transact([intervention("L.count", "R.count", op="swap")])
    assert (swapped.get("L.count").value, swapped.get("R.count").value) == (9, 2)


def test_sequential_rhs_and_invariant_errors():
    w = world({"L.count": 2, "R.count": 9})
    second = intervention("L.count", {"ref": "R.count"}, identifier="second")
    second.update(sequential=True, read_world="current")
    w.transact([intervention("R.count", 5), second])
    assert w.get("L.count").value == 5
    w.invariants.add("R.count")
    with pytest.raises(ProtocolError):
        w.transact([intervention()])


def test_transitive_invalidation_and_direct_write_protection():
    w = world({"x": 2, "one": 1})
    run([step("ADD", "y", ["x", "one"]), step("ADD", "z", ["y", "one"])], w)
    w.transact([intervention("x", 4)])
    assert not w.get("y").valid and not w.get("z").valid
    run([step("ADD", "y", ["x", "one"]), step("ADD", "z", ["y", "one"])], w)
    assert w.get("z").value == 6
    with pytest.raises(ProtocolError):
        run([step("SET", "x", ["one"])], w)


@pytest.mark.parametrize(
    "op,args,values,expected",
    [
        ("READ_FACT", ["x"], {"x": 3}, 3),
        ("SET", ["x"], {"x": 3}, 3),
        ("ADD", ["x", "y"], {"x": {"lo": 1, "hi": 2}, "y": 3}, {"lo": 4, "hi": 5}),
        ("TABLE_LOOKUP", ["table", "key"], {"table": {"red": 3}, "key": "red"}, 3),
        (
            "RULE_TRANSITION",
            ["table", "state", "event"],
            {"table": {"closed": {"open": "opened"}}, "state": "closed", "event": "open"},
            "opened",
        ),
        ("SELECT", ["list", "index"], {"list": ["a", "b"], "index": 1}, "b"),
    ],
)
def test_typed_whitelisted_operations(op, args, values, expected):
    w = world(values)
    run([step(op, "result", args)], w)
    assert w.get("result").value == expected and w.get("result").grounded


def test_unknown_unbacked_and_arbitrary_code_never_execute():
    w = world({"known": 1})
    run([step("ADD", "result", ["missing", "known"])], w)
    assert w.get("result").value is None
    with pytest.raises(ProtocolError):
        resolve({"literal": 3, "source": "missing"}, w, {})
    with pytest.raises(ProtocolError):
        run([step("__import__('os')", "result", [])], w)
    with pytest.raises(ProtocolError):
        execute([step("SET", "result", ["known"])] * 65, w, {}, {"builtin": BUILTIN_RULE})


def test_event_overlap_and_coverage():
    event = {
        "entity_id": "flower",
        "predicate": "water",
        "start": 1,
        "end": 2,
        "occurrence_id": "e1",
    }
    overlap = {**event, "start": 1.5, "end": 2.5, "occurrence_id": "other_window"}
    later = {**event, "start": 3, "end": 4, "occurrence_id": "e2"}
    assert count_events([event, overlap, later]) == 2
    w = world({"events1": [event], "events2": [overlap, later]})
    run([step("COUNT_EVENTS", "count", ["events1", "events2"])], w)
    assert w.get("count").value == 2
    w.cells["events2"].complete = False
    run([step("COUNT_EVENTS", "count", ["events1", "events2"])], w)
    assert w.get("count").value is None


@pytest.mark.parametrize(
    "a,rel,b,expected",
    [
        (2, "eq", 2, True),
        ({"lo": 1, "hi": 3}, "eq", 2, None),
        ({"lo": 4, "hi": 5}, "gt", 3, True),
        (None, "eq", 0, None),
        (True, "eq", False, False),
    ],
)
def test_three_valued_interval_comparison(a, rel, b, expected):
    assert compare(a, rel, b) is expected


def test_sort_select_cannot_hide_uncertain_rank():
    w = world({"levels": {"a": {"lo": 4, "hi": 6}, "b": 5}, "first": 0})
    run([step("SORT", "rank", ["levels"]), step("SELECT", "winner", ["rank", "first"])], w)
    assert w.get("rank").value["ambiguous"] and not w.get("winner").known


def test_compound_candidate_and_option_permutation():
    cs = candidates()["candidates"]
    w = world({"R.count": 2})
    w.id = "main"
    assert unique_entailed(audit_candidates(cs, {"main": w})) == "A"
    cs[0]["atoms"].append({**cs[0]["atoms"][0], "id": "a2", "key": "unseen"})
    assert unique_entailed(audit_candidates(cs, {"main": w})) is None
    permuted = candidates([{"label": "A", "text": "9"}, {"label": "B", "text": "2"}])["candidates"]
    assert unique_entailed(audit_candidates(permuted, {"main": w})) == "B"


@pytest.mark.parametrize("mechanism", ["S1", "S2"])
def test_conditional_future_is_supported_not_observed(mechanism):
    w = world({"ready": True})
    rules = {"r": {"basis": "hypothesis"}}
    h = {
        "id": "h1",
        "mechanism": mechanism,
        "output": "next",
        "value": "open",
        "conditions": [{"key": "ready", "relation": "eq", "expected": True}],
        "rule_id": "r",
    }
    conditional_hypotheses([h], w, rules, 3)
    assert w.get("next").value == "open" and not w.get("next").grounded
    assert w.get("next").kind == "derived"
    h2 = {**h, "id": "h2", "value": "wait"}
    conditional_hypotheses([h, h2], w, rules, 3)
    assert not w.get("next").known


def physics_fixture():
    values, objects = {}, []
    for eid, position, velocity in [
        ("A", [0, 0], [1, 0]),
        ("B", [10, 0], [-1, 0]),
        ("C", [5, 0], [0, 0]),
    ]:
        values.update(
            {
                eid + ".exists": True,
                eid + ".position": position,
                eid + ".velocity": velocity,
                eid + ".radius": 1,
                eid + ".reference": "world_2d_metric",
                eid + ".valid_until": 10,
                eid + ".delay": 0,
            }
        )
        objects.append(
            {
                "entity_id": eid,
                **{
                    name + "_key": eid + "." + name
                    for name in (
                        "exists",
                        "position",
                        "velocity",
                        "radius",
                        "reference",
                        "valid_until",
                        "delay",
                    )
                },
            }
        )
    values["horizon"] = 5
    w = world(values)
    for eid in ("A", "B", "C"):
        w.cells[eid + ".position"].unit = w.cells[eid + ".radius"].unit = "m"
        w.cells[eid + ".velocity"].unit = "m/s"
        w.cells[eid + ".position"].time = 0
    model = {"objects": objects, "horizon": {"ref": "horizon"}, "supports": [], "rule_id": "r"}
    return w, model, {"r": {"basis": "stipulated"}}


def test_s3_removal_discovers_new_interaction_and_replay():
    factual, model, rules = physics_fixture()
    physics(model, factual, rules, rules)
    assert factual.get("collision:A:C").value is True
    assert factual.get("collision:A:B").value is None  # after an earlier collision
    hypoth = factual.fork("remove")
    hypoth.transact([intervention("C.exists", False, op="remove")])
    physics(model, hypoth, rules, rules)
    assert hypoth.get("collision:A:B").value is True
    assert hypoth.get("collision:A:C").value is False
    replay = factual.fork("no-intervention")
    physics(model, replay, rules, rules)
    assert replay.get("collision:A:C").value == factual.get("collision:A:C").value
    assert factual.get("C.exists").value is True


def test_s3_delay_uses_shared_clock_and_unknown_screen_geometry():
    w, model, rules = physics_fixture()
    w.transact(
        [
            intervention("C.exists", False, op="remove"),
            intervention("A.delay", 2, op="delay", identifier="delay"),
        ]
    )
    physics(model, w, rules, rules)
    assert w.get("B.delay").value == 0
    assert next(e["contact_time"] for e in w.execution if e.get("pair") == ["A", "B"]) == 5
    w.cells["A.reference"].value = "screen"
    physics(model, w, rules, rules)
    assert w.get("collision:A:B").value is None


def trend_fixture():
    w = world(
        {
            "binding": {"time": 2021, "entities": ["A", "B"]},
            "universe": ["A", "B"],
            "bind_time": 2021,
            "t0": 2020,
            "t1": 2025,
            "base_time": 2025,
            "target": 2030,
            "A.start": 10,
            "A.end": 20,
            "A.base": 20,
            "B.start": 20,
            "B.end": 25,
            "B.base": 25,
        }
    )
    for k in w.cells:
        if k.startswith(("A.", "B.")):
            w.cells[k].unit = "units"
    model = {
        "output": "levels",
        "entities": ["A", "B"],
        "entity_binding_time": {"ref": "bind_time"},
        "trend_start": {"ref": "t0"},
        "trend_end": {"ref": "t1"},
        "base_time": {"ref": "base_time"},
        "target_time": {"ref": "target"},
        "method": "absolute",
        "binding_key": "binding",
        "universe_key": "universe",
        "series": [
            {
                "entity_id": e,
                "start_key": e + ".start",
                "end_key": e + ".end",
                "base_key": e + ".base",
            }
            for e in ("A", "B")
        ],
        "rule_id": "trend",
    }
    return w, model, {"trend": {"basis": "stipulated"}}


def test_s5_absolute_ratio_and_multiscenario_isolation():
    base, model, rules = trend_fixture()
    a, b = base.fork("2030"), base.fork("2035")
    trend(model, a, rules, rules)
    b.cells["target"].value = 2035
    trend(model, b, rules, rules)
    assert a.get("levels").value == {"A": 30, "B": 30}
    assert b.get("levels").value == {"A": 40, "B": 35}
    assert base.get("target").value == 2030
    model["method"] = "ratio"
    trend(model, a, rules, rules)
    assert a.get("levels").value == {"A": 40, "B": 31.25}


@pytest.mark.parametrize("bad", ["rank", "universe", "zero_ratio", "binding", "units"])
def test_s5_invalid_inputs_remain_unknown(bad):
    w, model, rules = trend_fixture()
    if bad == "rank":
        w.cells["A.start"].unit = "rank"
    elif bad == "universe":
        w.cells["universe"].value.append("C")
    elif bad == "zero_ratio":
        model["method"] = "ratio"
        w.cells["A.start"].value = 0
    elif bad == "binding":
        w.cells["binding"].value["time"] = 2020
    else:
        w.cells["B.end"].unit = "percent"
    trend(model, w, rules, rules)
    assert w.get("levels").value is None


def test_explicit_scenarios_are_not_truncated_to_three():
    store = FactStore()
    cs = candidates()["candidates"]
    cs = [
        {
            **copy.deepcopy(cs[0]),
            "label": str(i),
            "interventions": [intervention(identifier="i" + str(i))],
        }
        for i in range(8)
    ]
    for c in cs:
        c["atoms"][0]["scenario_id"] = c["label"]
    proposal = reason(scenarios=[scenario(c["label"], candidate_label=c["label"]) for c in cs])
    spec = task()
    spec["interventions"] = []
    worlds = execute_worlds(spec, cs, proposal, store, R7Config())
    assert len(worlds) == 9


def test_packed_composites_event_filters_and_top_k():
    events = [
        {"entity_id": "L", "predicate": "water", "start": 1, "end": 2, "occurrence_id": "a"},
        {"entity_id": "R", "predicate": "water", "start": 3, "end": 4, "occurrence_id": "b"},
    ]
    w = world({"events": events, "window": [0, 5], "left": 3, "middle": 2, "right": 1, "k": 2})
    run(
        [
            step(
                "COUNT_EVENTS",
                "left_count",
                ["events"],
                {"entity_id": "L", "time_window": {"ref": "window"}},
            ),
            step("SET", "pair", ["left", "right"], {"pack": "list"}),
            step(
                "SET",
                "levels",
                ["left", "middle", "right"],
                {"pack": "map", "keys": ["L", "M", "R"]},
            ),
            step("SORT", "order", ["levels"]),
            step("SELECT", "top", ["order", "k"], {"top_k": True}),
        ],
        w,
    )
    assert w.get("left_count").value == 1
    assert w.get("pair").value == [3, 1]
    assert w.get("top").value == ["L", "M"]


def test_intervention_sensitive_and_irrelevant_changes():
    base = world({"x": 2, "extra": 8, "one": 1})
    outcomes = []
    for target, value in [("x", 2), ("x", 4), ("extra", 99)]:
        w = base.fork(target + str(value))
        w.transact([intervention(target, value)])
        run([step("ADD", "result", ["x", "one"])], w)
        outcomes.append(w.get("result").value)
    assert outcomes == [3, 5, 3]


def test_question_scenario_interventions_do_not_mix():
    base = world({"L.count": 2, "R.count": 9})
    store = FactStore()
    store.versions[0]["cells"] = {k: c.to_dict() for k, c in base.cells.items()}
    spec = task()
    spec["interventions"] = [
        {**intervention("R.count", 5, identifier="five"), "scenario_id": "five"},
        {**intervention("R.count", 7, identifier="seven"), "scenario_id": "seven"},
    ]
    worlds = execute_worlds(
        spec,
        candidates()["candidates"],
        reason(scenarios=[scenario("five"), scenario("seven")]),
        store,
        R7Config(),
    )
    assert worlds["five"].get("R.count").value == 5
    assert worlds["seven"].get("R.count").value == 7


def test_physical_replay_failure_downgrades_mechanism():
    w, model, _ = physics_fixture()
    w.cells["collision:A:C"] = Cell(
        "collision:A:C", False, "observed", ["measured-no-contact"], grounded=True
    )
    store = FactStore()
    store.versions[0]["cells"] = {k: c.to_dict() for k, c in w.cells.items()}
    spec = task("S3")
    spec["interventions"] = []
    proposal = reason(
        rules=[{"id": "r", "basis": "stipulated"}], scenarios=[scenario(physics=[model])]
    )
    worlds = execute_worlds(spec, candidates()["candidates"], proposal, store, R7Config())
    assert worlds["factual"].execution[0]["status"] == "failed_downgraded"
    assert not worlds["main"].get("collision:A:C").grounded


def test_observed_postintervention_position_cannot_seed_counterfactual():
    w, model, rules = physics_fixture()
    w.transact([intervention("C.exists", False, op="remove")])
    for eid in ("A", "B"):
        w.cells[eid + ".position"].time = 1
    physics(model, w, rules, rules)
    assert w.get("collision:A:B").value is None
