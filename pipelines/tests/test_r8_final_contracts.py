from fractions import Fraction

import pytest
from r8_fakes import program
from test_r8_geometry import geometry_store

from qwen3vl_agent.r8.adapters import Adapters
from qwen3vl_agent.r8.config import R8Config
from qwen3vl_agent.r8.controller import Controller
from qwen3vl_agent.r8.evaluate import score
from qwen3vl_agent.r8.geometry import apply_rules
from qwen3vl_agent.r8.matcher import parse_option
from qwen3vl_agent.r8.types import ModelingError


@pytest.mark.parametrize(
    "text,unit,base",
    [
        ("280g", "kg", "280"),
        ("1.5cm²", "m^2", "3/20000"),
        ("2mm^2", "m^2", "1/500000"),
        ("2,000g", "kg", "2000"),
    ],
)
def test_attached_units_are_not_silently_reinterpreted(text, unit, base):
    parsed = parse_option(text, unit, "number")
    assert parsed["kind"] == "value"
    assert parsed["value"].base == Fraction(base)


@pytest.mark.parametrize("only", ["tl_region", "total_region"])
def test_one_positive_region_does_not_license_every_partition(only):
    store, geometry = geometry_store()
    store.state["relations"]["positive"]["objects"] = {"one": only}
    with pytest.raises(ModelingError, match="nondegeneracy"):
        apply_rules(geometry, store.state["relations"], R8Config())


def geometry_controller(joint=False):
    store, geometry = geometry_store()
    p = program()
    p.update(backend="constraints", geometry=geometry)
    p["query"] = {
        "nodes": [
            {"id": "solved", "op": "solve_target", "args": ["solver_target"], "params": {}},
            {
                "id": "twice",
                "op": "multiply",
                "args": ["solved", {"value": "2", "unit": "1", "source": "definition:two"}],
                "params": {},
            },
        ],
        "target_node": "twice",
        "output_unit": "m^2",
    }
    if joint:
        store.get("tl").update(
            raw_text="12 or 13", alternatives=["13"], alternatives_exhaustive=True
        )
    ctl = Controller.__new__(Controller)
    ctl.config, ctl.store, ctl.adapters = R8Config(), store, Adapters(store)
    ctl.state, ctl.settings, ctl.task = (
        {"program": p, "executions": []},
        {"constraints": True},
        {"output_unit": "m^2"},
    )
    ctl.save = lambda: None
    ctl.coverage_complete = lambda: True
    # Joint execution needs only the same immutable request question and scope contract.
    from types import SimpleNamespace

    ctl.request = SimpleNamespace(question=store.question)
    ctl.media = SimpleNamespace(contract=store.contract)
    return ctl


@pytest.mark.parametrize("joint", [False, True])
def test_geometry_postprocessing_runs_in_every_candidate_branch(joint):
    ctl = geometry_controller(joint)
    value, _ = ctl.execution()
    if joint:
        assert value["complete"]
        assert [q.value for q in value["candidate_values"]] == [24, Fraction(288, 13)]
    else:
        assert value.value == 24


@pytest.mark.parametrize("joint", [False, True])
def test_geometry_cannot_bypass_declared_failed_checks(joint):
    ctl = geometry_controller(joint)
    p = ctl.state["program"]
    p["query"]["nodes"].append(
        {"id": "wrong", "op": "equal", "args": ["twice", "tl"], "params": {}}
    )
    p["checks"] = ["wrong"]
    with pytest.raises(ModelingError, match="declared check"):
        ctl.execution()


def test_baseline_answer_and_verification_rates_are_separate():
    report = score(
        [
            {
                "request_id": "q",
                "mode": "A",
                "run_status": "completed",
                "result": {"prediction": "B", "status": "unresolved_modeling", "verified": False},
            }
        ],
        [{"request_id": "q", "answer": "B"}],
    )["experiments"][0]
    assert report["no_prediction_rate"] == 0
    assert report["unverified_rate"] == report["unresolved_rate"] == 1
