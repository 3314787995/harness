from dataclasses import replace

import pytest
from r8_fakes import FakeModel, observation, program, task
from test_r8_adapters import attempt, query
from test_r8_agent import setup as setup  # noqa: PLC0414 -- explicit pytest fixture re-export

from qwen3vl_agent.r8 import R8VideoAgent
from qwen3vl_agent.r8.evidence import numeric_in_span
from qwen3vl_agent.r8.matcher import parse_option
from qwen3vl_agent.r8.operators import calculate as calc
from qwen3vl_agent.r8.types import ModelingError
from qwen3vl_agent.r8.units import Quantity as Q


def test_package_count_unit_is_explicit():
    assert numeric_in_span("280", "If the recipe calls for 280g of honey")
    assert numeric_in_span("-100", "cash change −100.")
    assert [v.value for v in parse_option("0,100", "CNY", "tuple")["value"]] == [0, 100]
    result = calc(
        "minimum_required_packages", [Q.make(280, "g"), Q.make(30, "g")], {"count_unit": "jar"}
    )
    assert result.value == 10 and result.unit.label == "jar"
    result = calc(
        "maximum_affordable_count",
        [Q.make(10, "GBP"), Q.make(3, "GBP")],
        {"count_unit": "count:box"},
    )
    assert result.value == 3 and result.unit.label == "count:box"
    with pytest.raises(ModelingError):
        calc(
            "minimum_required_packages", [Q.make(280, "g"), Q.make(30, "g")], {"count_unit": "USD"}
        )


def test_interrupted_call_is_charged_on_resume(setup, tmp_path):
    config, request = setup

    def interrupted(_):
        raise KeyboardInterrupt("synthetic interruption")

    request = replace(request, checkpoint_path=str(tmp_path / "interrupted.jsonl"))
    with pytest.raises(KeyboardInterrupt):
        R8VideoAgent(FakeModel({"observe": interrupted}), config).solve(request)
    model = FakeModel()
    result = R8VideoAgent(model, config).solve(replace(request, resume=True))
    assert result.prediction == "B"
    assert "compile" not in [c[0]["stage"] for c in model.calls]
    assert result.resources["model_calls"] == 6
    assert any(r["status"] == "interrupted" for r in result.resources["receipts"])


def test_declared_false_check_blocks_verification(setup):
    config, request = setup

    def invalid(_):
        value = program()
        value["checks"] = ["nonexistent_check"]
        return value

    result = R8VideoAgent(FakeModel({"formalize": invalid}), config).solve(
        replace(request, mode="C")
    )
    assert not result.verified and result.prediction is None


def test_range_scan_repairs_missing_discovery(setup):
    config, request = setup
    calls = 0

    def compile_range(_):
        spec = task()
        spec.update(
            plan="range", coverage_need="all_attempts", answer_type="percentage", output_unit="%"
        )
        return spec

    def read_range(payload):
        nonlocal calls
        calls += 1
        value = observation(payload)
        value["attempts"] = [attempt("a", 1.0), attempt("b", 2.0, "failure")]
        for row in value["attempts"]:
            row["evidence_refs"] = ["F01"]
        value["discovery"]["complete"] = calls > 1
        return value

    def ratio_program(_):
        value = program()
        value["adapters"] = [query()]
        value["adapters"][0]["window"] = [0.0, 5.0]
        value["query"] = {
            "nodes": [
                {
                    "id": "s",
                    "op": "select_attribute",
                    "args": ["stats"],
                    "params": {"attribute": "successes"},
                },
                {
                    "id": "n",
                    "op": "select_attribute",
                    "args": ["stats"],
                    "params": {"attribute": "total"},
                },
                {"id": "ratio", "op": "ratio", "args": ["s", "n"], "params": {}},
                {"id": "percent", "op": "percentage", "args": ["ratio"], "params": {}},
            ],
            "target_node": "percent",
            "output_unit": "%",
        }
        return value

    fake = FakeModel({"compile": compile_range, "observe": read_range, "formalize": ratio_program})
    result = R8VideoAgent(fake, config).solve(
        replace(request, mode="F", choices=("A. 10%", "B. 50%"))
    )
    assert result.prediction == "B" and result.verified
    assert (
        not result.coverage[0]["discovery_complete"] and result.coverage[-1]["discovery_complete"]
    )
    assert result.trace["actions"][0]["kind"] == "directed_scan"
