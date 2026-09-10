import json
from dataclasses import replace
from fractions import Fraction

import pytest
from r8_fakes import FakeModel, observation, program, task, variable
from test_r8_adapters import attempt, query
from test_r8_agent import setup as setup  # noqa: PLC0414 -- explicit pytest fixture re-export
from test_r8_math import store

from qwen3vl_agent.r8 import R8Config, R8VideoAgent
from qwen3vl_agent.r8.adapters import Adapters
from qwen3vl_agent.r8.contracts import parse
from qwen3vl_agent.r8.evaluate import export_variables
from qwen3vl_agent.r8.matcher import match, parse_option
from qwen3vl_agent.r8.operators import calculate as calc
from qwen3vl_agent.r8.query_ir import Executor
from qwen3vl_agent.r8.types import Choice, ProtocolError
from qwen3vl_agent.r8.units import IntervalQuantity
from qwen3vl_agent.r8.units import Quantity as Q


def test_reference_vectors_and_countdown():
    s, packet = store()
    s.append(variable("other", "0.9", "CNY"), packet=packet)
    result = Executor(s, R8Config()).run(
        {
            "nodes": [
                {
                    "id": "average",
                    "op": "mean",
                    "args": [{"refs": ["price", "other"]}],
                    "params": {},
                }
            ],
            "target_node": "average",
            "output_unit": "CNY",
        }
    )
    assert result.value.value == Fraction("10.4")
    assert (
        calc("duration_to_minutes", [calc("parse_clock", ["1:48"], {"kind": "duration"})]).value
        == 108
    )


@pytest.mark.parametrize(
    "op,args,want",
    [
        ("digit_sum", ["1234"], 10),
        ("abs", [Q.make(-7)], 7),
        ("sum", [[Q.make(1), Q.make(2)]], 3),
        ("equal", [Q.make(1), Q.make(1)], True),
        ("less_equal", [Q.make(1), Q.make(2)], True),
        ("greater_equal", [Q.make(1), Q.make(2)], False),
        ("nonzero_denominator", [Q.make(0)], False),
        ("nonnegative", [Q.make(-1)], False),
        ("recompute", [Q.make(2), Q.make(2)], True),
    ],
)
def test_remaining_primitives(op, args, want):
    result = calc(op, args)
    assert (result.value if isinstance(result, Q) else result) == want


def test_all_original_punctuation_and_symbols():
    assert parse_option("2.6.", "1", "number")["value"].value == Fraction("2.6")
    assert parse_option("September 24, 2025.", "date", "date")["value"].isoformat() == "2025-09-24"
    assert parse_option("08:00.", "clock", "clock")["value"].value == 28800
    assert (
        parse_option("Heart-shaped mug ₱179, ice cream-shaped mug ₱249.", "PHP", "tuple")["value"][
            0
        ].unit.label
        == "PHP"
    )
    assert parse_option("11.", "GBP", "number")["value"].value == 11


def test_known_total_unknown_outcomes_propagate_interval():
    s, _ = store()
    adapter = Adapters(s)
    adapter.state["attempts"] = {r["id"]: r for r in [attempt("a", 1), attempt("b", 2, "unknown")]}
    values, _ = adapter.execute([query()], coverage_complete=True)
    graph = {
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
    result = Executor(s, R8Config(), values).run(graph)
    assert isinstance(result.value, IntervalQuantity)
    assert (result.value.lower, result.value.upper) == (50, 100)
    spec = task()
    spec.update(answer_type="percentage", output_unit="%")
    assert match(result.value, [Choice("A", "10%"), Choice("B", "80%")], spec).prediction == "B"
    assert match(result.value, [Choice("A", "50%"), Choice("B", "100%")], spec).prediction is None
    assert match(result.value, [], spec).prediction is None


def test_interval_cannot_use_none_at_boundary():
    spec = task()
    spec.update(output_unit="1", precision={"kind": "exact", "places": None, "question_span": ""})
    value = IntervalQuantity(Fraction(1), Fraction(2))
    assert (
        match(value, [Choice("A", "1"), Choice("B", "None of the others")], spec).prediction is None
    )


def test_joint_candidates_same_option(setup):
    config, request = setup

    def uncertain(payload):
        out = observation(payload)
        out["observations"][0].update(
            raw_text="19.8 or 19.9",
            value="19.8",
            alternatives=["19.9"],
            alternatives_exhaustive=True,
        )
        return out

    result = R8VideoAgent(FakeModel({"observe": uncertain}), config).solve(
        replace(request, mode="D")
    )
    assert result.prediction == "B" and result.status == "verified_at_option_precision"
    assert result.execution["joint_complete"] and result.execution["joint_total"] == 2
    assert all(v["value"] == "19.8" for v in result.variables.values() if v["id"] == "price")


def test_joint_truncation_and_open_alternatives(setup):
    config, request = setup

    def uncertain(payload):
        out = observation(payload)
        out["observations"][0].update(
            raw_text="19.8 or 19.9 or 20.0",
            value="19.8",
            alternatives=["19.9", "20.0"],
            alternatives_exhaustive=True,
        )
        return out

    result = R8VideoAgent(
        FakeModel({"observe": uncertain}), replace(config, max_joint_candidates=2)
    ).solve(replace(request, mode="D"))
    assert result.prediction is None and not result.execution["joint_complete"]

    def incomplete(payload):
        out = uncertain(payload)
        out["observations"][0]["alternatives_exhaustive"] = False
        return out

    result = R8VideoAgent(FakeModel({"observe": incomplete}), config).solve(
        replace(request, mode="D")
    )
    assert not result.verified


def test_directed_repair_actually_uses_crop(setup):
    config, request = setup

    def unreadable(payload):
        out = observation(payload)
        out["observations"][0].update(
            value=None, raw_text="unreadable", unresolved=["small decimal"]
        )
        out["requested_context"] = [
            {
                "reason": "read the small price",
                "frame_id": "F01",
                "bbox": [0.2, 0.2, 0.8, 0.8],
                "window": None,
            }
        ]
        return out

    fake = FakeModel({"observe": unreadable})
    result = R8VideoAgent(fake, config).solve(replace(request, mode="F"))
    assert result.prediction == "B"
    repair = next(c for c in fake.calls if c[0]["stage"] == "reread")
    assert len(repair[0]["input"]["evidence"]) == 2
    assert any(e.get("crop_transform") for e in result.evidence.values())
    assert result.trace["actions"][0]["kind"] == "directed"


def test_uniform_repair_has_different_frames(setup):
    config, request = setup
    called = 0

    def observed(payload):
        nonlocal called
        called += 1
        out = observation(payload)
        if called == 1:
            out["observations"][0].update(
                value=None, raw_text="unreadable", unresolved=["small decimal"]
            )
        return out

    fake = FakeModel({"observe": observed})
    result = R8VideoAgent(fake, config).solve(replace(request, mode="E"))
    assert result.prediction == "B" and result.trace["actions"][0]["kind"] == "uniform"
    packets = [c[0]["input"]["evidence"] for c in fake.calls if c[0]["stage"] == "observe"]
    assert {e["id"] for e in packets[0].values()} != {e["id"] for e in packets[1].values()}


def test_replay_export_and_oracle_gate(setup, tmp_path):
    config, request = setup
    result = R8VideoAgent(FakeModel(), config).solve(replace(request, mode="C"))
    file = tmp_path / "variables.json"
    artifact = export_variables(result, file)
    assert "prediction" not in artifact and "answer" not in artifact
    fake = FakeModel()
    replayed = R8VideoAgent(fake, config).solve(
        replace(request, mode="C", variables_input=str(file), comparison="fixed_evidence")
    )
    assert replayed.prediction == "B"
    assert "observe" not in [c[0]["stage"] for c in fake.calls]
    artifact["kind"] = "oracle_diagnostic"
    file.write_text(json.dumps(artifact), encoding="utf-8")
    with pytest.raises(ProtocolError, match="diagnostic"):
        R8VideoAgent(FakeModel(), config).solve(replace(request, variables_input=str(file)))


def test_schema_unknown_fields_and_duplicate_keys():
    from jsonschema import ValidationError

    with pytest.raises(ProtocolError):
        parse('{"prediction":"A","prediction":"B"}', "fallback")
    out = program()
    out["answer"] = "B"
    with pytest.raises(ValidationError):
        parse(json.dumps(out), "formalize")


def test_record_revisions_invalidate_dependents():
    s, packet = store()
    adapter = Adapters(s)
    out = observation({})
    out["attempts"] = [attempt("a", 1, "unknown")]
    adapter.ingest(out, packet, "first")
    values, _ = adapter.execute([query()], coverage_complete=True)
    parents = values["stats"]["parents"]
    s.add_derived("dependent", 1, parents)
    s.add_derived("independent", 1, ["price@1"])
    out["attempts"][0]["outcome"] = "success"
    adapter.ingest(out, packet, "revision")
    assert (
        not s.state["derived"]["dependent"]["valid"] and s.state["derived"]["independent"]["valid"]
    )
