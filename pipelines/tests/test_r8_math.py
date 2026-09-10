from fractions import Fraction

import pytest
from r8_fakes import task, variable

from qwen3vl_agent.r8.config import R8Config
from qwen3vl_agent.r8.evidence import EvidenceStore
from qwen3vl_agent.r8.matcher import match, parse_option
from qwen3vl_agent.r8.operators import calculate as calc
from qwen3vl_agent.r8.query_ir import Executor
from qwen3vl_agent.r8.types import Choice, InputContract, ModelingError, ProtocolError
from qwen3vl_agent.r8.units import Quantity as Q
from qwen3vl_agent.r8.units import number


def store():
    contract = InputContract(((0.0, 10.0),), ("video", "screen_text"), "full", "caller", None)
    result = EvidenceStore("given 19.9 and 6", contract)
    evidence = {
        "F01": {
            "id": "frame",
            "modality": "video",
            "timestamp_seconds": 1.0,
            "scope_hash": contract.fingerprint,
        }
    }
    result.present(evidence)
    result.state["entities"]["yogurt"] = {
        "id": "yogurt",
        "scope": "shopping",
        "snapshot": "receipt",
    }
    result.append(variable(), packet=evidence)
    result.append(variable("count", "6", "cup"), packet=evidence)
    return result, evidence


def test_yogurt_exact():
    result = calc("divide", [Q.make("19.9", "CNY"), Q.make(6, "cup")]).convert("CNY/cup")
    assert result.value == Fraction(199, 60)
    assert (
        match(
            result, [Choice("A", "2"), Choice("B", "3"), Choice("C", "4"), Choice("D", "5")], task()
        ).prediction
        == "B"
    )


@pytest.mark.parametrize(
    "op,a,b,want",
    [
        ("minimum_required_packages", "280", "30", 10),
        ("minimum_required_packages", "0", "30", 0),
        ("maximum_affordable_count", "280", "30", 9),
    ],
)
def test_integer_boundaries(op, a, b, want):
    assert calc(op, [Q.make(a), Q.make(b)]).value == want


@pytest.mark.parametrize(
    "op", ["divide", "ratio", "maximum_affordable_count", "minimum_required_packages"]
)
def test_zero_denominator(op):
    with pytest.raises(ModelingError):
        calc(op, [Q.make(2), Q.make(0)])


@pytest.mark.parametrize("a,b", [("USD", "CNY"), ("m", "g"), ("cup", "jar"), ("CNY/cup", "CNY")])
def test_incompatible(a, b):
    with pytest.raises(ModelingError):
        calc("add", [Q.make(1, a), Q.make(1, b)])


def test_units_and_pricing_basis():
    assert calc("add", [Q.make("1", "kg"), Q.make("500", "g")]).value == Fraction(3, 2)
    with pytest.raises(ModelingError):
        calc("add", [Q.make(1, "CNY", "sticker"), Q.make(1, "CNY", "per_item")])
    assert calc("percentage_point_difference", [Q.make("60", "%"), Q.make("45", "%")]).value == 15
    assert calc("percentage", [Q.make("3/5")]).value == 60


@pytest.mark.parametrize(
    "day,offset,want",
    [
        ("2024-02-28", 1, "2024-02-29"),
        ("2024-02-28", 2, "2024-03-01"),
        ("2025-09-30", 1, "2025-10-01"),
        ("2025-01-01", -1, "2024-12-31"),
    ],
)
def test_calendar(day, offset, want):
    parsed = calc("parse_date", [day])
    assert calc("add_days", [parsed, Q.make(offset, "day")]).isoformat() == want


def test_invalid_date_clock_duration():
    with pytest.raises(ModelingError):
        calc("parse_date", ["2025-09-31"])
    assert calc("duration_to_minutes", [Q.make(90, "s")]).value == Fraction(3, 2)
    clock = calc("parse_clock", ["11:30 PM"])
    assert calc("add_minutes", [clock, Q.make(90, "min")], {"wrap_24h": True}).value == 3600
    with pytest.raises(ModelingError):
        calc("parse_clock", ["25:61"])


def test_nested_object_query_and_ties():
    rows = [
        {"id": "a", "price": Q.make(2, "CNY"), "weight": Q.make(100, "g")},
        {"id": "b", "price": Q.make(2, "CNY"), "weight": Q.make(150, "g")},
        {"id": "c", "price": Q.make(3, "CNY"), "weight": Q.make(100, "g")},
    ]
    tied = calc("argmin", [rows], {"attribute": "price"})
    assert [x["id"] for x in tied] == ["a", "b"]
    weights = calc("select_attribute", [tied], {"attribute": "weight"})
    assert calc("mean", [weights]).value == 125
    assert [
        r["id"]
        for r in calc("filter", [rows, Q.make(100, "g")], {"attribute": "weight", "relation": "eq"})
    ] == ["a", "c"]
    assert calc("deduplicate", [rows + rows], {"key": "id"}) == rows
    assert calc("select_nth", [rows, Q.make(2)], {"index_base": 1})["id"] == "b"
    assert not calc("target_uniqueness", [weights])


def test_executor_provenance_and_local_invalidation():
    s, packet = store()
    graph = {
        "nodes": [
            {"id": "unit_price", "op": "divide", "args": ["price@1", "count@1"], "params": {}}
        ],
        "target_node": "unit_price",
        "output_unit": "CNY/cup",
    }
    out = Executor(s, R8Config()).run(graph)
    assert out.value.value == Fraction(199, 60)
    s.add_derived("only_count", 6, ["count@1"])
    s.append(variable(value="29.9"), packet=packet)
    assert not s.state["derived"]["query:unit_price"]["valid"]
    assert s.state["derived"]["only_count"]["valid"]
    with pytest.raises(ProtocolError):
        Executor(s, R8Config()).run(graph)


@pytest.mark.parametrize(
    "literal",
    [
        {"value": "19.9", "unit": "CNY", "source": "answer:B"},
        {"value": "8", "unit": "1", "source": "definition:two"},
        {"value": "1", "unit": "1"},
    ],
)
def test_unsourced_literals(literal):
    s, _ = store()
    with pytest.raises((ModelingError, ProtocolError)):
        Executor(s, R8Config()).run(
            {
                "nodes": [{"id": "x", "op": "add", "args": [literal, "price"], "params": {}}],
                "target_node": "x",
                "output_unit": "CNY",
            }
        )


@pytest.mark.parametrize("value", ["__import__('os')", "1e999999", "1/0", "NaN", True, "9" * 101])
def test_numeric_input_limits(value):
    with pytest.raises(ModelingError):
        number(value)


def test_matching_boundaries():
    spec = task()
    spec["precision"] = {"kind": "exact", "places": None, "question_span": ""}
    spec["output_unit"] = "1"
    choices = [Choice("C", "2.6"), Choice("G", "2.6")]
    result = match(Q.make("2.6"), choices, spec)
    assert result.prediction == "C" and result.labels == ["C", "G"]
    assert match(Q.make("2.61"), choices, spec).prediction is None
    spec["precision"]["kind"] = "closest"
    assert match(Q.make("2.61"), choices, spec).prediction == "C"
    spec["answer_type"] = "count"
    spec["output_unit"] = "arrow"
    assert match(Q.make(2, "arrow"), [Choice("A", "33.3%")], spec).status == "annotation_anomaly"
    assert parse_option("September 31, 2025", "date", "date")["kind"] == "invalid"


def test_none_is_not_unknown():
    spec = task()
    spec["output_unit"] = "1"
    spec["precision"]["kind"] = "exact"
    assert (
        match(Q.make(3), [Choice("A", "1"), Choice("B", "None of the others")], spec).prediction
        == "B"
    )
    assert (
        match(
            Q.make(3), [Choice("A", "unparsed answer"), Choice("B", "None of the others")], spec
        ).prediction
        is None
    )
