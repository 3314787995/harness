import pytest
from r8_fakes import observation, variable
from test_r8_adapters import query
from test_r8_math import store

from qwen3vl_agent.r8.adapters import Adapters
from qwen3vl_agent.r8.config import R8Config
from qwen3vl_agent.r8.matcher import match
from qwen3vl_agent.r8.query_ir import Executor
from qwen3vl_agent.r8.types import Choice, ModelingError, ProtocolError


def accounting():
    s, packet = store()
    for entity in ("shop", "bank", "supplier", "customer"):
        s.state["entities"][entity] = {"id": entity, "scope": "shopping", "snapshot": "receipt"}
    for key, value, role in (
        ("loan", "100", "amount"),
        ("purchase", "30", "amount"),
        ("sale", "50", "amount"),
        ("opening", "200", "opening_balance"),
        ("debt", "0", "opening_liability"),
        ("cost", "30", "sold_goods_cost_basis"),
    ):
        row = variable(key, value, "CNY")
        row.update(entity_id="shop", role=role)
        s.append(row, packet=packet)
    rows = []
    for key, payer, payee, amount, nature, time in (
        ("t1", "bank", "shop", "loan", "loan", 1.0),
        ("t2", "shop", "supplier", "purchase", "purchase", 2.0),
        ("t3", "customer", "shop", "sale", "sale", 3.0),
        ("t4", "shop", "bank", "loan", "repayment", 4.0),
    ):
        rows.append(
            {
                "id": key,
                "payer": payer,
                "payee": payee,
                "amount_ref": amount,
                "scope": "shopping",
                "snapshot": "receipt",
                "nature": nature,
                "stage": "stage",
                "time": time,
                "evidence_refs": ["frame"],
                "duplicate_of": None,
            }
        )
    adapter = Adapters(s)
    out = observation({})
    out["transactions"] = rows
    adapter.ingest(out, packet, "ledger")
    q = query()
    q.update(kind="transactions", scope="shopping", actor="shop", round_id="", operation="cashflow")
    return s, adapter, q


@pytest.mark.parametrize(
    "operation,initial,want",
    [
        ("cashflow", None, 20),
        ("balance", "opening", 220),
        ("liability", "debt", 0),
        ("realized_profit", "cost", 20),
    ],
)
def test_accounting_bases(operation, initial, want):
    _s, adapter, q = accounting()
    q.update(operation=operation, initial_ref=initial)
    result = adapter.transactions(q, True)
    assert result["value"].value == want and result["accounting_basis"] == operation


def test_cost_not_inferred_and_currency_not_converted():
    s, adapter, q = accounting()
    q["operation"] = "realized_profit"
    with pytest.raises(ModelingError, match="cost basis"):
        adapter.transactions(q, True)
    q.update(operation="cashflow", window=[0.0, 2.5])
    assert adapter.transactions(q, True)["value"].value == 70
    s.state["variables"][s.state["current"]["sale"]]["unit"] = "USD"
    q["window"] = None
    with pytest.raises(ModelingError, match="units"):
        adapter.transactions(q, True)


def test_multi_stage_tuple_query_retains_order():
    s, adapter, q = accounting()
    one = {**q, "id": "early", "window": [0.0, 2.5]}
    two = {**q, "id": "late", "window": [2.5, 5.0]}
    imported, _ = adapter.execute([one, two], coverage_complete=True)
    result = Executor(s, R8Config(), imported).run(
        {"nodes": [], "target_node": {"refs": ["early", "late"]}, "output_unit": "CNY"}
    )
    assert [v.value for v in result.value] == [70, -50]
    spec = {
        "answer_type": "tuple",
        "output_unit": "CNY",
        "precision": {"kind": "exact", "places": None},
    }
    assert (
        match(result.value, [Choice("A", "70, -50"), Choice("B", "-50, 70")], spec).prediction
        == "A"
    )


def test_stipulated_value_cannot_overwrite_observation():
    s, _packet = store()
    given = {
        "id": "price",
        "entity_id": "yogurt",
        "attribute": "price",
        "role": "total_price",
        "scope": "shopping",
        "snapshot": "receipt",
        "value": "19.9",
        "unit": "CNY",
        "unit_basis": "",
        "question_span": "19.9",
    }
    with pytest.raises(ProtocolError, match="separate IDs"):
        s.append(given, origin="hypothetical")
