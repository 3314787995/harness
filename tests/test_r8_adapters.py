from copy import deepcopy
from fractions import Fraction

import pytest
from test_r8_math import store

from qwen3vl_agent.r8.adapters import Adapters
from qwen3vl_agent.r8.types import ModelingError


def attempt(key, start, outcome="success", end=None, replay=None):
    return {
        "id": key,
        "actor_id": "player",
        "scope": "round",
        "round_id": "one",
        "start_time": start,
        "end_time": end if end is not None else start + 0.2,
        "outcome": outcome,
        "attributes": {"touches_cup": True if outcome == "success" else None},
        "evidence_refs": ["frame"],
        "replay_of": replay,
        "duplicate_of": None,
    }


def query():
    return {
        "id": "stats",
        "kind": "attempts",
        "scope": "round",
        "actor": "player",
        "round_id": "one",
        "boundary": "end",
        "window": [0.0, 10.0],
        "attribute": "",
        "operation": "",
        "rule_span": "",
        "complete_claim": True,
        "initial_ref": None,
    }


def test_unknown_outcome_interval_and_replay():
    s, _ = store()
    adapter = Adapters(s)
    adapter.state["attempts"] = {
        r["id"]: r
        for r in [attempt("a", 1), attempt("b", 2, "unknown"), attempt("replay", 5, replay="a")]
    }
    stats = adapter.attempts(query(), True)
    assert stats["total"] == 2 and stats["successes"] == 1 and stats["unknown"] == 1
    assert stats["ratio_interval"] == [Fraction(1, 2), Fraction(1)]
    stats = adapter.attempts(query(), False)
    assert stats["total"] is None and stats["ratio_interval"] is None


def test_missing_denominator_never_fabricated():
    s, _ = store()
    adapter = Adapters(s)
    adapter.state["attempts"]["a"] = attempt("a", 1)
    with pytest.raises(ModelingError, match="missing_denominator"):
        adapter.execute([query()], coverage_complete=False)


def test_open_boundary_and_duplicate_cycle():
    s, _ = store()
    adapter = Adapters(s)
    row = attempt("a", 1)
    row["end_time"] = None
    adapter.state["attempts"]["a"] = row
    assert adapter.attempts(query(), True)["total"] is None
    row["replay_of"] = "a"
    with pytest.raises(ModelingError):
        adapter.attempts(query(), True)


def test_alternative_rule_does_not_modify_history():
    s, _ = store()
    s.question = "If touching the cup counted as success."
    adapter = Adapters(s)
    row = attempt("a", 1, "failure")
    row["attributes"]["touches_cup"] = True
    adapter.state["attempts"]["a"] = row
    q = query()
    q.update(
        kind="replace_rule", attribute="touches_cup", operation="all_true", rule_span=s.question
    )
    before = deepcopy(adapter.state)
    result = adapter.replace_rule(q, True)
    assert result["successes"] == 1 and adapter.state == before


def test_unreadable_inventory_kept():
    s, _ = store()
    adapter = Adapters(s)
    adapter.state["items"] = {
        "p": {
            "id": "p",
            "entity_id": "yogurt",
            "scope": "shopping",
            "snapshot": "receipt",
            "readable": False,
            "attributes": {},
            "evidence_refs": ["frame"],
            "duplicate_of": None,
        }
    }
    q = query()
    q.update(scope="shopping", kind="inventory", attribute="price")
    result = adapter.inventory(q, True)
    assert len(result["rows"]) == 1 and result["unreadable"] == ["p"] and not result["complete"]
