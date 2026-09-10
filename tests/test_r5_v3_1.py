"""Merge-only regressions: recorded responses, lineage, bounded repair and resume."""
import ast
import hashlib
import json
from copy import deepcopy
from dataclasses import replace
from functools import partial
from pathlib import Path

import pytest

from qwen3vl_agent.models.base import ModelOutput
from qwen3vl_agent.r5 import R5Budget, R5Config, R5Request
from qwen3vl_agent.r5.checkpoint import Checkpoint
from qwen3vl_agent.r5.ledger import FactCardStore
from qwen3vl_agent.r5.prompts import prompt
from qwen3vl_agent.r5.runtime import ModelSession, RunContext
from qwen3vl_agent.r5.synthesis import parse_merge
from qwen3vl_agent.r5.types import ProtocolError
from test_r5 import FakeModel, observation, run, setup

FIXTURES = Path(__file__).parent / "fixtures"
RECORDED = json.loads((FIXTURES / "r5_merge_failures.json").read_text(encoding="utf-8"))


def session_for(model, **kwargs):
    return ModelSession(model, None, R5Config(**kwargs), RunContext(R5Budget()))


def call_merge(session, payload):
    return session.call("merge", payload,
                        parser=partial(parse_merge, units=payload["units"], config=session.config))


def child(units, key="child"):
    return {"id": key, "units": units, "protected": [], "ranges": [], "conflicts": []}


@pytest.mark.parametrize("row", RECORDED, ids=lambda r: r["request_id"])
def test_recorded_merge_accepts_summary_and_preserves_exact_missing_inputs(row):
    model = FakeModel(merge=ModelOutput(row["raw_response"], row["metadata"]))
    session = session_for(model)
    value = call_merge(session, row["payload"]).value
    original = json.loads(row["raw_response"])
    assert value["claims"] == original["claims"]
    assert value["omitted_refs"] == original["omitted_refs"] == []
    assert value["passthrough_refs"] == row["expected_passthrough_refs"]
    assert [c["role"] for c in model.calls] == ["merge"]
    assert not session.context.issues
    assert session.context.calls[0]["merge_validation"]["passthrough_refs"] == row["expected_passthrough_refs"]
    store = FactCardStore()
    store.state["facts"].update(deepcopy(row["input_facts"]))
    node = store.commit_node("recorded", [child(row["payload"]["units"])], value, "recorded-call")
    n = row["expected_claim_count"]
    assert len(node["units"]) == n + len(row["expected_passthrough_refs"])
    assert node["units"][n:] == [u for u in row["payload"]["units"] if u["id"] in row["expected_passthrough_refs"]]
    for ref in node["passthrough_refs"]:
        assert node["passthrough_lineage"][ref] == [ref]
        assert store.state["facts"][ref]["evidence_refs"] == row["input_facts"][ref]["evidence_refs"]
    all_leaves = {f for u in node["units"] for f in store.leaf_ids(u["id"])}
    assert all_leaves == {u["id"] for u in row["payload"]["units"]}


def test_passthrough_uses_input_order_and_id_identity_without_claim_limit_or_false_links():
    units = [{"id": f"f{i}", "statement": "Same wording.", "kind": kind}
             for i, kind in enumerate(["visual_observation", "screen_text", "visual_observation", "utterance"])]
    units += [deepcopy(units[1])]
    data = {"claims": [{"statement": "Summary.", "support_refs": ["f0"]}],
            "omitted_refs": [{"ref_id": "f3", "reason": "Explicit model omission."}],
            "passthrough_refs": ["forged"]}
    value = parse_merge(data, units, R5Config(max_claims=1))
    assert value["passthrough_refs"] == ["f1", "f2"]
    store = FactCardStore()
    store.state["facts"].update({u["id"]: u for u in units})
    node = store.commit_node("k", [child(units)], value, "call")
    assert len(node["units"]) == 3
    assert node["units"][1:] == units[1:3]
    assert store.leaf_ids(node["units"][0]["id"]) == ["f0"]
    assert node["conflicts"] == []


@pytest.mark.parametrize("bad,error", [
    ({"claims": [None]}, r"claims\[0\] must be an object"),
    ({"claims": [{"statement": "S", "support_refs": ["ghost"]}]}, "unknown input IDs"),
    ({"claims": [{"statement": "", "support_refs": ["f0"]}]}, "statement"),
    ({"claims": [{"statement": "S", "support_refs": []}]}, "needs an input reference"),
    ({"claims": [{"statement": "S", "support_refs": ["f0"]}], "omitted_refs": [None]}, "omitted_refs"),
])
def test_other_merge_protocol_errors_remain_errors(bad, error):
    with pytest.raises(ProtocolError, match=error):
        parse_merge(bad, [{"id": "f0"}, {"id": "f1"}], R5Config())


def first_only(p):
    unit = p["units"][0]
    return {"claims": [{"statement": unit["statement"], "support_refs": [unit["id"]]}]}


@pytest.mark.parametrize("gap", [False, True])
def test_multilevel_passthrough_reaches_composer_without_inflating_evidence(setup, gap):
    def observe(p):
        value = observation()(p)
        if gap and p["core"][0] == 2:
            value.update(truncated=True, unresolved=["known observation gap"])
        return value

    def compose(p):
        unit = p["units"][-1]
        return {"prediction": "D", "claims": [{"statement": unit["statement"], "support_refs": [unit["id"]]}]}

    result, model, _ = run(setup, FakeModel(observe=observe, merge=first_only, compose=compose),
                           duration=36, choices={"A": "first", "D": "last"})
    assert result.prediction == "D" and result.trace["tree_depth"] == 3
    store = FactCardStore(result.trace["fact_store"])
    selected = next(c for c in model.calls if c["role"] == "compose")["payload"]["units"][-1]["id"]
    expected = {r for f in store.leaf_ids(selected) for r in store.state["facts"][f]["evidence_refs"]}
    assert set(result.evidence_refs) == expected
    assert len(store.state["facts"]) > len(store.leaf_ids(selected))
    assert all(n["passthrough_refs"] for n in store.state["nodes"].values())
    assert any(ref.startswith("claim:") for n in store.state["nodes"].values() for ref in n["passthrough_refs"])
    assert result.coverage["complete"] is not gap
    assert result.completion_state == ("partial" if gap else "complete")
    assert result.verification_status == "not_performed"
    assert not any(c["role"] in {"repair", "audit"} for c in model.calls)
    assert not any("merge_incomplete" in issue for issue in result.unresolved_items)


@pytest.mark.parametrize("field", ["support_refs", "omitted_refs"])
def test_merge_repair_can_use_supplied_id_absent_from_original_output(field):
    payload = {"units": [{"id": "f0", "statement": "Input zero.", "kind": "visual_observation"},
                         {"id": "f1", "statement": "Input one.", "kind": "screen_text"}]}
    raw = {"claims": [{"statement": "Existing summary.", "support_refs": ["ghost" if field == "support_refs" else "f0"]}]}
    if field == "omitted_refs":
        raw["omitted_refs"] = [{"ref_id": "ghost", "reason": "Already stated reason."}]
    repaired = deepcopy(raw)
    if field == "support_refs":
        repaired["claims"][0]["support_refs"] = ["f1"]
    else:
        repaired["omitted_refs"][0]["ref_id"] = "f1"
    model = FakeModel(merge=raw, repair=repaired)
    session = session_for(model)
    value = call_merge(session, payload).value
    assert [c["role"] for c in model.calls] == ["merge", "repair"]
    repair = model.calls[1]["payload"]
    assert repair["input_units"] == payload["units"]
    assert "ghost" not in repair["allowed_references"]
    assert field in repair["error"] and session.context.calls[0]["repair_reason"] == repair["error"]
    assert "Return the Merge object itself" in prompt("repair", repair)
    assert value["passthrough_refs"] == (["f0"] if field == "support_refs" else [])


@pytest.mark.parametrize("field,error", [("statement", "factual statement"),
                                         ("reference", "evidence reference"),
                                         ("reason", "omission reason"),
                                         ("conflicts", "semantic conclusion")])
def test_merge_repair_does_not_invent_material_and_is_not_retried(field, error):
    raw = {"claims": [{"statement": "Existing.", "support_refs": ["ghost"]}]}
    fixed = {"claims": [{"statement": "Existing.", "support_refs": ["f0"]}]}
    if field == "statement":
        fixed["claims"][0]["statement"] = "Invented new fact."
    elif field == "reference":
        fixed["claims"][0]["support_refs"] = ["hypothesis:new"]
    elif field == "reason":
        fixed["omitted_refs"] = [{"ref_id": "f0", "reason": "Invented duplicate reason."}]
    else:
        fixed["conflicts"] = ["Invented conflict."]
    session = session_for(FakeModel(merge=raw, repair=fixed))
    payload = {"units": [{"id": "f0", "statement": "Input."}]}
    for _ in range(2):
        with pytest.raises(ProtocolError, match=error):
            call_merge(session, payload)
    assert [c["role"] for c in session.model.calls] == ["merge", "repair"]
    assert session.context.calls[0]["validation_error"]


def test_carry_capacity_failure_records_explicit_fallback(setup, monkeypatch):
    original = ModelSession.fits

    def fits(self, role, payload):
        if role == "merge" and any(c.startswith("node:") for c in payload["child_ids"]):
            return False  # Inject the existing context-capacity signal at the next level.
        return original(self, role, payload)

    monkeypatch.setattr(ModelSession, "fits", fits)
    result, model, _ = run(setup, FakeModel(merge=first_only), duration=16)
    assert result.prediction and result.completion_state == "partial"
    assert any("cannot_pack_two_summary_nodes" in issue for issue in result.unresolved_items)
    assert sum(c["role"] == "merge" for c in model.calls) == 2
    assert not any(c["role"] == "repair" for c in model.calls)
    compose = next(c for c in model.calls if c["role"] == "compose")["payload"]
    assert {u["id"] for u in compose["units"]} == set(result.trace["fact_store"]["facts"])


@pytest.mark.parametrize("stage", ["merge_returned", "merge_committed", "repair_returned"])
def test_passthrough_resume_preserves_receipts_lineage_and_budget(setup, tmp_path, monkeypatch, stage):
    original_save, interrupted = Checkpoint.save, False
    original_invoke = ModelSession.invoke

    def invoke(self, role, *args, **kwargs):
        if role == "merge":
            self.context.elapsed_sec += 17.0
        return original_invoke(self, role, *args, **kwargs)

    def save(self, value):
        nonlocal interrupted
        calls = value["context"]["calls"]
        hit = bool(value["work"]["store"]["nodes"]) if stage == "merge_committed" else (
            bool(calls) and calls[-1]["role"] == stage.split("_")[0] and calls[-1]["status"] == "returned")
        if hit and not interrupted:
            original_save(self, value)
            interrupted = True
            raise KeyboardInterrupt()
        original_save(self, value)

    monkeypatch.setattr(Checkpoint, "save", save)
    monkeypatch.setattr(ModelSession, "invoke", invoke)
    handlers = {"merge": first_only}
    if stage == "repair_returned":
        handlers = {"merge": {"claims": [{"statement": "Existing.", "support_refs": ["ghost"]}]},
                    "repair": lambda p: {"claims": [{"statement": "Existing.", "support_refs": [p["input_units"][0]["id"]]}]}}
    agent, model, _, video = setup(FakeModel(**handlers))
    req = R5Request(video_path=video, question="Summarize.", checkpoint_path=str(tmp_path / "resume.jsonl"),
                    budget=R5Budget(max_model_calls=8))
    with pytest.raises(KeyboardInterrupt):
        agent.solve(req)
    result = agent.solve(replace(req, resume=True))
    assert sum(c["role"] == "merge" for c in model.calls) == 1
    assert sum(c["role"] == "observe" for c in model.calls) == 3
    assert sum(c["role"] == "repair" for c in model.calls) == int(stage == "repair_returned")
    assert result.resources["model_calls"] == len(model.calls)
    assert result.resources["elapsed_sec"] >= 17
    assert result.resources["limits"]["max_model_calls"] == 8
    assert all(n["passthrough_refs"] for n in result.trace["fact_store"]["nodes"].values())
    previous = len(model.calls)
    assert agent.solve(replace(req, resume=True)).to_dict() == result.to_dict()
    assert len(model.calls) == previous


def test_unaffected_v3_inputs_configuration_and_budgets_are_frozen():
    root = Path(__file__).resolve().parents[1]
    contract = json.loads((FIXTURES / "r5_v3_frozen_contract.json").read_text(encoding="utf-8"))
    sha = lambda data: hashlib.sha256(data).hexdigest()
    for relative, expected in contract["files"].items():
        # Shared model/runner modules have since changed for other current R
        # pipelines. This historical freeze only governs R5-owned code/config;
        # current shared interfaces are exercised by the release regressions.
        if not (relative.startswith("qwen3vl_agent/r5/") or relative == "configs/r5_8b.yaml"):
            continue
        assert sha((root / relative).read_bytes()) == expected, relative
    for case in contract["prompt_cases"]:
        assert sha(prompt(case["role"], case["payload"]).encode()) == case["sha256"]
    for case in contract["functions"]:
        tree = ast.parse((root / case["path"]).read_text(encoding="utf-8"))
        node = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == case["function"])
        assert sha(ast.dump(node).encode()) == case["ast_sha256"], case["function"]
