"""Evidence-driven 5.2 contracts; fake models do not establish GPU accuracy."""
import copy
import json
import re
from dataclasses import replace
from pathlib import Path

import pytest

from qwen3vl_agent.r4.collection_contracts import ContractError, check_schema, diagnostics, parse_json, record_schema, validate
from qwen3vl_agent.r4.collection_prompts import build_prompt, example_cases, validate_example
from qwen3vl_agent.r4.controller import CollectionController, public_target
from qwen3vl_agent.r4.inventory import EvidenceStore
from qwen3vl_agent.r4.types import InventorySpec, SetSpec
from r4_v5_fakes import Model, task, card, setup
from test_r4_v5 import refs


def check(p, candidate, state="seen", support="direct"):
    ref = next(r for r, meta in p["catalog"].items() if meta["region"] == "core")
    return {"set": "items", "candidate": candidate, "state": state, "support": support,
            "refs": [ref] if state != "not_seen" else [], "facts": "specific fixture evidence"}


@pytest.mark.parametrize("ns", ["physical_instance", "semantic_category", "text_value", "task_item"])
def test_actual_observer_prompt_has_separate_requirements_and_valid_examples(tmp_path, ns):
    model = Model(task(ns))
    agent, req = setup(tmp_path, model)
    if ns == "task_item":
        # History adapter is covered by the selected history regression.
        t = InventorySpec.from_dict(task(ns)).sets[0]
        p = {"sets": [public_target(t)], "catalog": {}}
        text = build_prompt("discover_candidates", p, [t])
    else:
        agent.solve(req)
        call = next(c for c in model.calls if c["role"] == "discover_candidates")
        p = call["payload"]
        text = call["messages"][0]["content"][-1]["text"]
        t = InventorySpec.from_dict(task(ns)).sets[0]
    assert "requirements" in p["sets"][0] and "conditions" not in p["sets"][0]
    assert "state:seen/not_seen" not in text
    cases = example_cases("discover_candidates", [t])
    blocks = re.findall(r"```json\n(.*?)\n```", text, re.S)
    assert len(blocks) == len(cases)
    for block, (_, target, _) in zip(blocks, cases):
        validate_example("discover_candidates", target, json.loads(block))


def test_presence_prompt_only_checks_and_semantic_demo():
    t = InventorySpec.from_dict(task("semantic_category", "missing_members", candidates=["apple", "pear"])).sets[0]
    payload = {"sets": [public_target(t)], "candidates": {"items": ["apple", "pear"]}, "catalog": {}}
    text = build_prompt("discover_candidates", payload, [t])
    assert "boxes is an array" not in text and "output checks ONLY" in text
    assert "Preparation" in text and "support:direct/related/uncertain" in text
    for block, (_, target, _) in zip(re.findall(r"```json\n(.*?)\n```", text, re.S), example_cases("discover_candidates", [t])):
        validate_example("discover_candidates", target, json.loads(block))


def test_real_outputs_remain_rejected_and_feedback_is_specific():
    data = json.loads((Path(__file__).parent/"fixtures/r4_v5_1_observation_failures.json").read_text(encoding="utf-8"))
    for entry in (data[0], data[2]):
        ns = "physical_instance" if "077" in entry["question"] else "semantic_category"
        target = InventorySpec.from_dict(task(ns)).sets[0]
        row = parse_json(entry["output"])["records"][0]
        with pytest.raises(ContractError) as exc:
            validate(row, record_schema(target), "records[0]")
        feedback = diagnostics(exc.value.errors)
        assert any("Do NOT copy" in e["instruction"] for e in feedback)
        if ns == "physical_instance":
            assert any("1–3" in e["instruction"] for e in feedback)
    row = parse_json(data[-1]["output"])["checks"][0]
    t = SetSpec("activities", "semantic_category", "daily activity", candidates=("Eating a meal.",))
    with pytest.raises(ContractError):
        validate(row, check_schema(t))  # old seen did not assess direct vs related support


def test_observation_recovery_preserves_slot_and_same_media(tmp_path):
    def observe(p, m):
        row = card(p)
        if len(m.calls) == 2:
            row["conditions"]["predicate"] = p["sets"][0]["requirements"]["predicate"]
            row["boxes"] *= 8
        return {"records": [row], "coverage": "complete"}
    model = Model(task(), observe)
    agent, req = setup(tmp_path, model)
    result = agent.solve(req)
    assert result.prediction == "A" and len(model.calls) == 3
    assert len(result.inventory["cards"]) == 1 and not result.inventory["quarantined"]
    first, recovery = model.calls[1:]
    media = lambda c: [p for p in c["messages"][0]["content"] if p["type"] == "image"]
    assert media(first) == media(recovery)
    text = recovery["messages"][0]["content"][-1]["text"]
    assert "Do NOT copy" in text and "Do not erase records" in text and "box_count" in text


def test_empty_recovery_cannot_erase_member(tmp_path):
    def observe(p, m):
        if len(m.calls) == 2:
            row = card(p); row["conditions"]["predicate"] = "visually present"
            return {"records": [row], "coverage": "complete"}
        return {"records": [], "coverage": "complete"}
    model = Model(task(), observe)
    agent, req = setup(tmp_path, model)
    result = agent.solve(req)
    assert len(model.calls) == 3 and result.prediction is None
    assert result.result_status == "execution_failed" and not result.inventory["cards"]
    assert any(e["code"] == "recovery_slot_missing" for e in result.failure["errors"])


def test_only_inapplicable_check_can_be_explicitly_discarded(tmp_path):
    def observe(p, m):
        row = card(p, query_value="apple")
        bad = {"set": "items", "candidate": "fruit", "state": "not_seen", "refs": [], "facts": "unsupported branch"}
        return {"records": [row], "checks": [bad if len(m.calls) == 2 else None], "coverage": "complete"}
    model = Model(task("semantic_category"), observe)
    agent, req = setup(tmp_path, model)
    result = agent.solve(req)
    assert result.prediction == "A" and len(model.calls) == 3
    assert len(result.inventory["cards"]) == 1 and not result.inventory["checks"]
    assert not result.inventory["quarantined"]


@pytest.mark.parametrize("support,uncertainties,expected", [
    ("direct", [], "seen"), ("related", [], "unreadable"),
    ("uncertain", [], "unreadable"), ("direct", ["action not clear"], "unreadable")])
def test_positive_support_contract(support, uncertainties, expected):
    spec = InventorySpec.from_dict(task("semantic_category", "missing_members", candidates=["writing"]))
    store = EvidenceStore(spec); catalog, aliases, tile = refs()
    row = {"set": "items", "candidate": "writing", "state": "seen", "support": support,
           "uncertainties": uncertainties, "refs": ["F1"], "facts": "a pen lies beside paper"}
    key = store.commit_check(row, spec.sets[0], tile, aliases, catalog, "slot", "call")
    assert store.state["checks"][key]["state"] == expected
    assert store.state["checks"][key]["reported_state"] == "seen"


@pytest.mark.parametrize("initial_support", ["direct", "related"])
def test_check_conflict_or_ambiguity_is_reviewed_and_retracted(tmp_path, initial_support):
    def observe(p, m):
        return {"checks": [check(p, "writing"), check(p, "reading", support=initial_support)], "coverage": "complete"}
    def inspect(p, m):
        assert "check_review" in p and "choices" not in p and "existing" not in p
        return {"checks": [check(p, r["candidate"], "not_seen" if r["candidate"] == "reading" else "seen")
                           for r in p["check_review"]], "coverage": "complete"}
    model = Model(task("semantic_category", "missing_members", candidates=["writing", "reading"]), observe, inspect)
    agent, req = setup(tmp_path, model)
    result = agent.solve(replace(req, choices={"A": "writing", "B": "reading"}))
    assert result.prediction == "B", result.to_dict()
    assert len(model.calls) == 3 and result.resources["calls_by_purpose"]["qualification"] == 1
    assert result.resources["calls_by_purpose"]["focused"] == 0
    assert len(result.inventory["checks"]) == 2
    assert result.inventory["check_revisions"]
    assert all(not r.get("needs_review") for r in result.inventory["checks"].values())


def test_gap_recalculation_clears_and_reopens_after_positive_retraction(tmp_path):
    def observe(p, m):
        n = sum(c["role"] == "discover_candidates" for c in m.calls)
        return {"checks": [check(p, c, "not_seen" if c == "writing" else "unreadable" if n == 1 else "seen")
                           for c in p["candidates"]["items"]], "coverage": "complete"}
    model = Model(task("semantic_category", "missing_members", candidates=["writing", "reading"]), observe)
    agent, req = setup(tmp_path, model, duration=16)
    controller = CollectionController(agent, replace(req, choices={"A":"writing","B":"reading"}))
    controller.compile(); controller.store = EvidenceStore(controller.spec); controller.plan()
    a, b = list(controller.work["windows"].values())
    controller.observe(a)
    assert a["status"] == "partial"
    controller.observe(b)
    assert a["status"] == "complete" and not a["candidate_gaps"]
    row = next(r for r in controller.store.state["checks"].values() if r["candidate"] == "reading" and r["state"] == "seen")
    row["needs_review"] = True
    controller.refresh_gaps()
    assert a["status"] == "partial" and a["candidate_gaps"]
    row["needs_review"] = False
    controller.refresh_gaps()
    assert a["status"] == "complete"


def test_review_interrupt_replays_and_updates_once(tmp_path):
    def observe(p, m):
        return {"checks": [check(p, c) for c in p["candidates"]["items"]], "coverage": "complete"}
    def inspect(p, m):
        return {"checks": [check(p, r["candidate"], "not_seen" if r["candidate"] == "reading" else "seen")
                           for r in p["check_review"]], "coverage": "complete"}
    model = Model(task("semantic_category", "missing_members", candidates=["writing", "reading"]), observe, inspect)
    agent, req = setup(tmp_path, model)
    req = replace(req, choices={"A":"writing","B":"reading"}, checkpoint_path=str(tmp_path/"check.jsonl"))
    from qwen3vl_agent.r4.session import CollectionSession
    original = CollectionSession.parse
    def interrupted(self, rec):
        if rec["role"] == "inspect_existing":
            raise KeyboardInterrupt()
        return original(self, rec)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(CollectionSession, "parse", interrupted)
        with pytest.raises(KeyboardInterrupt):
            agent.solve(req)
    calls = len(model.calls)
    result = agent.solve(replace(req, resume=True))
    assert result.prediction == "B" and len(model.calls) == calls
    assert len(result.inventory["checks"]) == 2 and len(result.inventory["check_revisions"]) == 2
    again = agent.solve(replace(req, resume=True))
    assert again.prediction == "B" and len(model.calls) == calls


def test_missing_review_cannot_leave_a_trusted_positive(tmp_path):
    def observe(p, m):
        return {"checks": [check(p, c) for c in p["candidates"]["items"]], "coverage":"complete"}
    model = Model(task("semantic_category", "missing_members", candidates=["writing","reading"]), observe,
                  lambda p,m: {"checks":[], "coverage":"complete"})
    agent, req = setup(tmp_path, model)
    result = agent.solve(replace(req, choices={"A":"writing","B":"reading"}))
    assert result.prediction is None and result.result_status == "execution_failed"
    assert result.resources["calls_by_purpose"]["qualification"] == 1
    assert result.resources["calls_by_purpose"]["recovery"] <= 2
    assert any(r.get("needs_review") for r in result.inventory["checks"].values())


def test_qualification_pool_stays_bounded_for_persistent_conflict(tmp_path):
    def observe(p, m):
        n = sum(c["role"] == "discover_candidates" for c in m.calls) - 1
        names = ["writing", "reading", "drawing"]
        return {"checks": [check(p, c, "seen" if c == names[n] else "not_seen")
                           for c in p["candidates"]["items"]], "coverage":"complete"}
    def inspect(p, m):
        return {"checks": [check(p, r["candidate"]) for r in p["check_review"]], "coverage":"complete"}
    model = Model(task("semantic_category", "missing_members", candidates=["writing","reading","drawing"]), observe, inspect)
    agent, req = setup(tmp_path, model, duration=24)
    result = agent.solve(replace(req, choices={"A":"writing","B":"reading","C":"drawing"}))
    assert result.prediction is None and result.result_status == "budget_exhausted"
    assert result.resources["calls_by_purpose"]["qualification"] == 2
    assert result.resources["calls_by_purpose"]["focused"] == 0
    assert len(model.calls) == 6


def test_applicable_check_cannot_be_removed_with_null(tmp_path):
    def observe(p, m):
        row = check(p, "writing")
        if len(m.calls) == 2:
            row["state"] = "maybe"
            return {"checks":[row], "coverage":"complete"}
        return {"checks":[None], "coverage":"complete"}
    model = Model(task("semantic_category", "missing_members", candidates=["writing"]), observe)
    agent, req = setup(tmp_path, model)
    result = agent.solve(req)
    assert result.prediction is None and result.result_status == "execution_failed"
    assert not result.inventory["checks"] and result.inventory["quarantined"]
    assert len(model.calls) == 3


def test_finite_names_do_not_turn_category_count_into_presence(tmp_path):
    def observe(p, m):
        assert "candidates" not in p
        return {"records":[card(p, query_value="apple")], "coverage":"complete"}
    model = Model(task("semantic_category", candidates=["apple", "pear"]), observe)
    agent, req = setup(tmp_path, model)
    result = agent.solve(req)
    assert result.prediction == "A"
    text = model.calls[1]["messages"][0]["content"][-1]["text"]
    assert "finite_candidate_checks" not in text and "category_mapping" in text
