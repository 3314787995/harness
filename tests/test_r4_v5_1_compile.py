"""Compiler contract regression tests, not measurements of model accuracy."""
import copy
import json
import re
from dataclasses import replace
from pathlib import Path

import pytest

from qwen3vl_agent.r4 import R4Request
from qwen3vl_agent.r4.collection_contracts import (
    ContractError, UNIT_EQUIVALENCE, compile_diagnostics, compile_schema,
    parse_compile, parse_compile_json,
)
from qwen3vl_agent.r4.collection_prompts import (
    build_prompt, compile_example_cases, validate_compile_example,
)
from qwen3vl_agent.r4.controller import executor
from qwen3vl_agent.r4.types import ProtocolError
from r4_v5_fakes import Model, card, setup, task


FIXTURE = json.loads((Path(__file__).parent / "fixtures/r4_v5_compile_failures.json").read_text(encoding="utf-8"))
REPLAY = [(q, call) for q in FIXTURE["cases"] for call in q["calls"]]


def corrected(q):
    compiled = task(q["expected_namespace"])
    member = compiled["sets"][0]
    member.update(target=q["target"], count_unit=q["count_unit"], predicate="visually present")
    member.pop("equivalence")
    compiled.pop("version")
    if q["id"].endswith("251-1"):
        compiled["operations"][0]["op"] = "missing_members"
        member["candidates"] = [c["text"] for c in q["choices"]]
    return compiled


def request_for(q):
    return R4Request(q["question"], video_path="fixture.mp4", choices=q["choices"])


def test_examples_in_actual_model_messages_pass_real_acceptance(tmp_path):
    compiled = task()
    compiled["sets"][0].pop("equivalence")
    model = Model(compiled, lambda p, m: {"records": [card(p)], "coverage": "complete"})
    agent, request = setup(tmp_path, model)
    result = agent.solve(request)
    assert result.result_status == "supported", result.to_dict()
    prompt = model.calls[0]["messages"][0]["content"][-1]["text"]
    blocks = re.findall(r"```json\n(.*?)\n```", prompt, re.S)
    cases = compile_example_cases()
    assert len(blocks) == len(cases) == 9
    for raw, case in zip(blocks, cases):
        actual = parse_compile_json(raw)
        assert actual == case["response"]
        validate_compile_example({**case, "response": actual})
        spec = parse_compile(actual, R4Request(case["question"], video_path="format-demo.mp4"))
        assert spec.sets[0].namespace == case["expected"]["namespace"]
    assert "$.sets[].scope.kind" in prompt and "$.scope.kind" in prompt
    assert "same evidence" not in prompt and "committed slots" not in prompt
    assert "optional_membership" not in prompt
    assert len(prompt) < agent.config.budget.max_text_chars_per_call
    assert [c["kwargs"]["max_new_tokens"] for c in model.calls] == [512, 1024]
    assert result.resources["calls"][0]["default_assignments"]


@pytest.mark.parametrize("broken", ["scope", "semantic_target"])
def test_bad_program_example_stops_before_any_model_call(tmp_path, monkeypatch, broken):
    import qwen3vl_agent.r4.collection_prompts as prompts
    cases = compile_example_cases()
    if broken == "scope":
        cases[0]["response"]["scope"] = "full"
    else:
        cases[1]["response"]["sets"][0]["target"] = "triangle"
    monkeypatch.setattr(prompts, "compile_example_cases", lambda: cases)
    model = Model(task())
    agent, req = setup(tmp_path, model)
    result = agent.solve(req)
    assert not model.calls
    assert result.prediction is None and result.failure["stage"] == "compile_prompt"
    assert result.failure["code"] == "prompt_contract_error"
    assert result.resources["calls_by_purpose"]["recovery"] == 0


@pytest.mark.parametrize("namespace", list(UNIT_EQUIVALENCE))
@pytest.mark.parametrize("supplied", [False, True], ids=["omitted", "auto"])
def test_unit_defaults_are_program_owned_and_input_is_unchanged(namespace, supplied):
    raw = task(namespace)
    raw.pop("version")
    if supplied:
        raw["sets"][0]["equivalence"] = "auto"
    else:
        raw["sets"][0].pop("equivalence")
    before = copy.deepcopy(raw)
    assignments = []
    spec = parse_compile(raw, R4Request("fixture", video_path="v"), assignments=assignments)
    assert raw == before
    assert spec.version == 5 and spec.output_id == "answer"
    assert spec.sets[0].equivalence == UNIT_EQUIVALENCE[namespace]
    assert spec.sets[0].scope == {} and spec.scope == {"kind": "full"}
    entry = next(a for a in assignments if a["path"] == "$.sets[0].equivalence")
    assert entry["value"] == UNIT_EQUIVALENCE[namespace]
    assert entry["was_present"] == supplied
    assert entry["reason"] == "namespace_equivalence"


@pytest.mark.parametrize("path,value", [
    (("sets", 0, "equivalence"), "entity"),
    (("sets", 0, "normalization"), None),
    (("sets", 0, "membership"), None),
    (("sets", 0, "scope"), "full"),
    (("unresolved",), True),
    (("unresolved",), None),
])
def test_explicit_wrong_values_are_not_silently_defaulted(path, value):
    data = task("semantic_category")
    node = data
    for part in path[:-1]:
        node = node[part]
    node[path[-1]] = value
    before = copy.deepcopy(data)
    with pytest.raises(ContractError):
        parse_compile(data, R4Request("types?", video_path="v"))
    assert data == before


def test_combination_scope_and_cross_set_units_keep_their_contracts():
    raw = task("semantic_category", equivalence="combination", attribute_keys=["fruit", "color"])
    spec = parse_compile(raw, R4Request("combinations?", video_path="v"))
    assert spec.sets[0].equivalence == "combination"
    raw["sets"][0]["attribute_keys"] = []
    with pytest.raises(ContractError, match="dimensions"):
        parse_compile(raw, R4Request("combinations?", video_path="v"))
    raw = task(scope={"kind": "frame", "timestamp_sec": 12})
    raw["sets"][0].pop("equivalence")
    spec = parse_compile(raw, R4Request("at 12s?", video_path="v"))
    assert executor(spec.sets[0], spec.scope, spec.operations) == "local_instances"
    raw["sets"].append({**raw["sets"][0], "set_id": "others", "count_unit": "incompatible unit"})
    raw["operations"] = [{"operation_id": "u", "op": "UNION", "inputs": ["items", "others"]}]
    with pytest.raises(ProtocolError, match="consistent member units"):
        parse_compile(raw, R4Request("combined?", video_path="v"))


@pytest.mark.parametrize("q,call", REPLAY, ids=[f"{q['id']}-{i+1}" for i, (q, _) in enumerate(REPLAY)])
def test_six_real_bad_outputs_rejected_with_actionable_feedback(q, call):
    with pytest.raises(ContractError) as caught:
        parse_compile(parse_compile_json(call["raw_response"]), request_for(q))
    feedback = compile_diagnostics([{**e, "stage": "compile", "call_id": "compile:0"} for e in caught.value.errors])
    encoded = json.dumps(feedback)
    assert len(feedback) <= 8 and all(f["call_id"] == "compile:0" and f["stage"] == "compile" for f in feedback)
    assert "scope is an object using kind" in encoded
    assert "optional_membership" in encoded
    assert "unit_equivalence_conflict" in encoded
    if q["id"].endswith("225-1"):
        assert "Remove the r4.compile wrapper" in encoded
        codes = {f["code"] for f in feedback if f["path"] == "$"}
        assert {"schema_required", "schema_additionalProperties"} <= codes
    if q["id"].endswith("251-1"):
        assert "duplicate_json_field" in encoded and "Keep exactly one 'version'" in encoded


@pytest.mark.parametrize("q", FIXTURE["cases"], ids=[q["id"] for q in FIXTURE["cases"]])
def test_one_compile_recovery_preserves_question_and_then_observes(tmp_path, q):
    good = corrected(q)
    model = Model(good, compile_outputs=[q["calls"][0]["raw_response"], good])
    agent, req = setup(tmp_path, model)
    req = replace(req, question=q["question"], choices=request_for(q).choices)
    result = agent.solve(req)
    assert [c["role"] for c in model.calls[:3]] == ["compile", "compile", "discover_candidates"], result.to_dict()
    assert len([c for c in model.calls if c["role"] == "compile"]) == 2
    correction = model.calls[1]["messages"][0]["content"][-1]["text"]
    assert q["question"] in correction and "COMPILE_CORRECTION" in correction
    assert all(q["question"] == c["payload"]["question"] for c in model.calls[:2])
    assert "committed slots" not in correction and "same evidence" not in correction
    assert "FORMAT DEMO distinct_tools" in correction
    assert not any(secret in correction for secret in ("answer_label", "selection_review", "original_record"))
    # The word 'correct' occurs in instructions; no grading fields appear in the structured input.
    assert not ({"answer_label", "correct", "selection_review"} & model.calls[1]["payload"].keys())
    compiled = result.trace["compiled_task"]
    assert compiled["sets"][0]["namespace"] == q["expected_namespace"]
    first, second = result.resources["calls"][:2]
    assert first["validation"] == "invalid" and "accepted_task" not in first
    assert second["validation"] == "valid" and second["default_assignments"]
    assert result.resources["calls_by_purpose"]["recovery"] == 1


@pytest.mark.parametrize("q", FIXTURE["cases"], ids=[q["id"] for q in FIXTURE["cases"]])
def test_repeated_compile_failure_stops_at_two_and_keeps_both_errors(tmp_path, q):
    bad = q["calls"][0]["raw_response"]
    model = Model(corrected(q), compile_outputs=[bad, bad])
    agent, req = setup(tmp_path, model, request_kwargs={"checkpoint_path": str(tmp_path / "state.jsonl")})
    result = agent.solve(replace(req, question=q["question"]))
    assert len(model.calls) == 2 and all(c["role"] == "compile" for c in model.calls)
    assert result.prediction is None and result.result_status == "execution_failed"
    assert result.failure["repeated_invalid_output"] is True
    assert {e["call_id"] for e in result.failure["errors"]} == {"compile:0", "compile:1"}
    assert not result.inventory and result.trace["compiled_task"] is None
    assert all(c["validation_errors"] and c["model_feedback"] for c in result.resources["calls"])
    before = len(model.calls)
    again = agent.solve(replace(req, question=q["question"], resume=True))
    assert again.prediction is None and len(model.calls) == before


def test_complete_fences_duplicate_paths_and_truncation_are_distinct(tmp_path):
    raw = json.dumps(task())
    assert parse_compile_json(" \n```json\n" + raw + "\n```\n") == task()
    with pytest.raises(ContractError) as caught:
        parse_compile_json('{"sets":[{"namespace":"physical_instance","namespace":"semantic_category"}]}')
    assert caught.value.errors[0]["path"] == "$.sets[0].namespace"
    from qwen3vl_agent.models.base import ModelOutput
    model = Model(task(), compile_outputs=[ModelOutput(raw, {"output_tokens": 512}), ModelOutput(raw, {"finish_reason": "length"})])
    agent, req = setup(tmp_path, model)
    result = agent.solve(req)
    assert result.failure["code"] == "compile_recovery_failed" and len(model.calls) == 2
    assert all(e["code"] == "response_truncated" for e in result.failure["errors"])
    assert not result.inventory


def test_returned_compile_replays_once_and_protocol_change_is_rejected(tmp_path, monkeypatch):
    import qwen3vl_agent.r4.controller as controller
    model = Model(task(), lambda p, m: {"records": [card(p)], "coverage": "complete"})
    agent, req = setup(tmp_path, model, request_kwargs={"checkpoint_path": str(tmp_path / "state.jsonl")})
    real = controller.parse_compile
    monkeypatch.setattr(controller, "parse_compile", lambda *a, **kw: (_ for _ in ()).throw(KeyboardInterrupt()))
    with pytest.raises(KeyboardInterrupt):
        agent.solve(req)
    assert len(model.calls) == 1
    monkeypatch.setattr(controller, "parse_compile", real)
    result = agent.solve(replace(req, resume=True))
    assert result.result_status == "supported" and len(model.calls) == 2
    assert len([c for c in model.calls if c["role"] == "compile"]) == 1
    monkeypatch.setattr(controller, "VERSION", "r4-inventory-v5")
    with pytest.raises(ValueError, match="mismatch"):
        agent.solve(replace(req, resume=True))


def test_interrupted_compile_recovery_retains_its_charge(tmp_path):
    model = Model(task(), compile_outputs=["{}", KeyboardInterrupt()])
    agent, req = setup(tmp_path, model, request_kwargs={"checkpoint_path": str(tmp_path / "state.jsonl")})
    with pytest.raises(KeyboardInterrupt):
        agent.solve(req)
    result = agent.solve(replace(req, resume=True))
    assert len(model.calls) == 2 and result.result_status == "execution_failed"
    assert result.resources["calls_by_purpose"]["recovery"] == 1
