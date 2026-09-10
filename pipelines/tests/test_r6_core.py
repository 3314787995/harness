"""Question-shaped mechanism regressions; synthetic evidence is not benchmark validation."""

from copy import deepcopy
from dataclasses import asdict, replace

import pytest
from r6_fakes import FakeMedia, FakeModel, assessment, fact, query, relation, request

from qwen3vl_agent.r6 import R6Config, R6VideoAgent
from qwen3vl_agent.r6.acquisition import action, select_action
from qwen3vl_agent.r6.controller import gap
from qwen3vl_agent.r6.evaluate import public_request
from qwen3vl_agent.r6.logic import evaluate_expression, exact_set_status
from qwen3vl_agent.r6.schema import parse, validate_observation, validate_query
from qwen3vl_agent.r6.state import add_observation, evaluate_assessment, new_state, relation_sources
from qwen3vl_agent.r6.types import InputContract, ProtocolError


def ledger():
    state = new_state()
    sources = {"F01": {"id": "s", "modality": "video", "source_time": [1, 1]}}
    add_observation(state, {"records": [fact()], "gaps": []}, sources, job_key="one")
    return state


@pytest.mark.parametrize(
    "true,unknown,selection,expected",
    [
        ({2}, {3}, {2}, "unknown"),
        ({2}, {3}, {2, 3}, "unknown"),
        ({2, 3}, set(), {2, 3}, "supported"),
        ({2, 3}, set(), {2}, "contradicted"),
        ({2}, set(), {2, 5}, "contradicted"),
    ],
)
def test_e08_three_valued_exact_set(true, unknown, selection, expected):
    states = {
        i: "supported" if i in true else "unknown" if i in unknown else "contradicted"
        for i in (1, 2, 3)
    }
    assert exact_set_status(selection, states, universe_complete=True) == expected
    assert exact_set_status(selection, states, universe_complete=False) == "unknown"


def test_unknown_negation_and_none_of_are_not_absence():
    a = {"op": "atom", "id": "a"}
    assert evaluate_expression({"op": "not", "args": [a]}, {}) == "unknown"
    assert evaluate_expression({"op": "none_of", "args": [a]}, {"a": "unknown"}) == "unknown"
    assert evaluate_expression({"op": "none_of", "args": [a]}, {"a": "contradicted"}) == "supported"


def test_e14_true_event_can_be_partial_target_fit():
    state, spec = ledger(), query()
    value = assessment(spec, state["facts"], both_true=True)
    result = evaluate_assessment(value, spec, state["facts"], {}, state["sources"])
    assert result["candidates"][1]["factual_status"] == "supported"
    assert result["candidates"][1]["answer_target_fit"] == "partial"
    assert result["preferred_label"] == "A"


def test_e10_later_relation_cannot_replace_earlier_relation():
    state, spec = ledger(), query(relation_type="reveals")
    for atom in spec["atoms"]:
        atom["story_time"] = [0, 2]
    state["facts"]["F000001"]["story_time"] = [9, 10]
    value = assessment(spec, state["facts"], relation_rows=[relation(story_time=[9, 10])])
    result = evaluate_assessment(value, spec, state["facts"], {}, state["sources"])
    assert result["candidates"][0]["factual_status"] == "unknown"


def test_e09_visual_cannot_support_musical_atom():
    state = ledger()
    spec = query(required_modalities=["video", "audio"])
    result = evaluate_assessment(
        assessment(spec, state["facts"]), spec, state["facts"], {}, state["sources"]
    )
    assert all(c["factual_status"] == "unknown" for c in result["candidates"])


def test_cycles_and_fabricated_references_rejected():
    state = ledger()
    r = relation(parents=["r1"])
    with pytest.raises(ProtocolError, match="cyclic"):
        evaluate_assessment(
            assessment(query(), state["facts"], relation_rows=[r]),
            query(),
            state["facts"],
            {},
            state["sources"],
        )
    with pytest.raises(ProtocolError):
        relation_sources("missing", state["facts"], {})
    value = {"records": [fact("invented")], "gaps": [], "overflow": False}
    with pytest.raises(ProtocolError, match="not shown"):
        validate_observation(value, {"F01": state["sources"]["s"]}, query())


def test_same_source_rephrasing_not_independent_and_versions_preserved():
    state = ledger()
    source = {"F01": state["sources"]["s"]}
    add_observation(state, {"records": [fact()], "gaps": []}, source, job_key="two")
    assert len(state["facts"]) == 1
    conflicting = fact()
    conflicting["predicate"] = "The animal is a dog"
    add_observation(state, {"records": [conflicting], "gaps": []}, source, job_key="three")
    assert len(state["facts"]) == 2 and len(state["sources"]) == 1


@pytest.mark.parametrize("field", ["answer", "gold", "time_reference", "other_question_answer"])
def test_public_request_rejects_gold(tmp_path, field):
    data = asdict(request(tmp_path)) | {field: "secret"}
    with pytest.raises(ProtocolError, match="offline"):
        public_request(data)


def test_compiler_cannot_omit_choice_or_widen_access(tmp_path):
    item = request(tmp_path, allowed_intervals=[(0, 2)], reference_scope=(0, 1))
    contract = InputContract.resolve(item, 4, "hash")
    spec = query()
    with pytest.raises(ProtocolError, match="reference"):
        validate_query(spec, item, contract)
    spec["reference_interval"] = [0, 1]
    spec["initial_actions"] = [action("observe_clip", span=(0, 3))]
    with pytest.raises(ProtocolError, match="illegal"):
        validate_query(spec, item, contract)
    spec["initial_actions"] = []
    spec["option_claims"].pop()
    with pytest.raises(ProtocolError, match="all labels"):
        validate_query(spec, item, contract)


def test_illegal_and_repeated_actions_are_filtered(tmp_path):
    item = request(tmp_path, allowed_intervals=[(0, 2)])
    contract = InputContract.resolve(item, 4, "hash")
    state = new_state()
    invalid = action("observe_clip", span=(1, 3))
    legal = action("observe_clip", span=(0, 1))
    chosen = select_action([invalid, legal], state, contract, R6Config())
    assert chosen == legal
    from qwen3vl_agent.r6.acquisition import action_identity

    state["actions"].append({"identity": action_identity(legal)})
    assert select_action([legal], state, contract, R6Config()) is None


@pytest.mark.parametrize("scenario", ["E02", "E04", "E12"])
def test_direct_evidence_path_checks_full_proposition(tmp_path, scenario):
    model = FakeModel()
    result = R6VideoAgent(model, media_factory=FakeMedia).solve(
        request(tmp_path, request_id=scenario)
    )
    assert result.prediction == "A" and result.stop_reason == "EVIDENCE_SUFFICIENT"
    assert not result.forced_choice
    assert result.costs["model_calls"] == 4
    assert [c["role"] for c in model.calls] == [
        "compiler",
        "observer",
        "relation_checker",
        "verifier",
    ]
    assert result.trace["relations"] == {}


@pytest.mark.parametrize("scenario", ["E05", "E11"])
def test_ambiguous_input_is_not_reinterpreted_to_match_answer(tmp_path, scenario):
    spec = query()
    spec["ambiguities"] = ["Unclear whether the option describes an error or a correction"]
    model = FakeModel({"compiler": lambda body: deepcopy(spec)})
    result = R6VideoAgent(model, media_factory=FakeMedia).solve(
        request(tmp_path, request_id=scenario)
    )
    assert result.stop_reason == "INPUT_AMBIGUITY" and result.forced_choice
    assert not any(c["role"] == "verifier" for c in model.calls)


def test_invalid_protocol_repairs_count_and_force_original_label(tmp_path):
    model = FakeModel({"compiler": lambda body: "not json"})
    result = R6VideoAgent(model, media_factory=FakeMedia).solve(request(tmp_path))
    assert result.prediction == "A" and result.technical_fallback
    assert result.stop_reason == "TOOL_FAILURE" and result.costs["model_calls"] == 2
    assert result.costs["failed_calls"] == 2


def test_reserved_budget_no_unaccounted_final_call(tmp_path):
    config = replace(R6Config(), max_model_calls_total=4)
    model = FakeModel()
    result = R6VideoAgent(model, config, media_factory=FakeMedia).solve(request(tmp_path))
    assert result.stop_reason == "BUDGET_EXHAUSTED" and result.forced_choice
    assert result.costs["model_calls"] == len(model.calls) == 2


def test_verifier_does_not_receive_prior_choice_or_defence(tmp_path):
    model = FakeModel()
    R6VideoAgent(model, media_factory=FakeMedia).solve(request(tmp_path))
    body = next(c["body"] for c in model.calls if c["role"] == "verifier")
    import json

    serialized = json.dumps(body["input"])
    assert "preferred_label" not in serialized and "fit_reason" not in serialized
    assert "atom_assessments" not in serialized and body["source_manifest"]


def test_failed_verification_cannot_be_voted_into_sufficiency(tmp_path):
    def reject(body):
        return {
            "checks": [
                {
                    "check_id": c["check_id"],
                    "state": "unknown",
                    "source_ids": [],
                    "missing": ["identity unresolved"],
                }
                for c in body["input"]["checks"]
            ],
            "gaps": [gap("identity", "Resolve the animal")],
        }

    model = FakeModel({"verifier": reject})
    result = R6VideoAgent(
        model, replace(R6Config(), max_refinement_rounds=1), media_factory=FakeMedia
    ).solve(request(tmp_path))
    assert result.evidence_status == "insufficient" and result.forced_choice
    assert result.costs["model_calls"] <= 16


def test_unknown_fields_in_config_and_role_output_rejected():
    with pytest.raises(ProtocolError):
        R6Config.from_mapping({"secret_budget": 99})
    with pytest.raises(ProtocolError):
        parse('{"records": [], "gaps": [], "overflow": false, "answer": "A"}', "observer")
