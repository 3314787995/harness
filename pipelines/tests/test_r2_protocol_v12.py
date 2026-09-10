"""1.2 regressions based on the two real 1.1 runs; no GPU or grading input."""

import copy
import json
from dataclasses import replace
from pathlib import Path

import pytest
from test_r2 import FakeQwen, observation, query, record
from test_r2 import config as config  # noqa: PLC0414
from test_r2 import video as video  # noqa: PLC0414

from qwen3vl_agent.models.base import ModelOutput
from qwen3vl_agent.r2 import InputContract, R2Budget, R2Request, R2VideoAgent
from qwen3vl_agent.r2.context import fit_final_payload
from qwen3vl_agent.r2.contracts import repair_context, validate_final, validate_query
from qwen3vl_agent.r2.observation import validate_identity_handoff
from qwen3vl_agent.r2.planning import select_action
from qwen3vl_agent.r2.prompts import prompt
from qwen3vl_agent.r2.runtime import ModelSession
from qwen3vl_agent.r2.types import VERSION, BudgetExhausted, ProtocolError

CASES = json.loads(
    (Path(__file__).parent / "fixtures/r2_v11_gpu_regressions.json").read_text(encoding="utf-8")
)["cases"]
DIRECTION = next(c for c in CASES if c["observations"])
ROTATION = next(c for c in CASES if not c["observations"])


def test_real_state_fits_final_with_repair_headroom_and_preserves_source():
    payload = {
        "question": DIRECTION["question"],
        "options": [{"label": label, "text": "choice " + label} for label in "ABCDE"],
        "derived": DIRECTION["derived"],
        "observations": DIRECTION["observations"],
        "unresolved": DIRECTION["unresolved"],
        "raw_frames": [{"frame_id": "F01", "source_seconds": 2.5}],
        "raw_windows": [{"window_id": "raw", "frame_ids": ["frame1"]}],
    }
    original = copy.deepcopy(payload)
    fitted, diagnostic = fit_final_payload(payload, 60000)
    assert diagnostic["original_chars"] > 60000
    assert diagnostic["final_chars"] <= 40000
    assert fitted["observations"]
    assert payload == original
    for key in ("question", "options", "raw_frames", "raw_windows", "unresolved"):
        assert fitted[key] == payload[key]
    allowed = {r["id"] for r in fitted["observations"]}
    assert all(set(o["evidence_ids"]) <= allowed for o in fitted["derived"]["operations"])
    repair = {
        "error": "missing assessments",
        "previous_output": "x" * 12000,
        **repair_context("final", fitted, "{}"),
    }
    assert len(prompt("final", fitted, repair)) <= 60000


def test_protected_question_is_not_truncated_to_hide_budget_failure():
    payload = {
        "question": "x" * 10000,
        "options": [],
        "observations": [],
        "derived": {"operations": []},
    }
    with pytest.raises(BudgetExhausted, match="protected_terminal_context"):
        fit_final_payload(payload, 1000)
    assert len(payload["question"]) == 10000


def test_long_original_question_can_use_relaxed_headroom():
    payload = {
        "question": "q" * 60000,
        "options": [],
        "observations": [],
        "derived": {"operations": []},
    }
    fitted, diagnostic = fit_final_payload(payload, 120000)
    assert fitted["question"] == payload["question"]
    assert 60000 < diagnostic["final_chars"] <= diagnostic["target_chars"] <= 100000


def test_budget_preflight_diagnostic_is_not_charged_as_model_call(config):
    state = {}
    session = ModelSession(
        FakeQwen(), config, R2Budget(max_text_chars_per_call=10), state, lambda: None
    )
    with pytest.raises(BudgetExhausted, match="text_context_budget"):
        session.call("compile", "compile_intent", {})
    assert session.summary()["model_calls"] == 0
    assert state["preflight_failures"][0]["text_chars"] > 10
    assert not state["preflight_failures"][0]["model_invoked"]


def test_real_silent_identity_omission_is_rejected_but_explained_gap_is_legal():
    call = next(c for c in DIRECTION["calls"] if c["role"] == "observe")
    value = json.loads(call["raw"])
    required = [
        {"from_node": e["node_id"], "target_id": e["target_id"], "shared_frame_ids": ["F01"]}
        for e in call["payload"]["handoff"]["entities"]
    ]
    assert not value["associations"]
    with pytest.raises(ProtocolError, match="Missing identity handoff"):
        validate_identity_handoff(value, required)
    value["gaps"].append(
        {
            "kind": "identity",
            "slot_id": "hand_direction",
            "description": "Correspondence cannot be resolved",
        }
    )
    validate_identity_handoff(value, required)


def test_identity_link_must_cite_actual_shared_frame():
    value = observation(
        [record(0, point=[1, 2])],
        associations=[
            {
                "alternatives": [
                    {
                        "links": [
                            {"from_node": "old/E1", "to_entity": "E1", "kind": "same_entity"}
                        ],
                        "evidence_frames": ["F02"],
                    }
                ],
            }
        ],
    )
    requirements = [{"from_node": "old/E1", "target_id": "T1", "shared_frame_ids": ["F01"]}]
    with pytest.raises(ProtocolError, match="shared frame"):
        validate_identity_handoff(value, requirements)
    value["associations"][0]["alternatives"][0]["evidence_frames"] = ["F01"]
    validate_identity_handoff(value, requirements)


def test_missing_identity_retains_state_and_routes_to_bridge_v13(video, config):
    seen = set()

    def omit_once(value, payload):
        if payload.get("identity_requirements") and payload["window_id"] not in seen:
            seen.add(payload["window_id"])
            value["associations"] = []
        return value

    result = R2VideoAgent(FakeQwen(observer_hook=omit_once), config=config).solve(
        R2Request(str(video), "Direction?")
    )
    # 1.3 retains local measurements and explicitly marks missing correspondence;
    # this fake omits it in every new window, so no identity should become certain.
    assert result.completion_state == "partial", result.unresolved_items
    assert result.state_store["observations"]
    assert not result.state_store["associations"]
    assert result.trace["actions"][0]["action"] == "bridge_identity"
    assert any(g["source"] == "program_identity_contract" for g in result.state_store["gaps"])


def test_identity_gap_gets_bridge_not_dense_repeated_measurements(config):
    contract = InputContract.resolve(R2Request("v", "q"), 6)
    gaps = [{"kind": "identity", "description": "not joined", "span": [0, 5]}]
    first = select_action(gaps, [(0, 6)], contract, [], "revision1", config)
    assert first["action"] == "bridge_identity"
    assert len(first["windows"]) == 2
    assert all(w["fps"] == config.fps for w in first["windows"])
    assert (
        select_action(gaps, [(0, 6)], contract, [first["signature"]], "more points", config) is None
    )


def test_slot_identity_gap_without_explicit_span_can_close_after_bridge(video, config):
    def identity_gap(value, payload):
        if payload["window_id"] == "base_00001":
            value["associations"] = []
            value["gaps"] = [
                {"kind": "identity", "slot_id": "S1", "description": "Correspondence unknown"}
            ]
        return value

    result = R2VideoAgent(FakeQwen(observer_hook=identity_gap), config=config).solve(
        R2Request(str(video), "Direction?")
    )
    assert result.trace["actions"][0]["action"] == "bridge_identity"
    assert result.completion_state == "complete", result.unresolved_items
    assert all(g["resolved"] for g in result.state_store["gaps"])


def test_real_body_reference_without_question_relation_defaults_screen():
    q = validate_query(ROTATION["query"], question=ROTATION["question"])
    assert q["slots"][0]["reference_frame"] == "screen"
    assert ROTATION["query"]["slots"][0]["reference_frame"] == "body"
    explicit = query()
    explicit["slots"][0].update(reference_frame="body", reference_evidence="relative to the torso")
    assert (
        validate_query(explicit, question="Does it move relative to the torso?")["slots"][0][
            "reference_frame"
        ]
        == "body"
    )
    explicit["slots"][0].pop("reference_evidence")
    with pytest.raises(ProtocolError, match="reference_evidence"):
        validate_query(explicit, question="Does it move relative to the torso?")


def test_real_recheck_then_uncertified_final_retains_prediction_without_format_repair(
    video, config
):
    raw = [c["raw"] for c in ROTATION["calls"] if c["role"] == "final"]

    class ReplayFinal(FakeQwen):
        final_count = 0

        def generate(self, messages, **kwargs):
            body = json.loads(messages[0]["content"][-1]["text"].split("\n", 1)[1])
            if body["stage"] == "final":
                self.final_count += 1
                if self.final_count <= 2:
                    return ModelOutput(raw[self.final_count - 1])
                value = json.loads(raw[0])
                value["recheck"] = None
                return ModelOutput(json.dumps(value))
            return super().generate(messages, **kwargs)

    result = R2VideoAgent(ReplayFinal(), config=config).solve(
        R2Request(str(video), "Rotation?", choices=list("abcde"))
    )
    finals = [c for c in result.trace["calls"] if c["role"] == "final"]
    # 1.4 separates a usable prediction from failed evidence certification.
    # Structural failures still get one repair (covered by the next test).
    assert len(finals) == 2, result.unresolved_items
    assert [c["validation_status"] for c in finals] == ["accepted", "accepted"]
    assert finals[1]["program_annotations"]
    assert any("program_evidence_certification:" in str(g) for g in result.unresolved_items)
    assert result.prediction == "A" and result.completion_state == "partial"
    assert all(a["status"] == "unknown" for a in result.option_assessments)


def test_two_rounds_each_have_one_repair_and_resume_is_stable(video, config, tmp_path):
    count = 0

    def final_hook(value, _):
        nonlocal count
        count += 1
        if count in (1, 3):
            value["prediction"] = "bad-label"
        if count == 2:
            value["recheck"] = {"kind": "detail", "description": "Inspect contact", "span": [1, 2]}
        return value

    model = FakeQwen(final_hook=final_hook)
    agent = R2VideoAgent(model, config=config)
    request = R2Request(
        str(video),
        "Direction?",
        choices=["right", "left"],
        checkpoint_path=str(tmp_path / "resume.jsonl"),
    )
    result = agent.solve(request)
    assert result.resources["terminal_calls"] == 4
    assert result.prediction == "A"
    assert result.resources["frame_exposures"] == sum(
        len(c["frame_ids"]) for c in result.trace["calls"]
    )
    assert agent.solve(replace(request, resume=True)).to_dict() == result.to_dict()
    assert count == 4


def test_version_and_explicit_small_budget_still_bounded(config):
    assert VERSION == "r2-entity-time/1.5"
    assert R2Budget().terminal_call_reserve == 4
    state = {}
    session = ModelSession(FakeQwen(), config, R2Budget(max_model_calls=4), state, lambda: None)
    session.call("a", "compile_intent", {})
    session.call("b", "compile_intent", {})
    with pytest.raises(BudgetExhausted, match="model_call_budget"):
        session.call("c", "compile_intent", {})


def test_no_evidence_supported_assessments_are_still_rejected():
    call = [c for c in ROTATION["calls"] if c["role"] == "final"][-1]
    with pytest.raises(ProtocolError, match="need supporting observations"):
        validate_final(json.loads(call["raw"]), call["payload"])
