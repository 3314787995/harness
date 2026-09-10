"""1.4 regressions from the original two 1.3 GPU requests; no answer keys."""

import copy
import json
from dataclasses import replace
from pathlib import Path

import pytest
from test_r2 import FakeQwen, frame_map, observation, query, record, reduced
from test_r2 import config as config  # noqa: PLC0414
from test_r2 import video as video  # noqa: PLC0414
from test_r2_protocol_v13 import valid_final

from qwen3vl_agent.r2 import R2Request, R2VideoAgent
from qwen3vl_agent.r2.contracts import (
    certify_final,
    parse,
    stage_schema,
    validate_final,
    validate_observation,
)
from qwen3vl_agent.r2.observation import ObservationSpec
from qwen3vl_agent.r2.planning import select_action
from qwen3vl_agent.r2.state import StateStore
from qwen3vl_agent.r2.types import InputContract, ProtocolError

CASES = json.loads(
    (Path(__file__).parent / "fixtures/r2_v13_gpu_regressions.json").read_text(encoding="utf-8")
)["cases"]


def test_real_rotation_context_unknown_is_retained_and_unresolved():
    case = next(c for c in CASES if c["role"] == "observe")
    for call in case["calls"]:
        payload = call["payload"]
        raw = parse(call["raw"], "observe", stage_schema("observe", payload))
        before = copy.deepcopy(raw)
        spec = ObservationSpec.from_query(case["query"])
        frames = frame_map([r["source_seconds"] for r in payload["frames"]])
        value, annotations = spec.normalize(raw, payload["core"], frames)
        validate_observation(value, case["query"], frames, set(), spec=spec)
        assert raw == before
        assert annotations and value["records"]
        context = [
            r
            for r in value["records"]
            if frames[r["frame_id"]]["timestamp_seconds"] > payload["core"][1]
        ]
        assert len(context) == 3
        for r in context:
            t = frames[r["frame_id"]]["timestamp_seconds"]
            assert any(g["slot_id"] == r["slot_id"] and g["span"] == [t, t] for g in value["gaps"])
        store = StateStore()
        store.ingest(
            "w", value, frames, case["query"], {"span": [0, 5], "completed": False}, "replay"
        )
        assert store.observations
        assert reduced(case["query"], store)["status"] == "unresolved"


def test_known_gap_does_not_hide_unexplained_missing_measurement_at_another_time():
    q = query()
    spec = ObservationSpec.from_query(q)
    raw = observation([record(0, point=None, description="blurred"), record(1, point=None)])
    frames = frame_map([0, 4.5])
    value, _ = spec.normalize(raw, [0, 5], frames)
    assert value["gaps"][0]["span"] == [0, 0]
    with pytest.raises(ProtocolError, match="missing point"):
        validate_observation(value, q, frames, set(), spec=spec)


def test_existing_core_gap_does_not_suppress_explained_context_gap():
    q = query()
    spec = ObservationSpec.from_query(q)
    raw = observation([record(0, point=None, description="blurred context")])
    raw["gaps"] = [
        {"kind": "detail", "slot_id": "S1", "description": "core blurred", "span": [0, 4]}
    ]
    frames = frame_map([4.5])
    value, _ = spec.normalize(raw, [0, 4], frames)
    assert len(value["gaps"]) == 2
    validate_observation(value, q, frames, set(), spec=spec)


def test_point_gap_routes_to_nonempty_window_within_cutoff(config):
    contract = InputContract.resolve(R2Request("v", "q", observation_cutoff=4.5), 5)
    gap = {"kind": "detail", "span": [4.5, 4.5], "slot_id": "S1", "description": "blurred"}
    action = select_action([gap], [[0, 4.5]], contract, [], "revision", config)
    assert action and action["windows"]
    assert action["gap"]["span"] == [4.5, 4.5]
    assert all(0 <= w["span"][0] < w["span"][1] <= 4.5 for w in action["windows"])
    assert (
        select_action([gap], [[0, 4.5]], contract, [action["signature"]], "revision", config)
        is None
    )


def test_real_direction_prediction_retained_without_certifying_claims():
    case = next(c for c in CASES if c["role"] == "final")
    for call in case["calls"]:
        payload = call["payload"]
        raw = parse(call["raw"], "final", stage_schema("final", payload))
        before = copy.deepcopy(raw)
        assert raw["recheck"]["evidence_ids"]
        result = certify_final(raw, payload)
        assert result["prediction"] == raw["prediction"]
        assert all(a["status"] == "unknown" for a in result["assessments"])
        assert any(s.startswith("program_evidence_certification:") for s in result["unresolved"])
        assert result["recheck"] == raw["recheck"] and raw == before
        validate_final(result, payload)


@pytest.mark.parametrize(
    "failure",
    [
        "bad_label",
        "wrong_order",
        "duplicate_label",
        "bad_reference",
        "bad_recheck_reference",
        "bad_operation",
    ],
)
def test_prediction_separation_does_not_accept_invalid_identity_or_references(failure):
    _, payload, value = valid_final()
    if failure == "bad_label":
        value["prediction"] = "Z"
    elif failure == "wrong_order":
        value["assessments"].reverse()
    elif failure == "duplicate_label":
        value["assessments"][1]["label"] = "X"
    elif failure == "bad_reference":
        value["evidence_ids"] = ["not-shown"]
    elif failure == "bad_recheck_reference":
        value["recheck"] = {"kind": "detail", "description": "check", "evidence_ids": ["not-shown"]}
    else:
        value["assessments"][0]["operation_ids"] = ["invented"]
    with pytest.raises(ProtocolError):
        certify_final(value, payload)


def test_sufficient_measurements_keep_certification():
    _, payload, value = valid_final()
    assert certify_final(value, payload) == value
    assert value["assessments"][0]["status"] == "supported"


def test_controller_retains_choice_audits_downgrade_and_resumes_without_calls(
    video, config, tmp_path
):
    def unsupported(value, payload):
        for a in value["assessments"]:
            a.update(status="supported", evidence_ids=[], operation_ids=[])
        value["recheck"] = None
        return value

    model = FakeQwen(final_hook=unsupported)
    agent = R2VideoAgent(model, config=config)
    request = R2Request(
        str(video),
        "Direction?",
        choices=["one", "two"],
        checkpoint_path=str(tmp_path / "resume.jsonl"),
    )
    result = agent.solve(request)
    assert result.prediction == "A"
    assert result.completion_state == "partial" and result.support_level != "supported"
    assert all(a["status"] == "unknown" for a in result.option_assessments)
    finals = [c for c in result.trace["calls"] if c["role"] == "final"]
    assert len(finals) == 1
    assert json.loads(finals[0]["raw"])["assessments"][0]["status"] == "supported"
    assert (
        finals[0]["program_annotations"]["normalized_value"]["assessments"][0]["status"]
        == "unknown"
    )
    assert result.resources["frame_exposures"] == sum(
        len(c["frame_ids"]) for c in result.trace["calls"]
    )
    assert agent.solve(replace(request, resume=True)).to_dict() == result.to_dict()


def test_fixture_has_no_scoring_keys():
    def visit(value):
        if isinstance(value, dict):
            assert (
                not {"gold", "answer", "correct_answer", "ground_truth", "accuracy"} & value.keys()
            )
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(CASES)
