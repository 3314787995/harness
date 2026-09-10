"""1.3: retain uncertain state, short identity protocol, necessary answer evidence."""

import copy
import json
from pathlib import Path

import pytest
from test_r2 import FakeQwen, frame_map, observation, query, record, reduced, store_for
from test_r2 import config as config  # noqa: PLC0414
from test_r2 import video as video  # noqa: PLC0414

from qwen3vl_agent.r2 import R2Request, R2VideoAgent
from qwen3vl_agent.r2.contracts import parse, stage_schema, validate_final, validate_observation
from qwen3vl_agent.r2.evidence import assessment_policy
from qwen3vl_agent.r2.observation import (
    ObservationSpec,
    normalize_identity_handoff,
    validate_identity_handoff,
)
from qwen3vl_agent.r2.state import StateStore
from qwen3vl_agent.r2.types import ProtocolError

CASES = json.loads(
    (Path(__file__).parent / "fixtures/r2_v12_gpu_regressions.json").read_text(encoding="utf-8")
)["cases"]


def replay_observation(case, call):
    q = case["query"]
    payload = copy.deepcopy(call["payload"])
    requirements = payload.get("identity_requirements", [])
    for i, r in enumerate(requirements):
        r["key"] = f"K{i + 1}"
    value = parse(call["raw"], "observe", stage_schema("observe", payload))
    spec = ObservationSpec.from_query(q)
    value, _ = spec.normalize(value, payload["core"])
    value = normalize_identity_handoff(value, requirements)
    frames = frame_map([r["source_seconds"] for r in payload["frames"]])
    validate_observation(value, q, frames, {r["from_node"] for r in requirements}, spec=spec)
    validate_identity_handoff(value, requirements)
    return value, frames


def test_real_null_rotation_responses_preserve_unknown_without_fabricated_angles():
    case = next(c for c in CASES if not c["observations"])
    for call in [c for c in case["calls"] if c["role"] == "observe"]:
        value, frames = replay_observation(case, call)
        assert value["records"]
        assert all("orientation_angle" not in r for r in value["records"])
        assert all(r["rotation_type"] == "unknown" for r in value["records"])
        assert value["gaps"]
        store = StateStore()
        store.ingest("w", value, frames, case["query"], {"span": [0, 4], "completed": False}, "raw")
        assert reduced(case["query"], store)["status"] == "unresolved"
        assert '"orientation_angle": null' in call["raw"]


def test_real_missing_handoff_keeps_measurements_but_never_creates_identity():
    case = next(c for c in CASES if c["observations"])
    call = [c for c in case["calls"] if c["role"] == "observe"][-1]
    value, _ = replay_observation(case, call)
    assert value["records"] and not value["associations"]
    assert any(g.get("source") == "program_identity_contract" for g in value["gaps"])
    assert not json.loads(call["raw"])["gaps"]


def test_null_required_position_without_explanation_still_needs_repair():
    q = query()
    spec = ObservationSpec.from_query(q)
    value, _ = spec.normalize(observation([record(0, point=None)]), [0, 1])
    with pytest.raises(ProtocolError, match="missing point"):
        validate_observation(value, q, frame_map([0]), set(), spec=spec)


def short_requirement():
    return [
        {
            "key": "K1",
            "from_node": "old/E1",
            "target_id": "T1",
            "slot_ids": ["S1"],
            "shared_frame_ids": ["F01"],
        }
    ]


def test_short_match_binds_only_explicit_observed_correspondence():
    value = observation([record(0, point=[5, 6])])
    value["identity_matches"] = [
        {
            "key": "K1",
            "status": "same_entity",
            "entity_id": "E1",
            "evidence_frames": ["F01"],
            "reason": "Same visible feature on shared frame",
        }
    ]
    normalized = normalize_identity_handoff(value, short_requirement())
    validate_identity_handoff(normalized, short_requirement())
    link = normalized["associations"][0]["alternatives"][0]["links"][0]
    assert link == {"from_node": "old/E1", "to_entity": "E1", "kind": "same_entity"}
    assert not normalized["associations"][0]["relation_preserved"]
    assert not value["associations"]  # Raw output is untouched.


@pytest.mark.parametrize("failure", ["unknown_key", "no_frames", "wrong_frame", "duplicate"])
def test_short_match_bad_references_remain_protocol_errors(failure):
    value = observation([record(0, point=[5, 6])])
    match = {
        "key": "K1",
        "status": "same_entity",
        "entity_id": "E1",
        "evidence_frames": ["F01"],
        "reason": "shared feature",
    }
    if failure == "unknown_key":
        match["key"] = "K9"
    if failure == "no_frames":
        match["evidence_frames"] = []
    if failure == "wrong_frame":
        match["evidence_frames"] = ["F99"]
    value["identity_matches"] = [match] * (2 if failure == "duplicate" else 1)
    with pytest.raises(ProtocolError):
        normalized = normalize_identity_handoff(value, short_requirement())
        validate_identity_handoff(normalized, short_requirement())


def test_short_unknown_requires_no_invented_current_entity():
    value = observation([record(0, point=[5, 6])])
    value["identity_matches"] = [
        {"key": "K1", "status": "unknown", "reason": "Occluded at connection"}
    ]
    normalized = normalize_identity_handoff(value, short_requirement())
    assert not normalized["associations"]
    assert normalized["gaps"][0]["source"] == "observer_identity_match"


def valid_final():
    q = query()
    store = store_for(
        q, [record(0, point=[100, 200]), record(1, point=[200, 200]), record(2, point=[300, 200])]
    )
    op = reduced(q, store)
    payload = {
        "question": "motion?",
        "options": [{"label": "X", "text": "one"}, {"label": "Y", "text": "two"}],
        "observations": store.observations,
        "derived": {"sufficient": True, "operations": [op]},
    }
    payload["assessment_policy"] = assessment_policy(q, payload)
    refs = [r["id"] for r in store.observations]
    value = {
        "prediction": "X",
        "evidence_ids": refs,
        "weakest_premise": "measured movement",
        "recheck": None,
        "unresolved": [],
        "assessments": [
            {"label": "X", "status": "supported", "evidence_ids": refs, "operation_ids": ["Q1"]},
            {"label": "Y", "status": "unknown", "evidence_ids": []},
        ],
    }
    return q, payload, value


def test_ordered_measurements_can_enter_decisive_answer_contract():
    _, payload, value = valid_final()
    validate_final(parse(json.dumps(value), "final", stage_schema("final", payload)), payload)


@pytest.mark.parametrize(
    "failure",
    [
        "two_supported",
        "wrong_prediction",
        "single_pose",
        "unknown_operation",
        "incomplete",
        "posture_only",
    ],
)
def test_final_rejects_known_unsupported_claim_patterns(failure):
    q, payload, value = valid_final()
    if failure == "two_supported":
        value["assessments"][1].update(
            status="supported", evidence_ids=value["evidence_ids"], operation_ids=["Q1"]
        )
    elif failure == "wrong_prediction":
        value["prediction"] = "Y"
    elif failure == "single_pose":
        value["assessments"][0]["evidence_ids"] = value["evidence_ids"][:1]
    elif failure == "unknown_operation":
        value["assessments"][0]["operation_ids"] = ["unknown"]
    elif failure == "incomplete":
        payload["assessment_policy"]["complete_support_allowed"] = False
    else:
        for r in payload["observations"]:
            r.pop("source_point")
            r["value"] = "pointing_left"
        payload["assessment_policy"] = assessment_policy(q, payload)
    with pytest.raises(ProtocolError):
        validate_final(value, payload)


def test_real_multiple_support_final_is_rejected_not_silently_accepted():
    case = next(c for c in CASES if c["observations"])
    call = next(c for c in case["calls"] if c["role"] == "final")
    payload = copy.deepcopy(call["payload"])
    payload["assessment_policy"] = assessment_policy(case["query"], payload)
    with pytest.raises(ProtocolError, match="multiple whole options"):
        validate_final(json.loads(call["raw"]), payload)


def test_short_identity_controller_creates_real_association_and_audit(video, config):
    def short(value, payload):
        if payload.get("identity_requirements"):
            value["associations"] = []
            value["identity_matches"] = [
                {
                    "key": r["key"],
                    "status": "same_entity",
                    "entity_id": "E1",
                    "evidence_frames": r["shared_frame_ids"] or [payload["frames"][0]["frame_id"]],
                    "reason": "Shared visible feature",
                }
                for r in payload["identity_requirements"]
            ]
        return value

    result = R2VideoAgent(FakeQwen(observer_hook=short), config=config).solve(
        R2Request(str(video), "Direction?", choices=["one", "two"])
    )
    assert result.completion_state == "complete", result.unresolved_items
    assert result.state_store["associations"]
    assert any(
        "identity_matches" in c["raw"] and c.get("program_annotations")
        for c in result.trace["calls"]
    )
