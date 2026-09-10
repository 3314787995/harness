"""Regressions for the actual R2 empty-output failures; CPU only, no model weights."""

import json
from dataclasses import replace
from pathlib import Path

import pytest
from test_r2 import FakeQwen, frame_map, observation, query, record, reduced, store_for
from test_r2 import config as config  # noqa: PLC0414 - pytest fixture exports
from test_r2 import video as video  # noqa: PLC0414 - pytest fixture export

from qwen3vl_agent.models.base import ModelOutput
from qwen3vl_agent.r2 import R2Request, R2VideoAgent
from qwen3vl_agent.r2.contracts import (
    parse,
    repair_context,
    stage_schema,
    validate_final,
    validate_observation,
)
from qwen3vl_agent.r2.observation import ObservationSpec
from qwen3vl_agent.r2.prompts import prompt
from qwen3vl_agent.r2.types import ProtocolError

FAILURES = json.loads(
    (Path(__file__).parent / "fixtures/r2_gpu_empty_responses.json").read_text(encoding="utf-8")
)["cases"]


def wire_payload(q=None):
    q = q or query()
    return {
        "targets": q["targets"],
        "slots": q["slots"],
        "anchors": q["anchors"],
        "frames": [{"frame_id": f"F{i + 1:02d}", "source_seconds": i} for i in range(3)],
        "observation_spec": ObservationSpec.from_query(q).to_dict(),
    }


def validate_read(q, value):
    spec = ObservationSpec.from_query(q)
    value, _ = spec.normalize(value, (0, 2))
    return validate_observation(value, q, frame_map([0, 1, 2]), set(), spec=spec)


class ReplayFailure(FakeQwen):
    def __init__(self, case, **kwargs):
        super().__init__(**kwargs)
        self.case, self.seen = case, set()

    def generate(self, messages, **kwargs):
        body = json.loads(messages[0]["content"][-1]["text"].split("\n", 1)[1])
        role = body["stage"]
        if role in {"compile_intent", "observe", "final"} and role not in self.seen:
            self.seen.add(role)
            raw = next(c["raw"] for c in self.case["calls"] if c["role"] == role)
            return ModelOutput(raw)
        return super().generate(messages, **kwargs)


@pytest.mark.parametrize("case", FAILURES, ids=lambda c: c["source_request_id"])
def test_real_failures_recover_with_concrete_feedback_and_same_media(case, video, config):
    model = ReplayFailure(case)
    request = R2Request(
        str(video),
        "Describe the tracked object's motion",
        choices=["one", "two", "three", "four", "five"],
    )
    result = R2VideoAgent(model, config=config).solve(request)
    assert result.completion_state == "complete", result.unresolved_items
    assert not result.trace["actions"]
    calls = result.trace["calls"]
    first_observers = [c for c in calls if c["role"] == "observe"][:2]
    assert first_observers[0]["validation_status"] == "rejected"
    assert first_observers[0]["frame_ids"] == first_observers[1]["frame_ids"]
    assert first_observers[0]["pixels"] == first_observers[1]["pixels"]
    repaired = json.loads(first_observers[1]["prompt"].split("\n", 1)[1])
    assert repaired["format_repair"]["allowed_frame_ids"]
    assert repaired["format_repair"]["required_measurements"]["tasks"][0]["required_fields"] == [
        "point"
    ]
    terminal = [c for c in calls if c["role"] == "final"]
    feedback = json.loads(terminal[1]["prompt"].split("\n", 1)[1])["format_repair"]
    assert feedback["missing_labels"] == list("ABCDE")
    assert feedback["required_assessment_order"] == list("ABCDE")
    assert result.resources["terminal_calls"] == 2
    assert result.resources["frame_exposures"] == sum(len(c["frame_ids"]) for c in calls)


def test_repeated_empty_record_failure_does_not_schedule_visual_refinement(video, config):
    def empty(value, payload):
        value.update(entities=[], records=[], gaps=[], complete=True)
        return value

    def ask_to_recheck(value, payload):
        value["recheck"] = {"kind": "detail", "description": "look again", "span": [0, 1]}
        return value

    result = R2VideoAgent(
        FakeQwen(observer_hook=empty, final_hook=ask_to_recheck), config=config
    ).solve(R2Request(str(video), "direction?", choices=["right", "left"]))
    calls = [c for c in result.trace["calls"] if c["role"] == "observe"]
    assert len(calls) == 2 and all(c["validation_status"] == "rejected" for c in calls)
    assert "provide observed records" in calls[0]["validation_error"]
    assert not result.trace["actions"]
    assert result.completion_state == "partial" and result.support_level == "unsupported"
    assert result.state_store["observations"] == []
    assert result.coverage_manifest[0]["protocol_status"] == "failed"
    assert not result.coverage_manifest[0]["processed"]
    assert any(c["frame_ids"] for c in result.trace["calls"] if c["role"] == "final")
    assert result.resources["terminal_calls"] == 1


def test_named_missing_target_routes_to_localization_with_feedback(video, config):
    def missing_once(value, payload):
        if payload["window_id"] == "base_00000":
            value.update(
                entities=[],
                records=[],
                complete=True,
                gaps=[
                    {
                        "kind": "localization",
                        "slot_id": "S1",
                        "description": "Target not found in this window",
                        "span": [0, 3],
                    }
                ],
            )
        return value

    model = FakeQwen(observer_hook=missing_once)
    result = R2VideoAgent(model, config=config).solve(R2Request(str(video), "direction?"))
    first = result.coverage_manifest[0]
    assert first["processed"] and not first["completed"] and first["model_reported_complete"]
    assert result.trace["actions"][0]["action"] == "relocate"
    payloads = [p for role, p, _ in model.payloads if role == "observe"]
    assert "handoff" not in payloads[0]
    rechecks = [p["recheck_context"] for p in payloads if "recheck_context" in p]
    assert rechecks and rechecks[0]["gap"]["kind"] == "localization"
    assert rechecks[0]["needed_measurements"][0]["slot_id"] == "S1"
    assert "previous_observations" in rechecks[0]
    assert result.resources["terminal_calls"] <= 2


def test_occlusion_remains_unknown_and_normalization_is_audited(video, config):
    def occluded(value, payload):
        for r in value["records"]:
            r.pop("point")
            r.update(
                visibility="occluded", value=None, description="The target is behind the screen"
            )
        return value

    result = R2VideoAgent(FakeQwen(observer_hook=occluded), config=config).solve(
        R2Request(str(video), "direction?")
    )
    assert not result.value_state["sufficient"]
    assert all(
        r["visibility"] == "occluded" and "source_point" not in r
        for r in result.state_store["observations"]
    )
    first = next(c for c in result.trace["calls"] if c["role"] == "observe")
    assert json.loads(first["raw"])["gaps"] == []
    assert first["program_annotations"]["normalized_value"]["gaps"]
    assert first["validation_status"] == "accepted"


@pytest.mark.parametrize(
    "reference,missing",
    [("screen", "point"), ("scene", "reference_point"), ("body", "scale"), ("object", "scale")],
)
def test_required_geometry_fields_or_specific_gap(reference, missing):
    q = query()
    q["slots"][0]["property"] = (
        "direction"  # The operation, not a free-form name, determines measurements.
    )
    q["slots"][0]["reference_frame"] = reference
    row = record(0, point=[100, 200], reference_point=[50, 50], scale=300)
    row.pop(missing)
    value = observation([row])
    with pytest.raises(ProtocolError, match=missing):
        validate_read(q, value)
    value["gaps"] = [{"kind": "reference", "description": "Reference obscured", "slot_id": "S1"}]
    validate_read(q, value)
    assert reduced(q, store_for(q, [row, {**row, "frame_id": "F02"}]))["status"] == "unresolved"


def test_gap_for_other_frame_or_slot_cannot_excuse_missing_measurement():
    q = query()
    value = observation(
        [record(0)],
        gaps=[{"kind": "detail", "description": "Blurred later", "slot_id": "S1", "span": [1, 2]}],
    )
    with pytest.raises(ProtocolError, match="point"):
        validate_read(q, value)
    value["gaps"][0]["slot_id"] = "OTHER"
    with pytest.raises(ProtocolError, match="slot_id"):
        validate_read(q, value)


@pytest.mark.parametrize("mode", ["orbit", "heading", "self_spin"])
def test_rotation_type_is_observed_and_reaches_geometry(mode):
    q = query("rotation_pattern", {"rotation_type": "unknown"})
    rows = [
        record(
            i,
            rotation_type=mode,
            orientation_angle=a,
            feature_identifiable=True,
            adjacency_resolved=True,
            point=p,
            reference_point=[500, 250],
        )
        for i, (a, p) in enumerate(zip([0, 45, 90], [[600, 250], [570, 180], [500, 150]]))
    ]
    validate_read(q, observation(rows))
    result = reduced(q, store_for(q, rows))
    assert result["status"] == "supported"
    assert result["value"]["rotation_type"] == mode
    assert result["value"]["directions"] == ["counterclockwise"]


def test_rotation_agent_uses_visual_type_and_completes_program_path(video, config):
    def rotate(value, payload):
        times = {f["frame_id"]: f["source_seconds"] for f in payload["frames"]}
        for row in value["records"]:
            row.update(
                rotation_type="self_spin",
                orientation_angle=-20 * times[row["frame_id"]],
                feature_identifiable=True,
                adjacency_resolved=True,
            )
        return value

    model = FakeQwen(query("rotation_pattern", {"rotation_type": "unknown"}), observer_hook=rotate)
    result = R2VideoAgent(model, config=config).solve(R2Request(str(video), "rotation?"))
    assert result.completion_state == "complete", result.unresolved_items
    op = result.value_state["operations"][0]
    assert op["value"]["rotation_type"] == "self_spin"
    assert op["value"]["directions"] == ["clockwise"]


def test_interrupted_observer_keeps_same_frames_when_resumed(video, config, tmp_path):
    model = FakeQwen(interrupt_role="observe")
    request = R2Request(str(video), "direction?", checkpoint_path=str(tmp_path / "resume.jsonl"))
    with pytest.raises(KeyboardInterrupt):
        R2VideoAgent(model, config=config).solve(request)
    result = R2VideoAgent(model, config=config).solve(replace(request, resume=True))
    calls = [c for c in result.trace["calls"] if c["role"] == "observe"]
    assert calls[0]["status"] == "interrupted"
    assert calls[0]["frame_ids"] == calls[1]["frame_ids"]
    assert calls[0]["payload"] == calls[1]["payload"]


@pytest.mark.parametrize("visibility", ["occluded", "absent", "unknown"])
def test_unseen_target_cannot_supply_visual_coordinates(visibility):
    value = observation(
        [record(0, point=[10, 20], visibility=visibility, description="Target is not visible")]
    )
    with pytest.raises(ProtocolError, match="unseen point"):
        validate_read(query(), value)


@pytest.mark.parametrize(
    "failure",
    [
        "missing_type",
        "unknown_type",
        "mixed_types",
        "query_conflict",
        "unidentifiable",
        "disconnected",
        "missing_center",
    ],
)
def test_rotation_uncertainty_cannot_become_supported(failure):
    q = query("rotation_pattern", {"rotation_type": "unknown"})
    rows = [
        record(
            i,
            rotation_type="self_spin",
            orientation_angle=i * 30,
            feature_identifiable=True,
            adjacency_resolved=True,
        )
        for i in range(3)
    ]
    if failure == "missing_type":
        rows[0].pop("rotation_type")
    elif failure == "unknown_type":
        rows[0]["rotation_type"] = "unknown"
    elif failure == "mixed_types":
        rows[0]["rotation_type"] = "heading"
    elif failure == "query_conflict":
        q["operations"][0]["parameters"]["rotation_type"] = "orbit"
    elif failure == "unidentifiable":
        rows[1]["feature_identifiable"] = False
    elif failure == "disconnected":
        rows[1]["adjacency_resolved"] = False
    else:
        for r in rows:
            r["rotation_type"] = "orbit"
    assert reduced(q, store_for(q, rows))["status"] == "unresolved"


def test_observer_schema_is_selected_and_is_the_displayed_contract():
    payload = wire_payload()
    schema = stage_schema("observe", payload)
    body = json.loads(prompt("observe", payload).split("\n", 1)[1])
    assert body["output_schema"] == schema
    assert "containments" not in schema["properties"]
    assert "associations" not in schema["properties"]
    assert "orientation_angle" not in schema["properties"]["records"]["items"]["properties"]
    assert "cycle_marker" not in body["instruction"]
    assert len(prompt("observe", payload)) < 8000
    payload["handoff"] = {
        "entities": [{"node_id": "prior/E1"}],
        "previous_observations": [{"id": "prior/O1"}],
    }
    assert "associations" in stage_schema("observe", payload)["properties"]
    assert "superseded_observation_ids" in stage_schema("observe", payload)["properties"]
    identity = stage_schema("observe", wire_payload(query("identity_at_time")))
    assert "containments" in identity["properties"]


def final_payload():
    return {
        "options": [{"label": c, "text": "candidate " + c} for c in ["Z", "X", "Q"]],
        "observations": [{"id": "w/O1"}],
    }


def final_value():
    return {
        "prediction": "X",
        "assessments": [
            {"label": c, "status": "unknown", "evidence_ids": []} for c in ["Z", "X", "Q"]
        ],
        "evidence_ids": [],
        "weakest_premise": "Missing visual evidence",
        "unresolved": ["unknown"],
        "recheck": None,
    }


@pytest.mark.parametrize(
    "failure",
    [
        "empty",
        "duplicate",
        "wrong_order",
        "invalid_prediction",
        "invalid_reference",
        "unsupported_claim",
    ],
)
def test_final_contract_detects_actual_missing_order_and_evidence(failure):
    value, payload = final_value(), final_payload()
    if failure == "empty":
        value["assessments"] = []
    elif failure == "duplicate":
        value["assessments"][1]["label"] = "Z"
    elif failure == "wrong_order":
        value["assessments"].reverse()
    elif failure == "invalid_prediction":
        value["prediction"] = "A"
    elif failure == "invalid_reference":
        value["evidence_ids"] = ["invented"]
    else:
        value["assessments"][0]["status"] = "supported"
    with pytest.raises(ProtocolError):
        parsed = parse(json.dumps(value), "final", stage_schema("final", payload))
        validate_final(parsed, payload)
    feedback = repair_context("final", payload, json.dumps(value))
    assert feedback["required_assessment_order"] == ["Z", "X", "Q"]
    if failure == "empty":
        assert feedback["missing_labels"] == ["Z", "X", "Q"]
    if failure == "duplicate":
        assert feedback["duplicate_labels"] == ["Z"] and feedback["missing_labels"] == ["X"]


def test_all_unknown_and_free_text_remain_valid_without_fabricated_evidence():
    payload, value = final_payload(), final_value()
    parsed = parse(json.dumps(value), "final", stage_schema("final", payload))
    assert validate_final(parsed, payload) == value
    payload["options"] = []
    value.update(prediction="cannot resolve observed motion", assessments=[])
    assert (
        validate_final(parse(json.dumps(value), "final", stage_schema("final", payload)), payload)
        == value
    )


def test_previous_protocol_checkpoint_is_retained_but_not_resumed(
    video, config, tmp_path, monkeypatch
):
    from qwen3vl_agent.r2 import runtime

    path = tmp_path / "old.jsonl"
    request = R2Request(str(video), "direction?", checkpoint_path=str(path))
    with monkeypatch.context() as patch:
        patch.setattr(runtime, "VERSION", "r2-entity-time/1.0")
        R2VideoAgent(FakeQwen(), config=config).solve(request)
    old = path.read_bytes()
    with pytest.raises(ValueError):
        R2VideoAgent(FakeQwen(), config=config).solve(replace(request, resume=True))
    assert path.read_bytes() == old


def test_failure_fixtures_have_no_scoring_fields():
    for case in FAILURES:
        assert set(case) == {"source_request_id", "source_sha256", "calls"}
        assert all(set(c) == {"call_id", "role", "raw"} for c in case["calls"])
        assert sum(c["role"] == "final" for c in case["calls"]) == 2
