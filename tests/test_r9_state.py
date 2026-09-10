from copy import deepcopy

import pytest

from qwen3vl_agent.r9.audit import validate_visual_audit
from qwen3vl_agent.r9.schema import SCHEMAS, validate
from qwen3vl_agent.r9.spatial_state import SpatialState
from qwen3vl_agent.r9.types import MissingCapability, ProtocolError
from tests.r9_fakes import record, seeded_state, spec


def observation_payload():
    return {
        "observations": [
            {
                "id": "o1",
                "frame_id": "F01",
                "shot_id": "s1",
                "entity_candidate": "c1",
                "category": "chair",
                "visible_attributes": ["blue"],
                "facing_cues": [],
                "bbox_xyxy": [0, 0, 1, 1],
                "coordinate_space": "presented_image_normalized",
                "occlusion": "partial",
                "visible_extent": "partial",
                "observation_statement": "A blue chair is visible.",
            }
        ],
        "links": [
            {
                "entity_id": "e1",
                "candidate_ids": ["c1"],
                "status": "confirmed",
                "basis": "distinctive_attributes",
                "source_observation_ids": ["o1"],
                "valid_time": [1, 1],
            }
        ],
        "gaps": [],
    }


def evidence():
    return {
        "F01": {
            "id": "f1",
            "source_frame_id": "source1",
            "timestamp_seconds": 1,
            "view_box": [20, 30, 60, 70],
            "source_size": [100, 100],
        }
    }


def test_crop_mapping_and_observations_immutable():
    st = SpatialState(spec())
    p = observation_payload()
    st.observe(p, evidence(), "call1")
    assert st.data["observations"]["call1.o1"]["bbox_xyxy"] == [0.2, 0.3, 0.6, 0.7]
    snapshot = deepcopy(st.data)
    p["observations"][0]["observation_statement"] = "changed"
    with pytest.raises(ProtocolError, match="immutable"):
        st.observe(p, evidence(), "call1")
    assert st.data == snapshot


def test_transaction_does_not_commit_partial_invalid_sources():
    st = SpatialState(spec())
    before = deepcopy(st.data)
    p = observation_payload()
    p["links"][0]["source_observation_ids"] = ["invented"]
    with pytest.raises(ProtocolError):
        st.observe(p, evidence(), "call1")
    assert st.data == before


def test_appearance_only_cannot_confirm_identity():
    st = SpatialState(spec())
    p = observation_payload()
    p["links"][0]["basis"] = "appearance_only"
    st.observe(p, evidence(), "call1")
    with pytest.raises(MissingCapability):
        st.bound("e1")


def test_conflicting_binding_invalidates_dependencies():
    st = seeded_state()
    st.data["records"]["r1"] = record()
    child = record()
    child.update(record_id="r2", arguments=["e2"], parent_record_ids=["r1"])
    st.data["records"]["r2"] = child
    st.observe(observation_payload(), evidence(), "new")
    assert set(st.data["invalid_records"]) == {"r1", "r2"}
    with pytest.raises(MissingCapability):
        st.bound("e1", 1)


def test_binding_withdrawal_invalidates_old_position():
    st = seeded_state()
    st.data["records"]["r1"] = record()
    p = observation_payload()
    p["observations"] = []
    p["links"][0].update(status="distinct", candidate_ids=["c1"], source_observation_ids=["o1"])
    st.observe(p, {}, "withdraw")
    assert "r1" in st.data["invalid_records"]
    with pytest.raises(MissingCapability):
        st.bound("e1", 1)


def test_hypothesis_limit_is_not_a_temporal_history_limit():
    st = SpatialState(spec())
    for i in range(5):
        p, e = observation_payload(), evidence()
        e["F01"]["timestamp_seconds"] = i + 1
        p["links"][0]["valid_time"] = [i + 1, i + 1]
        st.observe(p, e, f"c{i}")
    assert not st.data["unresolved_links"]
    assert st.bound("e1", 1) and st.bound("e1", 5)


def test_identity_does_not_extend_position_valid_time():
    st = seeded_state()
    r = record()
    r["valid_time"] = [1, 2]
    st.data["records"]["r1"] = r
    assert st.bound("e1", 7)
    assert not st.select("bearing", time=7)


@pytest.mark.parametrize(
    "method", ["question_constraint", "predicted_geometry", "deterministic_transform"]
)
def test_model_cannot_forge_given_or_tool_geometry(method):
    st = seeded_state()
    r = record()
    r["method"] = method
    with pytest.raises(ProtocolError):
        st.relations({"frames": [], "records": [r], "gaps": []}, {"f1", "f2", "f3"}, "rel")
    assert not st.data["records"]


def test_builder_must_resee_raw_sources_and_not_upgrade_estimate():
    st = seeded_state()
    r = record(
        "distance", {"interval": [2, 2], "semantics": "closest_boundary"}, ["e1", "e2"], unit="m"
    )
    with pytest.raises(ProtocolError, match="presented"):
        st.relations({"frames": [], "records": [r], "gaps": []}, {"f1"}, "rel")
    st.relations({"frames": [], "records": [r], "gaps": []}, {"f1", "f2", "f3"}, "rel")
    assert st.data["records"]["rel.r1"]["status"] == "estimated"


def test_schema_does_not_accept_arbitrary_code_or_nan():
    s = spec()
    s["nodes"] = [
        {"id": "n", "operation": "exec", "entity_ids": [], "input_nodes": [], "parameters": {}}
    ]
    with pytest.raises(ProtocolError):
        validate(s, SCHEMAS["compile"])
    s = spec()
    s["direction_rule"]["back_threshold_degrees"] = float("nan")
    with pytest.raises(ProtocolError):
        validate(s, SCHEMAS["compile"])


def test_audit_cannot_omit_claims_or_reference_hidden_frames():
    a = {
        "checks": [
            {"record_id": "r1", "verdict": "supported", "frame_ids": ["F01"], "reason": "visible"}
        ],
        "gaps": [],
    }
    with pytest.raises(ProtocolError):
        validate_visual_audit(a, ["r1", "r2"], evidence())
    a["checks"][0]["frame_ids"] = ["F99"]
    with pytest.raises(ProtocolError):
        validate_visual_audit(a, ["r1"], evidence())
