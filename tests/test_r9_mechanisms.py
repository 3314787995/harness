"""Each mechanism traverses source validation, query execution and raw-frame audit."""

import json

import pytest

from qwen3vl_agent.models.base import ModelOutput
from qwen3vl_agent.r9 import R9Request, R9VideoAgent
from qwen3vl_agent.r9.operators import run_query
from qwen3vl_agent.r9.operators.area import floor_area
from qwen3vl_agent.r9.operators.distance import convert
from qwen3vl_agent.r9.types import InputContract, MissingCapability
from tests.r9_fakes import FakeModel, config, make_video, record, seeded_state, spec


def task_for(op):
    s = spec("Inspect the spatial relation at 3 seconds.", op)
    if op in {"heading_delta", "max_extent", "floor_area"}:
        s["entities"] = s["entities"][:1]
        s["query_frame"]["forward_role"] = None
    if op == "heading_delta":
        s["time"] = {
            "reference_anchor": {
                "kind": "start",
                "time_s": None,
                "description": "",
                "ordinal": None,
            },
            "query_anchor": {"kind": "time", "time_s": 3, "description": "", "ordinal": None},
        }
    if op == "nearest_object":
        s["measurement"]["geometry_semantics"] = "center"
    if op == "max_extent":
        s["measurement"] = {
            "geometry_semantics": "max_extent",
            "unit": "cm",
            "metric_required": True,
        }
    if op == "floor_area":
        s["measurement"] = {
            "geometry_semantics": "floor_area",
            "unit": "m2",
            "metric_required": True,
        }
    if op == "fill_turns":
        s["route"] = {
            "start_entity": "e1",
            "forward_entity": "e2",
            "waypoints": ["e2", "e3"],
            "candidate_paths": [],
            "stop_condition": "reach e3",
            "turn_indices": [1],
        }
    return s


class MechanismModel(FakeModel):
    def generate(self, messages, **kwargs):
        output = super().generate(messages, **kwargs)
        body = self.calls[-1]["body"]
        stage, p = body["stage"], body["input"]
        if stage == "observe" and self.task["operation"] == "heading_delta":
            value = json.loads(output.text)
            first = value["observations"][0]
            last = dict(first, id="o2", frame_id=list(p["frames"])[-1])
            value["observations"].append(last)
            value["links"][0]["source_observation_ids"].append("o2")
            return ModelOutput(json.dumps(value), output.metadata)
        if stage != "relations":
            return output
        observations = p["state"]["observations"]
        ids = [o["id"] for o in observations]
        times = [o["timestamp_s"] for o in observations]
        records, frames = [], []

        def add(predicate, value, args, unit=None, source_ids=None, valid_time=None):
            r = record(predicate, value, args, unit=unit)
            r.update(
                record_id=f"r{len(records) + 1}",
                source_observation_ids=source_ids or ids,
                valid_time=valid_time or [min(times), max(times)],
            )
            records.append(r)

        op = self.task["operation"]
        if op == "bearing":
            add(
                "bearing",
                {"horizontal": "left", "depth": "front", "angle_interval": None},
                ["e1", "e2", "e3"],
            )
        elif op == "heading_delta":
            ordered = sorted(observations, key=lambda o: o["timestamp_s"])
            for o, degrees in ((ordered[0], 0), (ordered[-1], 90)):
                add(
                    "heading",
                    {
                        "degrees": [degrees, degrees],
                        "kind": "body",
                        "anchor": "test",
                        "position_entity": "e1",
                    },
                    ["e1"],
                    source_ids=[o["id"]],
                    valid_time=[o["timestamp_s"], o["timestamp_s"]],
                )
        elif op == "nearest_object":
            for e, d in (("e2", 1), ("e3", 3)):
                add(
                    "distance",
                    {"interval": [d, d], "semantics": "center"},
                    ["e1", e],
                    "arbitrary_scene_unit",
                )
        elif op == "max_extent":
            add(
                "extent",
                {"dimensions": [[10, 10], [20, 20], [30, 30]], "complete": True},
                ["e1"],
                "cm",
            )
        elif op == "floor_area":
            add(
                "area",
                {
                    "regions": [{"id": "room", "polygon": [[0, 0], [2, 0], [2, 3], [0, 3]]}],
                    "coverage_complete": True,
                },
                ["e1"],
                "m2",
            )
        elif op == "fill_turns":
            add(
                "progress",
                {
                    "current": "e1",
                    "previous": None,
                    "heading": 0,
                    "completed_waypoints": [],
                    "instruction_index": 0,
                    "stop_reached": False,
                },
                ["e1"],
            )
            for a, b, heading in (("e1", "e2", 0), ("e2", "e3", 90)):
                add(
                    "edge",
                    {
                        "from": a,
                        "to": b,
                        "departure_heading": heading,
                        "arrival_heading": heading,
                        "length": None,
                        "cost_seconds": None,
                    },
                    [a, b],
                )
        elif op == "viewpoint_relation":
            frames = [
                {
                    "id": "camera",
                    "kind": "camera",
                    "component_id": "scene",
                    "plane": "3d",
                    "parent_frame_id": "query",
                    "source_observation_ids": ids,
                }
            ]
            add(
                "viewpoint",
                {
                    "relation": "opposite finish line",
                    "camera_frame": "camera",
                    "observer_frame": "query",
                    "landmark_ids": ["e2", "e3"],
                    "mirror_plane_id": None,
                    "axis_defined": True,
                },
                ["e1", "e2", "e3"],
            )
        return ModelOutput(
            json.dumps({"frames": frames, "records": records, "gaps": []}), output.metadata
        )


@pytest.mark.parametrize(
    "op,choices,expected,unit",
    [
        ("bearing", ["front-left", "back-right"], "A", None),
        ("heading_delta", ["right 90 degrees", "unchanged"], "A", None),
        ("nearest_object", ["sofa", "tv"], "A", None),
        ("max_extent", [], 30, "cm"),
        ("floor_area", [], 6, "m2"),
        ("fill_turns", ["right", "left"], "A", None),
        ("viewpoint_relation", ["opposite finish line", "beside the start"], "A", None),
    ],
)
def test_mechanism_full_loop(tmp_path, op, choices, expected, unit):
    p = make_video(tmp_path / "v.mp4")
    s = task_for(op)
    req = R9Request(
        str(p),
        s["source_spans"][0]["text"],
        choices=choices,
        output_unit=unit,
        fixed_frames=(0, 1, 2, 3),
        comparison="fixed_evidence",
    )
    model = MechanismModel(task=s)
    result = R9VideoAgent(model, config(tmp_path)).solve(req)
    assert result.prediction == expected
    assert not result.forced_answer
    assert result.trace["verification"]["can_stop"]
    assert [c["body"]["stage"] for c in model.calls] == ["compile", "observe", "relations", "audit"]


def test_stop_before_unneeded_object_binding():
    s = spec(operation="next_action")
    s["route"] = {
        "start_entity": "e1",
        "forward_entity": "e2",
        "waypoints": ["e2", "e3"],
        "candidate_paths": [],
        "stop_condition": "before the fork",
    }
    st = seeded_state(s)
    del st.data["entity_links"]["e3"]
    st.data["records"]["r1"] = record(
        "progress",
        {
            "current": "e1",
            "previous": None,
            "heading": 0,
            "completed_waypoints": [],
            "instruction_index": 0,
            "stop_reached": True,
        },
        ["e1"],
    )
    assert run_query(s, st, InputContract(((0, 10),))).value == "Stop"


def test_area_inside_diamond_overlap_and_unit_dimension():
    a = {"id": "a", "polygon": [[0, 0], [2, 0], [2, 2], [0, 2]]}
    b = {"id": "b", "polygon": [[1, 0], [2, 1], [1, 2], [0, 1]]}
    with pytest.raises(MissingCapability):
        floor_area([a, b], True)
    with pytest.raises(MissingCapability):
        convert([1, 1], "m2", "cm")


def test_terminal_reserve_survives_spent_visual_budget(tmp_path):
    p = make_video(tmp_path / "v.mp4")
    req = R9Request(str(p), "q", choices=["left", "right"], max_visual_exposures=4)
    model = FakeModel(missing=True)
    result = R9VideoAgent(model, config(tmp_path)).solve(req)
    assert result.forced_answer and result.prediction == "A"
    assert result.trace["resources"]["visual_exposures"] == 4
    assert model.calls[-1]["body"]["stage"] == "answer"
    assert len(model.calls[-1]["messages"][-1]["content"]) == 1


def test_invalid_request_result_never_calls_model(tmp_path):
    model = FakeModel()
    result = R9VideoAgent(model, config(tmp_path)).solve(
        {"video_path": "v", "question": "q", "answer": "A"}
    )
    assert result.status == "invalid_input" and not model.calls
