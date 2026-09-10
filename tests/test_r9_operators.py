import math

import pytest

from qwen3vl_agent.r9.operators import Executor, run_query
from qwen3vl_agent.r9.operators.area import floor_area, polygon_area
from qwen3vl_agent.r9.operators.direction import bearing, heading_delta, interval_bearing, turn
from qwen3vl_agent.r9.operators.distance import closest_boundary, convert, metric_scale, nearest
from qwen3vl_agent.r9.operators.route import compare_paths, next_action, replay
from qwen3vl_agent.r9.operators.viewpoint import viewpoint
from qwen3vl_agent.r9.types import InputContract, MissingCapability
from tests.r9_fakes import record, seeded_state, spec


def test_query_window_cannot_silently_reuse_an_unanchored_or_earlier_relation():
    s = spec()
    state = seeded_state(s)
    state.data["records"]["r1"] = record()
    contract = InputContract(((0, 10),))
    for anchor in (None, 1):
        result = run_query(s, state, contract, query_time=anchor, query_scope=(4, 8))
        assert result.value is None and result.gaps[0].kind == "time"
        assert result.gaps[0].time_interval == [4, 8]
    assert run_query(s, state, contract, query_time=5, query_scope=(4, 8)).value == "front-left"


def test_scoped_event_ordinal_keeps_initial_reference_outside_query_window():
    state = seeded_state()
    for i, bounds in enumerate(((1, 2), (5, 6)), 1):
        r = record(
            "event",
            {"description": "turn", "start_s": bounds[0], "end_s": bounds[1], "replay_of": None},
        )
        r["record_id"] = f"r{i}"
        state.data["records"][r["record_id"]] = r
    state.data["coverage"] = [{"span": [4, 8], "sequential": True, "completed": True}]
    executor = Executor(state, state.spec, InputContract(((0, 10),)), query_scope=(4, 8))
    assert executor.event_time("turn", 1) == 6
    assert executor.anchor_time({"kind": "start"}) == 0
    assert executor.anchor_time({"kind": "start"}, query=True) == 4


@pytest.mark.parametrize(
    "point,kind,expected",
    [
        ((-1, 1), "left_right", "left"),
        ((-1, 1), "four_quadrants", "front-left"),
        ((1, -1), "four_quadrants", "back-right"),
        ((-1, -1), "left_right_back", "back"),
        ((1, 1), "left_right_back", "right"),
    ],
)
def test_bearing(point, kind, expected):
    assert bearing((0, 0), (0, 2), point, kind, 135)[0] == expected


@pytest.mark.parametrize(
    "o,f,q",
    [
        ((0, 0), (0, 0), (1, 1)),
        ((0, 0), (0, 1), (0, 0)),
        ((0, 0), (0, 1), (0, 2)),
        ((0, 0), (0, 1), (1, 0)),
    ],
)
def test_bearing_degenerate(o, f, q):
    with pytest.raises(MissingCapability):
        bearing(o, f, q)


@pytest.mark.parametrize(
    "interval,kind,expected",
    [
        ([-60, -30], "four_quadrants", "front-left"),
        ([135, 150], "left_right_back", "back"),
        ([-150, -135], "left_right_back", "back"),
    ],
)
def test_angle_intervals(interval, kind, expected):
    assert interval_bearing(interval, kind, 135) == expected


@pytest.mark.parametrize("interval", [[80, 100], [-10, 10], [134, 136], [-180, 180]])
def test_crossing_angle_gaps(interval):
    with pytest.raises(MissingCapability):
        interval_bearing(
            interval, "left_right_back" if interval == [134, 136] else "four_quadrants", 135
        )


def test_initial_heading_does_not_drift():
    assert heading_delta([170, 170], [-170, -170]) == [20, 20]
    assert heading_delta([0, 0], [90, 90]) == [90, 90]
    assert heading_delta([0, 0], [0, 0]) == [0, 0]
    assert turn(90, 180) == "right"


def test_nearest_scale_free_and_tie():
    assert nearest({"a": [2, 3], "b": [4, 5]}) == "a"
    assert nearest({"a": [20, 30], "b": [40, 50]}) == "a"
    with pytest.raises(MissingCapability):
        nearest({"a": [2, 4], "b": [4, 5]})


def test_boundary_not_center():
    assert closest_boundary([[0, 0], [2, 0]], [[3, 0], [10, 0]]) == 1
    with pytest.raises(MissingCapability):
        closest_boundary([[0, 0]], [[3, 0]], complete_a=False)


def test_unit_conversion_and_squared_scale():
    assert convert([2, 3], "m", "cm") == [200, 300]
    assert convert([2, 2], "arbitrary_scene_unit2", "m2", [3, 3], 2) == [18, 18]
    with pytest.raises(MissingCapability):
        convert([1, 1], "arbitrary_scene_unit", "m")


def test_target_category_prior_is_circular():
    s = {"kind": "category_prior", "reference_entity": "bed", "meters_per_scene_unit": [1, 1]}
    with pytest.raises(MissingCapability):
        metric_scale(s, "bed")
    assert metric_scale(s, "table") == [1, 1]


def test_area_disjoint_coverage_and_revisit():
    a = {"id": "a", "polygon": [[0, 0], [1, 0], [1, 1], [0, 1]]}
    b = {"id": "b", "polygon": [[1, 0], [2, 0], [2, 1], [1, 1]]}
    assert floor_area([a, b], True) == 2
    for regions, coverage in [([a], False), ([a, a], True), ([a, dict(a, id="b")], True)]:
        with pytest.raises(MissingCapability):
            floor_area(regions, coverage)
    with pytest.raises(MissingCapability):
        polygon_area([[0, 0], [1, 1], [0, 1], [1, 0]])


def edges():
    return [
        {
            "from": "a",
            "to": "b",
            "departure_heading": 90,
            "arrival_heading": 90,
            "length": [1, 1],
            "cost_seconds": [2, 2],
        },
        {
            "from": "b",
            "to": "c",
            "departure_heading": 0,
            "arrival_heading": 0,
            "length": [1, 1],
            "cost_seconds": [3, 3],
        },
        {
            "from": "a",
            "to": "c",
            "departure_heading": 0,
            "arrival_heading": 0,
            "length": [3, 3],
            "cost_seconds": [8, 8],
        },
    ]


def test_route_uses_arrival_heading_and_stop():
    assert replay(["a", "b", "c"], 0, edges()) == ["right", "left"]
    p = {
        "current": "b",
        "previous": "a",
        "heading": 90,
        "completed_waypoints": ["b"],
        "instruction_index": 1,
        "stop_reached": False,
    }
    assert next_action(p, ["b", "c"], edges()) == "Turn left and move forward"
    assert next_action(dict(p, stop_reached=True), ["b", "c"], []) == "Stop"
    assert compare_paths([["a", "b", "c"], ["a", "c"]], edges()) == ["a", "b", "c"]
    e = edges()
    e[0]["cost_seconds"] = None
    with pytest.raises(MissingCapability):
        compare_paths([["a", "b", "c"], ["a", "c"]], e)


def test_topology_does_not_supply_turns():
    e = edges()
    e[0]["arrival_heading"] = None
    with pytest.raises(MissingCapability):
        replay(["a", "b", "c"], 0, e)


def test_viewpoint_axes_landmarks_and_screen_parent():
    frames = {
        "cam": {"kind": "inner_camera", "parent_frame_id": "outer"},
        "outer": {"kind": "camera"},
    }
    v = {
        "relation": "opposite finish line",
        "camera_frame": "cam",
        "observer_frame": "outer",
        "landmark_ids": ["arch", "line"],
        "mirror_plane_id": None,
        "axis_defined": True,
    }
    assert viewpoint(v, frames) == "opposite finish line"
    with pytest.raises(MissingCapability):
        viewpoint(dict(v, axis_defined=False), frames)
    frames["cam"]["parent_frame_id"] = None
    with pytest.raises(MissingCapability):
        viewpoint(v, frames)


def test_qualitative_left_does_not_prove_front_or_135():
    s = spec()
    st = seeded_state(s)
    st.data["records"]["r1"] = record(
        value={"horizontal": "left", "depth": "unknown", "angle_interval": None}
    )
    assert run_query(s, st, InputContract(((0, 10),))).value is None
    s["direction_rule"]["kind"] = "left_right"
    assert run_query(s, st, InputContract(((0, 10),))).value == "left"


@pytest.mark.parametrize(
    "op,predicate,value,entities,unit,expected",
    [
        (
            "absolute_distance",
            "distance",
            {"interval": [2, 2], "semantics": "closest_boundary"},
            ["e1", "e2"],
            "m",
            2,
        ),
        (
            "max_extent",
            "extent",
            {"dimensions": [[1, 1], [2, 2], [3, 3]], "complete": True},
            ["e1"],
            "cm",
            3,
        ),
        (
            "floor_area",
            "area",
            {
                "regions": [{"id": "r", "polygon": [[0, 0], [2, 0], [2, 3], [0, 3]]}],
                "coverage_complete": True,
            },
            ["e1"],
            "m2",
            6,
        ),
    ],
)
def test_numeric_query_engine(op, predicate, value, entities, unit, expected):
    s = spec(operation=op)
    s["entities"] = [e for e in s["entities"] if e["id"] in entities]
    s["measurement"] = {
        "geometry_semantics": "closest_boundary" if op == "absolute_distance" else op,
        "unit": unit,
        "metric_required": True,
    }
    st = seeded_state(s)
    r = record(predicate, value, entities, unit=unit)
    r["source_observation_ids"] = list(st.data["observations"])
    st.data["records"]["r1"] = r
    assert math.isclose(run_query(s, st, InputContract(((0, 10),))).value, expected)


def test_event_ordinal_requires_prefix_scan():
    s = spec()
    s["nodes"] = [
        {
            "id": "event",
            "operation": "event_select",
            "entity_ids": [],
            "input_nodes": [],
            "parameters": {"event_description": "generator damaged", "ordinal": 2},
        }
    ]
    s["output_node"] = "event"
    st = seeded_state(s)
    for n, t in enumerate((2, 5), 1):
        r = record(
            "event",
            {"description": "generator damaged", "start_s": t - 1, "end_s": t, "replay_of": None},
        )
        r["record_id"] = f"r{n}"
        st.data["records"][f"r{n}"] = r
    assert run_query(s, st, InputContract(((0, 10),))).value is None
    st.data["coverage"] = [{"span": [0, 5], "sequential": True, "completed": True}]
    assert run_query(s, st, InputContract(((0, 10),))).value == 5


def test_image_points_cannot_become_scene_geometry():
    s, st = spec(), seeded_state()
    for i, p in enumerate([[0, 0], [0, 2], [-1, 1]], 1):
        r = record("point", {"coordinates": p, "space": "image"}, [f"e{i}"], unit="pixel")
        r["record_id"] = f"r{i}"
        st.data["records"][f"r{i}"] = r
    assert run_query(s, st, InputContract(((0, 10),))).value is None
