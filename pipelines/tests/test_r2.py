"""R2 invariants using controlled observations and real CPU-decoded video; no model weights."""

import copy
import json
from dataclasses import replace
from fractions import Fraction
from types import SimpleNamespace

import pytest

from qwen3vl_agent.models.base import BaseVideoModel, ModelOutput
from qwen3vl_agent.models.qwen3vl import Qwen3VLModel
from qwen3vl_agent.r2 import InputContract, R2Budget, R2Config, R2Request, R2VideoAgent
from qwen3vl_agent.r2.agent import external_navigation
from qwen3vl_agent.r2.contracts import parse, validate_observation, validate_query
from qwen3vl_agent.r2.evaluate import preflight, read_manifest, run_manifest
from qwen3vl_agent.r2.media import R2Media, source_point
from qwen3vl_agent.r2.planning import make_windows, sample_times, select_action
from qwen3vl_agent.r2.reduce import reduce_operation
from qwen3vl_agent.r2.runtime import ModelSession
from qwen3vl_agent.r2.state import StateStore
from qwen3vl_agent.r2.types import BudgetExhausted, ProtocolError


def query(op="direction_sequence", parameters=None, slots=None):
    slots = slots or [
        {
            "id": "S1",
            "target_id": "T1",
            "property": "position",
            "description": "Read the target position",
            "reference_frame": "screen",
        }
    ]
    return {
        "targets": [{"id": "T1", "description": "the red object"}],
        "slots": slots,
        "operations": [
            {
                "id": "Q1",
                "op": op,
                "target_ids": ["T1"],
                "slot_ids": [s["id"] for s in slots],
                "parameters": parameters or {},
            }
        ],
        "scope": {"kind": "full", "interval": None, "description": ""},
        "anchors": [],
        "fast_motion": False,
        "unresolved": [],
    }


def observation(records, entities=None, **kwargs):
    return {
        "entities": entities
        or [{"id": "E1", "target_id": "T1", "description": "red object", "part_of": ""}],
        "records": records,
        "associations": [],
        "containments": [],
        "gaps": [],
        "reference_status": "stable",
        "candidate_coverage_complete": True,
        "complete": True,
        **kwargs,
    }


def record(index, value=None, **kwargs):
    return {
        "slot_id": "S1",
        "entity_id": "E1",
        "frame_id": f"F{index + 1:02d}",
        "visibility": "visible",
        "value": value,
        "basis": "visual_observation",
        **kwargs,
    }


def frame_map(times, prefix="f"):
    return {
        f"F{i + 1:02d}": {
            "id": f"{prefix}{i}",
            "source_frame_id": f"{prefix}{i}",
            "timestamp_seconds": t,
            "source_size": [1000, 500],
            "view_box": [0, 0, 1000, 500],
        }
        for i, t in enumerate(times)
    }


def store_for(q, records, times=None, **kwargs):
    times = times or list(range(len(records)))
    state = StateStore()
    frames = frame_map(times)
    coverage = {"span": [min(times), max(times)], "completed": True, "resolution_met": True}
    value = observation(records, **kwargs)
    state.ingest("w", value, frames, q, coverage, "call1")
    return state


def reduced(q, state, **kwargs):
    return reduce_operation(q["operations"][0], q, state, R2Config(), **kwargs)


@pytest.fixture
def video(tmp_path):
    import av
    from PIL import Image, ImageDraw

    path = tmp_path / "controlled.mp4"
    with av.open(str(path), "w") as output:
        stream = output.add_stream("mpeg4", rate=8)
        stream.width, stream.height, stream.pix_fmt = 96, 64, "yuv420p"
        for i in range(40):
            image = Image.new("RGB", (96, 64), "black")
            ImageDraw.Draw(image).rectangle((4 + i, 20, 12 + i, 28), fill="red")
            f = av.VideoFrame.from_image(image)
            f.pts, f.time_base = i, Fraction(1, 8)
            for packet in stream.encode(f):
                output.mux(packet)
        for packet in stream.encode():
            output.mux(packet)
    return path


@pytest.fixture
def config(tmp_path):
    return R2Config.from_mapping({"media": {"cache_dir": str(tmp_path / "cache")}})


class FakeQwen(BaseVideoModel):
    def __init__(
        self, q=None, invalid_role=None, interrupt_role=None, observer_hook=None, final_hook=None
    ):
        super().__init__("fake-qwen")
        self.q = q or query()
        self.payloads = []
        self.invalid_role, self.interrupt_role = invalid_role, interrupt_role
        self.observer_hook, self.final_hook = observer_hook, final_hook

    def load(self):
        self._loaded = True

    def unload(self):
        self._loaded = False

    def generate(self, messages, **kwargs):
        parts = messages[0]["content"]
        body = json.loads(parts[-1]["text"].split("\n", 1)[1])
        role, payload = body["stage"], body["input"]
        self.payloads.append((role, copy.deepcopy(payload), copy.deepcopy(parts)))
        if self.interrupt_role == role:
            self.interrupt_role = None
            raise KeyboardInterrupt("controlled interruption")
        if self.invalid_role == role:
            return ModelOutput("not JSON")
        if role.startswith("compile"):
            value = self.q
        elif role == "locate":
            fs = payload["frames"]
            value = {
                "candidates": [
                    {
                        "start_frame": fs[0]["frame_id"],
                        "end_frame": fs[-1]["frame_id"],
                        "reason": "target",
                    }
                ],
                "unresolved": [],
            }
        elif role == "observe":
            records = []
            for f in payload["frames"]:
                i = int(f["frame_id"][1:]) - 1
                records.append(record(i, "red", point=[100 + f["source_seconds"] * 60, 500]))
            value = observation(records)
            previous = payload.get("handoff", {}).get("entities", [])
            if previous:
                value["associations"] = [
                    {
                        "group_id": "G1",
                        "alternatives": [
                            {
                                "links": [
                                    {
                                        "from_node": e["node_id"],
                                        "to_entity": "E1",
                                        "kind": "same_entity",
                                    }
                                    for e in previous
                                ],
                                "evidence_frames": sorted(
                                    {
                                        fid
                                        for requirement in payload.get("identity_requirements", [])
                                        for fid in requirement["shared_frame_ids"]
                                    }
                                )
                                or [payload["frames"][0]["frame_id"]],
                            }
                        ],
                        "supersedes": [],
                        "unresolved_extra": False,
                        "relation_preserved": True,
                    }
                ]
            if self.observer_hook:
                value = self.observer_hook(value, payload)
            # The fake emits only fields available in this operation's wire contract.
            schema = body["output_schema"]
            value = {k: v for k, v in value.items() if k in schema["properties"]}
            fields = schema["properties"]["records"]["items"]["properties"]
            value["records"] = [
                {k: v for k, v in r.items() if k in fields} for r in value["records"]
            ]
        else:
            refs = [r["id"] for r in payload["observations"][:1]]
            operation_ids = []
            policy = payload.get("assessment_policy")
            if policy:
                refs = []
                if policy["complete_support_allowed"]:
                    for oid, op in policy["operations"].items():
                        if op["status"] == "supported" and op["evidence_ids"]:
                            operation_ids = [oid]
                            refs = op["evidence_ids"]
                            break
            labels = [c["label"] for c in payload["options"]]
            value = {
                "prediction": labels[0] if labels else "right",
                "evidence_ids": refs,
                "assessments": [
                    {
                        "label": label,
                        "status": ("supported" if i == 0 else "contradicted")
                        if refs
                        else "unknown",
                        "evidence_ids": refs,
                        **({"operation_ids": operation_ids} if policy else {}),
                    }
                    for i, label in enumerate(labels)
                ],
                "weakest_premise": "target correspondence",
                "recheck": None,
                "unresolved": [],
            }
            if self.final_hook:
                value = self.final_hook(value, payload)
        return ModelOutput(
            json.dumps(value), {"input_tokens": 11, "output_tokens": 7, "visual_tokens": 3}
        )


def test_direction_retains_reversal_with_identical_endpoints():
    q = query()
    state = store_for(
        q, [record(i, point=[x, 500]) for i, x in enumerate([200, 420, 650, 440, 210])]
    )
    result = reduced(q, state)
    assert result["status"] == "supported"
    assert result["value"]["direction_segments"] == ["right", "left"]


def test_endpoint_net_change_and_role_correspondence():
    q = query("endpoint_delta", {"allow_role_correspondence": True})
    state = store_for(q, [record(0, "red"), record(1, "green")])
    assert reduced(q, state)["value"]["S1"]["after"] == "green"
    assert reduced(q, state)["value"]["S1"]["changed"] is True


def test_overlapping_onsets_are_not_simultaneous():
    slots = [
        {
            "id": sid,
            "target_id": "T1",
            "property": sid,
            "description": "presence",
            "reference_frame": "screen",
        }
        for sid in ["root", "shoot"]
    ]
    q = query("state_sequence", slots=slots)
    records = [
        record(0, False, slot_id="root"),
        record(2, True, slot_id="root"),
        record(1, False, slot_id="shoot"),
        record(3, True, slot_id="shoot"),
    ]
    state = store_for(q, records, [0, 1, 2, 3])
    result = reduced(q, state)
    assert any(g["kind"] == "stage_order" for g in result["gaps"])
    assert result["value"]["established_order"] == []


def test_action_relation_sequence_preserves_order():
    q = query("relation_transition")
    state = store_for(
        q,
        [
            record(0, "arm outside sleeve"),
            record(1, "arm entering sleeve"),
            record(2, "arm inside sleeve"),
        ],
    )
    result = reduced(q, state)
    assert [x["value"] for x in result["value"]["sequences"]["S1"]][-1] == "arm inside sleeve"


def test_motion_filter_requires_motion_and_occlusion_blocks_negative():
    q = query("motion_condition_filter")
    state = store_for(
        q, [record(0, "circle", point=[100, 100]), record(1, "circle", point=[300, 100])]
    )
    assert reduced(q, state)["value"]["exists"]
    state = store_for(q, [record(0, None, visibility="occluded", basis="unresolved")])
    assert reduced(q, state)["status"] == "unresolved"


def test_path_preserves_rectangle_aspect():
    q = query("path_shape")
    state = store_for(
        q,
        [
            record(i, point=p)
            for i, p in enumerate([[100, 200], [700, 200], [700, 600], [100, 600], [100, 200]])
        ],
    )
    value = reduced(q, state)["value"]
    p = value["ordered_path"]
    assert value["closed"]
    assert (p[1][0] - p[0][0]) / (p[2][1] - p[1][1]) == pytest.approx(3)


def test_rotation_angle_wrap_and_alias():
    q = query("rotation_pattern", {"rotation_type": "self_spin"})
    records = [
        record(
            i,
            orientation_angle=a,
            feature_identifiable=True,
            adjacency_resolved=True,
            rotation_type="self_spin",
        )
        for i, a in enumerate([170, -170, -150])
    ]
    state = store_for(q, records)
    assert reduced(q, state)["value"]["signed_deltas_deg"] == [20, 20]
    records[1]["adjacency_resolved"] = False
    assert reduced(q, store_for(q, records))["status"] == "unresolved"


def test_speed_uses_real_time_differences():
    q = query("motion_property_trend", {"metric": "speed"})
    records = [record(i, point=[100 + i * 100, 100]) for i in range(4)]
    a = reduced(q, store_for(q, records, [0, 1, 2, 3]))["value"]["local_displacement_rates"]
    b = reduced(q, store_for(q, records, [0, 0.5, 1, 1.5]))["value"]["local_displacement_rates"]
    assert b[0][1] == pytest.approx(2 * a[0][1])


def test_frequency_and_amplitude_have_distinct_units():
    q = query("motion_property_trend", {"metric": "frequency", "phase_unit": "full circle"})
    records = [record(i, "top", phase="top", cycle_marker=True) for i in range(3)]
    result = reduced(q, store_for(q, records, [0, 1, 1.5]))
    assert result["value"]["measurements"] == [(1, 1), (1.5, 2)]
    q = query("motion_property_trend", {"metric": "amplitude"})
    q["slots"][0]["reference_frame"] = "body"
    records = [
        record(i, phase="peak", point=[500, y], reference_point=[500, 500], scale=500)
        for i, y in enumerate([400, 300, 200])
    ]
    result = reduced(q, store_for(q, records))
    assert result["status"] == "supported"
    assert result["value"]["trend"] == ["increase"]


def test_phase_continuation_and_drift():
    q = query("periodic_continuation")
    state = store_for(q, [record(i, phase=p) for i, p in enumerate("ABCABCA")])
    assert reduced(q, state)["value"]["next_phase"] == "B"
    state = store_for(q, [record(i, phase=p) for i, p in enumerate("ABCABD")])
    assert reduced(q, state)["status"] == "unresolved"


def identity_store(alternatives=1, reveal_at_end=False, unrelated=False):
    q = query("identity_at_time")
    store = StateStore()
    entities = [
        {"id": f"E{i}", "target_id": "T1", "description": "cup", "part_of": ""} for i in range(1, 4)
    ]
    for w, t in [("a", 0), ("b", 1)]:
        records = [record(0, entity_id=f"E{i}", rank=i) for i in range(1, 4)]
        containment = [
            {
                "hidden_target_id": "T1",
                "carrier_entity_id": "E1",
                "frame_id": "F01",
                "status": "visible_reveal",
            }
        ]
        associations = []
        if w == "b":
            choices = []
            for i in range(alternatives):
                perm = [1, 2, 3] if i % 2 == 0 else [1, 3, 2] if unrelated else [2, 1, 3]
                choices.append(
                    {
                        "links": [
                            {"from_node": f"a/E{j}", "to_entity": f"E{k}", "kind": "same_entity"}
                            for j, k in enumerate(perm, 1)
                        ],
                        "evidence_frames": ["F01"],
                    }
                )
            associations = [
                {
                    "group_id": "G",
                    "alternatives": choices,
                    "supersedes": [],
                    "unresolved_extra": False,
                    "relation_preserved": True,
                }
            ]
        value = observation(
            records,
            entities,
            associations=associations,
            containments=containment if (w == "b") == reveal_at_end else [],
        )
        store.ingest(
            w,
            value,
            frame_map([t], w),
            q,
            {"span": [t, t + 0.1], "completed": True, "resolution_met": True},
            w,
        )
    return q, store


@pytest.mark.parametrize("backward", [False, True])
def test_identity_propagates_in_both_directions(backward):
    q, store = identity_store(reveal_at_end=backward)
    value = reduced(q, store, query_time=0 if backward else 1)
    assert value["status"] == "supported"
    assert value["value"]["possible_ranks"] == [1]
    assert value["value"]["propagation"] == ("backward" if backward else "forward")


def test_identity_ambiguity_overflow_and_irrelevant_swaps():
    q, store = identity_store(2)
    assert reduced(q, store, query_time=1)["status"] == "unresolved"
    q, store = identity_store(9)
    assert reduced(q, store, query_time=1)["value"]["unexpanded_ambiguity"]
    q, store = identity_store(2, unrelated=True)
    assert reduced(q, store, query_time=1)["status"] == "supported"


def test_contract_disjoint_intervals_and_future_query():
    request = R2Request(
        "v",
        "q",
        allowed_time_intervals=((0, 2), (4, 6)),
        observation_cutoff=5,
        protocol_id="strict_prefix",
        query_time=8,
    )
    contract = InputContract.resolve(request, 10)
    assert contract.allowed_time_intervals == ((0, 2), (4, 5))
    assert not contract.permits_span((1, 4.5))
    assert not contract.permits(5.01)
    with pytest.raises(ValueError):
        R2Request("v", "q", protocol_id="strict_prefix")


def test_media_pts_cutoff_shared_frames_and_coordinate_mapping(video, config):
    request = R2Request(str(video), "q", observation_cutoff=1.01)
    contract = InputContract.resolve(request, 5)
    media = R2Media(config)
    a = media.extract(video, (0.25, 1.01), [0.25, 0.5, 0.75, 1.01], contract, fps=4)
    assert all(0.25 <= f.timestamp_seconds <= 1.01 for f in a.frames)
    assert all(
        m["pts"] * m["time_base"][0] / m["time_base"][1] == pytest.approx(m["timestamp_seconds"])
        for m in media.catalog.values()
    )
    b = media.extract(video, (0.5, 1.01), [0.625, 0.875], contract, fps=4, anchors=a.frames[1:])
    assert {f.id for f in a.frames[1:]} <= {f.id for f in b.frames}
    prepared = media.prepare(a)
    assert prepared.video_frame_metadata[0]["frames_indices"][0] == 2
    cropped = media.fixed_crop(a, [250, 250, 750, 750])
    meta = media.catalog[cropped.frames[0].id]
    assert source_point([500, 500], meta) == [48, 32]
    restored = media.extract(video, (0.25, 1.01), [0.5, 0.75], contract, anchors=cropped.frames)
    assert all(f.id == media.catalog[f.id]["source_frame_id"] for f in restored.frames)
    assert {f.id for f in a.frames} <= {f.id for f in restored.frames}
    with pytest.raises(ValueError):
        media.extract(video, (0, 2), [1.5], contract)


def test_native_timing_checks_reject_padding_and_fake_fps():
    tensor = SimpleNamespace(shape=(4, 3, 32, 32))
    selected = [
        {
            "fps": 8,
            "frames_indices": [8, 10, 12, 14],
            "source_timestamps": [1, 1.25, 1.5, 1.75],
            "frame_ids": ["a", "b", "c", "d"],
            "total_num_frames": 15,
        }
    ]
    assert Qwen3VLModel._validated_frame_metadata([tensor], selected)[0]["frames_indices"][0] == 8
    with pytest.raises(ValueError):
        Qwen3VLModel._validated_frame_metadata([SimpleNamespace(shape=(6,))], selected)
    selected[0]["fps"] = 4
    with pytest.raises(ValueError):
        Qwen3VLModel._validated_frame_metadata([tensor], selected)


def test_controller_options_hidden_and_checkpoint_resume(video, config, tmp_path):
    model = FakeQwen()
    request = R2Request(
        str(video),
        "Which direction?",
        choices=[
            {"label": "X", "text": "The red object moves right."},
            {"label": "Z", "text": "It moves left."},
        ],
        checkpoint_path=str(tmp_path / "checkpoint.jsonl"),
    )
    agent = R2VideoAgent(model, config=config)
    result = agent.solve(request)
    assert result.prediction == "X"
    assert result.completion_state == "complete", result.unresolved_items
    observer_payloads = [p for role, p, _ in model.payloads if role == "observe"]
    assert observer_payloads
    assert all(
        "options" not in p and "The red object moves right." not in json.dumps(p)
        for p in observer_payloads
    )
    previous = len(model.payloads)
    restored = agent.solve(replace(request, resume=True))
    assert restored.to_dict() == result.to_dict()
    assert restored.option_assessments == result.option_assessments
    assert len(model.payloads) == previous


def test_invalid_final_does_not_default_to_first_choice(video, config):
    model = FakeQwen(invalid_role="final")
    result = R2VideoAgent(model, config=config).solve(
        R2Request(str(video), "q", choices=["A: right", "B: left"])
    )
    assert result.prediction is None
    assert result.completion_state == "failed"
    assert result.resources["terminal_calls"] == 2


def test_interrupted_observer_resume_is_charged(video, config, tmp_path):
    model = FakeQwen(interrupt_role="observe")
    request = R2Request(str(video), "q", checkpoint_path=str(tmp_path / "checkpoint.jsonl"))
    with pytest.raises(KeyboardInterrupt):
        R2VideoAgent(model, config=config).solve(request)
    result = R2VideoAgent(model, config=config).solve(replace(request, resume=True))
    assert result.prediction == "right"
    assert any(c["status"] == "interrupted" for c in result.trace["calls"])
    assert result.resources["frame_exposures"] > result.resources["unique_source_frames"]


def test_manifest_gold_is_sidecar_only_and_preflight_decodes(video, config, tmp_path):
    path = tmp_path / "requests.jsonl"
    path.write_text(
        json.dumps(
            {
                "request_id": "r2-real",
                "pipeline_id": "R2",
                "video_path": video.name,
                "question": "direction?",
                "gold": "SECRET_GOLD",
                "evidence_time": 3.5,
            }
        )
        + "\n"
    )
    rows = read_manifest(path)
    assert rows[0][1]["gold"] == "SECRET_GOLD"
    report = preflight(rows, config)
    assert report["ready"] == 1, report
    assert report["model_calls"] == 0 and report["accuracy"] is None
    model = FakeQwen()
    summary = run_manifest(R2VideoAgent(model, config=config), rows, tmp_path / "run")
    assert summary["predictions"] == 1
    assert "SECRET_GOLD" not in json.dumps(model.payloads)


def test_subtitle_cue_crossing_cutoff_is_not_exposed(tmp_path):
    path = tmp_path / "subs.jsonl"
    path.write_text(
        json.dumps({"start_sec": 0, "end_sec": 1, "text": "allowed"})
        + "\n"
        + json.dumps({"start_sec": 0.5, "end_sec": 2, "text": "FUTURE_SECRET"})
        + "\n"
    )
    request = R2Request(
        "v",
        "q",
        subtitle_path=str(path),
        available_modalities=("video", "subtitle"),
        observation_cutoff=1,
    )
    result = external_navigation(request, InputContract.resolve(request, 3))
    assert len(result) == 1 and result[0]["text"] == "allowed"


def test_runtime_reserves_terminal_calls(config):
    state = {"calls": [], "stages": {}}
    session = ModelSession(FakeQwen(), config, R2Budget(max_model_calls=4), state, lambda: None)
    session.call("a", "compile_intent", {})
    session.call("b", "compile_intent", {})
    with pytest.raises(BudgetExhausted):
        session.call("c", "compile_intent", {})


def test_bad_observer_reference_and_duplicate_json_rejected():
    q = query()
    value = observation([record(8)])
    with pytest.raises(ProtocolError):
        validate_observation(value, q, frame_map([0]), set())
    with pytest.raises(ProtocolError):
        parse('{"x":1,"x":2}', "observe")


def test_compiler_cannot_change_scope_from_options():
    q = query()
    modified = copy.deepcopy(q)
    modified["scope"] = {"kind": "interval", "interval": [0, 999], "description": ""}
    with pytest.raises(ProtocolError):
        validate_query(modified, previous=q)


def test_sampling_caps_source_rate_and_phase_shift(config):
    contract = InputContract.resolve(R2Request("v", "q"), 10)
    w = make_windows([(0, 2)], contract, config, refined=True, shifted=True)[0]
    times, fps = sample_times(w, source_fps=8)
    assert fps == 8 and times[0] == pytest.approx(1 / 16)
    action = select_action(
        [{"kind": "phase_alias", "description": "alias", "span": [0, 1]}],
        [(0, 2)],
        contract,
        [],
        0,
        config,
    )
    assert action["action"] == "phase_shift"


def test_role_correspondence_does_not_merge_physical_entities_in_same_view():
    q = query("endpoint_delta", {"allow_role_correspondence": True})
    store = store_for(q, [record(0, "red"), record(1, "red")])
    es = [
        {"id": "E1", "target_id": "T1", "description": "old rod", "part_of": ""},
        {"id": "E2", "target_id": "T1", "description": "new rod", "part_of": ""},
    ]
    value = observation(
        [record(0, "red"), record(1, "green", entity_id="E2")],
        es,
        associations=[
            {
                "group_id": "roles",
                "alternatives": [
                    {
                        "links": [
                            {"from_node": "w/E1", "to_entity": "E1", "kind": "same_entity"},
                            {"from_node": "w/E1", "to_entity": "E2", "kind": "same_role"},
                        ],
                        "evidence_frames": ["F01", "F02"],
                    }
                ],
                "supersedes": [],
                "unresolved_extra": False,
                "relation_preserved": False,
            }
        ],
    )
    store.ingest(
        "next",
        value,
        frame_map([1, 2], "n"),
        q,
        {"span": [1, 2], "completed": True, "resolution_met": True},
        "c",
    )
    assert reduced(q, store)["status"] == "supported"
    q["operations"][0]["parameters"]["allow_role_correspondence"] = False
    assert reduced(q, store)["status"] == "unresolved"


def test_corrected_association_invalidates_derivation_without_overwriting_observations():
    q, store = identity_store(2)
    old = copy.deepcopy(store.observations)
    store.save_derived(reduced(q, store, query_time=1))
    value = observation(
        [record(0, rank=1)],
        associations=[
            {
                "group_id": "corrected",
                "alternatives": [
                    {
                        "links": [
                            {"from_node": "a/E1", "to_entity": "E1", "kind": "same_entity"},
                            {"from_node": "b/E1", "to_entity": "E1", "kind": "same_entity"},
                        ],
                        "evidence_frames": ["F01"],
                    }
                ],
                "supersedes": ["b/G"],
                "unresolved_extra": False,
                "relation_preserved": True,
            }
        ],
    )
    store.ingest(
        "correct",
        value,
        frame_map([2], "correct"),
        q,
        {"span": [2, 2.1], "completed": True, "resolution_met": True},
        "c",
    )
    assert store.observations[: len(old)] == old
    assert not store.derived[0]["valid"]
    assert reduced(q, store, query_time=1)["status"] == "supported"
    assert store.identity_domain("b/E1")["resolved"]


def test_possible_transfer_blocks_carrier_propagation():
    q, store = identity_store()
    store.containments.append(
        {"hidden_target_id": "T1", "status": "possible_transfer", "timestamp": 0.5}
    )
    assert reduced(q, store, query_time=1)["status"] == "unresolved"


def test_unknown_body_reference_blocks_speed():
    q = query("motion_property_trend", {"metric": "speed"})
    q["slots"][0]["reference_frame"] = "body"
    state = store_for(
        q, [record(i, point=[100 * i, 500]) for i in range(3)], reference_status="moving"
    )
    result = reduced(q, state)
    assert any(g["kind"] == "reference" for g in result["gaps"])


def test_odd_and_unrepresentable_frames_use_timestamped_images(video, config):
    media = R2Media(config)
    contract = InputContract.resolve(R2Request(str(video), "q"), 5)
    batch = media.extract(video, (0, 1), [0, 0.25, 0.5], contract)
    assert media.prepare(batch).kind == "timestamped_images"
    batch = media.extract(video, (0, 1), [0, 0.25, 0.5, 0.75], contract)
    media.catalog[batch.frames[1].id]["source_frame_index"] = None
    assert media.prepare(batch).kind == "timestamped_images"


def test_final_recheck_uses_remaining_loop_and_stops_at_two_calls(video, config):
    counter = [0]

    def final_hook(value, payload):
        counter[0] += 1
        if counter[0] == 1:
            value["recheck"] = {
                "kind": "temporal_resolution",
                "description": "inspect local turn",
                "span": [1, 1.5],
            }
        return value

    model = FakeQwen(final_hook=final_hook)
    result = R2VideoAgent(model, config=config).solve(R2Request(str(video), "direction?"))
    assert result.resources["terminal_calls"] == 2
    assert len(result.trace["actions"]) == 1
    assert result.trace["actions"][0]["complete"]
    assert all(
        len(w["source_frame_ids"]) <= 16
        for role, p, _ in model.payloads
        if role == "final"
        for w in p["raw_windows"]
    )


def test_no_progress_action_is_not_rescheduled(config):
    contract = InputContract.resolve(R2Request("v", "q"), 5)
    issues = [{"kind": "detail", "description": "unreadable", "span": [1, 2]}]
    a = select_action(issues, [(0, 5)], contract, [], "same-semantic-state", config)
    assert (
        select_action(issues, [(0, 5)], contract, [a["signature"]], "same-semantic-state", config)
        is None
    )


def test_failed_observer_format_repair_keeps_original_media(video, config):
    model = FakeQwen(invalid_role="observe")
    result = R2VideoAgent(model, config=config).solve(R2Request(str(video), "q"))
    calls = [c for c in result.trace["calls"] if c["role"] == "observe"]
    assert len(calls) == 2 and calls[0]["frame_ids"] == calls[1]["frame_ids"]
    assert result.support_level != "supported"


def test_runtime_rejects_prefix_expansion_in_compiled_scope(video, config):
    q = query()
    q["scope"] = {"kind": "interval", "interval": [0, 4], "description": ""}
    model = FakeQwen(q)
    result = R2VideoAgent(model, config=config).solve(
        R2Request(str(video), "q", observation_cutoff=1)
    )
    assert not [p for role, p, _ in model.payloads if role == "observe"]
    assert result.support_level != "supported"


def test_qwen_generate_injects_verified_timing_only_when_requested(monkeypatch):
    import contextlib
    import sys

    import numpy as np

    captured = {}

    class Inputs(dict):
        @property
        def input_ids(self):
            return self["input_ids"]

        def to(self, _):
            return self

    class Processor:
        def apply_chat_template(self, *args, **kwargs):
            return "prompt"

        def __call__(self, **kwargs):
            captured["processor"] = kwargs
            return Inputs(input_ids=np.array([[10, 11]]))

        def batch_decode(self, *args, **kwargs):
            return ["X"]

    def generate(**kwargs):
        captured["generation"] = kwargs
        return [np.array([10, 11, 20])]

    def process(*args, **kwargs):
        return (
            None,
            [
                (
                    np.zeros((4, 3, 32, 32)),
                    {"fps": 2, "frames_indices": [0, 1, 2, 3], "total_num_frames": 4},
                )
            ],
            {"do_sample_frames": False},
        )

    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            inference_mode=contextlib.nullcontext, cuda=SimpleNamespace(is_available=lambda: False)
        ),
    )
    monkeypatch.setitem(sys.modules, "qwen_vl_utils", SimpleNamespace(process_vision_info=process))
    model = Qwen3VLModel("fake")
    model._loaded = True
    model.processor = Processor()
    model.model = SimpleNamespace(generate=generate, device="cpu", config=SimpleNamespace())
    selected = [
        {
            "fps": 8,
            "frames_indices": [80, 82, 84, 86],
            "source_timestamps": [10, 10.25, 10.5, 10.75],
            "frame_ids": ["a", "b", "c", "d"],
            "total_num_frames": 100,
        }
    ]
    result = model.generate([{"role": "user", "content": "q"}], video_frame_metadata=selected)
    assert result.metadata["video_timing_verified"]
    assert captured["processor"]["video_metadata"][0]["frames_indices"][0] == 80
    assert captured["processor"]["do_sample_frames"] is False
    assert captured["processor"]["do_resize"] is False
    assert "video_frame_metadata" not in captured["generation"]
    model.generate([{"role": "user", "content": "q"}])
    assert captured["processor"]["video_metadata"][0]["frames_indices"][0] == 0


def test_cli_r2_dispatch_preserves_native_options(video, config, monkeypatch, tmp_path):
    import sys

    from qwen3vl_agent import cli

    model = FakeQwen()
    monkeypatch.setattr(cli, "build_model", lambda _: model)
    trace = tmp_path / "trace.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "cli",
            "--strategy",
            "r2",
            "--video",
            str(video),
            "--query",
            "direction?",
            "--choice",
            "Y: right",
            "--choice",
            "N: left",
            "--trace-output",
            str(trace),
        ],
    )
    cli.main()
    result = json.loads(trace.read_text())["r2"]
    assert result["prediction"] == "Y"
    assert result["completion_state"] == "complete"


def test_missing_media_is_reported_separately(config, tmp_path):
    report = preflight([(R2Request(str(tmp_path / "missing.mp4"), "q"), {})], config)
    assert report["missing_media"] == 1 and report["ready"] == 0


def test_query_scope_excludes_context_motion():
    q = query()
    state = store_for(
        q, [record(i, point=[x, 500]) for i, x in enumerate([700, 200, 400, 600, 100])]
    )
    value = reduced(q, state, query_spans=[(1, 3)])
    assert value["value"]["direction_segments"] == ["right"]
    assert len(value["evidence_ids"]) == 3


def test_identity_connection_cannot_bridge_unobserved_time():
    from qwen3vl_agent.r2.reduce import reduce_query

    q, state = identity_store()
    coverage = [w["coverage"] for w in state.windows.values()]
    result = reduce_query(q, state, R2Config(), [(0, 0.1), (1, 1.1)], coverage, 1)
    assert not result["sufficient"]
    assert any(g["kind"] == "coverage" and g["span"] == [0, 1] for g in result["gaps"])


def test_identity_history_uses_later_reveal_and_stops(video, config):
    q = query("identity_at_time")

    def hook(value, payload):
        for row in value["records"]:
            row["rank"] = 2
        last = payload["frames"][-1]
        if last["source_seconds"] >= 4:
            value["containments"] = [
                {
                    "hidden_target_id": "T1",
                    "carrier_entity_id": "E1",
                    "frame_id": last["frame_id"],
                    "status": "visible_reveal",
                }
            ]
        return value

    model = FakeQwen(q, observer_hook=hook)
    result = R2VideoAgent(model, config=config).solve(
        R2Request(str(video), "Which cup held it earlier?", query_scope=(0, 1), query_time=0.31)
    )
    op = result.value_state["operations"][0]
    assert op["value"]["propagation"] == "backward"
    assert op["value"]["query_time"] == 0.25
    assert op["value"]["possible_ranks"] == [2]
    assert len([p for role, p, _ in model.payloads if role == "observe"]) == 2
    assert result.value_state["sufficient"]


def test_identity_direct_query_reveal_stops_before_unneeded_window(video, config):
    q = query("identity_at_time")

    def hook(value, payload):
        for row in value["records"]:
            row["rank"] = 1
        value["containments"] = [
            {
                "hidden_target_id": "T1",
                "carrier_entity_id": "E1",
                "frame_id": payload["frames"][0]["frame_id"],
                "status": "visible_reveal",
            }
        ]
        return value

    model = FakeQwen(q, observer_hook=hook)
    result = R2VideoAgent(model, config=config).solve(
        R2Request(str(video), "Which cup?", query_time=0)
    )
    assert result.value_state["sufficient"]
    assert len([p for role, p, _ in model.payloads if role == "observe"]) == 1


def test_observation_conflict_requires_explicit_same_frame_correction():
    q = query("endpoint_delta")
    state = store_for(q, [record(0, "red"), record(1, "green")])
    raw = copy.deepcopy(state.observations)

    def reread(from_node, retired=()):
        return observation(
            [record(0, "blue")],
            superseded_observation_ids=list(retired),
            associations=[
                {
                    "group_id": "same",
                    "alternatives": [
                        {
                            "links": [
                                {"from_node": from_node, "to_entity": "E1", "kind": "same_entity"}
                            ],
                            "evidence_frames": ["F01"],
                        }
                    ],
                    "supersedes": [],
                    "unresolved_extra": False,
                    "relation_preserved": True,
                }
            ],
        )

    coverage = {"span": [0, 0.25], "completed": True, "resolution_met": True}
    state.ingest("conflict", reread("w/E1"), frame_map([0]), q, coverage, "call2")
    assert any(g["kind"] == "conflict" for g in reduced(q, state)["gaps"])
    state.save_derived(reduced(q, state))
    with pytest.raises(ProtocolError, match="same source frame"):
        state.ingest(
            "invalid", reread("w/E1", ["w/O001"]), frame_map([0], "other"), q, coverage, "call3"
        )
    assert "invalid/E1" not in state.entities
    state.ingest("correct", reread("w/E1", ["w/O001"]), frame_map([0]), q, coverage, "call4")
    assert state.observations[:2] == raw
    assert not state.derived[0]["valid"]
    assert reduced(q, state)["status"] == "supported"
    assert reduced(q, state)["value"]["S1"]["before"] == "blue"


def test_unseen_phase_is_not_removed_from_periodic_evidence():
    q = query("periodic_continuation")
    rows = [record(i, phase) for i, phase in enumerate(["A", "B", "A", None, "B"])]
    rows[3]["visibility"] = "occluded"
    state = store_for(q, rows)
    assert reduced(q, state)["status"] == "unresolved"


def test_second_final_invalid_cannot_reuse_provisional_prediction(video, config):
    count = 0

    def final_hook(value, _):
        nonlocal count
        count += 1
        if count == 1:
            value["recheck"] = {
                "kind": "detail",
                "description": "Read the contact again",
                "span": [1, 2],
            }
        else:
            value["prediction"] = "not-an-original-label"
        return value

    result = R2VideoAgent(FakeQwen(final_hook=final_hook), config=config).solve(
        R2Request(str(video), "Direction?", choices=("right", "left"))
    )
    assert result.prediction is None and result.completion_state == "failed"
    # 1.2: first final, post-recheck final, then its dedicated format repair.
    assert result.resources["terminal_calls"] == 3


def test_specific_motion_condition_is_not_replaced_by_any_motion():
    q = query("motion_condition_filter", {"motion_condition": "bouncing"})
    rows = [
        record(i, "red", point=[100 + i * 100, 100], condition_satisfied=False) for i in range(3)
    ]
    assert reduced(q, store_for(q, rows))["value"]["exists"] is False
    rows[1]["condition_satisfied"] = True
    assert reduced(q, store_for(q, rows))["value"]["exists"] is True
    for row in rows:
        row.pop("condition_satisfied")
    assert reduced(q, store_for(q, rows))["status"] == "unresolved"


def test_frequency_and_periodic_alias_need_connected_phase_evidence():
    q = query("motion_property_trend", {"metric": "frequency", "phase_unit": "full circle"})
    rows = [
        record(i, "top", phase="top", cycle_marker=True, adjacency_resolved=True) for i in range(3)
    ]
    assert reduced(q, store_for(q, rows))["status"] == "supported"
    rows[1]["adjacency_resolved"] = False
    assert reduced(q, store_for(q, rows))["status"] == "unresolved"
    q = query("periodic_continuation")
    rows = [record(i, phase=p, adjacency_resolved=True) for i, p in enumerate("ABCABCA")]
    assert reduced(q, store_for(q, rows))["status"] == "supported"
    rows[2]["adjacency_resolved"] = False
    assert reduced(q, store_for(q, rows))["status"] == "unresolved"


def test_real_vfr_does_not_invent_source_frame_indices(tmp_path, config):
    import av
    from PIL import Image

    path = tmp_path / "vfr.mp4"
    pts = [0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    with av.open(str(path), "w") as output:
        stream = output.add_stream("mpeg4", rate=8)
        stream.width, stream.height, stream.pix_fmt = 64, 64, "yuv420p"
        for value in pts:
            frame = av.VideoFrame.from_image(Image.new("RGB", (64, 64), "red"))
            frame.pts, frame.time_base = value, Fraction(1, 8)
            for packet in stream.encode(frame):
                output.mux(packet)
        for packet in stream.encode():
            output.mux(packet)
    media = R2Media(config)
    contract = InputContract.resolve(R2Request(str(path), "q"), 2)
    batch = media.extract(path, (0, 1.6), [p / 8 for p in pts], contract, fps=8)
    assert len(batch.frames) == len(pts)
    assert [media.catalog[f.id]["decoded_frame_index"] for f in batch.frames] == list(
        range(len(pts))
    )
    assert media.prepare(batch).kind == "timestamped_images"
    assert media.catalog[batch.frames[3].id]["timestamp_seconds"] == 0.5


def test_resume_rejects_changed_request_model_and_media(video, config, tmp_path):
    model = FakeQwen()
    agent = R2VideoAgent(model, config=config)
    request = R2Request(str(video), "q", checkpoint_path=str(tmp_path / "check.jsonl"))
    agent.solve(request)
    with pytest.raises(ValueError, match="mismatch"):
        agent.solve(replace(request, resume=True, protocol_id="different_protocol"))
    model.generation = {"temperature": 0.2}
    with pytest.raises(ValueError, match="mismatch"):
        agent.solve(replace(request, resume=True))
    model.generation = {}
    with video.open("ab") as output:
        output.write(b"\0")
    with pytest.raises(ValueError, match="mismatch"):
        agent.solve(replace(request, resume=True))


def test_interrupted_targeted_recheck_resumes_with_cost(video, config, tmp_path):
    interrupted = False

    def observer_hook(value, payload):
        nonlocal interrupted
        if payload["window_id"].startswith("repair") and not interrupted:
            interrupted = True
            raise KeyboardInterrupt("repair interrupted")
        return value

    def final_hook(value, _):
        if not interrupted:
            value["recheck"] = {"kind": "detail", "description": "recheck state", "span": [1, 2]}
        return value

    model = FakeQwen(observer_hook=observer_hook, final_hook=final_hook)
    agent = R2VideoAgent(model, config=config)
    request = R2Request(str(video), "q", checkpoint_path=str(tmp_path / "repair.jsonl"))
    with pytest.raises(KeyboardInterrupt):
        agent.solve(request)
    result = agent.solve(replace(request, resume=True))
    assert result.prediction is not None
    assert result.resources["model_calls"] == len(model.payloads)
    assert sum(c["status"] == "interrupted" for c in result.trace["calls"]) == 1
    assert all(a["complete"] for a in result.trace["actions"])


def test_batch_missing_media_does_not_discard_other_predictions(video, config, tmp_path):
    rows = [
        (R2Request(str(tmp_path / "missing.mp4"), "q", request_id="missing"), {}),
        (R2Request(str(video), "q", request_id="ready"), {}),
    ]
    out = tmp_path / "batch"
    result = run_manifest(R2VideoAgent(FakeQwen(), config=config), rows, out)
    assert result["requests"] == 2 and result["predictions"] == 1
    predictions = [json.loads(s) for s in (out / "predictions.jsonl").read_text().splitlines()]
    assert predictions[0]["prediction"] is None and predictions[1]["prediction"] is not None
