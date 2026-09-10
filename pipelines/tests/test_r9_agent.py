import json
from dataclasses import replace

import pytest

from qwen3vl_agent.r9 import R9Request, R9VideoAgent
from qwen3vl_agent.r9.evaluate import mra, preflight, public_request, score
from qwen3vl_agent.r9.protocol import match_choice
from qwen3vl_agent.r9.schema import SCHEMAS, validate
from qwen3vl_agent.r9.types import ProtocolError
from tests.r9_fakes import FakeModel, config, make_video, spec


def request(tmp_path, **kwargs):
    path = make_video(tmp_path / "video.mp4")
    return R9Request(
        str(path), "Where is the target?", choices=("A. front-left", "B. back-right"), **kwargs
    )


def test_end_to_end_grounded_and_original_option_order(tmp_path):
    req = request(tmp_path)
    model = FakeModel()
    result = R9VideoAgent(model, config(tmp_path)).solve(req)
    assert result.prediction == "A"
    assert result.status == "estimated"
    assert not result.forced_answer
    assert [c["body"]["stage"] for c in model.calls] == ["compile", "observe", "relations", "audit"]
    reordered = replace(req, choices=("A. back-right", "B. front-left"))
    assert R9VideoAgent(FakeModel(), config(tmp_path)).solve(reordered).prediction == "B"


def test_role_isolation_and_neutral_paths(tmp_path):
    req = request(tmp_path)
    model = FakeModel()
    result = R9VideoAgent(model, config(tmp_path)).solve(req)
    for call in model.calls:
        text = json.dumps(call["messages"])
        assert req.video_path not in text
        if call["body"]["stage"] in {"observe", "relations", "audit"}:
            assert "A. front-left" not in text
            assert "original_option_texts" not in text
            assert "ground_truth" not in text
        assert call["kwargs"]["temperature"] == 0
        assert call["kwargs"]["input_token_limit"] == 15184
    assert result.trace["resources"]["unique_source_frames"] == 4
    assert result.trace["resources"]["visual_exposures"] > 4


def test_public_query_window_reaches_question_compiler_and_final_fallback(tmp_path):
    req = request(tmp_path, query_scope=(0.5, 2.0))
    model = FakeModel(missing=True)
    result = R9VideoAgent(model, config(tmp_path)).solve(req)
    assert result.forced_answer
    assert model.calls[0]["body"]["input"]["public_protocol"]["query_scope"] == [0.5, 2.0]
    assert model.calls[-1]["body"]["input"]["query_scope"] == [0.5, 2.0]
    with pytest.raises(ProtocolError, match="query scope"):
        replace(req, query_time=2.5)


@pytest.mark.parametrize("mode", ["B0", "B1", "B2", "B3", "B4"])
def test_all_baseline_modes(tmp_path, mode):
    req = request(tmp_path, mode=mode)
    model = FakeModel()
    result = R9VideoAgent(model, config(tmp_path)).solve(req)
    assert result.prediction == "A"
    stages = [c["body"]["stage"] for c in model.calls]
    if mode == "B0":
        assert stages == ["answer"]
        assert result.trace["resources"]["visual_exposures"] == 0
    if mode == "B2":
        assert stages == ["compile", "answer"]
        assert "explicit_reference_frame" in model.calls[-1]["body"]["input"]


def test_unresolved_gap_triggers_reread_then_forced_output(tmp_path):
    req = request(tmp_path)
    result = R9VideoAgent(FakeModel(missing=True), config(tmp_path)).solve(req)
    assert result.prediction == "A" and result.forced_answer and result.status == "estimated"
    assert len(result.trace["actions"]) == 2
    assert result.unresolved_reasons


def test_numeric_forced_estimate_and_explicit_unresolved(tmp_path):
    req = replace(request(tmp_path), choices=(), output_protocol="numeric", output_unit="m")
    s = spec(req.question, "absolute_distance")
    s["measurement"] = {
        "geometry_semantics": "closest_boundary",
        "unit": "m",
        "metric_required": True,
    }
    result = R9VideoAgent(FakeModel(task=s, missing=True), config(tmp_path)).solve(req)
    assert result.prediction == 2.5 and result.unit == "m" and result.forced_answer
    result = R9VideoAgent(FakeModel(task=s, missing=True), config(tmp_path)).solve(
        replace(req, force_answer=False)
    )
    assert result.prediction is None and result.status == "unresolved"


def test_format_retry_and_budget_reserve(tmp_path):
    req = request(tmp_path, max_model_calls=4)
    result = R9VideoAgent(FakeModel(malformed_once=True), config(tmp_path)).solve(req)
    assert result.forced_answer and result.trace["resources"]["model_calls"] <= 4
    assert result.trace["receipts"][0]["status"] == "invalid_output"


def test_checkpoint_resume_no_repeat_and_version_rejection(tmp_path):
    req = request(tmp_path, checkpoint_path=str(tmp_path / "checkpoint.jsonl"))
    model = FakeModel(interrupt_at=3)
    agent = R9VideoAgent(model, config(tmp_path))
    with pytest.raises(KeyboardInterrupt):
        agent.solve(req)
    after = agent.solve(replace(req, resume=True))
    assert after.prediction == "A"
    stages = [c["body"]["stage"] for c in model.calls]
    assert stages.count("compile") == 1 and stages.count("observe") == 1
    assert after.trace["resources"]["failed_calls"] == 1
    count = len(model.calls)
    assert agent.solve(replace(req, resume=True)).prediction == "A" and len(model.calls) == count
    with pytest.raises(ValueError, match="mismatch"):
        agent.solve(replace(req, resume=True, question="changed"))


@pytest.mark.parametrize("vfr", [False, True])
def test_real_pts_crop_input_preflight(tmp_path, vfr):
    p = make_video(tmp_path / "v.mp4", vfr=vfr)
    r = R9Request(str(p), "input check", allowed_scope=(0.5, 2.5))
    report = preflight(r, config(tmp_path))
    assert report["status"] == "passed" and not report["model_loaded"]
    for packet in report["packets"]:
        assert all(0.5 <= v["timestamp_seconds"] <= 2.5 for v in packet["evidence"].values())
        crop = list(packet["crop_evidence"].values())[-1]
        assert crop["source_frame_id"] != crop["id"]
        assert crop["view_box"] != [0, 0, *crop["source_size"]]


def test_schema_unknown_fields_and_leaking_request():
    with pytest.raises(ProtocolError):
        validate(dict(spec(), answer="A"), SCHEMAS["compile"])
    with pytest.raises(ProtocolError):
        public_request({"video_path": "left/frame.mp4", "question": "q", "answer": "A"})
    with pytest.raises(ProtocolError):
        R9Request("v", "q", mode="B5")


def test_mra_strict_threshold_and_missing_denominator():
    assert mra(3.2, 2.9) == 0.8
    assert mra(150, 100) == 0
    assert mra(105, 100) == 0.9
    assert mra(100, 100) == 1
    answers = [
        {
            "request_id": "q",
            "dataset": "VSI",
            "native_task": "distance",
            "mechanism": "S3",
            "output_protocol": "numeric",
            "unit": "m",
            "answer": 2,
        }
    ]
    result = score([], answers)
    assert result["overall"]["mra"] == 0 and result["overall"]["missing_predictions"] == 1


def test_semantic_choice_maps_without_letter_cache():
    a = R9Request("v", "q", choices=["A. left, right", "B. right, left"])
    assert match_choice(["left", "right"], a.choices) == "A"
