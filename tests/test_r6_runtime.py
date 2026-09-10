"""Scope, processor limits, persistence and entry-point tests without model weights."""

import json
from copy import deepcopy
from dataclasses import asdict, replace
from fractions import Fraction
from pathlib import Path

import pytest
from r6_fakes import FakeMedia, FakeModel, query, request

from qwen3vl_agent.cli import build_parser
from qwen3vl_agent.models.qwen3vl import InputContextExceeded, VisualBudgetExceeded
from qwen3vl_agent.r6 import R6Config, R6VideoAgent
from qwen3vl_agent.r6.evaluate import main, preflight, run_requests, score
from qwen3vl_agent.r6.media import ScopedMedia
from qwen3vl_agent.r6.providers import TextSources
from qwen3vl_agent.r6.runtime import Session
from qwen3vl_agent.r6.types import InputContract, ProtocolError


def tiny_video(tmp_path):
    av = pytest.importorskip("av")
    np = pytest.importorskip("numpy")
    path = tmp_path / "pts.mp4"
    with av.open(str(path), "w") as container:
        stream = container.add_stream("mpeg4", rate=4)
        stream.width, stream.height, stream.pix_fmt = 64, 48, "yuv420p"
        for i in range(12):
            array = np.zeros((48, 64, 3), dtype=np.uint8)
            array[:, :, i % 3] = 100 + i * 10
            frame = av.VideoFrame.from_ndarray(array, format="rgb24")
            frame.pts, frame.time_base = i, Fraction(1, 4)
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return path


def test_real_pts_disjoint_scope_crop_and_cache_tamper(tmp_path):
    path = tiny_video(tmp_path)
    item = replace(
        request(tmp_path), video_path=str(path), allowed_intervals=((0.25, 1), (1.75, 2.75))
    )
    config = R6Config.from_mapping({"media": {"cache_dir": str(tmp_path / "cache")}})
    media = ScopedMedia(item, config)
    with pytest.raises(ProtocolError, match="boundary"):
        media.extract((0, 1.5), [0.5])
    batch = media.extract((0.25, 1), [0.3, 0.9], fps=2)
    assert all(0.25 <= f.timestamp_seconds <= 1 for f in batch.frames)
    assert all(media.catalog[f.id]["pts"] is not None for f in batch.frames)
    crop = media.crop(batch.frames[0].id, [0.25, 0.25, 0.75, 0.75])
    assert media.catalog[crop.id]["source_frame_id"] == batch.frames[0].id
    prepared = media.prepare(batch)
    assert {f.id for f in prepared.frames} == {f.id for f in batch.frames}
    restored = ScopedMedia(item, config)
    restored.restore(deepcopy(media.catalog))
    assert set(restored.catalog) == set(media.catalog)
    Path(batch.frames[0].path).write_bytes(b"tampered")
    with pytest.raises(ProtocolError, match="cache"):
        media.frame(batch.frames[0].id)
    fresh = ScopedMedia(item, config)
    with pytest.raises(ProtocolError, match="cache"):
        fresh.extract((0.25, 1), [0.3, 0.9], fps=2)


def test_real_preflight_no_model_load(tmp_path):
    item = replace(request(tmp_path), video_path=str(tiny_video(tmp_path)))
    config = R6Config.from_mapping({"media": {"cache_dir": str(tmp_path / "cache")}})
    result = preflight(item, config)
    assert result["status"] == "ready" and not result["model_loaded"]
    assert result["decoded_checks"] and result["audio_available"] is False


def test_text_filter_happens_before_search_and_views_keep_offsets(tmp_path):
    text = tmp_path / "asr.jsonl"
    rows = [
        {"start_sec": 0, "end_sec": 0.5, "text": "permitted words " * 400},
        {"start_sec": 0.9, "end_sec": 1.1, "text": "FORBIDDEN crossing"},
        {"start_sec": 2, "end_sec": 3, "text": "FORBIDDEN future"},
        {"start_sec": 0.1, "end_sec": 0.4, "text": "unaligned", "alignment_status": "unknown"},
    ]
    text.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    item = request(
        tmp_path,
        history_cutoff=1,
        allowed_modalities=["video", "asr"],
        subtitle_policy="aligned_only",
        asr_path=str(text),
    )
    contract = InputContract.resolve(item, 4, "hash")
    source = TextSources(item, contract)
    assert len(source.sources) > 1
    assert all(
        "FORBIDDEN" not in s["text"] and "unaligned" not in s["text"]
        for s in source.search("FORBIDDEN")
    )
    assert sorted(s["char_span"][0] for s in source.sources.values())[:2] == [0, 3000]
    assert len(source.issues) == 3


def test_subtitle_permission_required(tmp_path):
    with pytest.raises(ProtocolError, match="external text"):
        request(tmp_path, subtitle_path="not_allowed.srt")


def test_resume_completed_does_not_call_model_again(tmp_path):
    item = request(tmp_path, checkpoint_path=str(tmp_path / "checkpoint.jsonl"))
    model = FakeModel()
    first = R6VideoAgent(model, media_factory=FakeMedia).solve(item)
    before = len(model.calls)
    second = R6VideoAgent(model, media_factory=FakeMedia).solve(replace(item, resume=True))
    assert first.to_dict() == second.to_dict()
    assert len(model.calls) == before
    with pytest.raises(ValueError, match="mismatch"):
        R6VideoAgent(model, media_factory=FakeMedia).solve(
            replace(item, resume=True, question="Changed")
        )
    with pytest.raises(ValueError, match="mismatch"):
        R6VideoAgent(
            model, replace(R6Config(), observer_options="all"), media_factory=FakeMedia
        ).solve(replace(item, resume=True))


def test_mid_call_replay_reuses_validated_response_and_cost(tmp_path):
    state = {}
    model = FakeModel()
    session = Session(model, R6Config(), state, lambda: None)
    payload = {
        "choices": [{"label": "A", "text": "one"}, {"label": "B", "text": "two"}],
        "protocol": {"reference_scope": None},
    }
    first = session.call("compile", "compiler", payload)
    second = Session(model, R6Config(), deepcopy(state), lambda: None).call(
        "compile", "compiler", payload
    )
    assert first == second and len(model.calls) == 1
    with pytest.raises(ProtocolError, match="changed"):
        session.call("compile", "compiler", {**payload, "changed": True})


@pytest.mark.parametrize("error", ["input", "visual"])
def test_measured_processor_limit_splits_and_counts_failed_attempt(tmp_path, error):
    class LimitedModel(FakeModel):
        def generate(self, messages, **kwargs):
            body = json.loads(messages[1]["content"][0]["text"])
            if body["role"] == "observer" and len(body["source_manifest"]) > 16:
                if error == "input":
                    raise InputContextExceeded(18000, 16384)
                raise VisualBudgetExceeded("visual tokens exceed limit")
            return super().generate(messages, **kwargs)

    model = LimitedModel()
    result = R6VideoAgent(model, media_factory=FakeMedia).solve(request(tmp_path))
    assert result.prediction == "A"
    assert result.costs["model_calls"] == len(model.calls) + 1
    assert any(e["event"] == "observation_split" for e in result.trace["events"])
    assert result.costs["failed_calls"] == 1
    assert len(result.trace["facts"]) == 2


@pytest.mark.parametrize(
    "mode,roles",
    [("direct", ["answer"]), ("question_only", ["answer"]), ("captions", ["observer", "answer"])],
)
def test_baseline_modes_use_real_entrypoints_and_remain_unverified(tmp_path, mode, roles):
    model = FakeModel()
    result = R6VideoAgent(model, media_factory=FakeMedia).solve(request(tmp_path, mode=mode))
    assert [c["role"] for c in model.calls] == roles
    assert result.prediction == "A" and result.evidence_status == "insufficient"
    if mode == "question_only":
        assert model.calls[0]["body"]["source_manifest"] == {}
        assert result.costs["visual_exposures"] == 0


@pytest.mark.parametrize(
    "setting,field",
    [("neutral", "neutral_targets"), ("all", "choices"), ("question_only", "question")],
)
def test_observer_visibility_ablation(tmp_path, setting, field):
    model = FakeModel()
    R6VideoAgent(
        model, replace(R6Config(), observer_options=setting), media_factory=FakeMedia
    ).solve(request(tmp_path))
    observer = next(c for c in model.calls if c["role"] == "observer")["body"]["input"]
    assert field in observer and "preferred_label" not in observer
    if setting != "all":
        assert "choices" not in observer
    if setting == "question_only":
        assert "neutral_targets" not in observer and "requested_observation" not in observer


def test_audio_unavailable_is_capability_stop(tmp_path):
    model = FakeModel({"compiler": lambda body: query(required_modalities=["video", "audio"])})
    result = R6VideoAgent(model, media_factory=FakeMedia).solve(request(tmp_path))
    assert result.stop_reason == "MODALITY_UNAVAILABLE"
    assert result.costs["audio_seconds"] == 0 and result.forced_choice


def test_cli_and_offline_score_keep_missing_runs_in_denominator(tmp_path):
    args = build_parser().parse_args(
        ["--strategy", "r6", "--video", "x.mp4", "--query", "Q", "--choice", "a", "--choice", "b"]
    )
    assert args.strategy == "r6" and args.r6_mode == "pipeline"
    result = score(
        [{"request_id": "1", "prediction": "A"}],
        [{"request_id": "1", "answer": "A"}, {"request_id": "2", "answer": "B"}],
    )
    assert result["accuracy"] == 0.5 and result["questions"] == 2
    source = tmp_path / "request.json"
    source.write_text(json.dumps(asdict(request(tmp_path)) | {"answer": "A"}), encoding="utf-8")
    with pytest.raises(ProtocolError, match="offline"):
        main(["preflight", "--request", str(source), "--output", str(tmp_path / "preflight.json")])


def test_run_missing_media_does_not_load_model(tmp_path):
    item = replace(request(tmp_path), video_path=str(tmp_path / "missing.mp4"))
    model = FakeModel()
    predictions = run_requests(
        [item], model_settings={}, config=R6Config(), output=tmp_path / "run", model=model
    )
    assert predictions[0]["run_status"] == "media_unavailable"
    assert model.loaded is False and model.calls == []
