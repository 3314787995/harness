from dataclasses import replace

import pytest
from r8_fakes import FakeModel, make_video
from test_r8_agent import setup as setup  # noqa: PLC0414 -- explicit pytest fixture re-export

from qwen3vl_agent.cli import build_parser
from qwen3vl_agent.p01.config import P01Config
from qwen3vl_agent.r8 import R8Config, R8Request
from qwen3vl_agent.r8.evaluate import preflight, public_request, run_rows, score
from qwen3vl_agent.r8.media import ScopedMedia, sampling, windows
from qwen3vl_agent.r8.types import ProtocolError


def test_shared_cli_and_manifest_contract():
    args = build_parser().parse_args(
        ["--strategy", "r8", "--query", "q", "--video", "v.mp4", "--output-protocol", "numeric"]
    )
    assert args.r8_mode == "G"
    with pytest.raises(ProtocolError, match="answers"):
        public_request({"video_path": "v.mp4", "question": "q", "answer": "B"})


def test_pts_vfr_and_crop_permissions(tmp_path):
    video = make_video(tmp_path / "vfr.mp4", vfr=True)
    config = R8Config(media=P01Config(cache_dir=str(tmp_path / "cache")))
    request = R8Request(str(video), "read", allowed_scope=(0.5, 2.5))
    media = ScopedMedia(request, config, {})
    batch = media.extract((0.5, 2.5), [0.5, 0.75, 1.0, 1.75, 2.5])
    assert all(0.5 <= f.timestamp_seconds <= 2.5 for f in batch.frames)
    assert all(media.catalog[f.id]["pts"] is not None for f in batch.frames)
    prepared = media.prepare(batch)
    assert prepared.kind == "images" or not prepared.video_frame_metadata
    local = media.local_batch(batch.frames[1].id, [[0.1, 0.2, 0.8, 0.9]])
    assert len(local.frames) == 2 and not local.ordered
    crop = media.catalog[local.frames[-1].id]
    assert (
        crop["source_frame_id"] == batch.frames[1].id
        and crop["view_box"] != media.catalog[batch.frames[1].id]["view_box"]
    )
    with pytest.raises(ProtocolError):
        media.extract((0.0, 3.0), [1.0])


def test_scan_context_not_double_counted(setup):
    config, request = setup
    config = replace(config, core_seconds=1.0, context_seconds=0.25)
    media = ScopedMedia(request, config, {})
    plans = windows(media.contract, config)
    assert sum(p["core"][1] - p["core"][0] for p in plans) == pytest.approx(
        media.metadata.duration_seconds
    )
    assert plans[0]["context"][1] > plans[1]["context"][0]
    with pytest.raises(ProtocolError, match="density"):
        sampling((0.0, 10.0), config, fps=10.0)


def test_preflight_does_not_load_model(setup):
    config, request = setup
    result = preflight(request, config)
    assert (
        result["status"] == "passed"
        and not result["model_loaded"]
        and not result["semantic_accuracy_measured"]
    )


def test_score_keeps_anomaly_duplicates_and_experiments():
    answers = [
        {
            "request_id": key,
            "answer": "B",
            "video_id": video,
            "mechanism": "S2",
            "r8_role": "boundary" if key == "anomaly" else "mechanism",
        }
        for key, video in [("q1", "v1"), ("q2_same_stem", "v1"), ("anomaly", "v2")]
    ]
    predictions = [
        {
            "request_id": key,
            "mode": "G",
            "comparison": "end_to_end",
            "run_status": "completed",
            "result": {
                "prediction": "B" if key == "q1" else None,
                "status": "verified_exact"
                if key == "q1"
                else "annotation_anomaly"
                if key == "anomaly"
                else "unresolved_evidence",
            },
        }
        for key in ("q1", "q2_same_stem", "anomaly")
    ]
    predictions += [
        {
            "request_id": "q1",
            "mode": "C",
            "comparison": "fixed_evidence",
            "diagnostic": True,
            "run_status": "completed",
            "result": {"prediction": "B", "status": "verified_exact"},
        }
    ]
    report = score(predictions, answers)
    assert len(report["experiments"]) == 2
    full = next(r for r in report["experiments"] if r["comparison"] == "end_to_end")
    assert full["n"] == 3 and full["accuracy_all_requested"] == pytest.approx(1 / 3)
    assert full["annotation_anomaly_n"] == 1 and full["variable_accuracy"] == "unavailable"
    assert len(full["per_video"]) == 2


def test_missing_media_is_explicit_and_no_load(monkeypatch, tmp_path):
    model = FakeModel()
    monkeypatch.setattr("qwen3vl_agent.r8.evaluate.build_model", lambda _: model)
    results = run_rows(
        [{"video_path": str(tmp_path / "missing.mp4"), "question": "q"}],
        output=tmp_path / "out.jsonl",
        checkpoint_dir=tmp_path / "checkpoints",
    )
    assert results[0]["run_status"] == "media_unavailable" and results[0]["result"] is None
    assert not model.is_loaded and not model.calls
