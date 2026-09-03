from __future__ import annotations

import json
from pathlib import Path

import pytest

from qwen3vl_agent.config import load_config
from qwen3vl_agent.p01.config import P01Config
from qwen3vl_agent.p01.smoke import (
    load_smoke_questions,
    main,
    select_questions,
    summarize_records,
)


def _write_manifest(
    data_root: Path,
    *,
    records: list[dict[str, object]],
) -> None:
    benchmark_root = data_root / "Video-MME"
    subset = benchmark_root / "subsets" / "p01-smoke-v1"
    subset.mkdir(parents=True)
    videos = benchmark_root / "videos"
    videos.mkdir()
    for record in records:
        video = benchmark_root / str(record["video_path"])
        video.parent.mkdir(parents=True, exist_ok=True)
        video.write_bytes(b"fake-video")
    (subset / "questions.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


def _record(native_id: str, video_name: str, probe: str) -> dict[str, object]:
    return {
        "schema_version": 1,
        "benchmark": "Video-MME",
        "native_question_id": native_id,
        "native_task_type": "test",
        "probe_category": probe,
        "video_path": f"videos/{video_name}",
        "question": f"Question {native_id}?",
        "choices": [
            {"id": "A", "text": "first"},
            {"id": "B", "text": "second"},
        ],
        "answer_label": "A",
        "given_interval_sec": [2, 4] if native_id == "q2" else None,
        "media_status": "available",
    }


def test_manifest_loading_filtering_and_requested_order(tmp_path: Path) -> None:
    _write_manifest(
        tmp_path,
        records=[
            _record("q1", "b.mp4", "G01-static"),
            _record("q2", "a.mp4", "G42"),
        ],
    )

    questions = load_smoke_questions(tmp_path, benchmarks=("Video-MME",))

    assert [item.native_question_id for item in questions] == ["q1", "q2"]
    assert questions[1].given_interval is not None
    assert questions[1].given_interval.start_seconds == 2
    assert [item.native_question_id for item in select_questions(questions)] == [
        "q2",
        "q1",
    ]
    requested = select_questions(questions, question_ids=("q1", "q2"))
    assert [item.native_question_id for item in requested] == ["q1", "q2"]
    with pytest.raises(ValueError, match="unknown question ID"):
        select_questions(questions, question_ids=("missing",))


def test_manifest_rejects_media_path_escape(tmp_path: Path) -> None:
    record = _record("q1", "clip.mp4", "G01-static")
    record["video_path"] = "../outside.mp4"
    benchmark_root = tmp_path / "Video-MME"
    subset = benchmark_root / "subsets" / "p01-smoke-v1"
    subset.mkdir(parents=True)
    (subset / "questions.jsonl").write_text(
        json.dumps(record) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="stay relative"):
        load_smoke_questions(tmp_path, benchmarks=("Video-MME",))


def test_dry_run_writes_an_ordered_plan_without_loading_model(tmp_path: Path) -> None:
    _write_manifest(
        tmp_path / "data",
        records=[
            _record("q1", "b.mp4", "G01-static"),
            _record("q2", "a.mp4", "G42"),
        ],
    )
    config = tmp_path / "config.yaml"
    config.write_text("model: {}\np01: {}\n", encoding="utf-8")
    output = tmp_path / "run"

    exit_code = main(
        [
            "--config",
            str(config),
            "--data-root",
            str(tmp_path / "data"),
            "--output-dir",
            str(output),
            "--benchmark",
            "Video-MME",
            "--question-id",
            "q1",
            "--question-id",
            "q2",
            "--dry-run",
        ]
    )

    assert exit_code == 0
    plan = json.loads((output / "run_plan.json").read_text(encoding="utf-8"))
    assert [item["native_question_id"] for item in plan["questions"]] == ["q1", "q2"]


def test_summary_uses_atomic_item_records_and_reports_peak_memory() -> None:
    summary = summarize_records(
        [
            {
                "runner_status": "completed",
                "question": {
                    "probe_category": "G01-static",
                    "choices": [{"id": "A", "text": "first"}],
                    "answer_label": "A",
                },
                "result": {
                    "status": "answered",
                    "prediction": "A",
                    "decision_source": "initial",
                    "support_level": "strong",
                    "trace": {"stop_reason": "mandatory_mcq_prediction_emitted"},
                    "resources": {"model_call_count": 3},
                },
                "evaluation": {"label_exact_match": True},
                "runtime": {
                    "elapsed_seconds": 1.5,
                    "gpu": {"max_memory_reserved_bytes": 123},
                },
            },
            {
                "runner_status": "error",
                "question": {
                    "probe_category": "G01-static",
                    "choices": [{"id": "A", "text": "first"}],
                    "answer_label": "A",
                },
                "result": None,
                "evaluation": {"label_exact_match": None},
                "runtime": {
                    "elapsed_seconds": 0.5,
                    "gpu": {"max_memory_reserved_bytes": 100},
                },
            },
        ]
    )

    assert summary["record_count"] == 2
    assert summary["runner_status_counts"] == {"completed": 1, "error": 1}
    assert summary["mcq_label_accuracy"] == 1.0
    assert summary["mcq_total"] == 2
    assert summary["mcq_prediction_coverage"] == 0.5
    assert summary["mcq_strict_label_accuracy"] == 0.5
    assert summary["total_model_calls"] == 3
    assert summary["by_probe"]["G01-static"]["record_count"] == 2
    assert summary["peak_gpu_memory_reserved_bytes"] == 123
    assert summary["total_elapsed_seconds"] == 2.0


def test_4090d_fallback_config_is_valid_and_smaller_than_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project_root = Path(__file__).resolve().parents[1]
    monkeypatch.delenv("QWEN3VL_MODEL_PATH", raising=False)
    default = load_config(project_root / "configs" / "p01_8b.yaml")
    fallback = load_config(project_root / "configs" / "p01_4090d_safe.yaml")

    default_p01 = P01Config.from_mapping(default["p01"])
    fallback_p01 = P01Config.from_mapping(fallback["p01"])

    assert fallback["model"]["attn_implementation"] == "flash_attention_2"
    assert fallback_p01.normal_total_pixels < default_p01.normal_total_pixels
    assert fallback_p01.normal_max_pixels < default_p01.normal_max_pixels
    assert fallback_p01.image_min_pixels < default_p01.image_min_pixels
    assert fallback_p01.max_refinement_rounds == 1
    assert fallback_p01.max_model_calls == default_p01.max_model_calls
