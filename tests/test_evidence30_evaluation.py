from __future__ import annotations

import json
from pathlib import Path
from typing import Any, ClassVar, Self

import pyarrow as pa
import pyarrow.parquet as pq

from qwen3vl_agent.coarse_to_fine.cache import SubtitleCue, SubtitleTrack
from qwen3vl_agent.evaluation.evidence30 import (
    Evidence30Dataset,
    canonical_sha256,
    normalize_exposures,
    preflight_evidence30,
    score_relaxed_grounding,
    sha256_file,
)
from qwen3vl_agent.evaluation.evidence30_runner import (
    mark_locked_spent,
    run_evidence30_suite,
)
from qwen3vl_agent.evaluation.freeze import (
    create_freeze_manifest,
    verify_freeze_manifest,
    write_freeze_manifest,
)
from qwen3vl_agent.evaluation.runtime import (
    VideoMMEStrategySession,
    _sample_direct_subtitles,
)
from qwen3vl_agent.evaluation.videomme import VideoMMEQuestion
from qwen3vl_agent.models.base import BaseVideoModel, ModelOutput


def make_reference(question_id: str, split: str, video_id: str) -> dict[str, Any]:
    question = {
        "text": f"Question {question_id}?",
        "options": [
            {"option_id": f"O{index + 1}", "benchmark_label": label, "text": text}
            for index, (label, text) in enumerate(
                zip("ABCD", ("One.", "Two.", "Three.", "Four."))
            )
        ],
        "official_answer": {"option_id": "O2", "benchmark_label": "B"},
    }
    return {
        "schema_version": "videomme-evidence30/0.2.0",
        "annotation_id": f"test-{question_id}",
        "record_status": "locked",
        "split": split,
        "source": {
            "dataset": "Video-MME",
            "video_id": video_id,
            "question_id": question_id,
            "duration_bucket": "short",
            "domain": "test",
            "sub_category": "test",
            "task_type": "test",
            "video_sha256": None,
            "subtitle_sha256": None,
        },
        "question": question,
        "validity": {"status": "valid", "reason": "fixture"},
        "evidence_contract": {
            "primary_topology": "local",
            "secondary_topologies": [],
            "required_modalities": ["visual"],
            "answer_criterion": "direct_support",
            "evidence_slots": [
                {"slot_id": "S1", "description": "visible fact", "required": True}
            ],
            "sufficient_evidence_sets": [
                {
                    "set_id": "ES1",
                    "description": "fixture set",
                    "logic": "all_required",
                    "items": [
                        {
                            "evidence_id": "E1",
                            "slot_id": "S1",
                            "required": True,
                            "core_interval": {"start_sec": 1.0, "end_sec": 2.0},
                            "context_interval": {"start_sec": 0.5, "end_sec": 2.5},
                            "modality": "visual",
                            "observation_requirement": "point_frame",
                            "minimum_observation": {
                                "mode": "inspect",
                                "min_frames": 1,
                                "min_temporal_span_sec": 0,
                            },
                            "atomic_fact": "SECRET_ATOMIC_FACT",
                            "roles": ["support"],
                            "supports_option_ids": ["O2"],
                            "refutes_option_ids": [],
                            "source_references": {
                                "frame_timestamps_sec": [1.0],
                                "subtitle_cue_ids": [],
                                "ocr_text": None,
                            },
                        }
                    ],
                    "relations": [],
                }
            ],
            "global_coverage_contract": None,
        },
        "hard_negatives": [
            {
                "negative_id": "HN1",
                "interval": {"start_sec": 8.0, "end_sec": 9.0},
                "modalities": ["visual"],
                "plausibility": "SECRET_HARD_NEGATIVE",
                "insufficiency_reason": "fixture",
                "tempts_option_ids": ["O1"],
            }
        ],
        "annotation_notes": [],
        "review": {},
    }


def write_fixture_dataset(tmp_path: Path) -> dict[str, Path]:
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    dev = make_reference("001-1", "dev", "001")
    locked = make_reference("002-1", "locked", "002")
    dev_path = evidence / "dev.jsonl"
    locked_path = evidence / "locked.jsonl"
    excluded_path = evidence / "excluded.jsonl"
    dev_path.write_text(json.dumps(dev) + "\n", encoding="utf-8")
    locked_path.write_text(json.dumps(locked) + "\n", encoding="utf-8")
    excluded_path.write_text("", encoding="utf-8")

    parquet_path = tmp_path / "questions.parquet"
    rows = []
    for question_id, video_id, source_name in (
        ("001-1", "001", "vid1"),
        ("002-1", "002", "vid2"),
    ):
        rows.append(
            {
                "video_id": video_id,
                "duration": "short",
                "domain": "test",
                "sub_category": "test",
                "url": "",
                "videoID": source_name,
                "question_id": question_id,
                "task_type": "test",
                "question": f"Question {question_id}?",
                "options": ["A. One.", "B. Two.", "C. Three.", "D. Four."],
                "answer": "B",
            }
        )
    pq.write_table(pa.Table.from_pylist(rows), parquet_path)

    manifest = {
        "manifest_version": "videomme-evidence30/0.2.0",
        "source_files": {
            "question_parquet": {
                "path": str(parquet_path),
                "sha256": sha256_file(parquet_path),
            }
        },
        "selection": {
            "question_ids": ["001-1", "002-1"],
            "question_hashes": {
                "001-1": canonical_sha256(dev["question"]),
                "002-1": canonical_sha256(locked["question"]),
            },
            "trace_contaminated_question_ids": [],
        },
        "splits": {
            "dev_question_ids": ["001-1"],
            "locked_question_ids": ["002-1"],
        },
        "artifact_sha256": {
            "dev": sha256_file(dev_path),
            "locked": sha256_file(locked_path),
            "excluded": sha256_file(excluded_path),
            "manifest": None,
        },
    }
    (evidence / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    schema = tmp_path / "schema.json"
    schema.write_text("{}", encoding="utf-8")
    video_dir = tmp_path / "videos"
    subtitle_dir = tmp_path / "subtitles"
    video_dir.mkdir()
    subtitle_dir.mkdir()
    (video_dir / "vid1.mp4").write_bytes(b"fixture-video-1")
    (video_dir / "vid2.mp4").write_bytes(b"fixture-video-2")
    return {
        "evidence": evidence,
        "schema": schema,
        "parquet": parquet_path,
        "video_dir": video_dir,
        "subtitle_dir": subtitle_dir,
    }


def test_relaxed_grounding_uses_context_slots_and_modality() -> None:
    reference = make_reference("001-1", "dev", "001")
    reference["evidence_contract"]["evidence_slots"].append(
        {"slot_id": "S2", "description": "spoken fact", "required": True}
    )
    reference["evidence_contract"]["required_modalities"].append("subtitle")
    reference["evidence_contract"]["sufficient_evidence_sets"][0]["items"].append(
        {
            "evidence_id": "E2",
            "slot_id": "S2",
            "required": True,
            "core_interval": {"start_sec": 4.0, "end_sec": 5.0},
            "context_interval": {"start_sec": 3.5, "end_sec": 5.5},
            "modality": "subtitle",
        }
    )
    exposures = [
        {
            "stage": "inspect",
            "modality": "visual",
            "start_seconds": 0.5,
            "end_seconds": 0.5,
            "frame_id": "F1",
        },
        {
            "stage": "subtitle",
            "modality": "subtitle",
            "start_seconds": 5.5,
            "end_seconds": 6.0,
            "frame_id": None,
        },
    ]
    score = score_relaxed_grounding(reference, exposures)
    assert score["grounded"] is True
    assert score["slot_coverage"] == 1.0
    assert score["context_item_recall"] == 1.0
    assert score["core_item_recall"] == 0.0


def test_ocr_reference_accepts_visual_frame_exposure() -> None:
    reference = make_reference("001-1", "dev", "001")
    item = reference["evidence_contract"]["sufficient_evidence_sets"][0]["items"][0]
    item["modality"] = "ocr"
    reference["evidence_contract"]["required_modalities"] = ["ocr"]
    score = score_relaxed_grounding(
        reference,
        [
            {
                "stage": "inspect",
                "modality": "visual",
                "start_seconds": 1.5,
                "end_seconds": 1.5,
                "frame_id": "F1",
            }
        ],
    )
    assert score["grounded"] is True


def test_normalize_exposures_for_all_three_strategies() -> None:
    direct = normalize_exposures(
        "direct",
        {
            "direct_preprocessing": {
                "frames": [{"id": "F1", "timestamp_seconds": 1.0}],
                "subtitle_intervals": [{"start_seconds": 2.0, "end_seconds": 3.0}],
            }
        },
    )
    coarse = normalize_exposures(
        "coarse_to_fine",
        {
            "coarse_to_fine": {
                "glance": {"frames": [{"id": "F2", "timestamp_seconds": 4.0}]},
                "rounds": [
                    {
                        "round": 1,
                        "selection_frames": [],
                        "frames": [{"id": "F3", "timestamp_seconds": 5.0}],
                        "candidate_subtitles": {"W1": "[6.000s-7.000s] hello"},
                        "subtitles": "",
                    }
                ],
            }
        },
    )
    active = normalize_exposures(
        "active_tree",
        {
            "active_tree": {
                "events": [
                    {
                        "type": "active_observation",
                        "frames": [{"id": "F4", "timestamp_seconds": 8.0}],
                        "subtitles": "[9.000s-10.000s] world",
                    }
                ]
            }
        },
    )
    assert {(item["modality"], item["start_seconds"]) for item in direct} == {
        ("visual", 1.0),
        ("subtitle", 2.0),
    }
    assert {(item["modality"], item["start_seconds"]) for item in coarse} == {
        ("visual", 4.0),
        ("visual", 5.0),
        ("subtitle", 6.0),
    }
    assert {(item["modality"], item["start_seconds"]) for item in active} == {
        ("visual", 8.0),
        ("subtitle", 9.0),
    }


def test_direct_subtitle_sampling_spans_the_full_video() -> None:
    track = SubtitleTrack(
        SubtitleCue(float(index * 10), float(index * 10 + 2), f"cue-{index}-" * 8)
        for index in range(20)
    )
    sampled = _sample_direct_subtitles(
        track,
        duration_seconds=200.0,
        max_chars=400,
        segments=4,
    )
    assert len(sampled) <= 400
    assert "cue-0-" in sampled
    assert "cue-15-" in sampled or "cue-16-" in sampled


def test_preflight_checks_sources_and_artifact_hashes(tmp_path: Path) -> None:
    paths = write_fixture_dataset(tmp_path)
    dataset = Evidence30Dataset.load(paths["evidence"])
    report = preflight_evidence30(
        dataset,
        schema_path=paths["schema"],
        parquet_path=paths["parquet"],
        video_dir=paths["video_dir"],
        subtitle_dir=paths["subtitle_dir"],
        verify_media_hashes=False,
    )
    assert report["ok"] is True
    assert report["records"] == 2

    (paths["evidence"] / "dev.jsonl").write_text("{}\n", encoding="utf-8")
    broken = preflight_evidence30(
        dataset,
        schema_path=paths["schema"],
        parquet_path=paths["parquet"],
        video_dir=paths["video_dir"],
        subtitle_dir=paths["subtitle_dir"],
        verify_media_hashes=False,
    )
    assert broken["ok"] is False
    assert any("artifact hash mismatch" in error for error in broken["errors"])


class FakeSession:
    calls: ClassVar[list[tuple[str, str]]] = []

    def __init__(self, _: Any, *, strategy: str, **__: Any) -> None:
        self.strategy = strategy

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None:
        return None

    def evaluate(self, question: VideoMMEQuestion) -> dict[str, Any]:
        self.calls.append((self.strategy, question.question_id))
        exposure = {
            "stage": "fixture",
            "modality": "visual",
            "start_seconds": 1.0,
            "end_seconds": 1.0,
            "frame_id": "F1",
        }
        if self.strategy == "direct":
            metadata = {
                "input_tokens": 10,
                "output_tokens": 1,
                "direct_preprocessing": {"reader": "fixture"},
                "exposures": [exposure],
            }
        elif self.strategy == "coarse_to_fine":
            metadata = {
                "coarse_to_fine": {
                    "degraded": False,
                    "stop_reason": "confidence_threshold",
                    "budget": {
                        "unique_limit": 32,
                        "cumulative_limit": 64,
                        "unique_frames": 1,
                        "cumulative_views": 1,
                    },
                },
                "exposures": [exposure],
            }
        else:
            metadata = {
                "verified": False,
                "active_tree": {
                    "degraded": False,
                    "stop_reason": "evidence_search_exhausted_with_missing_slots",
                    "resources": {
                        "max_model_calls": 14,
                        "model_call_count": 2,
                        "input_tokens": 20,
                        "output_tokens": 2,
                    },
                },
                "exposures": [exposure],
            }
        return {
            **question.to_dict(),
            "strategy": self.strategy,
            "with_subtitles": True,
            "prediction": "B",
            "correct": True,
            "wall_seconds": 1.0,
            "model_output": "B",
            "metadata": metadata,
        }


def test_dev_runner_is_visible_and_resumable(tmp_path: Path) -> None:
    paths = write_fixture_dataset(tmp_path)
    dataset = Evidence30Dataset.load(paths["evidence"])
    output = tmp_path / "dev-run"
    FakeSession.calls = []
    progress: list[dict[str, Any]] = []
    summary = run_evidence30_suite(
        dataset=dataset,
        split="dev",
        strategies=("direct", "coarse_to_fine", "active_tree"),
        config={"model": {}},
        parquet_path=paths["parquet"],
        video_dir=paths["video_dir"],
        subtitle_dir=paths["subtitle_dir"],
        output_dir=output,
        project_root=tmp_path,
        session_factory=FakeSession,
        progress=progress.append,
    )
    assert summary["engineering_pass"] is True
    assert summary["visibility"] == "full"
    assert len(FakeSession.calls) == 3
    assert all(item["item"] is not None for item in progress)
    assert (output / "items.jsonl").is_file()

    run_evidence30_suite(
        dataset=dataset,
        split="dev",
        strategies=("direct", "coarse_to_fine", "active_tree"),
        config={"model": {}},
        parquet_path=paths["parquet"],
        video_dir=paths["video_dir"],
        subtitle_dir=paths["subtitle_dir"],
        output_dir=output,
        project_root=tmp_path,
        session_factory=FakeSession,
    )
    assert len(FakeSession.calls) == 3


def test_locked_runner_suppresses_items_and_reserves_freeze(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    paths = write_fixture_dataset(tmp_path)
    dataset = Evidence30Dataset.load(paths["evidence"])
    output = tmp_path / "locked-run"
    registry = tmp_path / "registry.json"
    monkeypatch.setattr(
        "qwen3vl_agent.evaluation.evidence30_runner.verify_freeze_manifest",
        lambda _, **__: {
            "ok": True,
            "expected_freeze_id": "freeze-1",
            "current_freeze_id": "freeze-1",
        },
    )
    progress: list[dict[str, Any]] = []
    summary = run_evidence30_suite(
        dataset=dataset,
        split="locked",
        strategies=("direct", "coarse_to_fine", "active_tree"),
        config={"model": {}},
        parquet_path=paths["parquet"],
        video_dir=paths["video_dir"],
        subtitle_dir=paths["subtitle_dir"],
        output_dir=output,
        project_root=tmp_path,
        freeze_manifest=tmp_path / "freeze.json",
        locked_registry=registry,
        session_factory=FakeSession,
        progress=progress.append,
    )
    public = json.dumps(summary)
    assert summary["visibility"] == "aggregate"
    assert "002-1" not in public
    assert all(item["item"] is None for item in progress)
    sealed = Path(summary["sealed_artifact"]["path"])
    assert "002-1" in sealed.read_text(encoding="utf-8")
    assert summary["sealed_artifact"]["sha256"] == sha256_file(sealed)

    dataset_id = sha256_file(paths["evidence"] / "manifest.json")
    mark_locked_spent(registry, dataset_id=dataset_id)
    state = json.loads(registry.read_text(encoding="utf-8"))
    assert state["datasets"][dataset_id]["status"] == "spent"


def test_freeze_detects_source_drift(tmp_path: Path) -> None:
    project = tmp_path / "project"
    package = project / "qwen3vl_agent"
    schemas = project / "schemas"
    evidence = project / "evidence"
    model = project / "model"
    package.mkdir(parents=True)
    schemas.mkdir()
    evidence.mkdir()
    model.mkdir()
    (package / "module.py").write_text("VALUE = 1\n", encoding="utf-8")
    (schemas / "schema.json").write_text("{}", encoding="utf-8")
    (project / "pyproject.toml").write_text("[project]\nname='fixture'\n", encoding="utf-8")
    config = project / "config.yaml"
    config.write_text("model: {}\n", encoding="utf-8")
    for name in ("manifest.json", "dev.jsonl", "locked.jsonl", "excluded.jsonl"):
        (evidence / name).write_text("{}\n" if name == "manifest.json" else "", encoding="utf-8")
    (model / "weights.bin").write_bytes(b"weights")
    manifest = create_freeze_manifest(
        project_root=project,
        config_path=config,
        evidence_root=evidence,
        model_path=model,
    )
    freeze_path = write_freeze_manifest(project / "freeze.json", manifest)
    assert verify_freeze_manifest(freeze_path)["ok"] is True
    assert verify_freeze_manifest(freeze_path, runtime_config={"model": {}})["ok"] is True
    assert (
        verify_freeze_manifest(
            freeze_path,
            runtime_config={"model": {"path": "different"}},
        )["ok"]
        is False
    )
    (package / "module.py").write_text("VALUE = 2\n", encoding="utf-8")
    assert verify_freeze_manifest(freeze_path)["ok"] is False


class CapturingModel(BaseVideoModel):
    def __init__(self) -> None:
        super().__init__("fixture")
        self.messages: list[list[dict[str, Any]]] = []

    def load(self) -> None:
        self._loaded = True

    def generate(self, messages: list[dict[str, Any]], **_: Any) -> ModelOutput:
        self.messages.append(messages)
        return ModelOutput("B", {"input_tokens": 10, "output_tokens": 1})

    def unload(self) -> None:
        self._loaded = False


def test_runtime_prompt_never_receives_reference_evidence(tmp_path: Path) -> None:
    video_dir = tmp_path / "videos"
    subtitle_dir = tmp_path / "subtitles"
    video_dir.mkdir()
    subtitle_dir.mkdir()
    (video_dir / "vid1.mp4").write_bytes(b"not-decoded-by-fake-model")
    question = VideoMMEQuestion(
        video_id="vid1",
        question_id="001-1",
        duration="short",
        domain="test",
        sub_category="test",
        task_type="test",
        question="Which option is visible?",
        options=("A. One.", "B. Two.", "C. Three.", "D. Four."),
        answer="B",
    )
    model = CapturingModel()
    with VideoMMEStrategySession(
        {
            "model": {},
            "evaluation": {
                "direct_frames": 16,
                "direct_min_pixels": 4096,
                "direct_max_pixels": 131072,
                "direct_total_pixels": 2097152,
                "direct_fps": 1.0,
                "direct_subtitle_max_chars": 8000,
                "direct_subtitle_segments": 16,
            },
        },
        strategy="direct",
        video_dir=video_dir,
        subtitle_dir=subtitle_dir,
        with_subtitles=False,
        native_video_decode=True,
        model=model,
    ) as session:
        result = session.evaluate(question)
    prompt = json.dumps(model.messages, ensure_ascii=False)
    assert result["prediction"] == "B"
    assert "SECRET_ATOMIC_FACT" not in prompt
    assert "SECRET_HARD_NEGATIVE" not in prompt
    assert "context_interval" not in prompt
