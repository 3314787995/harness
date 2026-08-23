from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from qwen3vl_agent.coarse_to_fine import CachedVideo, SubtitleCue, SubtitleTrack, TimeWindow
from qwen3vl_agent.coarse_to_fine.types import FrameRef
from qwen3vl_agent.evaluation.ablations import ABLATION_POLICY_ID, build_unified_answer_prompt
from qwen3vl_agent.evaluation.core_dense import (
    CORE_DENSE_POLICY_ID,
    CoreDenseConfig,
    CoreDenseSession,
    CoreWindowSpec,
    DevReferenceSet,
    build_core_packet,
    load_oracle_failure_rows,
    sample_core_specs,
    validate_core_packet,
)
from qwen3vl_agent.evaluation.videomme import VideoMMEQuestion
from qwen3vl_agent.models.base import BaseVideoModel, ModelOutput


def _question() -> VideoMMEQuestion:
    return VideoMMEQuestion(
        video_id="vid",
        question_id="001-1",
        duration="long",
        domain="test",
        sub_category="test",
        task_type="test",
        question="What happens?",
        options=("A. One", "B. Two", "C. Three", "D. Four"),
        answer="B",
    )


def _reference() -> dict[str, Any]:
    return {
        "question": {
            "official_answer": {"benchmark_label": "B", "option_id": "O2"},
        },
        "evidence_contract": {
            "evidence_slots": [
                {"slot_id": "S1", "required": True},
                {"slot_id": "S2", "required": True},
            ],
            "sufficient_evidence_sets": [
                {
                    "set_id": "ES1",
                    "items": [
                        {
                            "evidence_id": "E1",
                            "slot_id": "S1",
                            "required": True,
                            "modality": "visual",
                            "core_interval": {"start_sec": 20.0, "end_sec": 30.0},
                            "context_interval": {"start_sec": 0.0, "end_sec": 99.0},
                            "atomic_fact": "SECRET_VISUAL_FACT",
                        },
                        {
                            "evidence_id": "E2",
                            "slot_id": "S2",
                            "required": True,
                            "modality": "subtitle",
                            "core_interval": {"start_sec": 80.0, "end_sec": 90.0},
                            "context_interval": {"start_sec": 0.0, "end_sec": 99.0},
                            "atomic_fact": "SECRET_SUBTITLE_FACT",
                        },
                    ],
                }
            ],
        },
        "hard_negatives": [
            {
                "interval": {"start_sec": 40.0, "end_sec": 45.0},
                "modalities": ["visual"],
                "plausibility": "SECRET_HARD_NEGATIVE",
            }
        ],
    }


def _cached_video(tmp_path: Path, *, count: int = 100) -> CachedVideo:
    frames: list[FrameRef] = []
    for index in range(count):
        path = tmp_path / f"F{index:03d}.jpg"
        path.write_bytes(b"fixture")
        frames.append(FrameRef(f"F{index:03d}", float(index), str(path)))
    return CachedVideo(
        source_path=str(tmp_path / "vid.mp4"),
        cache_dir=str(tmp_path),
        duration_seconds=float(count),
        source_fps=30.0,
        width=640,
        height=360,
        sample_fps=1.0,
        frames=tuple(frames),
        cache_hit=True,
    )


def _config(tmp_path: Path) -> dict[str, Any]:
    return {
        "model": {},
        "core_dense": {
            "cache_dir": str(tmp_path / "cache"),
            "subtitle_max_chars": 1_000,
            "max_new_tokens": 16,
            "variants": {
                "core_16": {
                    "max_frames": 16,
                    "min_pixels": 4_096,
                    "max_pixels": 131_072,
                    "total_pixels": 2_097_152,
                },
                "core_32": {
                    "max_frames": 32,
                    "min_pixels": 4_096,
                    "max_pixels": 65_536,
                    "total_pixels": 2_097_152,
                },
            },
        },
    }


def _baseline_row(question_id: str, *, correct: bool) -> dict[str, Any]:
    return {
        "question_id": question_id,
        "policy_id": ABLATION_POLICY_ID,
        "packet_source": "oracle_context",
        "completed": True,
        "correct": correct,
        "prediction": "B" if correct else "A",
        "relaxed_grounding": {
            "grounded": True,
            "core_item_recall": 1.0,
        },
    }


def test_dev_reference_loader_does_not_require_locked_file(tmp_path: Path) -> None:
    root = tmp_path / "evidence"
    root.mkdir()
    manifest = {"splits": {"dev_question_ids": ["001-1"]}}
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    record = {"split": "dev", "source": {"question_id": "001-1"}}
    (root / "dev.jsonl").write_text(json.dumps(record) + "\n", encoding="utf-8")
    dataset = DevReferenceSet.load(root)
    assert dataset.question_ids == ("001-1",)
    assert not (root / "locked.jsonl").exists()


def test_oracle_failure_selection_is_dev_only_and_exact(tmp_path: Path) -> None:
    path = tmp_path / "items.jsonl"
    rows = [
        _baseline_row("001-1", correct=False),
        _baseline_row("002-1", correct=True),
        _baseline_row("003-1", correct=True),
    ]
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    failures = load_oracle_failure_rows(
        path,
        dev_question_ids=["001-1", "002-1", "003-1"],
        expected_count=1,
    )
    assert list(failures) == ["001-1"]

    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_baseline_row("LOCKED-1", correct=False)) + "\n")
    with pytest.raises(ValueError, match="non-dev"):
        load_oracle_failure_rows(
            path,
            dev_question_ids=["001-1", "002-1", "003-1"],
            expected_count=1,
        )


def test_core_sampling_balances_local_item_against_global_item(tmp_path: Path) -> None:
    cached = _cached_video(tmp_path)
    specs = [
        CoreWindowSpec("E-global", "S-global", "visual", TimeWindow("G", 0.0, 99.0)),
        CoreWindowSpec("E-local", "S-local", "visual", TimeWindow("L", 80.0, 90.0)),
    ]
    selected, source_count, allocations = sample_core_specs(cached, specs, total=16)
    assert len(selected) == 16
    assert source_count == 100
    assert [item["requested_frames"] for item in allocations] == [8, 8]
    assert sum(80.0 <= frame.timestamp_seconds <= 90.0 for frame in selected) >= 7


def test_core_packet_separates_visual_and_late_subtitle_windows(tmp_path: Path) -> None:
    cached = _cached_video(tmp_path)
    track = SubtitleTrack(
        [
            SubtitleCue(2.0, 3.0, "early distractor"),
            SubtitleCue(84.0, 85.0, "required late cue"),
        ]
    )
    config = CoreDenseConfig.from_mapping(_config(tmp_path)["core_dense"])
    packet = build_core_packet(
        "core_16",
        reference=_reference(),
        cached=cached,
        subtitle_track=track,
        config=config,
    )
    assert all(20.0 <= frame.timestamp_seconds <= 30.0 for frame in packet.frames)
    assert "required late cue" in packet.subtitles
    assert "early distractor" not in packet.subtitles
    score = validate_core_packet(_reference(), packet, max_frames=16)
    assert score["grounded"] is True
    assert score["core_item_recall"] == 1.0
    assert score["chosen_set_core_recall"] == 1.0


def test_core_packet_validation_rejects_missing_required_modality(tmp_path: Path) -> None:
    cached = _cached_video(tmp_path)
    config = CoreDenseConfig.from_mapping(_config(tmp_path)["core_dense"])
    packet = build_core_packet(
        "core_16",
        reference=_reference(),
        cached=cached,
        subtitle_track=None,
        config=config,
    )
    with pytest.raises(ValueError, match="every required item"):
        validate_core_packet(_reference(), packet, max_frames=16)


class CapturingModel(BaseVideoModel):
    def __init__(self) -> None:
        super().__init__("fixture")
        self.messages: list[list[dict[str, Any]]] = []
        self.kwargs: list[dict[str, Any]] = []

    def load(self) -> None:
        self._loaded = True

    def unload(self) -> None:
        self._loaded = False

    def generate(self, messages: list[dict[str, Any]], **kwargs: Any) -> ModelOutput:
        self.messages.append(messages)
        self.kwargs.append(kwargs)
        return ModelOutput("B", {"input_tokens": 10, "output_tokens": 1})


def test_core32_uses_same_total_pixels_and_unified_head_without_gold(
    tmp_path: Path,
) -> None:
    cached = _cached_video(tmp_path)
    track = SubtitleTrack([SubtitleCue(84.0, 85.0, "ordinary source subtitle")])
    raw_config = _config(tmp_path)
    config = CoreDenseConfig.from_mapping(raw_config["core_dense"])
    packet = build_core_packet(
        "core_32",
        reference=_reference(),
        cached=cached,
        subtitle_track=track,
        config=config,
    )
    model = CapturingModel()
    with CoreDenseSession(raw_config, model=model) as session:
        result = session.evaluate(_question(), packet, reference=_reference())
    media = model.messages[0][0]["content"][0]
    serialized = json.dumps(model.messages, ensure_ascii=False)
    prompt = build_unified_answer_prompt(_question(), packet)  # type: ignore[arg-type]
    assert result["policy_id"] == CORE_DENSE_POLICY_ID
    assert result["prediction"] == "B"
    assert media["max_frames"] == 32
    assert media["max_pixels"] == 65_536
    assert media["total_pixels"] == 2_097_152
    assert "SECRET_VISUAL_FACT" not in serialized
    assert "SECRET_SUBTITLE_FACT" not in serialized
    assert "SECRET_HARD_NEGATIVE" not in serialized
    assert "official_answer" not in prompt
    assert model.kwargs == [{"max_new_tokens": 16}]


def test_core_dense_config_requires_both_fixed_variants(tmp_path: Path) -> None:
    value = _config(tmp_path)["core_dense"]
    del value["variants"]["core_32"]
    with pytest.raises(ValueError, match="exactly"):
        CoreDenseConfig.from_mapping(value)
