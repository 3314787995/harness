from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from qwen3vl_agent.coarse_to_fine import CachedVideo, SubtitleCue, SubtitleTrack
from qwen3vl_agent.coarse_to_fine.types import FrameRef
from qwen3vl_agent.evaluation.ablations import (
    AblationConfig,
    EvidencePacket,
    UnifiedAnswerSession,
    build_evidence_packet,
    build_unified_answer_prompt,
    choose_oracle_set,
    load_dev_trace_rows,
    select_priority_frames,
)
from qwen3vl_agent.evaluation.evidence30 import score_relaxed_grounding
from qwen3vl_agent.evaluation.videomme import VideoMMEQuestion
from qwen3vl_agent.models.base import BaseVideoModel, ModelOutput


def _question() -> VideoMMEQuestion:
    return VideoMMEQuestion(
        video_id="vid",
        question_id="001-1",
        duration="short",
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
                {"slot_id": "S1", "description": "SECRET_SLOT", "required": True}
            ],
            "sufficient_evidence_sets": [
                {
                    "set_id": "ES-WIDE",
                    "items": [
                        {
                            "evidence_id": "E1",
                            "slot_id": "S1",
                            "required": True,
                            "modality": "visual",
                            "core_interval": {"start_sec": 2.0, "end_sec": 3.0},
                            "context_interval": {"start_sec": 0.0, "end_sec": 10.0},
                            "atomic_fact": "SECRET_ATOMIC_FACT",
                        }
                    ],
                },
                {
                    "set_id": "ES-NARROW",
                    "items": [
                        {
                            "evidence_id": "E2",
                            "slot_id": "S1",
                            "required": True,
                            "modality": "visual",
                            "core_interval": {"start_sec": 4.5, "end_sec": 5.5},
                            "context_interval": {"start_sec": 4.0, "end_sec": 6.0},
                            "atomic_fact": "SECRET_ALTERNATIVE_FACT",
                        }
                    ],
                },
            ],
        },
        "hard_negatives": [
            {
                "interval": {"start_sec": 8.0, "end_sec": 9.0},
                "modalities": ["visual"],
                "plausibility": "SECRET_HARD_NEGATIVE",
            }
        ],
    }


def _cached_video(tmp_path: Path) -> CachedVideo:
    frames: list[FrameRef] = []
    for index in range(12):
        path = tmp_path / f"F{index:02d}.jpg"
        path.write_bytes(b"fixture")
        frames.append(FrameRef(f"F{index:02d}", float(index), str(path)))
    return CachedVideo(
        source_path=str(tmp_path / "vid.mp4"),
        cache_dir=str(tmp_path),
        duration_seconds=12.0,
        source_fps=30.0,
        width=640,
        height=360,
        sample_fps=1.0,
        frames=tuple(frames),
        cache_hit=True,
    )


def _config(tmp_path: Path, *, max_frames: int = 4) -> AblationConfig:
    return AblationConfig.from_mapping(
        {
            "cache_dir": str(tmp_path / "cache"),
            "max_frames": max_frames,
            "subtitle_max_chars": 1_000,
            "min_pixels": 4_096,
            "max_pixels": 131_072,
            "total_pixels": 2_097_152,
        }
    )


def test_oracle_chooses_shortest_valid_context_set() -> None:
    assert choose_oracle_set(_reference())["set_id"] == "ES-NARROW"


def test_priority_selection_keeps_cited_frames_with_matched_budget(tmp_path: Path) -> None:
    cached = _cached_video(tmp_path)
    selected, priority_count = select_priority_frames(
        list(cached.frames),
        priority_ids={"F01", "F10"},
        limit=4,
    )
    assert len(selected) == 4
    assert priority_count == 2
    assert {frame.id for frame in selected} >= {"F01", "F10"}
    assert [frame.timestamp_seconds for frame in selected] == sorted(
        frame.timestamp_seconds for frame in selected
    )


def test_packets_replay_trace_and_oracle_without_gold_text(tmp_path: Path) -> None:
    cached = _cached_video(tmp_path)
    direct_frames = [cached.frames[index].to_dict() for index in (0, 3, 6, 9)]
    active_frames = [cached.frames[index].to_dict() for index in (1, 4, 5, 10, 11)]
    direct = {
        "metadata": {
            "direct_preprocessing": {"frames": direct_frames},
            "exposures": [
                {
                    "modality": "subtitle",
                    "start_seconds": 1.0,
                    "end_seconds": 2.0,
                }
            ],
        }
    }
    active = {
        "metadata": {
            "active_tree": {
                "events": [{"frames": active_frames}],
                "evidence_ledger": [{"source_frame_ids": ["F10"]}],
                "verification_attempts": [],
            },
            "exposures": [
                {
                    "modality": "subtitle",
                    "start_seconds": 4.0,
                    "end_seconds": 5.0,
                }
            ],
        }
    }
    track = SubtitleTrack(
        [
            SubtitleCue(1.0, 2.0, "early cue"),
            SubtitleCue(4.0, 5.0, "oracle cue"),
        ]
    )
    config = _config(tmp_path)
    packets = {
        source: build_evidence_packet(
            source,
            reference=_reference(),
            direct_row=direct,
            active_row=active,
            cached=cached,
            subtitle_track=track,
            config=config,
        )
        for source in ("direct_replay", "active_tree_replay", "oracle_context")
    }
    assert all(len(packet.frames) <= 4 for packet in packets.values())
    assert "F10" in {frame.id for frame in packets["active_tree_replay"].frames}
    assert packets["oracle_context"].oracle_set_id == "ES-NARROW"
    assert all(4.0 <= frame.timestamp_seconds <= 6.0 for frame in packets["oracle_context"].frames)
    for packet in packets.values():
        prompt = build_unified_answer_prompt(_question(), packet)
        assert "SECRET_ATOMIC_FACT" not in prompt
        assert "SECRET_ALTERNATIVE_FACT" not in prompt
        assert "SECRET_HARD_NEGATIVE" not in prompt
        assert "official_answer" not in prompt


def test_oracle_keeps_late_subtitle_separate_from_global_visual_window(
    tmp_path: Path,
) -> None:
    frames: list[FrameRef] = []
    for index in range(28):
        path = tmp_path / f"long-{index:02d}.jpg"
        path.write_bytes(b"fixture")
        frames.append(FrameRef(f"L{index:02d}", float(index * 100), str(path)))
    cached = CachedVideo(
        source_path=str(tmp_path / "long.mp4"),
        cache_dir=str(tmp_path),
        duration_seconds=2_706.0,
        source_fps=30.0,
        width=640,
        height=360,
        sample_fps=1.0,
        frames=tuple(frames),
        cache_hit=True,
    )
    reference = {
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
                            "modality": "visual",
                            "core_interval": {"start_sec": 0.0, "end_sec": 2_705.0},
                            "context_interval": {"start_sec": 0.0, "end_sec": 2_705.0},
                        },
                        {
                            "evidence_id": "E2",
                            "slot_id": "S2",
                            "modality": "subtitle",
                            "core_interval": {"start_sec": 2_427.0, "end_sec": 2_454.0},
                            "context_interval": {
                                "start_sec": 2_427.0,
                                "end_sec": 2_454.0,
                            },
                        },
                    ],
                }
            ],
        },
        "hard_negatives": [],
    }
    track = SubtitleTrack(
        [
            SubtitleCue(1.0, 2.0, "early " * 2_000),
            SubtitleCue(2_430.0, 2_431.0, "required late cue"),
        ]
    )
    packet = build_evidence_packet(
        "oracle_context",
        reference=reference,
        direct_row={},
        active_row={},
        cached=cached,
        subtitle_track=track,
        config=_config(tmp_path, max_frames=16),
    )
    assert "required late cue" in packet.subtitles
    assert "early " not in packet.subtitles
    assert score_relaxed_grounding(reference, packet.exposures())["grounded"] is True


def test_load_dev_trace_rejects_non_dev_rows(tmp_path: Path) -> None:
    path = tmp_path / "items.jsonl"
    path.write_text(
        json.dumps({"question_id": "LOCKED-1", "strategy": "direct"}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="non-dev"):
        load_dev_trace_rows(path, dev_question_ids=["001-1"])


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


def test_unified_session_prompt_never_receives_reference_semantics(tmp_path: Path) -> None:
    cached = _cached_video(tmp_path)
    packet = EvidencePacket(
        source="oracle_context",
        frames=tuple(cached.frames[4:6]),
        subtitles="[4.000s-5.000s] ordinary source subtitle",
        source_frame_count=2,
        subtitle_window_count=1,
        oracle_set_id="ES-NARROW",
    )
    model = CapturingModel()
    config = {
        "model": {},
        "ablation": _config(tmp_path, max_frames=2).to_dict(),
    }
    with UnifiedAnswerSession(config, model=model) as session:
        result = session.evaluate(_question(), packet, reference=_reference())
    serialized = json.dumps(model.messages, ensure_ascii=False)
    assert result["prediction"] == "B"
    assert "SECRET_ATOMIC_FACT" not in serialized
    assert "SECRET_ALTERNATIVE_FACT" not in serialized
    assert "SECRET_HARD_NEGATIVE" not in serialized
    assert "context_interval" not in serialized
    assert model.kwargs == [{"max_new_tokens": 16}]


def test_ablation_config_rejects_invalid_frame_budget(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="max_frames"):
        AblationConfig.from_mapping(
            {"cache_dir": str(tmp_path / "cache"), "max_frames": 0}
        )
