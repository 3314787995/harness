from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image

from qwen3vl_agent.coarse_to_fine import CachedVideo, FrameRef, SubtitleTrack, TimeWindow
from qwen3vl_agent.coarse_to_fine.adapters import AnswerRecord, MultipleChoiceAdapter
from qwen3vl_agent.coarse_to_fine.agent import CoarseToFineVideoAgent
from qwen3vl_agent.coarse_to_fine.config import CoarseToFineConfig
from qwen3vl_agent.coarse_to_fine.prompts import (
    build_glance_prompt,
    build_selection_prompt,
    parse_glance,
)
from qwen3vl_agent.coarse_to_fine.types import FrameBudget
from qwen3vl_agent.models.base import BaseVideoModel, ModelOutput


class FakeModel(BaseVideoModel):
    def __init__(self, responses: list[str]) -> None:
        super().__init__("fake", device="cpu", dtype="float32")
        self.responses = iter(responses)
        self.calls: list[dict[str, Any]] = []

    def load(self) -> None:
        self._loaded = True

    def unload(self) -> None:
        self._loaded = False

    def generate(self, messages, *, videos=None, images=None, **kwargs):
        self.calls.append(
            {
                "messages": messages,
                "videos": videos,
                "images": images,
                "kwargs": kwargs,
            }
        )
        return ModelOutput(next(self.responses), {"fake": True})


class FakeCache:
    def __init__(self, cached: CachedVideo) -> None:
        self.cached = cached

    def prepare(self, video_path: str) -> CachedVideo:
        assert video_path == self.cached.source_path
        return self.cached


def make_cached_video(tmp_path: Path) -> CachedVideo:
    frame_dir = tmp_path / "frames"
    frame_dir.mkdir()
    frames: list[FrameRef] = []
    for index in range(25):
        path = frame_dir / f"F{index:06d}.jpg"
        Image.new("RGB", (64, 48), color=(index * 5, 20, 40)).save(path)
        frames.append(FrameRef(f"F{index:06d}", index * 2.5, str(path)))
    return CachedVideo(
        source_path=str(tmp_path / "video.mp4"),
        cache_dir=str(tmp_path),
        duration_seconds=60.0,
        source_fps=30.0,
        width=1280,
        height=720,
        sample_fps=1.0,
        frames=tuple(frames),
        cache_hit=True,
    )


def test_local_search_refines_and_stops_on_confidence(tmp_path: Path) -> None:
    cached = make_cached_video(tmp_path)
    model = FakeModel(
        [
            '{"scope":"local"}',
            (
                '{"window_ids":["R1-W1","R1-W4"],"rationale":"likely",'
                '"evidence_needed":"object"}'
            ),
            (
                '{"answer":"C","reason":"coarse evidence","confidence":1,'
                '"evidence_frame_ids":[],"missing_evidence":"look closer"}'
            ),
            (
                '{"window_ids":["R2-W1","R2-W2"],"rationale":"refine",'
                '"evidence_needed":"detail"}'
            ),
            (
                '{"answer":"A","reason":"direct evidence","confidence":3,'
                '"evidence_frame_ids":["F000006"],"missing_evidence":""}'
            ),
        ]
    )
    config = CoarseToFineConfig(cache_dir=str(tmp_path), max_rounds=3)
    agent = CoarseToFineVideoAgent(model, config=config, cache=FakeCache(cached))
    agent.load()

    output = agent.generate(
        [{"role": "user", "content": "What happened?"}],
        videos=[cached.source_path],
        choices=["first", "second", "third", "fourth"],
    )

    trace = output.metadata["coarse_to_fine"]
    assert output.text == "A"
    assert trace["scope"] == "local"
    assert len(trace["rounds"]) == 2
    assert trace["stop_reason"] == "confidence_threshold"
    assert trace["budget"]["unique_frames"] <= config.unique_frame_budget
    assert trace["budget"]["cumulative_views"] <= config.cumulative_frame_budget
    assert model.calls[1]["images"]
    assert isinstance(model.calls[2]["videos"][0], list)


def test_global_question_bypasses_window_search(tmp_path: Path) -> None:
    cached = make_cached_video(tmp_path)
    model = FakeModel(
        [
            '{"scope":"global"}',
            (
                '{"answer":"B","reason":"seen throughout","confidence":3,'
                '"evidence_frame_ids":[],"missing_evidence":""}'
            ),
        ]
    )
    agent = CoarseToFineVideoAgent(
        model,
        config=CoarseToFineConfig(cache_dir=str(tmp_path)),
        cache=FakeCache(cached),
    )
    agent.load()

    output = agent.generate(
        [{"role": "user", "content": "What is the overall topic?"}],
        videos=[cached.source_path],
        choices=["one", "two", "three", "four"],
    )

    trace = output.metadata["coarse_to_fine"]
    assert output.text == "B"
    assert trace["stop_reason"] == "global_direct"
    assert len(model.calls) == 2


def test_explicit_absence_question_overrides_local_route(tmp_path: Path) -> None:
    cached = make_cached_video(tmp_path)
    model = FakeModel(
        [
            '{"scope":"local"}',
            '{"answer":"B","reason":"only B is absent","confidence":3,"missing_evidence":""}',
        ]
    )
    agent = CoarseToFineVideoAgent(
        model,
        config=CoarseToFineConfig(cache_dir=str(tmp_path)),
        cache=FakeCache(cached),
    )
    agent.load()

    output = agent.generate(
        [{"role": "user", "content": "Which activity is not shown in the video?"}],
        videos=[cached.source_path],
        choices=["one", "two", "three", "four"],
    )

    trace = output.metadata["coarse_to_fine"]
    assert output.text == "B"
    assert trace["scope"] == "global"
    assert trace["glance"]["model_scope"] == "local"
    assert trace["glance"]["controller_override"]
    assert trace["stop_reason"] == "global_direct"
    assert len(model.calls) == 2


def test_invalid_decision_is_marked_and_degrades_to_direct(tmp_path: Path) -> None:
    cached = make_cached_video(tmp_path)
    model = FakeModel(["not-json", "C"])
    agent = CoarseToFineVideoAgent(
        model,
        config=CoarseToFineConfig(cache_dir=str(tmp_path)),
        cache=FakeCache(cached),
    )
    agent.load()

    output = agent.generate(
        [{"role": "user", "content": "Question"}],
        videos=[cached.source_path],
        choices=["one", "two", "three", "four"],
    )

    trace = output.metadata["coarse_to_fine"]
    assert output.text == "C"
    assert trace["degraded"] is True
    assert trace["stop_reason"] == "degraded_direct_fallback"
    assert "DecisionError" in trace["degraded_reason"]


def test_truncated_reasoning_can_be_repaired_without_new_media(tmp_path: Path) -> None:
    cached = make_cached_video(tmp_path)
    model = FakeModel(
        [
            '{"scope":"local"}',
            '{"window_ids":["R1-W1"]}',
            '{"answer":"B","reason":"truncated',
            (
                '{"answer":"B","reason":"brief","confidence":3,'
                '"missing_evidence":""}'
            ),
        ]
    )
    agent = CoarseToFineVideoAgent(
        model,
        config=CoarseToFineConfig(
            cache_dir=str(tmp_path),
            protocol_repair_attempts=1,
        ),
        cache=FakeCache(cached),
    )
    agent.load()

    output = agent.generate(
        [{"role": "user", "content": "Question"}],
        videos=[cached.source_path],
        choices=["one", "two", "three", "four"],
    )

    trace = output.metadata["coarse_to_fine"]
    assert output.text == "B"
    assert trace["degraded"] is False
    assert trace["protocol_retries"][0]["status"] == "success"
    assert model.calls[3]["videos"] is None
    assert model.calls[3]["images"] is None


def test_subtitle_track_selects_overlapping_cues(tmp_path: Path) -> None:
    path = tmp_path / "demo.srt"
    path.write_text(
        "1\n00:00:01,000 --> 00:00:02,000\nfirst\n\n"
        "2\n00:00:08,000 --> 00:00:09,000\nsecond\n",
        encoding="utf-8",
    )
    track = SubtitleTrack.from_srt(path)
    text = track.text_for_windows(
        [TimeWindow("W", 7.5, 9.5)],
        max_chars=1_000,
    )
    assert "second" in text
    assert "first" not in text


def test_compact_glance_protocol_does_not_require_a_reason() -> None:
    adapter = MultipleChoiceAdapter(["one", "two"])
    prompt = build_glance_prompt("Where does the event happen?", adapter)
    decision = parse_glance('{"scope":"local"}')

    assert '{"scope":"global"}' in prompt
    assert decision.scope == "local"
    assert decision.reason == ""


def test_window_selector_receives_aligned_subtitles() -> None:
    adapter = MultipleChoiceAdapter(["one", "two"])
    windows = [TimeWindow("R1-W0", 0.0, 10.0)]
    prompt = build_selection_prompt(
        "What are they discussing?",
        adapter,
        windows,
        top_k=1,
        history=[],
        candidate_subtitles={"R1-W0": "[2s-4s] choose a best man"},
    )

    assert "R1-W0" in prompt
    assert "choose a best man" in prompt


def test_frame_budget_counts_unique_and_repeated_views() -> None:
    frames = [FrameRef("A", 0.0, "a.jpg"), FrameRef("B", 1.0, "b.jpg")]
    budget = FrameBudget(unique_limit=2, cumulative_limit=4)
    budget.consume(frames, purpose="first")
    budget.consume(frames, purpose="second")
    assert budget.to_dict()["unique_frames"] == 2
    assert budget.to_dict()["cumulative_views"] == 4


def test_multiple_choice_vote_uses_confidence_then_recency_for_ties() -> None:
    adapter = MultipleChoiceAdapter(["one", "two", "three", "four"])
    records = [
        AnswerRecord("A", 1, 1),
        AnswerRecord("B", 2, 2),
        AnswerRecord("A", 1, 3),
        AnswerRecord("B", 3, 4),
    ]
    assert adapter.vote(records) == "B"
