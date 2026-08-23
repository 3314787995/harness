from __future__ import annotations

import gc
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, Self

from qwen3vl_agent.active_tree import ActiveTreeConfig, ActiveTreeVideoAgent
from qwen3vl_agent.active_tree.replay import render_trace_html
from qwen3vl_agent.coarse_to_fine import (
    CoarseToFineConfig,
    CoarseToFineVideoAgent,
    SubtitleTrack,
    TimeWindow,
    VideoEvidenceCache,
)
from qwen3vl_agent.coarse_to_fine.adapters import MultipleChoiceAdapter
from qwen3vl_agent.coarse_to_fine.prompts import build_direct_prompt
from qwen3vl_agent.evaluation.evidence30 import (
    STRATEGIES,
    normalize_exposures,
    subtitle_exposures,
)
from qwen3vl_agent.evaluation.videomme import VideoMMEQuestion
from qwen3vl_agent.factory import build_model
from qwen3vl_agent.models.base import BaseVideoModel


def _sample_direct_subtitles(
    track: SubtitleTrack,
    *,
    duration_seconds: float,
    max_chars: int,
    segments: int,
) -> str:
    if max_chars <= 0 or not track.cues:
        return ""
    whole = TimeWindow("ROOT", 0.0, max(duration_seconds, 0.001), depth=0)
    full_text = track.text_for_windows([whole], max_chars=1_000_000_000)
    if len(full_text) <= max_chars:
        return full_text

    segment_count = max(1, segments)
    width = whole.end_seconds / segment_count
    windows = [
        TimeWindow(
            f"DIRECT-{index:02d}",
            index * width,
            whole.end_seconds if index == segment_count - 1 else (index + 1) * width,
            depth=0,
        )
        for index in range(segment_count)
    ]
    populated = [
        window
        for window in windows
        if track.text_for_windows([window], max_chars=1_000_000_000)
    ]
    if not populated:
        return ""
    per_segment = max(1, (max_chars - len(populated) + 1) // len(populated))
    chunks = [
        track.text_for_windows([window], max_chars=per_segment)
        for window in populated
    ]
    return "\n".join(chunk for chunk in chunks if chunk)


class VideoMMEStrategySession:
    """Reusable, single-strategy local evaluation session."""

    def __init__(
        self,
        config: Mapping[str, Any],
        *,
        strategy: str,
        video_dir: str | Path,
        subtitle_dir: str | Path,
        with_subtitles: bool = True,
        native_video_decode: bool = False,
        replay_dir: str | Path | None = None,
        model: BaseVideoModel | None = None,
        model_factory: Callable[[Mapping[str, Any] | None], BaseVideoModel] = build_model,
    ) -> None:
        if strategy not in STRATEGIES:
            raise ValueError(f"Unsupported strategy: {strategy}")
        self.config = dict(config)
        self.strategy = strategy
        self.video_dir = Path(video_dir).expanduser().resolve()
        self.subtitle_dir = Path(subtitle_dir).expanduser().resolve()
        self.with_subtitles = with_subtitles
        self.native_video_decode = native_video_decode
        self.replay_dir = (
            Path(replay_dir).expanduser().resolve() if replay_dir is not None else None
        )
        self.model = model or model_factory(self.config.get("model"))
        self.coarse_config = CoarseToFineConfig.from_mapping(
            self.config.get("coarse_to_fine")
        )
        self.active_config = ActiveTreeConfig.from_mapping(self.config.get("active_tree"))
        evaluation_config = self.config.get("evaluation", {})
        if not isinstance(evaluation_config, Mapping):
            raise TypeError("evaluation config must be a mapping")
        self.direct_frames = int(
            evaluation_config.get("direct_frames", self.coarse_config.global_frames)
        )
        if self.direct_frames < 1:
            raise ValueError("evaluation.direct_frames must be positive")
        model_config = self.config.get("model", {})
        model_video = model_config.get("video", {}) if isinstance(model_config, Mapping) else {}
        self.direct_video_options = {
            "min_pixels": int(
                evaluation_config.get(
                    "direct_min_pixels",
                    model_video.get("min_pixels", 4 * 32 * 32),
                )
            ),
            "max_pixels": int(
                evaluation_config.get(
                    "direct_max_pixels",
                    model_video.get("max_pixels", 256 * 32 * 32),
                )
            ),
            "total_pixels": int(
                evaluation_config.get(
                    "direct_total_pixels",
                    model_video.get("total_pixels", 4096 * 32 * 32),
                )
            ),
            "fps": float(
                evaluation_config.get(
                    "direct_fps",
                    model_video.get("fps", self.coarse_config.sample_fps),
                )
            ),
            "max_frames": self.direct_frames,
        }
        self.direct_subtitle_max_chars = int(
            evaluation_config.get("direct_subtitle_max_chars", 8_000)
        )
        self.direct_subtitle_segments = int(
            evaluation_config.get("direct_subtitle_segments", self.direct_frames)
        )
        if self.direct_video_options["min_pixels"] > self.direct_video_options["max_pixels"]:
            raise ValueError("evaluation direct_min_pixels exceeds direct_max_pixels")
        if self.direct_subtitle_max_chars < 0 or self.direct_subtitle_segments < 1:
            raise ValueError("invalid direct subtitle sampling configuration")
        self.search_agent: CoarseToFineVideoAgent | ActiveTreeVideoAgent | None = None
        if strategy == "coarse_to_fine":
            self.search_agent = CoarseToFineVideoAgent(
                self.model,
                config=self.coarse_config,
            )
        elif strategy == "active_tree":
            self.search_agent = ActiveTreeVideoAgent(
                self.model,
                config=self.active_config,
            )
        self.direct_cache = (
            VideoEvidenceCache(
                self.coarse_config.cache_dir,
                sample_fps=self.coarse_config.sample_fps,
                max_side=self.coarse_config.cache_max_side,
                jpeg_quality=self.coarse_config.cache_jpeg_quality,
                lru_size=self.coarse_config.cache_lru_size,
            )
            if strategy == "direct" and not native_video_decode
            else None
        )
        self._loaded = False

    def load(self) -> None:
        if self._loaded:
            return
        if self.search_agent is not None:
            self.search_agent.load()
        else:
            self.model.load()
        if self.replay_dir is not None:
            self.replay_dir.mkdir(parents=True, exist_ok=True)
        self._loaded = True

    def close(self) -> None:
        if not self._loaded:
            return
        if self.search_agent is not None:
            self.search_agent.unload()
        else:
            self.model.unload()
        self._loaded = False

    def __enter__(self) -> Self:
        self.load()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def evaluate(self, question: VideoMMEQuestion) -> dict[str, Any]:
        try:
            return self._evaluate_question(question)
        finally:
            self.release_item_resources()

    def release_item_resources(self) -> None:
        """Release per-question tensors while keeping model weights resident."""

        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:  # pragma: no cover - torch is an inference dependency
            return

    def _evaluate_question(self, question: VideoMMEQuestion) -> dict[str, Any]:
        if not self._loaded:
            raise RuntimeError("Evaluation session is not loaded")
        video_path = question.video_path(self.video_dir)
        subtitle_path = question.subtitle_path(self.subtitle_dir)
        if not video_path.is_file():
            raise FileNotFoundError(f"Missing Video-MME video: {video_path}")

        started = time.perf_counter()
        if self.search_agent is not None:
            result = self.search_agent.generate(
                [{"role": "user", "content": question.question}],
                videos=[str(video_path)],
                choices=question.options,
                subtitle_path=(
                    str(subtitle_path)
                    if self.with_subtitles and subtitle_path.is_file()
                    else None
                ),
            )
        else:
            result = self._evaluate_direct(question, video_path, subtitle_path)

        result.metadata["exposures"] = normalize_exposures(
            self.strategy,
            result.metadata,
        )
        prediction = MultipleChoiceAdapter(question.options).normalize(result.text)
        record: dict[str, Any] = {
            **question.to_dict(),
            "strategy": self.strategy,
            "with_subtitles": self.with_subtitles,
            "prediction": prediction,
            "correct": prediction == question.answer,
            "wall_seconds": time.perf_counter() - started,
            "model_output": result.text,
            "metadata": result.metadata,
        }
        if self.strategy == "active_tree" and self.replay_dir is not None:
            replay_path = render_trace_html(
                result.metadata["active_tree"],
                self.replay_dir / f"{question.question_id}.html",
            )
            record["replay_path"] = str(replay_path)
        return record

    def _evaluate_direct(
        self,
        question: VideoMMEQuestion,
        video_path: Path,
        subtitle_path: Path,
    ) -> Any:
        adapter = MultipleChoiceAdapter(question.options)

        direct_preprocessing: dict[str, Any]
        duration_seconds = 0.0
        if self.direct_cache is not None:
            cached = self.direct_cache.prepare(str(video_path))
            duration_seconds = cached.duration_seconds
            direct_frames = cached.uniform_frames(self.direct_frames)
            video_source: str | list[str] = [frame.path for frame in direct_frames]
            direct_preprocessing = {
                "reader": "cached_frames",
                "frame_count": len(direct_frames),
                "frames": [frame.to_dict() for frame in direct_frames],
                "cache": cached.to_dict(),
                "video_options": dict(self.direct_video_options),
            }
        else:
            video_source = str(video_path)
            direct_preprocessing = {"reader": "qwen_vl_utils_native"}
        subtitle_text = ""
        if self.with_subtitles and subtitle_path.is_file():
            track = SubtitleTrack.from_srt(subtitle_path)
            if duration_seconds <= 0 and track.cues:
                duration_seconds = max(cue.end_seconds for cue in track.cues)
            subtitle_text = _sample_direct_subtitles(
                track,
                duration_seconds=duration_seconds,
                max_chars=self.direct_subtitle_max_chars,
                segments=self.direct_subtitle_segments,
            )
        direct_preprocessing["subtitle_sampling"] = {
            "method": "uniform_time_segments",
            "max_chars": self.direct_subtitle_max_chars,
            "segments": self.direct_subtitle_segments,
            "actual_chars": len(subtitle_text),
        }
        direct_preprocessing["subtitle_intervals"] = [
            {
                "start_seconds": item.start_seconds,
                "end_seconds": item.end_seconds,
            }
            for item in subtitle_exposures(subtitle_text, stage="direct_answer")
        ]
        prompt = build_direct_prompt(
            question.question,
            adapter,
            subtitles=subtitle_text,
        )
        if self.direct_cache is not None:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "video",
                            "video": video_source,
                            **self.direct_video_options,
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            output = self.model.generate(messages)
        else:
            output = self.model.generate(
                [{"role": "user", "content": prompt}],
                videos=[video_source],
            )
        output.metadata["direct_preprocessing"] = direct_preprocessing
        output.metadata["stop_reason"] = "single_pass_complete"
        return output
