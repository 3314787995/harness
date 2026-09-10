from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any

from qwen3vl_agent.p01.config import P01Config
from qwen3vl_agent.r4.types import R4Budget, finite


@dataclass(frozen=True)
class R4Config:
    media: P01Config = field(default_factory=lambda: P01Config(cache_dir=".cache/qwen3vl_agent/r4-v5_7",
        normal_max_pixels=393216, normal_total_pixels=8388608, detail_max_pixels=1048576))
    budget: R4Budget = field(default_factory=R4Budget)
    core_frames: int = 16
    context_frames: int = 2
    max_frames_per_call: int = 24
    static_fps: float = 2.0
    motion_fps: float = 2.0
    refinement_fps: tuple[float, ...] = (6.0, 12.0)
    max_refinement_rounds: int = 2
    max_split_depth: int = 6
    min_split_sec: float = 0.25
    crop_overlap: float = 0.1
    identity_candidates: int = 3
    compiler_tokens: int = 512
    observer_tokens: int = 1024
    final_tokens: int = 512  # Reserved best-effort answer; total question budget is unchanged.
    review_tokens: int = 768
    max_observations_per_call: int = 12
    max_focused_calls: int = 4
    max_qualification_calls: int = 2
    max_identity_calls: int = 4
    max_recovery_calls: int = 2
    short_duration_sec: float = 120.0
    short_frame_exposures: int = 640
    local_max_calls: int = 7
    max_provider_segments: int = 64
    max_provider_text_chars: int = 16000
    text_window_sec: float = 60.0

    @classmethod
    def from_mapping(cls, data: dict[str, Any] | None) -> R4Config:
        values = dict(data or {})
        if set(values) - {f.name for f in fields(cls)}:
            raise ValueError("unknown R4 configuration field")
        if isinstance(values.get("media"), dict):
            values["media"] = P01Config.from_mapping(values["media"])
        if isinstance(values.get("budget"), dict):
            values["budget"] = R4Budget(**values["budget"])
        if "refinement_fps" in values:
            values["refinement_fps"] = tuple(values["refinement_fps"])
        result = cls(**values)
        result.validate()
        return result

    def validate(self) -> None:
        self.media.validate()
        for f in fields(self):
            value = getattr(self, f.name)
            if f.type == "int" and (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < (0 if f.name in {"max_refinement_rounds", "context_frames"} else 1)
            ):
                raise ValueError(f"invalid R4 {f.name}")
        if self.core_frames + 2 * self.context_frames > self.max_frames_per_call:
            raise ValueError("core/context exceed frame limit")
        if self.max_frames_per_call > self.media.caption_refine_max_frames:
            raise ValueError("frame limit exceeds shared media preparer")
        for value in (
            self.static_fps,
            self.motion_fps,
            self.text_window_sec,
            self.min_split_sec,
            *self.refinement_fps,
        ):
            if finite(value, minimum=0) == 0:
                raise ValueError("R4 rates and durations must be positive")
        if not 0 <= self.max_refinement_rounds <= 2:
            raise ValueError("at most two semantic refinement rounds")
        if tuple(sorted(set(self.refinement_fps))) != self.refinement_fps:
            raise ValueError("refinement rates must strictly increase")
        if self.max_refinement_rounds and not self.refinement_fps:
            raise ValueError("refinement rates required")
        if not 0 <= finite(self.crop_overlap) < 0.5:
            raise ValueError("invalid crop overlap")
