import math
from dataclasses import dataclass, field, fields

from qwen3vl_agent.p01.config import P01Config

from .types import R2Budget


@dataclass(frozen=True)
class R2Config:
    media: P01Config = field(default_factory=lambda: P01Config(cache_dir=".cache/qwen3vl_agent/r2"))
    budget: R2Budget = field(default_factory=R2Budget)
    core_sec: float = 4.0
    context_sec: float = 1.0
    fps: float = 4.0
    fast_core_sec: float = 2.0
    fast_context_sec: float = 0.5
    fast_fps: float = 8.0
    refine_core_sec: float = 1.0
    refine_context_sec: float = 0.25
    refine_fps: float = 16.0
    endpoint_frames: int = 6
    max_frames_per_call: int = 48
    max_refinements: int = 2
    max_hypotheses: int = 8
    locator_frames: int = 24
    locator_levels: int = 3
    compiler_tokens: int = 3072
    observer_tokens: int = 4096
    final_tokens: int = 3072
    max_records: int = 48
    position_tolerance: float = 0.005  # Fraction of original image diagonal.
    trend_tolerance: float = 0.10

    @classmethod
    def from_mapping(cls, value=None):
        if isinstance(value, cls):
            value.validate()
            return value
        data = dict(value or {})
        if set(data) - {f.name for f in fields(cls)}:
            raise ValueError("unknown R2 config fields")
        if isinstance(data.get("media"), dict):
            data["media"] = P01Config.from_mapping(data["media"])
        if isinstance(data.get("budget"), dict):
            data["budget"] = R2Budget(**data["budget"])
        result = cls(**data)
        result.validate()
        return result

    def validate(self):
        self.media.validate()
        for f in fields(self):
            v = getattr(self, f.name)
            if f.name in {"media", "budget"}:
                continue
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
                raise ValueError(f"invalid R2 setting {f.name}")
            if v < 0 or (v == 0 and f.name != "max_refinements"):
                raise ValueError(f"invalid R2 setting {f.name}")
            if isinstance(f.default, int) and not isinstance(v, int):
                raise TypeError(f"{f.name} must be an integer")
        if self.max_frames_per_call > 48 or self.max_frames_per_call < 8:
            raise ValueError("per-call cap must be 8..48")
        if self.max_refinements > 2 or not 4 <= self.endpoint_frames <= 8:
            raise ValueError("at most two refinements; endpoints require 4..8 frames")
        for prefix in ("", "fast_", "refine_"):
            count = (
                getattr(self, prefix + "core_sec") + 2 * getattr(self, prefix + "context_sec")
            ) * getattr(self, prefix + "fps")
            if count + 2 > self.max_frames_per_call:
                raise ValueError("window density plus shared anchors exceeds the frame cap")
