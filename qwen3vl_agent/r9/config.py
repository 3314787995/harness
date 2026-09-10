"""Frozen-model defaults for the R9 spatial controller."""

import math
from dataclasses import asdict, dataclass, field, fields

from qwen3vl_agent.p01.config import P01Config

MODEL = "Qwen/Qwen3-VL-8B-Instruct"
REVISION = "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"


@dataclass(frozen=True)
class R9Config:
    media: P01Config = field(default_factory=lambda: P01Config(cache_dir=".cache/qwen3vl_agent/r9"))
    model_revision: str = REVISION
    geometry_backend: str = "disabled"
    max_model_calls: int = 32
    terminal_call_reserve: int = 2
    initial_overview_frames: int = 32
    preferred_frames_per_call: int = 16
    max_video_frames_per_call: int = 32
    max_detail_images_per_call: int = 2
    max_unique_source_frames: int = 128
    max_visual_exposures: int = 256
    max_refinement_rounds: int = 3
    max_active_binding_hypotheses: int = 3
    working_context_target_tokens: int = 16384
    max_new_tokens_per_structured_call: int = 1200
    format_retries: int = 1
    scan_fps: float = 4.0
    max_seconds: float = 900.0
    max_query_nodes: int = 16

    @property
    def max_frames_per_call(self):
        return self.max_video_frames_per_call + self.max_detail_images_per_call

    @classmethod
    def from_mapping(cls, value=None):
        if isinstance(value, cls):
            result = value
        else:
            data = dict(value or {})
            unknown = set(data) - {f.name for f in fields(cls)}
            if unknown:
                raise ValueError(f"unknown R9 settings: {sorted(unknown)}")
            if isinstance(data.get("media"), dict):
                data["media"] = P01Config.from_mapping(data["media"])
            result = cls(**data)
        result.validate()
        return result

    def validate(self):
        self.media.validate()
        if self.model_revision != REVISION:
            raise ValueError("R9 requires the pinned Qwen3-VL-8B revision")
        if self.geometry_backend != "disabled":
            raise ValueError("frozen geometry backend / B5 is not enabled in R9 v1")
        for f in fields(self):
            if f.name in {"media", "model_revision", "geometry_backend"}:
                continue
            v = getattr(self, f.name)
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
                raise ValueError(f"invalid R9 setting {f.name}")
            if isinstance(f.default, int) and not isinstance(v, int):
                raise TypeError(f"{f.name} must be integer")
            if v < 0 or (
                v == 0
                and f.name
                not in {"format_retries", "max_refinement_rounds", "max_detail_images_per_call"}
            ):
                raise ValueError(f"invalid R9 setting {f.name}")
        if not 1 <= self.preferred_frames_per_call <= self.max_video_frames_per_call <= 32:
            raise ValueError("R9 video batches must contain at most 32 frames")
        if self.initial_overview_frames > self.max_unique_source_frames:
            raise ValueError("overview exceeds unique-frame budget")
        if self.terminal_call_reserve != 2 or self.max_model_calls < 4:
            raise ValueError("R9 reserves two calls and needs at least four calls")
        if self.max_detail_images_per_call > 2 or self.max_active_binding_hypotheses > 3:
            raise ValueError("R9 detail/hypothesis cap exceeded")
        if self.working_context_target_tokens <= self.max_new_tokens_per_structured_call:
            raise ValueError("context must leave room for input")

    def to_dict(self):
        return asdict(self)
