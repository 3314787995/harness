"""v5 engineering defaults; no claim of model-optimal settings."""
from dataclasses import dataclass, field, fields
import math
from collections.abc import Mapping
from qwen3vl_agent.p01.config import P01Config
from .types import R3Budget

@dataclass(frozen=True)
class R3Config:
    media: P01Config = field(default_factory=lambda: P01Config(cache_dir=".cache/qwen3vl_agent/r3-v54", normal_total_pixels=8192 * 1024))
    budget: R3Budget = field(default_factory=R3Budget)
    short_core_sec: float = 4.0
    short_fps: float = 8.0
    context_sec: float = 1.0
    long_core_sec: float = 32.0
    long_fps: float = 1.0
    tail_sec: float = 2.0
    neighbor_sec: float = 6.0
    neighbor_fps: float = 8.0
    refinement_fps: float = 16.0
    max_refinements: int = 12
    max_frames_per_call: int = 64
    visual_token_target: int = 8192
    visual_token_limit: int = 12288
    query_tokens: int = 512
    observer_tokens: int = 768
    refinement_tokens: int = 768
    final_tokens: int = 512
    seed: int = 0
    max_provider_segments: int = 64
    max_provider_text_chars: int = 16000

    @classmethod
    def from_mapping(cls, data=None):
        values = dict(data or {})
        unknown = set(values) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"R3 v5 configuration requires migration; unknown fields: {sorted(unknown)}")
        if isinstance(values.get("media"), Mapping):
            values["media"] = P01Config.from_mapping({"normal_total_pixels":8192*1024, **values["media"]})
        if isinstance(values.get("budget"), Mapping):
            values["budget"] = R3Budget(**values["budget"])
        result = cls(**values)
        result.validate()
        return result

    def validate(self):
        self.media.validate()
        for f in fields(self):
            v = getattr(self, f.name)
            if f.name in {"media", "budget"}:
                continue
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < (0 if f.name in {"seed", "max_refinements"} else 1):
                raise ValueError(f"invalid R3 v5 {f.name}")
            if f.type is int and not isinstance(v, int):
                raise ValueError(f"{f.name} must be an integer")
        if self.max_frames_per_call > min(64, self.media.caption_refine_max_frames):
            raise ValueError("R3 frame cap exceeds 64 or shared preparer cap")
        if self.visual_token_target > self.visual_token_limit or self.visual_token_limit > 12288:
            raise ValueError("invalid visual token caps")
        if self.max_refinements > 12:
            raise ValueError("at most twelve critical rereads per query")
