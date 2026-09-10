import math
from dataclasses import asdict, dataclass, field, fields

from qwen3vl_agent.p01.config import P01Config


@dataclass(frozen=True)
class R7Config:
    media: P01Config = field(default_factory=lambda: P01Config(cache_dir=".cache/qwen3vl_agent/r7"))
    initial_frames: int = 32
    max_frames_per_call: int = 48
    tail_focus_seconds: float = 3.0
    fine_motion_fps: float = 16.0
    window_seconds: float = 60.0
    max_refinements: int = 2
    max_latent_branches: int = 3
    short_model_calls: int = 10
    short_unique_frames: int = 128
    long_model_calls: int = 128
    long_unique_frames: int = 4096
    max_program_steps: int = 64
    max_text_chars: int = 100000
    compiler_tokens: int = 800
    candidates_tokens: int = 1600
    observer_tokens: int = 1200
    reasoner_tokens: int = 1600
    verifier_tokens: int = 800
    model_revision: str = "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"

    @classmethod
    def from_mapping(cls, value=None):
        if isinstance(value, cls):
            value.validate()
            return value
        data = dict(value or {})
        if set(data) - {f.name for f in fields(cls)}:
            raise ValueError("unknown R7 configuration field")
        if isinstance(data.get("media"), dict):
            data["media"] = P01Config.from_mapping(data["media"])
        result = cls(**data)
        result.validate()
        return result

    def validate(self):
        self.media.validate()
        for f in fields(self):
            if f.name in {"media", "model_revision"}:
                continue
            value = getattr(self, f.name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0
            ):
                raise ValueError(f"invalid R7 {f.name}")
            if value == 0 and f.name != "max_refinements":
                raise ValueError(f"invalid R7 {f.name}")
            if isinstance(f.default, int) and not isinstance(value, int):
                raise TypeError(f"integer required for {f.name}")
        if not 2 <= self.initial_frames <= self.max_frames_per_call <= 48:
            raise ValueError("initial/per-call frames must be 2..48")
        if self.max_refinements > 2 or self.max_latent_branches > 3:
            raise ValueError("at most 2 refinements / 3 latent branches per explicit scenario")

    def budget(self, contract, request):
        durations = [b - a for a, b in contract.allowed_time_intervals]
        n = sum(math.ceil(d / self.window_seconds) for d in durations)
        short = sum(durations) <= self.window_seconds
        calls = self.short_model_calls if short else min(self.long_model_calls, 2 * n + 10)
        unique = (
            self.short_unique_frames
            if short
            else min(self.long_unique_frames, self.initial_frames * n + 96)
        )
        calls = min(calls, request.max_model_calls or calls)
        return {
            "max_model_calls": calls,
            "max_unique_frames": unique,
            "max_frame_exposures": calls * self.max_frames_per_call,
            "max_media_pixels": calls * self.media.normal_total_pixels,
            "windows": n,
            "profile": "short" if short else "long",
        }

    def to_dict(self):
        return asdict(self)
