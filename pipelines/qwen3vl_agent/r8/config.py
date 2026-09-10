"""Frozen, explicit defaults for the training-free R8 controller."""

import math
from dataclasses import asdict, dataclass, field, fields

from qwen3vl_agent.p01.config import P01Config

MODEL = "Qwen/Qwen3-VL-8B-Instruct"
REVISION = "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"
SOLVER_VERSION = "5.1.0.0"


@dataclass(frozen=True)
class R8Config:
    media: P01Config = field(default_factory=lambda: P01Config(cache_dir=".cache/qwen3vl_agent/r8"))
    initial_frames: int = 24
    max_frames_per_call: int = 48
    max_crops: int = 3
    core_seconds: float = 6.0
    context_seconds: float = 1.0
    scan_fps: float = 4.0
    format_retries: int = 1
    semantic_recompiles: int = 2
    repair_rounds: int = 3
    max_joint_candidates: int = 8
    max_query_nodes: int = 64
    max_query_depth: int = 16
    max_numeric_digits: int = 100
    max_geometry_rules: int = 50
    solver_seconds: float = 5.0
    max_model_calls: int = 128
    max_unique_frames: int = 4096
    max_seconds: float = 900.0
    max_text_chars: int = 160000
    compiler_tokens: int = 768
    observer_tokens: int = 1024
    formalizer_tokens: int = 2048
    auditor_tokens: int = 1024
    model_revision: str = REVISION

    @classmethod
    def from_mapping(cls, value=None):
        if isinstance(value, cls):
            result = value
        else:
            data = dict(value or {})
            if set(data) - {f.name for f in fields(cls)}:
                raise ValueError("unknown R8 configuration field")
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
            ):
                raise ValueError(f"invalid {f.name}")
            if value < 0 or (
                value == 0
                and f.name
                not in {
                    "max_crops",
                    "context_seconds",
                    "format_retries",
                    "semantic_recompiles",
                    "repair_rounds",
                }
            ):
                raise ValueError(f"invalid {f.name}")
            if isinstance(f.default, int) and not isinstance(value, int):
                raise TypeError(f"{f.name} requires an integer")
        if not 2 <= self.initial_frames <= self.max_frames_per_call <= 48:
            raise ValueError("R8 requires 2 <= initial_frames <= max_frames_per_call <= 48")
        if self.model_revision != REVISION:
            raise ValueError("R8 protocol fixes the Qwen revision")

    def to_dict(self):
        return asdict(self)
