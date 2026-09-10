"""Document defaults; budgets are enforced by the controller, never the model."""

from dataclasses import asdict, dataclass, field, fields

from qwen3vl_agent.p01.config import P01Config

from .types import ProtocolError, finite

MODEL = "Qwen/Qwen3-VL-8B-Instruct"
REVISION = "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"


@dataclass(frozen=True)
class R6Config:
    media: P01Config = field(
        default_factory=lambda: P01Config(
            cache_dir=".cache/qwen3vl_agent/r6",
            normal_total_pixels=8192 * 1024,
            normal_min_pixels=32 * 1024,
            image_min_pixels=32 * 1024,
        )
    )
    model_revision: str = REVISION
    initial_overview_frames: int = 32
    core_window_seconds: float = 8.0
    context_seconds_each_side: float = 2.0
    fps_default: float = 2.0
    fps_for_fast_events: float = 4.0
    max_frames_per_call: int = 48
    crop_frames_per_request: int = 2
    max_relevant_facts_per_call: int = 10
    max_total_input_tokens_per_call: int = 16384
    max_visual_tokens_per_call: int = 8192
    max_new_tokens_compiler: int = 1024
    max_new_tokens_observer: int = 1024
    max_new_tokens_relation_checker: int = 1536
    max_new_tokens_verifier: int = 1024
    max_model_calls_total: int = 16
    reserve_model_calls_for_verification: int = 2
    max_refinement_rounds: int = 4
    schema_repair_attempts_per_call: int = 1
    observer_options: str = "neutral"
    refinement_policy: str = "targeted"
    relation_verification: bool = True
    direct_channel: bool = True
    audio_provider: str = "disabled"

    @classmethod
    def from_mapping(cls, value=None):
        if isinstance(value, cls):
            result = value
        else:
            data = dict(value or {})
            if set(data) - {f.name for f in fields(cls)}:
                raise ProtocolError("unknown R6 configuration fields")
            if isinstance(data.get("media"), dict):
                data["media"] = P01Config.from_mapping(data["media"])
            result = cls(**data)
        result.validate()
        return result

    def validate(self):
        self.media.validate()
        for f in fields(self):
            value = getattr(self, f.name)
            if isinstance(f.default, bool):
                if not isinstance(value, bool):
                    raise ProtocolError(f"{f.name} requires bool")
            elif isinstance(f.default, (int, float)):
                zero_ok = f.name in {
                    "context_seconds_each_side",
                    "max_refinement_rounds",
                    "schema_repair_attempts_per_call",
                }
                if not finite(value) or value < 0 or (value == 0 and not zero_ok):
                    raise ProtocolError(f"invalid {f.name}")
                if isinstance(f.default, int) and not isinstance(value, int):
                    raise ProtocolError(f"{f.name} requires integer")
        if self.model_revision != REVISION or self.audio_provider != "disabled":
            raise ProtocolError("R6 v1 uses pinned Qwen3-VL; real audio is not enabled")
        if self.observer_options not in {"neutral", "question_only", "all"}:
            raise ProtocolError("invalid observer_options")
        if self.refinement_policy not in {"targeted", "uniform"}:
            raise ProtocolError("invalid refinement_policy")
        if self.reserve_model_calls_for_verification != 2 or self.max_model_calls_total < 4:
            raise ProtocolError("reserve two verification calls; total must be at least four")
        if not self.initial_overview_frames <= self.max_frames_per_call <= 48:
            raise ProtocolError("overview/frame limit incompatible")
        if self.crop_frames_per_request > 2 or self.max_relevant_facts_per_call > 10:
            raise ProtocolError("detail/fact cap exceeds v1 contract")
        if self.schema_repair_attempts_per_call > 1:
            raise ProtocolError("at most one schema repair per logical call")

    def to_dict(self):
        return asdict(self)
