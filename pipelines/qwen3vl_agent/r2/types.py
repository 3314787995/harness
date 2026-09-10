"""Public request/result types and the immutable media access boundary."""

from __future__ import annotations

import itertools
import json
import math
from dataclasses import asdict, dataclass, field, fields
from typing import Any

from qwen3vl_agent.r1.types import R1Choice, normalize_choices

VERSION = "r2-entity-time/1.5"
OPERATIONS = (
    "endpoint_delta",
    "state_sequence",
    "relation_transition",
    "motion_condition_filter",
    "direction_sequence",
    "path_shape",
    "rotation_pattern",
    "motion_property_trend",
    "identity_at_time",
    "periodic_continuation",
)


class ProtocolError(ValueError):
    pass


class BudgetExhausted(RuntimeError):
    pass


def finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def interval(value: Any) -> tuple[float, float]:
    if hasattr(value, "start_seconds"):
        value = (value.start_seconds, value.end_seconds)
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError("interval requires [start, end]")
    if not all(finite(v) for v in value) or not 0 <= value[0] < value[1]:
        raise ValueError("interval requires finite 0 <= start < end")
    return tuple(float(v) for v in value)


@dataclass(frozen=True)
class R2Budget:
    max_model_calls: int = 128
    max_frame_exposures: int = 4096
    max_media_pixels: int = 1073741824
    max_visual_tokens: int = 1048576
    max_text_chars_per_call: int = 120000
    terminal_call_reserve: int = 4

    def __post_init__(self):
        for key, value in asdict(self).items():
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"invalid budget {key}")
        if self.terminal_call_reserve < 2 or self.max_model_calls < 4:
            raise ValueError("R2 requires at least two terminal slots and four total calls")

    def capped(self, other: R2Budget) -> R2Budget:
        return R2Budget(
            **{f.name: min(getattr(self, f.name), getattr(other, f.name)) for f in fields(self)}
        )


@dataclass(frozen=True)
class R2Request:
    video_path: str
    question: str
    choices: tuple[R1Choice, ...] = ()
    request_id: str = "r2-request"
    video_id: str = "video"
    group_id: str | None = None
    allowed_scope: tuple[float, float] | None = None
    allowed_time_intervals: tuple[tuple[float, float], ...] = ()
    query_scope: tuple[float, float] | str | None = None
    query_time: float | None = None
    observation_cutoff: float | None = None
    protocol_id: str = "full_video"
    available_modalities: tuple[str, ...] = ("video", "screen_text")
    subtitle_path: str | None = None
    asr_path: str | None = None
    execution_subtype: str | None = None
    output_protocol: str = "auto"
    budget: R2Budget = field(default_factory=R2Budget)
    checkpoint_path: str | None = None
    resume: bool = False

    def __post_init__(self):
        for name in ("video_path", "question", "request_id", "video_id", "protocol_id"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise ValueError(f"{name} is required")
        choices = self.choices
        if isinstance(choices, dict):
            choices = [{"label": k, "text": v} for k, v in choices.items()]
        object.__setattr__(self, "choices", normalize_choices(choices))
        if self.allowed_scope is not None:
            object.__setattr__(self, "allowed_scope", interval(self.allowed_scope))
        spans = tuple(interval(s) for s in self.allowed_time_intervals)
        if spans and self.allowed_scope is not None:
            raise ValueError("use allowed_scope or allowed_time_intervals, not both")
        if any(a[1] > b[0] for a, b in itertools.pairwise(spans)):
            raise ValueError("allowed intervals must be sorted and disjoint")
        object.__setattr__(self, "allowed_time_intervals", spans)
        if self.query_scope is not None and not isinstance(self.query_scope, str):
            object.__setattr__(self, "query_scope", interval(self.query_scope))
        for name in ("query_time", "observation_cutoff"):
            v = getattr(self, name)
            if v is not None and (not finite(v) or v < 0):
                raise ValueError(f"invalid {name}")
        if self.protocol_id == "strict_prefix" and self.observation_cutoff is None:
            raise ValueError("strict_prefix requires an external observation_cutoff")
        modalities = tuple(self.available_modalities)
        if "video" not in modalities or set(modalities) - {
            "video",
            "screen_text",
            "subtitle",
            "asr",
        }:
            raise ValueError(
                "R2 requires video; supported modalities are video/screen_text/subtitle/asr"
            )
        object.__setattr__(self, "available_modalities", modalities)
        for kind in ("subtitle", "asr"):
            if getattr(self, kind + "_path") and kind not in modalities:
                raise ValueError(f"{kind}_path requires explicit {kind} permission")
        if self.execution_subtype is not None and self.execution_subtype not in OPERATIONS:
            raise ValueError("unknown R2 operation")
        if isinstance(self.budget, dict):
            object.__setattr__(self, "budget", R2Budget(**self.budget))
        if not isinstance(self.budget, R2Budget):
            raise TypeError("budget must be R2Budget")
        if self.output_protocol not in {"auto", "multiple_choice", "free_text"}:
            raise ValueError("R2 supports auto, multiple_choice and free_text")
        if self.output_protocol == "multiple_choice" and not self.choices:
            raise ValueError("multiple_choice requires choices")
        if self.choices and self.output_protocol == "free_text":
            raise ValueError("choices conflict with free_text")
        if self.resume and not self.checkpoint_path:
            raise ValueError("resume requires checkpoint_path")


@dataclass(frozen=True)
class InputContract:
    allowed_time_intervals: tuple[tuple[float, float], ...]
    available_modalities: tuple[str, ...]
    query_time: float | None
    observation_cutoff: float | None
    protocol_id: str

    @classmethod
    def resolve(cls, request: R2Request, duration: float) -> InputContract:
        if not finite(duration) or duration <= 0:
            raise ValueError("video duration must be positive")
        spans = request.allowed_time_intervals or (request.allowed_scope or (0.0, duration),)
        if any(b > duration + 1e-6 for _, b in spans):
            raise ValueError("allowed interval exceeds source duration")
        cutoff = duration if request.observation_cutoff is None else request.observation_cutoff
        accessible = tuple((a, min(b, duration, cutoff)) for a, b in spans if a < min(b, cutoff))
        if not accessible:
            raise ValueError("no permitted media remains")
        # The queried time is not an evidence permission: a future query may be after cutoff.
        return cls(
            accessible,
            request.available_modalities,
            request.query_time,
            request.observation_cutoff,
            request.protocol_id,
        )

    def permits(self, timestamp: float) -> bool:
        return finite(timestamp) and any(
            a <= timestamp <= b for a, b in self.allowed_time_intervals
        )

    def permits_span(self, span) -> bool:
        return any(a <= span[0] <= span[1] <= b for a, b in self.allowed_time_intervals)

    def intersect(self, span) -> list[tuple[float, float]]:
        return [
            (max(a, span[0]), min(b, span[1]))
            for a, b in self.allowed_time_intervals
            if max(a, span[0]) < min(b, span[1])
        ]


@dataclass
class R2Result:
    prediction: str | None
    completion_state: str
    support_level: str
    answer_basis: str = "visual_estimates_and_program_reduction"
    value_state: dict = field(default_factory=dict)
    state_store: dict = field(default_factory=dict)
    coverage_manifest: list = field(default_factory=list)
    unresolved_items: list = field(default_factory=list)
    evidence_refs: list = field(default_factory=list)
    option_assessments: list = field(default_factory=list)
    resources: dict = field(default_factory=dict)
    trace: dict = field(default_factory=dict)

    def to_dict(self):
        # The public wire representation is identical before and after JSONL resume.
        return json.loads(
            json.dumps(
                {"pipeline_id": "R2", "version": VERSION, **asdict(self)},
                ensure_ascii=False,
                allow_nan=False,
            )
        )

    @property
    def text(self):
        return self.prediction or ""
