"""Public R7 contracts. Media permissions are supplied by the caller, never by Qwen."""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import re
from dataclasses import asdict, dataclass, field

VERSION = "r7-world-execution/1.0"
MECHANISMS = ("S1", "S2", "S3", "S4", "S5")
MODES = ("B0", "B1", "B2", "B3", "B4", "B4-uniform")


def plain(value):
    return json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False, default=str))


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, default=str).encode()
    ).hexdigest()


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def span(value):
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise ValueError("time interval requires [start, end]")
    if not all(finite(v) for v in value) or not 0 <= value[0] < value[1]:
        raise ValueError("time interval requires finite 0 <= start < end")
    return tuple(map(float, value))


class ProtocolError(ValueError):
    pass


class ModelFailure(RuntimeError):
    pass


class BudgetExhausted(RuntimeError):
    pass


@dataclass(frozen=True)
class Choice:
    label: str
    text: str


def normalize_choices(values):
    if isinstance(values, dict):
        values = [{"label": k, "text": v} for k, v in values.items()]
    out = []
    for i, value in enumerate(values):
        if isinstance(value, str):
            match = re.match(r"^\s*([A-Za-z0-9]+)[.)：:]\s+(.+)$", value, re.DOTALL)
            label, text = match.groups() if match else (chr(65 + i), value)
        elif hasattr(value, "label"):
            label, text = value.label, value.text
        else:
            label, text = value["label"], value["text"]
        if (
            not isinstance(label, str)
            or not label.strip()
            or not isinstance(text, str)
            or not text.strip()
        ):
            raise ValueError("nonempty option label/text required")
        out.append(Choice(label, text))
    if not out or len({x.label for x in out}) != len(out):
        raise ValueError("at least one option with unique labels required")
    return tuple(out)


@dataclass(frozen=True)
class R7Request:
    video_path: str
    question: str
    choices: tuple[Choice, ...]
    request_id: str = "r7-request"
    video_id: str = "video"
    group_id: str | None = None
    allowed_scope: tuple[float, float] | None = None
    allowed_time_intervals: tuple[tuple[float, float], ...] = ()
    observation_cutoff: float | None = None
    protocol_id: str = "full_video"
    protocol_source: str = "caller"
    target_visibility: str = "unresolved"
    available_modalities: tuple[str, ...] = ("video", "screen_text")
    subtitle_path: str | None = None
    query_time: float | None = None
    query_scope: tuple[float, float] | None = None
    execution_subtype: str | None = None
    output_protocol: str = "multiple_choice"
    max_model_calls: int | None = None
    mode: str = "B4"
    facts_input: str | None = None
    checkpoint_path: str | None = None
    resume: bool = False

    def __post_init__(self):
        for name in (
            "video_path",
            "question",
            "request_id",
            "video_id",
            "protocol_id",
            "protocol_source",
        ):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise ValueError(f"{name} is required")
        object.__setattr__(self, "choices", normalize_choices(self.choices))
        intervals = tuple(span(x) for x in self.allowed_time_intervals)
        if self.allowed_scope is not None:
            if intervals:
                raise ProtocolError("use allowed_scope OR allowed_time_intervals")
            object.__setattr__(self, "allowed_scope", span(self.allowed_scope))
        intervals = tuple(sorted(intervals))
        if any(a[1] > b[0] for a, b in itertools.pairwise(intervals)):
            raise ProtocolError("overlapping input intervals")
        object.__setattr__(self, "allowed_time_intervals", intervals)
        object.__setattr__(self, "available_modalities", tuple(self.available_modalities))
        if "video" not in self.available_modalities or set(self.available_modalities) - {
            "video",
            "screen_text",
            "subtitle",
        }:
            raise ProtocolError(
                "R7 accepts video/screen_text and explicitly permitted existing subtitles"
            )
        if bool(self.subtitle_path) != ("subtitle" in self.available_modalities):
            raise ProtocolError("subtitle path and permission must be supplied together")
        if self.protocol_id == "strict_prefix" and self.observation_cutoff is None:
            raise ProtocolError("strict_prefix requires an explicit cutoff")
        if self.observation_cutoff is not None and (
            not finite(self.observation_cutoff) or self.observation_cutoff <= 0
        ):
            raise ProtocolError("invalid cutoff")
        if self.query_time is not None and not finite(self.query_time):
            raise ProtocolError("invalid query time")
        if self.query_scope is not None:
            object.__setattr__(self, "query_scope", span(self.query_scope))
        if self.target_visibility not in {
            "unobserved_future",
            "counterfactual",
            "observed",
            "unresolved",
        }:
            raise ProtocolError("invalid visibility")
        if self.execution_subtype is not None and self.execution_subtype not in MECHANISMS:
            raise ValueError("R7 operation hint must be S1..S5")
        if self.mode not in MODES or self.output_protocol not in {"auto", "multiple_choice"}:
            raise ValueError("R7 supports B0..B4(-uniform), multiple choice")
        if self.max_model_calls is not None and (
            isinstance(self.max_model_calls, bool)
            or not isinstance(self.max_model_calls, int)
            or self.max_model_calls < 1
        ):
            raise ValueError("max_model_calls must be positive")
        if self.facts_input and self.mode not in {"B2", "B3", "B4", "B4-uniform"}:
            raise ValueError("fact replay requires B2..B4")
        if self.resume and not self.checkpoint_path:
            raise ValueError("resume requires a checkpoint")


@dataclass(frozen=True)
class InputContract:
    allowed_time_intervals: tuple[tuple[float, float], ...]
    available_modalities: tuple[str, ...]
    protocol_id: str
    protocol_source: str
    observation_cutoff: float | None

    @classmethod
    def resolve(cls, request, duration):
        if not finite(duration) or duration <= 0:
            raise ProtocolError("invalid video duration")
        intervals = request.allowed_time_intervals or (
            (request.allowed_scope,) if request.allowed_scope else ((0.0, duration),)
        )
        if any(b > duration + 0.1 for _, b in intervals):
            raise ProtocolError("declared observation interval exceeds media duration")
        end = min(duration, request.observation_cutoff) if request.observation_cutoff else duration
        allowed = tuple((a, min(b, end)) for a, b in intervals if a < min(b, end))
        if not allowed:
            raise ProtocolError("no permitted interval")
        return cls(
            allowed,
            request.available_modalities,
            request.protocol_id,
            request.protocol_source,
            request.observation_cutoff,
        )

    def permits(self, time):
        return finite(time) and any(a <= time <= b for a, b in self.allowed_time_intervals)

    def permits_span(self, value):
        return (
            len(value) == 2
            and all(finite(v) for v in value)
            and any(a <= value[0] <= value[1] <= b for a, b in self.allowed_time_intervals)
        )

    def intersect(self, value):
        return [
            (max(a, value[0]), min(b, value[1]))
            for a, b in self.allowed_time_intervals
            if max(a, value[0]) < min(b, value[1])
        ]

    @property
    def fingerprint(self):
        return digest(asdict(self))


@dataclass
class R7Result:
    prediction: str | None
    completion_state: str
    support_level: str
    stop_reason: str
    task_spec: dict = field(default_factory=dict)
    candidates: list = field(default_factory=list)
    facts: dict = field(default_factory=dict)
    scenarios: list = field(default_factory=list)
    assessments: list = field(default_factory=list)
    gaps: list = field(default_factory=list)
    coverage: list = field(default_factory=list)
    resources: dict = field(default_factory=dict)
    trace: dict = field(default_factory=dict)

    @property
    def text(self):
        return self.prediction or ""

    def to_dict(self):
        return plain({"pipeline_id": "R7", "version": VERSION, **asdict(self)})
