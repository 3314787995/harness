"""Public R9 contracts. Public requests contain no benchmark answers."""

import hashlib
import itertools
import json
import math
from dataclasses import asdict, dataclass, field
from typing import TypedDict

from qwen3vl_agent.r8.types import Choice, normalize_choices

VERSION = "r9-spatial-query/1.0"
OPERATIONS = (
    "bearing",
    "heading_delta",
    "nearest_object",
    "absolute_distance",
    "max_extent",
    "floor_area",
    "fill_turns",
    "next_action",
    "compare_paths",
    "viewpoint_relation",
)
HELPERS = ("event_select", "add", "subtract", "compare", "unit_convert", "collect")
MODES = ("B0", "B1", "B2", "B3", "B4")
STATUSES = ("evidence_supported", "estimated", "unresolved", "invalid_input")


class QuestionSpec(TypedDict):
    operation: str
    entities: list[dict]
    time: dict
    query_frame: dict
    direction_rule: dict
    measurement: dict
    route: dict | None
    required_capabilities: list[str]
    source_spans: list[dict]
    nodes: list[dict]
    output_node: str | None


class Observation(TypedDict):
    id: str
    frame_id: str
    source_frame_id: str
    timestamp_s: float
    shot_id: str
    entity_candidate: str
    observation_statement: str
    coordinate_space: str
    bbox_xyxy: list[float] | None


class EntityLinks(TypedDict):
    entity_id: str
    candidate_ids: list[str]
    status: str
    basis: str
    source_observation_ids: list[str]
    valid_time: list[float]


def plain(value):
    return json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))


def digest(value):
    return hashlib.sha256(
        json.dumps(plain(value), sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def finite(v):
    return not isinstance(v, bool) and isinstance(v, (float, int)) and math.isfinite(v)


class ProtocolError(ValueError):
    pass


class BudgetExhausted(RuntimeError):
    pass


class MissingCapability(ValueError):
    def __init__(self, kind, detail, *, entities=(), record_ids=()):
        self.gap = Gap(kind, detail, list(entities), list(record_ids))
        super().__init__(detail)


def span(v):
    if not isinstance(v, (list, tuple)) or len(v) != 2 or not all(finite(t) for t in v):
        raise ProtocolError("time interval requires two finite values")
    if not 0 <= v[0] < v[1]:
        raise ProtocolError("time interval requires 0 <= start < end")
    return tuple(map(float, v))


@dataclass(frozen=True)
class R9Request:
    video_path: str
    question: str
    choices: tuple[Choice, ...] = ()
    request_id: str = "r9-request"
    video_id: str = "video"
    allowed_scope: tuple[float, float] | None = None
    allowed_time_intervals: tuple[tuple[float, float], ...] = ()
    query_scope: tuple[float, float] | None = None
    query_time: float | None = None
    observation_cutoff: float | None = None
    output_protocol: str = "auto"
    output_unit: str | None = None
    force_answer: bool = True
    mode: str = "B4"
    comparison: str = "fixed_total_budget"
    fixed_frames: tuple[float, ...] = ()
    max_model_calls: int | None = None
    max_unique_source_frames: int | None = None
    max_visual_exposures: int | None = None
    max_seconds: float | None = None
    checkpoint_path: str | None = None
    resume: bool = False

    def __post_init__(self):
        for name in ("video_path", "question", "request_id", "video_id"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise ProtocolError(f"{name} required")
        object.__setattr__(self, "choices", normalize_choices(self.choices))
        intervals = tuple(sorted(span(v) for v in self.allowed_time_intervals))
        if self.allowed_scope is not None:
            if intervals:
                raise ProtocolError("allowed_scope and allowed_time_intervals are exclusive")
            object.__setattr__(self, "allowed_scope", span(self.allowed_scope))
        if any(a[1] > b[0] for a, b in itertools.pairwise(intervals)):
            raise ProtocolError("allowed intervals overlap")
        object.__setattr__(self, "allowed_time_intervals", intervals)
        if self.query_scope is not None:
            object.__setattr__(self, "query_scope", span(self.query_scope))
        for key in ("query_time", "observation_cutoff"):
            v = getattr(self, key)
            if v is not None and (not finite(v) or v < 0):
                raise ProtocolError(f"invalid {key}")
        if (
            self.query_scope
            and self.query_time is not None
            and not self.query_scope[0] <= self.query_time <= self.query_scope[1]
        ):
            raise ProtocolError("query time exceeds query scope")
        if self.mode not in MODES:
            raise ProtocolError("R9 supports B0-B4; B5 geometry is not enabled")
        if self.comparison not in {"fixed_evidence", "fixed_total_budget"}:
            raise ProtocolError("unknown comparison mode")
        times = tuple(self.fixed_frames)
        if any(not finite(t) or t < 0 for t in times) or list(times) != sorted(set(times)):
            raise ProtocolError("fixed frames require sorted distinct nonnegative timestamps")
        object.__setattr__(self, "fixed_frames", times)
        if self.comparison == "fixed_evidence" and len(times) > 32:
            raise ProtocolError(
                "fixed-evidence comparisons require at most 32 source timestamps for the single-call baselines"
            )
        if len(intervals) > 32:
            raise ProtocolError("at most 32 disjoint allowed intervals are supported")
        if self.comparison == "fixed_evidence" and not times and self.mode != "B0":
            raise ProtocolError("fixed_evidence requires explicit fixed_frames")
        if self.output_protocol not in {"auto", "multiple_choice", "numeric", "free_text"}:
            raise ProtocolError("unknown output protocol")
        if self.output_protocol == "multiple_choice" and not self.choices:
            raise ProtocolError("multiple choice requires original options")
        if self.output_protocol == "numeric" and self.choices:
            raise ProtocolError("numeric protocol cannot contain options")
        for name in (
            "max_model_calls",
            "max_unique_source_frames",
            "max_visual_exposures",
            "max_seconds",
        ):
            v = getattr(self, name)
            if v is not None and (not finite(v) or v <= 0):
                raise ProtocolError(f"invalid {name}")
            if v is not None and name != "max_seconds" and not isinstance(v, int):
                raise ProtocolError(f"{name} requires an integer")
        if not isinstance(self.force_answer, bool) or not isinstance(self.resume, bool):
            raise ProtocolError("force_answer/resume require booleans")

    @property
    def protocol(self):
        return (
            ("multiple_choice" if self.choices else "numeric")
            if self.output_protocol == "auto"
            else self.output_protocol
        )


@dataclass(frozen=True)
class InputContract:
    allowed_time_intervals: tuple[tuple[float, float], ...]

    @classmethod
    def resolve(cls, request, duration):
        if not finite(duration) or duration <= 0:
            raise ProtocolError("video duration unavailable")
        values = request.allowed_time_intervals or (request.allowed_scope or (0.0, duration),)
        cutoff = min(
            duration,
            request.observation_cutoff if request.observation_cutoff is not None else duration,
        )
        if any(a >= duration or b > duration + 1e-6 for a, b in values):
            raise ProtocolError("requested scope exceeds video duration")
        allowed = tuple((a, min(b, cutoff)) for a, b in values if a < min(b, cutoff))
        if not allowed:
            raise ProtocolError("no permitted video interval")
        result = cls(allowed)
        if request.query_scope and not result.permits_span(request.query_scope):
            raise ProtocolError("query scope exceeds allowed media")
        if request.query_time is not None and not result.permits(request.query_time):
            raise ProtocolError("query time exceeds allowed media")
        if any(not result.permits(t) for t in request.fixed_frames):
            raise ProtocolError("fixed frame outside permission")
        return result

    def permits(self, t):
        return finite(t) and any(a <= t <= b for a, b in self.allowed_time_intervals)

    def permits_span(self, v):
        return (
            len(v) == 2
            and all(finite(t) for t in v)
            and any(a <= v[0] <= v[1] <= b for a, b in self.allowed_time_intervals)
        )

    def intersect(self, v):
        return [
            (max(a, v[0]), min(b, v[1]))
            for a, b in self.allowed_time_intervals
            if max(a, v[0]) < min(b, v[1])
        ]

    @property
    def fingerprint(self):
        return digest(asdict(self))


@dataclass
class Gap:
    kind: str
    detail: str
    entities: list[str] = field(default_factory=list)
    record_ids: list[str] = field(default_factory=list)
    time_interval: list[float] | None = None
    frame_id: str | None = None
    bbox: list[float] | None = None


@dataclass
class QueryResult:
    status: str = "unresolved"
    value: object = None
    unit: str | None = None
    source_ids: list[str] = field(default_factory=list)
    record_ids: list[str] = field(default_factory=list)
    gaps: list[Gap] = field(default_factory=list)
    derivation: list[dict] = field(default_factory=list)
    scale_source: list[str] = field(default_factory=list)


@dataclass
class Verification:
    can_stop: bool = False
    hard_checks_passed: bool = False
    visual_checks: list[dict] = field(default_factory=list)
    gaps: list[Gap] = field(default_factory=list)


@dataclass
class R9Result:
    text: str
    prediction: object
    status: str
    forced_answer: bool
    semantic_answer: object
    unit: str | None
    unresolved_reasons: list[str]
    source_ids: list[str]
    scale_source: list[str]
    trace: dict

    def to_dict(self):
        return plain(asdict(self))
