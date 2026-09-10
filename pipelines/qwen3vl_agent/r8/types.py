"""Public contracts; model outputs cannot widen video or subtitle permissions."""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import re
from dataclasses import asdict, dataclass, field

VERSION = "r8-symbolic-execution/1.0"
MODES = tuple("ABCDEFG")
STATUSES = (
    "verified_exact",
    "verified_at_option_precision",
    "observed_answer",
    "unresolved_evidence",
    "unresolved_modeling",
    "solver_unknown",
    "annotation_anomaly",
    "forced_guess",
)
ORIGINS = ("observed", "given", "definition", "derived", "hypothetical")


def plain(value):
    return json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False, default=str))


def digest(value):
    return hashlib.sha256(
        json.dumps(plain(value), sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def finite(value):
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)


def span(value):
    if not isinstance(value, (list, tuple)) or len(value) != 2 or not all(finite(v) for v in value):
        raise ProtocolError("interval requires two finite numbers")
    if not 0 <= value[0] < value[1]:
        raise ProtocolError("interval requires 0 <= start < end")
    return tuple(map(float, value))


class ProtocolError(ValueError):
    pass


class ModelingError(ValueError):
    pass


class BudgetExhausted(RuntimeError):
    pass


class ModelFailure(RuntimeError):
    pass


@dataclass(frozen=True)
class Choice:
    label: str
    text: str


def normalize_choices(values):
    if isinstance(values, dict):
        values = [Choice(k, v) for k, v in values.items()]
    output = []
    for i, value in enumerate(values or ()):
        if isinstance(value, str):
            m = re.match(r"^\s*([A-Za-z0-9]+)[.):：]\s+(.+)$", value, re.DOTALL)
            label, text = m.groups() if m else (chr(65 + i), value)
        elif isinstance(value, dict):
            if set(value) != {"label", "text"}:
                raise ProtocolError("choice requires only label/text")
            label, text = value["label"], value["text"]
        else:
            label, text = value.label, value.text
        if (
            not isinstance(label, str)
            or not label.strip()
            or not isinstance(text, str)
            or not text.strip()
        ):
            raise ProtocolError("nonempty choice label/text required")
        output.append(Choice(label, text))
    if len({c.label for c in output}) != len(output):
        raise ProtocolError("duplicate option labels")
    return tuple(output)


@dataclass(frozen=True)
class R8Request:
    video_path: str
    question: str
    choices: tuple[Choice, ...] = ()
    request_id: str = "r8-request"
    video_id: str = "video"
    group_id: str | None = None
    allowed_scope: tuple[float, float] | None = None
    allowed_time_intervals: tuple[tuple[float, float], ...] = ()
    query_scope: tuple[float, float] | None = None
    query_time: float | None = None
    observation_cutoff: float | None = None
    available_modalities: tuple[str, ...] = ("video", "screen_text")
    subtitle_path: str | None = None
    protocol_id: str = "full_video"
    protocol_source: str = "caller"
    output_protocol: str = "auto"
    require_choice: bool = False
    max_model_calls: int | None = None
    max_unique_frames: int | None = None
    max_seconds: float | None = None
    execution_subtype: str | None = None
    mode: str = "G"
    comparison: str = "end_to_end"
    fixed_frames: tuple[float, ...] = ()
    variables_input: str | None = None
    diagnostic: bool = False
    checkpoint_path: str | None = None
    resume: bool = False

    def __post_init__(self):
        for key in (
            "video_path",
            "question",
            "request_id",
            "video_id",
            "protocol_id",
            "protocol_source",
        ):
            if not isinstance(getattr(self, key), str) or not getattr(self, key).strip():
                raise ProtocolError(f"{key} required")
        object.__setattr__(self, "choices", normalize_choices(self.choices))
        intervals = tuple(sorted(span(v) for v in self.allowed_time_intervals))
        if self.allowed_scope is not None:
            if intervals:
                raise ProtocolError("allowed_scope and allowed_time_intervals are exclusive")
            object.__setattr__(self, "allowed_scope", span(self.allowed_scope))
        if any(a[1] > b[0] for a, b in itertools.pairwise(intervals)):
            raise ProtocolError("overlapping allowed intervals")
        object.__setattr__(self, "allowed_time_intervals", intervals)
        if self.query_scope is not None:
            object.__setattr__(self, "query_scope", span(self.query_scope))
        for key in ("query_time", "observation_cutoff"):
            value = getattr(self, key)
            if value is not None and (not finite(value) or value < 0):
                raise ProtocolError(f"invalid {key}")
        if self.protocol_id == "strict_prefix" and self.observation_cutoff is None:
            raise ProtocolError("strict_prefix requires cutoff")
        modalities = tuple(self.available_modalities)
        object.__setattr__(self, "available_modalities", modalities)
        if "video" not in modalities or set(modalities) - {"video", "screen_text", "subtitle"}:
            raise ProtocolError("R8 accepts video, screen_text and existing permitted subtitles")
        if bool(self.subtitle_path) != ("subtitle" in modalities):
            raise ProtocolError("subtitle permission and path must occur together")
        if self.output_protocol not in {"auto", "multiple_choice", "numeric", "free_text"}:
            raise ProtocolError("unsupported R8 output protocol")
        if (self.require_choice or self.output_protocol == "multiple_choice") and not self.choices:
            raise ProtocolError("choice protocol requires original options")
        if self.mode not in MODES or self.comparison not in {"end_to_end", "fixed_evidence"}:
            raise ProtocolError("invalid R8 experiment")
        if self.execution_subtype not in {None, "S1", "S2", "S3", "S4", "S5"}:
            raise ProtocolError("mechanism hint must be S1..S5")
        object.__setattr__(self, "fixed_frames", tuple(self.fixed_frames))
        if any(not finite(t) or t < 0 for t in self.fixed_frames):
            raise ProtocolError("invalid fixed frame times")
        if self.comparison == "fixed_evidence" and not (self.fixed_frames or self.variables_input):
            raise ProtocolError("fixed_evidence needs explicit frames or variable replay")
        if self.variables_input and self.mode == "A":
            raise ProtocolError("mode A does not accept a variable table")
        if self.resume and not self.checkpoint_path:
            raise ProtocolError("resume requires checkpoint")
        for key in ("max_model_calls", "max_unique_frames", "max_seconds"):
            value = getattr(self, key)
            if value is not None and (not finite(value) or value <= 0):
                raise ProtocolError(f"invalid {key}")
            if key != "max_seconds" and value is not None and not isinstance(value, int):
                raise ProtocolError(f"{key} must be integer")


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
            raise ProtocolError("invalid media duration")
        intervals = request.allowed_time_intervals or (
            (request.allowed_scope,) if request.allowed_scope else ((0.0, duration),)
        )
        if any(b > duration + 0.1 for a, b in intervals):
            raise ProtocolError("permission exceeds media duration")
        end = (
            min(duration, request.observation_cutoff)
            if request.observation_cutoff is not None
            else duration
        )
        allowed = tuple((a, min(b, end)) for a, b in intervals if a < min(b, end))
        if not allowed:
            raise ProtocolError("no permitted interval")
        result = cls(
            allowed,
            request.available_modalities,
            request.protocol_id,
            request.protocol_source,
            request.observation_cutoff,
        )
        if request.query_scope and not result.permits_span(request.query_scope):
            raise ProtocolError("query scope exceeds permitted evidence")
        if any(not result.permits(t) for t in request.fixed_frames):
            raise ProtocolError("fixed frames exceed permitted evidence")
        return result

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
class EntityRecord:
    id: str
    description: str
    scope: str
    snapshot: str


@dataclass
class VariableRecord:
    id: str
    version: int
    entity_id: str
    attribute: str
    role: str
    scope: str
    snapshot: str
    raw_text: str
    value: object
    unit: str
    unit_basis: str
    origin: str
    evidence_refs: list[str] = field(default_factory=list)
    question_span: str = ""
    parents: list[str] = field(default_factory=list)
    event_time: float | None = None
    content_time: str | None = None
    valid_interval: list[float] | None = None
    alternatives: list = field(default_factory=list)
    alternatives_exhaustive: bool = False
    unresolved: list[str] = field(default_factory=list)
    visual_support_status: str = "unreviewed"
    valid: bool = True

    @property
    def ref(self):
        return f"{self.id}@{self.version}"


@dataclass
class EvidenceRecord:
    id: str
    modality: str
    timestamp_seconds: float
    scope_hash: str
    source_frame_id: str | None = None
    pts: int | None = None
    time_base: list[int] | None = None
    view_box: list[float] | None = None
    parent_id: str | None = None


@dataclass
class CoverageRecord:
    core: list[float]
    context: list[float]
    timestamps: list[float]
    frame_ids: list[str]
    max_gap_seconds: float
    requested_fps: float | None
    scan_completed: bool
    resolution_met: bool
    discovery_complete: bool = False
    open_event_boundaries: list = field(default_factory=list)
    possible_replays: list = field(default_factory=list)
    unreadable_items: list = field(default_factory=list)


@dataclass
class Defect:
    kind: str
    detail: str
    variable_refs: list[str] = field(default_factory=list)
    evidence_refs: list[str] = field(default_factory=list)
    window: list[float] | None = None
    bbox: list[float] | None = None
    blocking: bool = True
    answer_sensitive: bool = True
    dependent_nodes: int = 1


@dataclass
class Budget:
    max_model_calls: int
    max_unique_frames: int
    max_seconds: float


@dataclass
class TaskSpec:
    target: str
    target_entity: str
    answer_type: str
    output_unit: str
    coverage_need: str
    plan: str
    slots: list
    givens: list
    anchors: list
    semantic_constraints: list
    precision: dict
    scope: str
    snapshot: str


@dataclass
class QueryGraph:
    nodes: list
    target_node: str | dict
    output_unit: str


@dataclass
class ConstraintRecord:
    id: str
    relation: str
    args: list
    sources: list
    snapshot: str


@dataclass
class R8Result:
    prediction: str | None
    status: str
    verified: bool
    stop_reason: str
    value: object = None
    evidence_status: str = "unresolved"
    modeling_status: str = "unresolved"
    solver_status: str = "not_run"
    matching_status: str = "not_run"
    task_spec: dict = field(default_factory=dict)
    variables: dict = field(default_factory=dict)
    evidence: dict = field(default_factory=dict)
    coverage: list = field(default_factory=list)
    defects: list = field(default_factory=list)
    execution: dict = field(default_factory=dict)
    resources: dict = field(default_factory=dict)
    trace: dict = field(default_factory=dict)

    @property
    def text(self):
        return self.prediction or ""

    def to_dict(self):
        return plain({"pipeline_id": "R8", "version": VERSION, **asdict(self)})
