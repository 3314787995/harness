"""Public contracts for the already-selected R3 event-ledger executor."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from .query import QuerySpec

from qwen3vl_agent.p01.types import TimeSpan
from qwen3vl_agent.r1.types import R1Choice, normalize_choices, span_from

OPERATIONS = {
    "count_occurrences",
    "localize_event",
    "first_occurrence",
    "last_occurrence",
    "first_k",
    "last_k",
    "nth_occurrence",
    "order_events",
    "next_after_anchor",
    "previous_before_anchor",
    "event_duration",
    "cooccurrence_frequency",
}
UNITS = {
    "appearance_episode",
    "action_cycle",
    "state_transition",
    "scene_episode",
    "utterance_or_mention",
}
FACT_KINDS = {"visual_event", "screen_text_event", "utterance", "reported_event"}


class ProtocolError(ValueError):
    pass


class BudgetExhausted(RuntimeError):
    pass


@dataclass(frozen=True)
class Bracket:
    lo: float | None = None
    hi: float | None = None

    def __post_init__(self) -> None:
        if any(
            v is not None and (isinstance(v, bool) or not math.isfinite(v) or v < 0)
            for v in (self.lo, self.hi)
        ):
            raise ValueError("invalid temporal bracket")
        if self.lo is not None and self.hi is not None and self.lo > self.hi:
            raise ValueError("inverted temporal bracket")

    @classmethod
    def parse(cls, value: Any) -> Bracket:
        if isinstance(value, cls):
            return value
        if isinstance(value, dict):
            return cls(value.get("lo"), value.get("hi"))
        return cls(*(value or (None, None)))

    def to_list(self) -> list[float | None]:
        return [self.lo, self.hi]


@dataclass(frozen=True)
class R3Budget:
    max_model_calls: int = 4096
    terminal_call_reserve: int = 1
    max_provider_calls: int = 4096
    max_frame_exposures: int = 196608
    max_media_pixels: int = 51539607552
    max_visual_tokens: int | None = None
    max_text_chars_per_call: int = 60000

    def __post_init__(self) -> None:
        for key, val in asdict(self).items():
            if val is None and key == "max_visual_tokens":
                continue
            if (
                isinstance(val, bool)
                or not isinstance(val, int)
                or val < (0 if key == "max_provider_calls" else 1)
            ):
                raise ValueError(f"invalid R3 budget: {key}")
        if ((type(self) is R3Budget and self.terminal_call_reserve != 1)
            or (type(self) is not R3Budget and self.terminal_call_reserve < 2)
            or self.max_model_calls < self.terminal_call_reserve):
            raise ValueError("v5 reserves exactly one terminal call")


@dataclass(frozen=True)
class R3Request:
    video_path: str
    question: str
    request_id: str = "r3-request"
    video_id: str = "video"
    group_id: str | None = None
    choices: tuple[R1Choice, ...] = ()
    allowed_scope: TimeSpan | None = None
    query_scope: TimeSpan | str | None = None
    observation_cutoff: float | None = None
    query_spec: dict[str, Any] | QuerySpec | None = None
    execution_subtype: str | None = None
    benchmark_policy: dict[str, Any] = field(default_factory=dict)
    available_modalities: tuple[str, ...] = ("video", "screen_text")
    budget: R3Budget = field(default_factory=R3Budget)
    output_protocol: str = "auto"
    checkpoint_path: str | None = None
    resume: bool = False

    def __post_init__(self) -> None:
        if not self.video_path.strip() or not self.question.strip():
            raise ValueError("video_path and question are required")
        choices = (
            [{"label": label, "text": value} for label, value in self.choices.items()]
            if isinstance(self.choices, dict)
            else self.choices
        )
        object.__setattr__(self, "choices", normalize_choices(choices))
        object.__setattr__(self, "allowed_scope", span_from(self.allowed_scope))
        if not isinstance(self.query_scope, str):
            object.__setattr__(self, "query_scope", span_from(self.query_scope))
        if isinstance(self.budget, dict):
            object.__setattr__(self, "budget", R3Budget(**self.budget))
        if self.observation_cutoff is not None and (
            isinstance(self.observation_cutoff, bool)
            or not math.isfinite(self.observation_cutoff)
            or self.observation_cutoff <= 0
        ):
            raise ValueError("observation_cutoff must be a positive finite timestamp")
        if self.execution_subtype is not None and self.execution_subtype not in OPERATIONS:
            raise ValueError("unknown R3 operation")
        if "video" not in self.available_modalities or set(self.available_modalities) - {
            "video",
            "screen_text",
            "subtitle",
            "asr",
        }:
            raise ValueError(
                "R3 requires video and only supports permitted screen_text/subtitle/asr"
            )
        if self.output_protocol not in {
            "auto",
            "multiple_choice",
            "numeric",
            "free_text",
            "interval",
        }:
            raise ValueError("unknown output protocol")
        if self.output_protocol == "multiple_choice" and not self.choices:
            raise ValueError("multiple_choice requires choices")
        if self.choices and self.output_protocol not in {"auto", "multiple_choice"}:
            raise ValueError("choices conflict with output protocol")
        if not isinstance(self.benchmark_policy, dict):
            raise TypeError("benchmark_policy must be an object")
        for key in ("duration_option_intervals", "temporal_option_intervals"):
            bins = self.benchmark_policy.get(key, {})
            if not isinstance(bins, dict):
                raise TypeError(f"{key} must map option labels to finite intervals")
            for label, interval in bins.items():
                if (
                    label not in {c.label for c in self.choices}
                    or not isinstance(interval, (list, tuple))
                    or len(interval) != 2
                    or not all(
                        isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
                        for v in interval
                    )
                    or not 0 <= interval[0] <= interval[1]
                ):
                    raise ValueError(f"invalid {key} entry: {label}")
        if self.resume and not self.checkpoint_path:
            raise ValueError("resume requires checkpoint_path")


@dataclass(frozen=True)
class EventSpec:
    target_id: str
    description: str
    unit_kind: str = "appearance_episode"
    actor_constraint: str = ""
    object_constraint: str = ""
    start_criterion: str = ""
    completion_criterion: str = ""
    reset_criterion: str = ""
    inclusion_rule: str = "intersects"
    repeat_policy: str = "presentation"
    fact_kind: str = "visual_event"
    required_modalities: tuple[str, ...] = ()
    binding_description: str = ""
    requires_actor_binding: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.requires_actor_binding, bool):
            raise ProtocolError("requires_actor_binding must be boolean")
        if not self.target_id or not self.description or self.unit_kind not in UNITS:
            raise ProtocolError("invalid event target or unit")
        if self.inclusion_rule not in {"starts_inside", "completes_inside", "intersects"}:
            raise ProtocolError("invalid inclusion rule")
        if self.repeat_policy not in {"presentation", "world"} or self.fact_kind not in FACT_KINDS:
            raise ProtocolError("invalid repeat/fact semantics")
        if set(self.required_modalities) - {"video", "screen_text", "subtitle", "asr"}:
            raise ProtocolError("unknown required modality")
        if self.unit_kind in {"action_cycle", "state_transition"} and not self.completion_criterion:
            raise ProtocolError("cycles/transitions require an observable completion criterion")


@dataclass(frozen=True)
class Operation:
    operation_id: str
    op: str
    target_ids: tuple[str, ...]
    basis: str = "onset"
    selection: str = "all"
    k: int = 1
    group_by: str = "occurrence"
    project: str = "description"
    anchor_target_id: str | None = None
    anchor_selection: str = "unique"
    duration_aggregation: str = "single"
    duration_comparison: str = "longest"
    cooccurrence_targets: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.op not in OPERATIONS or not self.operation_id or not self.target_ids:
            raise ProtocolError("invalid temporal operation")
        if self.basis not in {"onset", "offset"}:
            raise ProtocolError("invalid ordering basis")
        if self.selection not in {
            "all",
            "unique",
            "first",
            "last",
            "first_per_category",
            "last_per_category",
        }:
            raise ProtocolError("invalid occurrence selector")
        if self.group_by not in {"occurrence", "category"}:
            raise ProtocolError("invalid grouping")
        if isinstance(self.k, bool) or not isinstance(self.k, int) or self.k < 1:
            raise ProtocolError("k must be a positive integer")
        if self.duration_aggregation not in {"single", "union", "sum", "compare"}:
            raise ProtocolError("invalid duration aggregation")
        if self.duration_comparison not in {"longest", "shortest"}:
            raise ProtocolError("invalid duration comparison")
        if self.anchor_selection not in {"unique", "first", "last"}:
            raise ProtocolError("invalid anchor selector")
        if self.op in {"next_after_anchor", "previous_before_anchor"} and not self.anchor_target_id:
            raise ProtocolError("adjacency operation requires an anchor target")


@dataclass(frozen=True)
class ScopeSpec:
    kind: str = "full"
    description: str = ""
    interval: tuple[float, float] | None = None
    relative_first_sec: float | None = None
    relative_last_sec: float | None = None
    ordinal: int | None = None

    def __post_init__(self) -> None:
        if self.kind not in {"full", "interval", "semantic"}:
            raise ProtocolError("invalid scope kind")
        if self.kind == "semantic" and not self.description:
            raise ProtocolError("semantic scope requires a description")
        if self.kind == "interval":
            if self.interval is None:
                raise ProtocolError("interval scope requires bounds")
            span_from(self.interval)
        if self.ordinal is not None and (
            isinstance(self.ordinal, bool) or not isinstance(self.ordinal, int) or self.ordinal < 1
        ):
            raise ProtocolError("semantic ordinal must be a positive integer")
        for val in (self.relative_first_sec, self.relative_last_sec):
            if val is not None and (isinstance(val, bool) or not math.isfinite(val) or val <= 0):
                raise ProtocolError("relative scope duration must be positive")
        if self.relative_first_sec is not None and self.relative_last_sec is not None:
            raise ProtocolError("choose first or last relative scope")


@dataclass(frozen=True)
class EventQuery:
    targets: tuple[EventSpec, ...]
    operations: tuple[Operation, ...]
    scope: ScopeSpec = field(default_factory=ScopeSpec)
    unresolved: tuple[str, ...] = ()
    version: int = 1

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EventQuery:
        try:
            targets = tuple(
                EventSpec(**{**t, "required_modalities": tuple(t.get("required_modalities", ()))})
                for t in data["targets"]
            )
            operations = tuple(
                Operation(
                    **{
                        **o,
                        "target_ids": tuple(o["target_ids"]),
                        "cooccurrence_targets": tuple(o.get("cooccurrence_targets", ())),
                    }
                )
                for o in data["operations"]
            )
            value = cls(
                targets,
                operations,
                ScopeSpec(**data.get("scope", {})),
                tuple(data.get("unresolved", ())),
                int(data.get("version", 1)),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ProtocolError(f"invalid EventQuery: {exc}") from exc
        ids = {t.target_id for t in targets}
        if (
            not 1 <= len(targets) <= 32
            or len(ids) != len(targets)
            or not 1 <= len(operations) <= 12
        ):
            raise ProtocolError("query requires unique targets and 1 to 12 operations")
        if len({o.operation_id for o in operations}) != len(operations):
            raise ProtocolError("duplicate operation_id")
        for op in operations:
            if {*op.target_ids, *op.cooccurrence_targets} - ids or (
                op.anchor_target_id and op.anchor_target_id not in ids
            ):
                raise ProtocolError("operation references unknown target")
        return value

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CoverageTile:
    tile_id: str
    core: tuple[float, float]
    context: tuple[float, float]
    fps: float
    observed: bool = False
    resolution_met: bool = False
    audit_needed: bool = True
    audit_done: bool = False
    audit_attempts: int = 0
    attempts: int = 0
    refinements: int = 0
    observation_ids: list[str] = field(default_factory=list)
    certificates: list[dict[str, Any]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    unresolved: list[str] = field(default_factory=list)
    external_coverage: str = "not_required"
    target_coverage: dict[str, str] = field(default_factory=dict)
    media_observations: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class EventRecord:
    event_id: str
    target_id: str
    actor_ref: str
    object_ref: str
    description: str
    category: str
    unit_kind: str
    fact_kind: str
    onset_bracket: Bracket
    offset_bracket: Bracket
    visible_span: tuple[float, float]
    evidence_refs: list[str]
    start_evidence_refs: list[str] = field(default_factory=list)
    completion_evidence_refs: list[str] = field(default_factory=list)
    reset_evidence_refs: list[str] = field(default_factory=list)
    observation_ids: list[str] = field(default_factory=list)
    member_ids: list[str] = field(default_factory=list)
    attributes: dict[str, Any] = field(default_factory=dict)
    cooccurrence: dict[str, str] = field(default_factory=dict)
    left_censored: bool = True
    right_censored: bool = True
    owner_tile_id: str | None = None
    status: str = "proposed"
    completed: bool = False
    replay_status: str = "unknown"
    unresolved_reasons: list[str] = field(default_factory=list)
    revision: int = 1
    verification: dict[str, Any] = field(default_factory=dict)
    lineage: list[str] = field(default_factory=list)
    historical_evidence_refs: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EventRecord:
        return cls(
            **{
                **data,
                "onset_bracket": Bracket.parse(data["onset_bracket"]),
                "offset_bracket": Bracket.parse(data["offset_bracket"]),
                "visible_span": tuple(data["visible_span"]),
            }
        )


@dataclass
class EventRelation:
    left_id: str
    right_id: str
    relation: str
    evidence_refs: list[str] = field(default_factory=list)
    reason: str = ""
    attempts: int = 0


@dataclass
class ObservationRecord:
    observation_id: str
    call_id: str
    tile_id: str
    kind: str
    core: tuple[float, float]
    source_refs: dict[str, dict[str, Any]]
    proposals: list[dict[str, Any]]
    unresolved: list[str]
    truncated: bool
    resolution_met: bool
    actual_max_gap_sec: float | None
    external_coverage: str = "not_required"
    stage_call_ids: dict[str, str] = field(default_factory=dict)
    visual: dict[str, Any] = field(default_factory=dict)
    fact_dispositions: list[dict[str, Any]] = field(default_factory=list)
    target_assessments: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class R3Result:
    prediction: str | None
    value_state: dict[str, Any]
    completion_state: str
    support_level: str
    answer_basis: str
    event_ledger: dict[str, Any] = field(default_factory=dict)
    coverage_manifest: list[dict[str, Any]] = field(default_factory=list)
    unresolved_items: list[str] = field(default_factory=list)
    evidence_refs: list[str] = field(default_factory=list)
    resources: dict[str, Any] = field(default_factory=dict)
    trace: dict[str, Any] = field(default_factory=dict)

    semantic_result: Any = None
    candidate_state: dict[str, Any] = field(default_factory=dict)
    evidence_gaps: list[dict[str, Any]] = field(default_factory=list)
    sampling_assumptions: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"pipeline_id": "R3", "version": "r3-5.4", **asdict(self)}

    @property
    def text(self) -> str:
        return self.prediction or ""
