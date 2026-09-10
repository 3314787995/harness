"""Public contracts for preselected inventory questions and permitted history."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from datetime import datetime
from itertools import pairwise
from typing import Any

from qwen3vl_agent.r1.types import R1Choice, normalize_choices, span_from
from qwen3vl_agent.r3.types import BudgetExhausted, ProtocolError, R3Budget

NAMESPACES = {"physical_instance", "semantic_category", "text_value", "task_item"}
OPERATIONS = {
    "list_members",
    "count_unique",
    "membership",
    "missing_members",
    "union",
    "intersection",
    "difference",
    "group_count",
    "compare_count",
    "argmax_count",
    "argmin_count",
    "remaining_quantity",
    "max_simultaneous_count",
    "ordered_counts",
}
OP_ALIASES = {"COUNT_DISTINCT": "count_unique", "UNION": "union", "INTERSECTION": "intersection",
              "DIFFERENCE": "difference", "GROUP_COUNT": "group_count", "ARGMAX": "argmax_count",
              "MAX_SIMULTANEOUS_COUNT": "max_simultaneous_count", "ORDERED_COUNTS": "ordered_counts"}
MODALITIES = {"video", "screen_text", "subtitle", "asr"}
RELATIONS = {"visually_present", "text_present", "mentioned", "planned", "completed"}
PREDICATE_KINDS = {"static", "moving", "exits", "enters", "task"}
TASK_PROJECTIONS = {"auto", "active_plan", "completed", "remaining"}
UPDATE_KINDS = {
    "create_plan",
    "add_item",
    "set_quantity",
    "cancel_item",
    "replace_item",
    "complete_item",
    "undo_completion",
}


def finite(value: Any, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("expected a finite number")
    if minimum is not None and value < minimum:
        raise ValueError("number below minimum")
    return float(value)


def timestamp(value: str | float) -> float:
    if isinstance(value, str):
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            raise ValueError("history timestamps require an explicit timezone")
        return parsed.timestamp()
    return finite(value)


def interval(value: Any) -> tuple[float, float]:
    span = span_from(value)
    if span is None:
        raise ValueError("interval is required")
    return (span.start_seconds, span.end_seconds)


@dataclass(frozen=True)
class R4Budget:
    """R4 owns its budget; deterministic rendering reserves no model call."""
    max_model_calls: int = 192
    terminal_call_reserve: int = 0
    max_provider_calls: int = 4096
    max_frame_exposures: int = 4096
    max_media_pixels: int = 4294967296
    max_visual_tokens: int | None = None
    max_text_chars_per_call: int = 60000
    max_generated_tokens: int = 32768
    visual_tokens_per_call: int = 8192

    def __post_init__(self) -> None:
        for key, value in asdict(self).items():
            if key == "max_visual_tokens" and value is None:
                continue
            minimum = 0 if key in {"terminal_call_reserve", "max_provider_calls"} else 1
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"invalid R4 budget: {key}")


@dataclass(frozen=True)
class ExternalFile:
    path: str
    kind: str = "subtitle"
    coverage_status: str = "unknown"
    covered_intervals: tuple[tuple[float, float], ...] = ()
    unresolved_intervals: tuple[tuple[float, float], ...] = ()
    alignment_error_sec: float | None = None
    provider_version: str = "file-v1"

    def __post_init__(self) -> None:
        if not self.path or self.kind not in {"subtitle", "asr"}:
            raise ValueError("invalid external file")
        if self.coverage_status not in {"complete", "partial", "unknown"}:
            raise ValueError("invalid external coverage status")
        for name in ("covered_intervals", "unresolved_intervals"):
            object.__setattr__(self, name, tuple(interval(v) for v in getattr(self, name)))
        if self.alignment_error_sec is not None:
            finite(self.alignment_error_sec, minimum=0)


@dataclass(frozen=True)
class HistoryMap:
    media_start: float
    media_end: float
    history_start: str | float

    def __post_init__(self) -> None:
        finite(self.media_start, minimum=0)
        finite(self.media_end, minimum=0)
        if self.media_end <= self.media_start:
            raise ValueError("empty history time map")
        timestamp(self.history_start)

    def at(self, media_time: float) -> float:
        return timestamp(self.history_start) + media_time - self.media_start


@dataclass(frozen=True)
class MediaSource:
    video_path: str
    entry_id: str = "source-1"
    source_id: str | None = None
    allowed_spans: tuple[tuple[float, float], ...] = ()
    recorded_at: str | None = None
    time_map: tuple[HistoryMap, ...] = ()
    available_modalities: tuple[str, ...] = ("video", "screen_text")
    external_files: tuple[ExternalFile, ...] = ()
    actor_bindings: dict[str, str] = field(default_factory=dict)
    task_context: str | None = None
    context_provenance: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.video_path or not self.entry_id:
            raise ValueError("source video_path and entry_id are required")
        object.__setattr__(self, "allowed_spans", tuple(interval(v) for v in self.allowed_spans))
        object.__setattr__(
            self,
            "time_map",
            tuple(v if isinstance(v, HistoryMap) else HistoryMap(**v) for v in self.time_map),
        )
        object.__setattr__(
            self,
            "external_files",
            tuple(
                v if isinstance(v, ExternalFile) else ExternalFile(**v) for v in self.external_files
            ),
        )
        if self.recorded_at is not None:
            timestamp(self.recorded_at)
        if self.recorded_at and self.time_map:
            raise ValueError("use recorded_at or time_map, not both")
        if not self.available_modalities or set(self.available_modalities) - MODALITIES:
            raise ValueError("invalid source modalities")
        spans = sorted(self.allowed_spans)
        if any(a[1] > b[0] for a, b in pairwise(spans)):
            raise ValueError("allowed source spans overlap")
        maps = sorted(self.time_map, key=lambda m: m.media_start)
        if any(a.media_end > b.media_start for a, b in pairwise(maps)):
            raise ValueError("history media mappings overlap")


@dataclass(frozen=True)
class R4Request:
    question: str
    sources: tuple[MediaSource, ...] = ()
    video_path: str | None = None
    request_id: str = "r4-request"
    video_id: str = "video"
    group_id: str | None = None
    native_labels: tuple[str, ...] = ()
    choices: tuple[R1Choice, ...] = ()
    allowed_scope: Any = None
    query_scope: Any = None
    observation_cutoff: float | None = None
    query_time: str | None = None
    available_modalities: tuple[str, ...] = ("video", "screen_text")
    external_files: tuple[ExternalFile, ...] = ()
    execution_subtype: str | None = None
    benchmark_policy: dict[str, Any] = field(default_factory=dict)
    output_protocol: str = "auto"
    budget: R4Budget = field(default_factory=R4Budget)
    checkpoint_path: str | None = None
    resume: bool = False

    def __post_init__(self) -> None:
        if not self.question.strip() or not self.request_id:
            raise ValueError("question and request_id are required")
        if bool(self.sources) == bool(self.video_path):
            raise ValueError("provide sources or one video_path")
        choices = self.choices
        if isinstance(choices, dict):
            choices = [{"label": k, "text": v} for k, v in choices.items()]
        object.__setattr__(self, "choices", normalize_choices(choices))
        object.__setattr__(
            self,
            "sources",
            tuple(v if isinstance(v, MediaSource) else MediaSource(**v) for v in self.sources),
        )
        object.__setattr__(
            self,
            "external_files",
            tuple(
                v if isinstance(v, ExternalFile) else ExternalFile(**v) for v in self.external_files
            ),
        )
        if len({s.entry_id for s in self.sources}) != len(self.sources):
            raise ValueError("source entry IDs must be unique")
        if self.sources and (self.allowed_scope is not None or self.external_files):
            raise ValueError("multi-source permissions/files belong on each source")
        if self.allowed_scope is not None:
            interval(self.allowed_scope)
        if self.observation_cutoff is not None:
            finite(self.observation_cutoff, minimum=0)
            if self.sources:
                raise ValueError("multi-source cutoff uses absolute query_time")
        if self.query_time is not None:
            timestamp(self.query_time)
        if set(self.available_modalities) - MODALITIES:
            raise ValueError("invalid modalities")
        if isinstance(self.budget, dict):
            object.__setattr__(self, "budget", R4Budget(**self.budget))
        if self.execution_subtype in OP_ALIASES:
            object.__setattr__(self, "execution_subtype", OP_ALIASES[self.execution_subtype])
        if self.execution_subtype and self.execution_subtype not in OPERATIONS:
            raise ValueError("unknown R4 operation")
        if self.output_protocol not in {"auto", "multiple_choice", "numeric", "free_text", "list"}:
            raise ValueError("unknown R4 output protocol")
        if self.output_protocol == "multiple_choice" and not self.choices:
            raise ValueError("multiple_choice requires choices")
        if self.choices and self.output_protocol not in {"auto", "multiple_choice"}:
            raise ValueError("choices conflict with output protocol")
        public = {
            "count_unit",
            "population",
            "normalization",
            "completion_predicate",
            "history_complete",
            "candidate_mode",
            "numeric_option_intervals",
            "unique_missing",
        }
        if not isinstance(self.benchmark_policy, dict) or set(self.benchmark_policy) - public:
            raise ValueError("benchmark_policy accepts public inventory conventions only")
        intervals = self.benchmark_policy.get("numeric_option_intervals", {})
        if intervals:
            if not isinstance(intervals, dict) or set(intervals) != {c.label for c in self.choices}:
                raise ValueError("numeric option intervals must cover every original label")
            for bounds in intervals.values():
                if (
                    not isinstance(bounds, (tuple, list))
                    or len(bounds) != 2
                    or finite(bounds[0]) > finite(bounds[1])
                ):
                    raise ValueError("invalid inclusive numeric option interval")
        if self.resume and not self.checkpoint_path:
            raise ValueError("resume requires checkpoint_path")

    def media_sources(self) -> tuple[MediaSource, ...]:
        return self.sources or (
            MediaSource(
                video_path=self.video_path,
                allowed_spans=(interval(self.allowed_scope),)
                if self.allowed_scope is not None
                else (),
                available_modalities=self.available_modalities,
                external_files=self.external_files,
            ),
        )


@dataclass(frozen=True)
class SetSpec:
    set_id: str
    namespace: str
    target: str
    predicate: str = "visible in the query scope"
    predicate_kind: str = "static"
    evidence_relation: str = "visually_present"
    required_modalities: tuple[str, ...] = ("video",)
    candidates: tuple[str, ...] = ()
    scope: dict[str, Any] = field(default_factory=dict)
    normalization: dict[str, Any] = field(default_factory=dict)
    population: str = "physical_objects"
    owner: str | None = None
    task_id: str | None = None
    task_projection: str = "auto"
    count_unit: str = ""
    membership: dict[str, str] = field(default_factory=dict)
    equivalence: str = "auto"
    attribute_keys: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.set_id or not self.target or self.namespace not in NAMESPACES:
            raise ProtocolError("invalid inventory set")
        if self.predicate_kind not in PREDICATE_KINDS:
            raise ProtocolError("invalid inventory predicate kind")
        if self.evidence_relation not in RELATIONS:
            raise ProtocolError("invalid evidence relation")
        if not self.required_modalities or set(self.required_modalities) - MODALITIES:
            raise ProtocolError("invalid required modalities")
        if self.evidence_relation in {"mentioned", "planned", "completed"} and not (
            set(self.required_modalities) & {"subtitle", "asr", "screen_text"}
        ):
            raise ProtocolError("language claims require a text modality")
        if (
            self.predicate_kind in {"moving", "exits", "enters"}
            and "video" not in self.required_modalities
        ):
            raise ProtocolError("motion requires video")
        if set(self.normalization) - {"casefold", "strip_punctuation", "aliases", "policy_id"}:
            raise ProtocolError("unknown normalization rule")
        if not isinstance(self.normalization.get("aliases", {}), dict):
            raise ProtocolError("aliases must be a mapping")
        if self.task_projection not in TASK_PROJECTIONS:
            raise ProtocolError("invalid task projection")
        if any(not isinstance(v, str) or not v for v in self.candidates):
            raise ProtocolError("candidates must be nonempty strings")
        if self.equivalence not in {"auto", "entity", "category", "literal", "combination", "task"}:
            raise ProtocolError("unknown equivalence rule")
        if not isinstance(self.membership, dict) or any(not isinstance(k, str) or not isinstance(v, str)
                                                       or not k or not v for k, v in self.membership.items()):
            raise ProtocolError("membership requires named text conditions")
        if set(self.membership) & {"target", "predicate"}:
            raise ProtocolError("membership cannot replace the target or predicate condition")
        if not isinstance(self.count_unit, str) or any(not isinstance(v, str) or not v for v in self.attribute_keys):
            raise ProtocolError("invalid member unit/attributes")


@dataclass(frozen=True)
class SetOperation:
    operation_id: str
    op: str
    inputs: tuple[str, ...]
    candidates: tuple[str, ...] = ()
    group_by: str = "category"
    compare: str = "greater"

    def __post_init__(self) -> None:
        object.__setattr__(self, "op", OP_ALIASES.get(self.op, self.op))
        if not self.operation_id or self.op not in OPERATIONS or not self.inputs:
            raise ProtocolError("invalid set operation")
        if self.compare not in {"greater", "less", "equal"}:
            raise ProtocolError("invalid count comparison")


@dataclass(frozen=True)
class InventorySpec:
    sets: tuple[SetSpec, ...]
    operations: tuple[SetOperation, ...]
    scope: dict[str, Any] = field(default_factory=lambda: {"kind": "full"})
    unresolved: tuple[str, ...] = ()
    version: int = 1
    choice_values: dict[str, Any] = field(default_factory=dict)
    output_id: str | None = None

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> InventorySpec:
        from qwen3vl_agent.r4.contracts import validate_semantics, validate_shape

        value = validate_shape("compile", value)
        result = cls(
            tuple(SetSpec(**s) for s in value["sets"]),
            tuple(SetOperation(**o) for o in value["operations"]),
            value.get("scope", {"kind": "full"}),
            tuple(value.get("unresolved", ())),
            value.get("version", 1),
            value.get("choice_values", {}),
            value.get("output_id"),
        )
        keys = [s.set_id for s in result.sets]
        if not keys or len(set(keys)) != len(keys) or not result.operations:
            raise ProtocolError("nonempty unique sets and operations required")
        if len({o.operation_id for o in result.operations}) != len(result.operations):
            raise ProtocolError("duplicate operation ID")
        by_id = {s.set_id: s for s in result.sets}
        kinds = {s.set_id: "set" for s in result.sets}
        for op in result.operations:
            if op.operation_id in by_id:
                raise ProtocolError("operation ID collides with a set or earlier operation")
            if set(op.inputs) - by_id.keys():
                raise ProtocolError("unknown/forward/cyclic operation input")
            input_kinds = [kinds[k] for k in op.inputs]
            if op.op in {"union", "intersection", "difference", "count_unique", "list_members", "membership", "missing_members", "group_count", "remaining_quantity", "max_simultaneous_count"} and any(k != "set" for k in input_kinds):
                raise ProtocolError("operator requires set operands")
            if op.op == "ordered_counts" and any(k not in {"set", "count"} for k in input_kinds):
                raise ProtocolError("ordered_counts requires sets or scalar counts")
            if op.op == "compare_count" and any(k not in {"set", "count"} for k in input_kinds):
                raise ProtocolError("compare_count requires sets or scalar counts")
            if op.op in {"argmax_count", "argmin_count"} and (any(k not in {"set", "count", "groups"} for k in input_kinds)
                    or ("groups" in input_kinds and len(input_kinds) != 1)):
                raise ProtocolError("group winner requires sets/counts or one group-count result")
            if op.op == "max_simultaneous_count" and by_id[op.inputs[0]].namespace != "physical_instance":
                raise ProtocolError("simultaneous instance count requires physical instances")
            if (
                op.op in {"union", "intersection", "difference"}
                and len({by_id[k].namespace for k in op.inputs}) != 1
            ):
                raise ProtocolError("set operations require matching namespaces")
            if op.op in {"difference", "compare_count"} and len(op.inputs) != 2:
                raise ProtocolError("binary operation requires two input sets")
            if (
                op.op
                in {
                    "count_unique",
                    "list_members",
                    "membership",
                    "missing_members",
                    "remaining_quantity",
                    "max_simultaneous_count",
                }
                and len(op.inputs) != 1
            ):
                raise ProtocolError("unary operation requires one input set")
            if op.op == "remaining_quantity" and by_id[op.inputs[0]].namespace != "task_item":
                raise ProtocolError("remaining_quantity requires task_item")
            if (
                len(op.inputs) > 1
                and len({(by_id[k].namespace, by_id[k].population, by_id[k].count_unit) for k in op.inputs}) != 1
            ):
                raise ProtocolError(
                    "multi-set reduction requires consistent member units and population"
                )
            if op.op in {"union", "intersection", "difference"}:
                first = by_id[op.inputs[0]]
                if any(
                    by_id[k].normalization != first.normalization
                    or by_id[k].population != first.population
                    or by_id[k].equivalence != first.equivalence
                    or by_id[k].attribute_keys != first.attribute_keys
                    for k in op.inputs
                ):
                    raise ProtocolError(
                        "set operands require compatible normalization and population"
                    )
            by_id[op.operation_id] = by_id[op.inputs[0]]
            kinds[op.operation_id] = ("set" if op.op in {"union", "intersection", "difference"} else
                                      "count" if op.op in {"count_unique", "max_simultaneous_count"} else
                                      "groups" if op.op == "group_count" else "value")
        if result.output_id is not None and result.output_id not in {o.operation_id for o in result.operations}:
            raise ProtocolError("output_id must reference an operation")
        for scope in [result.scope, *(s.scope for s in result.sets if s.scope)]:
            if scope.get("kind", "full") not in {
                "full",
                "interval",
                "frame",
                "semantic",
                "history",
            }:
                raise ProtocolError("unknown inventory scope")
            if scope.get("kind") == "interval":
                interval(scope["interval"])
            if scope.get("kind") == "frame":
                finite(scope["timestamp_sec"], minimum=0)
            if scope.get("kind") == "semantic" and not scope.get("description"):
                raise ProtocolError("semantic scope needs a description")
        validate_semantics(result)
        return result

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CoverageTile:
    tile_id: str
    entry_id: str
    core: tuple[float, float]
    context: tuple[float, float]
    fps: float
    kind: str = "video"
    set_ids: list[str] = field(default_factory=list)
    bbox: list[float] | None = None
    depth: int = 0
    status: str = "pending"
    audit_done: bool = False
    source_refs: list[str] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)
    max_gap_sec: float | None = None
    sizes: list[Any] = field(default_factory=list)
    children: list[str] = field(default_factory=list)


@dataclass
class ObservationRecord:
    observation_id: str
    set_id: str
    entry_id: str
    source_id: str
    local_id: str
    value: str
    category: str
    predicate_status: str
    evidence_relation: str
    evidence_refs: list[str]
    detections: list[dict[str, Any]] = field(default_factory=list)
    attributes: dict[str, Any] = field(default_factory=dict)
    raw_text: str | None = None
    candidate_values: list[str] = field(default_factory=list)
    visibility: str = "clear"
    population_status: str = "included"
    owner: str | None = None
    task_id: str | None = None
    issues: list[str] = field(default_factory=list)
    predicate_evidence: dict[str, Any] = field(default_factory=dict)
    reference_status: str = "validated"
    semantic_status: str = "model_reported"


@dataclass
class IdentityRelation:
    left: str
    right: str
    relation: str
    evidence_refs: list[str]
    reason: str
    relation_id: str = ""
    supersedes: list[str] = field(default_factory=list)


@dataclass
class R4Result:
    prediction: str | None
    value_state: dict[str, Any]
    completion_state: str
    support_level: str
    answer_basis: str
    inventory: dict[str, Any]
    coverage_manifest: list[dict[str, Any]]
    unresolved_items: list[str]
    evidence_refs: dict[str, Any]
    resources: dict[str, Any]
    trace: dict[str, Any]
    failure: dict[str, Any] | None = None
    result_status: str = "unresolved"
    answer_mode: str = "none"
    evidence_status: str = "unresolved"

    def to_dict(self) -> dict[str, Any]:
        return {"pipeline_id": "R4", **asdict(self)}


__all__ = [
    "BudgetExhausted",
    "CoverageTile",
    "ExternalFile",
    "HistoryMap",
    "IdentityRelation",
    "InventorySpec",
    "MediaSource",
    "ObservationRecord",
    "ProtocolError",
    "R4Budget",
    "R4Request",
    "R4Result",
    "SetOperation",
    "SetSpec",
]
