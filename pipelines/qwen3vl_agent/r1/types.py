"""Public R1 contracts. No model, decoder, or ASR implementation lives here."""

from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass, field
from typing import Any

from qwen3vl_agent.p01.types import TimeSpan


def span_from(value: Any) -> TimeSpan | None:
    if value is None:
        return None
    if isinstance(value, TimeSpan):
        span = value
    elif isinstance(value, dict):
        span = TimeSpan(float(value["start_sec"]), float(value["end_sec"]))
    else:
        if len(value) != 2:
            raise ValueError("a span requires start and end")
        span = TimeSpan(float(value[0]), float(value[1]))
    if not all(math.isfinite(t) for t in (span.start_seconds, span.end_seconds)):
        raise ValueError("span timestamps must be finite")
    if span.context_start_seconds is not None or span.context_end_seconds is not None:
        raise ValueError("R1 scopes must not contain implicit decode padding")
    return span


@dataclass(frozen=True)
class R1Choice:
    label: str
    text: str


def normalize_choices(values: Any) -> tuple[R1Choice, ...]:
    choices = []
    for i, item in enumerate(values or ()):
        if isinstance(item, R1Choice):
            choice = item
        elif isinstance(item, dict):
            choice = R1Choice(str(item["label"]), str(item["text"]))
        else:
            match = re.match(r"^([A-Za-z]|\d+)[.):]\s+(.+)$", str(item).strip(), re.DOTALL)
            choice = (
                R1Choice(match[1], match[2])
                if match
                else R1Choice(chr(65 + i) if i < 26 else str(i + 1), str(item).strip())
            )
        if not choice.label.strip() or not choice.text.strip():
            raise ValueError("choices require nonempty labels and text")
        choices.append(choice)
    if choices and (len(choices) < 2 or len({c.label for c in choices}) != len(choices)):
        raise ValueError("choices require at least two unique labels")
    return tuple(choices)


@dataclass(frozen=True)
class R1Budget:
    max_model_calls: int = 20
    terminal_call_reserve: int = 2
    max_provider_calls: int = 8
    max_frame_exposures: int = 2240
    max_media_pixels: int = 251658240
    max_visual_tokens: int | None = None
    max_text_chars_per_call: int = 60000

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < (0 if name == "max_provider_calls" else 1)
            ):
                raise ValueError(f"invalid budget: {name}")
        if self.terminal_call_reserve < 2 or self.max_model_calls < self.terminal_call_reserve:
            raise ValueError("reserve two terminal calls within the model-call budget")


@dataclass(frozen=True)
class R1Request:
    video_path: str
    question: str
    request_id: str = "r1-request"
    video_id: str = "video"
    group_id: str | None = None
    choices: tuple[R1Choice, ...] = ()
    allowed_scope: TimeSpan | None = None
    query_scope: TimeSpan | None = None
    available_modalities: tuple[str, ...] = ("video", "screen_text")
    output_protocol: str = "auto"
    failure_text: str = "Insufficient evidence."
    budget: R1Budget = field(default_factory=R1Budget)

    def __post_init__(self) -> None:
        if not self.video_path.strip() or not self.question.strip():
            raise ValueError("video_path and question are required")
        object.__setattr__(self, "choices", normalize_choices(self.choices))
        object.__setattr__(self, "allowed_scope", span_from(self.allowed_scope))
        object.__setattr__(self, "query_scope", span_from(self.query_scope))
        if isinstance(self.budget, dict):
            object.__setattr__(self, "budget", R1Budget(**self.budget))
        modalities = tuple(dict.fromkeys(self.available_modalities))
        if "video" not in modalities or set(modalities) - {
            "video",
            "screen_text",
            "subtitle",
            "asr",
        }:
            raise ValueError("R1 requires video; only screen_text, subtitle and asr may be added")
        object.__setattr__(self, "available_modalities", modalities)
        if self.output_protocol not in {"auto", "multiple_choice", "free_text", "numeric"}:
            raise ValueError("unknown output_protocol")
        if self.output_protocol == "multiple_choice" and not self.choices:
            raise ValueError("multiple_choice requires choices")
        if self.choices and self.output_protocol not in {"auto", "multiple_choice"}:
            raise ValueError("choices conflict with output_protocol")


@dataclass(frozen=True)
class QueryField:
    field_id: str
    description: str


@dataclass(frozen=True)
class QuerySpec:
    fields: tuple[QueryField, ...]
    anchor_description: str
    observation_modes: tuple[str, ...] = ("static",)
    coverage: str = "point"
    requires_reference: bool = False
    reference_description: str = ""
    reference_relation: str = "any"
    required_modalities: tuple[str, ...] = ()
    semantic_hint: str = ""
    requires_speaker_binding: bool = False


@dataclass(frozen=True)
class DiscriminantSpec:
    inspection_needs: tuple[str, ...] = ()
    target_union: tuple[str, ...] = ()
    observation_modes: tuple[str, ...] = ()


@dataclass
class SearchCandidate:
    candidate_id: str
    span: TimeSpan
    anchor_frame_ids: tuple[str, ...] = ()
    matched_anchor_conditions: tuple[str, ...] = ()
    unresolved_anchor_conditions: tuple[str, ...] = ()


@dataclass
class SearchState:
    checked_nodes: list[str] = field(default_factory=list)
    frontier: list[str] = field(default_factory=list)
    candidates: list[SearchCandidate] = field(default_factory=list)
    observed_candidates: list[str] = field(default_factory=list)
    navigation_frame_ids: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class EvidenceFact:
    fact_id: str
    statement: str
    structured_value: str
    subject_or_local_entity: str
    attribute: str
    source_id: str
    source_frame_ids: tuple[str, ...]
    source_segment_ids: tuple[str, ...]
    start_sec: float
    end_sec: float
    view_id: str
    observation_status: str
    supports_query_fields: tuple[str, ...]
    source_kind: str = "visual"
    uncertain_characters: str = ""
    original_frame_ids: tuple[str, ...] = ()
    crop_transforms: dict[str, Any] = field(default_factory=dict)


@dataclass
class CoverageRecord:
    planned_span: TimeSpan
    observed_frame_ids: tuple[str, ...] = ()
    actual_max_gap_sec: float | None = None
    spatial_resolution: list[tuple[int, int]] = field(default_factory=list)
    required_resolution_met: bool = False
    observation_completed: bool = False
    truncated: bool = False
    decode_failures: list[str] = field(default_factory=list)
    unresolved: list[str] = field(default_factory=list)
    coverage_kind: str = "base"


@dataclass
class EvidencePacket:
    packet_id: str
    candidate_id: str
    role: str
    span: TimeSpan
    facts: list[EvidenceFact] = field(default_factory=list)
    coverage: list[CoverageRecord] = field(default_factory=list)
    source_views: list[dict[str, Any]] = field(default_factory=list)
    anchor_match: str = "unresolved"
    target_binding: str = "unresolved"
    anchor_source_ids: list[str] = field(default_factory=list)
    target_source_ids: list[str] = field(default_factory=list)
    unresolved: list[str] = field(default_factory=list)
    crop_requests: list[dict[str, Any]] = field(default_factory=list)
    existence: str = "unknown"
    absence_basis: str = ""
    rechecks: int = 0
    locator_anchor_ids: tuple[str, ...] = ()
    speech_bindings: list[dict[str, Any]] = field(default_factory=list)
    fact_reviews: list[dict[str, Any]] = field(default_factory=list)
    review_request: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BindingRecord:
    left_packet_id: str
    right_packet_id: str
    relation: str
    source_frame_ids: tuple[str, ...]
    basis: str
    discriminating_features: tuple[str, ...] = ()
    feature_kinds: tuple[str, ...] = ()


@dataclass
class EvidenceBundle:
    bundle_id: str
    packets: list[EvidencePacket] = field(default_factory=list)
    bindings: list[BindingRecord] = field(default_factory=list)
    conflicts: list[dict[str, Any]] = field(default_factory=list)
    required_spans: list[TimeSpan] = field(default_factory=list)

    @property
    def facts(self) -> list[EvidenceFact]:
        return [fact for packet in self.packets for fact in packet.facts]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EvidenceAudit:
    sufficient: bool = False
    missing_fields: list[str] = field(default_factory=list)
    unresolved: list[str] = field(default_factory=list)
    coverage_complete: bool = False
    clear_fact_ids: list[str] = field(default_factory=list)


@dataclass
class R1Result:
    prediction: str | None
    answer_basis: str
    support_level: str
    completion_state: str
    evidence_bundle: EvidenceBundle | None = None
    claim_assessments: list[dict[str, Any]] = field(default_factory=list)
    unresolved_reasons: list[str] = field(default_factory=list)
    resources: dict[str, Any] = field(default_factory=dict)
    trace: dict[str, Any] = field(default_factory=dict)

    @property
    def output_text(self) -> str:
        return self.prediction or ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
