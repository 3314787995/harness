from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from qwen3vl_agent.coarse_to_fine.types import FrameRef

PRIMARY_MODES = ("static_visual", "dynamic_action", "ocr", "subscene_caption")
ANSWER_MODES = ("multiple_choice", "free_text")
VISIBILITIES = ("clear", "partial", "occluded", "not_visible", "conflicting")
COVERAGE_REQUIREMENTS = ("point", "sequence", "full_span", "text_consensus")
CLAIM_VERDICTS = ("entailed", "contradicted", "not_established")
OPTION_VERDICTS = ("entailed", "contradicted", "not_established")
SUPPORT_LEVELS = ("strong", "partial", "weak", "none")
DECISION_SOURCES = (
    "initial",
    "rescued",
    "terminal_fallback",
    "composer",
    "none",
)
RESULT_STATUSES = (
    "answered",
    "video_underdetermined",
    "pipeline_insufficient",
    "input_contract_violation",
    "protocol_error",
    "resource_exhausted",
)


class P01Error(RuntimeError):
    """Base class for bounded P01 controller failures."""


class ProtocolError(P01Error):
    """Raised when a model role violates its structured protocol."""


class ResourceExhausted(P01Error):
    """Raised when a hard P01 media or model-call budget is exhausted."""


class ContractViolation(P01Error):
    """Raised when an asserted P01 input violates the executor contract."""


@dataclass(frozen=True)
class TimeSpan:
    start_seconds: float
    end_seconds: float
    source: str = "candidate"
    context_start_seconds: float | None = None
    context_end_seconds: float | None = None

    def __post_init__(self) -> None:
        if self.start_seconds < 0:
            raise ValueError("span start must be non-negative")
        if self.end_seconds <= self.start_seconds:
            raise ValueError("span end must exceed start")
        if (
            self.context_start_seconds is not None
            and self.context_start_seconds > self.start_seconds
        ):
            raise ValueError("context start cannot follow evidence start")
        if self.context_end_seconds is not None and self.context_end_seconds < self.end_seconds:
            raise ValueError("context end cannot precede evidence end")

    @property
    def duration_seconds(self) -> float:
        return self.end_seconds - self.start_seconds

    @property
    def midpoint_seconds(self) -> float:
        return (self.start_seconds + self.end_seconds) / 2

    @property
    def decode_start_seconds(self) -> float:
        return (
            self.context_start_seconds
            if self.context_start_seconds is not None
            else self.start_seconds
        )

    @property
    def decode_end_seconds(self) -> float:
        return (
            self.context_end_seconds if self.context_end_seconds is not None else self.end_seconds
        )

    def contains_evidence(self, timestamp_seconds: float, *, tolerance: float = 1e-3) -> bool:
        return self.start_seconds - tolerance <= timestamp_seconds <= self.end_seconds + tolerance

    def contains_decode(self, timestamp_seconds: float, *, tolerance: float = 1e-3) -> bool:
        return (
            self.decode_start_seconds - tolerance
            <= timestamp_seconds
            <= self.decode_end_seconds + tolerance
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "start_seconds": round(self.start_seconds, 6),
            "end_seconds": round(self.end_seconds, 6),
            "duration_seconds": round(self.duration_seconds, 6),
            "source": self.source,
            "context_start_seconds": (
                round(self.context_start_seconds, 6)
                if self.context_start_seconds is not None
                else None
            ),
            "context_end_seconds": (
                round(self.context_end_seconds, 6) if self.context_end_seconds is not None else None
            ),
        }


@dataclass(frozen=True)
class P01Request:
    video_path: str
    question: str
    choices: tuple[str, ...] = ()
    given_interval: TimeSpan | None = None
    force_choice: bool = False

    def __post_init__(self) -> None:
        if not self.video_path.strip():
            raise ValueError("video_path must not be empty")
        if not self.question.strip():
            raise ValueError("question must not be empty")
        if self.choices and not 2 <= len(self.choices) <= 26:
            raise ValueError("multiple-choice requests require 2 to 26 choices")
        if self.force_choice and not self.choices:
            raise ValueError("force_choice requires multiple-choice options")


@dataclass(frozen=True)
class ObservationSlot:
    slot_id: str
    description: str
    required: bool = True
    value_type: str = "fact"

    def to_dict(self) -> dict[str, Any]:
        return {
            "slot_id": self.slot_id,
            "description": self.description,
            "required": self.required,
            "value_type": self.value_type,
        }


@dataclass(frozen=True)
class ObservationSpec:
    answer_mode: str
    primary_mode: str
    required_slots: tuple[ObservationSlot, ...]
    target_entities: tuple[str, ...] = ()
    target_actions: tuple[str, ...] = ()
    target_attributes: tuple[str, ...] = ()
    target_relations: tuple[str, ...] = ()
    detail_requests: tuple[str, ...] = ()
    temporal_hint: str | None = None
    coverage_requirement: str = "point"
    output_language: str = "same_as_question"

    def __post_init__(self) -> None:
        if self.answer_mode not in ANSWER_MODES:
            raise ValueError(f"unsupported answer mode: {self.answer_mode}")
        if self.primary_mode not in PRIMARY_MODES:
            raise ValueError(f"unsupported primary mode: {self.primary_mode}")
        if not self.required_slots or len(self.required_slots) > 4:
            raise ValueError("observation spec requires 1 to 4 slots")
        if self.coverage_requirement not in COVERAGE_REQUIREMENTS:
            raise ValueError(f"unsupported coverage requirement: {self.coverage_requirement}")

    @property
    def required_slot_ids(self) -> set[str]:
        return {slot.slot_id for slot in self.required_slots if slot.required}

    def to_dict(self) -> dict[str, Any]:
        return {
            "answer_mode": self.answer_mode,
            "primary_mode": self.primary_mode,
            "required_slots": [slot.to_dict() for slot in self.required_slots],
            "target_entities": list(self.target_entities),
            "target_actions": list(self.target_actions),
            "target_attributes": list(self.target_attributes),
            "target_relations": list(self.target_relations),
            "detail_requests": list(self.detail_requests),
            "temporal_hint": self.temporal_hint,
            "coverage_requirement": self.coverage_requirement,
            "output_language": self.output_language,
        }


@dataclass(frozen=True)
class CanonicalOption:
    option_id: str
    benchmark_label: str
    text: str

    def to_dict(self) -> dict[str, str]:
        return {
            "option_id": self.option_id,
            "benchmark_label": self.benchmark_label,
            "text": self.text,
        }


@dataclass(frozen=True)
class ClaimTest:
    claim_id: str
    statement: str
    slot_ids: tuple[str, ...] = ()
    predicate: str = "freeform"
    expected_value: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "claim_id": self.claim_id,
            "statement": self.statement,
            "slot_ids": list(self.slot_ids),
            "predicate": self.predicate,
            "expected_value": self.expected_value,
        }


@dataclass(frozen=True)
class OptionRule:
    option_id: str
    all_of: tuple[str, ...]
    none_of: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "option_id": self.option_id,
            "all_of": list(self.all_of),
            "none_of": list(self.none_of),
        }


@dataclass(frozen=True)
class DecisionSpec:
    claim_tests: tuple[ClaimTest, ...]
    option_rules: tuple[OptionRule, ...]
    cannot_determine_option_id: str | None = None

    def __post_init__(self) -> None:
        claim_ids = {claim.claim_id for claim in self.claim_tests}
        if not claim_ids:
            raise ValueError("decision spec requires at least one claim")
        for rule in self.option_rules:
            if not set(rule.all_of).issubset(claim_ids):
                raise ValueError(f"unknown all_of claim in rule {rule.option_id}")
            if not set(rule.none_of).issubset(claim_ids):
                raise ValueError(f"unknown none_of claim in rule {rule.option_id}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "claim_tests": [claim.to_dict() for claim in self.claim_tests],
            "option_rules": [rule.to_dict() for rule in self.option_rules],
            "cannot_determine_option_id": self.cannot_determine_option_id,
        }


@dataclass(frozen=True)
class BoundingBox:
    frame_id: str
    x1: int
    y1: int
    x2: int
    y2: int
    coordinate_space: str = "normalized_1000"

    def __post_init__(self) -> None:
        if self.coordinate_space != "normalized_1000":
            raise ValueError("bbox coordinate space must be normalized_1000")
        if not (0 <= self.x1 < self.x2 <= 1000 and 0 <= self.y1 < self.y2 <= 1000):
            raise ValueError("bbox coordinates must satisfy 0 <= x1 < x2 <= 1000")

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_id": self.frame_id,
            "x1": self.x1,
            "y1": self.y1,
            "x2": self.x2,
            "y2": self.y2,
            "coordinate_space": self.coordinate_space,
        }


@dataclass(frozen=True)
class EvidenceFact:
    fact_id: str
    slot_ids: tuple[str, ...]
    start_seconds: float
    end_seconds: float
    visibility: str
    statement: str
    source_frame_ids: tuple[str, ...]
    view_id: str

    def __post_init__(self) -> None:
        if self.visibility not in VISIBILITIES:
            raise ValueError(f"unsupported visibility: {self.visibility}")
        if self.start_seconds < 0 or self.end_seconds < self.start_seconds:
            raise ValueError("invalid evidence fact time range")
        if not self.fact_id or not self.statement:
            raise ValueError("evidence facts require an id and statement")

    @property
    def kind(self) -> str:
        raise NotImplementedError

    def common_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "fact_id": self.fact_id,
            "slot_ids": list(self.slot_ids),
            "start_seconds": round(self.start_seconds, 6),
            "end_seconds": round(self.end_seconds, 6),
            "visibility": self.visibility,
            "statement": self.statement,
            "source_frame_ids": list(self.source_frame_ids),
            "view_id": self.view_id,
        }


@dataclass(frozen=True)
class StaticFact(EvidenceFact):
    entity: str = ""
    attribute: str = ""
    relation: str = ""
    value: str = ""

    @property
    def kind(self) -> str:
        return "static"

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.common_dict(),
            "entity": self.entity,
            "attribute": self.attribute,
            "relation": self.relation,
            "value": self.value,
        }


@dataclass(frozen=True)
class EventFact(EvidenceFact):
    subject: str = ""
    initial_state: str = ""
    action: str = ""
    object: str = ""
    target: str = ""
    result: str = ""
    order: int | None = None

    @property
    def kind(self) -> str:
        return "event"

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.common_dict(),
            "subject": self.subject,
            "initial_state": self.initial_state,
            "action": self.action,
            "object": self.object,
            "target": self.target,
            "result": self.result,
            "order": self.order,
        }


@dataclass(frozen=True)
class TextFact(EvidenceFact):
    exact_text: str = ""
    uncertain_characters: str = ""
    bbox: BoundingBox | None = None
    consensus_frame_ids: tuple[str, ...] = ()

    @property
    def kind(self) -> str:
        return "text"

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.common_dict(),
            "exact_text": self.exact_text,
            "uncertain_characters": self.uncertain_characters,
            "bbox": self.bbox.to_dict() if self.bbox is not None else None,
            "consensus_frame_ids": list(self.consensus_frame_ids),
        }


Fact = StaticFact | EventFact | TextFact


@dataclass(frozen=True)
class CoverageManifest:
    mode: str
    observed_start_seconds: float
    observed_end_seconds: float
    sample_fps: float | None
    max_temporal_gap_seconds: float | None
    frame_ids: tuple[str, ...]
    shot_ids: tuple[str, ...] = ()
    roi_ids: tuple[str, ...] = ()
    context_only_frame_ids: tuple[str, ...] = ()
    full_span_coverage: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "observed_start_seconds": round(self.observed_start_seconds, 6),
            "observed_end_seconds": round(self.observed_end_seconds, 6),
            "sample_fps": self.sample_fps,
            "max_temporal_gap_seconds": self.max_temporal_gap_seconds,
            "frame_ids": list(self.frame_ids),
            "shot_ids": list(self.shot_ids),
            "roi_ids": list(self.roi_ids),
            "context_only_frame_ids": list(self.context_only_frame_ids),
            "full_span_coverage": self.full_span_coverage,
        }


@dataclass(frozen=True)
class SourceView:
    view_id: str
    kind: str
    frame_ids: tuple[str, ...]
    media_paths: tuple[str, ...]
    span: TimeSpan

    def to_dict(self) -> dict[str, Any]:
        return {
            "view_id": self.view_id,
            "kind": self.kind,
            "frame_ids": list(self.frame_ids),
            "media_paths": list(self.media_paths),
            "span": self.span.to_dict(),
        }


@dataclass(frozen=True)
class EvidencePacket:
    canonical_span: TimeSpan
    facts: tuple[Fact, ...]
    coverage_manifest: CoverageManifest
    missing_slot_ids: tuple[str, ...] = ()
    conflicts: tuple[str, ...] = ()
    source_views: tuple[SourceView, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "canonical_span": self.canonical_span.to_dict(),
            "facts": [fact.to_dict() for fact in self.facts],
            "coverage_manifest": self.coverage_manifest.to_dict(),
            "missing_slot_ids": list(self.missing_slot_ids),
            "conflicts": list(self.conflicts),
            "source_views": [view.to_dict() for view in self.source_views],
        }


@dataclass(frozen=True)
class CandidateObservation:
    candidate_id: str
    locator_rank: int
    target_visible: bool
    supporting_span: TimeSpan
    evidence: EvidencePacket
    visible_anchor: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "locator_rank": self.locator_rank,
            "target_visible": self.target_visible,
            "supporting_span": self.supporting_span.to_dict(),
            "visible_anchor": self.visible_anchor,
            "evidence": self.evidence.to_dict(),
        }


@dataclass(frozen=True)
class LocatorCandidate:
    node_id: str
    visible_anchor: str
    rank: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "visible_anchor": self.visible_anchor,
            "rank": self.rank,
        }


@dataclass(frozen=True)
class RefinementPlan:
    densify_fps: float | None = None
    extend_before_seconds: float = 0.0
    extend_after_seconds: float = 0.0
    crop_boxes: tuple[BoundingBox, ...] = ()
    target_slot_ids: tuple[str, ...] = ()
    target_claim_ids: tuple[str, ...] = ()
    reason: str = ""

    @property
    def actionable(self) -> bool:
        return bool(
            self.densify_fps
            or self.extend_before_seconds
            or self.extend_after_seconds
            or self.crop_boxes
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "densify_fps": self.densify_fps,
            "extend_before_seconds": self.extend_before_seconds,
            "extend_after_seconds": self.extend_after_seconds,
            "crop_boxes": [box.to_dict() for box in self.crop_boxes],
            "target_slot_ids": list(self.target_slot_ids),
            "target_claim_ids": list(self.target_claim_ids),
            "reason": self.reason,
            "actionable": self.actionable,
        }


@dataclass(frozen=True)
class ClaimVerdict:
    claim_id: str
    verdict: str
    evidence_fact_ids: tuple[str, ...] = ()
    verification_frame_ids: tuple[str, ...] = ()
    reason: str = ""

    def __post_init__(self) -> None:
        if self.verdict not in CLAIM_VERDICTS:
            raise ValueError(f"unsupported claim verdict: {self.verdict}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "claim_id": self.claim_id,
            "verdict": self.verdict,
            "evidence_fact_ids": list(self.evidence_fact_ids),
            "verification_frame_ids": list(self.verification_frame_ids),
            "reason": self.reason,
        }


@dataclass(frozen=True)
class OptionVerdict:
    option_id: str
    verdict: str
    entailed_support_count: int
    contradicted_support_count: int
    contradicted_exclusion_count: int
    entailed_exclusion_count: int

    def __post_init__(self) -> None:
        if self.verdict not in OPTION_VERDICTS:
            raise ValueError(f"unsupported option verdict: {self.verdict}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "option_id": self.option_id,
            "verdict": self.verdict,
            "entailed_support_count": self.entailed_support_count,
            "contradicted_support_count": self.contradicted_support_count,
            "contradicted_exclusion_count": self.contradicted_exclusion_count,
            "entailed_exclusion_count": self.entailed_exclusion_count,
        }


@dataclass(frozen=True)
class OptionAssessment:
    """Non-gating evidence summary emitted by a local decision pass."""

    option_id: str
    support_score: int
    contradiction_score: int
    discriminant_ids: tuple[str, ...] = ()
    evidence_fact_ids: tuple[str, ...] = ()
    source_frame_ids: tuple[str, ...] = ()
    reason: str = ""

    def __post_init__(self) -> None:
        if not 0 <= self.support_score <= 3:
            raise ValueError("support_score must be between 0 and 3")
        if not 0 <= self.contradiction_score <= 3:
            raise ValueError("contradiction_score must be between 0 and 3")

    def to_dict(self) -> dict[str, Any]:
        return {
            "option_id": self.option_id,
            "support_score": self.support_score,
            "contradiction_score": self.contradiction_score,
            "discriminant_ids": list(self.discriminant_ids),
            "evidence_fact_ids": list(self.evidence_fact_ids),
            "source_frame_ids": list(self.source_frame_ids),
            "reason": self.reason,
        }


@dataclass(frozen=True)
class ChoiceDecision:
    """A mandatory local MCQ choice plus inspectable, non-gating diagnostics."""

    selected_option_id: str
    option_assessments: tuple[OptionAssessment, ...]
    resolved_discriminant_ids: tuple[str, ...] = ()
    unresolved_discriminant_ids: tuple[str, ...] = ()
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "selected_option_id": self.selected_option_id,
            "option_assessments": [item.to_dict() for item in self.option_assessments],
            "resolved_discriminant_ids": list(self.resolved_discriminant_ids),
            "unresolved_discriminant_ids": list(self.unresolved_discriminant_ids),
            "reason": self.reason,
        }


@dataclass(frozen=True)
class EvidenceGrade:
    """Controller-computed evidence quality that never gates answer emission."""

    level: str
    clear_slot_ids: tuple[str, ...] = ()
    partial_slot_ids: tuple[str, ...] = ()
    missing_slot_ids: tuple[str, ...] = ()
    provenance_errors: tuple[str, ...] = ()
    modality_errors: tuple[str, ...] = ()
    conflicts: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.level not in SUPPORT_LEVELS:
            raise ValueError(f"unsupported evidence support level: {self.level}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "level": self.level,
            "clear_slot_ids": list(self.clear_slot_ids),
            "partial_slot_ids": list(self.partial_slot_ids),
            "missing_slot_ids": list(self.missing_slot_ids),
            "provenance_errors": list(self.provenance_errors),
            "modality_errors": list(self.modality_errors),
            "conflicts": list(self.conflicts),
        }


@dataclass
class ResourceLedger:
    max_model_calls: int
    model_calls: list[dict[str, Any]] = field(default_factory=list)
    unique_frame_ids: set[str] = field(default_factory=set)
    cumulative_frame_views: int = 0

    @property
    def remaining_model_calls(self) -> int:
        return max(0, self.max_model_calls - len(self.model_calls))

    def ensure_model_call(self, *, reserve: int = 0) -> None:
        if reserve < 0:
            raise ValueError("model-call reserve must be non-negative")
        if self.remaining_model_calls <= reserve:
            raise ResourceExhausted(
                f"model call safety limit reached: {self.max_model_calls}; "
                f"{reserve} call(s) reserved"
            )

    def record_model_call(
        self,
        *,
        role: str,
        prompt: str,
        raw_response: str,
        model_metadata: dict[str, Any],
        frames: tuple[FrameRef, ...] = (),
        media_paths: tuple[str, ...] = (),
        prompt_version: str = "p01-v2",
        media_config: dict[str, Any] | None = None,
        sampling: dict[str, Any] | None = None,
    ) -> None:
        new_ids = {frame.id for frame in frames} - self.unique_frame_ids
        self.unique_frame_ids.update(frame.id for frame in frames)
        self.cumulative_frame_views += len(frames)
        self.model_calls.append(
            {
                "call_index": len(self.model_calls) + 1,
                "role": role,
                "prompt_version": prompt_version,
                "prompt": prompt,
                "raw_response": raw_response,
                "model_metadata": dict(model_metadata),
                "frame_ids": [frame.id for frame in frames],
                "frames": [frame.to_dict() for frame in frames],
                "media_paths": list(media_paths),
                "media_config": dict(media_config or {}),
                "sampling": dict(sampling or {}),
                "new_unique_frames": len(new_ids),
            }
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_model_calls": self.max_model_calls,
            "model_call_count": len(self.model_calls),
            "unique_frames": len(self.unique_frame_ids),
            "cumulative_frame_views": self.cumulative_frame_views,
            "input_tokens": sum(
                int(call["model_metadata"].get("input_tokens", 0)) for call in self.model_calls
            ),
            "output_tokens": sum(
                int(call["model_metadata"].get("output_tokens", 0)) for call in self.model_calls
            ),
            "calls": list(self.model_calls),
        }


@dataclass(frozen=True)
class P01Result:
    status: str
    prediction: str | None
    decision_source: str
    support_level: str
    pipeline_outcome: str
    canonical_span: TimeSpan | None
    evidence: EvidencePacket | None
    decision: ChoiceDecision | None = None
    evidence_grade: EvidenceGrade | None = None
    claim_verdicts: tuple[ClaimVerdict, ...] = ()
    option_verdicts: tuple[OptionVerdict, ...] = ()
    missing_facts: tuple[str, ...] = ()
    trace: dict[str, Any] = field(default_factory=dict)
    resources: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.status not in RESULT_STATUSES:
            raise ValueError(f"unsupported P01 result status: {self.status}")
        if self.decision_source not in DECISION_SOURCES:
            raise ValueError(f"unsupported decision source: {self.decision_source}")
        if self.support_level not in SUPPORT_LEVELS:
            raise ValueError(f"unsupported support level: {self.support_level}")
        if self.status == "answered" and not (self.prediction or "").strip():
            raise ValueError("answered P01 results require a non-empty prediction")

    @property
    def output_text(self) -> str:
        return self.prediction or ""

    @property
    def verified_answer(self) -> None:
        """Deprecated v1 field. P01 v2 never claims binary verification."""

        return None

    @property
    def forced_prediction(self) -> str | None:
        """Deprecated v1 compatibility alias for the unified prediction."""

        return self.prediction

    @property
    def prediction_kind(self) -> str:
        """Deprecated v1 compatibility value used by older runners."""

        return "forced" if self.prediction is not None else "none"

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "prediction": self.prediction,
            "decision_source": self.decision_source,
            "support_level": self.support_level,
            "pipeline_outcome": self.pipeline_outcome,
            "verified_answer": self.verified_answer,
            "forced_prediction": self.forced_prediction,
            "prediction_kind": self.prediction_kind,
            "deprecated_fields": [
                "verified_answer",
                "forced_prediction",
                "prediction_kind",
            ],
            "canonical_span": (
                self.canonical_span.to_dict() if self.canonical_span is not None else None
            ),
            "evidence": self.evidence.to_dict() if self.evidence is not None else None,
            "decision": self.decision.to_dict() if self.decision is not None else None,
            "evidence_grade": (
                self.evidence_grade.to_dict() if self.evidence_grade is not None else None
            ),
            "claim_verdicts": [verdict.to_dict() for verdict in self.claim_verdicts],
            "option_verdicts": [verdict.to_dict() for verdict in self.option_verdicts],
            "missing_facts": list(self.missing_facts),
            "trace": self.trace,
            "resources": self.resources,
        }


class ASRAdapter(Protocol):
    def transcribe(
        self,
        video_path: str,
        start: float,
        end: float,
        language: str | None = None,
    ) -> list[dict[str, Any]]: ...


class OCRAdapter(Protocol):
    def transcribe(self, image_path: str, bbox: BoundingBox | None = None) -> str: ...
