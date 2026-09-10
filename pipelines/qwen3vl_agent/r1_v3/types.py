"""Append-only observations and local candidate relations; public R1 contracts stay intact."""

from dataclasses import dataclass, field
from typing import Any

from qwen3vl_agent.r1.media import MediaBatch
from qwen3vl_agent.r1.types import CoverageRecord, EvidenceFact, EvidencePacket


@dataclass(frozen=True)
class ObservationRecord:
    record_id: str
    call_id: str
    purpose: str
    raw: str
    batch: MediaBatch
    target: dict[str, Any]
    facts: tuple[EvidenceFact, ...]
    gaps: tuple[dict, ...]
    errors: tuple[dict, ...]
    reviews: tuple[dict, ...]
    coverage: CoverageRecord | None
    existence: str = "unknown"
    absence_basis: str = ""
    task_id: str = ""


@dataclass(frozen=True)
class FactEligibility:
    fact_id: str
    record_id: str
    source_valid: bool
    observation_status: str
    observation_clear: bool
    target_confirmed: bool
    query_fields: tuple[str, ...]
    refuted: bool
    answer_eligible: bool
    reasons: tuple[str, ...]


@dataclass
class QueryBindingTask:
    task_id: str
    candidate_id: str
    origin_record_ids: tuple[str, ...]
    fact_ids: tuple[str, ...]
    field_ids: tuple[str, ...]
    source_frame_ids: tuple[str, ...]
    status: str = "pending"
    attempted: bool = False
    result_record_ids: list[str] = field(default_factory=list)
    reason: str = "query_binding_unresolved"


@dataclass
class TerminalAudit:
    attempt: int
    call_ids: list[str]
    raw: str
    checks: dict[str, bool]
    failures: list[str]
    output_errors: list[str]
    passed: bool
    recovery: str = "not_needed"


@dataclass(frozen=True)
class CandidateLink:
    call_id: str
    left_id: str
    right_id: str
    relation: str
    left_source_ids: tuple[str, ...]
    right_source_ids: tuple[str, ...]
    basis: str
    basis_kind: str
    same_occurrence: bool = False


@dataclass
class V3Packet(EvidencePacket):
    observations: list[ObservationRecord] = field(default_factory=list)
    resolutions: list[dict] = field(default_factory=list)
    bound_fact_ids: list[str] = field(default_factory=list)
    target_record_ids: list[str] = field(default_factory=list)
    active_gaps: list[dict] = field(default_factory=list)
    active_errors: list[dict] = field(default_factory=list)
    aliases: list[str] = field(default_factory=list)
    merged_into: str | None = None
    fact_eligibility: list[FactEligibility] = field(default_factory=list)
    query_binding_tasks: list[QueryBindingTask] = field(default_factory=list)
