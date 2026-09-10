"""Public contracts for preselected, single-video factual synthesis."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any

from qwen3vl_agent.p01.types import TimeSpan
from qwen3vl_agent.r1.types import R1Choice, normalize_choices, span_from
from qwen3vl_agent.r3.types import BudgetExhausted, ProtocolError, R3Budget
from qwen3vl_agent.r4.types import ExternalFile, finite

OPERATIONS = {
    "factual_video_summary",
    "global_activity_synopsis",
    "content_genre_selection",
    "multi_segment_factual_description",
}
MODALITIES = {"video", "screen_text", "subtitle", "asr"}
FACT_KINDS = {"visual_observation", "screen_text", "utterance", "reported_event"}
@dataclass(frozen=True)
class R5Budget(R3Budget):
    max_model_calls: int = 64
    terminal_call_reserve: int = 2
    max_elapsed_sec: int = 600
    visual_deadline_sec: int = 420


@dataclass(frozen=True)
class R5Request:
    video_path: str
    question: str
    request_id: str = "r5-request"
    video_id: str = "video"
    group_id: str | None = None
    native_labels: tuple[str, ...] = ()
    choices: tuple[R1Choice, ...] = ()
    allowed_scope: TimeSpan | None = None
    query_scope: TimeSpan | None = None
    observation_cutoff: float | None = None
    execution_subtype: str | None = None
    available_modalities: tuple[str, ...] = ("video", "screen_text")
    external_files: tuple[ExternalFile, ...] = ()
    output_protocol: str = "auto"
    output_language: str = "same_as_question"
    length_instruction: str = "Concise, preserving the major stages and observed outcome."
    budget: R5Budget = field(default_factory=R5Budget)
    checkpoint_path: str | None = None
    resume: bool = False

    def __post_init__(self) -> None:
        if not self.video_path.strip() or not self.question.strip() or not self.request_id:
            raise ValueError("video_path, question and request_id are required")
        choices = self.choices
        if isinstance(choices, dict):
            choices = [{"label": k, "text": v} for k, v in choices.items()]
        object.__setattr__(self, "choices", normalize_choices(choices))
        for key in ("allowed_scope", "query_scope"):
            if getattr(self, key) is not None:
                object.__setattr__(self, key, span_from(getattr(self, key)))
        if self.observation_cutoff is not None:
            finite(self.observation_cutoff, minimum=0)
        if not self.available_modalities or set(self.available_modalities) - MODALITIES:
            raise ValueError("invalid R5 modalities")
        object.__setattr__(self, "available_modalities", tuple(self.available_modalities))
        object.__setattr__(
            self,
            "external_files",
            tuple(
                f if isinstance(f, ExternalFile) else ExternalFile(**f) for f in self.external_files
            ),
        )
        if any(f.kind not in self.available_modalities for f in self.external_files):
            raise ValueError("external file modality must be explicitly permitted")
        if self.execution_subtype and self.execution_subtype not in OPERATIONS:
            raise ValueError("unknown R5 execution subtype")
        if self.output_protocol not in {"auto", "multiple_choice", "free_text"}:
            raise ValueError("R5 supports multiple_choice and free_text")
        if self.output_protocol == "multiple_choice" and not self.choices:
            raise ValueError("multiple_choice requires choices")
        if self.choices and self.output_protocol == "free_text":
            raise ValueError("choices conflict with free_text")
        if isinstance(self.budget, dict):
            object.__setattr__(self, "budget", R5Budget(**self.budget))
        if self.resume and not self.checkpoint_path:
            raise ValueError("resume requires checkpoint_path")


@dataclass(frozen=True)
class SummarySpec:
    operation: str = "factual_video_summary"
    focus: str = "main content, activity, stages and observed outcome"
    required_modalities: tuple[str, ...] = ()
    unresolved: tuple[str, ...] = ()
    scope_interval: tuple[float, float] | None = None

    @classmethod
    def parse(cls, data: dict[str, Any], request: R5Request) -> SummarySpec:
        if "operation" not in data or "focus" not in data:
            raise ProtocolError("summary operation and focus are required")
        operation = request.execution_subtype or data["operation"]
        if operation not in OPERATIONS:
            raise ProtocolError("invalid summary operation")
        required = tuple(data.get("required_modalities", ()))
        if set(required) - MODALITIES:
            raise ProtocolError("invalid summary modality")
        focus = data["focus"]
        if not isinstance(focus, str) or not focus.strip() or len(focus) > 2000:
            raise ProtocolError("invalid synthesis focus")
        interval = (
            span_from(data["scope_interval"]) if data.get("scope_interval") is not None else None
        )
        return cls(
            operation,
            focus,
            required,
            tuple(str(v) for v in data.get("unresolved", ())),
            (interval.start_seconds, interval.end_seconds) if interval else None,
        )


@dataclass
class SegmentCard:
    card_id: str
    segment_id: str
    core_interval: list[float]
    fact_ids: list[str]
    local_entities: list[dict[str, Any]]
    local_transitions: list[dict[str, Any]]
    observation_call_id: str
    coverage_status: str
    unresolved: list[str] = field(default_factory=list)
    truncated: bool = False
    previous_card_id: str | None = None


@dataclass
class R5Result:
    prediction: str
    execution_subtype: str
    completion_state: str
    support_level: str
    answer_basis: str
    coverage: dict[str, Any]
    evidence_refs: list[str]
    unresolved_items: list[str]
    resources: dict[str, Any]
    trace: dict[str, Any]
    pipeline_id: str = "R5"
    verification_status: str = "not_performed"

    def to_dict(self) -> dict[str, Any]:
        # The live result and its JSON checkpoint have exactly the same public representation.
        return json.loads(json.dumps(asdict(self), ensure_ascii=False, allow_nan=False))


__all__ = [
    "BudgetExhausted",
    "ExternalFile",
    "ProtocolError",
    "R5Budget",
    "R5Request",
    "R5Result",
    "SegmentCard",
    "SummarySpec",
    "TimeSpan",
]
