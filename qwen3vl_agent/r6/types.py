"""Public, answer-blind contracts for source-grounded relationship reasoning."""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import re
from dataclasses import asdict, dataclass, field

VERSION = "r6-evidence-relations/1.0"
STOPS = (
    "EVIDENCE_SUFFICIENT",
    "BUDGET_EXHAUSTED",
    "NO_PROGRESS",
    "MODALITY_UNAVAILABLE",
    "INPUT_AMBIGUITY",
    "TOOL_FAILURE",
)
MODES = ("pipeline", "direct", "captions", "question_only")
MODALITIES = ("video", "subtitle", "asr", "audio")
RELATIONS = (
    "causal_support",
    "stated_reason",
    "motive",
    "social_relation",
    "reveals",
    "theme_mapping",
    "semantic_consistency",
    "rule_violation",
)


class ProtocolError(ValueError):
    pass


class BudgetExhausted(RuntimeError):
    pass


class ModelFailure(RuntimeError):
    pass


class ContextOverflow(ModelFailure):
    pass


def plain(value):
    return json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))


def digest(value):
    return hashlib.sha256(
        json.dumps(plain(value), sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def finite(value):
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)


def interval(value, *, point=False):
    if not isinstance(value, (list, tuple)) or len(value) != 2 or not all(map(finite, value)):
        raise ProtocolError("interval requires two finite numbers")
    if value[0] < 0 or value[1] < value[0] or (value[0] == value[1] and not point):
        raise ProtocolError("invalid time interval")
    return tuple(map(float, value))


@dataclass(frozen=True)
class Choice:
    label: str
    text: str


def normalize_choices(values):
    if isinstance(values, dict):
        values = [Choice(k, v) for k, v in values.items()]
    output = []
    for i, value in enumerate(values):
        if isinstance(value, str):
            match = re.match(r"^([A-Za-z0-9]+)[.):：]\s+(.+)$", value, re.DOTALL)
            label, text = match.groups() if match else (chr(65 + i), value)
        elif isinstance(value, dict) and set(value) == {"label", "text"}:
            label, text = value["label"], value["text"]
        elif isinstance(value, Choice):
            label, text = value.label, value.text
        else:
            raise ProtocolError("choices require label/text pairs or strings")
        if not all(isinstance(v, str) and v.strip() for v in (label, text)):
            raise ProtocolError("empty label or choice text")
        output.append(Choice(label, text))
    if len(output) < 2 or len({c.label for c in output}) != len(output):
        raise ProtocolError("R6 v1 requires at least two distinct choice labels")
    return tuple(output)


@dataclass(frozen=True)
class R6Request:
    video_path: str
    question: str
    choices: tuple[Choice, ...]
    request_id: str = "r6-request"
    video_id: str = "video"
    allowed_intervals: tuple[tuple[float, float], ...] = ()
    reference_scope: tuple[float, float] | None = None
    history_cutoff: float | None = None
    allowed_modalities: tuple[str, ...] = ("video",)
    subtitle_path: str | None = None
    asr_path: str | None = None
    subtitle_policy: str = "disabled"
    cross_question_cache_policy: str = "isolated"
    answer_protocol: str = "forced_choice"
    subtype: str = "auto"
    mode: str = "pipeline"
    checkpoint_path: str | None = None
    resume: bool = False

    def __post_init__(self):
        for name in ("video_path", "question", "request_id", "video_id"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise ProtocolError(f"{name} is required")
        object.__setattr__(self, "choices", normalize_choices(self.choices))
        spans = tuple(sorted(interval(s) for s in self.allowed_intervals))
        if any(a[1] > b[0] for a, b in itertools.pairwise(spans)):
            raise ProtocolError("allowed intervals overlap")
        object.__setattr__(self, "allowed_intervals", spans)
        if self.reference_scope is not None:
            object.__setattr__(self, "reference_scope", interval(self.reference_scope))
        if self.history_cutoff is not None and (
            not finite(self.history_cutoff) or self.history_cutoff <= 0
        ):
            raise ProtocolError("history cutoff must be positive")
        modalities = tuple(self.allowed_modalities)
        if not modalities or set(modalities) - set(MODALITIES):
            raise ProtocolError("unsupported modalities")
        object.__setattr__(self, "allowed_modalities", modalities)
        if self.subtitle_policy not in {"disabled", "aligned_only"}:
            raise ProtocolError("unsupported text policy")
        for kind in ("subtitle", "asr"):
            if getattr(self, kind + "_path") and (
                kind not in modalities or self.subtitle_policy != "aligned_only"
            ):
                raise ProtocolError("external text requires modality permission and aligned_only")
        if self.cross_question_cache_policy not in {"isolated", "sources_only"}:
            raise ProtocolError("only raw sources may be shared across questions")
        if self.answer_protocol not in {"forced_choice", "allow_abstention"}:
            raise ProtocolError("unsupported answer protocol")
        if self.subtype not in {"auto", "S1", "S2", "S3", "S4", "S5", "S6"}:
            raise ProtocolError("unsupported discussion subtype")
        if self.mode not in MODES or not isinstance(self.resume, bool):
            raise ProtocolError("invalid mode/resume")
        if self.resume and not self.checkpoint_path:
            raise ProtocolError("resume requires checkpoint_path")


@dataclass(frozen=True)
class InputContract:
    allowed_intervals: tuple[tuple[float, float], ...]
    reference_scope: tuple[float, float] | None
    history_cutoff: float | None
    allowed_modalities: tuple[str, ...]
    subtitle_policy: str
    cross_question_cache_policy: str
    answer_protocol: str
    video_id: str
    media_hash: str

    @classmethod
    def resolve(cls, request, duration, media_hash):
        if not finite(duration) or duration <= 0:
            raise ProtocolError("invalid media duration")
        end = min(duration, request.history_cutoff or duration)
        original = request.allowed_intervals or ((0.0, duration),)
        if any(b > duration + 1e-6 for _, b in original):
            raise ProtocolError("allowed interval exceeds media duration")
        spans = tuple((a, min(b, end)) for a, b in original if a < end)
        if not spans:
            raise ProtocolError("empty permitted evidence scope")
        return cls(
            spans,
            request.reference_scope,
            request.history_cutoff,
            request.allowed_modalities,
            request.subtitle_policy,
            request.cross_question_cache_policy,
            request.answer_protocol,
            request.video_id,
            media_hash,
        )

    def permits(self, t):
        return finite(t) and any(a <= t <= b for a, b in self.allowed_intervals)

    def permits_span(self, span):
        a, b = interval(span, point=True)
        return any(lo <= a <= b <= hi for lo, hi in self.allowed_intervals)

    @property
    def fingerprint(self):
        return digest(asdict(self))


@dataclass
class R6Result:
    request_id: str
    prediction: str | None
    evidence_status: str
    forced_choice: bool
    technical_fallback: bool
    stop_reason: str
    pending_gaps: list[dict] = field(default_factory=list)
    source_ids: list[str] = field(default_factory=list)
    costs: dict = field(default_factory=dict)
    trace: dict = field(default_factory=dict)
    pipeline_id: str = VERSION

    @property
    def text(self):
        return self.prediction or ""

    def to_dict(self):
        return plain(asdict(self))
