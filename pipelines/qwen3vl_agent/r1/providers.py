"""Injection boundary for the separately owned subtitle/ASR service."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Protocol

from qwen3vl_agent.p01.types import TimeSpan


@dataclass(frozen=True)
class ExternalSegment:
    source_id: str
    segment_id: str
    start_sec: float
    end_sec: float
    text: str
    kind: str
    speaker_id: str | None = None
    alignment_status: str = "unknown"

    def __post_init__(self) -> None:
        if (
            not self.source_id
            or not self.segment_id
            or not self.text.strip()
            or self.kind not in {"subtitle", "asr"}
            or not all(math.isfinite(t) for t in (self.start_sec, self.end_sec))
            or self.start_sec < 0
            or self.end_sec <= self.start_sec
        ):
            raise ValueError("invalid external evidence segment")
        if self.alignment_status not in {"aligned", "approximate", "unknown"}:
            raise ValueError("unknown alignment_status")


@dataclass(frozen=True)
class ProviderResult:
    items: tuple[ExternalSegment, ...] = ()
    available: bool = True
    truncated: bool = False
    cost: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


class ExternalEvidenceProvider(Protocol):
    def search(self, query: str, source_id: str, allowed_scope: TimeSpan) -> ProviderResult: ...

    def read(self, source_id: str, span: TimeSpan) -> ProviderResult: ...


class NullEvidenceProvider:
    def search(self, query: str, source_id: str, allowed_scope: TimeSpan) -> ProviderResult:
        return ProviderResult(available=False)

    def read(self, source_id: str, span: TimeSpan) -> ProviderResult:
        return ProviderResult(available=False)
