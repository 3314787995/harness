"""Compatibility adapter for separately owned subtitle/ASR providers."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any

from qwen3vl_agent.p01.types import TimeSpan
from qwen3vl_agent.r1.providers import (
    ExternalEvidenceProvider,
    ExternalSegment,
    NullEvidenceProvider,
    ProviderResult,
)
from qwen3vl_agent.r3.config import R3Config
from qwen3vl_agent.r3.runtime import RunContext


@dataclass(frozen=True)
class TemporalProviderResult(ProviderResult):
    """Optional extension; old ProviderResult consumers remain source-compatible."""

    coverage_status: str = "unknown"
    covered_intervals: tuple[tuple[float, float], ...] = ()
    alignment_error_sec: float | None = None


@dataclass
class ReadEvidence:
    items: list[ExternalSegment] = field(default_factory=list)
    coverage_status: str = "unknown"
    issues: list[str] = field(default_factory=list)
    alignment_error_sec: float | None = None


class ProviderAdapter:
    def __init__(
        self,
        provider: ExternalEvidenceProvider | None,
        config: R3Config,
        context: RunContext,
        source_id: str,
        allowed: TimeSpan,
        modalities: tuple[str, ...],
    ) -> None:
        self.provider = provider or NullEvidenceProvider()
        self.config, self.context = config, context
        self.source_id, self.allowed, self.modalities = source_id, allowed, set(modalities)

    def fetch(self, span: TimeSpan, *, query: str | None = None) -> ReadEvidence:
        ctx = self.context
        if not self.modalities & {"subtitle", "asr"}:
            return ReadEvidence(issues=["external_modality_not_permitted"])
        if len(ctx.provider_calls) >= ctx.budget.max_provider_calls:
            return ReadEvidence(issues=["provider_budget_limited"])
        bounded = TimeSpan(
            max(self.allowed.start_seconds, span.start_seconds),
            min(self.allowed.end_seconds, span.end_seconds),
        )
        record: dict[str, Any] = {
            "method": "search" if query else "read",
            "query": query,
            "source_id": self.source_id,
            "span": [bounded.start_seconds, bounded.end_seconds],
            "status": "started",
        }
        ctx.provider_calls.append(record)
        ctx.changed()
        try:
            result = (
                self.provider.search(query, self.source_id, bounded)
                if query
                else self.provider.read(self.source_id, bounded)
            )
            issues = []
            if not result.available:
                issues.append("dependency_missing")
            if result.error:
                issues.append("provider_error:" + str(result.error))
            if result.truncated:
                issues.append("provider_truncated")
            selected, chars, seen = [], 0, set()
            for segment in result.items:
                if not isinstance(segment, ExternalSegment):
                    issues.append("invalid_provider_segment")
                    continue
                if (
                    segment.source_id != self.source_id
                    or segment.kind not in self.modalities
                    or segment.start_sec < self.allowed.start_seconds
                    or segment.end_sec > self.allowed.end_seconds
                    or segment.end_sec <= bounded.start_seconds
                    or segment.start_sec >= bounded.end_seconds
                ):
                    issues.append("provider_source_or_scope_violation")
                    continue
                key = (segment.source_id, segment.segment_id)
                if key in seen:
                    continue
                if (
                    len(selected) >= self.config.max_provider_segments
                    or chars + len(segment.text) > self.config.max_provider_text_chars
                ):
                    issues.append("provider_truncated")
                    break
                seen.add(key)
                selected.append(segment)
                chars += len(segment.text)
            coverage = "unknown"
            alignment_error = getattr(result, "alignment_error_sec", None)
            if alignment_error is not None and (
                isinstance(alignment_error, bool)
                or not isinstance(alignment_error, (int, float))
                or not math.isfinite(alignment_error)
                or alignment_error < 0
            ):
                issues.append("invalid_provider_alignment_error")
                alignment_error = None
            if (
                not query
                and not issues
                and getattr(result, "coverage_status", "unknown") == "complete"
            ):
                from qwen3vl_agent.r3.planning import covered

                intervals = list(getattr(result, "covered_intervals", ()))
                valid = all(
                    isinstance(interval, (list, tuple))
                    and len(interval) == 2
                    and all(
                        isinstance(t, (int, float)) and not isinstance(t, bool) and math.isfinite(t)
                        for t in interval
                    )
                    and self.allowed.start_seconds
                    <= interval[0]
                    < interval[1]
                    <= self.allowed.end_seconds
                    for interval in intervals
                )
                if not valid:
                    issues.append("invalid_provider_coverage")
                elif covered((bounded.start_seconds, bounded.end_seconds), intervals):
                    coverage = "complete"
            record.update(
                status="returned",
                cost=result.cost,
                issues=issues,
                coverage_status=coverage,
                items=[asdict(s) for s in selected],
            )
            return ReadEvidence(selected, coverage, issues, alignment_error)
        except (RuntimeError, ValueError, OSError, TypeError, AttributeError) as exc:
            record.update(status="failed", error=str(exc))
            return ReadEvidence(issues=["provider_error:" + str(exc)])
        finally:
            ctx.changed()


__all__ = [
    "ExternalEvidenceProvider",
    "ExternalSegment",
    "NullEvidenceProvider",
    "ProviderAdapter",
    "ProviderResult",
    "TemporalProviderResult",
]
