"""Read-only consumption of separately produced subtitle/ASR evidence."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from qwen3vl_agent.p01.types import TimeSpan
from qwen3vl_agent.r1.providers import (
    ExternalEvidenceProvider,
    ExternalSegment,
    NullEvidenceProvider,
    ProviderResult,
)
from qwen3vl_agent.r3.planning import covered
from qwen3vl_agent.r4.types import ExternalFile, finite, interval


@dataclass(frozen=True)
class InventoryProviderResult(ProviderResult):
    coverage_status: str = "unknown"
    covered_intervals: tuple[tuple[float, float], ...] = ()
    unresolved_intervals: tuple[tuple[float, float], ...] = ()
    alignment_error_sec: float | None = None
    provider_version: str = "unknown"


def _clock(text: str) -> float:
    parts = text.replace(",", ".").split(":")
    if len(parts) not in {2, 3}:
        raise ValueError("invalid SRT/VTT timestamp")
    values = [float(p) for p in parts]
    return sum(v * 60**i for i, v in enumerate(reversed(values)))


def read_external_file(file: ExternalFile, source_id: str) -> list[ExternalSegment]:
    path = Path(file.path)
    text = path.read_text(encoding="utf-8-sig")
    rows = []
    if path.suffix.lower() == ".json":
        value = json.loads(text)
        rows = value.get("items", value.get("segments", [])) if isinstance(value, dict) else value
        if not isinstance(rows, list):
            raise ValueError("JSON evidence must contain a segment list")
    elif path.suffix.lower() in {".srt", ".vtt"}:
        for block in re.split(r"\n\s*\n", text.replace("\r\n", "\n")):
            lines = block.strip().splitlines()
            if not lines or lines[0].startswith(("WEBVTT", "NOTE", "STYLE", "REGION")):
                continue
            index = next((i for i, line in enumerate(lines) if "-->" in line), None)
            if index is None:
                raise ValueError("malformed subtitle cue")
            match = re.fullmatch(r"\s*(\S+)\s+-->\s+(\S+)(?:\s+.*)?", lines[index])
            if not match:
                raise ValueError("malformed subtitle timing")
            rows.append(
                {
                    "segment_id": f"cue-{len(rows)}",
                    "start_sec": _clock(match[1]),
                    "end_sec": _clock(match[2]),
                    "text": "\n".join(lines[index + 1 :]),
                }
            )
    else:
        raise ValueError("external files must be SRT, VTT or JSON")
    result, identities = [], set()
    for i, row in enumerate(rows):
        data = dict(row)
        if (
            data.get("source_id", source_id) != source_id
            or data.get("kind", file.kind) != file.kind
        ):
            raise ValueError("external file source/modality mismatch")
        data.update(source_id=source_id, kind=file.kind)
        data.setdefault("segment_id", f"segment-{i}")
        data.setdefault(
            "alignment_status", "aligned" if file.alignment_error_sec is not None else "unknown"
        )
        segment = ExternalSegment(**data)
        if segment.segment_id in identities:
            raise ValueError("duplicate external segment ID")
        identities.add(segment.segment_id)
        result.append(segment)
    return result


class FileEvidenceProvider:
    version = "r4-file-provider-v1"

    def __init__(self, source_id: str, files: list[dict[str, Any]], kind: str) -> None:
        self.source_id, self.kind = source_id, kind
        self.files = [ExternalFile(**f) for f in files if f["kind"] == kind]

    def read(self, source_id: str, span: TimeSpan) -> InventoryProviderResult:
        if source_id != self.source_id or not self.files:
            return InventoryProviderResult(available=False)
        segments, coverage, unresolved, errors = [], [], [], []
        for i, file in enumerate(self.files):
            for segment in read_external_file(file, source_id):
                if segment.end_sec > span.start_seconds and segment.start_sec < span.end_seconds:
                    values = asdict(segment)
                    values["segment_id"] = f"{self.kind}-file{i}:{segment.segment_id}"
                    segments.append(ExternalSegment(**values))
            if file.coverage_status == "complete":
                coverage.extend(file.covered_intervals)
            unresolved.extend(file.unresolved_intervals)
            errors.append(file.alignment_error_sec)
        return InventoryProviderResult(
            items=tuple(segments),
            coverage_status="complete"
            if covered((span.start_seconds, span.end_seconds), coverage)
            else "unknown",
            covered_intervals=tuple(coverage),
            unresolved_intervals=tuple(unresolved),
            alignment_error_sec=max(errors)
            if errors and all(e is not None for e in errors)
            else None,
            provider_version=self.version,
            cost={"file_reads": len(self.files)},
        )

    def search(self, query: str, source_id: str, allowed_scope: TimeSpan) -> ProviderResult:
        result = self.read(source_id, allowed_scope)
        return ProviderResult(
            items=tuple(s for s in result.items if query.casefold() in s.text.casefold()),
            available=result.available,
            cost=result.cost,
        )


class ProviderAdapter:
    def __init__(
        self,
        provider: ExternalEvidenceProvider | None,
        source: dict[str, Any],
        config: Any,
        context: Any,
        version_cache: dict[str, Any],
    ) -> None:
        self.provider, self.source = provider, source
        self.config, self.context, self.version_cache = config, context, version_cache

    def fetch(
        self, span: tuple[float, float], kind: str, *, query: str | None = None
    ) -> dict[str, Any]:
        ctx, source = self.context, self.source
        output = {"items": [], "issues": [], "complete": False, "alignment_error_sec": None}
        if kind not in source["available_modalities"]:
            output["issues"].append("dependency_missing:modality_not_permitted")
            return output
        if not any(a <= span[0] < span[1] <= b for a, b in source["allowed"]):
            raise ValueError("provider request outside source permissions")
        if len(ctx.provider_calls) >= ctx.budget.max_provider_calls:
            output["issues"].append("provider_budget_limited")
            return output
        provider = self.provider or (
            FileEvidenceProvider(source["source_id"], source["external_files"], kind)
            if source["external_files"]
            else NullEvidenceProvider()
        )
        record = {
            "method": "search" if query else "read",
            "entry_id": source["entry_id"],
            "span": list(span),
            "kind": kind,
            "status": "started",
        }
        ctx.provider_calls.append(record)
        ctx.changed()
        try:
            result = (
                provider.search(query, source["source_id"], TimeSpan(*span))
                if query
                else provider.read(source["source_id"], TimeSpan(*span))
            )
            version_key = f"provider-version:{source['entry_id']}:{kind}"
            version = getattr(result, "provider_version", None)
            if not query:
                if version_key in self.version_cache and self.version_cache[version_key] != version:
                    raise ValueError("provider version changed during the request")
                self.version_cache[version_key] = version
            if not result.available:
                output["issues"].append("dependency_missing")
            if result.error:
                output["issues"].append("provider_error:" + str(result.error))
            if result.truncated:
                output["issues"].append("provider_truncated")
            error = getattr(result, "alignment_error_sec", None)
            if error is not None:
                finite(error, minimum=0)
            output["alignment_error_sec"] = error
            chars, seen = 0, set()
            for segment in result.items:
                if not isinstance(segment, ExternalSegment):
                    raise TypeError("invalid external segment")
                if segment.kind != kind:
                    continue
                if segment.source_id != source["source_id"]:
                    raise ValueError("external source mismatch")
                margin = error or 0
                if not any(
                    a <= segment.start_sec - margin and segment.end_sec + margin <= b
                    for a, b in source["allowed"]
                ):
                    output["issues"].append("provider_cutoff_or_scope_violation")
                    continue
                if segment.end_sec <= span[0] or segment.start_sec >= span[1]:
                    continue
                key = f"{source['source_id']}:{kind}:{segment.segment_id}"
                values = asdict(segment)
                old = self.version_cache.get(key)
                if old is not None and old != values:
                    raise ValueError("provider changed an existing segment")
                self.version_cache[key] = values
                if key in seen:
                    continue
                seen.add(key)
                if (
                    len(output["items"]) >= self.config.max_provider_segments
                    or chars + len(segment.text) > self.config.max_provider_text_chars
                ):
                    output["issues"].append("provider_truncated")
                    break
                chars += len(segment.text)
                output["items"].append(values)
            intervals = [interval(v) for v in getattr(result, "covered_intervals", ())]
            unresolved = [interval(v) for v in getattr(result, "unresolved_intervals", ())]
            if any(not covered(v, source["allowed"]) for v in intervals):
                # A file may describe a larger, valid transcript; only its permitted intersection
                # is relevant. The actual segment contents were strictly filtered above.
                intervals = [
                    (max(a, x), min(b, y))
                    for a, b in intervals
                    for x, y in source["allowed"]
                    if max(a, x) < min(b, y)
                ]
            if any(a < span[1] and b > span[0] for a, b in unresolved):
                output["issues"].append("provider_unresolved_region")
            output["complete"] = bool(
                not query
                and not output["issues"]
                and getattr(result, "coverage_status", "unknown") == "complete"
                and covered(span, intervals)
            )
            record.update(
                status="returned",
                cost=result.cost,
                provider_version=getattr(result, "provider_version", None),
                **output,
            )
        except (ValueError, TypeError, RuntimeError, OSError, AttributeError) as exc:
            output.update(items=[], complete=False)
            output["issues"].append("provider_error:" + str(exc))
            record.update(status="failed", error=str(exc))
        finally:
            ctx.changed()
        return output


__all__ = [
    "ExternalEvidenceProvider",
    "ExternalSegment",
    "FileEvidenceProvider",
    "InventoryProviderResult",
    "ProviderAdapter",
    "ProviderResult",
]
