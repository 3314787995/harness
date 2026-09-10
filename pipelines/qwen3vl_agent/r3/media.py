"""Content-addressed source frames, reusing the existing timestamp-preserving media service."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from qwen3vl_agent.p01.media import SourceFrameStore
from qwen3vl_agent.p01.types import TimeSpan
from qwen3vl_agent.r1.media import MediaBatch, R1Media
from qwen3vl_agent.r3.checkpoint import file_digest


class R3SourceFrameStore(SourceFrameStore):
    def bind_source(self, path: str, source_sha256: str) -> None:
        self._bound_path = Path(path).expanduser().resolve()
        self._bound_digest = source_sha256

    def _source_key(self, source: Path) -> str:
        if source == getattr(self, "_bound_path", None):
            return self._bound_digest
        return file_digest(source)


class R3Media(R1Media):
    def extract(
        self, path: str, span: TimeSpan, timestamps: Any, allowed: TimeSpan, **kwargs: Any
    ) -> MediaBatch:
        batch = super().extract(path, span, timestamps, allowed, **kwargs)
        # Shared readers use a small rounding tolerance. An observation cutoff is
        # an access boundary, so R3 performs a strict check on the returned PTS.
        batch.frames = tuple(
            f
            for f in batch.frames
            if max(allowed.start_seconds, span.start_seconds)
            <= f.timestamp_seconds
            <= min(allowed.end_seconds, span.end_seconds)
        )
        if not batch.frames and "decoder_returned_no_permitted_frames" not in batch.errors:
            batch.errors.append("decoder_returned_no_permitted_frames")
        return batch
