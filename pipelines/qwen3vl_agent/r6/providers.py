"""Aligned text only; filter before indexing. Audio is an explicit unavailable capability."""

import json
import re
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Protocol

from qwen3vl_agent.r3.checkpoint import file_digest
from qwen3vl_agent.r4.providers import read_external_file

from .types import ProtocolError, digest, interval


class AudioObservationProvider(Protocol):
    """Future frozen audio observer; may not select answers or widen the supplied scope."""

    def observe(self, *, video_path: str, span: tuple[float, float], target: str) -> dict: ...


class UnavailableAudioProvider:
    available = False

    def observe(self, **kwargs):
        return {
            "status": "MODALITY_UNAVAILABLE",
            "records": [],
            "audio_seconds": 0,
            "reason": "R6 v1 has no actual audio model; ASR is not music evidence",
        }


class TextSources:
    def __init__(self, request, contract):
        self.sources, self.file_hashes, self.issues = {}, {}, []
        for kind in ("subtitle", "asr"):
            filename = getattr(request, kind + "_path")
            if not filename:
                continue
            path = Path(filename)
            sha = file_digest(path)
            self.file_hashes[kind] = sha
            if path.suffix.lower() == ".jsonl":
                rows = [
                    json.loads(line)
                    for line in path.read_text(encoding="utf-8-sig").splitlines()
                    if line.strip()
                ]
            else:
                file = SimpleNamespace(path=str(path), kind=kind, alignment_error_sec=0.0)
                rows = [asdict(s) for s in read_external_file(file, request.video_id)]
            seen = set()
            for index, row in enumerate(rows):
                permitted_fields = {
                    "source_id",
                    "segment_id",
                    "start_sec",
                    "end_sec",
                    "text",
                    "kind",
                    "speaker_id",
                    "alignment_status",
                }
                if not isinstance(row, dict) or set(row) - permitted_fields:
                    raise ProtocolError("text segments contain unsupported metadata")
                if not {"start_sec", "end_sec", "text"} <= row.keys():
                    raise ProtocolError("text requires timestamps and content")
                if row.get("source_id", request.video_id) != request.video_id:
                    raise ProtocolError("text belongs to another video")
                if row.get("kind", kind) != kind:
                    raise ProtocolError("text modality mismatch")
                timing = interval([row["start_sec"], row["end_sec"]])
                segment_id = str(row.get("segment_id", index))
                if segment_id in seen:
                    raise ProtocolError("duplicate text segment ID")
                seen.add(segment_id)
                if not contract.permits_span(timing):
                    self.issues.append({"kind": kind, "index": index, "reason": "outside_scope"})
                    continue
                if row.get("alignment_status", "aligned") != "aligned":
                    self.issues.append(
                        {"kind": kind, "index": index, "reason": "alignment_unknown"}
                    )
                    continue
                if not isinstance(row["text"], str) or not row["text"].strip():
                    raise ProtocolError("empty/non-string text segment")
                # Large cues split into source views with explicit character offsets, not hidden cuts.
                for start in range(0, len(row["text"]), 3000):
                    text = row["text"][start : start + 3000]
                    identity = {
                        "file": sha,
                        "segment": segment_id,
                        "offset": start,
                        "scope": contract.fingerprint,
                        "modality": kind,
                    }
                    sid = "R6T-" + digest(identity)[:24]
                    self.sources[sid] = {
                        "id": sid,
                        "modality": kind,
                        "text": text,
                        "source_time": list(timing),
                        "file_sha256": sha,
                        "segment_id": segment_id,
                        "speaker_id": row.get("speaker_id"),
                        "char_span": [start, start + len(text)],
                        "scope_hash": contract.fingerprint,
                        "source_frame_id": None,
                    }

    def search(self, query, *, span=None):
        tokens = set(re.findall(r"\w+", query.casefold()))
        rows = [
            s
            for s in self.sources.values()
            if span is None
            or max(span[0], s["source_time"][0]) <= min(span[1], s["source_time"][1])
        ]
        return sorted(
            rows,
            key=lambda s: (
                -sum(t in s["text"].casefold() for t in tokens),
                s["source_time"][0],
                s["id"],
            ),
        )
