from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from typing import Any

from qwen3vl_agent.p01.config import P01Config


@dataclass(frozen=True)
class R1Config:
    media: P01Config = field(default_factory=lambda: P01Config(cache_dir=".cache/qwen3vl_agent/r1"))
    max_candidates: int = 2
    max_refinements: int = 2
    max_relocations: int = 1
    overlap_sec: float = 2.0
    compiler_tokens: int = 1024
    observer_tokens: int = 3072
    final_tokens: int = 1536
    max_provider_segments: int = 64
    max_provider_text_chars: int = 16000
    # 16 frames at 8 fps cannot sustain a two-second overlap; preserve density explicitly.
    ocr_batch_max_frames: int = 32

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> R1Config:
        data = dict(value or {})
        unknown = set(data) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"unknown R1 config fields: {sorted(unknown)}")
        if isinstance(data.get("media"), Mapping):
            data["media"] = P01Config.from_mapping(data["media"])
        result = cls(**data)
        result.validate()
        return result

    def validate(self) -> None:
        if not isinstance(self.media, P01Config):
            raise TypeError("R1 media must be a P01Config or a configuration mapping")
        self.media.validate()
        for name in (
            "max_candidates",
            "compiler_tokens",
            "observer_tokens",
            "final_tokens",
            "max_provider_segments",
            "max_provider_text_chars",
            "ocr_batch_max_frames",
        ):
            if (
                isinstance(getattr(self, name), bool)
                or not isinstance(getattr(self, name), int)
                or getattr(self, name) < 1
            ):
                raise ValueError(f"invalid R1 {name}")
        if self.max_candidates > 2:
            raise ValueError("R1 observes at most two candidates per localization round")
        if not (
            self.media.ocr_search_fps * self.overlap_sec + 5
            < self.ocr_batch_max_frames
            <= self.media.caption_refine_max_frames
        ):
            raise ValueError("OCR batch cap must preserve two-second overlap within the media cap")
        if (
            any(
                isinstance(v, bool) or not isinstance(v, int)
                for v in (self.max_refinements, self.max_relocations)
            )
            or not 0 <= self.max_refinements <= 2
            or not 0 <= self.max_relocations <= 1
        ):
            raise ValueError("R1 allows at most two refinements and one relocation")
        for fps, cap in (
            (self.media.dynamic_fps, self.media.dynamic_max_frames),
            (self.media.dynamic_refine_fps, self.media.dynamic_refine_max_frames),
            (self.media.caption_fps, self.media.caption_max_frames),
            (self.media.caption_refine_fps, self.media.caption_refine_max_frames),
            (self.media.interval_medium_fps, self.media.interval_medium_max_frames),
        ):
            if not math.isfinite(fps) or cap - 5 <= fps * self.overlap_sec:
                raise ValueError("media sampling must fit a batch with two-second overlap")
        if self.overlap_sec != 2.0:
            raise ValueError("R1/1.0 uses a two-second overlap")
