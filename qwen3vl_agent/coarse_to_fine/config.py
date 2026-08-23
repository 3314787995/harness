from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class CoarseToFineConfig:
    cache_dir: str = ".cache/qwen3vl_agent"
    sample_fps: float = 1.0
    cache_max_side: int = 768
    cache_jpeg_quality: int = 85
    cache_lru_size: int = 4
    glance_frames: int = 4
    global_frames: int = 32
    initial_windows: int = 6
    select_top_k: int = 2
    split_parts: int = 2
    max_rounds: int = 3
    confidence_threshold: int = 3
    working_set_frames: int = 12
    unique_frame_budget: int = 32
    cumulative_frame_budget: int = 64
    subtitle_max_chars: int = 4_000
    min_window_seconds: float = 1.0
    decision_max_new_tokens: int = 128
    reasoning_max_new_tokens: int = 192
    protocol_repair_attempts: int = 0
    protocol_repair_max_new_tokens: int = 96
    contact_sheet_columns: int = 3

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> CoarseToFineConfig:
        if value is None:
            return cls()
        known = {item.name for item in fields(cls)}
        unknown = sorted(set(value) - known)
        if unknown:
            raise ValueError(f"Unknown coarse_to_fine config fields: {', '.join(unknown)}")
        config = cls(**dict(value))
        config.validate()
        return config

    def validate(self) -> None:
        positive_ints = {
            "cache_max_side": self.cache_max_side,
            "cache_jpeg_quality": self.cache_jpeg_quality,
            "cache_lru_size": self.cache_lru_size,
            "glance_frames": self.glance_frames,
            "global_frames": self.global_frames,
            "initial_windows": self.initial_windows,
            "select_top_k": self.select_top_k,
            "split_parts": self.split_parts,
            "max_rounds": self.max_rounds,
            "working_set_frames": self.working_set_frames,
            "unique_frame_budget": self.unique_frame_budget,
            "cumulative_frame_budget": self.cumulative_frame_budget,
            "decision_max_new_tokens": self.decision_max_new_tokens,
            "reasoning_max_new_tokens": self.reasoning_max_new_tokens,
            "protocol_repair_max_new_tokens": self.protocol_repair_max_new_tokens,
            "contact_sheet_columns": self.contact_sheet_columns,
        }
        for name, number in positive_ints.items():
            if number < 1:
                raise ValueError(f"{name} must be positive")
        if self.sample_fps <= 0:
            raise ValueError("sample_fps must be positive")
        if self.min_window_seconds <= 0:
            raise ValueError("min_window_seconds must be positive")
        if not 0 <= self.confidence_threshold <= 3:
            raise ValueError("confidence_threshold must be between 0 and 3")
        if self.protocol_repair_attempts < 0:
            raise ValueError("protocol_repair_attempts cannot be negative")
        if self.select_top_k > self.initial_windows:
            raise ValueError("select_top_k cannot exceed initial_windows")
        if self.working_set_frames > self.cumulative_frame_budget:
            raise ValueError("working_set_frames cannot exceed cumulative_frame_budget")

    @property
    def resolved_cache_dir(self) -> Path:
        return Path(self.cache_dir).expanduser().resolve()
