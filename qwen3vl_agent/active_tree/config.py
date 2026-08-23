from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ActiveTreeConfig:
    """Configuration for the training-free active evidence tree."""

    cache_dir: str = ".cache/qwen3vl_agent"
    sample_fps: float = 1.0
    cache_max_side: int = 768
    cache_jpeg_quality: int = 85
    cache_lru_size: int = 4

    scene_min_seconds: float = 4.0
    scene_max_seconds: float = 20.0
    scene_change_threshold: float = 0.16
    subtitle_gap_seconds: float = 3.0
    min_children: int = 3
    max_children: int = 6
    storyboard_frames: int = 4
    contact_sheet_columns: int = 4

    inspect_frames: int = 8
    motion_frames: int = 12
    detail_frames: int = 4
    compare_frames_per_node: int = 4
    subtitle_max_chars: int = 4_000
    breadth_subtitle_chars_per_node: int = 500

    min_active_observations: int = 1
    max_search_observations: int = 6
    max_repair_observations: int = 2
    max_stagnant_observations: int = 2
    max_navigation_actions: int = 4
    max_verification_repairs: int = 2
    protocol_repair_attempts: int = 1
    max_model_calls: int = 40
    frontier_max_nodes: int = 18
    allow_alignment_adjudication: bool = True
    allow_temporal_adjudication: bool = True
    enable_subtitle_retrieval_proposal: bool = True
    alignment_min_phrase_tokens: int = 2
    alignment_min_margin: float = 3.0
    temporal_min_gap_seconds: float = 0.5

    compiler_max_new_tokens: int = 256
    breadth_max_new_tokens: int = 512
    discriminator_max_new_tokens: int = 256
    planner_max_new_tokens: int = 256
    observer_max_new_tokens: int = 320
    verifier_max_new_tokens: int = 256

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> ActiveTreeConfig:
        if value is None:
            config = cls()
            config.validate()
            return config
        known = {item.name for item in fields(cls)}
        unknown = sorted(set(value) - known)
        if unknown:
            raise ValueError(f"Unknown active_tree config fields: {', '.join(unknown)}")
        config = cls(**dict(value))
        config.validate()
        return config

    def validate(self) -> None:
        positive_ints = {
            "cache_max_side": self.cache_max_side,
            "cache_jpeg_quality": self.cache_jpeg_quality,
            "cache_lru_size": self.cache_lru_size,
            "min_children": self.min_children,
            "max_children": self.max_children,
            "storyboard_frames": self.storyboard_frames,
            "contact_sheet_columns": self.contact_sheet_columns,
            "inspect_frames": self.inspect_frames,
            "motion_frames": self.motion_frames,
            "detail_frames": self.detail_frames,
            "compare_frames_per_node": self.compare_frames_per_node,
            "subtitle_max_chars": self.subtitle_max_chars,
            "breadth_subtitle_chars_per_node": self.breadth_subtitle_chars_per_node,
            "min_active_observations": self.min_active_observations,
            "max_search_observations": self.max_search_observations,
            "max_repair_observations": self.max_repair_observations,
            "max_stagnant_observations": self.max_stagnant_observations,
            "max_navigation_actions": self.max_navigation_actions,
            "max_verification_repairs": self.max_verification_repairs,
            "max_model_calls": self.max_model_calls,
            "frontier_max_nodes": self.frontier_max_nodes,
            "alignment_min_phrase_tokens": self.alignment_min_phrase_tokens,
            "compiler_max_new_tokens": self.compiler_max_new_tokens,
            "breadth_max_new_tokens": self.breadth_max_new_tokens,
            "discriminator_max_new_tokens": self.discriminator_max_new_tokens,
            "planner_max_new_tokens": self.planner_max_new_tokens,
            "observer_max_new_tokens": self.observer_max_new_tokens,
            "verifier_max_new_tokens": self.verifier_max_new_tokens,
        }
        for name, number in positive_ints.items():
            if number < 1:
                raise ValueError(f"{name} must be positive")
        if self.protocol_repair_attempts < 0:
            raise ValueError("protocol_repair_attempts must be non-negative")
        if self.sample_fps <= 0:
            raise ValueError("sample_fps must be positive")
        if self.scene_min_seconds <= 0:
            raise ValueError("scene_min_seconds must be positive")
        if self.scene_max_seconds <= self.scene_min_seconds:
            raise ValueError("scene_max_seconds must exceed scene_min_seconds")
        if not 0 <= self.scene_change_threshold <= 1:
            raise ValueError("scene_change_threshold must be between 0 and 1")
        if self.subtitle_gap_seconds < 0:
            raise ValueError("subtitle_gap_seconds must be non-negative")
        if self.alignment_min_margin < 0:
            raise ValueError("alignment_min_margin must be non-negative")
        if self.temporal_min_gap_seconds < 0:
            raise ValueError("temporal_min_gap_seconds must be non-negative")
        if self.min_children > self.max_children:
            raise ValueError("min_children cannot exceed max_children")
        if self.max_search_observations < self.min_active_observations:
            raise ValueError("max_search_observations cannot be below min_active_observations")

    @property
    def resolved_cache_dir(self) -> Path:
        return Path(self.cache_dir).expanduser().resolve()
