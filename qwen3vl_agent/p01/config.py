from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class P01Config:
    cache_dir: str = ".cache/qwen3vl_agent/p01"
    index_fps: float = 2.0
    index_max_side: int = 768
    cache_jpeg_quality: int = 85
    cache_lru_size: int = 2
    shot_min_seconds: float = 2.0
    shot_max_seconds: float = 20.0
    shot_change_threshold: float = 0.16

    short_video_threshold_sec: float = 20.0
    locator_branch_factor: int = 12
    locator_beam_width: int = 2
    locator_max_levels: int = 3
    max_candidate_spans: int = 2
    contact_sheet_columns: int = 3
    locator_tiles_per_page: int = 6
    locator_tile_width: int = 480
    locator_tile_height: int = 270
    locator_label_height: int = 36
    locator_sheet_max_pixels: int = 1024 * 32 * 32
    rescue_candidate_spans: int = 2
    global_rescue_max_frames: int = 128
    global_rescue_uniform_fraction: float = 0.5

    static_max_span_sec: float = 20.0
    dynamic_max_span_sec: float = 40.0
    ocr_max_span_sec: float = 20.0
    caption_max_span_sec: float = 50.0
    static_padding_sec: float = 2.0
    dynamic_padding_sec: float = 6.0
    ocr_padding_sec: float = 2.0
    caption_padding_sec: float = 8.0

    static_frames: int = 9
    static_max_frames: int = 16
    dynamic_fps: float = 6.0
    dynamic_refine_fps: float = 8.0
    dynamic_max_frames: int = 72
    dynamic_refine_max_frames: int = 96
    ocr_search_fps: float = 8.0
    ocr_max_frames: int = 16
    caption_fps: float = 4.0
    caption_max_frames: int = 96
    caption_refine_fps: float = 6.0
    caption_refine_max_frames: int = 112
    verification_max_frames: int = 16  # v1 compatibility; unused by v2.
    max_detail_images: int = 6
    decision_max_frames: int = 64

    interval_short_fps: float = 6.0
    interval_medium_fps: float = 3.0
    interval_long_fps: float = 1.5
    interval_short_max_frames: int = 72
    interval_medium_max_frames: int = 90
    interval_long_max_frames: int = 112
    interval_pattern_keyframes: int = 24
    interval_chunk_sec: float = 25.0
    interval_overlap_sec: float = 2.0
    max_interval_chunks: int = 8

    normal_min_pixels: int = 64 * 32 * 32
    normal_max_pixels: int = 256 * 32 * 32
    normal_total_pixels: int = 12288 * 32 * 32
    image_min_pixels: int = 128 * 32 * 32
    image_max_pixels: int = 256 * 32 * 32
    detail_max_pixels: int = 1024 * 32 * 32

    safe_normal_min_pixels: int = 32 * 32 * 32
    safe_normal_max_pixels: int = 192 * 32 * 32
    safe_normal_total_pixels: int = 8192 * 32 * 32
    safe_image_min_pixels: int = 64 * 32 * 32
    safe_image_max_pixels: int = 192 * 32 * 32
    safe_detail_max_pixels: int = 768 * 32 * 32
    safe_locator_sheet_max_pixels: int = 512 * 32 * 32
    oom_retry_attempts: int = 1

    max_refinement_rounds: int = 1
    max_rescue_rounds: int = 1
    protocol_repair_attempts: int = 1
    max_model_calls: int = 20
    terminal_call_reserve: int = 2

    observation_compiler_max_new_tokens: int = 512
    hypothesis_compiler_max_new_tokens: int = 1024
    locator_max_new_tokens: int = 256
    scout_max_new_tokens: int = 1024
    refinement_max_new_tokens: int = 1024
    verifier_max_new_tokens: int = 768
    decision_max_new_tokens: int = 1536
    rescue_locator_max_new_tokens: int = 384
    answer_max_new_tokens: int = 256
    repair_max_new_tokens: int = 1536

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> P01Config:
        if value is None:
            config = cls()
            config.validate()
            return config
        known = {item.name for item in fields(cls)}
        unknown = sorted(set(value) - known)
        if unknown:
            raise ValueError(f"Unknown p01 config fields: {', '.join(unknown)}")
        config = cls(**dict(value))
        config.validate()
        return config

    def validate(self) -> None:
        positive_ints = {
            "index_max_side": self.index_max_side,
            "cache_jpeg_quality": self.cache_jpeg_quality,
            "cache_lru_size": self.cache_lru_size,
            "locator_branch_factor": self.locator_branch_factor,
            "locator_beam_width": self.locator_beam_width,
            "locator_max_levels": self.locator_max_levels,
            "max_candidate_spans": self.max_candidate_spans,
            "contact_sheet_columns": self.contact_sheet_columns,
            "locator_tiles_per_page": self.locator_tiles_per_page,
            "locator_tile_width": self.locator_tile_width,
            "locator_tile_height": self.locator_tile_height,
            "locator_label_height": self.locator_label_height,
            "locator_sheet_max_pixels": self.locator_sheet_max_pixels,
            "rescue_candidate_spans": self.rescue_candidate_spans,
            "global_rescue_max_frames": self.global_rescue_max_frames,
            "static_frames": self.static_frames,
            "static_max_frames": self.static_max_frames,
            "dynamic_max_frames": self.dynamic_max_frames,
            "dynamic_refine_max_frames": self.dynamic_refine_max_frames,
            "ocr_max_frames": self.ocr_max_frames,
            "caption_max_frames": self.caption_max_frames,
            "caption_refine_max_frames": self.caption_refine_max_frames,
            "verification_max_frames": self.verification_max_frames,
            "max_detail_images": self.max_detail_images,
            "decision_max_frames": self.decision_max_frames,
            "max_interval_chunks": self.max_interval_chunks,
            "interval_short_max_frames": self.interval_short_max_frames,
            "interval_medium_max_frames": self.interval_medium_max_frames,
            "interval_long_max_frames": self.interval_long_max_frames,
            "interval_pattern_keyframes": self.interval_pattern_keyframes,
            "normal_min_pixels": self.normal_min_pixels,
            "normal_max_pixels": self.normal_max_pixels,
            "normal_total_pixels": self.normal_total_pixels,
            "image_min_pixels": self.image_min_pixels,
            "image_max_pixels": self.image_max_pixels,
            "detail_max_pixels": self.detail_max_pixels,
            "safe_normal_min_pixels": self.safe_normal_min_pixels,
            "safe_normal_max_pixels": self.safe_normal_max_pixels,
            "safe_normal_total_pixels": self.safe_normal_total_pixels,
            "safe_image_min_pixels": self.safe_image_min_pixels,
            "safe_image_max_pixels": self.safe_image_max_pixels,
            "safe_detail_max_pixels": self.safe_detail_max_pixels,
            "safe_locator_sheet_max_pixels": self.safe_locator_sheet_max_pixels,
            "oom_retry_attempts": self.oom_retry_attempts,
            "max_model_calls": self.max_model_calls,
            "terminal_call_reserve": self.terminal_call_reserve,
            "observation_compiler_max_new_tokens": (self.observation_compiler_max_new_tokens),
            "hypothesis_compiler_max_new_tokens": self.hypothesis_compiler_max_new_tokens,
            "locator_max_new_tokens": self.locator_max_new_tokens,
            "scout_max_new_tokens": self.scout_max_new_tokens,
            "refinement_max_new_tokens": self.refinement_max_new_tokens,
            "verifier_max_new_tokens": self.verifier_max_new_tokens,
            "decision_max_new_tokens": self.decision_max_new_tokens,
            "rescue_locator_max_new_tokens": self.rescue_locator_max_new_tokens,
            "answer_max_new_tokens": self.answer_max_new_tokens,
            "repair_max_new_tokens": self.repair_max_new_tokens,
        }
        for name, number in positive_ints.items():
            if number < 1:
                raise ValueError(f"{name} must be positive")

        positive_floats = {
            "index_fps": self.index_fps,
            "shot_min_seconds": self.shot_min_seconds,
            "shot_max_seconds": self.shot_max_seconds,
            "short_video_threshold_sec": self.short_video_threshold_sec,
            "static_max_span_sec": self.static_max_span_sec,
            "dynamic_max_span_sec": self.dynamic_max_span_sec,
            "ocr_max_span_sec": self.ocr_max_span_sec,
            "caption_max_span_sec": self.caption_max_span_sec,
            "dynamic_fps": self.dynamic_fps,
            "dynamic_refine_fps": self.dynamic_refine_fps,
            "ocr_search_fps": self.ocr_search_fps,
            "caption_fps": self.caption_fps,
            "caption_refine_fps": self.caption_refine_fps,
            "interval_short_fps": self.interval_short_fps,
            "interval_medium_fps": self.interval_medium_fps,
            "interval_long_fps": self.interval_long_fps,
            "interval_chunk_sec": self.interval_chunk_sec,
        }
        for name, number in positive_floats.items():
            if number <= 0:
                raise ValueError(f"{name} must be positive")

        non_negative = {
            "static_padding_sec": self.static_padding_sec,
            "dynamic_padding_sec": self.dynamic_padding_sec,
            "ocr_padding_sec": self.ocr_padding_sec,
            "caption_padding_sec": self.caption_padding_sec,
            "interval_overlap_sec": self.interval_overlap_sec,
            "protocol_repair_attempts": self.protocol_repair_attempts,
            "oom_retry_attempts": self.oom_retry_attempts,
        }
        for name, number in non_negative.items():
            if number < 0:
                raise ValueError(f"{name} must be non-negative")

        if not 0 <= self.shot_change_threshold <= 1:
            raise ValueError("shot_change_threshold must be between 0 and 1")
        if self.shot_max_seconds <= self.shot_min_seconds:
            raise ValueError("shot_max_seconds must exceed shot_min_seconds")
        if self.locator_beam_width > self.locator_branch_factor:
            raise ValueError("locator_beam_width cannot exceed locator_branch_factor")
        if self.max_candidate_spans > self.locator_beam_width:
            raise ValueError("max_candidate_spans cannot exceed locator_beam_width")
        if not 0 < self.global_rescue_uniform_fraction <= 1:
            raise ValueError("global_rescue_uniform_fraction must be in (0, 1]")
        if self.static_frames > self.static_max_frames:
            raise ValueError("static_frames cannot exceed static_max_frames")
        if self.interval_overlap_sec >= self.interval_chunk_sec:
            raise ValueError("interval overlap must be smaller than chunk size")
        if self.normal_min_pixels > self.normal_max_pixels:
            raise ValueError("normal_min_pixels cannot exceed normal_max_pixels")
        if self.image_min_pixels > self.image_max_pixels:
            raise ValueError("image_min_pixels cannot exceed image_max_pixels")
        if self.safe_normal_min_pixels > self.safe_normal_max_pixels:
            raise ValueError("safe_normal_min_pixels cannot exceed safe_normal_max_pixels")
        if self.safe_image_min_pixels > self.safe_image_max_pixels:
            raise ValueError("safe_image_min_pixels cannot exceed safe_image_max_pixels")
        if self.max_refinement_rounds != 1:
            raise ValueError("P01 requires exactly one refinement round")
        if self.max_rescue_rounds != 1:
            raise ValueError("P01 v2 requires exactly one rescue round")
        if self.terminal_call_reserve >= self.max_model_calls:
            raise ValueError("terminal_call_reserve must be smaller than max_model_calls")

    @property
    def resolved_cache_dir(self) -> Path:
        return Path(self.cache_dir).expanduser().resolve()

    def span_limit(self, mode: str) -> float:
        values = {
            "static_visual": self.static_max_span_sec,
            "dynamic_action": self.dynamic_max_span_sec,
            "ocr": self.ocr_max_span_sec,
            "subscene_caption": self.caption_max_span_sec,
        }
        try:
            return values[mode]
        except KeyError as exc:
            raise ValueError(f"unknown P01 observation mode: {mode}") from exc

    def padding(self, mode: str) -> float:
        values = {
            "static_visual": self.static_padding_sec,
            "dynamic_action": self.dynamic_padding_sec,
            "ocr": self.ocr_padding_sec,
            "subscene_caption": self.caption_padding_sec,
        }
        try:
            return values[mode]
        except KeyError as exc:
            raise ValueError(f"unknown P01 observation mode: {mode}") from exc

    def role_max_new_tokens(self, role: str) -> int:
        values = {
            "observation_compiler": self.observation_compiler_max_new_tokens,
            "hypothesis_compiler": self.hypothesis_compiler_max_new_tokens,
            "locator": self.locator_max_new_tokens,
            "candidate_scout": self.scout_max_new_tokens,
            "interval_scout": self.scout_max_new_tokens,
            "rescue_scout": self.scout_max_new_tokens,
            "refinement_extractor": self.refinement_max_new_tokens,
            "verifier": self.verifier_max_new_tokens,
            "initial_decision": self.decision_max_new_tokens,
            "final_decision": self.decision_max_new_tokens,
            "rescue_locator": self.rescue_locator_max_new_tokens,
            "answer_composer": self.answer_max_new_tokens,
            "protocol_repair": self.repair_max_new_tokens,
        }
        return values.get(role, self.scout_max_new_tokens)
