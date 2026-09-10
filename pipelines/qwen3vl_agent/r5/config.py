from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any

from qwen3vl_agent.p01.config import P01Config
from qwen3vl_agent.r5.types import R5Budget, finite


@dataclass(frozen=True)
class R5Config:
    media: P01Config = field(default_factory=lambda: P01Config(cache_dir=".cache/qwen3vl_agent/r5"))
    budget: R5Budget = field(default_factory=R5Budget)
    sample_fps: float = 4.0
    core_frames: int = 32
    context_frames: int = 8
    max_frames_per_call: int = 48
    merge_fan_in: int = 4
    max_split_depth: int = 1
    min_split_sec: float = 0.25
    max_facts_per_card: int = 8
    recovery_facts: int = 4
    max_recovery_calls: int = 3
    max_claims: int = 32
    max_statement_chars: int = 512
    max_provider_segments: int = 64
    max_provider_text_chars: int = 16000
    compiler_tokens: int = 2048
    observer_tokens: int = 2048
    recovery_tokens: int = 1024
    merge_tokens: int = 4096
    composer_tokens: int = 4096

    @classmethod
    def from_mapping(cls, data: dict[str, Any] | None) -> R5Config:
        values = dict(data or {})
        removed = set(values) & {"revisit_fps", "audit_batch_cards", "max_source_revisits",
                                 "max_revisit_calls", "audit_tokens"}
        if removed:
            raise ValueError("R5 v3 removed Auditor/revisit settings; remove these fields: "
                             + ", ".join(sorted(removed)))
        if set(values) - {f.name for f in fields(cls)}:
            raise ValueError("unknown R5 configuration field")
        if isinstance(values.get("media"), dict):
            values["media"] = P01Config.from_mapping(values["media"])
        if isinstance(values.get("budget"), dict):
            values["budget"] = R5Budget(**values["budget"])
        result = cls(**values)
        result.validate()
        return result

    def validate(self) -> None:
        self.media.validate()
        for f in fields(self):
            value = getattr(self, f.name)
            if f.type == "int" and (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < (0 if f.name in {"context_frames"} else 1)
            ):
                raise ValueError(f"invalid R5 {f.name}")
        for value in (self.sample_fps, self.min_split_sec):
            if finite(value, minimum=0) == 0:
                raise ValueError("R5 rates and durations must be positive")
        if self.core_frames + 2 * self.context_frames > self.max_frames_per_call:
            raise ValueError("core/context exceed per-call frame cap")
        if self.max_frames_per_call > self.media.caption_refine_max_frames:
            raise ValueError("frame cap exceeds the shared media service")
        if not 2 <= self.merge_fan_in <= 4:
            raise ValueError("R5 permits merge fan-in 2..4")
        if self.max_split_depth != 1:
            raise ValueError("R5 permits only one input-size split")
        if self.budget.visual_deadline_sec >= self.budget.max_elapsed_sec:
            raise ValueError("reserve time after the visual deadline for synthesis")
