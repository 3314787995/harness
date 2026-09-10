"""Per-request model/resource transaction boundary, including failed attempts."""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Any

from qwen3vl_agent.models.base import BaseVideoModel
from qwen3vl_agent.r1.config import R1Config
from qwen3vl_agent.r1.control import BudgetExhausted, ProtocolError, json_object
from qwen3vl_agent.r1.media import MediaBatch, PreparedMedia, R1Media
from qwen3vl_agent.r1.prompts import prompt, repair_prompt
from qwen3vl_agent.r1.types import R1Budget


@dataclass
class RunContext:
    budget: R1Budget
    calls: list[dict[str, Any]] = field(default_factory=list)
    provider_calls: list[dict[str, Any]] = field(default_factory=list)
    frame_exposures: int = 0
    media_pixels: int = 0
    visual_tokens_estimated: int = 0
    shown_frame_ids: set[str] = field(default_factory=set)
    issues: list[str] = field(default_factory=list)
    refinements: int = 0
    relocations: int = 0
    decoded_frames: int = 0

    def can_observe(self) -> bool:
        return len(self.calls) < self.budget.max_model_calls - self.budget.terminal_call_reserve

    def summary(self) -> dict[str, Any]:
        return {
            "limits": asdict(self.budget),
            "model_calls": len(self.calls),
            "provider_call_count": len(self.provider_calls),
            "calls": self.calls,
            "provider_calls": self.provider_calls,
            "frame_exposures": self.frame_exposures,
            "unique_frames": len(self.shown_frame_ids),
            "decoded_frames": self.decoded_frames,
            "media_pixels": self.media_pixels,
            "visual_tokens_estimated": self.visual_tokens_estimated,
            "refinements": self.refinements,
            "relocations": self.relocations,
        }


@dataclass
class CallResult:
    value: Any
    prepared: PreparedMedia | None
    call_id: str
    raw: str


class ModelSession:
    unrepaired_roles = frozenset()
    observer_roles = frozenset({"observe", "binding"})

    def __init__(
        self, model: BaseVideoModel, media: R1Media, config: R1Config, context: RunContext
    ) -> None:
        self.model, self.media, self.config, self.context = model, media, config, context

    def _prompt(self, role: str, payload: dict[str, Any]) -> str:
        return prompt(role, payload)

    def _repair_prompt(self, role: str, raw: str, error: str, payload: dict[str, Any]) -> str:
        return repair_prompt(role, raw, error)

    def _invoke(
        self, role: str, text: str, prepared: PreparedMedia | None, *, terminal: bool, tokens: int
    ) -> tuple[str, str, dict[str, Any]]:
        ctx, budget = self.context, self.context.budget
        reserve = 0 if terminal else budget.terminal_call_reserve
        if len(ctx.calls) >= budget.max_model_calls - reserve:
            raise BudgetExhausted("model_call_budget")
        if len(text) > budget.max_text_chars_per_call:
            raise BudgetExhausted("text_context_budget")
        frames = prepared.frames if prepared else ()
        pixels = prepared.pixels if prepared else 0
        estimated = (pixels + 1023) // 1024
        if ctx.frame_exposures + len(frames) > budget.max_frame_exposures:
            raise BudgetExhausted("frame_exposure_budget")
        if ctx.media_pixels + pixels > budget.max_media_pixels:
            raise BudgetExhausted("media_pixel_budget")
        if (
            budget.max_visual_tokens is not None
            and ctx.visual_tokens_estimated + estimated > budget.max_visual_tokens
        ):
            raise BudgetExhausted("estimated_visual_token_budget")
        record = {
            "call_id": f"call_{len(ctx.calls) + 1:03d}",
            "role": role,
            "prompt": text,
            "source_frame_ids": [f.id for f in frames],
            "media_pixels": pixels,
            "estimated_visual_tokens": estimated,
            "status": "started",
            "quality_limited": bool(prepared and prepared.quality_limited),
            "media_kind": prepared.kind if prepared else "text",
        }
        ctx.calls.append(record)
        ctx.frame_exposures += len(frames)
        ctx.media_pixels += pixels
        ctx.visual_tokens_estimated += estimated
        ctx.shown_frame_ids.update(f.id for f in frames)
        content = [*(prepared.parts if prepared else []), {"type": "text", "text": text}]
        started = time.monotonic()
        try:
            output = self.model.generate(
                [{"role": "user", "content": content}], max_new_tokens=tokens, temperature=0.0
            )
            record.update(status="returned", raw_response=output.text, metadata=output.metadata)
            actual = output.metadata.get("visual_tokens")
            if isinstance(actual, (int, float)) and actual > estimated:
                ctx.visual_tokens_estimated += int(actual) - estimated
            return output.text, record["call_id"], output.metadata
        except (RuntimeError, ValueError, OSError) as exc:
            record.update(status="failed", error=str(exc), error_type=type(exc).__name__)
            raise
        finally:
            record["elapsed_sec"] = time.monotonic() - started

    def call(
        self,
        role: str,
        payload: dict[str, Any],
        *,
        batch: MediaBatch | None = None,
        parser: Callable[[dict[str, Any], bool], Any] | None = None,
        terminal: bool = False,
    ) -> CallResult:
        text = self._prompt(role, payload)
        tokens = (
            self.config.final_tokens
            if terminal
            else self.config.observer_tokens
            if role in self.observer_roles
            else self.config.compiler_tokens
        )
        prepared = self.media.prepare(batch) if batch else None
        try:
            raw, call_id, metadata = self._invoke(
                role, text, prepared, terminal=terminal, tokens=tokens
            )
        except RuntimeError as exc:
            oom = type(exc).__name__ == "OutOfMemoryError" or (
                "out of memory" in str(exc).casefold()
                and any(s in str(exc).casefold() for s in ("cuda", "gpu"))
            )
            if not oom or batch is None:
                raise
            self.context.issues.append("oom_media_downgrade")
            # Optional CUDA cache release without adding a dependency to controller-only tests.
            import sys

            torch = sys.modules.get("torch")
            if torch is not None and torch.cuda.is_available():
                torch.cuda.empty_cache()
            prepared = self.media.prepare(batch, safe=True)
            raw, call_id, metadata = self._invoke(
                role, text, prepared, terminal=terminal, tokens=tokens
            )
        limited = bool(prepared and prepared.quality_limited)

        def parse(value: str) -> Any:
            data = json_object(value)
            if role == "observe" and (metadata.get("output_tokens", 0) >= tokens):
                data["truncated"] = True
            return parser(data, limited) if parser else data

        try:
            value = parse(raw)
        except (ProtocolError, KeyError, TypeError) as exc:
            original_record = next(c for c in self.context.calls if c["call_id"] == call_id)
            original_record.update(protocol_status="invalid", protocol_error=str(exc))
            if role in self.unrepaired_roles:
                raise
            self.context.issues.append(f"protocol_repair:{role}")
            repaired, _repair_id, _ = self._invoke(
                "repair", self._repair_prompt(role, raw, str(exc), payload), None,
                terminal=terminal, tokens=tokens
            )
            # Repair cannot change the displayed media/source set or the quality of the first call.
            self.context.calls[-1]["repairs_call_id"] = call_id
            self.context.calls[-1]["parsed_role"] = role

            # A repair may reformat an existing assertion, never invent a new source ID.
            def preserve_assertions(value: Any, key: str = "") -> None:
                if isinstance(value, dict):
                    for name, child in value.items():
                        preserve_assertions(child, name)
                elif isinstance(value, list):
                    for child in value:
                        preserve_assertions(child, key)
                elif (
                    isinstance(value, str)
                    and value
                    and key
                    in {
                        "statement",
                        "structured_value",
                        "prediction",
                        "fact_id",
                        "fact_ids",
                        "source_frame_ids",
                        "source_segment_ids",
                        "evidence_fact_ids",
                        "anchor_source_ids",
                        "target_source_ids",
                    }
                    and value not in raw
                    and json.dumps(value)[1:-1] not in raw
                ):
                    raise ProtocolError("repair invented an assertion or evidence reference")

            try:
                preserve_assertions(json_object(repaired))
                value = parse(repaired)
            except (ProtocolError, KeyError, TypeError) as error:
                self.context.calls[-1].update(protocol_status="invalid", protocol_error=str(error))
                raise
            self.context.calls[-1]["protocol_status"] = "valid"
            original_record["protocol_status"] = "repaired"
            return CallResult(value, prepared, call_id, repaired)
        self.context.calls[-1]["protocol_status"] = "valid"
        return CallResult(value, prepared, call_id, raw)
