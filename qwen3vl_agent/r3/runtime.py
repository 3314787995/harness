"""Per-question resource accounting and transactional model calls."""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Any

from qwen3vl_agent.models.base import BaseVideoModel
from qwen3vl_agent.r1.media import MediaBatch, PreparedMedia, R1Media
from qwen3vl_agent.r3.config import R3Config
from qwen3vl_agent.r3.prompts import prompt
from qwen3vl_agent.r3.types import BudgetExhausted, ProtocolError, R3Budget


def json_object(raw: str, *, strict: bool = False) -> dict[str, Any]:
    text = raw.strip()
    if not strict and text.startswith("```"):
        lines = text.splitlines()
        if len(lines) >= 3 and lines[-1].strip() == "```":
            text = "\n".join(lines[1:-1])
    def unique_keys(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate JSON key: " + key)
            result[key] = value
        return result

    try:
        value = json.loads(
            text, parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)),
            object_pairs_hook=unique_keys if strict else None,
        )
    except (ValueError, TypeError) as exc:
        raise ProtocolError("expected a complete finite JSON object") from exc
    if not isinstance(value, dict):
        raise ProtocolError("expected JSON object")
    return value


@dataclass
class RunContext:
    budget: R3Budget
    call_limit: int = 0
    base_reserve: int = 0
    call_pools: dict[str, int] = field(default_factory=dict)
    calls: list[dict[str, Any]] = field(default_factory=list)
    provider_calls: list[dict[str, Any]] = field(default_factory=list)
    frame_exposures: int = 0
    media_pixels: int = 0
    visual_tokens_estimated: int = 0
    decoded_frames: int = 0
    shown_frame_ids: list[str] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)
    on_change: Callable[[], None] | None = field(default=None, repr=False)

    def limit(self) -> int:
        return min(self.budget.max_model_calls, self.call_limit or self.budget.max_model_calls)

    def can_call(self, *, terminal: bool = False, optional: bool = False) -> bool:
        return self.can_calls(1, terminal=terminal, optional=optional)

    def can_calls(self, count: int, *, terminal: bool = False, optional: bool = False) -> bool:
        reserve = 0 if terminal else self.budget.terminal_call_reserve
        return len(self.calls) + count <= self.limit() - reserve - (self.base_reserve if optional else 0)

    def changed(self) -> None:
        if self.on_change:
            self.on_change()

    def pool_remaining(self, pool: str) -> int:
        return self.call_pools.get(pool, 0) - sum(c.get("pool") == pool for c in self.calls)

    def can_spend(self, pool: str, count: int = 1) -> bool:
        if not self.call_pools:
            return self.can_calls(count, terminal=pool == "terminal", optional=pool in {"review", "format"})
        if self.pool_remaining(pool) < count:
            return False
        reserve = 0 if pool == "terminal" else max(self.budget.terminal_call_reserve, self.pool_remaining("terminal"))
        if pool in {"review", "format"}:
            reserve += max(0, self.pool_remaining("base"))
        return len(self.calls) + count <= self.limit() - reserve

    def snapshot(self) -> dict[str, Any]:
        return {
            key: value for key, value in self.__dict__.items() if key not in {"budget", "on_change"}
        }

    def summary(self) -> dict[str, Any]:
        measured = [c.get("metadata", {}) for c in self.calls]
        return {
            **self.snapshot(),
            "limits": asdict(self.budget),
            "model_calls": len(self.calls),
            "provider_call_count": len(self.provider_calls),
            "unique_frames": len(set(self.shown_frame_ids)),
            "input_tokens_measured": sum(m.get("input_tokens", 0) or 0 for m in measured),
            "output_tokens_measured": sum(m.get("output_tokens", 0) or 0 for m in measured),
            "token_usage_complete": all(
                "input_tokens" in m and "output_tokens" in m for m in measured
            ),
        }


@dataclass
class CallResult:
    value: Any
    call_id: str
    prepared: PreparedMedia | None
    raw: str


class ModelSession:
    def __init__(
        self, model: BaseVideoModel, media: R1Media, config: R3Config, context: RunContext
    ) -> None:
        self.model, self.media, self.config, self.context = model, media, config, context

    def invoke(
        self,
        role: str,
        payload: dict[str, Any],
        prepared: PreparedMedia | None,
        *,
        terminal: bool,
        optional: bool,
        tokens: int,
        pool: str | None = None,
    ) -> tuple[str, str, dict[str, Any]]:
        ctx, budget = self.context, self.context.budget
        pool = pool or ("terminal" if terminal else "preparation")
        if ctx.call_pools and not ctx.can_spend(pool):
            raise BudgetExhausted("call_pool_unavailable:" + pool)
        if not ctx.can_call(terminal=terminal, optional=optional):
            raise BudgetExhausted("model_call_budget_or_required_coverage_reserve")
        text = prompt(role, payload)
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
            raise BudgetExhausted("visual_token_budget")
        call_id = f"call_{len(ctx.calls) + 1:06d}"
        record = {
            "call_id": call_id,
            "role": role,
            "pool": pool,
            "status": "started",
            "payload": payload,
            "source_frame_ids": [f.id for f in frames],
            "media_pixels": pixels,
            "visual_tokens_estimated": estimated,
            "media_kind": prepared.kind if prepared else "text",
            "quality_limited": bool(prepared and prepared.quality_limited),
        }
        ctx.calls.append(record)
        ctx.frame_exposures += len(frames)
        ctx.media_pixels += pixels
        ctx.visual_tokens_estimated += estimated
        ctx.shown_frame_ids.extend(f.id for f in frames)
        ctx.changed()  # A crashed or cancelled in-flight call is still charged on resume.
        started = time.monotonic()
        try:
            output = self.model.generate(
                [
                    {
                        "role": "user",
                        "content": [
                            *(prepared.parts if prepared else []),
                            {"type": "text", "text": text},
                        ],
                    }
                ],
                max_new_tokens=tokens,
                temperature=0.0,
            )
            record.update(status="returned", raw_response=output.text, metadata=output.metadata)
            actual = output.metadata.get("visual_tokens")
            if isinstance(actual, (int, float)) and actual > estimated:
                ctx.visual_tokens_estimated += int(actual) - estimated
            if (
                budget.max_visual_tokens is not None
                and ctx.visual_tokens_estimated > budget.max_visual_tokens
            ):
                ctx.issues.append("actual_visual_token_budget_exceeded")
                raise BudgetExhausted("actual_visual_token_budget_exceeded")
            return output.text, call_id, output.metadata
        except BaseException as exc:
            record.update(status="failed", error=str(exc), error_type=type(exc).__name__)
            raise
        finally:
            record["elapsed_sec"] = time.monotonic() - started
            ctx.changed()

    def call(
        self,
        role: str,
        payload: dict[str, Any],
        *,
        batch: MediaBatch | None = None,
        parser: Callable[[dict[str, Any]], Any] | None = None,
        terminal: bool = False,
        optional: bool = False,
    ) -> CallResult:
        if batch and len(batch.frames) > self.config.max_frames_per_call:
            raise BudgetExhausted("per_call_frame_cap")
        prepared = self.media.prepare(batch) if batch else None
        tokens = (
            self.config.final_tokens
            if terminal
            else (
                self.config.observer_tokens
                if role in {"observe", "relation"}
                else self.config.compiler_tokens
            )
        )
        try:
            raw, call_id, metadata = self.invoke(
                role, payload, prepared, terminal=terminal, optional=optional, tokens=tokens
            )
        except RuntimeError as exc:
            if (
                not batch
                or "out of memory" not in str(exc).lower()
                or isinstance(exc, BudgetExhausted)
            ):
                raise
            self.context.issues.append("oom_spatial_downgrade")
            prepared = self.media.prepare(batch, safe=True)
            raw, call_id, metadata = self.invoke(
                role, payload, prepared, terminal=terminal, optional=optional, tokens=tokens
            )

        def parse(text: str) -> Any:
            data = json_object(text)
            if role == "observe" and (
                metadata.get("finish_reason") in {"length", "max_tokens"}
                or (metadata.get("output_tokens", 0) or 0) >= tokens
            ):
                data["truncated"] = True
            return parser(data) if parser else data

        try:
            value = parse(raw)
        except (ProtocolError, KeyError, TypeError, ValueError) as exc:
            self.context.issues.append(f"protocol_repair:{role}")
            repaired, _, _ = self.invoke(
                "repair",
                {"original_role": role, "raw_response": raw, "error": str(exc)},
                None,
                terminal=terminal,
                optional=not terminal,
                tokens=tokens,
            )
            self.context.calls[-1]["repairs_call_id"] = call_id
            # No new source reference may be introduced by a format-only repair.
            original_tokens = set(raw.replace('"', " ").replace("'", " ").split())
            known_refs = {f.id for f in prepared.frames} if prepared else set()
            known_refs.update(s.get("segment_id", "") for s in payload.get("segments", []))
            for ref in known_refs:
                if ref and ref in repaired and ref not in raw and ref not in original_tokens:
                    raise ProtocolError("repair introduced an evidence reference")
            value = parse(repaired)
            raw = repaired
        return CallResult(value, call_id, prepared, raw)
