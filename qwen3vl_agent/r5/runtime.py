"""Bounded calls with complete receipts and an explicit remaining-stage reserve."""

from __future__ import annotations

import time
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from qwen3vl_agent.r3.runtime import RunContext as BaseContext
from qwen3vl_agent.r3.runtime import json_object
from qwen3vl_agent.r5.prompts import RULES, prompt
from qwen3vl_agent.r5.observation import decode_observation, PROTOCOL_VERSION
from qwen3vl_agent.r5.types import BudgetExhausted, ProtocolError


class NeedsSplit(RuntimeError):
    pass


class ModelExecutionError(RuntimeError):
    """An engine failure, never a schema error or a partial answer."""


def known_prediction(raw: str, choices: list[dict]) -> str | None:
    """Read only fully closed top-level fields, even if a later field is cut off."""
    labels = {c["label"] for c in choices}
    text = raw.strip()
    if text.startswith("```"):
        text = text.partition("\n")[2].lstrip()
    if not text.startswith("{"):
        return None
    decoder, pos = json.JSONDecoder(), 1
    try:
        while pos < len(text):
            while pos < len(text) and text[pos].isspace():
                pos += 1
            key, pos = decoder.raw_decode(text, pos)
            while pos < len(text) and text[pos].isspace():
                pos += 1
            if text[pos] != ":":
                return None
            pos += 1
            while pos < len(text) and text[pos].isspace():
                pos += 1
            value, pos = decoder.raw_decode(text, pos)
            if key == "prediction":
                return value.strip() if isinstance(value, str) and value.strip() in labels else None
            while pos < len(text) and text[pos].isspace():
                pos += 1
            if text[pos] != ",":
                return None
            pos += 1
    except (ValueError, IndexError):
        pass
    return None


@dataclass
class RunContext(BaseContext):
    required_reserve: int = 0
    elapsed_sec: float = 0.0
    clock_mark: float = field(default_factory=time.monotonic, repr=False)

    def tick(self) -> float:
        now = time.monotonic()
        self.elapsed_sec += max(0.0, now - self.clock_mark)
        self.clock_mark = now
        return self.elapsed_sec

    def changed(self) -> None:
        self.tick()
        super().changed()

    def snapshot(self) -> dict:
        self.tick()
        return {k: v for k, v in super().snapshot().items() if k != "clock_mark"}

    def can_observe(self) -> bool:
        return self.tick() < self.budget.visual_deadline_sec and self.can_call()

    def phase_calls(self, phase: str) -> int:
        return sum(c.get("attempt_phase") == phase and c["role"] == "observe" for c in self.calls)

    def can_call(self, *, terminal: bool = False, optional: bool = False) -> bool:
        return self.tick() < self.budget.max_elapsed_sec and len(self.calls) < self.limit() - self.required_reserve

    @property
    def remaining(self) -> int:
        return self.limit() - len(self.calls)


@dataclass
class CallResult:
    value: Any
    call_id: str
    prepared: Any


class ModelSession:
    def __init__(self, model: Any, media: Any, config: Any, context: RunContext):
        self.model, self.media, self.config, self.context = model, media, config, context

    def fits(self, role: str, payload: dict) -> bool:
        return len(prompt(role, payload)) <= self.context.budget.max_text_chars_per_call

    def invoke(self, role: str, payload: dict, prepared: Any, tokens: int,
               *, aliases: dict | None = None) -> tuple[str, str, dict]:
        ctx, budget = self.context, self.context.budget
        if not ctx.can_call():
            raise BudgetExhausted("model_call_budget_or_remaining_stage_reserve")
        if role == "observe" and not ctx.can_observe():
            raise BudgetExhausted("visual_time_budget")
        text = prompt(role, payload)
        if len(text) > budget.max_text_chars_per_call:
            raise NeedsSplit("text_context_budget")
        frames, pixels = (prepared.frames, prepared.pixels) if prepared else ((), 0)
        estimated = (pixels + 1023) // 1024
        if ctx.frame_exposures + len(frames) > budget.max_frame_exposures:
            raise BudgetExhausted("frame_exposure_budget")
        if ctx.media_pixels + pixels > budget.max_media_pixels:
            raise BudgetExhausted("media_pixel_budget")
        if budget.max_visual_tokens is not None and (
            ctx.visual_tokens_estimated + estimated > budget.max_visual_tokens
        ):
            raise BudgetExhausted("visual_token_budget")
        call_id = f"call_{len(ctx.calls) + 1:06d}"
        record = {
            "call_id": call_id,
            "role": role,
            "status": "started",
            "payload": payload,
            "source_frame_ids": [f.id for f in frames],
            "media_pixels": pixels,
            "visual_tokens_estimated": estimated,
            "text_chars": len(text),
            "media_kind": prepared.kind if prepared else "text",
            "protocol_version": PROTOCOL_VERSION,
            "evidence_aliases": aliases or {},
            "attempt_phase": payload.get("attempt_phase"),
            "segment_id": payload.get("segment_id"),
            "started_at_utc": time.time(),
            "output_token_limit": tokens,
        }
        ctx.calls.append(record)
        ctx.frame_exposures += len(frames)
        ctx.media_pixels += pixels
        ctx.visual_tokens_estimated += estimated
        ctx.shown_frame_ids.extend(f.id for f in frames)
        ctx.changed()
        start = time.monotonic()
        try:
            parts = [dict(p) for p in prepared.parts] if prepared else []
            if aliases:
                inverse = {ref: alias for alias, ref in aliases.items()}
                headers = {f"Frame {f.id} at {f.timestamp_seconds:.6f}s":
                           f"Frame {inverse[f.id]} at {f.timestamp_seconds:.6f}s" for f in frames}
                for part in parts:
                    if part.get("type") == "text" and part.get("text") in headers:
                        part["text"] = headers[part["text"]]
                if prepared and prepared.kind == "ordered_video":
                    parts.insert(0, {"type": "text", "text": "Supplied video frames in order: " +
                                    ", ".join(inverse[f.id] for f in frames)})
            output = self.model.generate(
                [
                    {
                        "role": "user",
                        "content": [
                            *parts,
                            {"type": "text", "text": text},
                        ],
                    }
                ],
                max_new_tokens=tokens,
                temperature=0.0,
            )
            record.update(status="returned", raw_response=output.text, metadata=output.metadata)
            actual = output.metadata.get("visual_tokens")
            if isinstance(actual, int) and actual > estimated:
                ctx.visual_tokens_estimated += actual - estimated
            if (
                budget.max_visual_tokens is not None
                and ctx.visual_tokens_estimated > budget.max_visual_tokens
            ):
                raise BudgetExhausted("actual_visual_token_budget_exceeded")
            return output.text, call_id, output.metadata
        except BaseException as exc:
            record.update(status="failed", error=str(exc), error_type=type(exc).__name__)
            recoverable_oom = (role == "observe" and isinstance(exc, RuntimeError)
                               and "out of memory" in str(exc).lower())
            if (isinstance(exc, Exception) and not isinstance(exc, BudgetExhausted)
                    and not recoverable_oom):
                raise ModelExecutionError(f"{type(exc).__name__}: {exc}") from exc
            raise
        finally:
            record["elapsed_sec"] = time.monotonic() - start
            ctx.changed()

    def call(
        self,
        role: str,
        payload: dict,
        *,
        batch: Any = None,
        parser: Callable[[dict], Any] | None = None,
        aliases: dict | None = None,
        tokens_override: int | None = None,
    ) -> CallResult:
        if batch and len(batch.frames) > self.config.max_frames_per_call:
            raise NeedsSplit("per_call_frame_cap")
        prepared = self.media.prepare(batch) if batch else None
        tokens = getattr(
            self.config,
            {
                "compile": "compiler_tokens",
                "observe": "observer_tokens",
                "merge": "merge_tokens",
                "compose": "composer_tokens",
            }[role],
        )
        tokens = tokens_override or tokens
        replay = next((c for c in reversed(self.context.calls)
                       if c["role"] == role and c["status"] == "returned"
                       and (role != "observe" or not c.get("observation_committed")) and c["payload"] == payload
                       and c.get("evidence_aliases") == (aliases or {})
                       and c.get("output_token_limit") == tokens), None)
        try:
            if replay is not None:
                raw, call_id, metadata = replay["raw_response"], replay["call_id"], replay["metadata"]
                replay["replayed_from_receipt"] = True
            else:
                raw, call_id, metadata = self.invoke(role, payload, prepared, tokens, aliases=aliases)
        except RuntimeError as exc:
            if role == "observe" and "out of memory" in str(exc).lower():
                raise NeedsSplit("oom_split_required") from exc
            raise

        def truncated(meta: dict) -> bool:
            return (
                meta.get("finish_reason") in {"length", "max_tokens"}
                or (meta.get("output_tokens", 0) or 0) >= tokens
            )

        receipt = next(c for c in self.context.calls if c["call_id"] == call_id)
        if receipt.get("validation_error"):
            raise ProtocolError(receipt["validation_error"])
        if truncated(metadata) and role == "merge":
            raise NeedsSplit("model_output_truncated")

        if role == "observe":
            value = decode_observation(raw, truncated=truncated(metadata))
            value = parser(value) if parser else value
            receipt["observation_validation"] = {
                k: value.get(k) for k in ("rejected", "normalizations", "truncated", "unresolved")
            }
            self.context.changed()
            return CallResult(value, call_id, prepared)

        def parse(text: str) -> Any:
            value = json_object(text)
            return parser(value) if parser else value

        result_metadata = metadata
        try:
            try:
                value = parse(raw)
            except (ValueError, TypeError, KeyError, ProtocolError) as exc:
                self.context.issues.append(f"protocol_repair:{role}")
                repair_payload = {
                    "original_role": role,
                    "original_call_id": call_id,
                    "target_protocol": RULES[role],
                    "allowed_references": sorted({u["id"] for u in payload.get("units", [])}
                                                 | {f["id"] for f in payload.get("facts", [])}
                                                 | set(payload.get("catalog", {}))),
                    "choices": payload.get("choices", []),
                    "raw_response": raw,
                    "error": str(exc),
                }
                if role == "merge":
                    repair_payload["input_units"] = [
                        {key: unit[key] for key in ("id", "statement", "kind") if key in unit}
                        for unit in payload["units"]
                    ]
                    receipt["repair_reason"] = str(exc)
                # A completed or interrupted repair is not a new allowance on resume.
                old_repair = next((c for c in self.context.calls
                                   if c["role"] == "repair" and c["payload"].get("original_call_id") == call_id), None)
                if old_repair is not None:
                    if old_repair["status"] != "returned":
                        raise ProtocolError("format repair was interrupted or failed")
                    repaired, repair_meta = old_repair["raw_response"], old_repair["metadata"]
                else:
                    repaired, _, repair_meta = self.invoke("repair", repair_payload, None, tokens)
                    self.context.calls[-1]["repairs_call_id"] = call_id
                result_metadata = repair_meta
                if truncated(repair_meta) and role != "compose":
                    raise ProtocolError("repair output truncated")
                if role == "compile":
                    try:
                        original = json_object(raw)
                    except (ValueError, TypeError, ProtocolError):
                        original = {}
                    if original.get("scope_interval") is not None and (
                        json_object(repaired).get("scope_interval") != original["scope_interval"]
                    ):
                        raise ProtocolError("format repair changed the compiled scope")
                if role == "compose":
                    original_prediction = known_prediction(raw, payload.get("choices", []))
                    if original_prediction is not None and known_prediction(repaired, payload.get("choices", [])) != original_prediction:
                        raise ProtocolError("format repair changed the original prediction")

                def original_material(value: Any):
                    if isinstance(value, dict):
                        for key, item in value.items():
                            if key == "statement" and isinstance(item, str):
                                yield "factual statement", item
                            elif role == "merge" and key == "reason" and isinstance(item, str):
                                yield "omission reason", item
                            elif role == "merge" and key == "conflicts" and isinstance(item, list):
                                for conflict in item:
                                    if isinstance(conflict, str):
                                        yield "semantic conclusion", conflict
                            elif key in {"support_refs", "evidence_refs", "evidence", "ref_id"}:
                                for ref in item if isinstance(item, list) else [item]:
                                    if isinstance(ref, str):
                                        yield "evidence reference", ref
                            else:
                                yield from original_material(item)
                    elif isinstance(value, list):
                        for item in value:
                            yield from original_material(item)
                for kind, item in original_material(json_object(repaired)):
                    if (role == "merge" and kind == "evidence reference"
                            and item in repair_payload["allowed_references"]):
                        continue
                    if json.dumps(item, ensure_ascii=False) not in raw and json.dumps(item) not in raw:
                        raise ProtocolError(f"format repair introduced or changed a {kind}")
                value = parse(repaired)
            if role == "merge":
                receipt["merge_validation"] = {
                    "claim_count": len(value["claims"]),
                    "passthrough_refs": value["passthrough_refs"],
                    "omitted_refs": value["omitted_refs"],
                }
            if role == "compose":
                if truncated(metadata) or truncated(result_metadata):
                    value["unresolved"].append("composer_output_truncated")
                receipt["answer_validation"] = {
                    k: value.get(k) for k in ("rejected", "rejected_references", "unresolved")
                }
        except (ValueError, TypeError, KeyError, ProtocolError) as exc:
            receipt["validation_error"] = str(exc)
            raise
        finally:
            self.context.changed()
        return CallResult(value, call_id, prepared)
