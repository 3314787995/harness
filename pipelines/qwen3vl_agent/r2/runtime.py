"""Per-call transactional receipts, bounded repairs and resumable Qwen stages."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import time
from dataclasses import asdict
from pathlib import Path

from qwen3vl_agent.r3.checkpoint import Checkpoint, file_digest

from .contracts import parse, repair_context, stage_schema
from .prompts import prompt
from .types import VERSION, BudgetExhausted, ProtocolError


def fingerprint(request, config, model, source_sha):
    package = Path(__file__).parent.parent
    implementation = hashlib.sha256()
    paths = list(Path(__file__).parent.glob("*.py")) + [
        package / p
        for p in (
            "models/qwen3vl.py",
            "models/base.py",
            "temporal_media.py",
            "r1/media.py",
            "r1/types.py",
            "p01/config.py",
            "r3/checkpoint.py",
        )
    ]
    for path in sorted(paths):
        implementation.update(path.relative_to(package).as_posix().encode())
        implementation.update(path.read_bytes())
    data = asdict(request)
    for key in ("resume", "checkpoint_path"):
        data.pop(key)
    return {
        "version": VERSION,
        "request": data,
        "config": asdict(config),
        "source_sha256": source_sha,
        "implementation": implementation.hexdigest(),
        "model": {
            "path": model.model_path,
            "dtype": getattr(model, "dtype", None),
            "generation": getattr(model, "generation", {}),
            "video": getattr(model, "video", {}),
            "attn_implementation": getattr(model, "attn_implementation", None),
            "use_cache": getattr(model, "use_cache", None),
            "device_map": getattr(model, "device_map", None),
            "resolved_revision": getattr(
                getattr(getattr(model, "model", None), "config", None), "_commit_hash", None
            ),
        },
        "external_sources": {
            kind: file_digest(p)
            for kind in ("subtitle", "asr")
            if (p := getattr(request, kind + "_path"))
        },
    }


def new_checkpoint(request, config, model, source_sha):
    return Checkpoint(
        request.checkpoint_path,
        fingerprint(request, config, model, source_sha),
        resume=request.resume,
    )


class ModelSession:
    def __init__(self, model, config, budget, state, save):
        self.model, self.config, self.budget, self.state, self.save = (
            model,
            config,
            budget,
            state,
            save,
        )
        state.setdefault("calls", [])
        state.setdefault("stages", {})

    def summary(self):
        calls = self.state["calls"]
        frames = [f for c in calls for f in c["frame_ids"]]
        catalog = self.state.get("media_catalog", {})
        return {
            "limits": asdict(self.budget),
            "model_calls": len(calls),
            "frame_exposures": len(frames),
            "unique_source_frames": len(
                {catalog.get(f, {}).get("source_frame_id", f) for f in frames}
            ),
            "media_pixels": sum(c["pixels"] for c in calls),
            "visual_tokens_estimated": sum(c["visual_tokens_estimated"] for c in calls),
            "input_tokens_measured": sum(
                c.get("metadata", {}).get("input_tokens", 0) or 0 for c in calls
            ),
            "output_tokens_measured": sum(
                c.get("metadata", {}).get("output_tokens", 0) or 0 for c in calls
            ),
            "visual_tokens_measured": sum(
                c.get("metadata", {}).get("visual_tokens", 0) or 0 for c in calls
            ),
            "token_usage_complete": all("input_tokens" in c.get("metadata", {}) for c in calls),
            "latency_seconds": sum(c.get("latency_seconds", 0) for c in calls),
            "terminal_calls": sum(c["role"] == "final" for c in calls),
        }

    def check(self, prepared, terminal=False):
        resources = self.summary()
        reserve = (
            0
            if terminal
            else min(self.budget.terminal_call_reserve, self.budget.max_model_calls - 2)
        )
        if resources["model_calls"] >= self.budget.max_model_calls - reserve:
            raise BudgetExhausted("model_call_budget")
        if terminal and resources["terminal_calls"] >= self.budget.terminal_call_reserve:
            raise BudgetExhausted("terminal_call_budget")
        frames = len(prepared.frames) if prepared else 0
        pixels = prepared.pixels if prepared else 0
        estimates = math.ceil(pixels / 1024)
        for consumed, extra, cap, name in (
            (
                resources["frame_exposures"],
                frames,
                self.budget.max_frame_exposures,
                "frame_exposure_budget",
            ),
            (resources["media_pixels"], pixels, self.budget.max_media_pixels, "media_pixel_budget"),
            (
                resources["visual_tokens_estimated"],
                estimates,
                self.budget.max_visual_tokens,
                "visual_token_budget",
            ),
        ):
            reserved = (
                0
                if terminal
                else min(
                    cap // 4,
                    self.budget.terminal_call_reserve
                    * (
                        self.config.max_frames_per_call
                        if "frame" in name
                        else self.config.media.normal_total_pixels
                        if "pixel" in name
                        else math.ceil(self.config.media.normal_total_pixels / 1024)
                    ),
                )
            )
            if consumed + extra > cap - reserved:
                raise BudgetExhausted(name)

    def call(self, key, role, payload, prepared=None, validator=None):
        stage = self.state["stages"].setdefault(key, {})
        if "value" in stage:
            return stage["value"], stage["call_id"]
        repair = None
        schema = stage_schema(role, payload)
        last_error = "stage interrupted"
        for attempt in range(2):
            call_key = f"{key}:{attempt}"
            receipt = next((c for c in self.state["calls"] if c["key"] == call_key), None)
            if receipt is None:
                self.check(prepared, terminal=role == "final")
                text = prompt(role, payload, repair, schema=schema)
                if role == "final" and repair and len(text) > self.budget.max_text_chars_per_call:
                    # Keep the same evidence and diagnostic fields; only shorten
                    # the rejected output excerpt, which is not visual evidence.
                    repair = copy.deepcopy(repair)
                    while (
                        repair.get("previous_output")
                        and len(text) > self.budget.max_text_chars_per_call
                    ):
                        repair["previous_output"] = repair["previous_output"][
                            : len(repair["previous_output"]) // 2
                        ]
                        text = prompt(role, payload, repair, schema=schema)
                if len(text) > self.budget.max_text_chars_per_call:
                    self.state.setdefault("preflight_failures", []).append(
                        {
                            "key": call_key,
                            "role": role,
                            "reason": "text_context_budget",
                            "text_chars": len(text),
                            "limit": self.budget.max_text_chars_per_call,
                            "model_invoked": False,
                        }
                    )
                    self.save()
                    raise BudgetExhausted("text_context_budget")
                receipt = {
                    "key": call_key,
                    "call_id": f"call_{len(self.state['calls']) + 1:06d}",
                    "role": role,
                    "status": "started",
                    "payload": copy.deepcopy(payload),
                    "prompt": text,
                    "output_schema": copy.deepcopy(schema),
                    "frame_ids": [f.id for f in prepared.frames] if prepared else [],
                    "pixels": prepared.pixels if prepared else 0,
                    "visual_tokens_estimated": math.ceil(prepared.pixels / 1024) if prepared else 0,
                    "media_kind": prepared.kind if prepared else "text",
                    "text_chars": len(text),
                }
                self.state["calls"].append(receipt)
                self.save()  # Charge before invocation, also if cancelled or the process dies.
                started = time.monotonic()
                try:
                    kwargs = {}
                    if prepared and prepared.video_frame_metadata:
                        kwargs["video_frame_metadata"] = prepared.video_frame_metadata
                    limit = (
                        self.config.final_tokens
                        if role == "final"
                        else self.config.observer_tokens
                        if role == "observe"
                        else self.config.compiler_tokens
                    )
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
                        max_new_tokens=limit,
                        **kwargs,
                    )
                    receipt.update(
                        status="returned",
                        raw=output.text,
                        metadata=output.metadata,
                        latency_seconds=time.monotonic() - started,
                    )
                    self.save()
                except Exception as exc:
                    receipt.update(
                        status="failed",
                        error=f"{type(exc).__name__}: {exc}",
                        latency_seconds=time.monotonic() - started,
                    )
                    self.save()
                    raise
            if receipt["status"] == "started":
                receipt["status"] = "interrupted"
                self.save()
            if receipt["status"] != "returned":
                last_error = receipt.get(
                    "error", "Previous invocation interrupted; its cost remains charged"
                )
                repair = {"error": last_error, "previous_output": ""}
                continue
            try:
                value = parse(receipt["raw"], role, schema=schema)
                if validator:
                    original = copy.deepcopy(value)
                    value = validator(value)
                    if value != original:
                        receipt["program_annotations"] = {
                            "basis": "program_validation_and_structural_defaults",
                            "normalized_value": copy.deepcopy(value),
                        }
                receipt["validation_status"] = "accepted"
                stage.update(value=value, call_id=receipt["call_id"])
                self.save()
                return value, receipt["call_id"]
            except ProtocolError as exc:
                last_error = str(exc)
                receipt["validation_error"] = last_error
                receipt["validation_status"] = "rejected"
                repair = {
                    "error": last_error,
                    "previous_output": receipt["raw"][:12000],
                    **repair_context(role, payload, receipt["raw"]),
                }
                self.save()
        raise ProtocolError(last_error)


def stable_key(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()[
        :20
    ]
