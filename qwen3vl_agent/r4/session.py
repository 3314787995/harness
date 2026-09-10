"""Durable role budgets and model returns, independent of R3/final-answer policies."""
from __future__ import annotations
import hashlib
import json
import time
from dataclasses import asdict
from pathlib import Path

from qwen3vl_agent.models.qwen3vl import VisualBudgetExceeded
from .collection_contracts import ContractError, parse_json, diagnostics, parse_compile_json, compile_diagnostics
from .collection_prompts import build_prompt
from .types import R4Budget, BudgetExhausted
from .prompts import VERSION


class StageFailure(Exception):
    def __init__(self, stage, code, message, *, errors=(), call_ids=(), window_id=None):
        self.failure = {"stage": stage, "code": code, "message": message, "errors": list(errors),
                        "call_ids": list(call_ids), "window_id": window_id}
        super().__init__(message)


class InputNeedsSplit(Exception):
    pass


class CollectionSession:
    def __init__(self, model, media, config, request, work, save, checkpoint_path=None):
        self.model, self.media, self.config, self.request = model, media, config, request
        self.work, self.changed, self.checkpoint_path = work, save, checkpoint_path
        caps = {}
        for k, v in asdict(request.budget).items():
            choices = [x for x in (v, getattr(config.budget, k)) if x is not None]
            caps[k] = min(choices) if choices else None
        caps["terminal_call_reserve"] = 0
        self.budget = R4Budget(**caps)
        self.state = work.setdefault("runtime", {"calls": {}, "provider_calls": [], "recovery_roots": {}, "limits": {}})
        self.provider_calls = self.state["provider_calls"]

    def configure(self, n, entity, duration, known_frame=False):
        self.state["limits"] = {"compile": 1, "base": n, "focused": self.config.max_focused_calls,
            "qualification": self.config.max_qualification_calls, "identity": self.config.max_identity_calls if entity else 0,
            "recovery": self.config.max_recovery_calls, "answer": 1,
            "total": min(self.budget.max_model_calls, self.config.local_max_calls if known_frame else
                         192 if duration > self.config.short_duration_sec else n + (14 if entity else 10)),
            "frames": min(self.budget.max_frame_exposures, 4096 if duration > self.config.short_duration_sec else self.config.short_frame_exposures)}
        self.changed()

    def usage(self):
        calls = [v for v in self.state["calls"].values() if v.get("charged")]
        return {"model_calls": len(calls), "calls_by_purpose": {k: sum(c["pool"] == k for c in calls) for k in
                  ("compile", "base", "focused", "qualification", "identity", "recovery", "answer")},
                "calls_by_role": {k: sum(c["role"] == k for c in calls) for k in sorted({c["role"] for c in calls})},
                "frame_exposures": sum(c["frames"] for c in calls),
                "media_pixels": sum(c.get("actual_pixels", c["pixels"]) for c in calls),
                "visual_tokens_measured": sum(c.get("actual_visual_tokens", 0) for c in calls),
                "visual_tokens": sum(c.get("actual_visual_tokens", c["estimated_visual_tokens"]) for c in calls),
                "visual_token_measurement_complete": all("actual_visual_tokens" in c for c in calls if c["frames"]),
                "generated_tokens": sum(c.get("output_tokens", c["token_reservation"]) for c in calls),
                "provider_calls": len(self.provider_calls), "limits": self.state["limits"],
                "calls": list(self.state["calls"].values())}

    def can_call(self, pool):
        u, lim = self.usage(), self.state["limits"]
        used = len(self.state["recovery_roots"]) if pool == "recovery" else u["calls_by_purpose"].get(pool, 0)
        reserve = 0 if pool == "answer" or u["calls_by_purpose"].get("answer", 0) else 1
        return u["model_calls"] < lim.get("total", self.budget.max_model_calls) - reserve and used < lim.get(pool, 0)

    def call(self, key, role, payload, *, pool, targets=(), prepared=None, aliases=None, catalog=None,
             root_id=None, recovery=False, feedback=None):
        previous = self.state["calls"].get(key)
        if previous:
            if previous["status"] == "returned":
                return previous
            if previous["status"] == "input_blocked":
                raise InputNeedsSplit(previous["error"])
            raise StageFailure(role, "interrupted_call", "A started call has no returned output; its budget remains charged", call_ids=[key])
        aliases, catalog = aliases or {}, catalog or {}
        try:
            prompt = build_prompt(role, payload, targets, feedback=feedback)
        except (ContractError, AssertionError, ValueError, KeyError, TypeError) as exc:
            raise StageFailure("observe_prompt" if role != "compile" else "compile_prompt", "prompt_contract_error",
                               str(exc), errors=getattr(exc, "errors", [])) from exc
        frames = list(prepared.frames) if prepared else []
        parts = []
        inverse = {v: k for k, v in aliases.items()}
        temporal = role in {"identity", "scope"} or any(s.predicate_kind != "static" or s.namespace == "task_item" for s in targets)
        if prepared:
            # R4 presents explicit ordered images: no hidden frame padding or forged video FPS.
            for frame, (width, height) in zip(frames, prepared.sizes):
                ref = inverse[frame.id]
                entry = payload.get("catalog", {}).get(ref, {})
                label = f"{ref} · {entry.get('region', 'core')}"
                if 'core_for' in entry:
                    label += ' · core_for=' + ','.join(entry['core_for'])
                if temporal:
                    label += f" · source_time={frame.timestamp_seconds:.6f}s"
                parts.extend([{"type": "text", "text": label}, {"type": "image", "image": frame.path,
                              "min_pixels": width * height, "max_pixels": width * height}])
        parts.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": parts}]
        u, lim = self.usage(), self.state["limits"]
        tokens = self.config.final_tokens if role == "best_effort" else self.config.compiler_tokens if role == "compile" else self.config.observer_tokens if role == "discover_candidates" and pool != "focused" else self.config.review_tokens
        if recovery:
            original=self.state["calls"].get(key.rsplit(":",1)[0]+":0")
            if original:
                tokens=original["token_reservation"]
        pixels = prepared.pixels if prepared else 0
        estimated = (pixels + 1023) // 1024
        if recovery and not self.can_call("recovery") and len(self.state["recovery_roots"]) >= lim["recovery"]:
            raise StageFailure(role, "question_recovery_exhausted", "The question has spent its two format recoveries; preserve this gap and continue other work", window_id=root_id)
        if not self.can_call(pool):
            raise BudgetExhausted("call_pool_exhausted:" + pool)
        reserve = pool != "answer" and not u["calls_by_purpose"].get("answer", 0)
        if u["generated_tokens"] + tokens + (self.config.final_tokens if reserve else 0) > self.budget.max_generated_tokens:
            raise BudgetExhausted("generated_token_budget")
        if u["frame_exposures"] + len(frames) + (8 if reserve else 0) > lim["frames"]:
            raise BudgetExhausted("frame_exposure_budget")
        if u["media_pixels"] + pixels * 2 + (16 * 1048576 if reserve else 0) > self.budget.max_media_pixels:
            raise BudgetExhausted("processed_pixel_budget")
        if len(prompt) > self.budget.max_text_chars_per_call:
            raise BudgetExhausted("text_input_budget")
        if estimated > self.budget.visual_tokens_per_call:
            raise InputNeedsSplit("estimated visual tokens exceed per-call cap")
        if recovery:
            if self.state["recovery_roots"].get(root_id):
                raise StageFailure(role, "recovery_exhausted", "This base window and descendants already spent protocol recovery", window_id=root_id)
            self.state["recovery_roots"][root_id] = key
        record = {"call_id": key, "role": role, "pool": pool, "status": "started", "charged": True,
                  "protocol": VERSION, "payload": payload, "messages": messages, "feedback": feedback,
                  "aliases": aliases, "catalog": catalog, "source_frame_ids": [f.id for f in frames],
                  "frames": len(frames), "pixels": pixels * 2, "estimated_visual_tokens": estimated,
                  "token_reservation": tokens, "root_id": root_id, "recovery": recovery,
                  "started_at": time.time(), "validation": "pending"}
        self.state["calls"][key] = record
        kwargs = {"max_new_tokens": tokens, "temperature": 0.0,
                  "visual_token_limit": min(self.budget.visual_tokens_per_call,
                      self.budget.max_visual_tokens - u["visual_tokens"] if self.budget.max_visual_tokens is not None else self.budget.visual_tokens_per_call),
                  "processed_pixel_limit": self.budget.max_media_pixels - u["media_pixels"]}
        if self.checkpoint_path:
            receipt = Path(self.checkpoint_path).parent / (Path(self.checkpoint_path).name + ".receipts") / (hashlib.sha256(key.encode()).hexdigest()[:20] + ".json")
            kwargs["preparation_receipt_path"] = str(receipt)
            record["preparation_receipt_path"] = str(receipt)
        record["generation_kwargs"] = kwargs
        self.changed()
        try:
            output = self.model.generate(messages, **kwargs)
            record.update(status="returned", raw_response=output.text, metadata=output.metadata)
            for dest, source in (("actual_visual_tokens", "visual_tokens"), ("actual_pixels", "processed_pixels"), ("output_tokens", "output_tokens")):
                actual = output.metadata.get(source)
                if isinstance(actual, int) and not isinstance(actual, bool) and actual >= 0:
                    record[dest] = actual
            # Durable return precedes parsing or any evidence commit.
            self.changed()
            return record
        except VisualBudgetExceeded as exc:
            record.update(status="input_blocked", charged=False, error=str(exc))
            if recovery:
                # The recovery started (including preparation), so its window allowance stays used.
                record["recovery_allowance_consumed"] = True
            raise InputNeedsSplit(str(exc)) from exc
        except BaseException as exc:
            record.update(error=str(exc), error_type=type(exc).__name__)
            if "out of memory" in str(exc).lower():
                record["status"] = "oom"
                raise InputNeedsSplit("generation_oom") from exc
            raise
        finally:
            record["elapsed_sec"] = time.time() - record["started_at"]
            self.changed()

    def parse(self, record):
        metadata = record.get("metadata", {})
        if metadata.get("finish_reason") in {"length", "max_tokens"} or (metadata.get("output_tokens", 0) or 0) >= record["token_reservation"]:
            message = ("Generation reached its limit; return a shorter complete task with required fields and necessary exceptions"
                       if record["role"] == "compile" else
                       "Generation reached its limit; return a shorter complete response, do not enumerate each frame")
            raise ContractError([{"path": "$", "code": "response_truncated", "message": message, "expected": "complete bounded JSON", "actual": "token limit"}])
        parser = parse_compile_json if record["role"] == "compile" else parse_json
        return parser(record["raw_response"])

    def validation(self, record, errors=(), *, committed=()):
        precise = [{**e, "call_id": record["call_id"], "stage": record["role"]} for e in errors]
        feedback = compile_diagnostics(precise) if record["role"] == "compile" else diagnostics(precise)
        record.update(validation="invalid" if errors else "valid", validation_errors=precise,
                      model_feedback=feedback, committed_slots=sorted(set(record.get("committed_slots", [])) | set(committed)))
        self.changed()
        return record["model_feedback"]
