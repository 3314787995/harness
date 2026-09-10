"""Charged-before-dispatch model calls, exact replay and version-checked checkpoints."""

from __future__ import annotations

import importlib.metadata
import json
import time
from dataclasses import asdict
from pathlib import Path

import jsonschema

from qwen3vl_agent.r3.checkpoint import Checkpoint, file_digest

from .contracts import SCHEMAS, parse
from .prompts import COMMON, PROMPTS
from .prompts import VERSION as PROMPT_VERSION
from .types import VERSION, BudgetExhausted, ModelFailure, ProtocolError, digest, plain


def software_fingerprint(model, config):
    packages = {}
    for name in ("torch", "transformers", "qwen-vl-utils", "av", "Pillow", "accelerate"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    processor = getattr(model, "_processor", None) or getattr(model, "processor", None)
    return {
        "pipeline": VERSION,
        "prompts": PROMPT_VERSION,
        "model_class": type(model).__module__ + "." + type(model).__name__,
        "model_path": str(getattr(model, "model_path", "unknown")),
        "model_revision": config.model_revision,
        "generation": plain(getattr(model, "generation", {})),
        "model_settings": {
            k: plain(getattr(model, k, None))
            for k in (
                "dtype",
                "device_map",
                "attn_implementation",
                "use_cache",
                "revision",
                "video",
            )
        },
        "loaded_commit": getattr(
            getattr(getattr(model, "model", None), "config", None), "_commit_hash", None
        ),
        "packages": packages,
        "chat_template_hash": digest(getattr(processor, "chat_template", None)),
        "processor_class": type(processor).__name__ if processor else None,
    }


def implementation_hash():
    root = Path(__file__).parent.parent
    files = list(Path(__file__).parent.glob("*.py"))
    for package in ("r1", "r2", "r3", "r4", "p01", "models"):
        files.extend((root / package).glob("*.py"))
    return digest({p.relative_to(root).as_posix(): file_digest(p) for p in sorted(set(files))})


def checkpoint_for(request, config, media, software):
    data = asdict(request)
    for key in ("resume", "checkpoint_path"):
        data.pop(key)
    fingerprint = {
        "request": data,
        "config": asdict(config),
        "software": software,
        "source_sha256": media.source_hash,
        "scope": asdict(media.contract),
        "implementation": implementation_hash(),
        "subtitle_sha256": file_digest(request.subtitle_path) if request.subtitle_path else None,
        "facts_sha256": file_digest(request.facts_input) if request.facts_input else None,
    }
    return Checkpoint(request.checkpoint_path, fingerprint, resume=request.resume)


class ModelSession:
    def __init__(self, model, config, limits, state, save):
        self.model, self.config, self.limits = model, config, limits
        self.state, self.save = state, save
        self.receipts = state.setdefault("receipts", [])
        self.jobs = state.setdefault("jobs", {})

    @property
    def remaining(self):
        return self.limits["max_model_calls"] - len(self.receipts)

    def resources(self):
        return {
            "limits": self.limits,
            "model_calls": len(self.receipts),
            "unique_frames": len({f for r in self.receipts for f in r["source_frame_ids"]}),
            "frame_exposures": sum(r["frame_count"] for r in self.receipts),
            "media_pixels": sum(r["pixels"] for r in self.receipts),
            "input_tokens": sum(r.get("input_tokens", 0) or 0 for r in self.receipts),
            "output_tokens": sum(r.get("output_tokens", 0) or 0 for r in self.receipts),
            "visual_tokens": sum(r.get("visual_tokens", 0) or 0 for r in self.receipts),
            "model_seconds": sum(r.get("elapsed_seconds", 0) for r in self.receipts),
            "failed_calls": sum(r["status"] not in {"ok", "started"} for r in self.receipts),
            "unknown_usage_calls": sum(r["status"] != "ok" for r in self.receipts),
            "receipts": plain(self.receipts),
        }

    def call(
        self, key, role, payload, validator=None, prepared=None, evidence=None, *, terminal=False
    ):
        evidence = evidence or {}
        visual_fingerprint = {
            k: {p: v for p, v in e.items() if p != "path"} for k, e in evidence.items()
        }
        identity = digest({"role": role, "payload": payload, "evidence": visual_fingerprint})
        cached = self.jobs.get(key)
        if cached:
            if cached["identity"] != identity:
                raise ProtocolError("stage key reused for different input")
            if cached.get("value") is not None:
                if validator:
                    validator(cached["value"])
                return plain(cached["value"]), cached["call_id"]
        self.jobs.setdefault(key, {"identity": identity, "invalid_outputs": 0})
        job = self.jobs[key]
        while job["invalid_outputs"] < 2:
            resources = self.resources()
            source_ids = sorted(
                {
                    e.get("source_frame_id", e["id"])
                    for e in evidence.values()
                    if e.get("modality") == "video"
                }
            )
            count, pixels = (len(prepared.frames), prepared.pixels) if prepared else (0, 0)
            previous = {f for r in self.receipts for f in r["source_frame_ids"]}
            if self.remaining <= (0 if terminal else 1):
                raise BudgetExhausted("model call cap / final-call reserve")
            if (
                len(previous | set(source_ids)) > self.limits["max_unique_frames"]
                or resources["frame_exposures"] + count > self.limits["max_frame_exposures"]
                or resources["media_pixels"] + pixels > self.limits["max_media_pixels"]
            ):
                raise BudgetExhausted("visual resource cap")
            body = {"stage": role, "input": payload, "schema": SCHEMAS[role]}
            if job["invalid_outputs"]:
                body["repair"] = {
                    "error": job["last_error"],
                    "instruction": "One format/contract repair. Do not invent absent input.",
                }
            encoded = json.dumps(body, ensure_ascii=False, allow_nan=False)
            if len(encoded) > self.config.max_text_chars:
                raise ProtocolError("bounded role context exceeded")
            parts = list(prepared.parts) if prepared else []
            parts.append({"type": "text", "text": "R7_JSON\n" + encoded})
            messages = [
                {"role": "system", "content": COMMON + "\n" + PROMPTS[role]},
                {"role": "user", "content": parts},
            ]
            call_id = f"call-{len(self.receipts) + 1:04d}"
            receipt = {
                "id": call_id,
                "key": key,
                "role": role,
                "status": "started",
                "input_hash": identity,
                "source_frame_ids": source_ids,
                "frame_count": count,
                "pixels": pixels,
                "text_chars": len(encoded),
                "evidence": visual_fingerprint,
                "video_frame_metadata": plain(prepared.video_frame_metadata) if prepared else [],
            }
            self.receipts.append(receipt)
            self.save()  # Charge before dispatch, including interrupted/failed calls.
            start = time.perf_counter()
            provider_returned = False
            try:
                tokens = getattr(
                    self.config,
                    {
                        "compile": "compiler_tokens",
                        "candidates": "candidates_tokens",
                        "observe": "observer_tokens",
                        "reason": "reasoner_tokens",
                        "verify": "verifier_tokens",
                        "final": "verifier_tokens",
                        "direct": "verifier_tokens",
                    }[role],
                )
                output = self.model.generate(
                    messages,
                    max_new_tokens=tokens,
                    temperature=0.0,
                    video_frame_metadata=prepared.video_frame_metadata if prepared else [],
                )
                provider_returned = True
                receipt["output"] = output.text
                for name in (
                    "input_tokens",
                    "output_tokens",
                    "visual_tokens",
                    "gpu_memory",
                    "video_timing_verified",
                ):
                    if name in output.metadata:
                        receipt[name] = plain(output.metadata[name])
                value = parse(output.text, role, validator)
            except (
                json.JSONDecodeError,
                jsonschema.ValidationError,
                ProtocolError,
                ValueError,
                KeyError,
                TypeError,
            ) as exc:
                if not provider_returned:
                    receipt["status"] = "model_error"
                    receipt["error"] = f"{type(exc).__name__}: {exc}"
                    raise ModelFailure(receipt["error"]) from exc
                receipt["status"] = "invalid_output"
                job["invalid_outputs"] += 1
                job["last_error"] = str(exc)[:1500]
            except BaseException as exc:
                receipt["status"] = (
                    "interrupted"
                    if isinstance(exc, (KeyboardInterrupt, SystemExit))
                    else "model_error"
                )
                receipt["error"] = f"{type(exc).__name__}: {exc}"
                if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                    raise
                raise ModelFailure(receipt["error"]) from exc
            else:
                receipt["status"] = "ok"
                job.update(value=value, call_id=call_id)
                return value, call_id
            finally:
                receipt["elapsed_seconds"] = time.perf_counter() - start
                self.save()
        raise ModelFailure("role contract failed after one repair: " + job.get("last_error", role))
