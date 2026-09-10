"""Before-dispatch accounting and exact, versioned replay."""

import importlib.metadata
import json
import time
from dataclasses import asdict
from pathlib import Path

from qwen3vl_agent.models.qwen3vl import InputContextExceeded
from qwen3vl_agent.r3.checkpoint import Checkpoint, file_digest

from .prompts import COMMON, PROMPTS
from .prompts import VERSION as PROMPT_VERSION
from .schema import parse, role_schema, validate
from .types import VERSION, BudgetExhausted, ProtocolError, digest, plain


def implementation_hash():
    root = Path(__file__).parent.parent
    files = list((root / "r9").rglob("*.py"))
    for name in ("r1", "r2", "p01", "models"):
        files.extend((root / name).glob("*.py"))
    files.append(root / "r3/checkpoint.py")
    return digest({p.relative_to(root).as_posix(): file_digest(p) for p in sorted(files)})


def software_fingerprint(model, config):
    versions = {}
    for p in ("torch", "transformers", "qwen-vl-utils", "av", "Pillow", "numpy", "jsonschema"):
        try:
            versions[p] = importlib.metadata.version(p)
        except importlib.metadata.PackageNotFoundError:
            versions[p] = None
    processor = getattr(model, "processor", None)
    loaded_config = getattr(getattr(model, "model", None), "config", None)
    loaded_commit = getattr(loaded_config, "_commit_hash", None)
    if loaded_commit is not None and loaded_commit != config.model_revision:
        raise ProtocolError("loaded model commit differs from the frozen revision")
    return {
        "pipeline": VERSION,
        "prompts": PROMPT_VERSION,
        "implementation": implementation_hash(),
        "model_class": type(model).__module__ + "." + type(model).__name__,
        "model_path": str(getattr(model, "model_path", "unknown")),
        "revision": config.model_revision,
        "loaded_commit": loaded_commit,
        "processor_class": type(processor).__name__ if processor else None,
        "chat_template_hash": digest(getattr(processor, "chat_template", None)),
        "packages": versions,
        "generation": getattr(model, "generation", {}),
        "model_settings": {
            k: getattr(model, k, None)
            for k in ("dtype", "device_map", "attn_implementation", "revision")
        },
    }


def checkpoint_for(request, config, media, software):
    public = asdict(request)
    for key in ("resume", "checkpoint_path"):
        public.pop(key)
    return Checkpoint(
        request.checkpoint_path,
        {
            "request": public,
            "config": asdict(config),
            "source_sha256": media.source_hash,
            "software": software,
        },
        resume=request.resume,
    )


class Session:
    def __init__(self, model, config, request, data, save):
        self.model, self.config, self.request, self.data, self.save = (
            model,
            config,
            request,
            data,
            save,
        )
        self.receipts = data.setdefault("receipts", [])
        self.jobs = data.setdefault("jobs", {})
        self.started = time.perf_counter()
        self.previous_seconds = data.get("elapsed_seconds", 0.0)
        self.limits = {
            k: min(getattr(config, k), getattr(request, k) or getattr(config, k))
            for k in (
                "max_model_calls",
                "max_unique_source_frames",
                "max_visual_exposures",
                "max_seconds",
            )
        }
        if self.limits["max_model_calls"] < 4:
            raise ProtocolError("R9 needs at least four calls including the terminal reserve")

    def persist(self):
        self.data["elapsed_seconds"] = self.previous_seconds + time.perf_counter() - self.started
        self.save()

    def resources(self):
        def total(key):
            values = [r.get(key) for r in self.receipts]
            return sum(values) if all(isinstance(v, (int, float)) for v in values) else None

        unique = {i for r in self.receipts for i in r["source_frame_ids"]}
        return {
            "limits": self.limits,
            "model_calls": len(self.receipts),
            "unique_source_frames": len(unique),
            "visual_exposures": sum(r["visual_exposures"] for r in self.receipts),
            "repeated_exposures": sum(r["visual_exposures"] for r in self.receipts) - len(unique),
            "crop_exposures": sum(r["crop_exposures"] for r in self.receipts),
            "presented_pixels": sum(r["presented_pixels"] for r in self.receipts),
            "input_tokens": total("input_tokens"),
            "output_tokens": total("output_tokens"),
            "visual_tokens": total("visual_tokens"),
            "model_seconds": total("latency_seconds"),
            "processed_pixels": total("processed_pixels"),
            "end_to_end_seconds": self.previous_seconds + time.perf_counter() - self.started,
            "failed_calls": sum(r["status"] != "ok" for r in self.receipts),
            "geometry_calls": 0,
            "geometry_seconds": 0.0,
        }

    def call(
        self, key, role, payload, prepared=None, evidence=None, validator=None, *, terminal=False
    ):
        evidence = evidence or {}
        schema = role_schema(role, payload)
        visual = {
            alias: {k: v for k, v in e.items() if k != "path"} for alias, e in evidence.items()
        }
        identity = digest({"role": role, "input": payload, "evidence": visual})
        job = self.jobs.setdefault(key, {"identity": identity, "invalid_outputs": 0})
        if job["identity"] != identity:
            raise ProtocolError("stage key reused with different input")
        if "value" in job:
            value = parse(json.dumps(job["value"]), role)
            validate(value, schema)
            if validator:
                validator(value)
            return plain(value)
        while job["invalid_outputs"] <= self.config.format_retries:
            resources = self.resources()
            reserve = 0 if terminal else self.config.terminal_call_reserve
            if len(self.receipts) >= self.limits["max_model_calls"] - reserve:
                raise BudgetExhausted("model call cap / terminal reserve")
            if resources["end_to_end_seconds"] >= self.limits["max_seconds"]:
                raise BudgetExhausted("wall-time budget exhausted")
            source_ids = sorted({e["source_frame_id"] for e in evidence.values()})
            previous = {i for r in self.receipts for i in r["source_frame_ids"]}
            if len(previous | set(source_ids)) > self.limits["max_unique_source_frames"]:
                raise BudgetExhausted("unique source-frame budget exhausted")
            count = len(prepared.frames) if prepared else 0
            if resources["visual_exposures"] + count > self.limits["max_visual_exposures"]:
                raise BudgetExhausted("visual exposure budget exhausted")
            body = {"stage": role, "input": payload, "schema": schema}
            if job["invalid_outputs"]:
                body["format_repair"] = job["last_error"]
            encoded = json.dumps(body, ensure_ascii=False, allow_nan=False)
            parts = list(prepared.parts) if prepared else []
            parts.append({"type": "text", "text": "R9_JSON\n" + encoded})
            receipt = {
                "id": f"call-{len(self.receipts) + 1:04d}",
                "key": key,
                "role": role,
                "status": "started",
                "source_frame_ids": source_ids,
                "visual_exposures": count,
                "crop_exposures": sum(e["id"] != e["source_frame_id"] for e in evidence.values()),
                "presented_pixels": prepared.pixels if prepared else 0,
                "evidence": visual,
                "input_hash": identity,
                "video_frame_metadata": prepared.video_frame_metadata if prepared else [],
            }
            self.receipts.append(receipt)
            self.persist()
            try:
                output = self.model.generate(
                    [
                        {"role": "system", "content": COMMON + "\n" + PROMPTS[role]},
                        {"role": "user", "content": parts},
                    ],
                    max_new_tokens=self.config.max_new_tokens_per_structured_call,
                    temperature=0.0,
                    num_beams=1,
                    input_token_limit=self.config.working_context_target_tokens
                    - self.config.max_new_tokens_per_structured_call,
                    video_frame_metadata=prepared.video_frame_metadata if prepared else [],
                )
                receipt["output"] = output.text
                receipt.update(
                    {
                        k: v
                        for k, v in output.metadata.items()
                        if k
                        in {
                            "input_tokens",
                            "output_tokens",
                            "visual_tokens",
                            "latency_seconds",
                            "gpu_memory",
                            "processed_visual_grids",
                            "processed_pixels",
                            "encoded_video_frames",
                            "video_timing_verified",
                        }
                    }
                )
                value = parse(output.text, role)
                validate(value, schema)
                if validator:
                    validator(value)
                job["value"] = value
                receipt["status"] = "ok"
                self.persist()
                return plain(value)
            except InputContextExceeded as exc:
                receipt.update(
                    status="context_exceeded", input_tokens=exc.actual_tokens, model_executed=False
                )
                self.persist()
                raise
            except ProtocolError as exc:
                job["invalid_outputs"] += 1
                job["last_error"] = str(exc)
                receipt.update(status="invalid_output", error=str(exc))
                self.persist()
            except BaseException as exc:
                receipt.update(status="model_error", error=type(exc).__name__ + ": " + str(exc))
                self.persist()
                raise
        raise ProtocolError(job["last_error"])
