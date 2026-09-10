"""Receipt-before-dispatch accounting, strict JSON repairs and fingerprinted replay."""

import importlib.metadata
import json
import time
from dataclasses import asdict
from pathlib import Path

from qwen3vl_agent.models.qwen3vl import InputContextExceeded, VisualBudgetExceeded
from qwen3vl_agent.r3.checkpoint import Checkpoint, file_digest

from .config import REVISION
from .prompts import COMMON, PROMPTS
from .schema import SCHEMAS, parse
from .types import (
    VERSION,
    BudgetExhausted,
    ContextOverflow,
    ModelFailure,
    ProtocolError,
    digest,
    plain,
)


def software_fingerprint(model, config):
    root = Path(__file__).parent.parent
    paths = list((root / "r6").glob("*.py"))
    paths += list((root / "models").glob("*.py"))
    paths += list((root / "r1").glob("*.py"))
    paths += list((root / "p01").glob("*.py"))
    paths += [
        root / "temporal_media.py",
        root / "r1/media.py",
        root / "r3/checkpoint.py",
        root / "r4/providers.py",
        root / "p01/config.py",
        root / "r4/types.py",
        root / "coarse_to_fine/cache.py",
        root / "coarse_to_fine/types.py",
    ]
    packages = {}
    for name in ("torch", "transformers", "qwen-vl-utils", "av", "Pillow", "jsonschema"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    loaded = getattr(getattr(getattr(model, "model", None), "config", None), "_commit_hash", None)
    if loaded is not None and loaded != REVISION:
        raise ProtocolError("loaded model differs from frozen revision")
    return {
        "pipeline": VERSION,
        "implementation": digest(
            {p.relative_to(root).as_posix(): file_digest(p) for p in sorted(paths)}
        ),
        "packages": packages,
        "model_class": type(model).__module__ + "." + type(model).__name__,
        "model_path": str(getattr(model, "model_path", "fake")),
        "revision": config.model_revision,
        "loaded_commit": loaded,
        "chat_template": digest(getattr(getattr(model, "processor", None), "chat_template", None)),
        "generation": getattr(model, "generation", {}),
    }


def checkpoint_for(request, config, media, texts, software):
    public = asdict(request)
    public.pop("checkpoint_path")
    public.pop("resume")
    return Checkpoint(
        request.checkpoint_path,
        {
            "request": public,
            "config": asdict(config),
            "protocol": asdict(media.contract),
            "text_files": texts.file_hashes,
            "software": software,
        },
        resume=request.resume,
    )


class Session:
    def __init__(self, model, config, state, save):
        self.model, self.config, self.state, self.save = model, config, state, save
        self.jobs = state.setdefault("jobs", {})
        self.receipts = state.setdefault("receipts", [])
        for receipt in self.receipts:
            if receipt["status"] == "started":
                receipt["status"] = "interrupted"
        self.tools = state.setdefault("tool_receipts", [])
        self.started = time.perf_counter()
        self.previous_seconds = state.get("elapsed_seconds", 0.0)

    def persist(self):
        self.state["elapsed_seconds"] = self.previous_seconds + time.perf_counter() - self.started
        self.save()

    def remaining(self, *, terminal=False):
        used = sum(r["role"] == "verifier" for r in self.receipts)
        reserve = 0 if terminal else max(0, self.config.reserve_model_calls_for_verification - used)
        return max(0, self.config.max_model_calls_total - len(self.receipts) - reserve)

    def resources(self):
        def total(key):
            values = [r.get(key) for r in self.receipts]
            return sum(values) if all(isinstance(v, (int, float)) for v in values) else None

        frames = [fid for r in self.receipts for fid in r["source_frame_ids"]]
        exposed = sum(r["visual_exposures"] for r in self.receipts)
        return {
            "model_calls": len(self.receipts),
            "model_call_limit": self.config.max_model_calls_total,
            "tool_calls": len(self.tools),
            "unique_source_frames": len(set(frames)),
            "visual_exposures": exposed,
            "repeated_exposures": exposed - len(set(frames)),
            "crop_exposures": sum(r["crop_exposures"] for r in self.receipts),
            "input_tokens": total("input_tokens"),
            "output_tokens": total("output_tokens"),
            "visual_tokens": total("visual_tokens"),
            "processed_pixels": total("processed_pixels"),
            "presented_pixels": sum(r.get("presented_pixels", 0) for r in self.receipts),
            "encoded_video_frames": sum(
                sum(r.get("encoded_video_frames", [])) for r in self.receipts
            ),
            "encoded_video_accounting_available": all(
                not r.get("video_frame_metadata") or "encoded_video_frames" in r
                for r in self.receipts
            ),
            "model_seconds": total("latency_seconds"),
            "text_characters": sum(r["text_characters"] for r in self.receipts),
            "audio_seconds": 0,
            "failed_calls": sum(r["status"] != "ok" for r in self.receipts),
            "end_to_end_seconds": self.previous_seconds + time.perf_counter() - self.started,
        }

    def tool(self, kind, payload, fn):
        receipt = {"kind": kind, "request": plain(payload), "status": "started"}
        self.tools.append(receipt)
        self.persist()
        started = time.perf_counter()
        try:
            result = fn()
            receipt["status"] = "ok"
            return result
        except (OSError, RuntimeError, ValueError, KeyError, TypeError) as exc:
            receipt.update(status="failed", error=str(exc))
            raise
        finally:
            receipt["seconds"] = time.perf_counter() - started
            self.persist()

    def call(
        self, key, role, payload, *, prepared=None, sources=None, validator=None, terminal=False
    ):
        sources = sources or {}
        visible = {
            alias: {k: v for k, v in s.items() if k != "path"} for alias, s in sources.items()
        }
        identity = digest(
            {
                "role": role,
                "payload": payload,
                "sources": visible,
                "pixels": prepared.sizes if prepared else [],
            }
        )
        job = self.jobs.setdefault(key, {"identity": identity, "attempts": 0})
        if job["identity"] != identity:
            raise ProtocolError("replay stage input changed")
        if "value" in job:
            value = parse(json.dumps(job["value"]), role)
            if validator:
                validator(value)
            return plain(value)
        if job.get("overflow"):
            raise ContextOverflow("replaying recorded processor overflow")
        while job["attempts"] <= self.config.schema_repair_attempts_per_call:
            if not self.remaining(terminal=terminal):
                raise BudgetExhausted("model call budget / reserved verification calls")
            job["attempts"] += 1
            prompt = {
                "role": role,
                "input": payload,
                "source_manifest": visible,
                "response_schema": SCHEMAS[role],
            }
            if job.get("error"):
                prompt["format_repair"] = {
                    "error": job["error"],
                    "instruction": "Regenerate from the same inputs.",
                }
            text = json.dumps(prompt, ensure_ascii=False)
            content = [{"type": "text", "text": text}, *(prepared.parts if prepared else [])]
            receipt = {
                "key": key,
                "role": role,
                "attempt": job["attempts"],
                "status": "started",
                "input_digest": identity,
                "source_ids": sorted(s["id"] for s in sources.values()),
                "source_manifest": visible,
                "prepared_sizes": prepared.sizes if prepared else [],
                "video_frame_metadata": prepared.video_frame_metadata if prepared else [],
                "presented_pixels": prepared.pixels if prepared else 0,
                "source_frame_ids": sorted(
                    {s["source_frame_id"] for s in sources.values() if s["modality"] == "video"}
                ),
                "visual_exposures": len(prepared.frames) if prepared else 0,
                "crop_exposures": sum(
                    s["id"] != s["source_frame_id"]
                    for s in sources.values()
                    if s["modality"] == "video"
                ),
                "text_characters": len(text),
                "model_executed": None,
            }
            self.receipts.append(receipt)
            self.persist()
            started = time.perf_counter()
            try:
                limit_role = "relation_checker" if role == "answer" else role
                result = self.model.generate(
                    [
                        {"role": "system", "content": COMMON + "\n" + PROMPTS[role]},
                        {"role": "user", "content": content},
                    ],
                    temperature=0.0,
                    do_sample=False,
                    max_new_tokens=getattr(self.config, "max_new_tokens_" + limit_role),
                    input_token_limit=self.config.max_total_input_tokens_per_call,
                    visual_token_limit=self.config.max_visual_tokens_per_call,
                    **(
                        {"video_frame_metadata": prepared.video_frame_metadata}
                        if prepared and prepared.video_frame_metadata
                        else {}
                    ),
                )
                receipt.update(
                    {
                        k: v
                        for k, v in result.metadata.items()
                        if k
                        in {
                            "input_tokens",
                            "output_tokens",
                            "visual_tokens",
                            "processed_pixels",
                            "latency_seconds",
                            "preparation",
                            "encoded_video_frames",
                            "gpu_memory",
                        }
                    }
                )
                receipt.update(raw_output=result.text, model_executed=True)
                value = parse(result.text, role)
                if validator:
                    validator(value)
                receipt["status"] = "ok"
                job["value"] = plain(value)
                self.persist()
                return value
            except InputContextExceeded as exc:
                receipt.update(
                    status="context_exceeded", input_tokens=exc.actual_tokens, model_executed=False
                )
                job["overflow"] = True
                raise ContextOverflow(str(exc)) from exc
            except VisualBudgetExceeded as exc:
                receipt.update(status="visual_budget_exceeded", model_executed=False)
                job["overflow"] = True
                raise ContextOverflow(str(exc)) from exc
            except ProtocolError as exc:
                job["error"] = str(exc)
                receipt.update(status="invalid_output", error=str(exc))
            except (RuntimeError, OSError, ValueError, TypeError, KeyError) as exc:
                receipt.update(status="model_error", error=f"{type(exc).__name__}: {exc}")
                # The model adapter raises ValueError for its measured visual budget gate.
                if "budget" in str(exc).lower() and "visual" in str(exc).lower():
                    job["overflow"] = True
                    raise ContextOverflow(str(exc)) from exc
                raise ModelFailure(str(exc)) from exc
            finally:
                receipt["attempt_seconds"] = time.perf_counter() - started
                self.persist()
        raise ProtocolError(job.get("error", "interrupted/failed attempts exhausted"))
