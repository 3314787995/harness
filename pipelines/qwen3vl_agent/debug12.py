"""Small, journaled runner for the fixed R1/R3/R4/R5 visual debug set."""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import importlib.metadata
import json
import os
import platform
import re
import sys
import time
import traceback
import uuid
from collections import Counter
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from qwen3vl_agent.config import load_config
from qwen3vl_agent.models.base import BaseVideoModel
from qwen3vl_agent.r1_v3.version import POLICY_ID as R1_V3_POLICY_ID

PROJECT = Path(__file__).resolve().parents[1]
PIPELINES = ("R1", "R3", "R4", "R5")
REQUEST_FIELDS = {"request_id", "video_id", "video_path", "question", "choices",
                  "available_modalities", "output_protocol"}
VERSION = "r1345-debug12-v1"


class FatalModelError(BaseException):
    """Escape controllers' broad Exception fallbacks after an engine failure."""


class PipelineExecutionError(RuntimeError):
    """A saved terminal pipeline failure; restarting unchanged must not spend more calls."""


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False, default=str)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def inside(root: Path, relative: str) -> Path:
    relative = PurePosixPath(relative)
    if relative.is_absolute() or ".." in relative.parts or ":" in str(relative) or "\\" in str(relative):
        raise ValueError(f"Expected portable relative path: {relative}")
    path = (root / relative).resolve()
    if root.resolve() not in path.parents or not path.is_file():
        raise ValueError(f"Missing file or path outside root: {relative}")
    return path


@dataclass
class DebugCase:
    row: dict[str, Any]
    media: dict[str, Any]
    evaluation: dict[str, Any]

    @property
    def id(self):
        return self.row["request_id"]

    @property
    def pipeline(self):
        return self.row["pipeline_id"]

    def request(self, output: Path | None = None, *, resume: bool = False):
        # Explicit allowlist: even a misplaced gold/review field cannot enter solve().
        fields = REQUEST_FIELDS | ({"query_spec"} if self.pipeline == "R3" else set())
        data = {key: copy.deepcopy(self.row[key]) for key in fields if key in self.row}
        if self.pipeline != "R1" and output is not None:
            checkpoint = output / "checkpoints" / f"{self.id}.jsonl"
            data.update(checkpoint_path=str(checkpoint), resume=resume and checkpoint.exists())
        types = importlib.import_module(f"qwen3vl_agent.{self.pipeline.lower()}.types")
        return getattr(types, self.pipeline + "Request")(**data)


def load_dataset(root: Path) -> list[DebugCase]:
    root = root.resolve()
    manifest = read_json(root / "manifest.json")
    if manifest["schema_version"] != VERSION:
        raise ValueError("Unsupported debug manifest")
    for relative, expected in manifest["files"].items():
        if file_hash(inside(root, relative)) != expected:
            raise ValueError(f"Dataset metadata hash mismatch: {relative}")
    answers_list = read_jsonl(root / "answers.jsonl")
    answers = {x["request_id"]: x for x in answers_list}
    if len(answers) != len(answers_list):
        raise ValueError("Duplicate answer IDs")
    media = manifest["media"]
    if len(media) != 9:
        raise ValueError("Expected exactly 9 videos")
    for info in media.values():
        video = inside(root, info["path"])
        if video.stat().st_size != info["bytes"] or file_hash(video) != info["sha256"]:
            raise ValueError(f"Video hash/size mismatch: {info['path']}")
    cases = []
    for row in read_jsonl(root / "requests.jsonl"):
        optional = {"query_spec"} if row.get("pipeline_id") == "R3" else set()
        if not REQUEST_FIELDS | {"pipeline_id"} <= set(row) or set(row) - REQUEST_FIELDS - {"pipeline_id"} - optional:
            raise ValueError("Inference manifest contains unexpected or missing fields")
        if row["pipeline_id"] not in PIPELINES or not re.fullmatch(r"R[1345]-[A-Za-z0-9-]+", row["request_id"]):
            raise ValueError("Invalid pipeline or request ID")
        info = media[row["video_id"]]
        if row["video_path"] != info["path"]:
            raise ValueError("Request/media mapping mismatch")
        row = {**row, "video_path": str(inside(root, row["video_path"]))}
        if row["available_modalities"] != ["video", "screen_text"] or row["output_protocol"] != "multiple_choice":
            raise ValueError("The debug set is visual-only multiple choice")
        answer = answers[row["request_id"]]
        choices = row["choices"]
        raw = answer["original_record"]
        expected = ([{"label": text[0], "text": text[3:]} for text in raw["options"]]
                    if "options" in raw else [{"label": chr(65 + i), "text": text} for i, text in enumerate(raw["candidates"])])
        label = raw["answer"] if "options" in raw else chr(65 + raw["candidates"].index(raw["answer"]))
        if choices != expected or row["question"] != raw["question"] or answer["answer_label"] != label:
            raise ValueError("Official question/options/answer mapping changed")
        case = DebugCase(row, info, answer)
        case.request()  # Validate the actual pipeline's public constructor, without a model.
        cases.append(case)
    if (len(cases) != 12 or len({c.id for c in cases}) != 12
            or set(answers) != {c.id for c in cases}
            or {c.row["video_id"] for c in cases} != set(media)
            or Counter(c.pipeline for c in cases) != Counter({p: 3 for p in PIPELINES})):
        raise ValueError("Expected twelve unique questions, three per pipeline, using all nine videos")
    return cases


def configurations(override: Path, output: Path, *, r1_version: str = "v3") -> dict:
    if r1_version != "v3":
        raise ValueError("Expected R1 version v1, v2 or v3")
    shared = load_config(override)["model"]
    if (shared.get("device_map") != "balanced" or shared.get("required_cuda_devices") != [0, 1]
            or shared.get("max_memory") != {0: "20GiB", 1: "20GiB"}
            or shared.get("forbid_offload") is not True or shared.get("dtype") != "bfloat16"
            or shared.get("attn_implementation") != "flash_attention_2"
            or shared.get("generation", {}).get("temperature") != 0.0):
        raise ValueError("Expected the locked BF16/FA2, balanced 2x4090D profile")
    if str(shared["path"]).replace("\\", "/").rstrip("/").split("/")[-1] != "Qwen3-VL-8B-Instruct":
        raise ValueError("Use Qwen3-VL-8B-Instruct; local directory must retain that name")
    result = {}
    for p in PIPELINES:
        key = f"r1_{r1_version}" if p == "R1" and r1_version != "v1" else p.lower()
        class_prefix = "R1" + r1_version.upper() if key in {"r1_v2", "r1_v3"} else p
        config = load_config(PROJECT / f"configs/{key}_8b.yaml")
        config["model"].update(copy.deepcopy(shared))
        # Keep evidence/cache with the run so it is retrievable after server shutdown.
        config[key]["media"]["cache_dir"] = str(output / "cache" / key)
        cls = getattr(importlib.import_module(f"qwen3vl_agent.{key}.config"), class_prefix + "Config")
        config[key] = asdict(cls.from_mapping(config[key]))
        result[p] = config
    if len({json_hash(c["model"]) for c in result.values()}) != 1:
        raise ValueError("All pipelines must use the same model configuration")
    return result


def preflight(cases: list[DebugCase], configs: dict) -> dict:
    import av

    videos = {}
    for case in cases:
        if case.row["video_id"] in videos:
            continue
        if case.pipeline == "R1" and ({"r1_v2", "r1_v3"} & configs["R1"].keys()):
            from qwen3vl_agent.r1.control import resolve_scopes
            from qwen3vl_agent.r1_v2.config import R1V2Config
            from qwen3vl_agent.r1_v2.media import R1V2IndexBuilder, index_report

            key = "r1_v3" if "r1_v3" in configs["R1"] else "r1_v2"
            builder = R1V2IndexBuilder(R1V2Config.from_mapping(configs["R1"][key]).media)
            metadata = builder.probe(case.row["video_path"])
            allowed, _ = resolve_scopes(case.request(), metadata.duration_seconds)
            index = builder.prepare_interval(case.row["video_path"], allowed, metadata=metadata)
            videos[case.row["video_id"]] = {
                "ready": True, "duration_seconds": metadata.duration_seconds,
                "pipeline_version": key.removeprefix("r1_"), "navigation_index": index_report(index),
            }
            continue
        with av.open(case.row["video_path"]) as container:
            stream = container.streams.video[0]
            duration = container.duration / av.time_base
            timestamps = []
            for target in (0.0, duration / 2, max(0.0, duration - 0.5)):
                container.seek(int(target / float(stream.time_base)), stream=stream)
                frame = next((f for f in container.decode(stream) if f.time is not None and f.time >= target - 0.1), None)
                if frame is None:
                    raise ValueError(f"Decode failed at {target}: {case.id}")
                timestamps.append(float(frame.time))
            videos[case.row["video_id"]] = {"duration_seconds": duration, "decoded_pts": timestamps, "ready": True}
    pipelines = {}
    for p in PIPELINES:
        entries = [(c.request(), {}) for c in cases if c.pipeline == p]
        if not entries:
            continue
        if p == "R1":
            pipelines[p] = {"requests": [{"request_id": r.request_id, "status": "ready"} for r, _ in entries]}
        else:
            evaluator = importlib.import_module(f"qwen3vl_agent.{p.lower()}.evaluate")
            cls = getattr(importlib.import_module(f"qwen3vl_agent.{p.lower()}.config"), p + "Config")
            pipelines[p] = evaluator.preflight(entries, cls.from_mapping(configs[p][p.lower()]))
    ready = all(r["status"] == "ready" for report in pipelines.values() for r in report["requests"])
    return {"ready": ready, "question_count": len(cases), "video_count": len(videos), "videos": videos,
            "pipelines": pipelines, "model_calls": 0, "torch_imported": "torch" in sys.modules,
            "pipeline_versions": pipeline_versions(configs),
            "note": "No model calls. R1 V2/V3 checks the real bounded shot index; other pipelines use metadata/decode preflight. No semantic accuracy claim."}


def environment() -> dict:
    packages = {}
    for name in ("torch", "torchvision", "transformers", "accelerate", "flash-attn", "qwen-vl-utils", "av", "Pillow", "PyYAML", "jsonschema"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {"python": platform.python_version(), "platform": platform.platform(), "packages": packages,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}


def r1_execution_identity(configs: dict) -> dict:
    return ({"r1_protocol": R1_V3_POLICY_ID, "r1_execution_branch": "v3_visual"}
            if "r1_v3" in configs.get("R1", {}) else {})


def identity(root: Path, configs: dict) -> dict:
    weights = Path(configs["R1"]["model"]["path"])
    model_files = {}
    if weights.is_dir():
        for path in sorted(weights.glob("*")):
            if path.is_file() and path.suffix in {".json", ".safetensors"}:
                model_files[path.name] = ({"bytes": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns}
                                          if path.suffix == ".safetensors" else file_hash(path))
    return {"runner": VERSION, "data_root": str(root), "manifest_sha256": file_hash(root / "manifest.json"),
            "configs": configs, "pipeline_versions": pipeline_versions(configs),
            **r1_execution_identity(configs),
            "environment": environment(), "model_files": model_files,
            "code": {p.relative_to(PROJECT).as_posix(): file_hash(p) for p in sorted((PROJECT / "qwen3vl_agent").rglob("*.py"))}}


def memory_snapshot(reset: bool = False) -> dict:
    torch = sys.modules.get("torch")
    if torch is None or not torch.cuda.is_available():
        return {}
    try:
        if reset:
            for i in range(torch.cuda.device_count()):
                torch.cuda.reset_peak_memory_stats(i)
        return {str(i): {"allocated_bytes": torch.cuda.memory_allocated(i), "reserved_bytes": torch.cuda.memory_reserved(i),
                         "max_allocated_bytes": torch.cuda.max_memory_allocated(i), "max_reserved_bytes": torch.cuda.max_memory_reserved(i)}
                for i in range(torch.cuda.device_count())}
    except Exception as exc:
        return {"measurement_error": str(exc)}


class JournaledModel(BaseVideoModel):
    def __init__(self, inner: BaseVideoModel, output: Path):
        super().__init__(inner.model_path, device=inner.device, dtype=inner.dtype)
        self.inner, self.output = inner, output
        self.question_id, self.sequence = "unassigned", 0

    def __getattr__(self, key):
        return getattr(self.inner, key)

    def load(self):
        if not self.inner.is_loaded:
            self.inner.load()
        self._loaded = True

    def unload(self):
        self.inner.unload()
        self._loaded = False

    def generate(self, messages, *, videos=None, images=None, **kwargs):
        directory = self.output / "calls" / self.question_id
        self.sequence += 1
        while (directory / f"{self.sequence:05d}.json").exists():
            self.sequence += 1
        path = directory / f"{self.sequence:05d}.json"
        role = re.search(r"R[1345]:[a-z_]+", str(messages))
        record = {"request_id": self.question_id, "state": "started", "role": role[0] if role else "navigation",
                  "messages": messages, "videos": videos, "images": images, "generation_kwargs": kwargs}
        atomic_json(path, record)
        print(f"[{self.question_id}] {self.sequence:05d} {record['role']}", flush=True)
        started = time.perf_counter()
        try:
            result = self.inner.generate(messages, videos=videos, images=images, **kwargs)
            record.update(state="completed", output=result.text, metadata=result.metadata)
            return result
        except BaseException as exc:
            record.update(state="error", error_type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc())
            oom = type(exc).__name__ == "OutOfMemoryError" or "out of memory" in str(exc).lower()
            # Existing pipelines may recover OOM by reducing media or splitting a window.
            # Other engine failures must not become an apparently completed fallback answer.
            from qwen3vl_agent.models.qwen3vl import VisualBudgetExceeded
            if isinstance(exc, Exception) and not oom and not isinstance(exc, VisualBudgetExceeded):
                raise FatalModelError(f"{type(exc).__name__}: {exc}") from exc
            raise
        finally:
            record.update(wall_seconds=time.perf_counter() - started, gpu_memory=memory_snapshot())
            atomic_json(path, record)


@contextmanager
def run_lock(output: Path):
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".run.lock").open("a+b") as stream:
        if stream.tell() == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream, fcntl.LOCK_UN)


def gpu_static_check(config: dict) -> dict:
    from qwen3vl_agent.p01.preflight import _check_gpu, _check_model_path, _check_packages

    checks = _check_packages("flash_attention_2") + _check_gpu(23.0, required_gpus=2)
    checks.append(_check_model_path(config["path"]))
    versions = environment()["packages"]
    expected = {"torch": "2.8.0+cu128", "torchvision": "0.23.0+cu128",
                "transformers": "4.57.6", "flash-attn": "2.8.3.post1"}
    mismatches = {p: {"actual": versions[p], "expected": v} for p, v in expected.items() if versions[p] != v}
    if mismatches:
        raise RuntimeError(f"Pinned runtime mismatch; run prepare.sh: {mismatches}")
    import torch
    if torch.cuda.device_count() != 2 or any("4090" not in torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())):
        raise RuntimeError("Expose exactly the intended two 4090D cards via CUDA_VISIBLE_DEVICES=0,1")
    report = {"ready": all(c.status != "error" for c in checks), "checks": [asdict(c) for c in checks]}
    if not report["ready"]:
        raise RuntimeError(json.dumps(report, ensure_ascii=False))
    return report


def pipeline_versions(configs):
    from qwen3vl_agent.r3.prompts import PROMPT_VERSION as r3_version
    from qwen3vl_agent.r4.prompts import VERSION as r4_version
    from qwen3vl_agent.r5.observation import PROTOCOL_VERSION as r5_version

    return {"R1": "v3" if "r1_v3" in configs.get("R1", {}) else "v2" if "r1_v2" in configs.get("R1", {}) else "v1", "R3": r3_version, "R4": r4_version, "R5": r5_version}


def default_agent(pipeline, model, config):
    if pipeline == "R1":
        from qwen3vl_agent.r1_v3 import R1V3VideoAgent
        return R1V3VideoAgent(model, config=config.get("r1_v3", config.get("r1")))
    module = importlib.import_module(f"qwen3vl_agent.{pipeline.lower()}.agent")
    return getattr(module, pipeline + "VideoAgent")(model, config=config[pipeline.lower()])


def ordered(cases: list[DebugCase]) -> list[DebugCase]:
    gates = [next(c for c in cases if c.pipeline == p) for p in PIPELINES if any(c.pipeline == p for c in cases)]
    return gates + [c for c in cases if c.id not in {g.id for g in gates}]


def write_summary(cases, output, signature):
    records = []
    for case in ordered(cases):
        path = output / "items" / f"{case.id}.json"
        if path.exists():
            row = read_json(path)
            if row.get("run_signature") != signature or row.get("request_id") != case.id:
                raise ValueError("Saved item belongs to another run")
            if row.get("status") in {"completed", "failed"}:
                records.append(row)
    predictions = [{k: r.get(k) for k in ("request_id", "pipeline_id", "status", "result_status", "answer_mode", "evidence_status", "prediction", "answer_label", "correct", "support_level", "answer_basis", "verification_status", "evidence_ref_count", "completion_state", "failure", "wall_seconds", "model_calls", "gpu_memory")} for r in records]

    def has_evidence(row):
        if row.get("pipeline_id") == "R3":
            return row.get("answer_basis") in {"visual_direct", "visual_reduction", "partial_choice"}
        if row.get("pipeline_id") == "R5":
            return bool(row.get("evidence_ref_count", len(row.get("result", {}).get("evidence_refs", []))))
        return row.get("support_level") in {"supported", "partial"} and row.get("answer_basis") != "unbacked_fallback"

    def counts(rows, planned):
        correct = sum(r["correct"] is True for r in rows)
        return {"planned": planned, "completed": sum(r["status"] == "completed" for r in rows), "terminal_count": len(rows),
                "answered": sum(r["prediction"] is not None for r in rows),
                "no_prediction": sum(r["prediction"] is None for r in rows),
                "answers_with_evidence": sum(r["prediction"] is not None and has_evidence(r) for r in rows),
                "unverified_answers": sum(r["prediction"] is not None and r.get("verification_status") == "not_performed" for r in rows),
                "unbacked_fallbacks": sum(r.get("answer_basis") in {"unbacked_fallback", "forced_choice"} for r in rows),
                "supported_correct": sum(r.get("correct") is True and r.get("support_level") == "supported"
                    and r.get("completion_state") == "complete" and r.get("answer_basis") not in {"unbacked_fallback", "forced_choice", "partial_choice", "protocol_failure"} for r in rows),
                "partial_correct": sum(r.get("correct") is True and r.get("support_level") == "partial"
                    and r.get("answer_basis") != "unbacked_fallback" for r in rows),
                "unbacked_fallback_correct": sum(r.get("correct") is True and r.get("answer_basis") in {"unbacked_fallback", "forced_choice"} for r in rows),
                "r3_supported_accuracy": sum(r.get("correct") is True and r.get("answer_basis") in {"visual_direct","visual_reduction"} for r in rows if r.get("pipeline_id")=="R3") / max(1, sum(r.get("pipeline_id")=="R3" for r in rows)),
                "r3_fallback_ratio": sum(r.get("answer_basis") in {"partial_choice","forced_choice"} for r in rows if r.get("pipeline_id")=="R3") / max(1, sum(r.get("pipeline_id")=="R3" for r in rows)),
                "answer_basis_counts": {basis: sum(r.get("answer_basis")==basis for r in rows) for basis in ("visual_direct","visual_reduction","partial_choice","forced_choice","unresolved","protocol_failure")},
                "r4_answer_mode_counts": {mode: sum(r.get("pipeline_id") == "R4" and r.get("answer_mode") == mode for r in rows) for mode in ("determined","best_effort","none")},
                "r4_result_status_counts": {s: sum(r.get("pipeline_id") == "R4" and r.get("result_status") == s for r in rows)
                    for s in ("supported", "unresolved", "budget_exhausted", "execution_failed")},
                "failed": sum(r["status"] == "failed" for r in rows), "correct": correct,
                "answer_match_rate_planned": correct / planned if planned else None}

    summary = {**counts(records, len(cases)), "pending": [c.id for c in cases if c.id not in {r['request_id'] for r in records}],
               "model_calls": sum(r["model_calls"] for r in records),
               "by_pipeline": {p: counts([r for r in records if r["pipeline_id"] == p], sum(c.pipeline == p for c in cases)) for p in PIPELINES},
               "note": "12 curated debug cases; diagnostic answer match, not an official benchmark score."}
    atomic_json(output / "summary.json", summary)
    atomic_json(output / "diagnostics.json", predictions)
    atomic_json(output / "r4_missing_predictions.json", [{"request_id": c.id,
        "reason": next((r.get("result_status") or r["status"] for r in records if r["request_id"] == c.id), "pending")}
        for c in cases if c.pipeline == "R4" and not any(r["request_id"] == c.id and r.get("prediction") is not None for r in records)])
    (output / "r4_native_predictions.jsonl").write_text("".join(json.dumps({"request_id": r["request_id"], "prediction": r["prediction"]}) + "\n"
        for r in records if r["pipeline_id"] == "R4" and r["prediction"] is not None), encoding="utf-8")
    (output / "predictions.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in predictions), encoding="utf-8")
    return summary


def run_cases(cases, selected, configs, output, run_identity, *, resume=False, model_factory=None,
              agent_factory=default_agent, gpu_check=gpu_static_check):
    output = output.resolve()
    run_identity = {**run_identity, **r1_execution_identity(configs),
                    "pipeline_versions": pipeline_versions(configs),
                    "planned_request_ids": [c.id for c in ordered(cases)]}
    signature = json_hash(run_identity)
    with run_lock(output):
        plan_path = output / "run_plan.json"
        if plan_path.exists():
            if not resume or read_json(plan_path)["signature"] != signature:
                raise ValueError("Existing run needs --resume and identical data, code, model and configuration")
        else:
            if (output / "items").exists() or (output / "checkpoints").exists():
                raise ValueError("New run directory contains previous inference state")
            atomic_json(plan_path, {"signature": signature, "identity": run_identity, "order": [c.id for c in ordered(cases)]})
        pending = []
        for case in ordered(selected):
            path = output / "items" / f"{case.id}.json"
            if path.exists():
                previous = read_json(path)
                if previous.get("run_signature") != signature or previous.get("request_id") != case.id:
                    raise ValueError("Mismatched saved item")
                if previous.get("status") == "failed":
                    if case.pipeline == "R4":
                        print(f"[{case.id}] saved terminal failure; skipped without retry", flush=True)
                        continue
                    write_summary(cases, output, signature)
                    raise PipelineExecutionError(f"Saved terminal failure for {case.id}; inspect it and use a new result directory after fixing code/config. No model loaded.")
                if previous.get("status") == "completed":
                    print(f"[{case.id}] completed; skipped", flush=True)
                    continue
            pending.append(case)
        if not pending:
            return write_summary(cases, output, signature)
        if model_factory is None:
            from qwen3vl_agent.factory import build_model
            model_factory = build_model
        model, current, request, started = None, None, None, None
        try:
            gpu_report = {"phase": "static", "ready": False, "environment": environment()}
            atomic_json(output / "gpu_preflight.json", gpu_report)
            if gpu_check:
                gpu_report["static"] = gpu_check(configs["R1"]["model"])
            inner = model_factory(configs["R1"]["model"])
            model = JournaledModel(inner, output)
            gpu_report["phase"] = "loading_model"
            atomic_json(output / "gpu_preflight.json", gpu_report)
            model.load()
            gpu_report.update(phase="loaded", ready=True, hf_device_map=getattr(inner, "hf_device_map", {}), gpu_memory=memory_snapshot())
            atomic_json(output / "gpu_preflight.json", gpu_report)
            atomic_json(output / "environment.json", environment())
            agents = {p: agent_factory(p, model, configs[p]) for p in PIPELINES if any(c.pipeline == p for c in pending)}
            for current in pending:
                request = None
                model.question_id, model.sequence = current.id, 0
                before = memory_snapshot(reset=True)
                started = time.perf_counter()
                request = current.request(output, resume=resume)
                try:
                    result = agents[current.pipeline].solve(request)
                except Exception as exc:
                    # R4 question-local faults are isolated. Model/device fatal errors are BaseExceptions
                    # from JournaledModel, and storage failures still propagate at the write boundary.
                    if current.pipeline != "R4" or isinstance(exc, OSError):
                        raise
                    from qwen3vl_agent.r4.types import R4Result
                    result = R4Result(None, {"results": []}, "execution_error", "unsupported", "pipeline_failure",
                        {}, [], [str(exc)], {}, {}, {"traceback": traceback.format_exc()},
                        {"stage": "execution", "code": type(exc).__name__, "message": str(exc)}, "execution_failed")
                trace = result.to_dict()
                failure = trace.get("failure") if current.pipeline == "R4" else None
                # Some public agents report input failures as normal result objects.
                issues = (trace.get("unresolved_reasons", []) + trace.get("unresolved_items", [])
                          + trace.get("issues", []) + trace.get("value_state", {}).get("issues", []))
                if (result.completion_state == "input_error"
                        or any(str(issue).startswith("input_error:") for issue in issues)):
                    if current.pipeline == "R4":
                        failure = failure or {"stage": "input", "code": "input_error", "message": str(issues)}
                    else:
                        atomic_json(output / "partial" / f"{current.id}.json", trace)
                        raise RuntimeError(f"Native input failure: {current.id}: {issues}")
                prediction = None if failure and not (current.pipeline == "R4" and trace.get("answer_mode") == "best_effort") else result.prediction
                if current.pipeline == "R4" and trace.get("result_status") in {"unresolved", "budget_exhausted", "execution_failed"} and trace.get("answer_mode") != "best_effort":
                    prediction = None
                if current.pipeline == "R5" and prediction == "":
                    prediction = None
                if (prediction is not None or current.pipeline not in {"R3", "R4", "R5"}) and prediction not in {c["label"] for c in current.row["choices"]}:
                    if current.pipeline == "R4":
                        failure = {"stage": "output", "code": "invalid_prediction", "message": f"Invalid native prediction: {prediction!r}"}
                        prediction = None
                    else:
                        atomic_json(output / "partial" / f"{current.id}.json", trace)
                        raise RuntimeError(f"Invalid native prediction: {prediction!r}")
                resources = trace.get("resources", {})
                record = {"run_signature": signature, "status": "failed" if failure else "completed", "request_id": current.id,
                          "pipeline_id": current.pipeline, "prediction": prediction,
                          "pipeline_version": pipeline_versions(configs).get(current.pipeline),
                          "answer_label": current.evaluation["answer_label"], "correct": prediction == current.evaluation["answer_label"] if prediction is not None else (False if current.pipeline == "R3" else None),
                          "support_level": result.support_level, "answer_basis": trace.get("answer_basis"), "completion_state": result.completion_state,
                          "result_status": ("execution_failed" if failure else trace.get("result_status", "supported" if prediction is not None else "unresolved")) if current.pipeline == "R4" else trace.get("result_status"),
                          "verification_status": trace.get("verification_status"), "evidence_ref_count": len(trace.get("evidence_refs", [])),
                          "wall_seconds": time.perf_counter() - started, "model_calls": resources.get("model_calls", len(resources.get("calls", []))),
                          "gpu_memory_before": before, "gpu_memory": memory_snapshot(), "request": asdict(request),
                          "evaluation": current.evaluation, "result": trace, "failure": failure,
                          **({"answer_mode":trace.get("answer_mode","none"), "evidence_status":trace.get("evidence_status")} if current.pipeline == "R4" else {})}
                atomic_json(output / "items" / f"{current.id}.json", record)
                write_summary(cases, output, signature)
                if failure and current.pipeline != "R4":
                    raise PipelineExecutionError(f"{current.id}: {failure['stage']}: {failure.get('message', failure['code'])}")
                print(f"[{current.id}] prediction={prediction} correct={record['correct']} support={result.support_level}", flush=True)
            return write_summary(cases, output, signature)
        except BaseException as exc:
            failure = {"run_signature": signature, "request_id": current.id if current else None,
                       "error_type": type(exc).__name__, "error": str(exc), "traceback": traceback.format_exc(), "gpu_memory": memory_snapshot()}
            atomic_json(output / "failures" / f"{uuid.uuid4().hex}.json", failure)
            atomic_json(output / "fatal_error.json", failure)
            if current and current.pipeline in {"R3", "R4"} and not (output / "items" / f"{current.id}.json").exists():
                atomic_json(output / "items" / f"{current.id}.json", {
                    "run_signature": signature, "status": "failed", "request_id": current.id,
                    "pipeline_id": current.pipeline, "pipeline_version": pipeline_versions(configs)[current.pipeline],
                    "prediction": None, "answer_label": current.evaluation["answer_label"], "correct": None,
                    "support_level": "none" if current.pipeline == "R3" else "unsupported", "completion_state": "execution_error",
                    **({"result_status": "execution_failed"} if current.pipeline == "R4" else {}),
                    "wall_seconds": time.perf_counter() - started if started else 0,
                    "model_calls": model.sequence if model else 0, "gpu_memory": failure["gpu_memory"],
                    "request": asdict(request) if request else None, "evaluation": current.evaluation,
                    "failure": {"stage": "execution", "code": type(exc).__name__, "message": str(exc)},
                    "result": None,
                })
            write_summary(cases, output, signature)
            raise
        finally:
            if model is not None:
                original_error = sys.exc_info()[1]
                try:
                    model.unload()
                except Exception as exc:
                    atomic_json(output / "cleanup_error.json", {"error": str(exc), "traceback": traceback.format_exc()})
                    if original_error is None:
                        raise


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    default_data = PROJECT / "data/r1345_debug12"
    if not default_data.exists():
        default_data = PROJECT.parent / "data/r1345_debug12"
    parser.add_argument("--data-root", type=Path, default=default_data)
    parser.add_argument("--config", type=Path, default=PROJECT / "configs/r1345_dual4090d.yaml")
    parser.add_argument("--output-dir", type=Path, default=PROJECT / "runs/r1345-debug12")
    parser.add_argument("--pipeline", choices=PIPELINES, action="append")
    parser.add_argument("--r1-version", choices=("v3",), default="v3")
    parser.add_argument("--question-id", action="append")
    parser.add_argument("--stage", choices=("all", "gates"), default="all")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    output, root = args.output_dir.resolve(), args.data_root.resolve()
    try:
        cases = load_dataset(root)
        configs = configurations(args.config, output, r1_version=args.r1_version)
        selected = [c for c in cases if (not args.pipeline or c.pipeline in args.pipeline)]
        if args.question_id:
            unknown = set(args.question_id) - {c.id for c in selected}
            if unknown:
                raise ValueError(f"Unknown or excluded question IDs: {sorted(unknown)}")
            selected = [c for c in selected if c.id in args.question_id]
        planned = list(selected)
        if args.stage == "gates":
            selected = [next(c for c in selected if c.pipeline == p) for p in PIPELINES if any(c.pipeline == p for c in selected)]
        if not selected:
            raise ValueError("No selected questions")
        if args.preflight:
            with run_lock(output):
                report = preflight(selected, configs)
                atomic_json(output / "preflight.json", report)
            print(json.dumps({k: report[k] for k in ("ready", "question_count", "video_count", "model_calls", "torch_imported")}))
            return 0 if report["ready"] else 2
        result = run_cases(planned, selected, configs, output, identity(root, configs), resume=args.resume)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (Exception, FatalModelError) as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        return 2
