"""Batch execution and no-model preflight for already-selected R4 JSONL requests."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from dataclasses import asdict, fields, replace
from pathlib import Path
from typing import Any

from qwen3vl_agent.config import load_config
from qwen3vl_agent.r3.evaluate import atomic_text, json_text
from qwen3vl_agent.r4.agent import R4VideoAgent
from qwen3vl_agent.r4.checkpoint import implementation_digest, file_digest, model_signature
from qwen3vl_agent.r4.config import R4Config
from qwen3vl_agent.r4.planning import planned_calls, resolve_sources
from qwen3vl_agent.r4.types import R4Request, R4Result


def read_manifest(path: str | Path) -> list[tuple[R4Request, dict[str, Any]]]:
    path = Path(path).resolve()
    allowed = {f.name for f in fields(R4Request)} - {"checkpoint_path", "resume"}
    rows, seen = [], set()

    def absolute(value: str) -> str:
        file = Path(value).expanduser()
        return str((path.parent / file).resolve() if not file.is_absolute() else file.resolve())

    def resolve_files(data: dict[str, Any]) -> None:
        if data.get("video_path"):
            data["video_path"] = absolute(data["video_path"])
        for file in data.get("external_files", []):
            file["path"] = absolute(file["path"])

    for line_number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            if not isinstance(row, dict) or row.get("pipeline_id", "R4") != "R4":
                raise ValueError("manifest must contain already-selected R4 objects")
            request_id = row.get("request_id")
            if not isinstance(request_id, str) or not request_id.strip() or request_id in seen:
                raise ValueError("unique nonempty request_id required")
            if {"checkpoint_path", "resume"} & row.keys():
                raise ValueError("batch runner owns checkpoint paths and resume")
            original = copy.deepcopy(row)
            data = {key: value for key, value in row.items() if key in allowed}
            resolve_files(data)
            for source in data.get("sources", []):
                resolve_files(source)
            request = R4Request(**data)
        except (ValueError, TypeError, KeyError, AttributeError) as exc:
            raise ValueError(f"{path}:{line_number}: {exc}") from exc
        seen.add(request_id)
        rows.append(
            (
                request,
                {**{k: v for k, v in row.items() if k not in allowed}, "original_record": original},
            )
        )
    if not rows:
        raise ValueError("R4 manifest is empty")
    return rows


def preflight(
    rows: list[tuple[R4Request, dict[str, Any]]], config: R4Config, probe: Any = None
) -> dict[str, Any]:
    if probe is None:
        from qwen3vl_agent.r1.config import R1Config
        from qwen3vl_agent.r4.media import R4Media

        probe = R4Media(R1Config(media=config.media))
    import math

    reports = []
    for request, _ in rows:
        report = {"request_id": request.request_id}
        try:
            sources, issues = resolve_sources(request, probe)
            if not any(s["allowed"] for s in sources.values()):
                issues.append("no_permitted_media")
            estimates = {}
            for name, fps in (("static", config.static_fps), ("motion", config.motion_fps)):
                n, external = 0, 0
                for source in sources.values():
                    rate = min(fps, source["source_fps"]) if source["source_fps"] else fps
                    for a, b in source["allowed"]:
                        n += max(1, math.ceil((b - a) * rate / config.core_frames))
                        external += math.ceil((b - a) / config.text_window_sec) * len(
                            set(source["available_modalities"]) & {"asr", "subtitle"}
                        )
                    for file in source["external_files"]:
                        from qwen3vl_agent.r4.providers import read_external_file
                        from qwen3vl_agent.r4.types import ExternalFile

                        read_external_file(ExternalFile(**file), source["source_id"])
                estimates[name] = {
                    "visual_windows": n,
                    "external_windows": external,
                    "planned_model_calls": planned_calls(n + external),
                }
            cap = min(config.budget.max_model_calls, request.budget.max_model_calls)
            report.update(
                status="ready" if not issues else "input_incomplete",
                issues=issues,
                estimates=estimates,
                model_call_cap=cap,
                budget_limited_possible=any(
                    v["planned_model_calls"] > cap for v in estimates.values()
                ),
                sources=[
                    {
                        "entry_id": s["entry_id"],
                        "source_id": s["source_id"],
                        "allowed": s["allowed"],
                    }
                    for s in sources.values()
                ],
                note="Full allowed-scope upper planning estimate; semantic query binding is not executed.",
            )
        except (ValueError, RuntimeError, OSError, ImportError) as exc:
            report.update(status="input_error", error=str(exc))
        reports.append(report)
    return {
        "pipeline_id": "R4",
        "mode": "preflight",
        "model_calls": 0,
        "semantic_coverage": False,
        "requests": reports,
    }


def reject_saved_failures(rows, destination: Path) -> list[str]:
    """Compatibility name: v5 reports terminal failures but does not retry/block later questions."""
    failed = []
    for request, _ in rows:
        stem = hashlib.sha256(request.request_id.encode()).hexdigest()[:24]
        path = destination / "sidecars" / (stem + ".json")
        if path.exists() and json.loads(path.read_text(encoding="utf-8")).get("failure"):
            failed.append(request.request_id)
    return failed


def run_manifest(
    agent: R4VideoAgent,
    rows: list[tuple[R4Request, dict[str, Any]]],
    output_dir: str | Path,
    *,
    resume: bool = False,
) -> dict[str, Any]:
    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    identity = {
        "requests": [asdict(r) for r, _ in rows],
        "evaluation_metadata": [x for _, x in rows],
        "config": asdict(agent.config),
        "implementation": implementation_digest(),
        "protocol": __import__("qwen3vl_agent.r4.prompts", fromlist=["VERSION"]).VERSION,
        "model_path": getattr(getattr(agent, "model", None), "model_path", None),
        "model_files": model_signature(agent.model) if getattr(agent, "model", None) is not None else None,
        "model_config": {k: getattr(getattr(agent, "model", None), k, None) for k in
                         ("dtype", "device", "device_map", "max_memory", "attn_implementation", "revision")},
        "media_hashes": {s.video_path: file_digest(s.video_path) for r, _ in rows for s in r.media_sources()},
        "text_hashes": {f.path: file_digest(f.path) for r, _ in rows for s in r.media_sources() for f in s.external_files},
    }
    identity_path = destination / "run_manifest.json"
    encoded = json_text(identity)
    if identity_path.exists():
        if not resume or identity_path.read_text(encoding="utf-8") != encoded:
            raise ValueError("existing output requires resume and identical manifest/config")
    else:
        if any(p.name != "preflight.json" for p in destination.iterdir()):
            raise ValueError("new R4 output directory must be empty")
        atomic_text(identity_path, encoded)
    if resume:
        reject_saved_failures(rows, destination)
    predictions, native_predictions, diagnostics = [], [], []

    def flush():
        answered = {p["request_id"] for p in predictions if p["prediction"] is not None}
        attempted = {p["request_id"] for p in predictions}
        absent = [
            {
                "request_id": r.request_id,
                "status": "no_prediction" if r.request_id in attempted else "pending",
            }
            for r, _ in rows
            if r.request_id not in answered
        ]
        atomic_text(
            destination / "predictions.jsonl",
            "".join(json.dumps(p, ensure_ascii=False, allow_nan=False) + "\n" for p in predictions),
        )
        atomic_text(
            destination / "native_predictions.jsonl",
            "".join(
                json.dumps(p, ensure_ascii=False, allow_nan=False) + "\n"
                for p in native_predictions
            ),
        )
        atomic_text(destination / "unanswered.json", json_text(absent))
        pending = [
            {"request_id": r.request_id, "completion_state": "pending"}
            for r, _ in rows
            if r.request_id not in attempted
        ]
        atomic_text(
            destination / "diagnostics.json", json_text({"requests": diagnostics + pending})
        )
        summary = {
            "pipeline_id": "R4",
            "request_count": len(rows),
            "attempted": len(predictions),
            "terminal_count": len(predictions),
            "completed": sum(not d["failure"] for d in diagnostics),
            "failed": sum(bool(d["failure"]) for d in diagnostics),
            "answered": len(answered),
            "no_prediction": len(predictions) - len(answered),
            "pending": len(rows) - len(predictions),
            "supported_count": sum(d["support_level"] == "supported" for d in diagnostics),
            "model_calls": sum(d["resources"].get("model_calls", 0) or 0 for d in diagnostics),
            "by_operation": {
                op: {
                    "count": sum(op in d["operations"] for d in diagnostics),
                    "supported_count": sum(
                        op in d["operations"] and d["support_level"] == "supported"
                        for d in diagnostics
                    ),
                }
                for op in sorted({op for d in diagnostics for op in d["operations"]})
            },
            "correct": sum(d["correct"] is True for d in diagnostics),
            "scored_count": sum(d["answer_label"] is not None for d in diagnostics),
            "accuracy": (sum(d["correct"] is True for d in diagnostics) / len(rows))
                        if all(x.get("answer_label", x.get("answer")) is not None for _, x in rows) else None,
            "answer_mode_counts": {mode:sum(d.get("answer_mode")==mode for d in diagnostics) for mode in ("determined","best_effort","none")},
            "result_status_counts": {s: sum(d.get("result_status") == s for d in diagnostics)
                for s in ("supported", "unresolved", "budget_exhausted", "execution_failed")},
            "identity_accuracy": None,
            "negative_evidence_accuracy": None,
            "note": "Native predictions contain answered requests only; unanswered.json includes failures and pending requests. Report the full request count with official scores.",
        }
        atomic_text(destination / "summary.json", json_text(summary))
        return summary

    loaded = False
    for request, extra in rows:
        stem = hashlib.sha256(request.request_id.encode()).hexdigest()[:24]
        checkpoint = destination / "checkpoints" / (stem + ".jsonl")
        sidecar = destination / "sidecars" / (stem + ".json")
        if resume and sidecar.exists():
            saved = json.loads(sidecar.read_text(encoding="utf-8"))
            result = R4Result(
                **{f.name: saved[f.name] for f in fields(R4Result) if f.name in saved}
            )
        else:
            # Loading/device availability is a batch prerequisite, not a recoverable question error.
            if not loaded and hasattr(agent, "load"):
                agent.load()
                loaded = True
            try:
                result = agent.solve(
                    replace(
                        request,
                        checkpoint_path=str(checkpoint),
                        resume=resume and checkpoint.exists(),
                    )
                )
            except Exception as exc:  # noqa: BLE001 - persist terminal failures at the batch boundary
                if isinstance(exc, OSError) or any(t in str(exc).lower() for t in ("device-side assert", "illegal memory access", "cuda driver")):
                    raise
                result = R4Result(
                    None,
                    {"results": []},
                    "execution_error",
                    "unsupported",
                    "pipeline_failure",
                    {},
                    [],
                    [str(exc)],
                    {},
                    {},
                    {},
                    {"stage": "execution", "code": type(exc).__name__, "message": str(exc)},
                    "execution_failed",
                )
        if result.completion_state in {"input_error", "dependency_missing"} and not result.failure:
            result.failure = {
                "stage": "input",
                "code": result.completion_state,
                "message": str(result.unresolved_items),
            }
            result.prediction = None
        labels = {c.label for c in request.choices}
        if result.prediction is not None and labels and result.prediction not in labels:
            result.failure = {"stage": "output", "code": "invalid_prediction", "message": "Prediction is not an original option"}
        if result.failure:
            result.result_status = "execution_failed"
            if result.answer_mode != "best_effort" or result.failure.get("code") == "invalid_prediction":
                result.prediction = None
        if result.result_status != "supported" and result.answer_mode != "best_effort":
            result.prediction = None
        gold = extra.get("answer_label", extra.get("answer"))
        predictions.append(
            {
                "request_id": request.request_id,
                "video_id": request.video_id,
                "group_id": request.group_id,
                "prediction": result.prediction,
            }
        )
        if result.prediction is not None:
            native_predictions.append(
                {
                    **extra.get("original_record", {**asdict(request), **extra}),
                    "prediction": result.prediction,
                }
            )
        diagnostics.append(
            {
                "request_id": request.request_id,
                "native_labels": request.native_labels,
                "execution_subtype": request.execution_subtype,
                "operations": [r["op"] for r in result.value_state.get("results", [])],
                "completion_state": result.completion_state,
                "support_level": result.support_level,
                "unresolved_items": result.unresolved_items,
                "resources": result.resources,
                "prediction": result.prediction,
                "failure": result.failure,
                "result_status": result.result_status, "answer_mode":result.answer_mode, "evidence_status":result.evidence_status,
                "answer_label": gold,
                "correct": result.prediction == gold if result.prediction is not None and gold is not None else None,
            }
        )
        atomic_text(
            sidecar,
            json_text(
                {"request": asdict(request), "evaluation_metadata": extra, **result.to_dict()}
            ),
        )
        flush()
    return flush()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--config", default="configs/r4_8b.yaml")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    rows = read_manifest(args.manifest)
    configuration = load_config(args.config)
    config = R4Config.from_mapping(configuration.get("r4"))
    if args.preflight:
        result = preflight(rows, config)
        atomic_text(Path(args.output_dir).resolve() / "preflight.json", json_text(result))
        print(json_text(result), end="")
        return
    from qwen3vl_agent.factory import build_model

    if args.resume:
        reject_saved_failures(rows, Path(args.output_dir).resolve())
    model = build_model(
        {
            "path": "Qwen/Qwen3-VL-8B-Instruct",
            "dtype": "bfloat16",
            "attn_implementation": "flash_attention_2",
            **configuration.get("model", {}),
        }
    )
    agent = R4VideoAgent(model, config)
    try:
        print(json_text(run_manifest(agent, rows, args.output_dir, resume=args.resume)), end="")
    finally:
        if model.is_loaded:
            agent.unload()


if __name__ == "__main__":
    main()
