"""Explicitly selected R2 manifests; gold is retained only in evaluation sidecars."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, fields, replace
from pathlib import Path

from qwen3vl_agent.config import load_config
from qwen3vl_agent.r3.evaluate import atomic_text, json_text

from .agent import R2VideoAgent
from .config import R2Config
from .media import R2Media
from .planning import make_windows, sample_times
from .runtime import stable_key
from .types import InputContract, R2Request, R2Result


def read_manifest(path):
    path = Path(path).resolve()
    permitted = {f.name for f in fields(R2Request)} - {"resume", "checkpoint_path"}
    rows, seen = [], set()
    for lineno, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            if not isinstance(row, dict) or row.get("pipeline_id", "R2") != "R2":
                raise ValueError("manifest requires already-selected R2 requests")
            if {"resume", "checkpoint_path"} & row.keys():
                raise ValueError("batch runner manages checkpoints")
            rid = row.get("request_id")
            if not isinstance(rid, str) or not rid.strip() or rid in seen:
                raise ValueError("unique request_id is required")
            data = {k: v for k, v in row.items() if k in permitted}
            for key in ("video_path", "subtitle_path", "asr_path"):
                if data.get(key):
                    p = Path(data[key]).expanduser()
                    data[key] = str(
                        (path.parent / p).resolve() if not p.is_absolute() else p.resolve()
                    )
            request = R2Request(**data)
        except (ValueError, TypeError, KeyError) as exc:
            raise ValueError(f"{path}:{lineno}: {exc}") from exc
        seen.add(rid)
        rows.append((request, {k: v for k, v in row.items() if k not in permitted}))
    if not rows:
        raise ValueError("empty R2 manifest")
    return rows


def preflight(rows, config):
    from qwen3vl_agent.models.qwen3vl import Qwen3VLModel

    media = R2Media(config)
    reports = []
    for request, _ in rows:
        result = {"request_id": request.request_id, "video_path": request.video_path}
        try:
            if not Path(request.video_path).is_file():
                result.update(status="missing_media")
                reports.append(result)
                continue
            metadata = media.probe(request.video_path)
            contract = InputContract.resolve(request, metadata.duration_seconds)
            spans = contract.allowed_time_intervals
            if request.query_scope is not None and not isinstance(request.query_scope, str):
                spans = contract.intersect(request.query_scope)
                if not spans:
                    raise ValueError("query scope has no permitted observation interval")
            if request.execution_subtype == "identity_at_time":
                spans = contract.allowed_time_intervals
            windows = make_windows(spans, contract, config)
            budget = request.budget.capped(config.budget)
            exposures = sum(
                min(config.max_frames_per_call, len(sample_times(w, metadata.source_fps)[0]) + 2)
                for w in windows
            )
            estimates = {
                "base_windows": len(windows),
                "base_calls_plus_compile_and_final": len(windows) + 4,
                "call_cap": budget.max_model_calls,
                "base_frame_exposures_upper_estimate": exposures,
                "terminal_frame_reserve_upper_estimate": 2 * config.max_frames_per_call,
                "frame_exposure_cap": budget.max_frame_exposures,
                "base_media_pixels_upper_estimate": len(windows) * config.media.normal_total_pixels,
                "media_pixel_cap": budget.max_media_pixels,
                "base_plan_may_exceed_budget": len(windows) + 4 > budget.max_model_calls
                or exposures + 2 * config.max_frames_per_call > budget.max_frame_exposures
                or (len(windows) + 2) * config.media.normal_total_pixels > budget.max_media_pixels,
                "semantic_localization_cost_not_estimated": True,
            }
            inspected = []
            for window in windows[:1] + windows[-1:] if len(windows) > 1 else windows:
                times, fps = sample_times(window, metadata.source_fps)
                batch = media.extract(request.video_path, window["span"], times, contract, fps=fps)
                if not batch.frames:
                    raise ValueError("decoder produced no permitted frames")
                prepared = media.prepare(batch)
                if prepared.video_frame_metadata:
                    from types import SimpleNamespace

                    tensors = [
                        SimpleNamespace(shape=(len(m["frame_ids"]),))
                        for m in prepared.video_frame_metadata
                    ]
                    Qwen3VLModel._validated_frame_metadata(tensors, prepared.video_frame_metadata)
                inspected.append(
                    {
                        "span": window["span"],
                        "frame_count": len(batch.frames),
                        "media_kind": prepared.kind,
                        "all_pts_permitted": all(
                            contract.permits(f.timestamp_seconds) for f in batch.frames
                        ),
                        "source_times": [f.timestamp_seconds for f in batch.frames],
                        "coverage": media.coverage(batch, window["id"], False),
                        "pts": [media.catalog[f.id]["pts"] for f in batch.frames],
                        "time_bases": [media.catalog[f.id]["time_base"] for f in batch.frames],
                        "model_pixels": prepared.pixels,
                    }
                )
            result.update(
                status="ready",
                duration_seconds=metadata.duration_seconds,
                input_contract=asdict(contract),
                estimates=estimates,
                decoded_checks=inspected,
            )
        except (ValueError, RuntimeError, OSError, ImportError) as exc:
            result.update(status="input_error", error=f"{type(exc).__name__}: {exc}")
        reports.append(result)
    return {
        "pipeline_id": "R2",
        "mode": "preflight",
        "model_calls": 0,
        "semantic_coverage": False,
        "accuracy": None,
        "requests": reports,
        "ready": sum(r["status"] == "ready" for r in reports),
        "missing_media": sum(r["status"] == "missing_media" for r in reports),
        "input_errors": sum(r["status"] == "input_error" for r in reports),
    }


def run_manifest(agent, rows, output_dir, *, resume=False):
    out = Path(output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    identity = {
        "requests": [asdict(r) for r, _ in rows],
        "evaluation_metadata": [x for _, x in rows],
        "config": asdict(agent.config),
    }
    manifest = out / "run_manifest.json"
    if manifest.exists():
        if not resume or manifest.read_text(encoding="utf-8") != json_text(identity):
            raise ValueError(
                "existing run requires resume and identical request/config/evaluation metadata"
            )
    elif any(p.name != "preflight.json" for p in out.iterdir()):
        raise ValueError("new output directory must be empty except preflight.json")
    else:
        atomic_text(manifest, json_text(identity))
    predictions, diagnostics = [], []
    for request, extra in rows:
        key = stable_key(request.request_id)
        checkpoint = out / "checkpoints" / (key + ".jsonl")
        try:
            metadata = agent.media.probe(request.video_path)
            InputContract.resolve(request, metadata.duration_seconds)
        except (OSError, ValueError, RuntimeError) as exc:
            result = R2Result(
                None,
                "failed",
                "unsupported",
                unresolved_items=[f"input_error: {type(exc).__name__}: {exc}"],
                resources={"model_calls": 0, "frame_exposures": 0},
            )
        else:
            result = agent.solve(
                replace(
                    request, checkpoint_path=str(checkpoint), resume=resume and checkpoint.exists()
                )
            )
        prediction = {
            "request_id": request.request_id,
            "video_id": request.video_id,
            "group_id": request.group_id,
            "prediction": result.prediction,
        }
        predictions.append(prediction)
        diagnostics.append(
            {
                **prediction,
                "completion_state": result.completion_state,
                "support_level": result.support_level,
                "resources": result.resources,
            }
        )
        atomic_text(
            out / "sidecars" / (key + ".json"),
            json_text(
                {"request": asdict(request), "evaluation": extra, "result": result.to_dict()}
            ),
        )
        atomic_text(
            out / "predictions.jsonl",
            "".join(json.dumps(p, ensure_ascii=False) + "\n" for p in predictions),
        )
        atomic_text(out / "diagnostics.json", json_text(diagnostics))
    summary = {
        "pipeline_id": "R2",
        "requests": len(rows),
        "predictions": sum(p["prediction"] is not None for p in predictions),
        "supported": sum(d["support_level"] == "supported" for d in diagnostics),
        "model_calls": sum(d["resources"]["model_calls"] for d in diagnostics),
        "accuracy": None,
    }
    atomic_text(out / "summary.json", json_text(summary))
    return summary


def main():
    parser = argparse.ArgumentParser(description="Run explicitly selected R2 questions")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    settings = R2Config.from_mapping(config.get("r2"))
    rows = read_manifest(args.manifest)
    if args.preflight:
        result = preflight(rows, settings)
        atomic_text(Path(args.output_dir) / "preflight.json", json_text(result))
    else:
        from qwen3vl_agent.factory import build_model

        model_settings = {
            "path": "Qwen/Qwen3-VL-8B-Instruct",
            "dtype": "bfloat16",
            **config.get("model", {}),
        }
        agent = R2VideoAgent(build_model(model_settings), config=settings)
        agent.load()
        try:
            result = run_manifest(agent, rows, args.output_dir, resume=args.resume)
        finally:
            agent.unload()
    print(json_text(result))


if __name__ == "__main__":
    main()
