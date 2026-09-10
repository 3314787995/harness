"""R8 public CLI: decode preflight, answer-blind runs, and separate offline scoring."""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from dataclasses import asdict, fields, replace
from fractions import Fraction
from pathlib import Path

from qwen3vl_agent.config import load_config
from qwen3vl_agent.factory import build_model
from qwen3vl_agent.r3.checkpoint import file_digest

from .config import MODEL, REVISION, R8Config
from .controller import R8VideoAgent
from .media import ScopedMedia, sampling
from .runtime import software_fingerprint
from .types import MODES, ProtocolError, R8Request, digest, plain

CATALOG_FIELDS = {
    "source_id",
    "dataset",
    "mechanism",
    "r8_role",
    "availability",
    "video_url",
    "source_url",
}


def read_jsonl(path):
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(plain(value), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def public_request(row, root=None):
    allowed = {f.name for f in fields(R8Request)} | CATALOG_FIELDS
    if not isinstance(row, dict) or set(row) - allowed:
        raise ProtocolError(
            "manifest accepts only public R8Request/catalog fields; keep answers in offline file"
        )
    values = {k: v for k, v in row.items() if k not in CATALOG_FIELDS}
    if root:
        for key in ("video_path", "subtitle_path", "variables_input", "checkpoint_path"):
            if values.get(key) and not Path(values[key]).is_absolute():
                values[key] = str((Path(root) / values[key]).resolve())
    return R8Request(**values)


def model_and_config(path=None):
    config = load_config(path) if path else {}
    r8 = R8Config.from_mapping(config.get("r8"))
    model = dict(config.get("model") or {})
    model.setdefault("path", MODEL)
    model.setdefault("revision", REVISION)
    model.setdefault("dtype", "bfloat16")
    model.setdefault("device", "auto")
    model.setdefault("attn_implementation", "flash_attention_2")
    model["generation"] = {**model.get("generation", {}), "temperature": 0.0}
    if model["revision"] != REVISION:
        raise ProtocolError("R8 fixes the model revision")
    if model["path"] != MODEL and not Path(model["path"]).is_dir():
        raise ProtocolError("R8 requires Qwen3-VL-8B-Instruct or its local pinned snapshot")
    return model, r8


def preflight(request, config):
    media = ScopedMedia(
        request, config, {"mode": "CPU_decode_preflight", "model": MODEL, "revision": REVISION}
    )
    packets = []
    for window in media.contract.allowed_time_intervals:
        times = sampling(window, replace(config, initial_frames=min(4, config.initial_frames)))[0]
        batch = media.extract(window, times)
        prepared = media.prepare(batch)
        crop = media.local_batch(batch.frames[0].id, [[0.2, 0.2, 0.8, 0.8]])
        local = media.prepare(crop)
        packets.append(
            {
                "kind": prepared.kind,
                "parts_types": [p["type"] for p in prepared.parts],
                "frame_count": len(prepared.frames),
                "pixels": prepared.pixels,
                "video_frame_metadata": prepared.video_frame_metadata,
                "evidence": media.evidence(prepared, window),
                "crop_kind": local.kind,
                "crop_frames": [media.get_evidence(f.id) for f in crop.frames],
            }
        )
    return {
        "status": "passed",
        "validation_kind": "real_source_decode_and_input_structure_only",
        "model_loaded": False,
        "semantic_accuracy_measured": False,
        "request_id": request.request_id,
        "video_path": request.video_path,
        "video_sha256": media.source_hash,
        "contract": asdict(media.contract),
        "metadata": asdict(media.metadata),
        "packets": packets,
    }


def export_variables(result, path):
    """Export observations only; predictions/answers/derived computations are not replay inputs."""
    from .contracts import OBS_VAR

    data = result.to_dict() if hasattr(result, "to_dict") else result
    trace = data["trace"]
    variables = []
    for row in data["variables"].values():
        if row["valid"] and row["origin"] == "observed":
            variables.append({key: row[key] for key in OBS_VAR["properties"]})
    artifact = {
        "format": "r8-variable-replay/1",
        "kind": "oracle_diagnostic" if trace["diagnostic"] else "observed_replay",
        "question_sha256": trace["question_sha256"],
        "video_sha256": trace["video_sha256"],
        "scope_hash": trace["scope_hash"],
        "entities": list(trace["entities"].values()),
        "evidence": data["evidence"],
        "variables": variables,
        "relations": list(trace["geometry_relations"].values()),
        "adapters": {k: list(v.values()) for k, v in trace["adapter_records"].items()},
    }
    write_json(path, artifact)
    return artifact


def run_rows(
    rows,
    *,
    config_path=None,
    output,
    checkpoint_dir,
    resume=False,
    root=None,
    mode=None,
    comparison=None,
    diagnostic=False,
):
    output = Path(output)
    if output.exists() and not resume:
        raise ProtocolError("output already exists; use --resume or a new output")
    model_settings, config = model_and_config(config_path)
    # Validate the complete manifest before loading weights or starting any expensive call.
    requests = []
    for row in rows:
        request = public_request(row, root)
        request = replace(
            request,
            mode=mode or request.mode,
            comparison=comparison or request.comparison,
            diagnostic=diagnostic or request.diagnostic,
        )
        requests.append(request)
    keys = [r.request_id for r in requests]
    if len(keys) != len(set(keys)):
        raise ProtocolError("duplicate request IDs in one experiment manifest")
    previous = {r["request_id"]: r for r in read_jsonl(output)} if output.exists() else {}
    model = build_model(model_settings)
    software = software_fingerprint(model, config)
    agent = R8VideoAgent(model, config)
    loaded = False
    output.parent.mkdir(parents=True, exist_ok=True)
    results = []
    try:
        for request in requests:
            identity = asdict(request)
            for key in ("resume", "checkpoint_path"):
                identity.pop(key)
            fingerprint = digest(
                {
                    "request": identity,
                    "config": asdict(config),
                    "software": software,
                    "video_sha256": file_digest(request.video_path)
                    if Path(request.video_path).is_file()
                    else None,
                }
            )
            if request.request_id in previous:
                if previous[request.request_id]["run_fingerprint"] != fingerprint:
                    raise ProtocolError(
                        "resumed output request/model/config/media fingerprint mismatch"
                    )
                results.append(previous[request.request_id])
                continue
            item = {
                "request_id": request.request_id,
                "video_id": request.video_id,
                "mode": request.mode,
                "comparison": request.comparison,
                "diagnostic": request.diagnostic,
                "run_fingerprint": fingerprint,
            }
            if not Path(request.video_path).is_file():
                item.update(
                    run_status="media_unavailable", error="source video not present", result=None
                )
            else:
                path = Path(checkpoint_dir) / (
                    digest(
                        {
                            "id": request.request_id,
                            "mode": request.mode,
                            "comparison": request.comparison,
                        }
                    )[:24]
                    + ".jsonl"
                )
                path.parent.mkdir(parents=True, exist_ok=True)
                if not loaded:
                    agent.load()
                    loaded = True
                try:
                    result = agent.solve(
                        replace(request, checkpoint_path=str(path), resume=resume and path.exists())
                    )
                    item.update(run_status="completed", result=result.to_dict())
                except Exception as exc:  # noqa: BLE001 -- batch boundary preserves the error record
                    item.update(
                        run_status="engineering_error",
                        error=f"{type(exc).__name__}: {exc}",
                        result=None,
                    )
            with output.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(plain(item), ensure_ascii=False) + "\n")
                stream.flush()
            results.append(item)
    finally:
        if loaded:
            agent.unload()
    return results


def score(predictions, answers):
    gold = {r["request_id"]: r for r in answers}
    if len(gold) != len(answers):
        raise ProtocolError("duplicate offline answer ID")
    groups = defaultdict(list)
    seen = set()
    for item in predictions:
        key = (
            item.get("mode", "G"),
            item.get("comparison", "end_to_end"),
            bool(item.get("diagnostic")),
        )
        identity = (key, item["request_id"])
        if identity in seen:
            raise ProtocolError("duplicate prediction in one experiment")
        seen.add(identity)
        if item["request_id"] not in gold:
            raise ProtocolError("prediction lacks an offline answer record")
        source = gold[item["request_id"]]
        result = item.get("result") or {}
        prediction = result.get("prediction")
        answer = source["answer"]
        correct = prediction == answer
        if source.get("answer_type") == "numeric" and prediction is not None:
            try:
                correct = Fraction(prediction) == Fraction(str(answer))
            except (ValueError, ZeroDivisionError):
                correct = False
        groups[key].append(
            {
                "request_id": item["request_id"],
                "video_id": source.get("video_id", item.get("video_id")),
                "mechanism": source.get("mechanism", "unassigned"),
                "r8_role": source.get("r8_role", "unassigned"),
                "correct": correct,
                "status": result.get("status"),
                "run_status": item.get("run_status"),
                "verified": result.get("verified", False),
                "prediction": prediction,
                "resources": result.get("resources", {}),
            }
        )

    def summary(rows):
        n = len(rows)
        executed = [r for r in rows if r["run_status"] == "completed"]
        return {
            "n": n,
            "correct": sum(r["correct"] for r in rows),
            "accuracy_all_requested": sum(r["correct"] for r in rows) / n if n else None,
            "executed_n": len(executed),
            "accuracy_executed": sum(r["correct"] for r in executed) / len(executed)
            if executed
            else None,
            "unresolved_rate": sum(
                r["status"] in {"unresolved_evidence", "unresolved_modeling", "solver_unknown"}
                for r in rows
            )
            / n
            if n
            else None,
            "no_prediction_rate": sum(r["prediction"] is None for r in rows) / n if n else None,
            "unverified_rate": sum(not r["verified"] for r in rows) / n if n else None,
            "forced_guess_rate": sum(r["status"] == "forced_guess" for r in rows) / n
            if n
            else None,
            "annotation_anomaly_n": sum(r["status"] == "annotation_anomaly" for r in rows),
            "media_unavailable_n": sum(r["run_status"] == "media_unavailable" for r in rows),
            "engineering_error_n": sum(r["run_status"] == "engineering_error" for r in rows),
        }

    reports = []
    for key, rows in sorted(groups.items()):
        by_video, by_mechanism, by_role = defaultdict(list), defaultdict(list), defaultdict(list)
        for row in rows:
            by_video[str(row["video_id"])].append(row)
            by_mechanism[row["mechanism"]].append(row)
            by_role[row["r8_role"]].append(row)
        rng = random.Random(8)
        videos = list(by_video)
        values = []
        if len(videos) >= 2:
            for _ in range(1000):
                selected = [r for _ in videos for r in by_video[rng.choice(videos)]]
                values.append(sum(r["correct"] for r in selected) / len(selected))
            values.sort()
        cost = {}
        for metric in (
            "model_calls",
            "unique_frames",
            "frame_exposures",
            "repeated_frame_exposures",
            "input_tokens",
            "output_tokens",
            "visual_tokens",
            "model_seconds",
            "elapsed_seconds",
        ):
            measured = [r["resources"].get(metric) for r in rows if r["run_status"] == "completed"]
            cost[metric] = (
                sum(measured)
                if measured and all(isinstance(v, (int, float)) for v in measured)
                else "unavailable"
            )
        reports.append(
            {
                "mode": key[0],
                "comparison": key[1],
                "diagnostic": key[2],
                **summary(rows),
                "per_video": {v: summary(r) for v, r in by_video.items()},
                "mechanisms": {v: summary(r) for v, r in by_mechanism.items()},
                "r8_roles": {v: summary(r) for v, r in by_role.items()},
                "cost": cost,
                "cluster_bootstrap_95_percent": [values[25], values[974]]
                if values
                else "unavailable",
                "variable_accuracy": "unavailable",
                "formula_accuracy": "unavailable",
                "source_reading_accuracy": "unavailable",
                "event_discovery_recall": "unavailable",
                "human_labels_present": False,
                "rows": rows,
            }
        )
    return {
        "format": "r8-offline-score/1",
        "experiments": reports,
        "scope": "only requested IDs in predictions; original duplicate/anomalous questions retained",
        "unrequested_answer_records": len(set(gold) - {r["request_id"] for r in predictions}),
    }


def parser():
    p = argparse.ArgumentParser(
        description="Training-free R8: CPU preflight, answer-blind run, offline score"
    )
    sub = p.add_subparsers(dest="command", required=True)
    for name in ("preflight", "run"):
        child = sub.add_parser(name)
        child.add_argument("--config")
        child.add_argument("--request", help="one public R8Request JSON")
        child.add_argument("--manifest", help="public request JSONL; no answers")
        child.add_argument("--video")
        child.add_argument("--question", default="Decode input preflight")
        child.add_argument("--choice", action="append", default=[])
        child.add_argument("--output", required=True)
        child.add_argument("--mode", choices=MODES)
        child.add_argument("--comparison", choices=("end_to_end", "fixed_evidence"))
        child.add_argument("--diagnostic", action="store_true")
        if name == "run":
            child.add_argument("--checkpoint-dir", default=".cache/qwen3vl_agent/r8/checkpoints")
            child.add_argument("--resume", action="store_true")
    child = sub.add_parser("score")
    child.add_argument("--predictions", required=True)
    child.add_argument("--answers", required=True)
    child.add_argument("--output", required=True)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    if args.command == "score":
        report = score(read_jsonl(args.predictions), read_jsonl(args.answers))
        write_json(args.output, report)
        print(f"R8 offline report: {Path(args.output).resolve()}")
        return
    if sum(bool(v) for v in (args.request, args.manifest, args.video)) != 1:
        raise SystemExit("supply exactly one of --request / --manifest / --video")
    root = (
        Path(args.manifest or args.request).resolve().parent
        if args.manifest or args.request
        else Path.cwd()
    )
    rows = (
        read_jsonl(args.manifest)
        if args.manifest
        else [json.loads(Path(args.request).read_text(encoding="utf-8"))]
        if args.request
        else [{"video_path": args.video, "question": args.question, "choices": args.choice}]
    )
    if args.command == "preflight":
        _, config = model_and_config(args.config)
        results = []
        for row in rows:
            request = public_request(row, root)
            if not Path(request.video_path).is_file():
                results.append(
                    {
                        "request_id": request.request_id,
                        "status": "media_unavailable",
                        "model_loaded": False,
                    }
                )
            else:
                results.append(preflight(request, config))
        write_json(args.output, {"results": results})
        print(f"R8 CPU decode preflight: {Path(args.output).resolve()}")
    else:
        results = run_rows(
            rows,
            config_path=args.config,
            output=args.output,
            checkpoint_dir=args.checkpoint_dir,
            resume=args.resume,
            root=root,
            mode=args.mode,
            comparison=args.comparison,
            diagnostic=args.diagnostic,
        )
        print(
            json.dumps(
                {
                    "requests": len(results),
                    "completed": sum(r["run_status"] == "completed" for r in results),
                    "output": str(Path(args.output).resolve()),
                }
            )
        )


if __name__ == "__main__":
    main()
