"""Answer-blind run/preflight and separate offline scoring."""

import argparse
import json
from dataclasses import asdict, fields, replace
from decimal import Decimal
from pathlib import Path

from qwen3vl_agent.config import load_config
from qwen3vl_agent.factory import build_model
from qwen3vl_agent.r3.checkpoint import file_digest

from .acquisition import overview
from .config import MODEL, REVISION, R9Config
from .controller import R9VideoAgent
from .media import ScopedMedia
from .runtime import software_fingerprint
from .types import MODES, ProtocolError, R9Request, digest, finite, plain


def read_jsonl(path):
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(plain(value), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def public_request(row, root=None):
    if not isinstance(row, dict) or set(row) - {f.name for f in fields(R9Request)}:
        raise ProtocolError(
            "run manifest accepts only public R9Request fields; answers/classification belong offline"
        )
    values = dict(row)
    for key in ("video_path", "checkpoint_path"):
        if root and values.get(key) and not Path(values[key]).is_absolute():
            values[key] = str((Path(root) / values[key]).resolve())
    return R9Request(**values)


def model_and_config(path=None):
    data = load_config(path) if path else {}
    config = R9Config.from_mapping(data.get("r9"))
    model = dict(data.get("model") or {})
    model.setdefault("path", MODEL)
    model.setdefault("revision", REVISION)
    model.setdefault("dtype", "bfloat16")
    model.setdefault("device", "auto")
    model.setdefault("attn_implementation", "sdpa")
    model["generation"] = {**model.get("generation", {}), "temperature": 0.0}
    if model["revision"] != REVISION or (
        model["path"] != MODEL and not Path(model["path"]).is_dir()
    ):
        raise ProtocolError("R9 requires the pinned Qwen3-VL-8B-Instruct snapshot")
    return model, config


def preflight(request, config):
    media = ScopedMedia(request, config)
    packets = []
    for action in overview(
        request,
        media.contract,
        replace(config, initial_overview_frames=min(4, config.initial_overview_frames)),
    ):
        batch = media.extract(action["time_interval"], action["times"])
        prepared = media.prepare(batch)
        crop = media.prepare(media.local_batch(batch.frames[0].id, [[0.2, 0.2, 0.8, 0.8]]))
        packets.append(
            {
                "kind": prepared.kind,
                "frame_count": len(prepared.frames),
                "pixels": prepared.pixels,
                "video_frame_metadata": prepared.video_frame_metadata,
                "evidence": media.evidence(prepared),
                "crop_evidence": media.evidence(crop),
            }
        )
    return {
        "status": "passed",
        "validation_kind": "real_PTS_decode_and_input_structure",
        "model_loaded": False,
        "semantic_accuracy_measured": False,
        "source_sha256": media.source_hash,
        "contract": asdict(media.contract),
        "metadata": asdict(media.metadata),
        "packets": packets,
    }


def run_rows(
    rows,
    *,
    config_path=None,
    output,
    checkpoint_dir,
    root=None,
    resume=False,
    mode=None,
    model=None,
):
    settings, config = model_and_config(config_path)
    requests = [public_request(row, root) for row in rows]
    if mode:
        requests = [replace(r, mode=mode) for r in requests]
    if len({r.request_id for r in requests}) != len(requests):
        raise ProtocolError("duplicate request ID in run manifest")
    output = Path(output)
    if output.exists() and not resume:
        raise ProtocolError("output exists; use --resume or a new output")
    previous = {r["request_id"]: r for r in read_jsonl(output)} if output.exists() else {}
    if set(previous) - {r.request_id for r in requests}:
        raise ProtocolError("resumed output contains requests outside this manifest")
    model = model or build_model(settings)
    software = software_fingerprint(model, config)
    agent = R9VideoAgent(model, config)
    loaded, results = False, []
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        for request in requests:
            public = asdict(request)
            for k in ("resume", "checkpoint_path"):
                public.pop(k)
            signature = digest(
                {
                    "request": public,
                    "config": asdict(config),
                    "software": software,
                    "source_sha256": file_digest(request.video_path)
                    if Path(request.video_path).is_file()
                    else None,
                }
            )
            if request.request_id in previous:
                item = previous[request.request_id]
                if item["run_fingerprint"] != signature:
                    raise ProtocolError("resumed run media/config/request/software mismatch")
                results.append(item)
                continue
            item = {
                "request_id": request.request_id,
                "mode": request.mode,
                "comparison": request.comparison,
                "run_fingerprint": signature,
            }
            if not Path(request.video_path).is_file():
                item.update(run_status="media_unavailable", result=None)
            else:
                checkpoint = Path(checkpoint_dir) / (
                    digest(
                        {
                            "request_id": request.request_id,
                            "mode": request.mode,
                            "comparison": request.comparison,
                        }
                    )[:24]
                    + ".jsonl"
                )
                try:
                    if not loaded:
                        agent.load()
                        loaded = True
                    result = agent.solve(
                        replace(
                            request,
                            checkpoint_path=str(checkpoint),
                            resume=resume and checkpoint.exists(),
                        )
                    )
                    item.update(run_status="completed", result=result.to_dict())
                except Exception as exc:  # noqa: BLE001 -- retain per-question engineering failures
                    item.update(
                        run_status="engineering_error",
                        result=None,
                        error=type(exc).__name__ + ": " + str(exc),
                    )
            results.append(item)
            # Atomic complete-file rewrite also makes an interrupted final row recoverable.
            temp = output.with_suffix(output.suffix + ".tmp")
            temp.write_text(
                "".join(json.dumps(r, ensure_ascii=False, allow_nan=False) + "\n" for r in results),
                encoding="utf-8",
            )
            temp.replace(output)
    finally:
        if loaded:
            agent.unload()
    return results


def mra(prediction, target):
    if not finite(prediction) or not finite(target) or target <= 0:
        return 0.0
    error = abs(Decimal(str(prediction)) - Decimal(str(target))) / Decimal(str(target))
    return sum(error < Decimal(50 - i * 5) / 100 for i in range(10)) / 10


def score(predictions, answers):
    if len({a["request_id"] for a in answers}) != len(answers) or len(
        {p["request_id"] for p in predictions}
    ) != len(predictions):
        raise ProtocolError("duplicate IDs in scoring input")
    gold = {a["request_id"]: a for a in answers}
    if {p["request_id"] for p in predictions} - set(gold):
        raise ProtocolError("prediction without an offline answer")
    by_id = {p["request_id"]: p for p in predictions}
    rows = []
    for key, a in gold.items():
        p = by_id.get(key, {"run_status": "missing_prediction", "result": None})
        result = p.get("result") or {}
        pred = result.get("prediction")
        numeric = a["output_protocol"] == "numeric"
        unit_ok = not numeric or result.get("unit") == a.get("unit")
        correct = pred == a["answer"] and unit_ok if not numeric else None
        value = mra(pred, a["answer"]) if numeric and unit_ok else 0.0 if numeric else None
        rows.append(
            {
                "request_id": key,
                "dataset": a["dataset"],
                "native_task": a["native_task"],
                "mechanism": a["mechanism"],
                "boundary": a.get("boundary", False),
                "run_status": p["run_status"],
                "status": result.get("status"),
                "numeric": numeric,
                "correct": correct,
                "mra": value,
                "forced": result.get("forced_answer", False),
                "resources": result.get("trace", {}).get("resources", {}),
            }
        )

    def summary(group):
        mcq = [r for r in group if not r["numeric"]]
        numeric = [r for r in group if r["numeric"]]
        forced = [r for r in mcq if r["forced"]]
        costs = {}
        for key in (
            "model_calls",
            "processed_pixels",
            "unique_source_frames",
            "visual_exposures",
            "repeated_exposures",
            "crop_exposures",
            "presented_pixels",
            "input_tokens",
            "output_tokens",
            "visual_tokens",
            "model_seconds",
            "end_to_end_seconds",
        ):
            available = [r["resources"][key] for r in group if finite(r["resources"].get(key))]
            costs[key] = {
                "sum": sum(available) if available else None,
                "available_questions": len(available),
            }
        return {
            "count": len(group),
            "mcq_count": len(mcq),
            "accuracy": sum(r["correct"] for r in mcq) / len(mcq) if mcq else None,
            "numeric_count": len(numeric),
            "mra": sum(r["mra"] for r in numeric) / len(numeric) if numeric else None,
            "unresolved_count": sum(r["status"] in {"unresolved", "invalid_input"} for r in group),
            "forced_count": sum(r["forced"] for r in group),
            "forced_rate": sum(r["forced"] for r in group) / len(group) if group else None,
            "unresolved_rate": sum(r["status"] in {"unresolved", "invalid_input"} for r in group)
            / len(group)
            if group
            else None,
            "unresolved_before_fallback_count": sum(
                r["forced"] or r["status"] in {"unresolved", "invalid_input"} for r in group
            ),
            "forced_accuracy": sum(r["correct"] for r in forced) / len(forced) if forced else None,
            "missing_media": sum(r["run_status"] == "media_unavailable" for r in group),
            "engineering_errors": sum(r["run_status"] == "engineering_error" for r in group),
            "missing_predictions": sum(r["run_status"] == "missing_prediction" for r in group),
            "costs": costs,
        }

    groups = {
        field: {
            str(value): summary([r for r in rows if r[field] == value])
            for value in sorted({r[field] for r in rows}, key=str)
        }
        for field in ("dataset", "native_task", "mechanism", "boundary")
    }
    return {
        "overall": summary(rows),
        "groups": groups,
        "per_question": rows,
        "denominator": "all entries in the supplied offline answer manifest; missing runs score zero",
        "videomme_v2_scoring": "per-question only; filtered spatial sets are not complete official dependency groups",
        "process_correctness_metrics": None,
        "process_annotation_note": "independent annotations required",
    }


def parser():
    p = argparse.ArgumentParser(description="R9 source-linked spatial video pipeline")
    sub = p.add_subparsers(dest="command", required=True)
    for name in ("run", "preflight"):
        c = sub.add_parser(name)
        c.add_argument("--config")
        c.add_argument("--request")
        c.add_argument("--manifest")
        c.add_argument("--video")
        c.add_argument("--question", default="Inspect permitted video input")
        c.add_argument("--choice", action="append", default=[])
        c.add_argument("--output", required=True)
        c.add_argument("--mode", choices=MODES)
        if name == "run":
            c.add_argument("--checkpoint-dir", default=".cache/qwen3vl_agent/r9/checkpoints")
            c.add_argument("--resume", action="store_true")
    c = sub.add_parser("score")
    c.add_argument("--predictions", required=True)
    c.add_argument("--answers", required=True)
    c.add_argument("--output", required=True)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    if args.command == "score":
        write_json(args.output, score(read_jsonl(args.predictions), read_jsonl(args.answers)))
        return
    if sum(bool(v) for v in (args.request, args.manifest, args.video)) != 1:
        raise SystemExit("supply exactly one request, manifest or video")
    root = (
        Path(args.request or args.manifest).resolve().parent
        if args.request or args.manifest
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
        output = []
        for row in rows:
            request = public_request(row, root)
            output.append(
                {
                    "request_id": request.request_id,
                    **(
                        preflight(request, config)
                        if Path(request.video_path).is_file()
                        else {"status": "media_unavailable", "model_loaded": False}
                    ),
                }
            )
        write_json(args.output, {"results": output})
    else:
        run_rows(
            rows,
            config_path=args.config,
            root=root,
            output=args.output,
            checkpoint_dir=args.checkpoint_dir,
            resume=args.resume,
            mode=args.mode,
        )
    print(str(Path(args.output).resolve()))


if __name__ == "__main__":
    main()
