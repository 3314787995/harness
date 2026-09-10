"""R7 batch/preflight/score CLI. Gold is accepted only by the offline score command."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict, fields, replace
from pathlib import Path

from qwen3vl_agent.config import load_config
from qwen3vl_agent.r3.evaluate import atomic_text, json_text

from .agent import R7VideoAgent
from .config import R7Config
from .media import ScopedMedia, sampling, windows
from .types import MODES, R7Request, digest, plain


def write_json(path, value):
    atomic_text(Path(path), json_text(value))


def read_manifest(path, *, mode=None):
    path = Path(path).resolve()
    allowed = {f.name for f in fields(R7Request)} - {"checkpoint_path", "resume"}
    metadata_fields = {
        "pipeline_id",
        "benchmark",
        "native_task",
        "source",
        "boundary",
        "example_id",
    }
    rows, seen = [], set()
    for index, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        raw = json.loads(line)
        if set(raw) - allowed - metadata_fields or raw.get("pipeline_id", "R7") != "R7":
            raise ValueError(
                f"{path}:{index}: unsupported field; answers/checkpoints are not runtime inputs"
            )
        data = {k: v for k, v in raw.items() if k in allowed}
        for key in ("video_path", "subtitle_path", "facts_input"):
            if data.get(key):
                value = Path(data[key]).expanduser()
                data[key] = str(
                    (path.parent / value).resolve() if not value.is_absolute() else value.resolve()
                )
        if mode:
            data["mode"] = mode
        request = R7Request(**data)
        if request.request_id in seen:
            raise ValueError("duplicate request_id")
        seen.add(request.request_id)
        rows.append((request, {k: v for k, v in raw.items() if k in metadata_fields}))
    if not rows:
        raise ValueError("empty manifest")
    return rows


def preflight(rows, config):
    reports = []
    for request, info in rows:
        report = {
            "request_id": request.request_id,
            "video_id": request.video_id,
            "boundary": info.get("boundary", False),
            "video_path": request.video_path,
        }
        try:
            if not Path(request.video_path).is_file():
                report["status"] = "missing_media"
            else:
                media = ScopedMedia(
                    request, config, {"mode": "preflight", "model_revision": config.model_revision}
                )
                all_windows = windows(media.contract, config)
                checked = []
                for window in (
                    [all_windows[0], all_windows[-1]] if len(all_windows) > 1 else all_windows
                ):
                    times, fps = sampling(
                        window,
                        config,
                        tail=request.execution_subtype == "S1",
                        source_fps=media.metadata.source_fps,
                    )
                    batch = media.extract(window, times, fps=fps)
                    prepared = media.prepare(batch)
                    if prepared.video_frame_metadata:
                        from types import SimpleNamespace

                        from qwen3vl_agent.models.qwen3vl import Qwen3VLModel

                        Qwen3VLModel._validated_frame_metadata(
                            [
                                SimpleNamespace(shape=(len(m["frame_ids"]),))
                                for m in prepared.video_frame_metadata
                            ],
                            prepared.video_frame_metadata,
                        )
                    checked.append(
                        {
                            "coverage": media.coverage(batch, False),
                            "kind": prepared.kind,
                            "all_pts_permitted": all(
                                media.contract.permits(f.timestamp_seconds) for f in batch.frames
                            ),
                            "pixels": prepared.pixels,
                        }
                    )
                report.update(
                    status="ready",
                    source_sha256=media.source_hash,
                    duration_seconds=media.metadata.duration_seconds,
                    contract=asdict(media.contract),
                    budget=config.budget(media.contract, request),
                    decoded_checks=checked,
                    subtitle_cues=len(media.subtitles),
                )
        except (ValueError, OSError, RuntimeError, ImportError, KeyError) as exc:
            report.update(status="input_error", error=f"{type(exc).__name__}: {exc}")
        reports.append(report)
    counts = Counter(r["status"] for r in reports)
    return {
        "pipeline_id": "R7",
        "mode": "preflight",
        "model_loaded": False,
        "accuracy": None,
        "questions": len(rows),
        "unique_videos": len({r.video_id for r, _ in rows}),
        "counts": dict(counts),
        "ready": counts["ready"] == len(rows),
        "items": reports,
    }


def run_manifest(rows, agent, output_dir, *, resume=False):
    root = Path(output_dir).resolve()
    identity = {
        "requests": [
            {k: v for k, v in asdict(r).items() if k not in {"resume", "checkpoint_path"}}
            for r, _ in rows
        ],
        "config": asdict(agent.config),
    }
    identity = plain(identity)
    path = root / "run_identity.json"
    if (
        root.exists()
        and any(root.iterdir())
        and (
            not resume
            or not path.is_file()
            or json.loads(path.read_text(encoding="utf-8")) != identity
        )
    ):
        raise ValueError("output exists; matching identity and --resume required")
    root.mkdir(parents=True, exist_ok=True)
    write_json(path, identity)
    predictions, diagnostics = [], []
    for request, info in rows:
        stem = digest(request.request_id)[:20]
        checkpoint = root / "checkpoints" / (stem + ".jsonl")
        current = replace(
            request, checkpoint_path=str(checkpoint), resume=resume and checkpoint.exists()
        )
        result = agent.solve(current)
        diagnostics.append({"request_id": request.request_id, **result.to_dict()})
        predictions.append(
            {
                "request_id": request.request_id,
                "video_id": request.video_id,
                "group_id": request.group_id,
                "prediction": result.prediction,
                "mode": request.mode,
                "completion_state": result.completion_state,
                "support_level": result.support_level,
                "mechanisms": result.task_spec.get("mechanisms", []),
                "single_candidate": len(request.choices) == 1,
                "boundary": info.get("boundary", False),
                "target_visibility": request.target_visibility,
                "semantic_visibility": result.task_spec.get("target_visibility", "unresolved"),
                "protocol_id": request.protocol_id,
                "resources": {k: v for k, v in result.resources.items() if k != "receipts"},
            }
        )
        write_json(root / "diagnostics" / (stem + ".json"), diagnostics[-1])
        if result.task_spec and result.facts.get("observations"):
            artifact = {
                "identity": {
                    "question": request.question,
                    "options": [asdict(c) for c in request.choices],
                    "scope": result.trace["input_contract"],
                    "source_sha256": result.trace["source_sha256"],
                },
                "task_spec": result.task_spec,
                "candidates": result.candidates,
                "facts": result.facts,
                "software": result.trace["software"],
            }
            write_json(root / "facts" / (stem + ".json"), artifact)
        atomic_text(
            root / "predictions.jsonl",
            "".join(json.dumps(p, ensure_ascii=False) + "\n" for p in predictions),
        )
    summary = {
        "pipeline_id": "R7",
        "questions": len(rows),
        "counts": dict(Counter(p["completion_state"] for p in predictions)),
        "predictions_present": sum(p["prediction"] is not None for p in predictions),
        "model_calls": sum(p["resources"].get("model_calls", 0) for p in predictions),
        "accuracy": None,
        "score_note": "Run the offline score command with a separate gold file.",
    }
    write_json(root / "summary.json", summary)
    return summary


def score(predictions_path, gold_path):
    predictions = [
        json.loads(s)
        for s in Path(predictions_path).read_text(encoding="utf-8").splitlines()
        if s.strip()
    ]
    gold_rows = [
        json.loads(s) for s in Path(gold_path).read_text(encoding="utf-8").splitlines() if s.strip()
    ]
    if len({p["request_id"] for p in predictions}) != len(predictions) or len(
        {g["request_id"] for g in gold_rows}
    ) != len(gold_rows):
        raise ValueError("duplicate prediction/gold IDs")
    gold = {g["request_id"]: g["answer"] for g in gold_rows}
    if {p["request_id"] for p in predictions} - gold.keys():
        raise ValueError("missing gold labels")
    groups = {
        "all": predictions,
        "K_gt_1": [p for p in predictions if not p["single_candidate"]],
        "single_candidate": [p for p in predictions if p["single_candidate"]],
        "boundary": [p for p in predictions if p["boundary"]],
        "mechanism_development": [p for p in predictions if not p["boundary"]],
        "declared_future": [
            p for p in predictions if p["target_visibility"] == "unobserved_future"
        ],
        "visibility_unresolved": [p for p in predictions if p["target_visibility"] == "unresolved"],
    }
    for mechanism in ("S1", "S2", "S3", "S4", "S5"):
        groups[mechanism] = [
            p for p in predictions if mechanism in p["mechanisms"] and not p["boundary"]
        ]
    scores = {}
    for name, rows in groups.items():
        n = len(rows)
        correct = sum(
            p["prediction"] is not None and p["prediction"] == gold[p["request_id"]] for p in rows
        )
        acc = correct / n if n else None
        # Wilson interval; video-correlated samples still require grouped resampling in research.
        if n:
            z = 1.959963984540054
            center = (acc + z * z / (2 * n)) / (1 + z * z / n)
            half = z * ((acc * (1 - acc) / n + z * z / (4 * n * n)) ** 0.5) / (1 + z * z / n)
            interval = [center - half, center + half]
        else:
            interval = None
        scores[name] = {"n": n, "correct": correct, "accuracy": acc, "wilson95": interval}
    return {
        "pipeline_id": "R7",
        "scores": scores,
        "official_group_score": None,
        "note": "Development/boundary subset; not held-out or official Video-MME-v2 group scoring. Wilson intervals assume independent questions.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("preflight", "run", "score"))
    parser.add_argument("--manifest")
    parser.add_argument("--config")
    parser.add_argument("--output", required=True)
    parser.add_argument("--mode", choices=MODES)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--predictions")
    parser.add_argument("--gold")
    parser.add_argument("--facts-dir", help="Reuse matching R7 observation artifacts for B2/B3/B4")
    args = parser.parse_args()
    if args.command == "score":
        if not args.predictions or not args.gold:
            parser.error("score requires --predictions and --gold")
        value = score(args.predictions, args.gold)
        write_json(args.output, value)
    else:
        if not args.manifest:
            parser.error("--manifest is required")
        mapping = load_config(args.config) if args.config else {}
        config = R7Config.from_mapping(mapping.get("r7"))
        rows = read_manifest(args.manifest, mode=args.mode)
        if args.facts_dir:
            rows = [
                (
                    replace(
                        r,
                        facts_input=str(
                            (Path(args.facts_dir) / f"{digest(r.request_id)[:20]}.json").resolve()
                        ),
                    ),
                    info,
                )
                for r, info in rows
            ]
            missing = [r.request_id for r, _ in rows if not Path(r.facts_input).is_file()]
            if missing:
                parser.error("missing observation replay artifacts: " + ", ".join(missing))
        if args.command == "preflight":
            value = preflight(rows, config)
            write_json(args.output, value)
        else:
            from qwen3vl_agent.factory import build_model

            model_config = dict(mapping.get("model") or {})
            model_config.setdefault("path", "Qwen/Qwen3-VL-8B-Instruct")
            model_config.setdefault("revision", config.model_revision)
            agent = R7VideoAgent(build_model(model_config), config=config)
            agent.load()
            try:
                value = run_manifest(rows, agent, args.output, resume=args.resume)
            finally:
                agent.unload()
    print(
        json.dumps({k: v for k, v in value.items() if k != "items"}, ensure_ascii=False, indent=2)
    )


if __name__ == "__main__":
    main()
