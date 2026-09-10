"""Answer-blind preflight/run, and a separate offline scorer."""

import argparse
import json
from collections import Counter
from dataclasses import fields, replace
from pathlib import Path

from qwen3vl_agent.config import load_config
from qwen3vl_agent.factory import build_model

from .config import MODEL, REVISION, R6Config
from .controller import R6VideoAgent
from .media import ScopedMedia
from .providers import TextSources
from .types import MODES, ProtocolError, R6Request, digest, plain


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
    if not isinstance(row, dict) or set(row) - {f.name for f in fields(R6Request)}:
        raise ProtocolError(
            "runtime accepts only R6Request fields; answers/annotations belong offline"
        )
    data = dict(row)
    for key in ("video_path", "subtitle_path", "asr_path", "checkpoint_path"):
        if root and data.get(key) and not Path(data[key]).is_absolute():
            data[key] = str((Path(root) / data[key]).resolve())
    return R6Request(**data)


def model_and_config(path=None):
    raw = load_config(path) if path else {}
    config = R6Config.from_mapping(raw.get("r6"))
    model = dict(raw.get("model") or {})
    model.setdefault("path", MODEL)
    model.setdefault("revision", REVISION)
    model.setdefault("dtype", "bfloat16")
    model.setdefault("device", "auto")
    model.setdefault("attn_implementation", "sdpa")
    model["generation"] = {**model.get("generation", {}), "temperature": 0.0, "do_sample": False}
    if model["revision"] != REVISION:
        raise ProtocolError("R6 requires the pinned Qwen3-VL-8B revision")
    if model["path"] != MODEL and not Path(model["path"]).is_dir():
        raise ProtocolError("model must be the official ID or an existing frozen local snapshot")
    return model, config


def preflight(request, config):
    if not Path(request.video_path).is_file():
        return {
            "status": "media_unavailable",
            "model_loaded": False,
            "request_id": request.request_id,
        }
    media = ScopedMedia(request, config)
    texts = TextSources(request, media.contract)
    decoded = []
    if "video" in request.allowed_modalities:
        for span in list(
            dict.fromkeys(
                (media.contract.allowed_intervals[0], media.contract.allowed_intervals[-1])
            )
        ):
            a, b = span
            batch = media.extract(span, [a + (b - a) * 0.25, a + (b - a) * 0.75])
            prepared = media.prepare(batch)
            decoded.append(
                {
                    "source_ids": [f.id for f in prepared.frames],
                    "kind": prepared.kind,
                    "pixels": prepared.pixels,
                    "video_metadata": prepared.video_frame_metadata,
                }
            )
    return {
        "status": "ready",
        "request_id": request.request_id,
        "model_loaded": False,
        "media_hash": media.source_hash,
        "allowed_intervals": media.contract.allowed_intervals,
        "decoded_checks": decoded,
        "aligned_text_views": len(texts.sources),
        "text_issues": texts.issues,
        "audio_available": False,
    }


def run_requests(requests, *, model_settings, config, output, model=None, resume=False):
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    signature = digest(
        {
            "requests": [
                r.__dict__ | {"choices": [c.__dict__ for c in r.choices]} for r in requests
            ],
            "config": config.to_dict(),
            "model": model_settings,
        }
    )
    identity = root / "run_identity.json"
    if identity.exists() and (
        not resume or json.loads(identity.read_text())["signature"] != signature
    ):
        raise ProtocolError("existing output requires resume and an unchanged run signature")
    write_json(identity, {"signature": signature, "audio_available": False})
    agent = R6VideoAgent(model or build_model(model_settings), config)
    predictions = []
    loaded = False
    try:
        for request in requests:
            if not Path(request.video_path).is_file():
                predictions.append(
                    {
                        "request_id": request.request_id,
                        "prediction": None,
                        "run_status": "media_unavailable",
                    }
                )
                continue
            if not loaded:
                agent.load()
                loaded = True
            key = digest({"id": request.request_id, "mode": request.mode})[:24]
            checkpoint = root / "checkpoints" / (key + ".jsonl")
            item = replace(
                request, checkpoint_path=str(checkpoint), resume=resume and checkpoint.exists()
            )
            result = agent.solve(item)
            full = result.to_dict()
            write_json(root / "traces" / (key + ".json"), full)
            predictions.append(
                {k: v for k, v in full.items() if k != "trace"}
                | {"run_status": "completed", "trace_file": f"traces/{key}.json"}
            )
    finally:
        if loaded:
            agent.unload()
    (root / "predictions.jsonl").write_text(
        "".join(json.dumps(p, ensure_ascii=False) + "\n" for p in predictions), encoding="utf-8"
    )
    write_json(
        root / "summary.json",
        {
            "questions": len(predictions),
            "accuracy": None,
            "stop_reasons": dict(
                Counter(p.get("stop_reason", p["run_status"]) for p in predictions)
            ),
            "run_signature": signature,
            "note": "Runtime never receives standard answers.",
        },
    )
    return predictions


def score(predictions, answers):
    def keyed(rows):
        result = {}
        for row in rows:
            if row["request_id"] in result:
                raise ProtocolError("duplicate request ID in score input")
            result[row["request_id"]] = row
        return result

    predicted, gold = keyed(predictions), keyed(answers)
    if set(predicted) - gold.keys():
        raise ProtocolError("prediction has no corresponding offline label")
    rows = []
    for rid, answer in gold.items():
        p = predicted.get(rid, {})
        rows.append(
            {
                "request_id": rid,
                "correct": p.get("prediction") == answer["answer"],
                "prediction": p.get("prediction"),
                "answer": answer["answer"],
                "forced_choice": p.get("forced_choice", False),
                "evidence_status": p.get("evidence_status", "missing"),
                "subtype": answer.get("subtype", "unspecified"),
            }
        )
    return {
        "questions": len(rows),
        "correct": sum(r["correct"] for r in rows),
        "accuracy": sum(r["correct"] for r in rows) / len(rows) if rows else None,
        "forced_choice_count": sum(r["forced_choice"] for r in rows),
        "per_question": rows,
        "denominator": "all supplied offline labels, including missing predictions",
        "videomme_v2": "per-question only; no official group score for this selected dev set",
        "process_correctness": None,
        "process_note": "requires independent evidence review",
    }


def parser():
    p = argparse.ArgumentParser(description="R6 source-grounded relationship pipeline")
    sub = p.add_subparsers(dest="command", required=True)
    for name in ("preflight", "run"):
        c = sub.add_parser(name)
        group = c.add_mutually_exclusive_group(required=True)
        group.add_argument("--request", help="single answer-blind request JSON")
        group.add_argument("--manifest", help="answer-blind requests JSONL")
        c.add_argument("--config")
        c.add_argument("--mode", choices=MODES)
        c.add_argument(
            "--output", required=True, help="JSON report for preflight, output directory for run"
        )
        if name == "run":
            c.add_argument("--resume", action="store_true")
    c = sub.add_parser("score")
    for field in ("predictions", "answers", "output"):
        c.add_argument("--" + field, required=True)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    if args.command == "score":
        write_json(args.output, score(read_jsonl(args.predictions), read_jsonl(args.answers)))
        return
    path = Path(args.manifest or args.request).resolve()
    rows = read_jsonl(path) if args.manifest else [json.loads(path.read_text(encoding="utf-8-sig"))]
    requests = [public_request(r, path.parent) for r in rows]
    if not requests or len({r.request_id for r in requests}) != len(requests):
        raise ProtocolError("manifest must contain distinct, nonempty request IDs")
    if args.mode:
        requests = [replace(r, mode=args.mode) for r in requests]
    model, config = model_and_config(args.config)
    if args.command == "preflight":
        write_json(args.output, {"results": [preflight(r, config) for r in requests]})
    else:
        run_requests(
            requests, model_settings=model, config=config, output=args.output, resume=args.resume
        )


if __name__ == "__main__":
    main()
