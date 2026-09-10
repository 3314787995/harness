"""Run a preselected R5 JSONL manifest, or inspect its inputs without loading a model."""

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
from qwen3vl_agent.r5.agent import R5VideoAgent
from qwen3vl_agent.r5.checkpoint import implementation_digest
from qwen3vl_agent.r5.observation import PROTOCOL_VERSION
from qwen3vl_agent.r5.config import R5Config
from qwen3vl_agent.r5.planning import estimate, make_plan, resolve_source
from qwen3vl_agent.r5.providers import read_external_file
from qwen3vl_agent.r5.types import R5Request


def read_manifest(path: str | Path) -> list[tuple[R5Request, dict[str, Any]]]:
    path = Path(path).resolve()
    permitted = {f.name for f in fields(R5Request)} - {"checkpoint_path", "resume"}
    rows, seen = [], set()

    def absolute(value: str) -> str:
        file = Path(value).expanduser()
        return str((path.parent / file).resolve() if not file.is_absolute() else file.resolve())

    for number, text in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not text.strip():
            continue
        try:
            row = json.loads(text)
            if not isinstance(row, dict) or row.get("pipeline_id", "R5") != "R5":
                raise ValueError("manifest must contain preselected R5 requests")
            key = row.get("request_id")
            if not isinstance(key, str) or not key.strip() or key in seen:
                raise ValueError("unique nonempty request_id required")
            if {"checkpoint_path", "resume"} & row.keys():
                raise ValueError("batch runner owns checkpoints and resume")
            original = copy.deepcopy(row)
            data = {k: copy.deepcopy(v) for k, v in row.items() if k in permitted}
            data["video_path"] = absolute(data["video_path"])
            for item in data.get("external_files", []):
                item["path"] = absolute(item["path"])
            request = R5Request(**data)
        except (ValueError, TypeError, KeyError, AttributeError) as exc:
            raise ValueError(f"{path}:{number}: {exc}") from exc
        seen.add(key)
        rows.append(
            (
                request,
                {
                    **{k: v for k, v in row.items() if k not in permitted},
                    "original_record": original,
                },
            )
        )
    if not rows:
        raise ValueError("R5 manifest is empty")
    return rows


def preflight(rows: list[tuple[R5Request, dict]], config: R5Config, probe: Any = None) -> dict:
    if probe is None:
        from qwen3vl_agent.r1.config import R1Config
        from qwen3vl_agent.r5.media import R5Media

        probe = R5Media(R1Config(media=config.media))
    reports = []
    for request, _ in rows:
        report = {"request_id": request.request_id}
        try:
            source, scope = resolve_source(request, probe)
            plan = make_plan(scope, config, source["source_fps"])
            planning = estimate(plan, config, request.available_modalities)
            files = []
            for file in request.external_files:
                items = read_external_file(file, source["source_id"])
                files.append(
                    {
                        "path": file.path,
                        "kind": file.kind,
                        "segments": len(items),
                        "coverage_status": file.coverage_status,
                    }
                )
            cap = min(config.budget.max_model_calls, request.budget.max_model_calls)
            report.update(
                status="ready",
                source_id=source["source_id"],
                estimates=planning,
                model_call_cap=cap,
                budget_limited_possible=planning["planned_model_calls"] > cap,
                query_scope=[scope.start_seconds, scope.end_seconds],
                external_files=files,
                unavailable_file_modalities=sorted(
                    set(request.available_modalities)
                    & {"subtitle", "asr"} - {f.kind for f in request.external_files}
                ),
            )
        except (OSError, ValueError, RuntimeError, TypeError, ImportError) as exc:
            report.update(status="input_error", error=str(exc))
        reports.append(report)
    return {
        "pipeline_id": "R5",
        "mode": "preflight",
        "model_calls": 0,
        "semantic_coverage": False,
        "requests": reports,
    }


def run_manifest(
    agent: R5VideoAgent,
    rows: list[tuple[R5Request, dict]],
    output_dir: str | Path,
    *,
    resume: bool = False,
) -> dict:
    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    run_identity = {
        "protocol_version": PROTOCOL_VERSION,
        "implementation": implementation_digest(),
        "requests": [asdict(r) for r, _ in rows],
        "evaluation_metadata": [x for _, x in rows],
        "config": asdict(agent.config),
    }
    identity_path = destination / "run_manifest.json"
    encoded = json_text(run_identity)
    if identity_path.exists():
        if not resume or identity_path.read_text(encoding="utf-8") != encoded:
            raise ValueError("existing output requires resume and identical manifest/config")
    else:
        if any(p.name != "preflight.json" for p in destination.iterdir()):
            raise ValueError("new R5 output directory must be empty")
        atomic_text(identity_path, encoded)
    predictions, native, diagnostics = [], [], []
    for request, extra in rows:
        stem = hashlib.sha256(request.request_id.encode()).hexdigest()[:24]
        checkpoint = destination / "checkpoints" / (stem + ".jsonl")
        result = agent.solve(
            replace(request, checkpoint_path=str(checkpoint), resume=resume and checkpoint.exists())
        )
        prediction = result.prediction or None
        predictions.append(
            {
                "request_id": request.request_id,
                "video_id": request.video_id,
                "group_id": request.group_id,
                "prediction": prediction,
            }
        )
        native.append({**extra.get("original_record", {}), "prediction": prediction})
        diagnostics.append(
            {
                "request_id": request.request_id,
                "execution_subtype": result.execution_subtype,
                "completion_state": result.completion_state,
                "support_level": result.support_level,
                "answer_basis": result.answer_basis,
                "verification_status": result.verification_status,
                "evidence_ref_count": len(result.evidence_refs),
                "coverage": result.coverage,
                "unresolved_items": result.unresolved_items,
                "resources": result.resources,
            }
        )
        atomic_text(
            destination / "sidecars" / (stem + ".json"),
            json_text(
                {"request": asdict(request), "evaluation_metadata": extra, **result.to_dict()}
            ),
        )
        for name, values in (("predictions", predictions), ("native_predictions", native)):
            atomic_text(
                destination / (name + ".jsonl"),
                "".join(json.dumps(v, ensure_ascii=False, allow_nan=False) + "\n" for v in values),
            )
        atomic_text(destination / "diagnostics.json", json_text({"requests": diagnostics}))
        if result.completion_state == "input_error":
            raise RuntimeError(f"R5 input failure: {request.request_id}: {result.unresolved_items}")
    summary = {
        "pipeline_id": "R5",
        "request_count": len(predictions),
        "answered_count": sum(bool(p["prediction"]) for p in predictions),
        "answers_with_evidence": sum(bool(p["prediction"]) and d["evidence_ref_count"] > 0
                                     for p, d in zip(predictions, diagnostics)),
        "unverified_answers": sum(bool(p["prediction"]) and d["verification_status"] == "not_performed"
                                   for p, d in zip(predictions, diagnostics)),
        "coverage_complete_count": sum(d["coverage"].get("complete", False) for d in diagnostics),
        "model_calls": sum(d["resources"].get("model_calls", 0) for d in diagnostics),
        "accuracy": None,
        "note": "Use the native evaluator; support and coverage diagnostics are not accuracy scores.",
    }
    atomic_text(destination / "summary.json", json_text(summary))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--config", default="configs/r5_8b.yaml")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    rows = read_manifest(args.manifest)
    configuration = load_config(args.config)
    config = R5Config.from_mapping(configuration.get("r5"))
    if args.preflight:
        result = preflight(rows, config)
        atomic_text(Path(args.output_dir).resolve() / "preflight.json", json_text(result))
        print(json_text(result), end="")
        return
    from qwen3vl_agent.factory import build_model

    model = build_model(
        {
            "path": "Qwen/Qwen3-VL-8B-Instruct",
            "dtype": "bfloat16",
            "attn_implementation": "flash_attention_2",
            **configuration.get("model", {}),
        }
    )
    agent = R5VideoAgent(model, config)
    agent.load()
    try:
        print(json_text(run_manifest(agent, rows, args.output_dir, resume=args.resume)), end="")
    finally:
        agent.unload()


if __name__ == "__main__":
    main()
