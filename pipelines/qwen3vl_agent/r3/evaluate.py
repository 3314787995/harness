"""Batch execution for an explicitly preselected R3 JSONL manifest.

Run with ``python -m qwen3vl_agent.r3.evaluate --help``. Benchmark annotations
stay in the sidecar and never become fields of the inference request.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, fields, replace
from pathlib import Path
from typing import Any

from qwen3vl_agent.config import load_config
from qwen3vl_agent.r3.agent import R3VideoAgent
from qwen3vl_agent.r3.config import R3Config
from qwen3vl_agent.r3.planning import make_tiles, planned_calls, resolve_allowed
from qwen3vl_agent.r3.types import R3Request


def read_manifest(path: str | Path) -> list[tuple[R3Request, dict[str, Any]]]:
    path = Path(path).resolve()
    allowed = {f.name for f in fields(R3Request)} - {"checkpoint_path", "resume"}
    rows, seen = [], set()
    for line_number, text in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not text.strip():
            continue
        try:
            row = json.loads(text)
            if not isinstance(row, dict) or row.get("pipeline_id", "R3").upper() != "R3":
                raise ValueError("manifest must contain already-selected R3 objects")
            request_id = row.get("request_id")
            if not isinstance(request_id, str) or not request_id.strip() or request_id in seen:
                raise ValueError("a nonempty, unique request_id is required")
            if {"checkpoint_path", "resume"} & row.keys():
                raise ValueError("checkpoint paths and resume are managed by the batch runner")
            data = {key: value for key, value in row.items() if key in allowed}
            video = Path(data["video_path"]).expanduser()
            data["video_path"] = str(
                (path.parent / video).resolve() if not video.is_absolute() else video.resolve()
            )
            request = R3Request(**data)
        except (ValueError, TypeError, KeyError, AttributeError) as exc:
            raise ValueError(f"{path}:{line_number}: {exc}") from exc
        seen.add(request_id)
        rows.append((request, {key: value for key, value in row.items() if key not in allowed}))
    if not rows:
        raise ValueError("R3 manifest is empty")
    return rows


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"


def preflight(
    rows: list[tuple[R3Request, dict[str, Any]]], config: R3Config, probe: Any = None
) -> dict[str, Any]:
    """No model load and no semantic coverage claims; estimates precede compilation."""
    if probe is None:
        from qwen3vl_agent.r1.config import R1Config
        from qwen3vl_agent.r1.media import R1Media

        probe = R1Media(R1Config(media=config.media))
    report = []
    for request, _ in rows:
        row: dict[str, Any] = {"request_id": request.request_id, "video_path": request.video_path}
        try:
            metadata = probe.probe(request.video_path)
            allowed = resolve_allowed(request, metadata.duration_seconds)
            scope = (
                request.query_scope
                if request.query_scope is not None and not isinstance(request.query_scope, str)
                else allowed
            )
            if (
                scope.start_seconds < allowed.start_seconds
                or scope.end_seconds > allowed.end_seconds
            ):
                raise ValueError("query scope exceeds allowed scope/cutoff")
            estimates = {}
            for name, rate in (("episode", config.episode_fps), ("cycle", config.cycle_fps)):
                rate = min(rate, metadata.source_fps) if metadata.source_fps else rate
                tiles = make_tiles(scope, allowed, rate, config)
                estimates[name] = {
                    "fps": rate,
                    "base_windows": len(tiles),
                    "planned_model_calls": planned_calls(len(tiles)),
                }
            row.update(
                status="ready",
                allowed_scope=asdict(allowed),
                estimates=estimates,
                model_call_cap=min(request.budget.max_model_calls, config.budget.max_model_calls),
                external_provider_required_at_runtime=bool(
                    set(request.available_modalities) & {"asr", "subtitle"}
                ),
            )
        except (ValueError, RuntimeError, OSError, ImportError) as exc:
            row.update(status="input_error", error=str(exc))
        report.append(row)
    return {
        "pipeline_id": "R3",
        "mode": "preflight",
        "model_calls": 0,
        "semantic_coverage": False,
        "requests": report,
    }


def run_manifest(
    agent: R3VideoAgent,
    rows: list[tuple[R3Request, dict[str, Any]]],
    output_dir: str | Path,
    *,
    resume: bool = False,
) -> dict[str, Any]:
    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    identity = {
        "requests": [asdict(request) for request, _ in rows],
        "evaluation_metadata": [extra for _, extra in rows],
        "config": asdict(agent.config),
    }
    identity_path = destination / "run_manifest.json"
    encoded = json_text(identity)
    if identity_path.exists():
        if not resume or identity_path.read_text(encoding="utf-8") != encoded:
            raise ValueError("existing output requires --resume and an identical manifest/config")
    else:
        if any(p.name != "preflight.json" for p in destination.iterdir()):
            raise ValueError("new R3 output directory must be empty")
        atomic_text(identity_path, encoded)
    predictions, diagnostics = [], []
    for request, extra in rows:
        stem = hashlib.sha256(request.request_id.encode()).hexdigest()[:24]
        checkpoint = destination / "checkpoints" / (stem + ".jsonl")
        # Even a finalized checkpoint is reopened by solve so video/config fingerprints
        # are checked. Skipping based on the existence of a prediction is unsafe.
        result = agent.solve(
            replace(request, checkpoint_path=str(checkpoint), resume=resume and checkpoint.exists())
        )
        predictions.append(
            {
                "request_id": request.request_id,
                "video_id": request.video_id,
                "group_id": request.group_id,
                "prediction": result.prediction,
            }
        )
        diagnostics.append(
            {
                "request_id": request.request_id,
                "completion_state": result.completion_state,
                "support_level": result.support_level,
                "resources": result.resources,
            }
        )
        atomic_text(
            destination / "sidecars" / (stem + ".json"),
            json_text(
                {"request": asdict(request), "evaluation_metadata": extra, **result.to_dict()}
            ),
        )
        atomic_text(
            destination / "predictions.jsonl",
            "".join(json.dumps(p, ensure_ascii=False, allow_nan=False) + "\n" for p in predictions),
        )
        atomic_text(destination / "diagnostics.json", json_text({"requests": diagnostics}))
    summary = {
        "pipeline_id": "R3",
        "request_count": len(predictions),
        "supported_count": sum(d["support_level"] == "supported" for d in diagnostics),
        "model_calls": sum(d["resources"].get("model_calls", 0) for d in diagnostics),
        "accuracy": None,
        "event_diagnostics": None,
        "note": "Native predictions only; use the official evaluator. No event-level scores inferred.",
    }
    atomic_text(destination / "summary.json", json_text(summary))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--config", default="configs/r3_8b.yaml")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="Validate media and estimate budgets without loading Qwen",
    )
    args = parser.parse_args()
    rows = read_manifest(args.manifest)
    configuration = load_config(args.config)
    config = R3Config.from_mapping(configuration.get("r3"))
    if args.preflight:
        result = preflight(rows, config)
        atomic_text(Path(args.output_dir).resolve() / "preflight.json", json_text(result))
        print(json_text(result), end="")
        return
    from qwen3vl_agent.factory import build_model

    model_config = {
        "path": "Qwen/Qwen3-VL-8B-Instruct",
        "dtype": "bfloat16",
        "attn_implementation": "flash_attention_2",
        **configuration.get("model", {}),
    }
    agent = R3VideoAgent(build_model(model_config), config=config)
    agent.load()
    try:
        print(json_text(run_manifest(agent, rows, args.output_dir, resume=args.resume)), end="")
    finally:
        agent.unload()


if __name__ == "__main__":
    main()
