from __future__ import annotations

import importlib.metadata
import json
import platform
import sys
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from qwen3vl_agent.config import load_config
from qwen3vl_agent.evaluation.evidence30 import POLICY_ID, canonical_sha256, sha256_file

_PACKAGES = (
    "accelerate",
    "av",
    "jsonschema",
    "pyarrow",
    "PyYAML",
    "qwen-vl-utils",
    "torch",
    "torchvision",
    "transformers",
)


def _relative_hashes(root: Path, files: list[Path]) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): sha256_file(path)
        for path in sorted(files)
        if path.is_file()
    }


def _source_hashes(project_root: Path, config_path: Path) -> dict[str, str]:
    files = list((project_root / "qwen3vl_agent").rglob("*.py"))
    files.extend((project_root / "schemas").glob("*.json"))
    files.extend([project_root / "pyproject.toml", config_path])
    unique = sorted({path.resolve() for path in files if path.is_file()})
    result: dict[str, str] = {}
    for path in unique:
        try:
            key = path.relative_to(project_root).as_posix()
        except ValueError:
            key = str(path)
        result[key] = sha256_file(path)
    return result


def _environment() -> dict[str, Any]:
    packages: dict[str, str | None] = {}
    for package in _PACKAGES:
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
    accelerator: dict[str, Any] = {"cuda_available": False, "devices": []}
    try:
        import torch

        accelerator["cuda_available"] = bool(torch.cuda.is_available())
        accelerator["torch_cuda"] = torch.version.cuda
        if torch.cuda.is_available():
            accelerator["devices"] = [
                torch.cuda.get_device_name(index)
                for index in range(torch.cuda.device_count())
            ]
    except Exception as exc:  # noqa: BLE001 - environment capture must remain diagnostic
        accelerator["probe_error"] = f"{type(exc).__name__}: {exc}"
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": packages,
        "accelerator": accelerator,
    }


def build_freeze_payload(
    *,
    project_root: str | Path,
    config_path: str | Path,
    evidence_root: str | Path,
    model_path: str | Path,
) -> dict[str, Any]:
    project = Path(project_root).expanduser().resolve()
    config = Path(config_path).expanduser().resolve()
    evidence = Path(evidence_root).expanduser().resolve()
    model = Path(model_path).expanduser().resolve()
    if not config.is_file():
        raise FileNotFoundError(config)
    if not evidence.is_dir():
        raise FileNotFoundError(evidence)
    if not model.is_dir():
        raise FileNotFoundError(model)

    annotation_files = [
        evidence / name
        for name in ("manifest.json", "dev.jsonl", "locked.jsonl", "excluded.jsonl")
    ]
    missing_annotations = [str(path) for path in annotation_files if not path.is_file()]
    if missing_annotations:
        raise FileNotFoundError(
            "Missing Evidence30 artifacts: " + ", ".join(missing_annotations)
        )
    model_files = [path for path in model.rglob("*") if path.is_file()]
    return {
        "policy_id": POLICY_ID,
        "project_root": str(project),
        "config_path": str(config),
        "evidence_root": str(evidence),
        "model_path": str(model),
        "resolved_config": load_config(config),
        "source_files": _source_hashes(project, config),
        "annotation_files": _relative_hashes(evidence, annotation_files),
        "model_files": _relative_hashes(model, model_files),
        "environment": _environment(),
    }


def create_freeze_manifest(
    *,
    project_root: str | Path,
    config_path: str | Path,
    evidence_root: str | Path,
    model_path: str | Path,
) -> dict[str, Any]:
    payload = build_freeze_payload(
        project_root=project_root,
        config_path=config_path,
        evidence_root=evidence_root,
        model_path=model_path,
    )
    return {
        "freeze_version": 1,
        "freeze_id": canonical_sha256(payload),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "payload": payload,
    }


def write_freeze_manifest(path: str | Path, manifest: dict[str, Any]) -> Path:
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return target


def verify_freeze_manifest(
    path: str | Path,
    *,
    runtime_config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    manifest = json.loads(source.read_text(encoding="utf-8"))
    payload = manifest["payload"]
    current = build_freeze_payload(
        project_root=payload["project_root"],
        config_path=payload["config_path"],
        evidence_root=payload["evidence_root"],
        model_path=payload["model_path"],
    )
    current_id = canonical_sha256(current)
    expected_id = str(manifest["freeze_id"])
    runtime_config_match = (
        runtime_config is None
        or canonical_sha256(dict(runtime_config))
        == canonical_sha256(payload["resolved_config"])
    )
    return {
        "ok": current_id == expected_id and runtime_config_match,
        "expected_freeze_id": expected_id,
        "current_freeze_id": current_id,
        "runtime_config_match": runtime_config_match,
        "manifest_path": str(source),
    }
