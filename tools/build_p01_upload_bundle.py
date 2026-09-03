from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_IMPORT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_IMPORT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_IMPORT_ROOT))

from qwen3vl_agent.p01.smoke import BENCHMARKS, load_smoke_questions

BUNDLE_SCHEMA_VERSION = 2
EXCLUDED_DIRECTORY_NAMES = {
    ".codex_artifacts",
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    ".mypy_cache",
    ".cache",
    "runs",
    "tmp",
    "build",
    "dist",
    "htmlcov",
}
EXCLUDED_FILE_NAMES = {
    "local.yaml",
    ".coverage",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(project_root: Path, *args: str) -> str | None:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip()


def git_snapshot(project_root: Path) -> dict[str, Any]:
    status = _git(project_root, "status", "--short")
    return {
        "commit": _git(project_root, "rev-parse", "HEAD"),
        "branch": _git(project_root, "rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(status),
        "status": status.splitlines() if status else [],
    }


def _ignore_project_files(directory: str, names: list[str]) -> set[str]:
    current = Path(directory)
    ignored: set[str] = set()
    for name in names:
        should_ignore = (
            name in EXCLUDED_DIRECTORY_NAMES
            or name in EXCLUDED_FILE_NAMES
            or name.endswith((".pyc", ".pyo"))
            or name == ".env"
            or name.startswith(".env.")
            or (current.name == "configs" and name == "local.yaml")
        )
        if should_ignore:
            ignored.add(name)
    return ignored


def _load_inventory(data_root: Path, benchmark: str, subset: str) -> dict[str, dict[str, Any]]:
    path = data_root / benchmark / "subsets" / subset / "media_inventory.jsonl"
    inventory: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            video_path = str(record.get("video_path") or "")
            if not video_path:
                raise ValueError(f"{path}:{line_number}: video_path is missing")
            if record.get("status") != "available":
                raise ValueError(f"{path}:{line_number}: media is not available")
            inventory[video_path] = record
    return inventory


def _copy_data(
    source_root: Path,
    target_root: Path,
    *,
    subset: str,
) -> dict[str, Any]:
    questions = load_smoke_questions(source_root, benchmarks=BENCHMARKS, subset=subset)
    by_benchmark: dict[str, dict[str, Any]] = {}
    for benchmark in BENCHMARKS:
        source_subset = source_root / benchmark / "subsets" / subset
        target_subset = target_root / benchmark / "subsets" / subset
        shutil.copytree(source_subset, target_subset)
        inventory = _load_inventory(source_root, benchmark, subset)
        selected = [item for item in questions if item.benchmark == benchmark]
        copied_paths: set[str] = set()
        for question in selected:
            relative = question.relative_video_path
            if relative in copied_paths:
                continue
            copied_paths.add(relative)
            record = inventory.get(relative)
            if record is None:
                raise ValueError(f"{benchmark}: inventory is missing {relative}")
            source = source_root / benchmark / relative
            destination = target_root / benchmark / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            if destination.stat().st_size != int(record["size_bytes"]):
                raise ValueError(f"copied media size mismatch: {destination}")
            expected_hash = str(record.get("sha256") or "").lower()
            if expected_hash and sha256_file(destination) != expected_hash:
                raise ValueError(f"copied media hash mismatch: {destination}")
        by_benchmark[benchmark] = {
            "question_count": len(selected),
            "video_count": len(copied_paths),
        }
    return {
        "question_count": len(questions),
        "video_count": len({(item.benchmark, item.relative_video_path) for item in questions}),
        "benchmarks": by_benchmark,
    }


def _payload_entries(root: Path) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix()
        if relative in {"SHA256SUMS", "bundle_manifest.json"}:
            continue
        category = "data" if relative.startswith("data/") else "code"
        if relative == "RUNBOOK.md":
            category = "documentation"
        entries.append(
            {
                "path": relative,
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "category": category,
            }
        )
    return entries


def _write_checksums(root: Path) -> None:
    paths = sorted(path for path in root.rglob("*") if path.is_file() and path.name != "SHA256SUMS")
    lines = [f"{sha256_file(path)}  {path.relative_to(root).as_posix()}" for path in paths]
    (root / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_bundle(
    *,
    project_root: str | Path,
    data_root: str | Path,
    output_dir: str | Path,
    subset: str = "p01-smoke-v1",
    runbook_path: str | Path | None = None,
) -> dict[str, Any]:
    project = Path(project_root).expanduser().resolve()
    data = Path(data_root).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    if not (project / "pyproject.toml").is_file():
        raise FileNotFoundError(f"project root is invalid: {project}")
    if output.exists():
        raise FileExistsError(f"output directory already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent)).resolve()
    try:
        code_target = staging / "code"
        shutil.copytree(
            project,
            code_target,
            ignore=_ignore_project_files,
            dirs_exist_ok=False,
        )
        data_summary = _copy_data(data, staging / "data", subset=subset)
        runbook = (
            Path(runbook_path).expanduser().resolve()
            if runbook_path is not None
            else project / "docs" / "p01_rental_runbook.md"
        )
        if not runbook.is_file():
            raise FileNotFoundError(f"runbook does not exist: {runbook}")
        shutil.copy2(runbook, staging / "RUNBOOK.md")
        snapshot = git_snapshot(project)
        marker = {
            "schema_version": BUNDLE_SCHEMA_VERSION,
            "bundle_type": "p01-smoke-v2",
            "built_at": datetime.now(timezone.utc).isoformat(),
        }
        (staging / ".p01-upload-bundle.json").write_text(
            json.dumps(marker, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        entries = _payload_entries(staging)
        manifest = {
            **marker,
            "project_snapshot": snapshot,
            "data_summary": data_summary,
            "payload_file_count": len(entries),
            "payload_size_bytes": sum(item["size_bytes"] for item in entries),
            "files": entries,
        }
        (staging / "bundle_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        _write_checksums(staging)
        os.replace(staging, output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return manifest


def create_tar_archive(bundle_dir: str | Path, archive_path: str | Path) -> dict[str, Any]:
    bundle = Path(bundle_dir).expanduser().resolve()
    archive = Path(archive_path).expanduser().resolve()
    if not (bundle / ".p01-upload-bundle.json").is_file():
        raise ValueError(f"not a P01 upload bundle: {bundle}")
    if archive.exists():
        raise FileExistsError(f"archive already exists: {archive}")
    archive.parent.mkdir(parents=True, exist_ok=True)
    temporary = archive.with_suffix(archive.suffix + ".tmp")
    try:
        with tarfile.open(temporary, mode="w", format=tarfile.PAX_FORMAT) as tar:
            tar.add(bundle, arcname=bundle.name, recursive=True)
        os.replace(temporary, archive)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    digest = sha256_file(archive)
    checksum_path = archive.with_suffix(archive.suffix + ".sha256")
    checksum_path.write_text(f"{digest}  {archive.name}\n", encoding="utf-8")
    return {
        "path": str(archive),
        "size_bytes": archive.stat().st_size,
        "sha256": digest,
        "checksum_path": str(checksum_path),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a portable P01 smoke upload bundle from the current worktree."
    )
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--subset", default="p01-smoke-v1")
    parser.add_argument("--runbook")
    parser.add_argument("--archive")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = build_bundle(
        project_root=args.project_root,
        data_root=args.data_root,
        output_dir=args.output_dir,
        subset=args.subset,
        runbook_path=args.runbook,
    )
    result: dict[str, Any] = {
        "bundle_dir": str(Path(args.output_dir).expanduser().resolve()),
        "data_summary": manifest["data_summary"],
        "payload_file_count": manifest["payload_file_count"],
        "payload_size_bytes": manifest["payload_size_bytes"],
        "project_snapshot": manifest["project_snapshot"],
    }
    if args.archive:
        result["archive"] = create_tar_archive(args.output_dir, args.archive)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
