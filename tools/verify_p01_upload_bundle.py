from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_bundle(bundle_root: str | Path) -> dict[str, Any]:
    root = Path(bundle_root).expanduser().resolve()
    marker = root / ".p01-upload-bundle.json"
    checksums = root / "SHA256SUMS"
    manifest_path = root / "bundle_manifest.json"
    errors: list[str] = []
    if not marker.is_file():
        errors.append("missing .p01-upload-bundle.json")
    if not checksums.is_file():
        errors.append("missing SHA256SUMS")
    if not manifest_path.is_file():
        errors.append("missing bundle_manifest.json")
    if marker.is_file():
        try:
            marker_payload = json.loads(marker.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            errors.append(f".p01-upload-bundle.json is invalid: {exc}")
        else:
            if marker_payload.get("bundle_type") != "p01-smoke-v2":
                errors.append("upload bundle is not a P01 v2 bundle")
    checked = 0
    if checksums.is_file():
        for line_number, line in enumerate(
            checksums.read_text(encoding="utf-8").splitlines(),
            start=1,
        ):
            if not line.strip():
                continue
            try:
                expected, relative = line.split("  ", maxsplit=1)
            except ValueError:
                errors.append(f"SHA256SUMS:{line_number}: malformed line")
                continue
            candidate = Path(relative)
            if candidate.is_absolute() or ".." in candidate.parts:
                errors.append(f"SHA256SUMS:{line_number}: unsafe path {relative}")
                continue
            target = (root / candidate).resolve()
            if not target.is_relative_to(root):
                errors.append(f"SHA256SUMS:{line_number}: path escapes bundle")
                continue
            if not target.is_file():
                errors.append(f"missing file: {relative}")
                continue
            actual = sha256_file(target)
            checked += 1
            if actual.lower() != expected.lower():
                errors.append(f"hash mismatch: {relative}")
    manifest: dict[str, Any] = {}
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            errors.append(f"bundle_manifest.json is invalid: {exc}")
        else:
            summary = manifest.get("data_summary") or {}
            if summary.get("question_count") != 25:
                errors.append("bundle manifest does not contain 25 questions")
            if summary.get("video_count") != 20:
                errors.append("bundle manifest does not contain 20 videos")
    return {
        "schema_version": 2,
        "bundle_root": str(root),
        "checked_file_count": checked,
        "data_summary": manifest.get("data_summary"),
        "errors": errors,
        "ready": not errors,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Verify a P01 upload bundle.")
    parser.add_argument("--bundle-root", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = verify_bundle(args.bundle_root)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
