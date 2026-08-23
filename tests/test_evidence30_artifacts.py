"""Collection-level acceptance checks for the v0.2 Evidence30 artifacts."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

from jsonschema import Draft202012Validator, FormatChecker

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = ROOT / "annotations" / "videomme_evidence30"
SCHEMA_PATH = ROOT / "schemas" / "videomme_evidence_annotation.schema.json"

DEV_QIDS = {
    "007-2", "212-3", "389-2", "102-1", "154-2", "251-3", "050-1",
    "314-2", "434-2", "496-1", "522-3", "604-2", "647-2", "700-1",
    "730-1", "780-2", "845-1", "884-2",
}
LOCKED_QIDS = {
    "197-3", "445-2", "504-1", "573-2", "599-2", "634-2", "673-2",
    "717-2", "754-2", "792-2", "847-2", "895-3",
}


def read_jsonl(name: str) -> list[dict]:
    path = ARTIFACT_ROOT / name
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def canonical_sha256(value: dict) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def all_records() -> list[dict]:
    return read_jsonl("dev.jsonl") + read_jsonl("locked.jsonl")


def test_v02_records_validate_against_schema() -> None:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    validator = Draft202012Validator(schema, format_checker=FormatChecker())
    records = all_records()
    errors = [error for record in records for error in validator.iter_errors(record)]
    assert not errors, "\n".join(f"{error.json_path}: {error.message}" for error in errors)


def test_v02_split_status_review_and_identity_contract() -> None:
    dev = read_jsonl("dev.jsonl")
    locked = read_jsonl("locked.jsonl")
    records = dev + locked
    qids = {record["source"]["question_id"] for record in records}
    vids = {record["source"]["video_id"] for record in records}
    assert len(dev) == 18
    assert len(locked) == 12
    assert qids == DEV_QIDS | LOCKED_QIDS
    assert {record["source"]["question_id"] for record in dev} == DEV_QIDS
    assert {record["source"]["question_id"] for record in locked} == LOCKED_QIDS
    assert len(qids) == 30 == len(vids)
    assert all(record["record_status"] == "locked" for record in records)
    assert all(record["validity"]["status"] == "valid" for record in records)
    assert all(record["review"]["mode"] == "single_annotator" for record in records)
    assert all(record["review"]["status"] == "locked" for record in records)
    assert all(record["review"]["annotator_a"]["reviewer_id"] == "primary-annotator" for record in records)
    assert all(record["review"]["annotator_b"] is None for record in records)
    assert all(record["review"]["adjudicator"] is None for record in records)
    assert all(record["schema_version"] == "videomme-evidence30/0.2.0" for record in records)


def test_v02_evidence_references_intervals_and_content_hashes() -> None:
    for record in all_records():
        options = {option["option_id"] for option in record["question"]["options"]}
        contract = record["evidence_contract"]
        evidence_ids = set()
        for evidence_set in contract["sufficient_evidence_sets"]:
            for item in evidence_set["items"]:
                evidence_ids.add(item["evidence_id"])
                core = item["core_interval"]
                context = item["context_interval"]
                assert 0 <= core["start_sec"] <= core["end_sec"]
                assert 0 <= context["start_sec"] <= context["end_sec"]
                assert context["start_sec"] <= core["start_sec"]
                assert core["end_sec"] <= context["end_sec"]
                assert set(item["supports_option_ids"]) <= options
                assert set(item["refutes_option_ids"]) <= options
                assert item["modality"] in contract["required_modalities"]
            for relation in evidence_set["relations"]:
                assert set(relation["source_evidence_ids"]) <= evidence_ids
        assert evidence_ids
        for negative in record["hard_negatives"]:
            assert 0 <= negative["interval"]["start_sec"] <= negative["interval"]["end_sec"]
            assert set(negative["tempts_option_ids"]) <= options
        expected = copy.deepcopy(record)
        expected["review"]["content_sha256"] = None
        assert record["review"]["content_sha256"] == canonical_sha256(expected)


def test_v02_manifest_and_versioned_copy_are_consistent() -> None:
    manifest = json.loads((ARTIFACT_ROOT / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["manifest_version"] == "videomme-evidence30/0.2.0"
    assert manifest["status"] == "locked_single_annotator"
    assert manifest["completion_label"] == "success"
    assert manifest["distribution"] == {"total": 30, "dev": 18, "locked": 12, "unique_video_ids": 30}
    assert len(manifest["selection"]["question_hashes"]) == 30
    assert len(manifest["selection"]["video_hashes"]) == 30
    assert all(len(value) == 64 for value in manifest["selection"]["question_hashes"].values())
    assert all(value is not None and len(value) == 64 for value in manifest["selection"]["video_hashes"].values())
    for name, digest in (("dev.jsonl", "dev"), ("locked.jsonl", "locked"), ("excluded.jsonl", "excluded")):
        assert manifest["artifact_sha256"][digest] == file_sha256(ARTIFACT_ROOT / name)
    for name in ("dev.jsonl", "locked.jsonl", "excluded.jsonl", "manifest.json"):
        assert (ARTIFACT_ROOT / "0.2.0" / name).read_bytes() == (ARTIFACT_ROOT / name).read_bytes()
    old_manifest = json.loads((ARTIFACT_ROOT / "0.1.0" / "manifest.json").read_text(encoding="utf-8"))
    assert old_manifest["manifest_version"] == "videomme-evidence30/0.1.0"
    assert len(read_jsonl("0.1.0/dev.jsonl")) == 3
