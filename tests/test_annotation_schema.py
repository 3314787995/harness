from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator, ValidationError

ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = ROOT / "schemas" / "videomme_evidence_annotation.schema.json"
TEMPLATE_PATH = ROOT / "schemas" / "videomme_evidence_annotation.template.json"


def _load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_annotation_schema_and_draft_template_are_valid() -> None:
    schema = _load(SCHEMA_PATH)
    template = _load(TEMPLATE_PATH)

    Draft202012Validator.check_schema(schema)
    Draft202012Validator(
        schema,
        format_checker=Draft202012Validator.FORMAT_CHECKER,
    ).validate(template)


def test_locked_record_cannot_reuse_incomplete_draft_template() -> None:
    schema = _load(SCHEMA_PATH)
    record = copy.deepcopy(_load(TEMPLATE_PATH))
    record["record_status"] = "locked"
    record["split"] = "locked"

    with pytest.raises(ValidationError):
        Draft202012Validator(
            schema,
            format_checker=Draft202012Validator.FORMAT_CHECKER,
        ).validate(record)


def test_v02_single_annotator_locked_record_is_valid() -> None:
    schema = _load(SCHEMA_PATH)
    record = copy.deepcopy(_load(TEMPLATE_PATH))
    record["record_status"] = "locked"
    record["split"] = "dev"
    record["source"]["video_id"] = "test-video"
    record["source"]["question_id"] = "test-video-1"
    record["source"]["video_sha256"] = "0" * 64
    record["question"]["text"] = "A test question"
    record["question"]["options"] = [
        {"option_id": f"O{i}", "benchmark_label": label, "text": f"Option {i}"}
        for i, label in enumerate("ABCD", 1)
    ]
    record["question"]["official_answer"] = {"option_id": "O1", "benchmark_label": "A"}
    record["validity"] = {"status": "valid", "reason": "test"}
    record["evidence_contract"] = {
        "primary_topology": "local",
        "secondary_topologies": [],
        "required_modalities": ["visual"],
        "answer_criterion": "direct_support",
        "evidence_slots": [{"slot_id": "S1", "description": "test", "required": True}],
        "sufficient_evidence_sets": [{
            "set_id": "ES1",
            "description": "test",
            "logic": "all_required",
            "items": [{
                "evidence_id": "E1",
                "slot_id": "S1",
                "required": True,
                "core_interval": {"start_sec": 1.0, "end_sec": 2.0},
                "context_interval": {"start_sec": 0.0, "end_sec": 3.0},
                "modality": "visual",
                "observation_requirement": "point_frame",
                "minimum_observation": {"mode": "inspect", "min_frames": 1, "min_temporal_span_sec": 0},
                "atomic_fact": "A visible test fact.",
                "roles": ["support"],
                "supports_option_ids": ["O1"],
                "refutes_option_ids": ["O2", "O3", "O4"],
                "source_references": {"frame_timestamps_sec": [1.5], "subtitle_cue_ids": [], "ocr_text": None},
            }],
            "relations": [],
        }],
        "global_coverage_contract": None,
    }
    record["review"] = {
        "mode": "single_annotator",
        "status": "locked",
        "annotator_a": {"reviewer_id": "primary-annotator", "completed_at": "2026-08-22T00:00:00Z"},
        "annotator_b": None,
        "adjudicator": None,
        "agreement": {"validity": None, "topology": None, "modalities": None, "sufficient_set": None, "notes": "single annotator"},
        "locked_at": "2026-08-22T00:00:00Z",
        "content_sha256": "1" * 64,
        "notes": "v0.2 single-annotator lock",
    }
    Draft202012Validator(
        schema,
        format_checker=Draft202012Validator.FORMAT_CHECKER,
    ).validate(record)
