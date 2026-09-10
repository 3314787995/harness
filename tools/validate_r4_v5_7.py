"""Only R4 5.7 new contracts and directly affected regressions; no GPU calls."""
import json
from pathlib import Path
import subprocess
import sys

TESTS = [
    "tests/test_r4_v5_7.py",
    "tests/test_r4_v5.py::test_category_aliases_and_excluded_shapes",
    "tests/test_r4_v5.py::test_missing_candidate_empty_positive_refs_are_valid",
    "tests/test_r4_v5.py::test_refs_integer_coordinates_crop_and_context",
    "tests/test_r4_v5.py::test_identity_can_be_revoked_and_box_change_is_not_proof",
    "tests/test_r4_v5.py::test_qualified_local_objects_complete_without_identity",
    "tests/test_r4_v5_composition.py::test_literal_case_and_attribute_combination_have_different_keys",
    "tests/test_r4_v5_composition.py::test_history_plan_completion_through_text_adapter",
    "tests/test_r1345_debug_runner.py::test_request_drops_gold_and_review",
    "tests/test_r1345_debug_runner.py::test_resume_skips_completed_items_and_loads_once",
    "tests/test_r1345_debug_runner.py::test_fatal_call_is_journaled_and_stops_batch",
    "tests/test_r1345_debug_runner.py::test_r4_failure_is_saved_and_resume_skips_before_model_loading",
    "tests/test_r1345_debug_runner.py::test_r4_wrong_and_unknown_predictions_continue_without_inflating_score"
]


def main():
    root = Path(__file__).resolve().parents[1]
    out = root / ".codex_artifacts/r4-v5_7"
    out.mkdir(parents=True, exist_ok=True)
    selected = TESTS[1:] if "--regression-only" in sys.argv else TESTS
    name = "regression" if "--regression-only" in sys.argv else "targeted"
    cmd = [sys.executable, "-X", "utf8", "-m", "pytest", *selected, "-q", "--tb=short", "--junitxml=" + str(out/(name+".xml"))]
    result = subprocess.run(cmd, cwd=root, capture_output=True, text=True, encoding="utf-8")
    print(result.stdout, end="")
    print(result.stderr, end="", file=sys.stderr)
    (out/(name+".log")).write_text(result.stdout+result.stderr, encoding="utf-8")
    (out/(name+".json")).write_text(json.dumps({"command":cmd,"exit_code":result.returncode,"gpu_inference":False},indent=2),encoding="utf-8")
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
