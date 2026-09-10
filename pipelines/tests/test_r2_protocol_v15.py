"""1.5: replay real 1.4 blockers and guarantee bounded terminal preparation."""

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_r2 import FakeQwen, frame_map
from test_r2 import config as config  # noqa: PLC0414
from test_r2 import video as video  # noqa: PLC0414
from test_r2_protocol_v13 import valid_final

from qwen3vl_agent.r2 import InputContract, R2Budget, R2Request, R2VideoAgent
from qwen3vl_agent.r2.context import fit_final_payload, summarize_gaps
from qwen3vl_agent.r2.contracts import (
    certify_final,
    parse,
    repair_context,
    stage_schema,
    validate_observation,
)
from qwen3vl_agent.r2.observation import ObservationSpec, normalize_identity_handoff
from qwen3vl_agent.r2.prompts import prompt
from qwen3vl_agent.r2.runtime import ModelSession
from qwen3vl_agent.r2.state import StateStore
from qwen3vl_agent.r2.types import ProtocolError

REAL = json.loads(
    (Path(__file__).parent / "fixtures/r2_v14_gpu_regressions.json").read_text(encoding="utf-8")
)


def test_real_optional_null_bbox_responses_are_accepted_without_crop_invention():
    for call in REAL["calls"]:
        payload = call["payload"]
        raw = parse(call["raw"], "observe", stage_schema("observe", payload))
        assert raw["gaps"][0]["bbox"] is None
        original = copy.deepcopy(raw)
        spec = ObservationSpec.from_query(REAL["query"])
        frames = frame_map([f["source_seconds"] for f in payload["frames"]])
        normalized, changes = spec.normalize(raw, payload["core"], frames)
        requirements = payload.get("identity_requirements", [])
        normalized = normalize_identity_handoff(normalized, requirements)
        validate_observation(
            normalized, REAL["query"], frames, {r["from_node"] for r in requirements}, spec=spec
        )
        assert "gaps:unavailable_bbox" in changes
        assert all("bbox" not in g for g in normalized["gaps"])
        assert raw == original


@pytest.mark.parametrize("bbox", [[1, 2], [-1, 0, 2, 3], "unknown"])
def test_malformed_non_null_bbox_stays_invalid(bbox):
    payload = REAL["calls"][0]["payload"]
    value = json.loads(REAL["calls"][0]["raw"])
    value["gaps"][0]["bbox"] = bbox
    with pytest.raises(ProtocolError):
        parse(json.dumps(value), "observe", stage_schema("observe", payload))


def test_final_null_bbox_is_normalized_without_losing_prediction_or_recheck():
    _, payload, value = valid_final()
    value["recheck"] = {"kind": "detail", "description": "blurred", "bbox": None}
    parsed = parse(json.dumps(value), "final", stage_schema("final", payload))
    normalized = certify_final(parsed, payload)
    assert normalized["prediction"] == value["prediction"]
    assert normalized["recheck"] == {"kind": "detail", "description": "blurred"}
    assert parsed["recheck"]["bbox"] is None


def test_gap_summary_is_bounded_and_does_not_merge_disjoint_spans_or_resolve_gaps():
    gaps = [
        {"kind": "detail", "slot_id": f"S{i % 30}", "description": "x" * 5000, "span": [i, i]}
        for i in range(1000)
    ]
    original = copy.deepcopy(gaps)
    summaries, diagnostic = summarize_gaps(gaps)
    assert len(summaries) == 16 and diagnostic["groups"] == 30
    assert diagnostic["records"] == 1000
    assert all("span" not in s and len(s["example_spans"]) <= 3 for s in summaries)
    assert all(len(t) <= 400 for s in summaries for t in s["examples"])
    assert gaps == original


def test_real_rotation_state_reaches_terminal_model_and_deduplicates_raw_windows(config):
    # Replays inference state and source-time metadata, not video inference or gold answers.
    model = FakeQwen(final_hook=lambda v, p: {**v, "prediction": p["options"][-1]["label"]})
    agent = R2VideoAgent(model, config=config)
    catalog = copy.deepcopy(REAL["media_catalog"])

    class ReplayMedia:
        def __init__(self):
            self.catalog = catalog

        def frame(self, fid):
            return SimpleNamespace(id=fid, timestamp_seconds=catalog[fid]["timestamp_seconds"])

        def prepare(self, batch):
            return SimpleNamespace(
                frames=batch.frames,
                pixels=0,
                parts=[],
                kind="metadata_replay",
                video_frame_metadata=None,
            )

    agent.media = ReplayMedia()
    state = {
        "coverage": copy.deepcopy(REAL["coverage"]),
        "notes": [REAL["failure"]],
        "final_round": 0,
    }
    budget = R2Budget()
    session = ModelSession(model, config, budget, state, lambda: None)
    request = R2Request("metadata-replay", REAL["question"], choices=REAL["options"])
    contract = InputContract.resolve(request, 5.032682)
    derived = copy.deepcopy(REAL["derived"])
    result = agent._finish(
        request, REAL["query"], derived, contract, state, StateStore(REAL["store"]), session
    )
    assert result["prediction"] == request.choices[-1].label
    assert all(a["status"] == "unknown" for a in result["assessments"])
    assert session.summary()["terminal_calls"] == 1
    assert not state.get("preflight_failures")
    payload = model.payloads[-1][1]
    assert payload["observations"] and payload["raw_frames"]
    assert len(payload["raw_windows"]) <= len(payload["raw_frames"]) <= 48
    assert len({json.dumps(w, sort_keys=True) for w in payload["raw_windows"]}) == len(
        payload["raw_windows"]
    )
    assert all(
        f["source_seconds"] == catalog[f["view_frame_id"]]["timestamp_seconds"]
        for f in payload["raw_frames"]
    )
    assert derived == REAL["derived"]
    assert state["final_context"][-1]["final_chars"] <= 100000
    assert session.summary()["frame_exposures"] == len(state["calls"][0]["frame_ids"])


def fallback_payload():
    _, payload, _ = valid_final()
    payload["unresolved"] = ["long internal diagnostic " * 15000]
    payload["raw_frames"] = [{"frame_id": "F01", "source_seconds": 1.25}]
    payload["raw_windows"] = [{"window_id": "w", "frame_ids": ["f1"]}]
    return payload


def test_best_effort_preserves_question_options_media_and_some_observations():
    payload = fallback_payload()
    original = copy.deepcopy(payload)
    fitted, diagnostic = fit_final_payload(payload, 30000)
    assert diagnostic["mode"] == "best_effort"
    assert fitted["observations"]
    assert fitted["derived"]["sufficient"] is False
    assert fitted["assessment_policy"]["complete_support_allowed"] is False
    assert fitted["unresolved"] and fitted["context_omissions"]["best_effort"]
    for key in ("question", "options", "raw_frames", "raw_windows"):
        assert fitted[key] == original[key]
    assert payload == original
    assert len(prompt("final", fitted)) <= diagnostic["target_chars"]


def test_best_effort_format_repair_is_bounded_and_resumable(config):
    payload, diagnostic = fit_final_payload(fallback_payload(), 60000)
    count = 0

    def final_hook(value, _):
        nonlocal count
        count += 1
        value["prediction"] = "bad" if count == 1 else "Y"
        return value

    model = FakeQwen(final_hook=final_hook)
    state = {}
    session = ModelSession(
        model, config, R2Budget(max_text_chars_per_call=60000), state, lambda: None
    )
    result, _ = session.call(
        "final:0", "final", payload, validator=lambda v: certify_final(v, payload)
    )
    assert result["prediction"] == "Y" and session.summary()["terminal_calls"] == 2
    assert all(c["text_chars"] <= 60000 for c in state["calls"])
    assert (
        session.call("final:0", "final", payload, validator=lambda v: certify_final(v, payload))[0]
        == result
    )
    assert count == 2 and diagnostic["mode"] == "best_effort"
    assert (
        len(
            prompt(
                "final",
                payload,
                {
                    "error": "bad label",
                    "previous_output": "x" * 12000,
                    **repair_context("final", payload, "{}"),
                },
            )
        )
        <= 60000
    )


def test_duplicate_raw_metadata_does_not_evict_evidence():
    _, payload, _ = valid_final()
    payload["raw_windows"] = [{"window_id": "w", "frame_ids": ["f1"]}] * 10000
    fitted, diagnostic = fit_final_payload(payload, 60000)
    assert len(fitted["raw_windows"]) == 1
    assert fitted["observations"]
    assert diagnostic["omissions"]["duplicate_raw_windows"] == 9999


def test_controller_still_finishes_after_observer_protocol_failure(video, config):
    result = R2VideoAgent(FakeQwen(invalid_role="observe"), config=config).solve(
        R2Request(str(video), "Direction?", choices=["one", "two"])
    )
    assert result.prediction is not None and result.completion_state == "partial"
    assert result.support_level != "supported" and result.resources["terminal_calls"] == 1


@pytest.mark.parametrize("choices", [[], ["one", "two"]])
def test_best_effort_controller_keeps_uncertainty_and_media_cutoff(video, config, choices):
    class LargeDiagnosticAgent(R2VideoAgent):
        def _finish(self, request, query, derived, contract, state, store, session):
            state["notes"].append("internal diagnostic " * 20000)
            return super()._finish(request, query, derived, contract, state, store, session)

    model = FakeQwen()
    result = LargeDiagnosticAgent(model, config=config).solve(
        R2Request(str(video), "Direction?", choices=choices, observation_cutoff=1.0)
    )
    assert result.prediction is not None and result.completion_state == "partial"
    assert result.support_level != "supported"
    assert result.trace["final_context"][-1]["mode"] == "best_effort"
    assert any("terminal_best_effort:" in str(g) for g in result.unresolved_items)
    payload = next(p for role, p, _ in model.payloads if role == "final")
    assert payload["raw_frames"] and payload["observations"]
    assert all(f["source_seconds"] <= 1.0 for f in payload["raw_frames"])


def test_fixture_is_scoring_free():
    def visit(value):
        if isinstance(value, dict):
            assert (
                not {"gold", "answer", "correct_answer", "ground_truth", "accuracy"} & value.keys()
            )
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(REAL)
