"""Only V3.1 failure replay, binding routing, terminal recovery and version isolation."""

import copy
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_r1 import query
from test_r1_v2 import real_video as real_video  # noqa: PLC0414
from test_r1_v3 import (
    FakeModel,
    append_unit,
    build,
    final,
    observation,
    request,
    state_with_candidates,
)

from qwen3vl_agent.coarse_to_fine.types import FrameRef
from qwen3vl_agent.p01.types import TimeSpan
from qwen3vl_agent.r1.control import audit_bundle, json_object
from qwen3vl_agent.r1.media import MediaBatch
from qwen3vl_agent.r1.types import EvidenceBundle, QueryField, QuerySpec, R1Request
from qwen3vl_agent.r1_v3.evidence import modality_state, refresh_binding_tasks
from qwen3vl_agent.r1_v3.prompts import prompt, repair_prompt
from qwen3vl_agent.r1_v3.terminal import audit_terminal
from qwen3vl_agent.r1_v3.types import V3Packet
from qwen3vl_agent.r1_v3.version import POLICY_ID

SAVED = json.loads(
    (Path(__file__).parent / "fixtures/r1_v3_1_failures.json").read_text(encoding="utf-8")
)


@pytest.mark.parametrize("saved", SAVED["cases"], ids=lambda x: x["id"])
def test_saved_failures_record_binding_modality_and_terminal_citations(saved):
    p = saved["observe"]["payload"]
    spec = QuerySpec(
        **{**p["query"], "fields": tuple(QueryField(**f) for f in p["query"]["fields"])}
    )
    batch = MediaBatch(TimeSpan(**p["batch_span"]), tuple(FrameRef(**f) for f in p["frames"]))
    packet = V3Packet("replay", "candidate", "answer", batch.span)
    raw = json_object(saved["observe"]["raw"])
    append_unit(packet, batch, spec, raw)
    tasks = refresh_binding_tasks(packet, spec)
    bundle = EvidenceBundle("replay", [packet], required_spans=[batch.span])
    state = SimpleNamespace(
        request=R1Request("unused.mp4", p["question"]), query=spec, bundles=[bundle]
    )
    modalities = modality_state(state, bundle)
    if saved["id"].endswith(("007-1", "073-1")):
        assert tasks and tasks[0].field_ids == ("Q1",)
        assert all(not f.supports_query_fields for f in packet.observations[0].facts)
        assert all(not f.answer_eligible for f in packet.fact_eligibility)
        assert any(f.observation_clear and f.target_confirmed for f in packet.fact_eligibility)
        assert not audit_bundle(bundle, spec).sufficient
    else:
        assert not tasks and audit_bundle(bundle, spec).sufficient
    if saved["id"].endswith("073-1"):
        assert modalities["screen_text"]["allowed"] and modalities["screen_text"]["input_present"]
        assert modalities["screen_text"]["reading_status"] == "readable"
        assert not modalities["screen_text"]["answer_fact_ids"]
    terminal = saved["final"]
    data = json_object(terminal["raw"])
    original = copy.deepcopy(data)
    audit, _ = audit_terminal(terminal["payload"], data, raw=terminal["raw"])
    assert any(x.startswith("terminal_references_empty:choice:") for x in audit.failures)
    assert not audit.passed and data == original  # No automatic field/answer correction.


def unassociated(p):
    result = observation(p)
    for fact in result["facts"]:
        fact["supports_query_fields"] = []
    return result


def test_binding_recovery_replays_sources_and_creates_new_fact_ids(real_video, tmp_path):
    seen = []

    def handler(p):
        seen.append(p)
        return unassociated(p) if p["purpose"] == "initial" else observation(p)

    agent, model = build(tmp_path, observe=handler)
    result = agent.solve(request(real_video, query_scope=TimeSpan(2, 8)))
    assert result.support_level == "supported", result.unresolved_reasons
    assert [p["purpose"] for p in seen] == ["initial", "query_binding"]
    review = seen[1]
    assert review["question"] == seen[0]["question"]
    assert review["missing_fields"][0]["field_id"] == "Q1"
    assert {f["id"] for f in review["frames"]} == set(
        review["query_binding_task"]["source_frame_ids"]
    )
    assert {f["id"] for f in review["frames"]} <= {f["id"] for f in seen[0]["frames"]}
    records = result.trace["observation_records"]
    assert records[0]["facts"][0]["supports_query_fields"] == ()
    assert records[1]["facts"][0]["supports_query_fields"] == ("Q1",)
    assert records[0]["facts"][0]["fact_id"] != records[1]["facts"][0]["fact_id"]
    assert records[1]["coverage"]["coverage_kind"] == "detail"
    assert result.trace["evidence_state"][0]["query_binding_tasks"][0]["status"] == "resolved"
    assert result.resources["refinements"] == 1
    assert len([c for c in model.calls if c["role"] == "final"]) == 1


def test_repeated_unassociated_background_stops_without_dense_or_auto_q1(real_video, tmp_path):
    def background(p):
        out = unassociated(p)
        out["facts"][0].update(
            statement="A background sign says STORE.",
            structured_value="STORE",
            subject_or_local_entity="background_sign",
            source_kind="screen_text",
        )
        return out

    agent, model = build(
        tmp_path,
        observe=background,
        query=query(required_modalities=["screen_text"], observation_modes=["ocr"]),
    )
    # One real OCR batch: its recheck must not manufacture another binding task.
    result = agent.solve(request(real_video, query_scope=TimeSpan(2, 4)))
    assert [c["payload"]["purpose"] for c in model.calls if c["role"] == "observe"] == [
        "initial",
        "query_binding",
    ]
    assert result.resources["refinements"] == 1
    assert result.trace["controller_stop_reason"] == "no_progress"
    assert result.support_level == "partial" and result.unresolved_reasons
    assert result.completion_state != "modality_unavailable"
    assert not any("required_modality_unavailable" in x for x in result.unresolved_reasons)
    tasks = result.trace["evidence_state"][0]["query_binding_tasks"]
    assert len(tasks) == 1 and tasks[0]["attempted"] and tasks[0]["status"] == "unresolved"
    assert all(
        not f["answer_eligible"] for f in result.trace["evidence_state"][0]["fact_eligibility"]
    )
    assert result.trace["terminal_call_count"] == 1


@pytest.mark.parametrize("kind", ["detail", "temporal_context"])
def test_binding_review_routes_only_explicit_visual_gap_with_shared_budget(
    real_video, tmp_path, kind
):
    def handler(p):
        if p["purpose"] == "initial":
            return unassociated(p)
        out = observation(p)
        if p["purpose"] == "query_binding":
            out["facts"] = []
            out["gaps"] = [
                {
                    "kind": kind,
                    "reason": "Need clearer surface or preceding action.",
                    "field_ids": ["Q1"],
                    **({"direction": "before"} if kind == "temporal_context" else {}),
                }
            ]
        return out

    agent, model = build(tmp_path, observe=handler)
    result = agent.solve(
        request(real_video, allowed_scope=TimeSpan(2, 8), query_scope=TimeSpan(2, 8))
    )
    seen = [c["payload"] for c in model.calls if c["role"] == "observe"]
    assert [p["purpose"] for p in seen] == ["initial", "query_binding", kind]
    assert result.resources["refinements"] == 2
    assert all(2 <= f["timestamp_seconds"] <= 8 for p in seen for f in p["frames"])
    assert result.support_level == "supported", result.unresolved_reasons


def test_binding_failed_call_is_charged_and_not_reissued(real_video, tmp_path):
    agent, _ = build(tmp_path, observe=lambda p: unassociated(p))
    state = state_with_candidates(agent, real_video, spans=((2, 8),))
    packet = state.bundles[0].packets[0]
    agent.model.handlers["observe"] = RuntimeError("synthetic read failure")
    assert agent._query_binding_recheck(state, packet)
    assert state.context.refinements == 1
    assert not agent._query_binding_recheck(state, packet)
    assert state.context.refinements == 1
    assert len(packet.query_binding_tasks) == 1


@pytest.mark.parametrize(
    "available,status,expected",
    [
        (("video", "screen_text"), "unreadable", False),
        (("video",), "clear", True),
    ],
)
def test_modality_permission_is_distinct_from_readability(
    real_video, tmp_path, available, status, expected
):
    def handler(p):
        out = observation(p, status=status)
        out["facts"][0]["source_kind"] = "screen_text"
        return out

    agent, _ = build(tmp_path, observe=handler)
    state = state_with_candidates(agent, real_video, spans=((2, 8),))
    state.request = replace(state.request, available_modalities=available)
    state.query = replace(state.query, required_modalities=("screen_text",))
    assert bool(agent._missing_modalities(state, state.bundles[0])) == expected
    state.bundles = []
    state.context.shown_frame_ids.clear()
    assert agent._missing_modalities(state, EvidenceBundle("missing")) == [
        "required_modality_unavailable:screen_text"
    ]


def test_retired_candidate_reading_is_input_history_not_answer_evidence(real_video, tmp_path):
    def handler(p):
        out = unassociated(p)
        out["target"].update(status="mismatched", description="A different object")
        out["facts"][0]["source_kind"] = "screen_text"
        return out

    agent, _ = build(tmp_path, observe=handler)
    state = state_with_candidates(agent, real_video, spans=((2, 8),))
    state.query = replace(state.query, required_modalities=("screen_text",))
    result = agent._finish(state)
    assert result.completion_state == "evidence_unresolved"
    assert result.trace["modality_state"]["screen_text"]["reading_status"] == "readable"
    assert result.trace["modality_state"]["screen_text"]["answer_fact_ids"] == []
    assert not any("required_modality_unavailable" in x for x in result.unresolved_reasons)


@pytest.mark.parametrize("kind,expected", [("visual", "partial"), ("screen_text", "supported")])
def test_available_modality_still_requires_eligible_reading_for_supported(
    real_video, tmp_path, kind, expected
):
    def handler(p):
        out = observation(p)
        out["facts"][0]["source_kind"] = kind
        return out

    agent, model = build(
        tmp_path,
        observe=handler,
        query=query(required_modalities=["screen_text"], observation_modes=["ocr"]),
    )
    result = agent.solve(request(real_video, query_scope=TimeSpan(2, 4)))
    assert result.support_level == expected, result.unresolved_reasons
    assert result.trace["terminal_call_count"] == 1
    assert not any(c["role"] == "terminal_review" for c in model.calls)
    if kind == "visual":
        assert result.completion_state == "evidence_unresolved"
        assert "required_modality_evidence_unresolved:screen_text" in result.unresolved_reasons
        assert not result.trace["terminal_evidence_audit"]["sufficient"]


def broken_final(payload):
    result = final(payload)
    for item in result["choice_assessments"]:
        if item["status"] == "rejected":
            item["fact_ids"] = []
    return result


def test_terminal_recovers_empty_exclusion_refs_once_on_same_frozen_media(real_video, tmp_path):
    agent, model = build(tmp_path, final=broken_final)
    result = agent.solve(request(real_video, query_scope=TimeSpan(2, 8)))
    calls = [c for c in model.calls if c["role"] in {"final", "terminal_review"}]
    assert [c["role"] for c in calls] == ["final", "terminal_review"]
    assert result.support_level == "supported", result.unresolved_reasons
    for key in ("question", "choices", "frozen_bundle", "frames", "evidence_audit"):
        assert calls[0]["payload"][key] == calls[1]["payload"][key]
    assert calls[0]["content"][:-1] == calls[1]["content"][:-1]
    assert result.resources["refinements"] == 0
    audits = result.trace["terminal_audits"]
    assert "terminal_references_empty:choice:B" in audits[0]["failures"]
    assert audits[1]["passed"] and result.trace["terminal_call_count"] == 2
    assert not any(c["role"] == "repair" for c in model.calls)


@pytest.mark.parametrize(
    "fault",
    [
        "json",
        "unknown_ref",
        "missing_option",
        "missing_basis",
        "contradiction",
        "new_claim",
        "claim_type",
    ],
)
def test_terminal_identifiable_output_failure_recovers(real_video, tmp_path, fault):
    def malformed(p):
        out = final(p)
        if fault == "json":
            return "{broken"
        if fault == "unknown_ref":
            out["choice_assessments"][1]["fact_ids"] = ["NEW_FACT"]
        if fault == "missing_option":
            out["choice_assessments"].pop()
        if fault == "missing_basis":
            out["choice_assessments"][1].pop("basis")
        if fault == "contradiction":
            out["answer_supported"] = False
        if fault == "new_claim":
            out["claims"][0]["statement"] = "An unseen event occurs."
        if fault == "claim_type":
            out["claims"][0]["statement"] = {"invalid": "object"}
        return out

    agent, model = build(tmp_path, final=malformed)
    result = agent.solve(request(real_video, query_scope=TimeSpan(2, 8)))
    assert result.support_level == "supported", result.unresolved_reasons
    assert result.trace["terminal_call_count"] == 2
    assert not any(c["role"] == "repair" for c in model.calls)


def test_terminal_first_pass_and_evidence_insufficiency_do_not_retry(real_video, tmp_path):
    agent, model = build(tmp_path)
    result = agent.solve(request(real_video, query_scope=TimeSpan(2, 8)))
    assert result.support_level == "supported" and result.trace["terminal_call_count"] == 1
    model.handlers["observe"] = unassociated
    result = agent.solve(request(real_video, query_scope=TimeSpan(2, 8)))
    assert result.support_level == "partial" and result.trace["terminal_call_count"] == 1
    assert "terminal_frozen_evidence_insufficient" in result.unresolved_reasons


@pytest.mark.parametrize(
    "responses",
    [
        [RuntimeError("CUDA out of memory"), broken_final],
        [broken_final, RuntimeError("CUDA out of memory")],
        [RuntimeError("CUDA out of memory"), RuntimeError("CUDA out of memory")],
        ["{broken", "{broken"],
        [broken_final, broken_final],
    ],
)
def test_any_terminal_failure_including_oom_is_limited_to_two_calls(
    real_video, tmp_path, responses
):
    # A single list is shared across roles to model the sequence of actual invocations.
    queued = list(responses)
    agent, model = build(tmp_path, final=queued, terminal_review=queued)
    result = agent.solve(request(real_video, query_scope=TimeSpan(2, 8)))
    terminal = [c for c in model.calls if c["role"] in {"final", "terminal_review"}]
    assert len(terminal) == 2 and not queued
    assert result.support_level != "supported" and result.unresolved_reasons
    assert not any(c["role"] == "repair" for c in model.calls)


def test_second_failure_keeps_usable_prediction_and_latest_uncertainty(real_video, tmp_path):
    agent, _ = build(tmp_path, final=broken_final, terminal_review="{broken")
    result = agent.solve(request(real_video, query_scope=TimeSpan(2, 8)))
    assert result.prediction == "A" and result.support_level == "partial"
    assert "terminal_json_unparseable" in result.unresolved_reasons
    assert len(result.trace["terminal_audits"]) == 2


def test_terminal_recovery_media_budget_does_not_issue_or_charge_extra_call(real_video, tmp_path):
    agent, model = build(tmp_path)
    state = state_with_candidates(agent, real_video, spans=((2, 8),))

    def first(p):
        state.context.frame_exposures = state.request.budget.max_frame_exposures
        return broken_final(p)

    model.handlers["final"] = first
    result = agent._finish(state)
    assert result.trace["terminal_call_count"] == 1
    assert result.support_level == "partial"
    assert "terminal_recovery_media_budget" in result.unresolved_reasons


def test_r1_30_completed_state_rejected_before_model_loading(tmp_path):
    from test_r1345_debug_runner import case

    from qwen3vl_agent import debug12

    cases = [case("R1")]
    configs = {"R1": {"model": {}, "r1_v3": {}}}
    old = {
        "r1_protocol": "r1-local-evidence/3.0",
        "r1_execution_branch": "v3_visual",
        "pipeline_versions": {"R1": "v3", "R3": "v1", "R4": "v1", "R5": "v1"},
        "planned_request_ids": [cases[0].id],
    }
    old["pipeline_versions"] = debug12.pipeline_versions(configs)
    out = tmp_path / "old"
    debug12.atomic_json(
        out / "run_plan.json", {"signature": debug12.json_hash(old), "identity": old}
    )
    debug12.atomic_json(out / "items" / (cases[0].id + ".json"), {"status": "completed"})
    with pytest.raises(ValueError, match="identical"):
        debug12.run_cases(
            cases,
            cases,
            configs,
            out,
            {},
            resume=True,
            model_factory=lambda _: pytest.fail("model must not load"),
            gpu_check=None,
        )
    assert debug12.r1_execution_identity(configs)["r1_protocol"] == POLICY_ID


def test_v3_runner_fatal_call_saved_before_stop(tmp_path):
    from test_r1345_debug_runner import case

    from qwen3vl_agent import debug12

    item = case("R1")
    model = FakeModel(query=RuntimeError("synthetic engine failure"))

    def agent_factory(_pipeline, wrapped, _config):
        def solve(req):
            wrapped.generate(
                [
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": "R1:query\nINPUT_JSON:\n{}"}],
                    }
                ]
            )

        return SimpleNamespace(solve=solve)

    with pytest.raises(debug12.FatalModelError):
        debug12.run_cases(
            [item],
            [item],
            {"R1": {"model": {}, "r1_v3": {}}},
            tmp_path / "run",
            {},
            model_factory=lambda _: model,
            agent_factory=agent_factory,
            gpu_check=None,
        )
    call = json.loads((tmp_path / "run/calls/R1-TEST-1/00001.json").read_text())
    assert call["state"] == "error"
    assert (tmp_path / "run/fatal_error.json").exists()


def test_prompt_rules_use_actual_fields_without_case_specific_answers():
    p = {
        "query": {
            "fields": [{"field_id": "Q7", "description": "container material"}],
            "observation_modes": ["static"],
            "coverage": "point",
        }
    }
    assert "Q7" in prompt("observe", p) and "MUST list" in prompt("observe", p)
    for text in (
        prompt("query", {}),
        repair_prompt("query", "{}", "reference_relation invalid", {}),
    ):
        assert "method, action or source" in text and "not subtitle or ASR" in text
        assert "reference_relation" in text
    final_prompt = prompt("final", {})
    assert (
        "cause/effect" in final_prompt and "COMPLETE" in final_prompt and '"basis"' in final_prompt
    )
