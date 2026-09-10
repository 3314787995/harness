"""Only the V3 protocol, recovery, state and public-entry regressions (no GPU)."""

import copy
import json
from dataclasses import asdict
from pathlib import Path

import pytest
from test_r1 import FakeModel as LegacyFakeModel
from test_r1 import final as legacy_final
from test_r1 import locate, observe, query
from test_r1_v2 import real_video as real_video  # noqa: PLC0414 -- shared pytest fixture

from qwen3vl_agent.coarse_to_fine.types import FrameRef
from qwen3vl_agent.p01.config import P01Config
from qwen3vl_agent.p01.types import TimeSpan
from qwen3vl_agent.r1.agent import _State
from qwen3vl_agent.r1.control import audit_bundle, parse_query
from qwen3vl_agent.r1.media import MediaBatch
from qwen3vl_agent.r1.runtime import RunContext
from qwen3vl_agent.r1.types import EvidenceBundle, R1Request, SearchCandidate
from qwen3vl_agent.r1_v3 import R1V3Config, R1V3VideoAgent
from qwen3vl_agent.r1_v3.observation import parse_observation, rebuild
from qwen3vl_agent.r1_v3.types import ObservationRecord, V3Packet
from qwen3vl_agent.r1_v3.version import POLICY_ID


def final(payload):
    result = legacy_final(payload)
    for item in result["choice_assessments"]:
        item["basis"] = "The cited target colour supports this value or excludes its alternative."
    return result


class FakeModel(LegacyFakeModel):
    def __init__(self, **handlers):
        super().__init__(**{"final": final, "terminal_review": final, **handlers})


def observation(payload, **kwargs):
    old = observe(payload, **kwargs)
    response = {
        "target": {
            "status": "matched",
            "description": "the requested target person",
            "source_frame_ids": old["anchor_source_ids"],
            "unresolved_conditions": [],
        },
        "facts": old["facts"],
        "gaps": [],
    }
    if payload["query"]["coverage"] != "point":
        response["coverage_gaps"] = []
    if payload["query"]["coverage"] == "existence":
        response.update(existence="present", absence_basis="")
    return response


def build(tmp_path, **handlers):
    model = FakeModel(observe=handlers.pop("observe", observation), **handlers)
    agent = R1V3VideoAgent(
        model, config=R1V3Config(media=P01Config(cache_dir=str(tmp_path / "cache")))
    )
    return agent, model


def request(real_video, **kwargs):
    return R1Request(
        str(real_video),
        "What colour is the target person's clothing?",
        choices=("red", "green"),
        **kwargs,
    )


def test_empty_gaps_complete_without_repair(real_video, tmp_path):
    agent, model = build(tmp_path)
    result = agent.solve(request(real_video))
    assert result.support_level == "supported", result.unresolved_reasons
    assert result.resources["refinements"] == 0
    assert not any(c["role"] == "repair" for c in model.calls)
    assert result.trace["navigation_index"]["shot_count"] == 5
    prompt = next(c for c in result.resources["calls"] if c["role"] == "observe")["prompt"]
    schema = prompt.split("OUTPUT_SCHEMA:\n")[1].split("\nINPUT_JSON:")[0]
    assert set(json.loads(schema)) == {"target", "facts", "gaps"}
    assert "speech_binding" not in prompt


@pytest.mark.parametrize(
    "fault", ["missing_target_refs", "invalid_fact", "broken_json", "invented_action"]
)
def test_original_frames_repair_units_without_text_repair(real_video, tmp_path, fault):
    seen = []

    def handler(p):
        answer = observation(p)
        seen.append(p)
        if len(seen) == 1:
            if fault == "broken_json":
                return "{broken"
            if fault == "missing_target_refs":
                answer["target"]["source_frame_ids"] = []
            if fault == "invalid_fact":
                bad = copy.deepcopy(answer["facts"][0])
                bad["source_frame_ids"] = ["UNSHOWN"]
                answer["facts"].append(bad)
            if fault == "invented_action":
                answer["review_request"] = {"kind": "before", "reason": "invented"}
        return answer

    agent, model = build(tmp_path, observe=handler)
    result = agent.solve(request(real_video))
    assert result.support_level == "supported", result.unresolved_reasons
    assert result.resources["refinements"] == 1 and result.resources["relocations"] == 0
    assert seen[1]["purpose"] == "output_binding"
    assert seen[0]["frames"] == seen[1]["frames"]
    assert seen[0]["batch_span"] == seen[1]["batch_span"]
    assert not any(c["role"] == "repair" for c in model.calls)
    records = result.trace["observation_records"]
    assert records[0]["errors"] and not records[1]["errors"]
    if fault == "invalid_fact":
        assert len(records[0]["facts"]) == 1
    if fault == "missing_target_refs":
        assert records[0]["target"]["status"] == "unresolved"
        assert records[0]["target"]["source_frame_ids"] == []
    assert result.trace["evidence_state"][0]["resolutions"]


def test_compiler_prompt_and_one_failed_repair_still_block(real_video, tmp_path):
    broken = query(reference_relation="", requires_reference=False)
    agent, _model = build(tmp_path, query=broken, repair=broken)
    result = agent.solve(request(real_video))
    repairs = [c for c in result.resources["calls"] if c["role"] == "repair"]
    assert len(repairs) == 1
    assert 'MUST be "any"' in repairs[0]["prompt"]
    assert "original compilation input" in repairs[0]["prompt"]
    assert "query_compiler_unresolved" in result.unresolved_reasons
    assert result.support_level != "supported"


def unit_packet():
    batch = MediaBatch(
        TimeSpan(0, 3), (FrameRef("F1", 1, "unused.png"), FrameRef("F2", 2, "unused.png"))
    )
    return V3Packet("p", "c", "answer", batch.span), batch, parse_query(query())


def append_unit(packet, batch, spec, data):
    record_id = f"{packet.packet_id}.o{len(packet.observations) + 1}"
    parsed = parse_observation(
        data,
        False,
        packet=packet,
        batch=batch,
        query=spec,
        source_id="video",
        record_id=record_id,
        review_ids=[r.record_id for r in packet.observations],
    )
    record = ObservationRecord(
        record_id,
        record_id,
        "test",
        json.dumps(data),
        batch,
        parsed["target"],
        parsed["facts"],
        parsed["gaps"],
        parsed["errors"],
        parsed["reviews"],
        None,
    )
    packet.observations.append(record)
    rebuild(packet)
    return record


def unit_data(batch, spec, **kwargs):
    return observation(
        {"frames": [f.to_dict() for f in batch.frames], "query": asdict(spec)}, **kwargs
    )


def test_empty_later_view_preserves_confirmed_target_and_fact():
    packet, batch, spec = unit_packet()
    first = append_unit(packet, batch, spec, unit_data(batch, spec))
    empty = unit_data(batch, spec)
    empty["target"].update(status="unresolved", source_frame_ids=[])
    empty["facts"] = []
    append_unit(packet, batch, spec, empty)
    assert packet.anchor_match == "matched"
    assert packet.bound_fact_ids == [first.facts[0].fact_id]
    assert first.target["status"] == "matched"


def test_new_target_cannot_take_old_unbound_facts():
    packet, batch, spec = unit_packet()
    first = unit_data(batch, spec)
    first["target"].update(status="unresolved", source_frame_ids=[])
    append_unit(packet, batch, spec, first)
    next_view = unit_data(batch, spec)
    next_view["facts"] = []
    append_unit(packet, batch, spec, next_view)
    assert packet.anchor_match == "matched" and packet.bound_fact_ids == []
    assert packet.facts[0].observation_status == "partial"
    assert audit_bundle(EvidenceBundle("b", [packet]), spec).missing_fields == ["Q1"]


def test_explicit_target_refutation_requires_original_sources():
    packet, batch, spec = unit_packet()
    first = append_unit(packet, batch, spec, unit_data(batch, spec))
    negative = unit_data(batch, spec)
    negative["target"]["status"] = "mismatched"
    negative["facts"] = []
    append_unit(packet, batch, spec, negative)
    assert "target_observation_conflict" in packet.unresolved
    negative["reviews"] = [
        {
            "record_id": first.record_id,
            "judgment": "refuted",
            "source_frame_ids": ["F1"],
            "basis": "The cited original target is a different object.",
        }
    ]
    append_unit(packet, batch, spec, negative)
    assert "target_observation_conflict" in packet.unresolved  # original was F2
    negative["reviews"][0]["source_frame_ids"] = ["F2"]
    append_unit(packet, batch, spec, negative)
    assert packet.anchor_match == "mismatched"
    assert "target_observation_conflict" not in packet.unresolved


def test_fact_conflict_is_resolved_only_by_source_review():
    packet, batch, spec = unit_packet()
    first = append_unit(packet, batch, spec, unit_data(batch, spec, value="red"))
    blue = unit_data(batch, spec, value="blue")
    append_unit(packet, batch, spec, blue)
    bundle = EvidenceBundle("b", [packet])
    assert "conflicting_facts" in audit_bundle(bundle, spec).unresolved
    blue["reviews"] = [
        {
            "record_id": first.record_id,
            "fact_id": first.facts[0].fact_id,
            "judgment": "refuted",
            "source_frame_ids": ["F2"],
            "basis": "Re-read the original source.",
        }
    ]
    append_unit(packet, batch, spec, blue)
    assert audit_bundle(bundle, spec).sufficient
    assert first.facts[0].structured_value == "red"


@pytest.mark.parametrize("unit", ["target", "fact"])
def test_unresolved_review_cannot_reactivate_refuted_evidence(unit):
    packet, batch, spec = unit_packet()
    first = append_unit(packet, batch, spec, unit_data(batch, spec, value="red"))
    checked = unit_data(batch, spec, value="blue")
    if unit == "target":
        checked["target"]["status"] = "mismatched"
        checked["facts"] = []
    review = {
        "record_id": first.record_id,
        "fact_id": first.facts[0].fact_id if unit == "fact" else "",
        "judgment": "refuted",
        "source_frame_ids": ["F2"],
        "basis": "The original source refutes the earlier reading.",
    }
    checked["reviews"] = [review]
    append_unit(packet, batch, spec, checked)
    later = copy.deepcopy(checked)
    later["target"].update(status="unresolved", source_frame_ids=[])
    later["facts"] = []
    # The requested refutation is downgraded because the actual original source is absent.
    later["reviews"][0]["source_frame_ids"] = ["F1"]
    latest = append_unit(packet, batch, spec, later)
    assert latest.reviews[0]["judgment"] == "unresolved"
    if unit == "target":
        assert packet.anchor_match == "mismatched"
        assert first.record_id not in packet.target_record_ids
    else:
        assert packet.fact_reviews[0]["judgment"] == "refuted"
        assert audit_bundle(EvidenceBundle("b", [packet]), spec).sufficient


def test_background_fact_retires_and_searches_new_position(real_video, tmp_path):
    counter = 0

    def locator(p):
        nonlocal counter
        counter += 1
        found = locate(p)
        if counter > 1:
            found["candidates"][0]["anchor_frame_ids"] = [p["nodes"][0]["frames"][-1]["id"]]
        return found

    first = True

    def handler(p):
        nonlocal first
        out = observation(p)
        if first:
            first = False
            out["target"].update(status="unresolved", source_frame_ids=[])
            out["facts"][0].update(
                statement="A background label is visible.", supports_query_fields=[]
            )
            out["gaps"] = [
                {"kind": "target_identity", "reason": "No target relationship was established."}
            ]
        return out

    agent, _model = build(tmp_path, locator=locator, observe=handler)
    result = agent.solve(request(real_video))
    assert result.support_level == "supported", result.unresolved_reasons
    assert result.resources["relocations"] == 1 and result.resources["refinements"] == 0
    rounds = result.trace["searches"]["answer"]["rounds"]
    assert {f["id"] for f in rounds[1]["shown_frames"]} - {
        f["id"] for f in rounds[0]["shown_frames"]
    }
    assert all("background" not in f.statement for f in result.evidence_bundle.facts)


def state_with_candidates(agent, real_video, spans=((2, 8), (2, 8))):
    req = request(real_video)
    ctx = RunContext(req.budget)
    state = _State(
        req,
        ctx,
        agent.session_type(agent.model, agent.media, agent.config, ctx),
        TimeSpan(0, 65),
        None,
        parse_query(query()),
    )
    agent._ensure_index(state)
    for name, span in zip(("left", "right"), spans):
        packet = agent._new_packet(state, SearchCandidate(name, TimeSpan(*span)), "answer")
        bundle = EvidenceBundle(name, [packet], required_spans=[packet.span])
        state.bundles.append(bundle)
        agent._observe_packet(state, packet)
    return state


@pytest.mark.parametrize(
    "relation,kind,expected",
    [
        ("same", "same_source_target", True),
        ("different", "discriminating_features", False),
        ("unresolved", "unresolved", False),
        ("same", "same_colour", False),
    ],
)
def test_local_candidate_links_do_not_merge_by_node_or_colour(
    real_video, tmp_path, relation, kind, expected
):
    def compare(p):
        return {
            "relation": relation,
            "basis_kind": kind,
            "same_occurrence": True,
            "basis": "Both groups identify their target occurrence.",
            "left_source_ids": p["left"]["source_frame_ids"],
            "right_source_ids": p["right"]["source_frame_ids"],
        }

    agent, model = build(tmp_path, candidate_review=compare)
    state = state_with_candidates(agent, real_video)
    ids = [f.fact_id for b in state.bundles for f in b.facts]
    assert not agent._decisive(state)
    assert agent._compare(state, *state.bundles)
    assert agent._decisive(state) is expected
    assert state.context.refinements == 1
    assert not agent._compare(state, *state.bundles)  # same task has no new source
    assert state.context.refinements == 1
    call = next(c for c in model.calls if c["role"] == "candidate_review")
    assert not any(part["type"] == "video" for part in call["content"])
    if expected:
        packet = state.bundles[0].packets[0]
        assert packet.aliases == ["right"]
        assert [f.fact_id for f in packet.facts] == ids


def test_candidate_review_reopens_only_with_new_records(real_video, tmp_path):
    def compare(p):
        return {
            "relation": "unresolved",
            "basis_kind": "unresolved",
            "same_occurrence": False,
            "basis": "The two target occurrences are still ambiguous.",
            "left_source_ids": p["left"]["source_frame_ids"],
            "right_source_ids": p["right"]["source_frame_ids"],
        }

    agent, model = build(tmp_path, candidate_review=compare)
    state = state_with_candidates(agent, real_video)
    assert agent._compare(state, *state.bundles)
    assert not agent._compare(state, *state.bundles)
    assert state.context.refinements == 1
    left = state.bundles[0].packets[0]
    batch = left.observations[0].batch
    new = unit_data(batch, state.query)
    new["target"]["description"] = "The target with a newly verified distinctive marking."
    append_unit(left, batch, state.query, new)
    assert agent._compare(state, *state.bundles)
    assert state.context.refinements == 2
    calls = [c for c in model.calls if c["role"] == "candidate_review"]
    assert (
        calls[0]["payload"]["left"]["source_frame_ids"]
        == calls[1]["payload"]["left"]["source_frame_ids"]
    )
    assert calls[0]["payload"]["left"]["record_ids"] != calls[1]["payload"]["left"]["record_ids"]


def test_detail_crop_and_fixed_scope(real_video, tmp_path):
    def handler(p):
        answer = observation(p, status="clear" if p["crop_transforms"] else "unreadable")
        if not p["crop_transforms"]:
            answer["gaps"] = [
                {
                    "kind": "detail",
                    "reason": "The target attribute is unreadable.",
                    "field_ids": ["Q1"],
                    "crop": {"frame_id": p["frames"][0]["id"], "bbox_xyxy_1000": [0, 0, 500, 500]},
                }
            ]
        return answer

    agent, model = build(tmp_path, observe=handler)
    result = agent.solve(request(real_video, allowed_scope=(7.2, 12.2), query_scope=(7.2, 12.2)))
    assert result.support_level == "supported", result.unresolved_reasons
    assert result.resources["refinements"] == 1 and result.resources["relocations"] == 0
    assert not any(c["role"] == "locator" for c in model.calls)
    assert all(
        7.2 <= f["timestamp_seconds"] <= 12.2
        for c in model.calls
        if c["role"] == "observe"
        for f in c["payload"]["frames"]
    )


def test_ordered_coverage_and_empty_later_batches(real_video, tmp_path):
    def handler(p):
        out = observation(p)
        if p["batch_span"]["start_seconds"] > 0:
            out["target"].update(status="unresolved", source_frame_ids=[])
            out["facts"] = []
        return out

    agent, _model = build(
        tmp_path, observe=handler, query=query(observation_modes=["ordered"], coverage="sequence")
    )
    result = agent.solve(request(real_video, query_scope=(0, 25)))
    assert result.support_level == "supported", result.unresolved_reasons
    assert result.resources["refinements"] == 0
    assert next(iter(result.trace["candidate_audits"].values()))["coverage_complete"]
    assert len(result.trace["observation_records"]) >= 3


def test_duplicate_skipped_action_does_not_spend_budget(real_video, tmp_path):
    agent, _model = build(tmp_path)
    state = state_with_candidates(agent, real_video)
    packet = state.bundles[0].packets[0]
    batch = packet.observations[0].batch
    assert agent._observe_batch(
        state, packet, batch, state.query, purpose="manual_review", refinement=True
    )
    assert not agent._observe_batch(
        state, packet, batch, state.query, purpose="manual_review", refinement=True
    )
    assert state.context.refinements == 1
    assert state.trace["action_decisions"][-1]["status"] == "skipped_duplicate"


def test_external_text_delegates_v2_without_loading_model(real_video, tmp_path, monkeypatch):
    from qwen3vl_agent.r1.types import R1Result
    from qwen3vl_agent.r1_v2 import R1V2VideoAgent

    seen = []

    def solve(self, req):
        seen.append((self.model, req))
        return R1Result(
            "A",
            "best_effort",
            "partial",
            "evidence_unresolved",
            trace={"policy_id": "r1-local-evidence/2.0"},
        )

    monkeypatch.setattr(R1V2VideoAgent, "solve", solve)
    agent, model = build(tmp_path)
    req = request(real_video, available_modalities=("video", "subtitle"))
    result = agent.solve(req)
    assert seen == [(model, req)] and not model.calls
    assert result.trace["execution_branch"] == "v2_external_text_compat"


def test_entry_config_and_version_defaults():
    from qwen3vl_agent import cli, debug12
    from qwen3vl_agent.config import load_config
    from qwen3vl_agent.r1 import R1Config, R1VideoAgent
    from qwen3vl_agent.r1_v2 import R1V2VideoAgent

    config = load_config(Path(__file__).parents[1] / "configs/r1_v3_8b.yaml")
    parsed = R1V3Config.from_mapping(config["r1_v3"])
    before, after = asdict(R1Config()), asdict(R1V3Config())
    before["media"].pop("cache_dir")
    after["media"].pop("cache_dir")
    assert before == after and parsed.observer_tokens == 3072
    assert type(debug12.default_agent("R1", FakeModel(), {"r1": {}})) is R1V3VideoAgent
    assert type(debug12.default_agent("R1", FakeModel(), {"r1_v2": {}})) is R1V3VideoAgent
    assert type(debug12.default_agent("R1", FakeModel(), {"r1_v3": {}})) is R1V3VideoAgent
    assert debug12.pipeline_versions({"R1": {"r1_v3": {}}})["R1"] == "v3"
    assert (
        cli.build_parser().parse_args(["--query", "x", "--strategy", "r1-v3"]).strategy == "r1-v3"
    )


def test_failed_visual_call_charged_once_and_can_recover(real_video, tmp_path):
    agent, _model = build(
        tmp_path, observe=[RuntimeError("controlled observation failure"), observation]
    )
    result = agent.solve(request(real_video, query_scope=(2, 8)))
    assert result.support_level == "supported", result.unresolved_reasons
    assert result.resources["refinements"] == 1
    assert result.resources["model_calls"] <= 20
    assert result.trace["observation_records"][0]["errors"]


def test_unrecoverable_output_stops_without_dense_or_relocation(real_video, tmp_path):
    agent, model = build(tmp_path, observe="{broken")
    result = agent.solve(request(real_video, query_scope=(2, 8)))
    assert result.support_level != "supported"
    assert result.trace["controller_stop_reason"] == "no_progress"
    assert result.resources["refinements"] == 1 and result.resources["relocations"] == 0
    assert [c["payload"]["purpose"] for c in model.calls if c["role"] == "observe"] == [
        "initial",
        "output_binding",
    ]


def test_original_sequence_replay_completes_only_original_span(real_video, tmp_path):
    agent, _model = build(
        tmp_path,
        observe=["{broken", observation, observation, observation],
        query=query(observation_modes=["ordered"], coverage="sequence"),
    )
    result = agent.solve(request(real_video, query_scope=(0, 25)))
    assert result.support_level == "supported", result.unresolved_reasons
    records = result.trace["observation_records"]
    assert records[0]["batch"]["span"] == records[-1]["batch"]["span"]
    assert records[0]["batch"]["frames"] == records[-1]["batch"]["frames"]
    assert result.resources["refinements"] == 1


@pytest.mark.parametrize("fixed", [False, True])
def test_only_explicit_context_gap_expands_window(real_video, tmp_path, fixed):
    count = 0

    def handler(p):
        nonlocal count
        count += 1
        out = observation(p)
        if count == 1:
            out["gaps"] = [
                {
                    "kind": "temporal_context",
                    "direction": "before",
                    "reason": "The preceding action is needed.",
                }
            ]
        return out

    agent, model = build(tmp_path, observe=handler)
    req = request(real_video)
    ctx = RunContext(req.budget)
    state = _State(
        req,
        ctx,
        agent.session_type(model, agent.media, agent.config, ctx),
        TimeSpan(0, 65),
        TimeSpan(12, 16) if fixed else None,
        parse_query(query()),
    )
    packet = agent._new_packet(state, SearchCandidate("one", TimeSpan(12, 16)), "answer")
    bundle = EvidenceBundle("b", [packet], required_spans=[packet.span])
    state.bundles = [bundle]
    agent._observe_packet(state, packet)
    assert agent._detail(state, bundle, packet)
    assert packet.span.start_seconds == (12 if fixed else 6)
    assert not packet.active_gaps
    assert ctx.refinements == 1


def test_temporal_ambiguity_remains_a_competitor():
    packet, batch, spec = unit_packet()
    data = unit_data(batch, spec)
    data["target"].update(status="unresolved", source_frame_ids=[])
    data["facts"] = []
    data["gaps"] = [
        {"kind": "temporal_selection", "reason": "An earlier matching occurrence is unresolved."}
    ]
    append_unit(packet, batch, spec, data)
    assert not R1V3VideoAgent._retired(packet)
    assert not audit_bundle(EvidenceBundle("b", [packet]), spec).sufficient


def test_same_object_without_same_occurrence_cannot_merge(real_video, tmp_path):
    def compare(p):
        return {
            "relation": "same",
            "same_occurrence": False,
            "basis_kind": "same_source_target",
            "basis": "The object may appear in different occurrences.",
            "left_source_ids": p["left"]["source_frame_ids"],
            "right_source_ids": p["right"]["source_frame_ids"],
        }

    agent, _model = build(tmp_path, candidate_review=compare)
    state = state_with_candidates(agent, real_video)
    agent._compare(state, *state.bundles)
    assert state.trace["candidate_links"][0]["relation"] == "unresolved"
    assert not agent._decisive(state)


def test_merged_candidate_span_and_original_fact_ids(real_video, tmp_path):
    def compare(p):
        return {
            "relation": "same",
            "same_occurrence": True,
            "basis_kind": "discriminating_features",
            "basis": "The same distinctive emblem identifies the same continuous occurrence.",
            "left_source_ids": p["left"]["source_frame_ids"],
            "right_source_ids": p["right"]["source_frame_ids"],
        }

    agent, _model = build(tmp_path, candidate_review=compare)
    state = state_with_candidates(agent, real_video, spans=((2, 8), (5, 11)))
    ids = [f.fact_id for b in state.bundles for f in b.facts]
    agent._compare(state, *state.bundles)
    packet = state.bundles[0].packets[0]
    assert packet.span == TimeSpan(2, 11)
    assert [f.fact_id for f in packet.facts] == ids
    assert all(
        packet.span.start_seconds <= f.start_sec <= f.end_sec <= packet.span.end_seconds
        for f in packet.facts
    )


def test_candidate_comparison_uses_original_crop_sources():
    packet, batch, spec = unit_packet()
    batch.frames = (*batch.frames, FrameRef("CROP", 2, "unused-crop.png"))
    batch.crops = {"CROP": {"source_frame_id": "F2"}}
    data = unit_data(batch, spec)
    data["target"]["source_frame_ids"] = ["CROP"]
    data["facts"][0]["source_frame_ids"] = ["CROP"]
    append_unit(packet, batch, spec, data)
    assert R1V3VideoAgent._comparison_sources(packet) == ["F2"]


def test_discrete_visual_reference_keeps_original_binding_contract(real_video, tmp_path):
    def binding(p):
        return {
            "relation": "same",
            "source_frame_ids": p["left"]["shown_frame_ids"] + p["right"]["shown_frame_ids"],
            "basis": "Both targets show the same distinctive embroidered emblem.",
            "discriminating_features": ["distinctive embroidered emblem"],
            "feature_kinds": ["marking"],
        }

    agent, model = build(
        tmp_path,
        query=query(
            requires_reference=True,
            reference_description="the reference person",
            reference_relation="after",
        ),
        binding=binding,
    )
    result = agent.solve(request(real_video))
    assert result.support_level == "supported", result.unresolved_reasons
    assert any(c["role"] == "binding" for c in model.calls)
    assert result.evidence_bundle.bindings[0].relation == "same"


def test_actual_v3_runner_gold_isolation_resume_and_version_signature(real_video, tmp_path):
    from test_r1345_debug_runner import case

    from qwen3vl_agent.debug12 import run_cases

    class LoadingModel(FakeModel):
        loads = unloads = 0

        def load(self):
            self.loads += 1

        def unload(self):
            self.unloads += 1

    model = LoadingModel(observe=observation)
    cases = [case("R1", i) for i in (1, 2)]
    for item in cases:
        item.row["video_path"] = str(real_video)
    configs = {"R1": {"model": {}, "r1_v3": {"media": {"cache_dir": str(tmp_path / "cache")}}}}
    out = tmp_path / "results"
    kwargs = {"model_factory": lambda _: model, "gpu_check": None}
    summary = run_cases(cases, cases, configs, out, {"r1_version": "v3"}, **kwargs)
    assert summary["completed"] == 2 and model.loads == model.unloads == 1
    assert "DO_NOT_SEND" not in json.dumps(model.calls)
    saved = json.loads((out / "items" / (cases[0].id + ".json")).read_text())
    assert saved["pipeline_version"] == "v3"
    count = len(model.calls)
    run_cases(cases, cases, configs, out, {"r1_version": "v3"}, resume=True, **kwargs)
    assert len(model.calls) == count and model.loads == 1
    with pytest.raises(ValueError, match="identical"):
        run_cases(cases, cases, configs, out, {"r1_version": "v2"}, resume=True, **kwargs)


def test_cli_and_module_execute_v3(real_video, tmp_path, monkeypatch, capsys):
    import importlib
    import sys

    from qwen3vl_agent import cli

    config = {"model": {}, "r1_v3": {"media": {"cache_dir": str(tmp_path / "cache")}}}
    monkeypatch.setattr(cli, "load_config", lambda _: config)
    monkeypatch.setattr(cli, "build_model", lambda _: FakeModel(observe=observation))
    trace = tmp_path / "cli.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "qwen3vl-agent",
            "--config",
            "unused.yaml",
            "--strategy",
            "r1-v3",
            "--video",
            str(real_video),
            "--query",
            "What colour is the target clothing?",
            "--choice",
            "red",
            "--choice",
            "green",
            "--trace-output",
            str(trace),
        ],
    )
    cli.main()
    assert json.loads(trace.read_text())["r1_v3"]["trace"]["policy_id"] == POLICY_ID
    module = importlib.import_module("qwen3vl_agent.r1_v3.__main__")
    monkeypatch.setattr(module, "load_config", lambda _: config)
    monkeypatch.setattr(module, "build_model", lambda _: FakeModel(observe=observation))
    req = tmp_path / "request.json"
    req.write_text(json.dumps(asdict(request(real_video))))
    trace = tmp_path / "module.json"
    monkeypatch.setattr(
        sys,
        "argv",
        ["r1_v3", "--config", "unused.yaml", "--request", str(req), "--trace-output", str(trace)],
    )
    module.main()
    assert json.loads(trace.read_text())["trace"]["policy_id"] == POLICY_ID
