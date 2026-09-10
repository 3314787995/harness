"""Adversarial role outputs and composition regressions."""

import json
from copy import deepcopy
from dataclasses import replace

import pytest
from r6_fakes import FakeMedia, FakeModel, assessment, fact, query, relation, request
from test_r6_core import ledger
from test_r6_runtime import tiny_video

from qwen3vl_agent.r6 import R6Config, R6VideoAgent
from qwen3vl_agent.r6.acquisition import action
from qwen3vl_agent.r6.controller import Controller, gap
from qwen3vl_agent.r6.evaluate import run_requests
from qwen3vl_agent.r6.schema import validate_observation
from qwen3vl_agent.r6.state import evaluate_assessment, verification_checks
from qwen3vl_agent.r6.types import ProtocolError


def test_conflicting_shared_atom_cannot_support_two_inconsistent_choices():
    state, spec = ledger(), query()
    spec["option_claims"][1]["logic"] = {"op": "not", "args": [{"op": "atom", "id": "a0"}]}
    value = assessment(spec, state["facts"])
    value["candidates"][1]["atom_assessments"][0]["atom_id"] = "a0"
    result = evaluate_assessment(value, spec, state["facts"], {}, state["sources"])
    assert all(c["factual_status"] == "unknown" for c in result["candidates"])
    assert any(w["reason"] == "conflicting_shared_atom" for w in result["normalizations"])


def test_contradicted_relation_cannot_be_positive_motive_support():
    state, spec = ledger(), query(relation_type="reveals")
    row = relation()
    row["support_state"] = "contradicted"
    result = evaluate_assessment(
        assessment(spec, state["facts"], relation_rows=[row]),
        spec,
        state["facts"],
        {},
        state["sources"],
    )
    assert result["candidates"][0]["selection_status"] == "unknown"


def test_e08_overview_cannot_certify_real_occasion_universe():
    state, spec = ledger(), query()
    state["facts"]["F000001"]["story_time"] = [1, 1]
    spec["answer_operator"] = "exact_set"
    spec["option_claims"][0]["selection_set"] = [1]
    spec["option_claims"][1]["selection_set"] = [1, 2]
    value = assessment(
        spec, state["facts"], relation_rows=[relation(story_time=[1, 1])], both_true=True
    )
    value.update(
        universe_complete=True,
        coverage_fact_ids=["F000001"],
        occasions=[
            {"index": 1, "status": "supported", "fact_ids": ["F000001"], "relation_ids": ["r1"]}
        ],
    )
    result = evaluate_assessment(
        value, spec, state["facts"], {}, state["sources"], coverage_complete=False
    )
    assert result["universe_complete"] is False
    assert result["candidates"][0]["selection_status"] == "unknown"
    complete = evaluate_assessment(
        value, spec, state["facts"], {}, state["sources"], coverage_complete=True
    )
    assert complete["candidates"][0]["selection_status"] == "supported"
    assert complete["candidates"][1]["selection_status"] == "contradicted"
    assert complete["competitors"][0]["addressed"]


def test_verifier_receives_negative_selection_and_exact_expression():
    state, spec = ledger(), query()
    spec["answer_operator"] = "false_statement"
    spec["option_claims"][0]["selection_polarity"] = "negative"
    for option in spec["option_claims"]:
        option["text"] = "Original literal choice " + option["label"]
    value = assessment(spec, state["facts"])
    value["candidates"][0]["atom_assessments"][0]["status"] = "contradicted"
    normalized = evaluate_assessment(value, spec, state["facts"], {}, state["sources"])
    check = verification_checks(normalized, spec, state["facts"], {})[0]
    assert check["selection_polarity"] == "negative"
    assert check["expression"] == spec["option_claims"][0]["logic"]
    assert check["literal_proposition"] == "Original literal choice A"


def test_text_fact_keeps_attribution_and_quote_must_exist():
    sources = {"T01": {"modality": "subtitle", "text": "I trust her."}}
    record = fact("T01")
    value = {"records": [record], "gaps": [], "overflow": False}
    with pytest.raises(ProtocolError, match="attribution"):
        validate_observation(value, sources, query())
    record.update(kind="attributed_statement", speaker="p", quote_or_paraphrase="quote")
    with pytest.raises(ProtocolError, match="quote text"):
        validate_observation(value, sources, query())
    record["predicate"] = "I trust her."
    validate_observation(value, sources, query())


def test_checker_prose_cannot_leak_prior_answer_to_observer(tmp_path):
    controller = Controller(R6VideoAgent(FakeModel(), media_factory=FakeMedia), request(tmp_path))
    controller.state["query_spec"] = query()
    payload = controller.observer_payload("SECRET: choose A and justify the previous prediction")
    assert "SECRET" not in json.dumps(payload)


@pytest.mark.parametrize("policy,expected_frames", [("targeted", 1), ("uniform", 8)])
def test_targeted_and_uniform_refinement_execute_different_sampling(
    tmp_path, policy, expected_frames
):
    count = 0

    def checker(body):
        nonlocal count
        count += 1
        data = body["input"]
        value = assessment(data["query_spec"], {f["id"]: f for f in data["facts"]})
        if count == 1:
            value["gaps"] = [gap("local_fact", "Check the event", span=[0, 0.5])]
            value["actions"] = [
                action("observe_clip", span=[0, 0.5], query="Check event", gap_index=0)
            ]
        return value

    model = FakeModel({"relation_checker": checker})
    result = R6VideoAgent(
        model, replace(R6Config(), refinement_policy=policy), media_factory=FakeMedia
    ).solve(request(tmp_path))
    observers = [c for c in result.trace["receipts"] if c["role"] == "observer"]
    assert observers[1]["visual_exposures"] == expected_frames
    assert result.stop_reason == "EVIDENCE_SUFFICIENT"


@pytest.mark.parametrize(
    "direct_channel,verification,observer_count", [(False, True, 2), (True, False, 1)]
)
def test_shortcut_and_verification_ablations_are_executable(
    tmp_path, direct_channel, verification, observer_count
):
    model = FakeModel()
    result = R6VideoAgent(
        model,
        replace(R6Config(), direct_channel=direct_channel, relation_verification=verification),
        media_factory=FakeMedia,
    ).solve(request(tmp_path))
    assert sum(c["role"] == "observer" for c in model.calls) == observer_count
    if not verification:
        assert result.evidence_status == "insufficient" and not result.trace["verifications"]


def test_interrupted_call_is_charged_and_resume_preserves_facts(tmp_path):
    class InterruptOnce(FakeModel):
        interrupted = False

        def generate(self, messages, **kwargs):
            role = json.loads(messages[1]["content"][0]["text"])["role"]
            if role == "relation_checker" and not self.interrupted:
                self.interrupted = True
                raise KeyboardInterrupt
            return super().generate(messages, **kwargs)

    model = InterruptOnce()
    item = request(tmp_path, checkpoint_path=str(tmp_path / "interrupted.jsonl"))
    agent = R6VideoAgent(model, media_factory=FakeMedia)
    with pytest.raises(KeyboardInterrupt):
        agent.solve(item)
    result = agent.solve(replace(item, resume=True))
    assert result.stop_reason == "EVIDENCE_SUFFICIENT"
    assert result.costs["model_calls"] == 5 and len(result.trace["facts"]) == 1
    assert any(r["status"] == "interrupted" for r in result.trace["receipts"])


def test_real_media_batch_entrypoint_serializes_and_resumes(tmp_path):
    item = replace(request(tmp_path), video_path=str(tiny_video(tmp_path)))
    config = R6Config.from_mapping({"media": {"cache_dir": str(tmp_path / "cache")}})
    model = FakeModel()
    first = run_requests(
        [item], model_settings={}, config=config, output=tmp_path / "run", model=model
    )
    count = len(model.calls)
    second = run_requests(
        [item], model_settings={}, config=config, output=tmp_path / "run", model=model, resume=True
    )
    assert first == second and len(model.calls) == count == 4
    trace = json.loads((tmp_path / "run" / first[0]["trace_file"]).read_text())
    assert trace["trace"]["receipts"][1]["source_manifest"]


def test_development_inputs_preserve_all_original_choices_and_exclude_gold():
    from pathlib import Path

    from qwen3vl_agent.r6.evaluate import public_request, read_jsonl

    root = Path(__file__).resolve().parents[1] / "examples/r6"
    rows = read_jsonl(root / "requests.jsonl")
    gold = {r["request_id"]: r for r in read_jsonl(root / "answers.jsonl")}
    assert len(rows) == len(gold) == 9
    for row in rows:
        item = public_request(row, root)
        assert gold[item.request_id]["answer"] in {c.label for c in item.choices}
        assert not ({"answer", "time_reference", "gold", "native_task"} & row.keys())
        assert len(item.choices) == len(row["choices"])


def test_unreadable_existing_media_is_a_recorded_zero_model_failure(tmp_path):
    model = FakeModel()
    result = R6VideoAgent(model).solve(request(tmp_path))
    assert result.stop_reason == "TOOL_FAILURE" and result.costs["model_calls"] == 0
    assert result.technical_fallback and not model.calls


def test_real_aligned_subtitle_can_take_direct_evidence_path(tmp_path):
    path = tmp_path / "dialogue.srt"
    path.write_text("1\n00:00:00,000 --> 00:00:01,000\nI trust her.\n", encoding="utf-8")
    spec = query(required_modalities=["subtitle"])
    model = FakeModel({"compiler": lambda body: deepcopy(spec)})
    item = request(
        tmp_path,
        subtitle_path=str(path),
        subtitle_policy="aligned_only",
        allowed_modalities=("video", "subtitle"),
    )
    result = R6VideoAgent(model, media_factory=FakeMedia).solve(item)
    assert result.stop_reason == "EVIDENCE_SUFFICIENT"
    assert result.costs["visual_exposures"] == 0
    assert all(f["kind"] == "attributed_statement" for f in result.trace["facts"].values())
    assert result.source_ids and all(s.startswith("R6T-") for s in result.source_ids)


def test_resume_finishes_remaining_overflow_children(tmp_path):
    from qwen3vl_agent.models.qwen3vl import InputContextExceeded

    class InterruptedSplit(FakeModel):
        small_observations = 0

        def generate(self, messages, **kwargs):
            body = json.loads(messages[1]["content"][0]["text"])
            if body["role"] == "observer":
                if len(body["source_manifest"]) > 16:
                    raise InputContextExceeded(18000, 16384)
                self.small_observations += 1
                if self.small_observations == 2:
                    raise KeyboardInterrupt
            return super().generate(messages, **kwargs)

    model = InterruptedSplit()
    item = request(tmp_path, checkpoint_path=str(tmp_path / "split.jsonl"))
    agent = R6VideoAgent(model, media_factory=FakeMedia)
    with pytest.raises(KeyboardInterrupt):
        agent.solve(item)
    result = agent.solve(replace(item, resume=True))
    assert len(result.trace["facts"]) == 2 and result.stop_reason == "EVIDENCE_SUFFICIENT"
    assert result.costs["model_calls"] == 7
