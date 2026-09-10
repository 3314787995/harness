"""Direct-answer contracts, using the recorded Composer failures without inference."""
import json
from dataclasses import replace
from pathlib import Path

import pytest

from qwen3vl_agent.models.base import ModelOutput
from qwen3vl_agent.r5 import R5Budget, R5Config, R5Request, R5Result
from qwen3vl_agent.r5.evaluate import run_manifest
from qwen3vl_agent.r5.checkpoint import Checkpoint
from qwen3vl_agent.r5.planning import post_calls, merge_calls
from qwen3vl_agent.r5.synthesis import parse_draft
from test_r5 import FakeModel, composer, run, setup  # shared synthetic media; no GPU


def request(choices=True):
    return R5Request(video_path="unused.mp4", question="What happens?",
                     choices={"A": "first", "D": "last"} if choices else ())


def test_three_recorded_composer_answers_survive_the_legacy_options_shape():
    rows = json.loads((Path(__file__).parent / "fixtures/r5_composer_failures.json").read_text(encoding="utf-8"))
    actual = []
    for row in rows:
        p = row["payload"]
        req = R5Request(video_path="unused.mp4", question=p["question"], choices=p["choices"])
        value = parse_draft(json.loads(row["raw_response"]), p["units"], p["facts"], req, R5Config())
        assert value["prediction"] == row["expected_prediction"]
        assert not value["rejected"] and not value["rejected_references"]
        assert all(c["support_refs"] for c in value["claims"])
        actual.append(value["prediction"])
    assert actual == ["A", "D", "A"]


@pytest.mark.parametrize("claims", [None, "broken", [], [None, {"statement": "", "support_refs": ["ghost"]}]])
def test_valid_mcq_survives_bad_auxiliary_fields_without_repair(setup, claims):
    result, model, _ = run(setup, FakeModel(compose={"prediction": "D", "claims": claims,
                                                   "options": {"label": ["D"]}}),
                           duration=2, choices=request().choices)
    assert result.prediction == "D" and result.answer_basis == "composer"
    assert result.support_level == "none" and result.completion_state == "partial"
    assert result.verification_status == "not_performed"
    assert not any(c["role"] in {"audit", "repair"} for c in model.calls)


@pytest.mark.parametrize("is_mcq", [True, False])
def test_bad_references_and_claims_are_isolated_without_erasing_good_text(setup, is_mcq):
    def output(p):
        ref = p["units"][0]["id"]
        return {"prediction": "D", "claims": [
            {"id": "duplicate", "statement": "First sentence.", "support_refs": [ref, "hypothesis:invented", {}]},
            {"id": "duplicate", "statement": "Second sentence.", "support_refs": ["unknown"]},
            {"statement": "", "support_refs": [ref]},
        ]}
    result, model, _ = run(setup, FakeModel(compose=output), duration=2,
                           choices=request(is_mcq).choices)
    assert result.prediction == ("D" if is_mcq else "First sentence. Second sentence.")
    draft = result.trace["composer"]
    assert len(draft["claims"]) == 2 and len({c["id"] for c in draft["claims"]}) == 2
    assert len(draft["rejected"]) == 1 and len(draft["rejected_references"]) == 3
    assert result.evidence_refs and all(ref.startswith("frame:") for ref in result.evidence_refs)
    assert result.support_level == "unverified" and result.completion_state == "partial"
    assert [c["role"] for c in model.calls] == ["compile", "observe", "compose"]


def test_unreferenced_free_text_is_retained_and_explicitly_unverified(setup):
    result, model, _ = run(setup, FakeModel(compose={"claims": [{"statement": "A tentative summary."}]}), duration=2)
    assert result.prediction == "A tentative summary."
    assert result.answer_basis == "composer" and result.support_level == "none"
    assert result.verification_status == "not_performed" and not result.evidence_refs
    assert "answer_claims_without_references" in result.unresolved_items
    assert not any(c["role"] == "repair" for c in model.calls)


@pytest.mark.parametrize("raw", ['{"prediction":"Z"}', '{}', 'not JSON'])
def test_invalid_core_answer_repairs_once_then_returns_empty(setup, raw):
    result, model, _ = run(setup, FakeModel(compose=raw), duration=2, choices=request().choices)
    assert result.prediction == "" and result.answer_basis == "no_answer"
    repairs = [c for c in model.calls if c["role"] == "repair"]
    assert len(repairs) == 1
    assert repairs[0]["payload"]["choices"] == [{"label": "A", "text": "first"}, {"label": "D", "text": "last"}]
    assert "no_valid_answer" in result.unresolved_items


def test_composer_and_repair_can_consume_the_last_two_calls(setup):
    raw = '{"prediction":"D","claims":['
    result, model, _ = run(setup, FakeModel(compose=raw, repair={"prediction": "D", "claims": []}),
                           duration=2, choices=request().choices, budget=R5Budget(max_model_calls=4))
    assert result.prediction == "D"
    assert [c["role"] for c in model.calls] == ["compile", "observe", "compose", "repair"]
    assert result.resources["model_calls"] == 4
    assert post_calls(20, R5Config()) == merge_calls(20) + 2


@pytest.mark.parametrize("repaired,reason", [
    ({"prediction": "A", "claims": []}, "changed the original prediction"),
    ({"prediction": "D", "claims": [{"statement": "Invented new content."}]}, "factual statement"),
])
def test_format_repair_cannot_change_a_known_selection_or_invent_content(setup, repaired, reason):
    result, _, _ = run(setup, FakeModel(compose='{"prediction":"D","claims":[', repair=repaired),
                       duration=2, choices=request().choices)
    assert result.prediction == "" and any(reason in i for i in result.unresolved_items)


def test_valid_json_answer_is_not_discarded_at_output_limit(setup):
    raw = json.dumps({"prediction": "D", "claims": []})
    result, model, _ = run(setup, FakeModel(compose=ModelOutput(raw, {"finish_reason": "length"})),
                           duration=2, choices=request().choices)
    assert result.prediction == "D" and "composer_output_truncated" in result.unresolved_items
    assert not any(c["role"] == "repair" for c in model.calls)


def test_no_observed_facts_does_not_start_composer_or_default_to_a(setup):
    result, model, _ = run(setup, FakeModel(observe={"facts": [], "truncated": False}),
                           duration=2, choices=request().choices)
    assert result.prediction == "" and "no_observed_facts" in result.unresolved_items
    assert not any(c["role"] == "compose" for c in model.calls)


@pytest.mark.parametrize("phase", ["merge", "compose", "repair"])
def test_returned_terminal_receipt_resumes_without_repeating_a_call(setup, tmp_path, monkeypatch, phase):
    original, interrupted = Checkpoint.save, False

    def save(self, value):
        nonlocal interrupted
        original(self, value)
        calls = value["context"]["calls"]
        if calls and calls[-1]["role"] == phase and calls[-1]["status"] == "returned" and not interrupted:
            interrupted = True
            raise KeyboardInterrupt()

    monkeypatch.setattr(Checkpoint, "save", save)
    model = FakeModel(compose="{}") if phase == "repair" else FakeModel()
    agent, model, _, video = setup(model)
    req = R5Request(video_path=video, question="Summarize.", checkpoint_path=str(tmp_path / "resume.jsonl"))
    with pytest.raises(KeyboardInterrupt):
        agent.solve(req)
    previous = sum(c["role"] == phase for c in model.calls)
    result = agent.solve(replace(req, resume=True))
    assert sum(c["role"] == phase for c in model.calls) == previous
    assert sum(c["role"] == "observe" for c in model.calls) == 3
    assert result.resources["model_calls"] == len(model.calls)
    assert any(c.get("replayed_from_receipt") for c in result.resources["calls"])
    count = len(model.calls)
    assert agent.solve(replace(req, resume=True)).to_dict() == result.to_dict()
    assert len(model.calls) == count


@pytest.mark.parametrize("old_version", ["r5-observation-v2", "r5-direct-v3"])
def test_old_protocol_checkpoint_is_rejected(setup, tmp_path, monkeypatch, old_version):
    import qwen3vl_agent.r5.agent as module
    from qwen3vl_agent.r5.observation import PROTOCOL_VERSION
    agent, _, _, video = setup()
    req = R5Request(video_path=video, question="Summarize.", checkpoint_path=str(tmp_path / "old.jsonl"))
    monkeypatch.setattr(module, "PROTOCOL_VERSION", old_version)
    agent.solve(req)
    monkeypatch.setattr(module, "PROTOCOL_VERSION", PROTOCOL_VERSION)
    with pytest.raises(ValueError, match="protocol changed"):
        agent.solve(replace(req, resume=True))


@pytest.mark.parametrize("key", ["audit_tokens", "audit_batch_cards", "max_source_revisits", "max_revisit_calls", "revisit_fps"])
def test_removed_configuration_has_explicit_migration_message(key):
    with pytest.raises(ValueError, match="removed Auditor/revisit settings"):
        R5Config.from_mapping({key: 4})


def test_fatal_composer_exception_is_preserved_and_propagated(setup, tmp_path):
    agent, _, _, video = setup(FakeModel(compose=ValueError("engine failure")))
    path = tmp_path / "fatal.jsonl"
    with pytest.raises(RuntimeError, match="engine failure"):
        agent.solve(R5Request(video_path=video, question="Summarize.", checkpoint_path=str(path)))
    assert "engine failure" in path.read_text(encoding="utf-8")


@pytest.mark.parametrize("phase", ["compile", "merge", "compose", "repair"])
def test_text_stage_oom_is_fatal_not_a_protocol_fallback(setup, tmp_path, phase):
    handlers = {phase: RuntimeError("CUDA out of memory")}
    if phase == "repair":
        handlers["compose"] = "{}"
    agent, model, _, video = setup(FakeModel(**handlers))
    path = tmp_path / "oom.jsonl"
    with pytest.raises(RuntimeError, match="CUDA out of memory"):
        agent.solve(R5Request(video_path=video, question="Summarize.", checkpoint_path=str(path)))
    assert model.calls[-1]["role"] == phase
    assert "CUDA out of memory" in path.read_text(encoding="utf-8")


def test_standalone_runner_records_empty_answer_as_null_and_continues(setup, tmp_path):
    agent, _, _, video = setup(FakeModel(compose={"prediction": "invalid"}))
    req = replace(request(), video_path=video)
    destination = tmp_path / "evaluation"
    summary = run_manifest(agent, [(req, {}), (replace(req, request_id="second"), {})], destination)
    assert summary["request_count"] == 2 and summary["answered_count"] == 0
    rows = [json.loads(line) for line in (destination / "predictions.jsonl").read_text().splitlines()]
    assert all(row["prediction"] is None for row in rows)


def test_standalone_runner_saves_input_error_before_stopping(tmp_path):
    class Agent:
        config = R5Config()
        calls = 0

        def solve(self, req):
            self.calls += 1
            return R5Result("", "factual_video_summary", "input_error", "none", "no_answer",
                            {"complete": False}, [], ["Cannot read video"], {}, {})

    agent = Agent()
    with pytest.raises(RuntimeError, match="R5 input failure"):
        run_manifest(agent, [(request(), {}), (replace(request(), request_id="second"), {})], tmp_path)
    assert agent.calls == 1
    assert len(list((tmp_path / "sidecars").glob("*.json"))) == 1
    assert "Cannot read video" in (tmp_path / "diagnostics.json").read_text()
