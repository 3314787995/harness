"""Only the new runner contracts; no fake re-test of pipeline reasoning."""

import json
from dataclasses import asdict
from types import SimpleNamespace

import pytest

from qwen3vl_agent.debug12 import (
    PIPELINES,
    DebugCase,
    FatalModelError,
    PipelineExecutionError,
    default_agent,
    run_cases,
)
from qwen3vl_agent.models.base import BaseVideoModel, ModelOutput


def case(pipeline, index=1):
    return DebugCase(
        {"pipeline_id": pipeline, "request_id": f"{pipeline}-TEST-{index}", "video_id": "v",
         "video_path": "video.mp4", "question": "What is visible?",
         "choices": [{"label": "A", "text": "A cup"}, {"label": "B", "text": "A plate"}],
         "available_modalities": ["video", "screen_text"], "output_protocol": "multiple_choice",
         "gold": "DO_NOT_SEND_GOLD", "inspection": "DO_NOT_SEND_REVIEW"},
        {}, {"answer_label": "A", "private_annotation": "DO_NOT_SEND_GOLD"},
    )


class Model(BaseVideoModel):
    def __init__(self):
        super().__init__("Qwen/Qwen3-VL-8B-Instruct")
        self.loads = self.unloads = self.calls = 0

    def load(self):
        self.loads += 1
        self._loaded = True

    def unload(self):
        self.unloads += 1
        self._loaded = False

    def generate(self, messages, **kwargs):
        self.calls += 1
        if messages[0]["content"] == "fail":
            raise RuntimeError("synthetic CUDA failure")
        return ModelOutput("{}", {})


def agent_factory(fail_id=None, seen=None):
    def build(pipeline, model, config):
        def solve(request):
            if seen is not None:
                seen.append((pipeline, request.request_id))
            model.generate([{"role": "user", "content": "fail" if request.request_id == fail_id else "okay"}])
            payload = {"prediction": "A", "support_level": "supported", "completion_state": "complete", "resources": {"model_calls": 1}}
            return SimpleNamespace(**payload, to_dict=lambda: payload)
        return SimpleNamespace(solve=solve)
    return build


@pytest.mark.parametrize("pipeline", PIPELINES)
def test_request_drops_gold_and_review(pipeline, tmp_path):
    model = Model()
    agent = default_agent(pipeline, model, {pipeline.lower(): {}})
    assert agent.model is model and model.loads == 0
    request = case(pipeline).request(tmp_path, resume=True)
    encoded = json.dumps(asdict(request))
    assert "DO_NOT_SEND" not in encoded
    assert "gold" not in asdict(request) and "inspection" not in asdict(request)
    assert type(request).__name__ == pipeline + "Request"
    assert list(request.available_modalities) == ["video", "screen_text"]
    assert [c.label for c in request.choices] == ["A", "B"]
    if pipeline != "R1":
        assert request.resume is False
        checkpoint = tmp_path / "checkpoints" / f"{request.request_id}.jsonl"
        checkpoint.parent.mkdir(exist_ok=True)
        checkpoint.write_text("checkpoint", encoding="utf-8")
        assert case(pipeline).request(tmp_path, resume=True).resume is True


def test_resume_skips_completed_items_and_loads_once(tmp_path):
    cases = [case(p) for p in PIPELINES]
    model, seen = Model(), []
    configs = {p: {"model": {}, p.lower(): {}} for p in PIPELINES}
    kwargs = dict(model_factory=lambda cfg: model, agent_factory=agent_factory(seen=seen), gpu_check=None)
    result = run_cases(cases, cases[:1], configs, tmp_path, {"signature": "fixed"}, **kwargs)
    assert result["completed"] == 1
    result = run_cases(cases, cases, configs, tmp_path, {"signature": "fixed"}, resume=True, **kwargs)
    assert result["completed"] == 4
    assert len(seen) == model.calls == 4
    assert model.loads == model.unloads == 2  # one load per process/run, not per agent
    run_cases(cases, cases, configs, tmp_path, {"signature": "fixed"}, resume=True, **kwargs)
    assert model.loads == 2  # a completed run never loads weights
    with pytest.raises(ValueError, match="identical"):
        run_cases(cases, cases, configs, tmp_path, {"signature": "changed"}, resume=True, **kwargs)


@pytest.mark.parametrize("pipeline", ["R1", "R3", "R5"])
def test_fatal_call_is_journaled_and_stops_batch(tmp_path, pipeline):
    cases = [case(pipeline, i) for i in (1, 2, 3)]
    model, seen = Model(), []
    configs = {p: {"model": {}, p.lower(): {}} for p in PIPELINES}
    with pytest.raises(FatalModelError, match="synthetic CUDA failure"):
        run_cases(cases, cases, configs, tmp_path, {"signature": "fixed"},
                  model_factory=lambda cfg: model, agent_factory=agent_factory(cases[1].id, seen), gpu_check=None)
    assert model.loads == model.unloads == 1
    assert [x[1] for x in seen] == [cases[0].id, cases[1].id]
    assert (tmp_path / "items" / f"{cases[0].id}.json").exists()
    failed_item = tmp_path / "items" / f"{cases[1].id}.json"
    if pipeline == "R3":
        assert json.loads(failed_item.read_text())["status"] == "failed"
        before = model.loads
        with pytest.raises(PipelineExecutionError, match="Saved terminal failure"):
            run_cases(cases, cases, configs, tmp_path, {"signature": "fixed"}, resume=True,
                      model_factory=lambda cfg: model, agent_factory=agent_factory(), gpu_check=None)
        assert model.loads == before
    else:
        assert not failed_item.exists()
    failure = json.loads((tmp_path / "fatal_error.json").read_text())
    assert failure["request_id"] == cases[1].id
    call = json.loads(next((tmp_path / "calls" / cases[1].id).glob("*.json")).read_text())
    assert call["state"] == "error" and "synthetic CUDA failure" in call["error"]
    assert json.loads((tmp_path / "summary.json").read_text())["completed"] == 1


def r4_results_factory(outcomes, seen):
    def build(pipeline, model, config):
        def solve(request):
            seen.append(request.request_id)
            model.generate([{"role": "user", "content": "okay"}])
            prediction, failure = outcomes[request.request_id]
            payload = {"prediction": prediction, "failure": failure,
                       "support_level": "supported" if prediction else "unsupported",
                       "completion_state": "compile_error" if failure else "complete" if prediction else "evidence_incomplete",
                       "resources": {"model_calls": 1}}
            return SimpleNamespace(**payload, to_dict=lambda: payload)
        return SimpleNamespace(solve=solve)
    return build


@pytest.mark.parametrize("stage", ["compile", "observe"])
def test_r4_failure_is_saved_and_resume_skips_before_model_loading(tmp_path, stage):
    cases = [case("R4", i) for i in (1, 2)]
    model, seen = Model(), []
    failure = {"stage": stage, "code": "response_contract_failed", "message": "invalid response"}
    outcomes = {cases[0].id: (None, failure), cases[1].id: ("A", None)}
    configs = {p: {"model": {}, p.lower(): {}} for p in PIPELINES}
    kwargs = dict(model_factory=lambda cfg: model, agent_factory=r4_results_factory(outcomes, seen), gpu_check=None)
    run_cases(cases, cases, configs, tmp_path, {"signature": "r4-fixed"}, **kwargs)
    saved = json.loads((tmp_path / "items" / f"{cases[0].id}.json").read_text())
    assert saved["status"] == "failed" and saved["prediction"] is saved["correct"] is None
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["planned"] == 2 and summary["failed"] == 1 and summary["answered"] == 1
    assert summary["pending"] == [] and summary["terminal_count"] == 2
    assert seen == [c.id for c in cases] and model.loads == model.unloads == 1
    assert summary["r4_result_status_counts"] == {"execution_failed":1,"supported":1,"unresolved":0,"budget_exhausted":0}
    run_cases(cases, cases, configs, tmp_path, {"signature": "r4-fixed"}, resume=True, **kwargs)
    assert model.loads == model.unloads == 1 and model.calls == 2


def test_r4_wrong_and_unknown_predictions_continue_without_inflating_score(tmp_path):
    cases = [case("R4", i) for i in (1, 2, 3, 4)]
    model, seen = Model(), []
    outcomes = {c.id: (p, None) for c, p in zip(cases, ["A", "B", None, "A"])}
    configs = {p: {"model": {}, p.lower(): {}} for p in PIPELINES}
    kwargs = dict(model_factory=lambda cfg: model, agent_factory=r4_results_factory(outcomes, seen), gpu_check=None)
    summary = run_cases(cases, cases, configs, tmp_path, {"signature": "r4-fixed"}, **kwargs)
    assert seen == [c.id for c in cases]
    assert summary["planned"] == summary["completed"] == 4
    assert summary["answered"] == 3 and summary["no_prediction"] == 1
    assert summary["correct"] == 2 and summary["answer_match_rate_planned"] == 0.5
    assert summary["failed"] == 0
    row = json.loads((tmp_path / "items" / f"{cases[2].id}.json").read_text())
    assert row["prediction"] is row["correct"] is None
    run_cases(cases, cases, configs, tmp_path, {"signature": "r4-fixed"}, resume=True, **kwargs)
    assert model.loads == 1 and model.calls == 4


def test_r5_empty_answers_continue_and_verification_does_not_change_accuracy(tmp_path):
    cases = [case("R5", i) for i in (1, 2, 3, 4)]
    outcomes = {c.id: p for c, p in zip(cases, ["A", "B", "", "A"])}
    model, seen = Model(), []

    def factory(pipeline, model, config):
        def solve(request):
            seen.append(request.request_id)
            prediction = outcomes[request.request_id]
            payload = {"prediction": prediction, "completion_state": "partial",
                       "support_level": "unverified" if prediction else "none",
                       "answer_basis": "composer" if prediction else "no_answer",
                       "verification_status": "not_performed",
                       "evidence_refs": ["frame:source"] if prediction else [],
                       "resources": {"model_calls": 0}}
            return SimpleNamespace(**payload, to_dict=lambda: payload)
        return SimpleNamespace(solve=solve)

    configs = {p: {"model": {}, p.lower(): {}} for p in PIPELINES}
    kwargs = dict(model_factory=lambda cfg: model, agent_factory=factory, gpu_check=None)
    result = run_cases(cases, cases, configs, tmp_path, {"signature": "r5-direct-v3"}, **kwargs)
    assert result["completed"] == 4 and result["no_prediction"] == 1
    assert result["correct"] == 2 and result["answer_match_rate_planned"] == .5
    assert result["unverified_answers"] == result["answers_with_evidence"] == 3
    assert result["unbacked_fallbacks"] == result["supported_correct"] == 0
    empty = json.loads((tmp_path / "items" / f"{cases[2].id}.json").read_text())
    assert empty["prediction"] is empty["correct"] is None and empty["status"] == "completed"
    run_cases(cases, cases, configs, tmp_path, {"signature": "r5-direct-v3"}, resume=True, **kwargs)
    assert model.loads == 1 and seen == [c.id for c in cases]
