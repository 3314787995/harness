import json
from dataclasses import replace

import pytest
from r8_fakes import FakeModel, make_video, observation

from qwen3vl_agent.p01.config import P01Config
from qwen3vl_agent.r8 import R8Config, R8Request, R8VideoAgent
from qwen3vl_agent.r8.types import ModelFailure, ProtocolError


@pytest.fixture
def setup(tmp_path):
    video = make_video(tmp_path / "video.mp4")
    config = R8Config(media=P01Config(cache_dir=str(tmp_path / "cache")))
    request = R8Request(
        str(video), "Which price is closest per yogurt?", choices=("2", "3", "4", "5")
    )
    return config, request


def test_full_controller_and_role_isolation(setup):
    config, request = setup
    fake = FakeModel()
    result = R8VideoAgent(fake, config).solve(request)
    assert result.prediction == "B" and result.verified
    assert result.value["value"] == "199/60"
    roles = [c[0]["stage"] for c in fake.calls]
    assert roles == ["compile", "observe", "formalize", "reread", "audit"]
    for body, kwargs, messages in fake.calls:
        assert kwargs["temperature"] == 0
        if body["stage"] in {"compile", "observe", "reread", "formalize", "audit"}:
            assert "choices" not in body["input"] and "prediction" not in body["input"]
        if body["stage"] == "reread":
            assert "19.9" not in json.dumps(body["input"])
    assert result.resources["visual_tokens"] == "unavailable"


@pytest.mark.parametrize(
    "mode,role",
    [
        ("A", "direct"),
        ("B", "model_math"),
        ("C", "audit"),
        ("D", "audit"),
        ("E", "audit"),
        ("F", "audit"),
        ("G", "reread"),
    ],
)
def test_modes(setup, mode, role):
    config, request = setup
    fake = FakeModel()
    result = R8VideoAgent(fake, config).solve(replace(request, mode=mode))
    roles = [c[0]["stage"] for c in fake.calls]
    assert role in roles
    assert ("reread" in roles) == (mode == "G")
    assert result.trace["mode"] == mode
    if mode in {"A", "B"}:
        assert result.prediction == "B" and not result.verified


def test_resume_no_new_calls_and_fingerprint(setup, tmp_path):
    config, request = setup
    request = replace(request, checkpoint_path=str(tmp_path / "checkpoint.jsonl"))
    first = R8VideoAgent(FakeModel(), config).solve(request)
    fake = FakeModel()
    resumed = R8VideoAgent(fake, config).solve(replace(request, resume=True))
    assert first.prediction == resumed.prediction and not fake.calls
    with pytest.raises(ValueError, match="mismatch"):
        R8VideoAgent(fake, config).solve(
            replace(request, resume=True, question="changed closest question")
        )


def test_budget_and_forced_fallback(setup):
    config, request = setup
    fake = FakeModel()
    result = R8VideoAgent(fake, config).solve(
        replace(request, max_model_calls=2, require_choice=True)
    )
    assert result.status == "forced_guess" and not result.verified and result.prediction == "B"
    assert [c[0]["stage"] for c in fake.calls] == ["compile", "fallback"]


def test_engineering_failure_never_becomes_a(setup):
    config, request = setup
    fake = FakeModel({"observe": lambda _: RuntimeError("test model failure")})
    with pytest.raises(ModelFailure):
        R8VideoAgent(fake, config).solve(replace(request, require_choice=True))
    assert all(c[0]["stage"] != "fallback" for c in fake.calls)


def test_source_forgery_rejected(setup):
    config, request = setup

    def wrong(payload):
        output = observation(payload)
        output["observations"][0]["evidence_refs"] = ["forged_frame"]
        return output

    fake = FakeModel({"observe": wrong})
    with pytest.raises(ModelFailure):
        R8VideoAgent(fake, config).solve(request)
    assert sum(c[0]["stage"] == "observe" for c in fake.calls) == 2


def test_open_numeric_generate(setup):
    config, request = setup
    fake = FakeModel()
    result = R8VideoAgent(fake, config).generate(
        [{"role": "user", "content": request.question}],
        videos=[request.video_path],
        output_protocol="numeric",
    )
    assert result.text == "199/60" and result.metadata["r8"]["verified"]


def test_permissions(setup):
    config, request = setup
    fake = FakeModel()
    result = R8VideoAgent(fake, config).solve(
        replace(request, allowed_scope=(1.0, 3.0), observation_cutoff=2.5)
    )
    assert all(1 <= e["timestamp_seconds"] <= 2.5 for e in result.evidence.values())
    with pytest.raises(ProtocolError):
        R8VideoAgent(fake, config).solve(replace(request, fixed_frames=(99.0,)))
