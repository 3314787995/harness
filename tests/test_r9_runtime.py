import json
import sys
from dataclasses import asdict
from types import SimpleNamespace

import pytest

from qwen3vl_agent.models.base import ModelOutput
from qwen3vl_agent.models.qwen3vl import InputContextExceeded, Qwen3VLModel
from qwen3vl_agent.r9 import R9Request, R9VideoAgent
from qwen3vl_agent.r9.evaluate import run_rows
from qwen3vl_agent.r9.media import ScopedMedia
from qwen3vl_agent.r9.runtime import Session
from qwen3vl_agent.r9.types import BudgetExhausted, ProtocolError
from tests.r9_fakes import FakeModel, config, make_video


def test_one_frame_clip_is_not_padded(tmp_path):
    p = make_video(tmp_path / "tiny.mp4", frames=1)
    req = R9Request(str(p), "q", choices=["left", "right"], mode="B1")
    result = R9VideoAgent(FakeModel(), config(tmp_path)).solve(req)
    assert result.trace["resources"]["unique_source_frames"] == 1
    assert result.trace["resources"]["visual_exposures"] == 1


@pytest.mark.parametrize("limit", ["max_unique_source_frames", "max_visual_exposures"])
def test_visual_budget_blocks_dispatch_and_is_not_reset(tmp_path, limit):
    path = make_video(tmp_path / "v.mp4")
    req = R9Request(str(path), "q", **{limit: 2})
    cfg = config(tmp_path)
    media = ScopedMedia(req, cfg)
    batch = media.prepare(media.extract((0, 2), [0, 0.5, 1, 1.5]))
    m = FakeModel()
    session = Session(m, cfg, req, {}, lambda: None)
    with pytest.raises(BudgetExhausted):
        session.call("test", "answer", {}, batch, media.evidence(batch), terminal=True)
    assert not m.calls and not session.receipts


def test_repeated_parent_and_crop_are_separate_exposures(tmp_path):
    path = make_video(tmp_path / "v.mp4")
    req = R9Request(str(path), "q", choices=["left", "right"], mode="B1")
    cfg = config(tmp_path)
    media = ScopedMedia(req, cfg)
    f = media.extract((0, 2), [0.5]).frames[0]
    prepared = media.prepare(media.local_batch(f.id, [[0.1, 0.1, 0.8, 0.8]]))

    class AnswerModel(FakeModel):
        def generate(self, *args, **kwargs):
            return ModelOutput(
                json.dumps(
                    {"semantic_answer": "left", "unit": None, "source_ids": [], "reason": "test"}
                )
            )

    data = {}
    session = Session(AnswerModel(), cfg, req, data, lambda: None)
    for key in ("a", "b"):
        session.call(key, "answer", {}, prepared, media.evidence(prepared), terminal=True)
    costs = session.resources()
    assert costs["unique_source_frames"] == 1 and costs["visual_exposures"] == 4
    assert costs["crop_exposures"] == 2
    assert costs["input_tokens"] is None


def test_processor_grid_and_actual_pixels_telemetry():
    class Tensor:
        def numel(self):
            return 3 * 16 * 16 * 8

    assert Qwen3VLModel._processed_pixels({"pixel_values": Tensor()}) == 2048
    assert Qwen3VLModel._visual_grids({"image_grid_thw": [[1, 4, 4]]}) == {
        "image_grid_thw": [[1, 4, 4]]
    }


def test_context_guard_runs_before_device_transfer_or_generation(monkeypatch):
    class Inputs(dict):
        input_ids = SimpleNamespace(shape=(1, 2000))
        transferred = False

        def to(self, _):
            self.transferred = True
            raise AssertionError("over-budget inputs must not transfer to GPU")

    inputs = Inputs()

    class Processor:
        def apply_chat_template(self, *args, **kwargs):
            return "prompt"

        def __call__(self, **kwargs):
            assert kwargs["do_resize"] is False
            return inputs

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace())
    monkeypatch.setitem(
        sys.modules,
        "qwen_vl_utils",
        SimpleNamespace(process_vision_info=lambda *a, **k: (None, None, {})),
    )
    model = Qwen3VLModel("fixture")
    model._loaded, model.model, model.processor = True, object(), Processor()
    with pytest.raises(InputContextExceeded) as exc:
        model.generate([{"role": "user", "content": "q"}], input_token_limit=1000)
    assert exc.value.actual_tokens == 2000 and not inputs.transferred


def test_context_guard_splits_without_hidden_dropped_frames(tmp_path):
    p = make_video(tmp_path / "v.mp4")
    req = R9Request(str(p), "q", choices=["front-left", "back-right"])

    class TooLarge(FakeModel):
        split = False

        def generate(self, messages, **kwargs):
            body = json.loads(messages[-1]["content"][-1]["text"].split("R9_JSON\n")[1])
            if body["stage"] == "observe" and not self.split:
                self.split = True
                raise InputContextExceeded(17000, 15184)
            return super().generate(messages, **kwargs)

    result = R9VideoAgent(TooLarge(), config(tmp_path)).solve(req)
    assert result.trace["resources"]["failed_calls"] == 1
    actions = result.trace["actions"]
    assert actions[0]["context_split"]
    assert actions[1]["times"] + actions[2]["times"] == actions[0]["times"]


def test_run_missing_media_no_model_load_and_gold_rejected(tmp_path):
    model = FakeModel()
    row = asdict(R9Request(str(tmp_path / "missing.mp4"), "q"))
    results = run_rows(
        [row], output=tmp_path / "run.jsonl", checkpoint_dir=tmp_path / "cp", model=model
    )
    assert results[0]["run_status"] == "media_unavailable" and not model.is_loaded
    with pytest.raises(ProtocolError):
        run_rows(
            [dict(row, answer="A")],
            output=tmp_path / "bad.jsonl",
            checkpoint_dir=tmp_path / "cp",
            model=model,
        )


def test_invalid_fixed_evidence_and_scope_are_explicit(tmp_path):
    with pytest.raises(ProtocolError):
        R9Request("v", "q", comparison="fixed_evidence", fixed_frames=tuple(range(33)))
    p = make_video(tmp_path / "v.mp4")
    with pytest.raises(ProtocolError):
        ScopedMedia(R9Request(str(p), "q", allowed_scope=(0, 100)), config(tmp_path))


def test_true_model_failure_is_engineering_error(tmp_path):
    p = make_video(tmp_path / "v.mp4")

    class Broken(FakeModel):
        def generate(self, *args, **kwargs):
            raise RuntimeError("model unavailable")

    row = asdict(R9Request(str(p), "q", mode="B0"))
    rows = run_rows(
        [row], output=tmp_path / "out.jsonl", checkpoint_dir=tmp_path / "cp", model=Broken()
    )
    assert rows[0]["run_status"] == "engineering_error" and rows[0]["result"] is None


def test_empty_decoder_output_is_engineering_error_without_forced_answer(tmp_path, monkeypatch):
    path = make_video(tmp_path / "v.mp4")
    from qwen3vl_agent.p01.types import TimeSpan
    from qwen3vl_agent.r1.media import MediaBatch
    from qwen3vl_agent.r2.media import R2Media

    monkeypatch.setattr(R2Media, "extract", lambda *a, **k: MediaBatch(TimeSpan(0, 1), ()))
    model = FakeModel()
    rows = run_rows(
        [asdict(R9Request(str(path), "q"))],
        output=tmp_path / "run.jsonl",
        checkpoint_dir=tmp_path / "cp",
        model=model,
    )
    assert rows[0]["run_status"] == "engineering_error" and rows[0]["result"] is None
    assert all(c["body"]["stage"] != "answer" for c in model.calls)
