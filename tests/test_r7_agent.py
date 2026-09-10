import json
from dataclasses import asdict, replace
from pathlib import Path

import pytest
from r7_fakes import QUESTION, ScriptedQwen, candidates, fact, make_video, reason, scenario, task

from qwen3vl_agent.r7 import R7Config, R7Request, R7VideoAgent
from qwen3vl_agent.r7.contracts import validate_candidates, validate_observation
from qwen3vl_agent.r7.evaluate import preflight, read_manifest, run_manifest, score
from qwen3vl_agent.r7.media import ScopedMedia
from qwen3vl_agent.r7.types import InputContract, ProtocolError


@pytest.fixture
def video(tmp_path):
    return make_video(tmp_path / "synthetic.mp4")


@pytest.fixture
def config(tmp_path):
    return R7Config.from_mapping({"media": {"cache_dir": str(tmp_path / "cache")}})


def request(video, **kwargs):
    return R7Request(str(video), QUESTION, ["A. 2", "B. 9"], **kwargs)


def test_s4_end_to_end_fact_world_intervention_and_permutation(video, config):
    fake = ScriptedQwen()
    result = R7VideoAgent(fake, config=config).solve(request(video))
    assert result.prediction == "A", result.to_dict()
    assert result.support_level == "entailed_by_execution"
    assert result.resources["model_calls"] == 4
    factual, hypothetical = result.scenarios
    assert factual["cells"]["R.count"]["value"] == 9
    assert hypothetical["cells"]["R.count"]["value"] == 2
    for role, payload in fake.payloads:
        if role in {"compile", "observe"}:
            assert "options" not in payload and "candidates" not in payload
    swapped = replace(request(video), choices=["A. 9", "B. 2"])
    other = R7VideoAgent(ScriptedQwen(), config=config).solve(swapped)
    assert other.prediction == "B"


@pytest.mark.parametrize("mechanism", ["S1", "S2"])
def test_s1_s2_end_to_end_conditional_support(video, config, mechanism):
    spec = task(mechanism)
    spec.update(interventions=[], target_visibility="unobserved_future")
    proposal = reason(
        rules=[
            {
                "id": "r",
                "basis": "hypothesis",
                "source_span": "",
                "description": "synthetic contextual rule",
            }
        ],
        scenarios=[
            scenario(
                hypotheses=[
                    {
                        "id": "h",
                        "mechanism": mechanism,
                        "output": "next",
                        "value": 2,
                        "conditions": [{"key": "L.count", "relation": "eq", "expected": 2}],
                        "rule_id": "r",
                    }
                ]
            )
        ],
    )
    fake = ScriptedQwen(
        spec=spec, proposal=proposal, candidate_hook=lambda p: candidates(p["options"], key="next")
    )
    result = R7VideoAgent(fake, config=config).solve(request(video))
    assert result.prediction == "A", result.to_dict()
    assert result.support_level == "conditional"
    assert any(r == "verify" for r, _ in fake.payloads)


@pytest.mark.parametrize("mechanism", ["S3", "S5"])
def test_s3_s5_full_controller_uses_execution_module(video, config, mechanism):
    from test_r7_execution import physics_fixture, trend_fixture

    w, model, _rules = physics_fixture() if mechanism == "S3" else trend_fixture()
    spec = task(mechanism)
    spec.update(
        targets=[{"id": k, "description": k} for k in {key.split(".")[0] for key in w.cells}],
        slots=[
            {
                "id": k,
                "entity_id": k.split(".")[0],
                "predicate": k.rsplit(".", 1)[-1],
                "description": k,
            }
            for k in w.cells
        ],
        interventions=[],
        stipulations=[
            {
                "id": "law",
                "key": "law",
                "value": "synthetic law",
                "source_span": "Set right to original left.",
            }
        ],
    )
    rule_id = model["rule_id"]
    proposal = reason(
        rules=[
            {
                "id": rule_id,
                "basis": "stipulated",
                "source_span": "Set right to original left.",
                "description": "synthetic law",
            }
        ],
        scenarios=[scenario(**{"physics" if mechanism == "S3" else "trends": [model]})],
    )

    def observer(payload):
        return {
            "entities": [],
            "facts": [
                fact(
                    k,
                    c.value,
                    unit=c.unit,
                    entity=k.split(".")[0],
                    time=payload["frames"][0]["source_seconds"],
                )
                for k, c in w.cells.items()
            ],
            "gaps": [],
        }

    key = "collision:A:C" if mechanism == "S3" else "levels"
    expected = (
        {"2": True, "9": False}
        if mechanism == "S3"
        else {"2": {"A": 30, "B": 30}, "9": {"A": 40, "B": 35}}
    )
    fake = ScriptedQwen(
        spec=spec,
        proposal=proposal,
        observe_hook=observer,
        candidate_hook=lambda p: candidates(p["options"], key=key, expected=expected),
    )
    result = R7VideoAgent(fake, config=config).solve(request(video))
    assert result.prediction == "A", result.to_dict()
    assert any(e.get("mechanism") == mechanism for s in result.scenarios for e in s["execution"])


def test_cutoff_scope_cache_crop_and_forbidden_suffix_invariance(tmp_path, config):
    first = make_video(tmp_path / "one.mp4", suffix_color="blue")
    second = make_video(tmp_path / "two.mp4", suffix_color="green")
    scopes, pixels = [], []
    for path in (first, second):
        media = ScopedMedia(
            request(path, protocol_id="strict_prefix", observation_cutoff=2),
            config,
            {"test": "fixed"},
        )
        batch = media.extract((0, 2), [0, 0.5, 1, 1.5, 2])
        prepared = media.prepare(batch)
        from PIL import Image

        pixels.append([Image.open(f.path).tobytes() for f in prepared.frames])
        scopes.append([f.id for f in batch.frames])
        with pytest.raises(ProtocolError):
            media.extract((0, 3), [2.5])
        with pytest.raises(ProtocolError):
            media.frame("foreign-frame")
        with pytest.raises(ProtocolError):
            media.prepare(replace(batch, span=type(batch.span)(0, 3)))
        crop = media.crop(batch.frames[0], [0, 0, 0.5, 0.5])
        assert media.get_evidence(crop.id)["source_frame_id"] == batch.frames[0].id
        with pytest.raises(ProtocolError):
            media.crop(batch.frames[0], [-0.1, 0, 0.5, 0.5])
        with pytest.raises(ProtocolError):
            media.crop(replace(batch.frames[0], timestamp_seconds=4), [0, 0, 0.5, 0.5])
    assert pixels[0] == pixels[1]
    assert scopes[0] == scopes[1]


def test_cache_tampering_and_scope_change_rejected(video, config):
    media = ScopedMedia(request(video, allowed_scope=(0, 2)), config, {})
    frame = media.extract((0, 2), [0]).frames[0]
    other = ScopedMedia(request(video, allowed_scope=(1, 2)), config, {})
    with pytest.raises(ProtocolError):
        other.frame(frame.id)
    Path(frame.path).write_bytes(b"tampered")
    with pytest.raises(ProtocolError):
        media.frame(frame.id)


def test_odd_vfr_and_nonlinear_sampling_retain_true_time(tmp_path, config):
    video = make_video(tmp_path / "vfr.mp4", variable_pts=True)
    media = ScopedMedia(request(video), config, {})
    batch = media.extract((0, 3), [0, 0.25, 1, 2, 3])
    prepared = media.prepare(batch)
    assert prepared.kind == "timestamped_images"
    assert all(f"{f.timestamp_seconds:.6f}s" in json.dumps(prepared.parts) for f in prepared.frames)
    assert all(media.get_evidence(f.id)["pts"] is not None for f in batch.frames)


def test_existing_subtitles_cannot_cross_cutoff_or_become_measured_fact(video, config, tmp_path):
    subtitles = tmp_path / "subtitles.jsonl"
    subtitles.write_text(
        "\n".join(
            json.dumps(r)
            for r in [
                {"start_sec": 0.1, "end_sec": 0.8, "text": "allowed"},
                {"start_sec": 1.5, "end_sec": 2.5, "text": "FORBIDDEN"},
                {"start_sec": 3, "end_sec": 4, "text": "FORBIDDEN"},
            ]
        ),
        encoding="utf-8",
    )
    req = request(
        video,
        subtitle_path=str(subtitles),
        available_modalities=("video", "screen_text", "subtitle"),
        observation_cutoff=2,
    )
    media = ScopedMedia(req, config, {})
    assert [s["text"] for s in media.read_subtitles((0, 2))] == ["allowed"]
    batch = media.extract((0, 2), [0, 1])
    evidence = media.evidence(media.prepare(batch), (0, 2))
    observation = {
        "entities": [],
        "facts": [fact("L.count", 2, refs=["T01"], time=0.1)],
        "gaps": [],
    }
    with pytest.raises(ProtocolError):
        validate_observation(observation, task(), evidence, [])


@pytest.mark.parametrize(
    "kwargs",
    [
        {"protocol_id": "strict_prefix"},
        {"observation_cutoff": -1},
        {"allowed_time_intervals": [(0, 2), (1, 3)]},
        {"available_modalities": ("video", "asr")},
        {"subtitle_path": "unpermitted.jsonl"},
    ],
)
def test_invalid_public_protocols(kwargs):
    with pytest.raises((ValueError, ProtocolError)):
        request("not-needed.mp4", **kwargs)


def test_source_frame_before_nonzero_start_is_never_selected(video, config):
    media = ScopedMedia(request(video, allowed_scope=(0.2, 1.3)), config, {})
    batch = media.extract((0.2, 1.3), [0.2, 1.3])
    assert all(0.2 <= f.timestamp_seconds <= 1.3 for f in batch.frames)


@pytest.mark.parametrize("role", ["compile", "candidates", "observe", "reason"])
def test_invalid_json_one_repair_no_fake_answer_and_charged(video, config, role):
    fake = ScriptedQwen(invalid_role=role)
    result = R7VideoAgent(fake, config=config).solve(request(video))
    assert result.prediction is None
    assert result.completion_state == "execution_error"
    assert sum(r == role for r, _ in fake.payloads) == 2
    assert result.resources["failed_calls"] == 2


def test_timeout_charged_without_answer(video, config):
    result = R7VideoAgent(ScriptedQwen(error_role="observe"), config=config).solve(request(video))
    assert result.prediction is None and result.resources["model_calls"] == 3
    assert result.resources["receipts"][-1]["status"] == "model_error"


def test_interrupted_call_resume_and_completed_replay(video, config, tmp_path):
    checkpoint = tmp_path / "resume.jsonl"
    req = request(video, checkpoint_path=str(checkpoint))
    interrupted = ScriptedQwen(interrupt_role="reason")
    with pytest.raises(KeyboardInterrupt):
        R7VideoAgent(interrupted, config=config).solve(req)
    resumed_model = ScriptedQwen()
    result = R7VideoAgent(resumed_model, config=config).solve(replace(req, resume=True))
    assert result.prediction == "A", result.to_dict()
    assert result.resources["model_calls"] == 5
    assert [r for r, _ in resumed_model.payloads] == ["reason"]
    replay_model = ScriptedQwen()
    replay = R7VideoAgent(replay_model, config=config).solve(replace(req, resume=True))
    assert replay.to_dict() == result.to_dict() and not replay_model.payloads
    mismatch = R7VideoAgent(ScriptedQwen(), config=config).solve(
        replace(req, resume=True, observation_cutoff=2)
    )
    assert mismatch.prediction is None and mismatch.completion_state == "input_error"


def test_budget_reserves_final_and_single_candidate(video, config):
    single = R7VideoAgent(ScriptedQwen(), config=config).solve(
        replace(request(video), choices=["Q. only"])
    )
    assert single.prediction == "Q" and single.support_level == "noninformative"
    assert single.resources["model_calls"] == 0
    limited = R7VideoAgent(ScriptedQwen(), config=config).solve(request(video, max_model_calls=3))
    assert limited.prediction in {"A", "B"}, limited.to_dict()
    assert limited.completion_state == "budget_limited" and limited.resources["model_calls"] == 3


def gap_verifier(value, payload):
    value["gaps"] = [
        {
            "kind": "perceptual",
            "affects_candidates": ["A", "B"],
            "neutral_query": "Read left count more clearly.",
            "window": [0, 1],
            "action": "observe",
            "frame_id": None,
            "bbox": None,
            "impact": 3,
            "resolvability": 3,
        }
    ]
    return value


def test_no_new_information_stops_refinement(video, config):
    spec = task()
    spec["interventions"] = []
    fake = ScriptedQwen(
        spec=spec,
        candidate_hook=lambda p: candidates(p["options"], key="missing"),
        verify_hook=gap_verifier,
    )
    result = R7VideoAgent(fake, config=config).solve(request(video))
    assert result.prediction == "A", result.to_dict()
    assert result.stop_reason == "no_new_information"
    assert sum(r == "observe" for r, _ in fake.payloads) == 2


def test_new_observation_triggers_reexecution(video, config):
    count = 0

    def observe(payload):
        nonlocal count
        count += 1
        t = payload["frames"][0]["source_seconds"]
        return {
            "entities": [],
            "facts": [
                fact(
                    "L.count",
                    None if count == 1 else 2,
                    time=t,
                    kind="unknown" if count == 1 else "observed",
                ),
                fact("R.count", 9, time=t),
            ],
            "gaps": [],
        }

    fake = ScriptedQwen(observe_hook=observe, verify_hook=gap_verifier)
    result = R7VideoAgent(fake, config=config).solve(request(video))
    assert result.prediction == "A", result.to_dict()
    assert result.support_level == "entailed_by_execution"
    assert sum(r == "reason" for r, _ in fake.payloads) == 2
    assert result.facts["versions"][1]["cells"]["L.count"]["value"] is None
    assert result.facts["versions"][-1]["cells"]["L.count"]["value"] == 2


@pytest.mark.parametrize("mode", ["B0", "B1", "B2", "B3", "B4", "B4-uniform"])
def test_all_ablation_modes_are_callable(video, config, mode):
    result = R7VideoAgent(ScriptedQwen(), config=config).solve(request(video, mode=mode))
    assert result.prediction in {"A", "B"}, result.to_dict()
    assert result.trace["mode"] == mode


def test_manifest_gold_separation_preflight_batch_score_and_fact_replay(video, config, tmp_path):
    manifest = tmp_path / "questions.jsonl"
    row = asdict(request(video))
    row.pop("checkpoint_path")
    row.pop("resume")
    manifest.write_text(json.dumps(row) + "\n", encoding="utf-8")
    rows = read_manifest(manifest)
    report = preflight(rows, config)
    assert report["ready"] and report["accuracy"] is None and not report["model_loaded"]
    output = tmp_path / "output"
    summary = run_manifest(rows, R7VideoAgent(ScriptedQwen(), config=config), output)
    assert summary["predictions_present"] == 1 and summary["accuracy"] is None
    fact_artifact = next((output / "facts").glob("*.json"))
    replay_model = ScriptedQwen()
    result = R7VideoAgent(replay_model, config=config).solve(
        request(video, mode="B3", facts_input=str(fact_artifact))
    )
    assert result.prediction == "A", result.to_dict()
    assert [r for r, _ in replay_model.payloads] == ["reason"]
    gold = tmp_path / "answers.jsonl"
    gold.write_text(
        json.dumps({"request_id": rows[0][0].request_id, "answer": "A"}), encoding="utf-8"
    )
    scoring = score(output / "predictions.jsonl", gold)
    assert scoring["scores"]["all"]["accuracy"] == 1 and scoring["official_group_score"] is None
    row["answer"] = "A"
    manifest.write_text(json.dumps(row), encoding="utf-8")
    with pytest.raises(ValueError):
        read_manifest(manifest)


def test_long_budget_is_duration_dependent_without_claiming_coverage(config):
    req = request("unused")
    contract = InputContract.resolve(req, 181)
    budget = config.budget(contract, req)
    assert budget["windows"] == 4
    assert budget["max_model_calls"] == 18 and budget["max_unique_frames"] == 224


def test_candidate_missing_compound_span_is_rejected(video):
    req = request(video)
    result = candidates()
    result["candidates"][0]["atoms"][0]["text_span"] = "fabricated"
    with pytest.raises(ProtocolError):
        validate_candidates(result, req)


def test_cli_single_r7_and_protocol_exit(monkeypatch, video, config, tmp_path, capsys):
    from qwen3vl_agent import cli

    fake = ScriptedQwen()
    monkeypatch.setattr(cli, "build_model", lambda _: fake)
    monkeypatch.setattr(cli, "load_config", lambda _: {"r7": asdict(config)})
    monkeypatch.setattr(
        "sys.argv",
        [
            "qwen3vl-agent",
            "--strategy",
            "r7",
            "--config",
            "fake.yaml",
            "--query",
            QUESTION,
            "--video",
            str(video),
            "--choice",
            "A. 2",
            "--choice",
            "B. 9",
        ],
    )
    cli.main()
    assert capsys.readouterr().out.strip() == "A"


def test_uniform_control_matches_refinement_frame_count(video, config):
    counts = []
    for mode in ("B4", "B4-uniform"):
        spec = task()
        spec["interventions"] = []
        fake = ScriptedQwen(
            spec=spec,
            candidate_hook=lambda p: candidates(p["options"], key="missing"),
            verify_hook=gap_verifier,
        )
        result = R7VideoAgent(fake, config=config).solve(request(video, mode=mode))
        assert result.prediction == "A", result.to_dict()
        counts.append(
            [r["frame_count"] for r in result.resources["receipts"] if r["role"] == "observe"][-1]
        )
    assert counts[0] == counts[1]


def test_checkpoint_interruption_after_refinement_commit(video, config, tmp_path, monkeypatch):
    from qwen3vl_agent.r3.checkpoint import Checkpoint

    original = Checkpoint.save

    def interrupt_after_write(self, state):
        original(self, state)
        pending = state.get("pending_refinement")
        if pending and pending["key"] in state.get("completed_observations", []):
            raise KeyboardInterrupt("after observation committed")

    monkeypatch.setattr(Checkpoint, "save", interrupt_after_write)
    spec = task()
    spec["interventions"] = []

    def fake():
        return ScriptedQwen(
            spec=spec,
            candidate_hook=lambda p: candidates(p["options"], key="missing"),
            verify_hook=gap_verifier,
        )

    req = request(video, checkpoint_path=str(tmp_path / "commit.jsonl"))
    with pytest.raises(KeyboardInterrupt):
        R7VideoAgent(fake(), config=config).solve(req)
    monkeypatch.setattr(Checkpoint, "save", original)
    resumed = fake()
    result = R7VideoAgent(resumed, config=config).solve(replace(req, resume=True))
    assert result.stop_reason == "no_new_information", result.to_dict()
    assert not resumed.payloads


def test_actual_model_payload_unchanged_by_forbidden_suffix(tmp_path, config):
    messages = []
    for name, color in (("a", "blue"), ("b", "green")):
        path = make_video(tmp_path / (name + ".mp4"), suffix_color=color)
        fake = ScriptedQwen()
        result = R7VideoAgent(fake, config=config).solve(
            request(path, observation_cutoff=2, protocol_id="strict_prefix")
        )
        assert result.prediction == "A"
        # Compare all textual role payloads, including post-observation evidence/state IDs.
        messages.append(fake.payloads)
    assert messages[0] == messages[1]


def test_shared_wrapper_model_revision_is_forwarded():
    from qwen3vl_agent.factory import build_model

    model = build_model({"path": "Qwen/Qwen3-VL-8B-Instruct", "revision": "pinned"})
    assert model.revision == "pinned"


def test_high_resolution_crop_uses_detail_budget_and_keeps_parent(tmp_path, config):
    import av
    from PIL import Image

    video = tmp_path / "large.mp4"
    with av.open(str(video), "w") as output:
        stream = output.add_stream("mpeg4", rate=8)
        stream.width, stream.height, stream.pix_fmt = 1200, 800, "yuv420p"
        for _ in range(2):
            frame = av.VideoFrame.from_image(Image.new("RGB", (1200, 800), "red"))
            for packet in stream.encode(frame):
                output.mux(packet)
        for packet in stream.encode():
            output.mux(packet)
    media = ScopedMedia(request(video), config, {})
    batch = media.extract((0, 0.25), [0])
    parent = batch.frames[0]
    crop = media.crop(parent, [0, 0, 0.9, 0.9])
    batch.frames = (parent, crop)
    batch.ordered = False
    batch.crops = {crop.id: media.catalog[crop.id]["crop_transform"]}
    prepared = media.prepare(batch)
    assert prepared.sizes[1][0] * prepared.sizes[1][1] > config.media.normal_max_pixels
    assert media.get_evidence(crop.id)["source_frame_id"] == parent.id


@pytest.mark.parametrize("operator", ["will_not", "least_likely"])
def test_negative_query_does_not_take_positive_truth_shortcut(video, config, operator):
    spec = task()
    spec["query_operator"] = operator

    def negative_verdict(value, payload):
        assert payload["question"] == QUESTION
        value["prediction"] = "B"
        return value

    fake = ScriptedQwen(spec=spec, verify_hook=negative_verdict)
    result = R7VideoAgent(fake, config=config).solve(request(video))
    assert result.prediction == "B", result.to_dict()
    assert any(role == "verify" for role, _ in fake.payloads)
