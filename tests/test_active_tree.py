from __future__ import annotations

from itertools import pairwise
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from qwen3vl_agent.active_tree import ActiveTreeConfig, ActiveTreeVideoAgent, SceneTreeBuilder
from qwen3vl_agent.active_tree.alignment import align_evidence_to_options
from qwen3vl_agent.active_tree.prompts import (
    ObserverDecision,
    ObserverFact,
    build_observer_prompt,
    canonicalize_options,
    parse_observer,
    parse_planner,
    parse_verification,
)
from qwen3vl_agent.active_tree.replay import render_trace_html
from qwen3vl_agent.active_tree.temporal import compose_temporal_option
from qwen3vl_agent.active_tree.types import (
    AtomicEvidence,
    EvidenceLedger,
    EvidenceSlot,
    PlannedAction,
    ProtocolError,
    SceneNode,
    SceneTree,
    TaskContract,
)
from qwen3vl_agent.coarse_to_fine import CachedVideo, FrameRef
from qwen3vl_agent.coarse_to_fine.cache import SubtitleCue, SubtitleTrack
from qwen3vl_agent.models.base import BaseVideoModel, ModelOutput


class FakeModel(BaseVideoModel):
    def __init__(self, responses: list[str]) -> None:
        super().__init__("fake", device="cpu", dtype="float32")
        self.responses = iter(responses)
        self.calls: list[dict[str, Any]] = []

    def load(self) -> None:
        self._loaded = True

    def unload(self) -> None:
        self._loaded = False

    def generate(self, messages, *, videos=None, images=None, **kwargs):
        self.calls.append(
            {
                "messages": messages,
                "videos": videos,
                "images": images,
                "kwargs": kwargs,
            }
        )
        return ModelOutput(next(self.responses), {"input_tokens": 10, "output_tokens": 5})


class FakeCache:
    def __init__(self, cached: CachedVideo) -> None:
        self.cached = cached

    def prepare(self, video_path: str) -> CachedVideo:
        assert video_path == self.cached.source_path
        return self.cached


def make_cached_video(tmp_path: Path, *, duration_seconds: float = 60.0) -> CachedVideo:
    frame_dir = tmp_path / "frames"
    frame_dir.mkdir(exist_ok=True)
    frames: list[FrameRef] = []
    for index in range(int(duration_seconds) + 1):
        path = frame_dir / f"F{index:06d}.jpg"
        color = (10, 20, 30) if index < 30 else (220, 210, 200)
        Image.new("RGB", (64, 48), color=color).save(path)
        frames.append(FrameRef(f"F{index:06d}", float(index), str(path)))
    return CachedVideo(
        source_path=str(tmp_path / "video.mp4"),
        cache_dir=str(tmp_path),
        duration_seconds=duration_seconds,
        source_fps=30.0,
        width=1280,
        height=720,
        sample_fps=1.0,
        frames=tuple(frames),
        cache_hit=True,
    )


def test_scene_tree_is_scene_aware_bounded_and_persistent(tmp_path: Path) -> None:
    cached = make_cached_video(tmp_path)
    config = ActiveTreeConfig(
        cache_dir=str(tmp_path),
        scene_min_seconds=4.0,
        scene_max_seconds=15.0,
        scene_change_threshold=0.1,
    )
    builder = SceneTreeBuilder(config)

    tree = builder.build(cached)
    root_children = tree.children(tree.root_id)
    leaves = sorted(
        (node for node in tree.nodes.values() if node.is_leaf and node.id != tree.root_id),
        key=lambda node: node.start_seconds,
    )

    assert 1 <= len(root_children) <= config.max_children
    assert leaves[0].start_seconds == 0.0
    assert leaves[-1].end_seconds == cached.duration_seconds
    assert all(
        left.end_seconds == right.start_seconds
        for left, right in pairwise(leaves)
    )
    assert tree.index_path and Path(tree.index_path).is_file()

    cached_tree = builder.build(cached)
    assert cached_tree.cache_hit is True
    assert cached_tree.to_dict()["node_count"] == tree.to_dict()["node_count"]


def test_active_tree_runs_option_blind_breadth_search_and_dual_verification(
    tmp_path: Path,
) -> None:
    cached = make_cached_video(tmp_path)
    model = FakeModel(
        [
            (
                '{"primary_topology":"local","required_modalities":["visual"],'
                '"answer_criterion":"direct_support","slots":[{"slot_id":"S1",'
                '"description":"find the decisive action","required":true,"constraint":""}]}'
            ),
            '{"facts":[],"missing_evidence":"inspect a branch"}',
            (
                '{"slots":[{"slot_id":"S1","description":"find the decisive action",'
                '"required":true,"constraint":""}],"option_tests":['
                '{"option_id":"O1","support_test":"one","refute_test":"not one"},'
                '{"option_id":"O2","support_test":"two","refute_test":"not two"},'
                '{"option_id":"O3","support_test":"three","refute_test":"not three"},'
                '{"option_id":"O4","support_test":"four","refute_test":"not four"}],'
                '"answer_criterion":"direct_support"}'
            ),
            (
                '{"actions":[{"kind":"observe","node_id":"L0-N0000",'
                '"mode":"inspect","slot_id":"S1","compare_node_ids":[],'
                '"expected_new_evidence":"decisive action"}]}'
            ),
            (
                '{"facts":[{"node_id":"L0-N0000","slot_ids":["S1"],'
                '"start_seconds":1.0,"end_seconds":2.0,"modality":"visual",'
                '"fact":"the second option action is visible","supports_option_ids":["O2"],'
                '"refutes_option_ids":[],"source_frame_ids":["F000001"],"subtitle_refs":[]}],'
                '"missing_evidence":""}'
            ),
            (
                '{"candidate_option_text":"two","decisive_evidence_id":"EV0001",'
                '"strongest_alternative_text":"one","alternative_refuted":true,'
                '"missing_slot_ids":[],"counterevidence":"","reason":"complete"}'
            ),
            (
                '{"candidate_option_text":"two","decisive_evidence_id":"EV0001",'
                '"strongest_alternative_text":"one","alternative_refuted":true,'
                '"missing_slot_ids":[],"counterevidence":"",'
                '"reason":"independent match"}'
            ),
        ]
    )
    config = ActiveTreeConfig(
        cache_dir=str(tmp_path),
        scene_max_seconds=20.0,
        scene_change_threshold=1.0,
        protocol_repair_attempts=0,
    )
    agent = ActiveTreeVideoAgent(model, config=config, cache=FakeCache(cached))
    agent.load()

    output = agent.generate(
        [{"role": "user", "content": "What happened?"}],
        videos=[cached.source_path],
        choices=["one", "two", "three", "four"],
    )

    trace = output.metadata["active_tree"]
    assert output.text == "B"
    assert trace["verified"] is True
    assert trace["stop_reason"] == "dual_verifier_agreement"
    assert trace["events"][0]["type"] == "root_breadth_observation"
    assert trace["events"][1]["type"] == "option_reveal"
    assert any(event["type"] == "active_observation" for event in trace["events"])
    assert trace["evidence_ledger"][0]["supports_option_ids"] == []
    assert [call["role"] for call in trace["resources"]["calls"]] == [
        "task_compiler",
        "breadth_observer",
        "option_discriminator",
        "planner",
        "observer",
        "completeness_verifier",
        "blind_skeptic",
    ]
    assert "OPTIONS" not in model.calls[0]["messages"][0]["content"]
    assert model.calls[1]["images"]
    assert model.calls[1]["videos"] is None
    assert "OPTIONS" not in model.calls[4]["messages"][0]["content"]
    assert model.calls[-1]["videos"]

    replay = render_trace_html(trace, tmp_path / "replay.html")
    replay_text = replay.read_text(encoding="utf-8")
    assert "Active Evidence Tree replay" in replay_text
    assert "dual_verifier_agreement" in replay_text


def test_missing_slots_stop_without_spending_calls_on_answer_verifiers(
    tmp_path: Path,
) -> None:
    cached = make_cached_video(tmp_path)
    model = FakeModel(
        [
            (
                '{"primary_topology":"local","required_modalities":["visual"],'
                '"answer_criterion":"direct_support","slots":[{"slot_id":"S1",'
                '"description":"find the decisive action","required":true,'
                '"constraint":""}]}'
            ),
            '{"facts":[],"missing_evidence":"not visible at breadth"}',
            "{}",
            (
                '{"actions":[{"kind":"observe","node_id":"L0-N0000",'
                '"mode":"inspect","slot_id":"S1","compare_node_ids":[],'
                '"expected_new_evidence":"decisive action"}]}'
            ),
            '{"facts":[],"missing_evidence":"still missing"}',
        ]
    )
    agent = ActiveTreeVideoAgent(
        model,
        config=ActiveTreeConfig(
            cache_dir=str(tmp_path),
            scene_max_seconds=20.0,
            scene_change_threshold=1.0,
            max_search_observations=1,
            protocol_repair_attempts=0,
        ),
        cache=FakeCache(cached),
    )
    agent.load()

    output = agent.generate(
        [{"role": "user", "content": "What happened?"}],
        videos=[cached.source_path],
        choices=["one", "two", "three", "four"],
    )

    trace = output.metadata["active_tree"]
    roles = [call["role"] for call in trace["resources"]["calls"]]
    assert trace["verified"] is False
    assert trace["stop_reason"] == "evidence_search_exhausted_with_missing_slots"
    assert "completeness_verifier" not in roles
    assert "blind_skeptic" not in roles


def test_dialogue_guard_replaces_generic_contract_and_requires_subtitles() -> None:
    generic = TaskContract(
        "local",
        [EvidenceSlot("S1", "observable fact to find")],
        ["visual"],
    )

    guarded = ActiveTreeVideoAgent._guard_contract(
        "What are the people arguing about?",
        generic,
        subtitles_available=True,
    )

    assert guarded.required_modalities == ["subtitle"]
    assert guarded.primary_topology == "local"
    assert guarded.slots[0].description != "observable fact to find"


def test_breadth_evidence_routes_search_but_does_not_complete_required_slot() -> None:
    ledger = EvidenceLedger()
    ledger.add(
        [
            AtomicEvidence(
                "EV0001",
                "N1",
                ("S1",),
                0.0,
                1.0,
                "subtitle",
                "a routing clue",
                (),
                (),
                (),
                ("a routing clue",),
                "breadth",
            )
        ]
    )

    assert ledger.missing_slot_ids(TaskContract("local", [EvidenceSlot("S1", "fact")])) == [
        "S1"
    ]
    assert ledger.covered_modalities() == set()


def test_subtitle_only_contract_normalizes_planner_observation_mode() -> None:
    action = PlannedAction("observe", "N1", "detail_ocr", "S1")

    normalized = ActiveTreeVideoAgent._required_mode_action(
        action,
        TaskContract("local", [EvidenceSlot("S1", "spoken topic")], ["subtitle"]),
        object(),  # only availability is relevant to this pure controller guard
    )

    assert normalized.mode == "subtitle"
    assert normalized.node_id == "N1"


def test_enumerated_action_sequence_forces_visual_event_grounding() -> None:
    guarded = ActiveTreeVideoAgent._guard_contract(
        (
            "In which order do these events happen?\n"
            "(a) The cloth is removed.\n(b) The speaker introduces himself.\n"
            "(c) A feather appears."
        ),
        TaskContract(
            "sequence",
            [
                EvidenceSlot("S1", "The cloth is removed"),
                EvidenceSlot("S2", "The speaker introduces himself"),
                EvidenceSlot("S3", "A feather appears"),
            ],
            ["subtitle", "ocr"],
        ),
        subtitles_available=True,
    )

    assert guarded.required_modalities == ["visual", "subtitle"]
    assert guarded.slots[0].constraint == "ground the visible event and its timestamp"
    assert guarded.slots[1].constraint == "ground the spoken event and its timestamp"


def test_visual_sequence_normalizes_ocr_plan_to_motion() -> None:
    normalized = ActiveTreeVideoAgent._required_mode_action(
        PlannedAction("observe", "N1", "detail_ocr", "S1"),
        TaskContract("sequence", [EvidenceSlot("S1", "event")], ["visual"]),
        None,
    )

    assert normalized.mode == "motion"


def test_sequence_visual_observer_prompt_is_target_blind() -> None:
    prompt = build_observer_prompt(
        "In which order do the hidden target events happen?",
        TaskContract(
            "sequence",
            [EvidenceSlot("S1", "remove the white cloth")],
            ["visual"],
        ),
        action=PlannedAction(
            "observe",
            "N1",
            "motion",
            "S1",
            expected_new_evidence="white cloth is removed",
        ),
        nodes=[
            SceneNode(
                id="N1",
                start_seconds=10.0,
                end_seconds=20.0,
                level=1,
                parent_id="ROOT",
            )
        ],
        frame_lines=["F000010 @ 10.000s", "F000020 @ 20.000s"],
        subtitles="the white cloth is removed",
    )

    assert "target-blind visual transcriber" in prompt
    assert "hidden target events" not in prompt
    assert "remove the white cloth" not in prompt
    assert "white cloth is removed" not in prompt
    assert "F000010 @ 10.000s" in prompt
    assert "exactly TWO distinct supplied frames" in prompt


def test_sequence_fine_event_prompt_reveals_target_but_hides_question() -> None:
    contract = TaskContract(
        "sequence",
        [EvidenceSlot("S1", "remove the white cloth from the frame")],
        ["visual"],
    )
    prompt = build_observer_prompt(
        "SECRET FULL QUESTION",
        contract,
        action=PlannedAction("observe", "N1", "event_verify", "S1"),
        nodes=[SceneNode("N1", 10.0, 12.0, 2, parent_id="N0")],
        frame_lines=["F000010 @ 10.000s", "F000011 @ 11.000s"],
        subtitles="SECRET SUBTITLE",
    )

    assert "fine-grained visual event validator" in prompt
    assert "remove the white cloth from the frame" in prompt
    assert "SECRET FULL QUESTION" not in prompt
    assert "SECRET SUBTITLE" not in prompt
    assert "one before the transition and one after it" in prompt


def test_fine_event_sampling_includes_localized_node_boundaries(tmp_path: Path) -> None:
    cached = make_cached_video(tmp_path)
    agent = ActiveTreeVideoAgent(
        FakeModel([]),
        config=ActiveTreeConfig(cache_dir=str(tmp_path), detail_frames=4),
    )

    frames, _ = agent._observation_frames(
        cached,
        [SceneNode("N1", 10.2, 20.7, 2, parent_id="N0")],
        "event_verify",
    )

    assert len(frames) == 2
    assert frames[0].id == cached.nearest_frame(10.2).id
    assert frames[-1].id == cached.nearest_frame(19.7).id


def test_sequence_uses_matching_visual_breadth_clue_as_observation_proposal() -> None:
    ledger = EvidenceLedger()
    ledger.add(
        [
            AtomicEvidence(
                "EV0001",
                "N2",
                ("S1",),
                90.0,
                120.0,
                "visual",
                "The magician removes the white cloth from the photo frame",
                (),
                (),
                ("F000099", "F000100"),
                (),
                "breadth",
            )
        ]
    )

    actions = ActiveTreeVideoAgent._sequence_breadth_actions(
        TaskContract(
            "sequence",
            [EvidenceSlot("S1", "The magician removes the white cloth from the photo frame")],
            ["visual"],
        ),
        ledger,
    )

    assert len(actions) == 1
    assert actions[0].node_id == "N2"
    assert actions[0].mode == "motion"


def test_sequence_sweep_chooses_earliest_uncovered_root_branch() -> None:
    tree = SceneTree(
        "ROOT",
        {
            "ROOT": SceneNode(
                "ROOT",
                0.0,
                120.0,
                0,
                child_ids=["N0", "N1", "N2", "N3"],
            ),
            **{
                f"N{index}": SceneNode(
                    f"N{index}",
                    float(index * 30),
                    float((index + 1) * 30),
                    1,
                    parent_id="ROOT",
                )
                for index in range(4)
            },
        },
    )
    ledger = EvidenceLedger()
    for index, slot_id in enumerate(("S1", "S2"), start=1):
        ledger.add(
            [
                AtomicEvidence(
                    f"EV{index:04d}",
                    f"N{index - 1}",
                    (slot_id,),
                    float(index),
                    float(index),
                    "visual",
                    f"event {slot_id}",
                    (),
                    (),
                    (f"F{index}",),
                    (),
                    "motion",
                )
            ]
        )

    action = ActiveTreeVideoAgent._sequence_sweep_action(
        tree,
        TaskContract(
            "sequence",
            [
                EvidenceSlot("S1", "event one"),
                EvidenceSlot("S2", "event two"),
                EvidenceSlot("S3", "event three"),
            ],
            ["visual"],
        ),
        ledger,
        [
            PlannedAction("observe", "N3", "overview").signature(),
            PlannedAction("observe", "N0", "motion", "S1").signature(),
            PlannedAction("observe", "N1", "motion", "S2").signature(),
            PlannedAction("observe", "N2", "motion", "S3").signature(),
        ],
    )

    assert action is not None
    assert action.node_id == "N3"
    assert action.slot_id == "S3"


def test_subtitle_evidence_requires_grounded_quote_and_discards_interpretation(
    tmp_path: Path,
) -> None:
    cached = make_cached_video(tmp_path)
    tree = SceneTreeBuilder(ActiveTreeConfig(cache_dir=str(tmp_path))).build(cached)
    node = tree.root
    grounded_quote = "I sort of already asked Chandler to be my best man"
    decision = ObserverDecision(
        facts=(
            ObserverFact(
                node_id=node.id,
                slot_ids=("S1",),
                start_seconds=13.0,
                end_seconds=16.0,
                modality="subtitle",
                fact="They are deciding who should be the best man.",
                supports_option_ids=("O2",),
                refutes_option_ids=("O1",),
                source_frame_ids=(),
                subtitle_refs=(grounded_quote,),
            ),
            ObserverFact(
                node_id=node.id,
                slot_ids=("S1",),
                start_seconds=1.0,
                end_seconds=2.0,
                modality="subtitle",
                fact="They are arguing about getting married a thousand times.",
                supports_option_ids=("O4",),
                refutes_option_ids=(),
                source_frame_ids=(),
                subtitle_refs=("They are getting married a thousand times",),
            ),
        ),
        missing_evidence="",
    )

    evidence = ActiveTreeVideoAgent._materialize_evidence(
        decision,
        tree,
        TaskContract("local", [EvidenceSlot("S1", "dialogue subject")], ["subtitle"]),
        [],
        mode="subtitle",
        default_slot_id="S1",
        subtitles_available=True,
        allowed_subtitle_text=(
            "[13.0s-16.0s] I sort of already asked Chandler to be my best man."
        ),
        allowed_frame_times={},
        next_index=1,
    )

    assert len(evidence) == 1
    assert evidence[0].fact == grounded_quote
    assert evidence[0].subtitle_refs == (grounded_quote,)
    assert evidence[0].supports_option_ids == ()
    assert evidence[0].refutes_option_ids == ()


def test_subtitle_reference_selection_keeps_relevant_late_lines() -> None:
    references = (
        "[0.5s-5.2s] hey guys here is the ring",
        "[5.2s-12.0s] yes yes a thousand times yes",
        "[9.7s-13.0s] any ideas for the bachelor party",
        "[16.5s-19.5s] decide who your best man will be",
    )

    selected = ActiveTreeVideoAgent._select_grounded_subtitle_refs(
        references,
        "\n".join(references),
        relevance_text="The discussion includes the bachelor party and best man choice.",
        limit=2,
    )

    assert selected == (references[2], references[3])


def test_sequence_visual_fact_is_rebound_by_blind_caption_not_requested_slot(
    tmp_path: Path,
) -> None:
    cached = make_cached_video(tmp_path)
    tree = SceneTreeBuilder(ActiveTreeConfig(cache_dir=str(tmp_path))).build(cached)
    evidence = ActiveTreeVideoAgent._materialize_evidence(
        ObserverDecision(
            (
                ObserverFact(
                    tree.root_id,
                    ("S3",),
                    0.0,
                    1.0,
                    "visual",
                    "The white cloth is removed from the photo frame",
                    (),
                    (),
                    ("F000010", "F000011"),
                    (),
                ),
            ),
            "",
        ),
        tree,
        TaskContract(
            "sequence",
            [
                EvidenceSlot("S1", "Take the white cloth away from the photo frame"),
                EvidenceSlot("S2", "Lead a self introduction"),
                EvidenceSlot("S3", "Produce a feather"),
            ],
            ["visual", "subtitle"],
        ),
        [],
        mode="motion",
        default_slot_id="S3",
        subtitles_available=False,
        allowed_subtitle_text="",
        allowed_frame_times={"F000010": 10.0, "F000011": 11.0},
        next_index=1,
    )

    assert len(evidence) == 1
    assert evidence[0].slot_ids == ("S1",)
    assert evidence[0].start_seconds == 10.0
    assert evidence[0].end_seconds == 11.0
    assert evidence[0].observation_mode == "motion"


def test_sequence_static_state_is_routing_clue_not_active_event(
    tmp_path: Path,
) -> None:
    cached = make_cached_video(tmp_path)
    tree = SceneTreeBuilder(ActiveTreeConfig(cache_dir=str(tmp_path))).build(cached)
    ledger = EvidenceLedger()
    evidence = ActiveTreeVideoAgent._materialize_evidence(
        ObserverDecision(
            (
                ObserverFact(
                    tree.root_id,
                    (),
                    0.0,
                    1.0,
                    "visual",
                    "A man holds a photo frame covered by a white cloth",
                    (),
                    (),
                    ("F000010", "F000011"),
                    (),
                ),
            ),
            "",
        ),
        tree,
        TaskContract(
            "sequence",
            [EvidenceSlot("S1", "Take the white cloth away from the photo frame")],
            ["visual"],
        ),
        [],
        mode="motion",
        default_slot_id="S1",
        subtitles_available=False,
        allowed_subtitle_text="",
        allowed_frame_times={"F000010": 10.0, "F000011": 11.0},
        next_index=1,
    )
    ledger.add(evidence)

    assert len(ledger.routing_items) == 1
    assert ledger.active_items == ()
    assert ledger.missing_slot_ids(
        TaskContract(
            "sequence",
            [EvidenceSlot("S1", "Take the white cloth away from the photo frame")],
            ["visual"],
        )
    ) == ["S1"]

    action = ActiveTreeVideoAgent._sequence_refinement_action(
        tree,
        TaskContract(
            "sequence",
            [EvidenceSlot("S1", "Take the white cloth away from the photo frame")],
            ["visual"],
        ),
        ledger,
        [],
    )
    assert action is not None
    assert action.node_id in tree.root.child_ids
    assert action.mode == "event_verify"

    breadth_ledger = EvidenceLedger()
    breadth_ledger.add(
        ActiveTreeVideoAgent._materialize_evidence(
            ObserverDecision(
                (
                    ObserverFact(
                        tree.root_id,
                        (),
                        0.0,
                        1.0,
                        "visual",
                        "A man holds a photo frame covered by a white cloth",
                        (),
                        (),
                        ("F000010",),
                        (),
                    ),
                ),
                "",
            ),
            tree,
            TaskContract(
                "sequence",
                [EvidenceSlot("S1", "Take the white cloth away from the photo frame")],
                ["visual"],
            ),
            [],
            mode="breadth",
            default_slot_id=None,
            subtitles_available=False,
            allowed_subtitle_text="",
            allowed_frame_times={"F000010": 10.0},
            next_index=1,
        )
    )
    assert breadth_ledger.items[0].observation_mode == "breadth"
    assert breadth_ledger.routing_items == ()


def test_sequence_before_after_disappearance_grounds_removal_event(
    tmp_path: Path,
) -> None:
    cached = make_cached_video(tmp_path)
    tree = SceneTreeBuilder(ActiveTreeConfig(cache_dir=str(tmp_path))).build(cached)
    evidence = ActiveTreeVideoAgent._materialize_evidence(
        ObserverDecision(
            (
                ObserverFact(
                    tree.root_id,
                    ("S1",),
                    10.0,
                    19.0,
                    "visual",
                    (
                        "The white cloth covers the photo frame in the first frame; "
                        "it is not visible in the last frame"
                    ),
                    (),
                    (),
                    ("F000010",),
                    (),
                ),
            ),
            "",
        ),
        tree,
        TaskContract(
            "sequence",
            [EvidenceSlot("S1", "Take the white cloth away from the photo frame")],
            ["visual"],
        ),
        [],
        mode="event_verify",
        default_slot_id="S1",
        subtitles_available=False,
        allowed_subtitle_text="",
        allowed_frame_times={"F000010": 10.0, "F000019": 19.0},
        next_index=1,
    )

    assert len(evidence) == 1
    assert evidence[0].observation_mode == "event_verify"
    assert evidence[0].slot_ids == ("S1",)
    assert evidence[0].source_frame_ids == ("F000010", "F000019")


def test_exact_alignment_finds_option_unique_phrase_without_using_gold_answer() -> None:
    options = canonicalize_options(
        [
            "Whether the person in blue will get married.",
            "Who should be chosen as the best man.",
            "Whether the suited man should get married.",
            "How many times the suited man is marrying.",
        ]
    )
    evidence = (
        AtomicEvidence(
            "EV0001",
            "N1",
            ("S1",),
            65.0,
            78.0,
            "subtitle",
            "I want you to be my best man",
            (),
            (),
            (),
            ("I want you to be my best man",),
            "subtitle",
        ),
    )

    alignment = align_evidence_to_options(
        options,
        evidence,
        min_phrase_tokens=2,
        min_margin=3.0,
    )

    assert alignment.option_id == "O2"
    assert alignment.evidence_id == "EV0001"
    assert alignment.matched_phrase == "best man"
    assert alignment.confident is True


def test_alignment_abstains_without_option_unique_phrase() -> None:
    alignment = align_evidence_to_options(
        canonicalize_options(["opens the door", "closes the door"]),
        (
            AtomicEvidence(
                "EV0001",
                "N1",
                ("S1",),
                0.0,
                1.0,
                "visual",
                "a person stands near the door",
                (),
                (),
                ("F000001",),
                (),
                "inspect",
            ),
        ),
        min_phrase_tokens=2,
        min_margin=3.0,
    )

    assert alignment.confident is False


def test_option_aware_subtitle_retrieval_proposes_matching_root_branch(
    tmp_path: Path,
) -> None:
    tree = SceneTree(
        "ROOT",
        {
            "ROOT": SceneNode("ROOT", 0.0, 100.0, 0, child_ids=["N1", "N2"]),
            "N1": SceneNode("N1", 0.0, 50.0, 1, parent_id="ROOT"),
            "N2": SceneNode("N2", 50.0, 100.0, 1, parent_id="ROOT"),
        },
    )
    subtitles = SubtitleTrack(
        [
            SubtitleCue(3.0, 5.0, "Here is the ring"),
            SubtitleCue(70.0, 74.0, "I want you to be my best man"),
        ]
    )
    agent = ActiveTreeVideoAgent(
        FakeModel([]),
        config=ActiveTreeConfig(cache_dir=str(tmp_path)),
    )

    action, alignment = agent._subtitle_retrieval_action(
        canonicalize_options(
            [
                "Whether someone gets married.",
                "Who should be the best man.",
                "Whether a wedding should happen.",
                "How many weddings happen.",
            ]
        ),
        TaskContract("local", [EvidenceSlot("S1", "spoken topic")], ["subtitle"]),
        subtitles,
        tree,
    )

    assert action is not None
    assert action.node_id == "N2"
    assert action.mode == "subtitle"
    assert alignment.option_id == "O2"


def test_temporal_compositor_maps_grounded_slot_times_to_symbolic_option() -> None:
    contract = TaskContract(
        "sequence",
        [
            EvidenceSlot("S1", "remove cloth"),
            EvidenceSlot("S2", "self introduction"),
            EvidenceSlot("S3", "produce feather"),
        ],
        ["visual", "subtitle"],
    )
    ledger = EvidenceLedger()
    for index, (slot_id, timestamp) in enumerate(
        (("S1", 101.0), ("S2", 65.0), ("S3", 141.0)),
        start=1,
    ):
        ledger.add(
            [
                AtomicEvidence(
                    f"EV{index:04d}",
                    f"N{index}",
                    (slot_id,),
                    timestamp,
                    timestamp,
                    "visual",
                    slot_id,
                    (),
                    (),
                    (f"F{index}",),
                    (),
                    "motion",
                )
            ]
        )

    composition = compose_temporal_option(
        canonicalize_options(["(a)(b)(c).", "(b)(c)(a).", "(c)(b)(a).", "(b)(a)(c)."]),
        contract,
        ledger,
        min_gap_seconds=0.5,
    )

    assert composition.confident is True
    assert composition.option_id == "O4"
    assert composition.option_pattern == "(b)(a)(c)"
    assert composition.slot_order == ("S2", "S1", "S3")


def test_observer_rejects_nested_object_recovery_without_top_level_facts() -> None:
    with pytest.raises(ProtocolError):
        parse_observer(
            '{"node_id":"N1","fact":"partial nested object"}',
            valid_node_ids={"N1"},
            valid_slot_ids={"S1"},
            valid_option_ids=set(),
            valid_frame_ids=set(),
        )


def test_observer_normalizes_motion_mode_and_bounds_visual_citations() -> None:
    decision = parse_observer(
        '{"facts":[{"node_id":"N1","slot_ids":["S1"],'
        '"start_seconds":0,"end_seconds":10,"modality":"motion",'
        '"fact":"the target action occurs","supports_option_ids":[],'
        '"refutes_option_ids":[],"source_frame_ids":["F1","F2","F3"],'
        '"subtitle_refs":["not allowed"]}],"missing_evidence":""}',
        valid_node_ids={"N1"},
        valid_slot_ids={"S1"},
        valid_option_ids=set(),
        valid_frame_ids={"F1", "F2", "F3"},
    )

    assert decision.facts[0].modality == "visual"
    assert decision.facts[0].source_frame_ids == ("F1", "F2")
    assert decision.facts[0].subtitle_refs == ()

    endpoint_decision = parse_observer(
        '{"facts":[{"node_id":"N1","slot_ids":["S1"],'
        '"start_seconds":0,"end_seconds":10,"modality":"visual",'
        '"fact":"a transition","supports_option_ids":[],'
        '"refutes_option_ids":[],"source_frame_ids":["F1","F2","F3"],'
        '"subtitle_refs":[]}]}',
        valid_node_ids={"N1"},
        valid_slot_ids={"S1"},
        valid_option_ids=set(),
        valid_frame_ids={"F1", "F2", "F3"},
        prefer_frame_endpoints=True,
    )
    assert endpoint_decision.facts[0].source_frame_ids == ("F1", "F3")


def test_planner_recovers_a_complete_action_from_truncated_outer_json() -> None:
    actions = parse_planner(
        '{"actions":[{"kind":"observe","node_id":"N1","mode":"subtitle",'
        '"slot_id":"S1","compare_node_ids":[],"expected_new_evidence":"quote"},'
    )

    assert len(actions) == 1
    assert actions[0].node_id == "N1"


def test_verifier_cannot_claim_sufficiency_without_discriminative_audit_fields() -> None:
    decision = parse_verification(
        '{"candidate_option_text":"O1: one","decisive_evidence_id":"EV9999",'
        '"strongest_alternative_text":"O2: two","alternative_refuted":true,'
        '"missing_slot_ids":[],"counterevidence":"","reason":"brief"}',
        options=canonicalize_options(["one", "two"]),
        valid_evidence_ids={"EV0001"},
    )

    assert decision.candidate_option_id == "O1"
    assert decision.sufficient is False
    assert decision.citations_valid is False
