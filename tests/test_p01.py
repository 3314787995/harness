from __future__ import annotations

import json
import re
from collections.abc import Callable, Sequence
from itertools import pairwise
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from qwen3vl_agent.cli import build_parser
from qwen3vl_agent.coarse_to_fine.cache import CachedVideo
from qwen3vl_agent.coarse_to_fine.types import FrameRef
from qwen3vl_agent.models.base import BaseVideoModel, ModelOutput
from qwen3vl_agent.p01 import P01Config, P01Request, P01VideoAgent, TimeSpan
from qwen3vl_agent.p01.control import (
    interval_chunks,
    parse_question_interval,
    reconcile_interval,
    uniform_timestamps,
)
from qwen3vl_agent.p01.media import (
    FrameMetrics,
    P01IndexBuilder,
    P01VideoIndex,
    ShotRef,
    SourceFrameStore,
)
from qwen3vl_agent.p01.prompts import (
    build_hypothesis_compiler_prompt,
    build_locator_prompt,
    build_observation_compiler_prompt,
    build_scout_prompt,
    build_verifier_prompt,
    compile_decision_spec,
    parse_decision_spec,
    parse_locator,
    parse_rescue_locator,
    parse_scout,
    parse_verifier,
)
from qwen3vl_agent.p01.types import (
    BoundingBox,
    CanonicalOption,
    ClaimTest,
    ContractViolation,
    CoverageManifest,
    EvidencePacket,
    ObservationSlot,
    ObservationSpec,
    ProtocolError,
    StaticFact,
    TextFact,
)

Response = str | Callable[[str, list[dict[str, Any]]], str]


class RoleFakeModel(BaseVideoModel):
    def __init__(self, handlers: dict[str, Response | list[Response]]) -> None:
        super().__init__("fake")
        self.handlers = handlers
        self.calls: list[dict[str, Any]] = []

    def load(self) -> None:
        self._loaded = True

    def unload(self) -> None:
        self._loaded = False

    def generate(
        self,
        messages: list[dict[str, Any]],
        *,
        videos: list[str | list[str]] | None = None,
        images: list[str] | None = None,
        **kwargs: Any,
    ) -> ModelOutput:
        del videos, images
        prompt = _prompt_text(messages)
        role = _prompt_role(prompt)
        self.calls.append({"role": role, "prompt": prompt, "messages": messages, "kwargs": kwargs})
        response = self.handlers[role]
        if isinstance(response, list):
            if not response:
                raise AssertionError(f"No fake response left for {role}")
            response = response.pop(0)
        text = response(prompt, messages) if callable(response) else response
        return ModelOutput(
            text=text,
            metadata={"input_tokens": 10, "output_tokens": 5, "latency_seconds": 0.01},
        )


class FakeIndexBuilder:
    def __init__(self, index: P01VideoIndex) -> None:
        self.index = index

    def prepare(self, video_path: str | Path) -> P01VideoIndex:
        del video_path
        return self.index


class FakeSourceStore:
    def __init__(self, image_path: Path, *, only_first: bool = False) -> None:
        self.image_path = image_path
        self.only_first = only_first
        self.extract_calls: list[dict[str, Any]] = []
        self.crop_calls: list[tuple[FrameRef, BoundingBox]] = []

    def extract(
        self,
        video_path: str | Path,
        timestamps: Sequence[float],
        *,
        purpose: str,
        max_side: int | None = None,
    ) -> tuple[FrameRef, ...]:
        del video_path, max_side
        values = list(timestamps)
        if self.only_first and values:
            values = values[:1]
        self.extract_calls.append({"purpose": purpose, "timestamps": values})
        result: list[FrameRef] = []
        seen: set[str] = set()
        for timestamp in values:
            frame_id = f"SRC-{round(timestamp * 1000):09d}"
            if frame_id in seen:
                continue
            seen.add(frame_id)
            result.append(FrameRef(frame_id, float(timestamp), str(self.image_path)))
        return tuple(result)

    def crop(
        self,
        frame: FrameRef,
        bbox: BoundingBox,
        *,
        padding_fraction: float = 0.1,
    ) -> FrameRef:
        del padding_fraction
        self.crop_calls.append((frame, bbox))
        return FrameRef(f"CROP-{frame.id}", frame.timestamp_seconds, str(self.image_path))


def _prompt_text(messages: list[dict[str, Any]]) -> str:
    content = messages[-1]["content"]
    if isinstance(content, str):
        return content
    return "\n".join(
        str(part.get("text", ""))
        for part in content
        if isinstance(part, dict) and part.get("type") == "text"
    )


def _prompt_role(prompt: str) -> str:
    markers = (
        ("Repair a malformed structured response", "protocol_repair"),
        ("compile a question-only observation contract", "observation_compiler"),
        ("locate visible evidence in a temporal contact sheet", "locator"),
        ("question-only local evidence observer (interval_scout)", "interval_scout"),
        ("question-only local evidence observer (candidate_scout)", "candidate_scout"),
        ("label-free visual discriminants", "hypothesis_compiler"),
        ("bounded global rescue locator", "rescue_locator"),
        ("discriminator-aware rescue observer (rescue_scout)", "rescue_scout"),
        ("one allowed targeted re-observation", "refinement_extractor"),
        ("blinded claim verifier", "verifier"),
        ("initial_decision local multiple-choice decision pass", "initial_decision"),
        ("final_decision local multiple-choice decision pass", "final_decision"),
        ("Write a concise best-effort answer", "answer_composer"),
    )
    for marker, role in markers:
        if marker in prompt:
            return role
    raise AssertionError(f"Unknown P01 prompt: {prompt[:100]!r}")


def _frame_map(prompt: str) -> list[tuple[str, float]]:
    return [
        (match.group(1), float(match.group(2)))
        for match in re.finditer(r"- visual \d+: (\S+) @ ([0-9.]+)s", prompt)
    ]


def _observation(
    mode: str = "static_visual",
    *,
    answer_mode: str = "multiple_choice",
    coverage: str = "point",
) -> str:
    return json.dumps(
        {
            "answer_mode": answer_mode,
            "primary_mode": mode,
            "required_slots": [
                {"description": "observable target", "required": True, "value_type": "fact"}
            ],
            "target_entities": ["target"],
            "target_actions": [],
            "target_attributes": ["requested property"],
            "target_relations": [],
            "detail_requests": [],
            "temporal_hint": None,
            "coverage_requirement": coverage,
            "output_language": "same_as_question",
        }
    )


def _decision() -> str:
    return json.dumps(
        {
            "claim_tests": [
                {
                    "claim_id": "C1",
                    "statement": "the target is red",
                    "slot_ids": ["S1"],
                    "predicate": "equals",
                    "expected_value": "red",
                },
                {
                    "claim_id": "C2",
                    "statement": "the target is blue",
                    "slot_ids": ["S1"],
                    "predicate": "equals",
                    "expected_value": "blue",
                },
            ],
            "option_rules": [
                {"option_id": "O1", "all_of": ["C1"], "none_of": []},
                {"option_id": "O2", "all_of": ["C2"], "none_of": []},
            ],
            "cannot_determine_option_id": None,
        }
    )


def _choice_decision(
    selected: str = "O1",
    *,
    unresolved: Sequence[str] = (),
    support: int = 3,
) -> Callable[[str, list[dict[str, Any]]], str]:
    def respond(prompt: str, _messages: list[dict[str, Any]]) -> str:
        frame_ids = [item[0] for item in _frame_map(prompt)]
        citation = frame_ids[:1]
        other = "O2" if selected == "O1" else "O1"
        return json.dumps(
            {
                "selected_option_id": selected,
                "option_assessments": [
                    {
                        "option_id": selected,
                        "support_score": support,
                        "contradiction_score": 0,
                        "discriminant_ids": ["C1" if selected == "O1" else "C2"],
                        "evidence_fact_ids": [],
                        "source_frame_ids": citation,
                        "reason": "visible local support",
                    },
                    {
                        "option_id": other,
                        "support_score": 0,
                        "contradiction_score": 3 if support >= 2 else 0,
                        "discriminant_ids": ["C2" if other == "O2" else "C1"],
                        "evidence_fact_ids": [],
                        "source_frame_ids": citation,
                        "reason": "less consistent",
                    },
                ],
                "resolved_discriminant_ids": [
                    claim_id for claim_id in ("C1", "C2") if claim_id not in unresolved
                ],
                "unresolved_discriminant_ids": list(unresolved),
                "reason": "mandatory local decision",
            }
        )

    return respond


def _rescue_locator(
    prompt: str,
    _messages: list[dict[str, Any]],
) -> str:
    frame_id = _frame_map(prompt)[0][0]
    return json.dumps(
        {"candidates": [{"frame_id": frame_id, "visible_anchor": "weak rescue anchor"}]}
    )


def _invisible_scout(_prompt: str, _messages: list[dict[str, Any]]) -> str:
    return json.dumps(
        {
            "target_visible": False,
            "supporting_start": None,
            "supporting_end": None,
            "visible_anchor": "",
            "facts": [],
            "missing_slot_ids": ["S1"],
            "conflicts": [],
        }
    )


def _verifier(
    verdict_c1: str = "entailed",
    verdict_c2: str = "contradicted",
    *,
    citations: int = 1,
) -> Callable[[str, list[dict[str, Any]]], str]:
    def respond(prompt: str, _messages: list[dict[str, Any]]) -> str:
        frame_ids = [item[0] for item in _frame_map(prompt)]
        cited = frame_ids[:citations]
        return json.dumps(
            {
                "verdicts": [
                    {
                        "claim_id": "C1",
                        "verdict": verdict_c1,
                        "evidence_fact_ids": [],
                        "verification_frame_ids": (
                            cited if verdict_c1 != "not_established" else []
                        ),
                        "reason": "independent view",
                    },
                    {
                        "claim_id": "C2",
                        "verdict": verdict_c2,
                        "evidence_fact_ids": [],
                        "verification_frame_ids": (
                            cited if verdict_c2 != "not_established" else []
                        ),
                        "reason": "independent view",
                    },
                ]
            }
        )

    return respond


def _static_scout(
    *,
    visibility: str = "clear",
    value: str = "red",
) -> Callable[[str, list[dict[str, Any]]], str]:
    def respond(prompt: str, _messages: list[dict[str, Any]]) -> str:
        frame_id, timestamp = _frame_map(prompt)[0]
        return json.dumps(
            {
                "target_visible": True,
                "supporting_start": timestamp,
                "supporting_end": timestamp + 0.1,
                "visible_anchor": "target object",
                "facts": [
                    {
                        "kind": "static",
                        "slot_ids": ["S1"],
                        "start_seconds": timestamp,
                        "end_seconds": timestamp,
                        "visibility": visibility,
                        "statement": f"the target is {value}",
                        "source_frame_ids": [frame_id],
                        "entity": "target",
                        "attribute": "color",
                        "relation": "",
                        "value": value,
                    }
                ],
                "missing_slot_ids": [] if visibility == "clear" else ["S1"],
                "conflicts": [],
            }
        )

    return respond


def _event_scout(visibility: str) -> Callable[[str, list[dict[str, Any]]], str]:
    def respond(prompt: str, _messages: list[dict[str, Any]]) -> str:
        frame_id, timestamp = _frame_map(prompt)[0]
        return json.dumps(
            {
                "target_visible": True,
                "supporting_start": timestamp,
                "supporting_end": timestamp + 0.25,
                "visible_anchor": "visible action",
                "facts": [
                    {
                        "kind": "event",
                        "slot_ids": ["S1"],
                        "start_seconds": timestamp,
                        "end_seconds": timestamp,
                        "visibility": visibility,
                        "statement": "the person lifts the cup",
                        "source_frame_ids": [frame_id],
                        "subject": "person",
                        "initial_state": "cup on table",
                        "action": "lifts",
                        "object": "cup",
                        "target": "",
                        "result": "cup raised",
                        "order": 1,
                    }
                ],
                "missing_slot_ids": [] if visibility == "clear" else ["S1"],
                "conflicts": [],
            }
        )

    return respond


def _ocr_scout(prompt: str, _messages: list[dict[str, Any]]) -> str:
    frame_map = [item for item in _frame_map(prompt) if not item[0].startswith("CROP-")]
    first_id, timestamp = frame_map[0]
    consensus = [item[0] for item in frame_map[:3]]
    return json.dumps(
        {
            "target_visible": True,
            "supporting_start": timestamp,
            "supporting_end": timestamp + 0.25,
            "visible_anchor": "text panel",
            "facts": [
                {
                    "kind": "text",
                    "slot_ids": ["S1"],
                    "start_seconds": timestamp,
                    "end_seconds": timestamp,
                    "visibility": "clear",
                    "statement": "the panel reads red",
                    "source_frame_ids": consensus,
                    "exact_text": "RED",
                    "uncertain_characters": "",
                    "bbox": {
                        "frame_id": first_id,
                        "x1": 100,
                        "y1": 200,
                        "x2": 800,
                        "y2": 700,
                    },
                    "consensus_frame_ids": consensus,
                }
            ],
            "missing_slot_ids": [],
            "conflicts": [],
        }
    )


def _not_established_verifier(
    prompt: str,
    _messages: list[dict[str, Any]],
) -> str:
    claim_ids = re.findall(r"^- (C\d+|VF\d+):", prompt, flags=re.MULTILINE)
    return json.dumps(
        {
            "verdicts": [
                {
                    "claim_id": claim_id,
                    "verdict": "not_established",
                    "evidence_fact_ids": [],
                    "verification_frame_ids": [],
                    "reason": "still occluded",
                }
                for claim_id in claim_ids
            ]
        }
    )


def _make_index(
    tmp_path: Path,
    config: P01Config,
    *,
    duration: float = 10.0,
    shot_spans: Sequence[tuple[float, float]] | None = None,
) -> tuple[P01VideoIndex, Path]:
    image_path = tmp_path / "frame.jpg"
    Image.new("RGB", (64, 48), "red").save(image_path)
    times = [min(duration, float(index)) for index in range(int(duration) + 1)]
    if times[-1] != duration:
        times.append(duration)
    frames = tuple(
        FrameRef(f"NAV-{index:04d}", timestamp, str(image_path))
        for index, timestamp in enumerate(dict.fromkeys(times))
    )
    cached = CachedVideo(
        source_path=str(tmp_path / "video.mp4"),
        cache_dir=str(tmp_path),
        duration_seconds=duration,
        source_fps=30.0,
        width=64,
        height=48,
        sample_fps=config.index_fps,
        frames=frames,
        cache_hit=False,
    )
    metrics = tuple(
        FrameMetrics(
            frame.id,
            frame.timestamp_seconds,
            change_score=0.05,
            motion_score=frame.timestamp_seconds / max(duration, 1),
            clarity_score=0.8,
            text_score=0.4,
            is_black=False,
            is_duplicate=False,
        )
        for frame in frames
    )
    spans = shot_spans or [(0.0, duration)]
    shots = tuple(
        ShotRef(
            f"S{index:04d}",
            TimeSpan(start, end, source="shot"),
            tuple(frame.id for frame in frames if start <= frame.timestamp_seconds <= end),
        )
        for index, (start, end) in enumerate(spans)
    )
    nodes, root_id = P01IndexBuilder(config)._build_tree(shots)
    return P01VideoIndex(cached, metrics, shots, nodes, root_id), image_path


def _run_agent(
    model: RoleFakeModel,
    index: P01VideoIndex,
    image_path: Path,
    request: P01Request,
    *,
    store: FakeSourceStore | None = None,
    config: P01Config | None = None,
) -> tuple[Any, FakeSourceStore]:
    source_store = store or FakeSourceStore(image_path)
    agent = P01VideoAgent(
        model,
        config=config or P01Config(cache_dir=str(image_path.parent / "cache")),
        index_builder=FakeIndexBuilder(index),
        source_store=source_store,
    )
    agent.load()
    try:
        return agent.solve(request), source_store
    finally:
        agent.unload()


def test_observation_prompt_is_choice_blind() -> None:
    prompt = build_observation_compiler_prompt(
        "What color is the object?", answer_mode="multiple_choice"
    )
    assert "scarlet candidate" not in prompt
    assert "answer choices" in prompt
    assert "how a visible result is produced" in prompt
    assert "localization anchor" in prompt


def test_locator_and_scout_prompts_do_not_treat_answer_uncertainty_as_absence() -> None:
    contract = ObservationSpec(
        answer_mode="multiple_choice",
        primary_mode="dynamic_action",
        required_slots=(ObservationSlot("S1", "observable action", True, "event"),),
        coverage_requirement="sequence",
    )
    span = TimeSpan(1, 3)
    locator = build_locator_prompt(
        "What does the vlogger do?",
        contract,
        (("S0001", span),),
        max_candidates=1,
    )
    scout = build_scout_prompt(
        "What does the vlogger do?",
        contract,
        "S0001",
        span,
        (),
    )

    assert "does not mean that the final answer is already" in scout
    assert "visible hands" in scout
    assert "never timestamps" in scout
    assert "1-4 decisive frame IDs" in scout
    assert "set target_visible=true" in scout
    assert "Use an empty candidates list only" in locator
    assert "with no time range" in locator


def test_scout_resolves_unambiguous_timestamp_citations_to_frame_ids() -> None:
    frames = (
        FrameRef("SRC-A", 10.2102, "a.jpg"),
        FrameRef("SRC-B", 14.9153, "b.jpg"),
    )
    decision = parse_scout(
        json.dumps(
            {
                "target_visible": True,
                "supporting_start": 10.21,
                "supporting_end": 14.915,
                "visible_anchor": "visible fire",
                "facts": [
                    {
                        "kind": "static",
                        "slot_ids": ["S1"],
                        "start_seconds": "10.21s",
                        "end_seconds": "14.915s",
                        "visibility": "clear",
                        "statement": "Smoke rises from the fire.",
                        "source_frame_ids": [10.21, 14.915],
                        "entity": "smoke",
                        "attribute": "source",
                        "relation": "rises from",
                        "value": "fire",
                    }
                ],
                "missing_slot_ids": [],
                "conflicts": [],
            }
        ),
        candidate_span=TimeSpan(10, 15),
        valid_slot_ids={"S1"},
        frames=frames,
        view_id="V1",
        fact_prefix="T",
    )
    assert decision.facts[0].source_frame_ids == ("SRC-A", "SRC-B")


def test_rescue_locator_accepts_one_unambiguous_decorated_frame_id() -> None:
    frames = (
        FrameRef("F000007", 3.5, "a.jpg"),
        FrameRef("F000011", 5.5, "b.jpg"),
    )
    decision = parse_rescue_locator(
        json.dumps(
            {
                "candidates": [
                    {
                        "frame_id": "F000007 @ 3.500s",
                        "visible_anchor": "visible target",
                    }
                ]
            }
        ),
        frames=frames,
        max_candidates=2,
    )
    assert decision.frame_ids == ("F000007",)


def test_ocr_verifier_prompt_requires_two_independent_adjacent_citations() -> None:
    frames = (FrameRef("SRC-A", 1.0, "a.jpg"), FrameRef("SRC-B", 1.25, "b.jpg"))
    fact = TextFact(
        fact_id="F1",
        slot_ids=("S1",),
        start_seconds=1.0,
        end_seconds=1.25,
        visibility="clear",
        statement="OPEN",
        source_frame_ids=("SRC-A", "SRC-B"),
        view_id="V1",
        exact_text="OPEN",
        bbox=BoundingBox("SRC-A", 100, 100, 900, 900),
        consensus_frame_ids=("SRC-A", "SRC-B"),
    )
    prompt = build_verifier_prompt(
        "What text is shown?",
        (ClaimTest("C1", "The text is OPEN", ("S1",), "equals", "OPEN"),),
        (fact,),
        frames,
        TimeSpan(1.0, 1.5),
    )
    assert "exactly two distinct adjacent verification frame IDs" in prompt
    assert "One readable frame is insufficient" in prompt


def test_verifier_resolves_unambiguous_timestamp_citations_to_frame_ids() -> None:
    frames = (FrameRef("SRC-A", 9.8102, "a.jpg"),)
    verdicts = parse_verifier(
        json.dumps(
            {
                "verdicts": [
                    {
                        "claim_id": "C1",
                        "verdict": "entailed",
                        "evidence_fact_ids": [],
                        "verification_frame_ids": ["9.810"],
                        "reason": "visible",
                    }
                ]
            }
        ),
        valid_claim_ids={"C1"},
        valid_fact_ids=set(),
        frames=frames,
    )
    assert verdicts[0].verification_frame_ids == ("SRC-A",)


def test_verifier_resolves_visual_indices_and_downgrades_invalid_citations() -> None:
    frames = (FrameRef("SRC-A", 1.0, "a.jpg"), FrameRef("SRC-B", 2.0, "b.jpg"))
    verdicts = parse_verifier(
        json.dumps(
            {
                "verdicts": [
                    {
                        "claim_id": "C1",
                        "verdict": "entailed",
                        "evidence_fact_ids": [],
                        "verification_frame_ids": ["visual 2"],
                        "reason": "visible",
                    },
                    {
                        "claim_id": "C2",
                        "verdict": "contradicted",
                        "evidence_fact_ids": [],
                        "verification_frame_ids": ["unknown"],
                        "reason": "not visible",
                    },
                ]
            }
        ),
        valid_claim_ids={"C1", "C2"},
        valid_fact_ids=set(),
        frames=frames,
    )
    assert verdicts[0].verification_frame_ids == ("SRC-B",)
    assert verdicts[0].verdict == "entailed"
    assert verdicts[1].verdict == "not_established"
    assert "downgraded" in verdicts[1].reason


def test_locator_normalizes_burned_in_id_with_appended_time() -> None:
    decision = parse_locator(
        json.dumps(
            {
                "candidates": [
                    {
                        "node_id": "L1_N0003830_1910s",
                        "visible_anchor": "two men",
                    }
                ]
            }
        ),
        valid_node_ids={"L1-N0003", "L1-N0007"},
        max_candidates=2,
    )
    assert decision.node_ids == ("L1-N0003",)


def test_hypothesis_rejects_non_discriminative_rules_and_placeholders() -> None:
    options = (
        CanonicalOption("O1", "A", "Brown."),
        CanonicalOption("O2", "B", "Purple."),
    )
    payload = {
        "claim_tests": [
            {
                "claim_id": "C1",
                "statement": "The mountains are [COLOR].",
                "slot_ids": ["S1"],
                "predicate": "has_color",
                "expected_value": "COLOR",
            }
        ],
        "option_rules": [
            {"option_id": "O1", "all_of": ["C1"], "none_of": []},
            {"option_id": "O2", "all_of": ["C1"], "none_of": []},
        ],
        "cannot_determine_option_id": None,
    }
    with pytest.raises(ProtocolError, match="identical rules"):
        parse_decision_spec(json.dumps(payload), options=options, valid_slot_ids={"S1"})


def test_hypothesis_prompt_requires_atomic_label_free_discriminants() -> None:
    spec = ObservationSpec(
        answer_mode="multiple_choice",
        primary_mode="static_visual",
        required_slots=(ObservationSlot("S1", "visible clothing depiction"),),
    )
    options = (
        CanonicalOption("O1", "A", "A lovable magic dragon."),
        CanonicalOption("O2", "B", "A tree with lots of leaves."),
    )
    prompt = build_hypothesis_compiler_prompt("What is depicted on the shirt?", options, spec)
    assert "smallest concrete observations" in prompt
    assert "claim is atomic" in prompt
    assert "under 30 words" in prompt
    assert "actor, direct" in prompt
    assert "Cannot be determined" in prompt
    assert "claim-ID strings only" in prompt
    assert "Use internal IDs such as O1 only in option_rules" in prompt
    assert '"all_of":["C1"]' in prompt


def test_hypothesis_lifts_inline_claim_objects_from_option_rules() -> None:
    options = (
        CanonicalOption("O1", "A", "A globe trophy."),
        CanonicalOption("O2", "B", "A torch."),
    )
    payload = {
        "claim_tests": [],
        "option_rules": [
            {
                "option_id": "O1",
                "all_of": [
                    {
                        "claim_id": "C1",
                        "statement": "The man stands next to a globe trophy.",
                        "slot_ids": ["S1"],
                        "predicate": "matches",
                        "expected_value": "a globe trophy",
                    }
                ],
                "none_of": [],
            },
            {
                "option_id": "O2",
                "all_of": [
                    {
                        "claim_id": "C2",
                        "statement": "The man stands next to a torch.",
                        "slot_ids": ["S1"],
                        "predicate": "matches",
                        "expected_value": "a torch",
                    }
                ],
                "none_of": [],
            },
        ],
        "cannot_determine_option_id": None,
    }
    decision = parse_decision_spec(
        json.dumps(payload),
        options=options,
        valid_slot_ids={"S1"},
    )
    assert [claim.claim_id for claim in decision.claim_tests] == ["C1", "C2"]
    assert [rule.all_of for rule in decision.option_rules] == [("C1",), ("C2",)]


def test_deterministic_hypothesis_compiler_wraps_odd_options_without_labels() -> None:
    spec = ObservationSpec(
        answer_mode="multiple_choice",
        primary_mode="static_visual",
        required_slots=(ObservationSlot("S1", "visible clothing depiction"),),
    )
    options = (
        CanonicalOption("O1", "A", "A lovable magic dragon."),
        CanonicalOption("O2", "B", "Cannot be determined."),
    )
    decision = compile_decision_spec(
        "What is depicted on the man's clothes?",
        options,
        spec,
    )
    assert len(decision.claim_tests) == 2
    assert decision.option_rules[0].all_of == ("C1",)
    assert decision.option_rules[1].all_of == ("C2",)
    assert decision.cannot_determine_option_id == "O2"
    assert all(
        "O1" not in claim.statement and "O2" not in claim.statement
        for claim in decision.claim_tests
    )


def test_interval_parsing_reconciliation_and_conflict() -> None:
    parsed = parse_question_interval("What happens from 01:02 to 01:05?", 100)
    assert parsed is not None
    assert (parsed.start_seconds, parsed.end_seconds) == (62, 65)
    assert reconcile_interval(TimeSpan(62.2, 65.2), parsed, 100) is not None
    with pytest.raises(ContractViolation):
        reconcile_interval(TimeSpan(70, 75), parsed, 100)


def test_uniform_sampling_honors_cap_without_losing_tail() -> None:
    values = uniform_timestamps(TimeSpan(0, 24), fps=4, max_frames=48)
    assert len(values) == 48
    assert values[0] == 0
    assert values[-1] == 24


def test_interval_chunk_limit_adapts_without_losing_coverage() -> None:
    chunks = interval_chunks(TimeSpan(0, 200), P01Config())
    assert len(chunks) == 8
    assert chunks[0].start_seconds == 0
    assert chunks[-1].end_seconds == 200
    assert all(left.end_seconds >= right.start_seconds for left, right in pairwise(chunks))


def test_index_flags_bad_frames_without_deleting_them_and_bounds_tree(tmp_path: Path) -> None:
    config = P01Config(cache_dir=str(tmp_path / "cache"))
    paths = []
    for index, color in enumerate(("black", "black", "white")):
        path = tmp_path / f"quality-{index}.jpg"
        Image.new("RGB", (64, 48), color).save(path)
        paths.append(path)
    frames = tuple(
        FrameRef(f"F{index}", float(index), str(path)) for index, path in enumerate(paths)
    )
    builder = P01IndexBuilder(config)
    metrics = builder._measure_frames(frames)
    assert len(metrics) == len(frames)
    assert metrics[0].is_black
    assert metrics[1].is_duplicate

    shots = tuple(
        ShotRef(f"S{index:04d}", TimeSpan(index * 2, index * 2 + 2), ()) for index in range(30)
    )
    nodes, root_id = builder._build_tree(shots)
    assert root_id == "ROOT"
    assert all(len(node.child_ids) <= 12 for node in nodes.values())
    root_children = [nodes[node_id] for node_id in nodes[root_id].child_ids]
    assert len(root_children) == 12
    durations = [node.span.duration_seconds for node in root_children]
    assert max(durations) - min(durations) <= 2.0


def test_canonical_span_keeps_in_span_provenance_from_a_wider_static_fact() -> None:
    fact = StaticFact(
        fact_id="F1",
        slot_ids=("S1",),
        start_seconds=11,
        end_seconds=29.5,
        visibility="clear",
        statement="The mountains are purple.",
        source_frame_ids=("OUT-A", "IN-A", "IN-B", "OUT-B"),
        view_id="V1",
        entity="mountains",
        attribute="color",
        value="purple",
    )
    coverage = CoverageManifest(
        mode="static_visual",
        observed_start_seconds=12.25,
        observed_end_seconds=28.25,
        sample_fps=None,
        max_temporal_gap_seconds=None,
        frame_ids=("OUT-A", "IN-A", "IN-B", "OUT-B"),
        context_only_frame_ids=("OUT-A", "OUT-B"),
    )
    kept = P01VideoAgent._valid_decisive_facts((fact,), coverage, TimeSpan(12.25, 28.25))
    assert len(kept) == 1
    assert kept[0].source_frame_ids == ("IN-A", "IN-B")
    assert (kept[0].start_seconds, kept[0].end_seconds) == (12.25, 28.25)


def test_valid_ocr_fact_satisfies_slot_even_when_an_earlier_fact_is_weak(
    tmp_path: Path,
) -> None:
    spec = ObservationSpec(
        answer_mode="multiple_choice",
        primary_mode="ocr",
        required_slots=(ObservationSlot("S1", "phone text", True, "text"),),
        coverage_requirement="text_consensus",
    )
    common = {
        "slot_ids": ("S1",),
        "start_seconds": 1.0,
        "end_seconds": 2.0,
        "visibility": "clear",
        "statement": "HELLO",
        "view_id": "V1",
    }
    weak = TextFact(fact_id="F1", source_frame_ids=("A",), **common)
    strong = TextFact(
        fact_id="F2",
        source_frame_ids=("A", "B"),
        exact_text="HELLO",
        bbox=BoundingBox("A", 100, 100, 900, 900),
        consensus_frame_ids=("A", "B"),
        **common,
    )
    packet = EvidencePacket(
        canonical_span=TimeSpan(0, 3),
        facts=(weak, strong),
        coverage_manifest=CoverageManifest(
            mode="ocr",
            observed_start_seconds=0,
            observed_end_seconds=3,
            sample_fps=4,
            max_temporal_gap_seconds=0.25,
            frame_ids=("A", "B"),
        ),
    )
    agent = P01VideoAgent(
        RoleFakeModel({}),
        config=P01Config(cache_dir=str(tmp_path / "cache")),
    )
    assert agent._packet_sufficiency(spec, packet)["sufficient"] is True


def test_real_pyav_index_source_decode_and_normalized_crop(tmp_path: Path) -> None:
    import av

    video_path = tmp_path / "tiny.mp4"
    with av.open(str(video_path), mode="w") as container:
        stream = container.add_stream("mpeg4", rate=4)
        stream.width = 64
        stream.height = 48
        stream.pix_fmt = "yuv420p"
        for index in range(8):
            image = Image.new("RGB", (64, 48), (index * 20, 30, 200))
            frame = av.VideoFrame.from_image(image)
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)

    config = P01Config(cache_dir=str(tmp_path / "cache"))
    builder = P01IndexBuilder(config)
    metadata = builder.probe(video_path)
    assert metadata.duration_seconds > 0
    interval_index = builder.prepare_interval(video_path, TimeSpan(0.5, 1.25), metadata=metadata)
    assert (
        len(interval_index.cached_video.frames) < metadata.duration_seconds * config.index_fps + 1
    )
    assert all(
        0.5 - 1e-3 <= frame.timestamp_seconds <= 1.25 + 1e-3
        for frame in interval_index.cached_video.frames
    )
    assert interval_index.shots[0].span.source == "explicit_interval_index"

    index = builder.prepare(video_path)
    assert index.duration_seconds > 0
    assert index.cached_video.width == 64
    assert len(index.metrics) == len(index.cached_video.frames)

    store = SourceFrameStore(config)
    frames = store.extract(video_path, (0.1, 0.6, 1.2), purpose="test")
    assert len(frames) == 3
    assert all(Path(frame.path).is_file() for frame in frames)
    bbox = BoundingBox(frames[0].id, 100, 100, 900, 900)
    crop = store.crop(frames[0], bbox)
    assert Path(crop.path).is_file()
    with Image.open(crop.path) as image:
        assert image.size != (64, 48)
        assert image.width <= 64 and image.height <= 48


def test_cli_exposes_p01_contract_flags() -> None:
    args = build_parser().parse_args(
        [
            "--query",
            "question",
            "--video",
            "clip.mp4",
            "--strategy",
            "p01",
            "--given-interval",
            "2",
            "4",
            "--force-choice",
        ]
    )
    assert args.strategy == "p01"
    assert args.given_interval == [2.0, 4.0]
    assert args.force_choice


def test_g01_static_mcq_is_answered_and_blind_until_hypothesis(tmp_path: Path) -> None:
    config = P01Config(cache_dir=str(tmp_path / "cache"))
    index, image_path = _make_index(tmp_path, config)
    model = RoleFakeModel(
        {
            "observation_compiler": _observation(),
            "candidate_scout": _static_scout(),
            "hypothesis_compiler": _decision(),
            "initial_decision": _choice_decision(),
        }
    )
    result, _store = _run_agent(
        model,
        index,
        image_path,
        P01Request(
            str(tmp_path / "video.mp4"),
            "What color is the target?",
            ("scarlet candidate", "azure candidate"),
        ),
        config=config,
    )
    assert result.status == "answered"
    assert result.prediction == "A"
    assert result.verified_answer is None
    assert result.support_level == "strong"
    roles = [call["role"] for call in model.calls]
    assert roles == [
        "observation_compiler",
        "candidate_scout",
        "hypothesis_compiler",
        "initial_decision",
    ]
    assert result.trace["hypothesis_compiler"]["kind"] == ("model_label_free_discriminants")
    blind_prompts = [
        call["prompt"]
        for call in model.calls
        if call["role"] in {"observation_compiler", "candidate_scout"}
    ]
    assert all("scarlet candidate" not in prompt for prompt in blind_prompts)
    decision_prompt = next(
        call["prompt"] for call in model.calls if call["role"] == "initial_decision"
    )
    assert "scarlet candidate" in decision_prompt
    assert "option_rules" in decision_prompt


def test_dynamic_action_gets_exactly_one_refinement(tmp_path: Path) -> None:
    config = P01Config(cache_dir=str(tmp_path / "cache"))
    index, image_path = _make_index(tmp_path, config)
    model = RoleFakeModel(
        {
            "observation_compiler": _observation("dynamic_action", coverage="sequence"),
            "candidate_scout": _event_scout("partial"),
            "hypothesis_compiler": _decision(),
            "refinement_extractor": _event_scout("clear"),
            "initial_decision": _choice_decision(),
        }
    )
    result, _store = _run_agent(
        model,
        index,
        image_path,
        P01Request(str(tmp_path / "video.mp4"), "What does the person do?", ("red", "blue")),
        config=config,
    )
    assert result.status == "answered"
    assert [call["role"] for call in model.calls].count("refinement_extractor") == 1
    assert result.evidence is not None
    assert any(fact.visibility == "clear" for fact in result.evidence.facts)


def test_clear_but_non_discriminative_action_fact_triggers_refinement(tmp_path: Path) -> None:
    config = P01Config(cache_dir=str(tmp_path / "cache"))
    index, image_path = _make_index(tmp_path, config)

    def generic_fire_fact(prompt: str, _messages: list[dict[str, Any]]) -> str:
        frame_id, timestamp = _frame_map(prompt)[0]
        return json.dumps(
            {
                "target_visible": True,
                "supporting_start": timestamp,
                "supporting_end": timestamp + 0.25,
                "visible_anchor": "visible smoke",
                "facts": [
                    {
                        "kind": "event",
                        "slot_ids": ["S1"],
                        "start_seconds": timestamp,
                        "end_seconds": timestamp,
                        "visibility": "clear",
                        "statement": "The man uses fire to generate smoke.",
                        "source_frame_ids": [frame_id],
                        "subject": "man",
                        "initial_state": "",
                        "action": "uses",
                        "object": "fire",
                        "target": "",
                        "result": "smoke",
                        "order": 1,
                    }
                ],
                "missing_slot_ids": [],
                "conflicts": [],
            }
        )

    discriminants = json.dumps(
        {
            "claim_tests": [
                {
                    "claim_id": "C1",
                    "statement": "A piece of cloth is burned.",
                    "slot_ids": ["S1"],
                    "predicate": "method",
                    "expected_value": "burning a piece of cloth",
                },
                {
                    "claim_id": "C2",
                    "statement": "A bonfire is lit.",
                    "slot_ids": ["S1"],
                    "predicate": "method",
                    "expected_value": "lighting a bonfire",
                },
            ],
            "option_rules": [
                {"option_id": "O1", "all_of": ["C1"], "none_of": []},
                {"option_id": "O2", "all_of": ["C2"], "none_of": []},
            ],
            "cannot_determine_option_id": None,
        }
    )
    model = RoleFakeModel(
        {
            "observation_compiler": _observation("dynamic_action", coverage="sequence"),
            "candidate_scout": generic_fire_fact,
            "hypothesis_compiler": discriminants,
            "refinement_extractor": generic_fire_fact,
            "initial_decision": _choice_decision(),
        }
    )
    result, _store = _run_agent(
        model,
        index,
        image_path,
        P01Request(
            str(tmp_path / "video.mp4"),
            "How is the smoke generated?",
            ("By burning cloth", "By lighting a bonfire"),
        ),
        config=config,
    )
    assert result.status == "answered"
    assert [call["role"] for call in model.calls].count("refinement_extractor") == 1
    plan = result.trace["refinement_plan"]
    assert set(plan["target_claim_ids"]) == {"C1", "C2"}
    assert "option discriminants" in plan["reason"]


def test_explicit_interval_continues_to_targeted_refinement_after_blind_miss(
    tmp_path: Path,
) -> None:
    config = P01Config(cache_dir=str(tmp_path / "cache"))
    index, image_path = _make_index(tmp_path, config)
    blind_miss = json.dumps(
        {
            "target_visible": False,
            "supporting_start": None,
            "supporting_end": None,
            "visible_anchor": "",
            "facts": [],
            "missing_slot_ids": ["S1"],
            "conflicts": [],
        }
    )
    model = RoleFakeModel(
        {
            "observation_compiler": _observation("dynamic_action", coverage="sequence"),
            "interval_scout": blind_miss,
            "hypothesis_compiler": _decision(),
            "refinement_extractor": _event_scout("clear"),
            "initial_decision": _choice_decision(),
        }
    )
    result, _store = _run_agent(
        model,
        index,
        image_path,
        P01Request(
            str(tmp_path / "video.mp4"),
            "What does the vlogger do in the supplied interval?",
            ("red", "blue"),
            given_interval=TimeSpan(0, 3),
        ),
        config=config,
    )
    assert result.status == "answered", result.to_dict()
    assert [call["role"] for call in model.calls] == [
        "observation_compiler",
        "interval_scout",
        "hypothesis_compiler",
        "refinement_extractor",
        "initial_decision",
    ]


def test_g07_ocr_uses_original_bbox_consensus_and_one_crop_batch(tmp_path: Path) -> None:
    config = P01Config(cache_dir=str(tmp_path / "cache"))
    index, image_path = _make_index(tmp_path, config)
    model = RoleFakeModel(
        {
            "observation_compiler": _observation("ocr", coverage="text_consensus"),
            "candidate_scout": _ocr_scout,
            "hypothesis_compiler": _decision(),
            "refinement_extractor": _ocr_scout,
            "initial_decision": _choice_decision(),
        }
    )
    result, store = _run_agent(
        model,
        index,
        image_path,
        P01Request(str(tmp_path / "video.mp4"), "What text is shown?", ("red", "blue")),
        config=config,
    )
    assert result.status == "answered"
    assert [call["role"] for call in model.calls].count("refinement_extractor") == 1
    assert store.crop_calls
    assert len(store.crop_calls) <= config.max_detail_images * 2
    assert all(not bbox.frame_id.startswith("CROP-") for _frame, bbox in store.crop_calls)
    assert result.evidence is not None
    text_facts = [fact for fact in result.evidence.facts if fact.kind == "text"]
    assert text_facts
    assert all(len(fact.consensus_frame_ids) >= 2 for fact in text_facts)
    refinement_call = next(call for call in model.calls if call["role"] == "refinement_extractor")
    part_types = [part["type"] for part in refinement_call["messages"][-1]["content"]]
    assert "video" in part_types
    assert "image" in part_types


def test_g39_composes_from_evidence_and_local_media_without_verifier(tmp_path: Path) -> None:
    config = P01Config(cache_dir=str(tmp_path / "cache"))
    index, image_path = _make_index(tmp_path, config)

    model = RoleFakeModel(
        {
            "observation_compiler": _observation(
                "subscene_caption",
                answer_mode="free_text",
                coverage="sequence",
            ),
            "candidate_scout": _event_scout("clear"),
            "answer_composer": "这个人拿起了杯子。",
        }
    )
    result, _store = _run_agent(
        model,
        index,
        image_path,
        P01Request(str(tmp_path / "video.mp4"), "画面中的人做了什么？"),
        config=config,
    )
    assert result.prediction == "这个人拿起了杯子。"
    assert "verifier" not in [call["role"] for call in model.calls]
    assert model.calls[-1]["role"] == "answer_composer"
    composer = model.calls[-1]["prompt"]
    assert "cup" in composer
    assert isinstance(model.calls[-1]["messages"][-1]["content"], list)


def test_second_candidate_wins_deterministic_arbitration(tmp_path: Path) -> None:
    config = P01Config(cache_dir=str(tmp_path / "cache"))
    index, image_path = _make_index(
        tmp_path,
        config,
        duration=30,
        shot_spans=((0, 15), (15, 30)),
    )

    def locate(_prompt: str, _messages: list[dict[str, Any]]) -> str:
        return json.dumps(
            {
                "candidates": [
                    {"node_id": "S0000", "visible_anchor": "first target"},
                    {"node_id": "S0001", "visible_anchor": "second target"},
                ]
            }
        )

    def scout(prompt: str, messages: list[dict[str, Any]]) -> str:
        if "CANDIDATE: S0000" in prompt:
            return _static_scout(visibility="partial")(prompt, messages)
        return _static_scout(visibility="clear")(prompt, messages)

    model = RoleFakeModel(
        {
            "observation_compiler": _observation(),
            "locator": locate,
            "candidate_scout": scout,
            "hypothesis_compiler": _decision(),
            "initial_decision": _choice_decision(),
        }
    )
    result, _store = _run_agent(
        model,
        index,
        image_path,
        P01Request(str(tmp_path / "video.mp4"), "What color is the target?", ("red", "blue")),
        config=config,
    )
    selected = next(item for item in result.trace["candidate_ranking"] if item["selected"])
    assert selected["candidate_id"] == "S0001"
    assert [call["role"] for call in model.calls].count("candidate_scout") == 2


def test_no_locator_anchor_runs_one_bounded_rescue_and_still_answers(tmp_path: Path) -> None:
    config = P01Config(cache_dir=str(tmp_path / "cache"))
    index, image_path = _make_index(
        tmp_path,
        config,
        duration=30,
        shot_spans=((0, 15), (15, 30)),
    )
    model = RoleFakeModel(
        {
            "observation_compiler": _observation(),
            "locator": json.dumps({"candidates": []}),
            "hypothesis_compiler": _decision(),
            "rescue_locator": _rescue_locator,
            "rescue_scout": _invisible_scout,
            "final_decision": _choice_decision("O1", support=0, unresolved=("C1", "C2")),
        }
    )
    result, _store = _run_agent(
        model,
        index,
        image_path,
        P01Request(
            str(tmp_path / "video.mp4"),
            "What color is the target?",
            ("red", "Cannot be determined"),
            force_choice=True,
        ),
        config=config,
    )
    assert result.status == "answered"
    assert result.prediction == "A"
    assert result.verified_answer is None
    assert result.forced_prediction == "A"
    assert result.trace["stop_reason"] == "mandatory_mcq_prediction_emitted"
    assert result.trace["bounded_rescue"]["outcome"] == "packet_selected"
    assert [call["role"] for call in model.calls].count("rescue_locator") == 1
    assert [call["role"] for call in model.calls].count("rescue_scout") <= 2
    rescue_prompts = [
        call["prompt"] for call in model.calls if call["role"] in {"rescue_locator", "rescue_scout"}
    ]
    assert all("O1" not in prompt and "O2" not in prompt for prompt in rescue_prompts)
    assert all("(A)" not in prompt and "(B)" not in prompt for prompt in rescue_prompts)


@pytest.mark.parametrize(
    "only_first",
    [False, True],
)
def test_occlusion_is_diagnostic_but_never_suppresses_mcq_output(
    tmp_path: Path,
    only_first: bool,
) -> None:
    config = P01Config(cache_dir=str(tmp_path / "cache"))
    index, image_path = _make_index(tmp_path, config, duration=2)
    model = RoleFakeModel(
        {
            "observation_compiler": _observation(
                "dynamic_action",
                coverage="full_span",
            ),
            "interval_scout": _event_scout("occluded"),
            "hypothesis_compiler": _decision(),
            "refinement_extractor": _event_scout("occluded"),
            "initial_decision": _choice_decision(support=1),
            "rescue_scout": _event_scout("occluded"),
            "final_decision": _choice_decision(support=1),
        }
    )
    store = FakeSourceStore(image_path, only_first=only_first)
    result, _store = _run_agent(
        model,
        index,
        image_path,
        P01Request(
            str(tmp_path / "video.mp4"),
            "What happens in the interval?",
            ("red", "blue"),
            given_interval=TimeSpan(0, 2),
        ),
        store=store,
        config=config,
    )
    assert result.status == "answered"
    assert result.prediction == "A"
    assert result.support_level in {"weak", "none"}
    assert result.verified_answer is None


def test_g42_context_only_frame_cannot_be_decisive(tmp_path: Path) -> None:
    config = P01Config(cache_dir=str(tmp_path / "cache"))
    index, image_path = _make_index(tmp_path, config, duration=8)

    def scout(prompt: str, _messages: list[dict[str, Any]]) -> str:
        frame_map = _frame_map(prompt)
        valid_id, valid_time = next(item for item in frame_map if 2 <= item[1] <= 4)
        facts = []
        for fact_id, timestamp, statement in (
            ("OUTSIDE-FRAME", 1.0, "bad outside fact"),
            (valid_id, valid_time, "the target is red"),
        ):
            facts.append(
                {
                    "kind": "static",
                    "slot_ids": ["S1"],
                    "start_seconds": timestamp,
                    "end_seconds": timestamp,
                    "visibility": "clear",
                    "statement": statement,
                    "source_frame_ids": [fact_id],
                    "entity": "target",
                    "attribute": "color",
                    "relation": "",
                    "value": "red",
                }
            )
        return json.dumps(
            {
                "target_visible": True,
                "supporting_start": 2,
                "supporting_end": 4,
                "visible_anchor": "target",
                "facts": facts,
                "missing_slot_ids": [],
                "conflicts": [],
            }
        )

    model = RoleFakeModel(
        {
            "observation_compiler": _observation(),
            "interval_scout": scout,
            "hypothesis_compiler": _decision(),
            "initial_decision": _choice_decision(),
        }
    )
    result, _store = _run_agent(
        model,
        index,
        image_path,
        P01Request(
            str(tmp_path / "video.mp4"),
            "What color is shown?",
            ("red", "blue"),
            given_interval=TimeSpan(2, 4),
        ),
        config=config,
    )
    assert result.evidence is not None
    context_ids = set(result.evidence.coverage_manifest.context_only_frame_ids)
    assert not context_ids
    assert all(not (set(fact.source_frame_ids) & context_ids) for fact in result.evidence.facts)
    assert all(fact.statement != "bad outside fact" for fact in result.evidence.facts)
    assert all(
        2 <= timestamp <= 4 for call in _store.extract_calls for timestamp in call["timestamps"]
    )


def test_mcq_prediction_is_mandatory_even_without_force_choice(tmp_path: Path) -> None:
    config = P01Config(cache_dir=str(tmp_path / "cache"))
    index, image_path = _make_index(tmp_path, config)
    model = RoleFakeModel(
        {
            "observation_compiler": _observation(),
            "candidate_scout": _static_scout(),
            "hypothesis_compiler": _decision(),
            "initial_decision": _choice_decision(support=0, unresolved=("C1", "C2")),
            "rescue_locator": _rescue_locator,
            "rescue_scout": _invisible_scout,
            "final_decision": _choice_decision(support=0, unresolved=("C1", "C2")),
        }
    )
    result, _store = _run_agent(
        model,
        index,
        image_path,
        P01Request(
            str(tmp_path / "video.mp4"),
            "What color is the target?",
            ("red", "blue"),
            force_choice=False,
        ),
        config=config,
    )
    assert result.verified_answer is None
    assert result.prediction == "A"
    assert result.forced_prediction == "A"
    assert result.prediction_kind == "forced"


def test_protocol_repair_succeeds_without_replaying_media(tmp_path: Path) -> None:
    config = P01Config(cache_dir=str(tmp_path / "cache"))
    index, image_path = _make_index(tmp_path, config)
    model = RoleFakeModel(
        {
            "observation_compiler": "not json",
            "protocol_repair": _observation(),
            "candidate_scout": _static_scout(),
            "hypothesis_compiler": _decision(),
            "initial_decision": _choice_decision(),
        }
    )
    result, _store = _run_agent(
        model,
        index,
        image_path,
        P01Request(str(tmp_path / "video.mp4"), "Question?", ("red", "blue")),
        config=config,
    )
    assert result.status == "answered"
    repair = next(call for call in model.calls if call["role"] == "protocol_repair")
    assert isinstance(repair["messages"][-1]["content"], str)
    assert result.trace["protocol_repairs"][0]["media_replayed"] is False


def test_intermediate_protocol_failure_degrades_without_direct_video_fallback(
    tmp_path: Path,
) -> None:
    config = P01Config(cache_dir=str(tmp_path / "cache"))
    index, image_path = _make_index(tmp_path, config)
    model = RoleFakeModel(
        {
            "observation_compiler": "bad",
            "protocol_repair": "still bad",
            "candidate_scout": _static_scout(),
            "hypothesis_compiler": _decision(),
            "initial_decision": _choice_decision(),
        }
    )
    result, _store = _run_agent(
        model,
        index,
        image_path,
        P01Request(str(tmp_path / "video.mp4"), "Question?", ("red", "blue")),
        config=config,
    )
    assert result.status == "answered"
    assert result.prediction == "A"
    assert result.trace["degraded_stages"][0]["stage"] == "observation_compiler"
    assert "direct" not in [call["role"] for call in model.calls]


def test_long_g42_adapts_to_eight_chunks_and_contract_error_remains_distinct(
    tmp_path: Path,
) -> None:
    config = P01Config(cache_dir=str(tmp_path / "cache"))
    index, image_path = _make_index(tmp_path, config, duration=200)

    resource_model = RoleFakeModel(
        {
            "observation_compiler": _observation(
                "subscene_caption",
                answer_mode="free_text",
                coverage="full_span",
            ),
            "interval_scout": _event_scout("clear"),
            "answer_composer": "A person lifts a cup.",
        }
    )
    resource, _store = _run_agent(
        resource_model,
        index,
        image_path,
        P01Request(
            str(tmp_path / "video.mp4"),
            "Describe this interval.",
            given_interval=TimeSpan(0, 200),
        ),
        config=config,
    )
    assert resource.status == "answered"
    assert resource.prediction
    assert len(resource.trace["interval_chunks"]) == 8

    conflict_model = RoleFakeModel({})
    conflict, _store = _run_agent(
        conflict_model,
        index,
        image_path,
        P01Request(
            str(tmp_path / "video.mp4"),
            "What happens from 00:10 to 00:20?",
            given_interval=TimeSpan(30, 40),
        ),
        config=config,
    )
    assert conflict.status == "input_contract_violation"
    assert conflict_model.calls == []


def test_malformed_terminal_decision_recovers_legal_label_and_never_abstains(
    tmp_path: Path,
) -> None:
    config = P01Config(cache_dir=str(tmp_path / "cache"), max_model_calls=5)
    index, image_path = _make_index(tmp_path, config)
    model = RoleFakeModel(
        {
            "observation_compiler": _observation(),
            "candidate_scout": _static_scout(),
            "hypothesis_compiler": _decision(),
            "initial_decision": "Answer: B",
            "protocol_repair": "still malformed",
        }
    )
    result, _store = _run_agent(
        model,
        index,
        image_path,
        P01Request(str(tmp_path / "video.mp4"), "What color?", ("red", "blue")),
        config=config,
    )
    assert result.status == "answered"
    assert result.prediction == "B"
    assert result.decision_source == "terminal_fallback"
    assert result.resources["model_call_count"] == 5
    assert result.trace["decision_fallbacks"][0]["kind"] == "legal_label_extraction"


def test_cuda_oom_retries_same_decision_once_with_recorded_safe_visual_budget(
    tmp_path: Path,
) -> None:
    config = P01Config(cache_dir=str(tmp_path / "cache"), max_model_calls=6)
    index, image_path = _make_index(tmp_path, config)

    def oom(_prompt: str, _messages: list[dict[str, Any]]) -> str:
        raise RuntimeError("CUDA out of memory")

    model = RoleFakeModel(
        {
            "observation_compiler": _observation(),
            "candidate_scout": _static_scout(),
            "hypothesis_compiler": _decision(),
            "initial_decision": [oom, _choice_decision()],
        }
    )
    result, _store = _run_agent(
        model,
        index,
        image_path,
        P01Request(str(tmp_path / "video.mp4"), "What color?", ("red", "blue")),
        config=config,
    )
    assert result.prediction == "A"
    decision_calls = [
        call for call in result.resources["calls"] if call["role"] == "initial_decision"
    ]
    assert len(decision_calls) == 2
    assert decision_calls[0]["model_metadata"]["is_cuda_oom"] is True
    assert decision_calls[1]["media_config"]["safe_budget"] is True
    assert result.trace["oom_retries"][0]["retry_index"] == 1


def test_empty_g39_composer_uses_nonempty_event_fact_fallback(tmp_path: Path) -> None:
    config = P01Config(cache_dir=str(tmp_path / "cache"))
    index, image_path = _make_index(tmp_path, config)
    model = RoleFakeModel(
        {
            "observation_compiler": _observation(
                "subscene_caption",
                answer_mode="free_text",
                coverage="sequence",
            ),
            "candidate_scout": _event_scout("clear"),
            "answer_composer": "",
        }
    )
    result, _store = _run_agent(
        model,
        index,
        image_path,
        P01Request(str(tmp_path / "video.mp4"), "What happens?"),
        config=config,
    )
    assert result.status == "answered"
    assert result.prediction == "the person lifts the cup"
    assert result.decision_source == "terminal_fallback"
    assert result.pipeline_outcome == "completed_with_degradation"


def test_long_explicit_interval_rescue_uses_at_most_two_scout_calls(
    tmp_path: Path,
) -> None:
    config = P01Config(cache_dir=str(tmp_path / "cache"))
    index, image_path = _make_index(tmp_path, config, duration=200)
    model = RoleFakeModel(
        {
            "observation_compiler": _observation(
                "static_visual",
                coverage="full_span",
            ),
            "interval_scout": _invisible_scout,
            "hypothesis_compiler": _decision(),
            "refinement_extractor": _invisible_scout,
            "initial_decision": _choice_decision(
                support=0,
                unresolved=("C1", "C2"),
            ),
            "rescue_scout": _invisible_scout,
            "final_decision": _choice_decision(
                support=0,
                unresolved=("C1", "C2"),
            ),
        }
    )
    result, _store = _run_agent(
        model,
        index,
        image_path,
        P01Request(
            str(tmp_path / "video.mp4"),
            "What color is shown in this interval?",
            ("red", "blue"),
            given_interval=TimeSpan(0, 200),
        ),
        config=config,
    )
    assert result.prediction == "A"
    assert [call["role"] for call in model.calls].count("interval_scout") == 8
    assert [call["role"] for call in model.calls].count("rescue_scout") == 2
    assert len(result.trace["bounded_rescue"]["explicit_rescue_chunks"]) == 2
    assert result.canonical_span is not None
    assert (
        result.canonical_span.start_seconds,
        result.canonical_span.end_seconds,
    ) == (0, 200)
