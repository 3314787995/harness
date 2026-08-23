from __future__ import annotations

import hashlib
import logging
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any, TypeVar

from qwen3vl_agent.active_tree.alignment import (
    EvidenceAlignment,
    align_evidence_to_options,
)
from qwen3vl_agent.active_tree.config import ActiveTreeConfig
from qwen3vl_agent.active_tree.prompts import (
    ObserverDecision,
    VerificationDecision,
    build_breadth_prompt,
    build_completeness_prompt,
    build_discriminator_prompt,
    build_observer_prompt,
    build_planner_prompt,
    build_skeptic_prompt,
    build_task_compiler_prompt,
    canonicalize_options,
    parse_discriminator,
    parse_observer,
    parse_planner,
    parse_task_contract,
    parse_verification,
    repair_prompt,
)
from qwen3vl_agent.active_tree.scene_tree import SceneTreeBuilder
from qwen3vl_agent.active_tree.temporal import (
    TemporalComposition,
    compose_temporal_option,
)
from qwen3vl_agent.active_tree.types import (
    AtomicEvidence,
    CanonicalOption,
    EvidenceLedger,
    EvidenceSlot,
    ModelCallLimit,
    PlannedAction,
    ProtocolError,
    ResourceLedger,
    SceneNode,
    SceneTree,
    TaskContract,
)
from qwen3vl_agent.coarse_to_fine.cache import (
    CachedVideo,
    SubtitleTrack,
    VideoEvidenceCache,
    build_contact_sheet,
)
from qwen3vl_agent.coarse_to_fine.types import FrameRef
from qwen3vl_agent.models.base import BaseVideoModel, ModelOutput

logger = logging.getLogger(__name__)
T = TypeVar("T")


class ActiveTreeVideoAgent:
    """Training-free active sensing controller over a scene-aware temporal tree."""

    def __init__(
        self,
        model: BaseVideoModel,
        *,
        config: ActiveTreeConfig | Mapping[str, Any] | None = None,
        cache: VideoEvidenceCache | None = None,
        tree_builder: SceneTreeBuilder | None = None,
    ) -> None:
        self.model = model
        self.config = (
            config if isinstance(config, ActiveTreeConfig) else ActiveTreeConfig.from_mapping(config)
        )
        self.config.validate()
        self.cache = cache or VideoEvidenceCache(
            self.config.cache_dir,
            sample_fps=self.config.sample_fps,
            max_side=self.config.cache_max_side,
            jpeg_quality=self.config.cache_jpeg_quality,
            lru_size=self.config.cache_lru_size,
        )
        self.tree_builder = tree_builder or SceneTreeBuilder(self.config)
        self._loaded = False

    def load(self) -> None:
        if self._loaded:
            return
        self.model.load()
        self._loaded = True

    def unload(self) -> None:
        self.model.unload()
        self._loaded = False

    def generate(
        self,
        messages: list[dict[str, Any]],
        *,
        videos: list[str] | None = None,
        images: list[str] | None = None,
        choices: Sequence[str] | None = None,
        subtitle_path: str | None = None,
        **_: Any,
    ) -> ModelOutput:
        if not self._loaded:
            raise RuntimeError("Agent is not loaded. Call load() first.")
        if images:
            raise ValueError("active_tree accepts video input only")
        if not videos or len(videos) != 1:
            raise ValueError("active_tree requires exactly one video")
        if not choices or not 2 <= len(choices) <= 26:
            raise ValueError("active_tree currently requires multiple-choice options")
        question = self._latest_user_text(messages)
        if not question:
            raise ValueError("messages must contain a non-empty user question")

        options = canonicalize_options(tuple(str(item) for item in choices))
        resources = ResourceLedger(self.config.max_model_calls)
        trace: dict[str, Any] = {
            "strategy": "active_tree",
            "question": question,
            "options": [item.to_dict() for item in options],
            "video_path": videos[0],
            "subtitle_path": subtitle_path,
            "events": [],
            "verification_attempts": [],
            "protocol_failures": [],
            "verified": False,
            "degraded": False,
        }
        started = time.perf_counter()
        try:
            cached = self.cache.prepare(videos[0])
            subtitles = SubtitleTrack.from_srt(subtitle_path) if subtitle_path else None
            tree = self.tree_builder.build(cached, subtitles)
            trace["cache"] = cached.to_dict()
            trace["tree"] = tree.to_dict()
            trace["subtitle_cues"] = len(subtitles.cues) if subtitles else 0
            final_option_id, verified, stop_reason = self._run(
                question,
                options,
                cached,
                subtitles,
                tree,
                resources,
                trace,
            )
        except Exception as exc:  # noqa: BLE001 - active-tree must return an auditable result
            logger.warning("Active-tree stopped unverified: %s", exc)
            trace["degraded"] = True
            trace["fatal_error"] = f"{type(exc).__name__}: {exc}"
            final_option_id = self._best_effort_option(options, EvidenceLedger(), [])
            verified = False
            stop_reason = "controller_error_unverified"

        option = next(item for item in options if item.option_id == final_option_id)
        trace["final_option_id"] = option.option_id
        trace["final_benchmark_label"] = option.benchmark_label
        trace["verified"] = verified
        trace["stop_reason"] = stop_reason
        trace["resources"] = resources.to_dict()
        trace["wall_seconds"] = time.perf_counter() - started
        return ModelOutput(
            option.benchmark_label,
            {
                "active_tree": trace,
                "wall_seconds": trace["wall_seconds"],
                "verified": verified,
            },
        )

    def _run(
        self,
        question: str,
        options: list[CanonicalOption],
        cached: CachedVideo,
        subtitles: SubtitleTrack | None,
        tree: SceneTree,
        resources: ResourceLedger,
        trace: dict[str, Any],
    ) -> tuple[str, bool, str]:
        contract = self._compile_contract(question, subtitles is not None, resources, trace)
        trace["question_only_contract"] = contract.to_dict()
        ledger = EvidenceLedger()

        breadth_added, breadth_signatures = self._breadth_observe(
            question,
            contract,
            cached,
            subtitles,
            tree,
            ledger,
            resources,
            trace,
        )
        trace["events"].append(
            {
                "type": "option_reveal",
                "after_root_breadth": True,
                "breadth_new_evidence": breadth_added,
            }
        )
        contract = self._discriminate(
            question,
            options,
            contract,
            ledger,
            resources,
            trace,
        )
        trace["evidence_contract"] = contract.to_dict()

        expanded = {tree.root_id}
        current_node_id = tree.root_id
        observed_signatures = list(breadth_signatures)
        search_observations = 0
        navigation_actions = 0
        stagnant_observations = 0
        repair_round = 0
        observation_limit = self.config.max_search_observations
        next_verification_at = self.config.min_active_observations
        repair_directive = ""
        candidate_history: list[str] = []

        for sequence_action in self._sequence_breadth_actions(contract, ledger):
            trace["events"].append(
                {
                    "type": "sequence_breadth_proposal",
                    "action": sequence_action.to_dict(),
                }
            )
            added = self._execute_observation(
                question,
                options,
                contract,
                sequence_action,
                cached,
                subtitles,
                tree,
                ledger,
                resources,
                trace,
            )
            observed_signatures.append(sequence_action.signature())
            search_observations += 1
            current_node_id = str(sequence_action.node_id)
            stagnant_observations = 0 if added else stagnant_observations + 1

        retrieval_action, retrieval_alignment = self._subtitle_retrieval_action(
            options,
            contract,
            subtitles,
            tree,
        )
        if retrieval_action is not None:
            trace["events"].append(
                {
                    "type": "option_aware_retrieval_proposal",
                    "action": retrieval_action.to_dict(),
                    "alignment": retrieval_alignment.to_dict(),
                }
            )
            added = self._execute_observation(
                question,
                options,
                contract,
                retrieval_action,
                cached,
                subtitles,
                tree,
                ledger,
                resources,
                trace,
            )
            observed_signatures.append(retrieval_action.signature())
            search_observations += 1
            current_node_id = str(retrieval_action.node_id)
            stagnant_observations = 0 if added else 1

        while True:
            if resources.remaining_model_calls == 0:
                trace["evidence_ledger"] = ledger.to_list()
                return (
                    self._best_effort_option(options, ledger, candidate_history),
                    False,
                    "model_call_budget_exhausted",
                )
            missing_slots = ledger.missing_slot_ids(contract)
            refinement_action = self._sequence_refinement_action(
                tree,
                contract,
                ledger,
                observed_signatures,
            )
            sweep_action = (
                None
                if refinement_action is not None
                else self._sequence_sweep_action(
                    tree,
                    contract,
                    ledger,
                    observed_signatures,
                )
            )
            deterministic_sequence_action = refinement_action or sweep_action
            should_verify = (
                search_observations >= next_verification_at
                and not missing_slots
                and deterministic_sequence_action is None
            )
            if (
                missing_slots
                and search_observations >= observation_limit
                and deterministic_sequence_action is None
            ):
                trace["evidence_ledger"] = ledger.to_list()
                return (
                    self._best_effort_option(options, ledger, candidate_history),
                    False,
                    "evidence_search_exhausted_with_missing_slots",
                )
            if should_verify:
                if resources.remaining_model_calls < 2:
                    trace["evidence_ledger"] = ledger.to_list()
                    return (
                        self._best_effort_option(options, ledger, candidate_history),
                        False,
                        "insufficient_budget_for_dual_verification",
                    )
                try:
                    verification = self._verify(
                        question,
                        options,
                        contract,
                        ledger,
                        cached,
                        resources,
                        trace,
                        attempt=repair_round + 1,
                    )
                except ModelCallLimit:
                    trace["evidence_ledger"] = ledger.to_list()
                    return (
                        self._best_effort_option(options, ledger, candidate_history),
                        False,
                        "model_call_budget_exhausted_during_verification",
                    )
                for decision in verification[:2]:
                    if decision.candidate_option_id:
                        candidate_history.append(decision.candidate_option_id)
                passed, final_option_id = self._verification_passed(
                    verification[0],
                    verification[1],
                    ledger,
                    contract,
                    verification[2],
                    verification[3],
                )
                if passed and final_option_id is not None:
                    trace["evidence_ledger"] = ledger.to_list()
                    return final_option_id, True, "dual_verifier_agreement"
                alignment_passed, auditor = self._alignment_adjudication(
                    verification[0],
                    verification[1],
                    verification[2],
                    ledger,
                    contract,
                )
                if alignment_passed and verification[2].option_id is not None:
                    trace["events"].append(
                        {
                            "type": "evidence_alignment_adjudication",
                            "auditor": auditor,
                            "alignment": verification[2].to_dict(),
                        }
                    )
                    trace["evidence_ledger"] = ledger.to_list()
                    return (
                        verification[2].option_id,
                        True,
                        "grounded_alignment_adjudication",
                    )
                temporal_passed, temporal_auditor = self._temporal_adjudication(
                    verification[0],
                    verification[1],
                    verification[3],
                    ledger,
                    contract,
                )
                if temporal_passed and verification[3].option_id is not None:
                    trace["events"].append(
                        {
                            "type": "temporal_composition_adjudication",
                            "auditor": temporal_auditor,
                            "composition": verification[3].to_dict(),
                        }
                    )
                    trace["evidence_ledger"] = ledger.to_list()
                    return (
                        verification[3].option_id,
                        True,
                        "grounded_temporal_adjudication",
                    )
                if repair_round >= self.config.max_verification_repairs:
                    trace["evidence_ledger"] = ledger.to_list()
                    return (
                        self._best_effort_option(options, ledger, candidate_history),
                        False,
                        "verification_repairs_exhausted",
                    )
                repair_round += 1
                observation_limit += self.config.max_repair_observations
                next_verification_at = (
                    search_observations + self.config.max_repair_observations
                )
                repair_directive = self._repair_directive(
                    (verification[0], verification[1]),
                    ledger,
                    contract,
                )
                previous = current_node_id
                current = tree.node(current_node_id)
                current_node_id = current.parent_id or tree.root_id
                trace["events"].append(
                    {
                        "type": "forced_backtrack",
                        "repair_round": repair_round,
                        "from_node_id": previous,
                        "to_node_id": current_node_id,
                        "directive": repair_directive,
                    }
                )
                stagnant_observations = 0
                continue

            visible = self._visible_frontier(tree, expanded, current_node_id)
            if (
                deterministic_sequence_action is not None
                and deterministic_sequence_action.node_id is not None
                and deterministic_sequence_action.node_id
                not in {node.id for node in visible}
            ):
                visible = [
                    tree.node(deterministic_sequence_action.node_id),
                    *visible,
                ]
            if deterministic_sequence_action is not None:
                planned = [deterministic_sequence_action]
                trace["events"].append(
                    {
                        "type": (
                            "sequence_clue_refinement"
                            if refinement_action is not None
                            else "sequence_temporal_sweep"
                        ),
                        "action": deterministic_sequence_action.to_dict(),
                    }
                )
            else:
                try:
                    planned = self._plan(
                        question,
                        options,
                        contract,
                        ledger,
                        current_node_id=current_node_id,
                        visible_nodes=visible,
                        expanded_node_ids=expanded,
                        observation_signatures=observed_signatures,
                        repair_directive=repair_directive,
                        resources=resources,
                        trace=trace,
                    )
                except (ProtocolError, ModelCallLimit) as exc:
                    trace["protocol_failures"].append(
                        {"role": "planner", "error": f"{type(exc).__name__}: {exc}"}
                    )
                    planned = []
            normalized_planned: list[PlannedAction] = []
            for planned_action in planned:
                normalized = self._required_mode_action(
                    planned_action,
                    contract,
                    subtitles,
                )
                normalized_planned.append(normalized)
                if normalized != planned_action:
                    trace["events"].append(
                        {
                            "type": "controller_mode_override",
                            "reason": "the evidence contract normalized observation mode",
                            "from_action": planned_action.to_dict(),
                            "to_action": normalized.to_dict(),
                        }
                    )
            action, rejected = self._select_action(
                normalized_planned,
                tree,
                visible,
                contract,
                subtitles,
                expanded,
                current_node_id,
                observed_signatures,
                search_observations,
            )
            if action is None:
                action = self._fallback_action(
                    tree,
                    visible,
                    contract,
                    subtitles,
                    expanded,
                    current_node_id,
                    observed_signatures,
                    ledger,
                )
                trace["events"].append(
                    {
                        "type": "controller_fallback",
                        "reason": "no legal planner action",
                        "rejected": rejected,
                        "action": action.to_dict(),
                    }
                )

            if (
                action.kind in {"observe", "compare"}
                and self._required_mode_action(action, contract, subtitles) != action
            ):
                previous_action = action
                action = self._required_mode_action(action, contract, subtitles)
                trace["events"].append(
                    {
                        "type": "controller_mode_override",
                        "reason": "the evidence contract normalized observation mode",
                        "from_action": previous_action.to_dict(),
                        "to_action": action.to_dict(),
                    }
                )

            if action.kind in {"verify", "answer"}:
                if search_observations >= next_verification_at:
                    observation_limit = search_observations
                    continue
                action = self._fallback_action(
                    tree,
                    visible,
                    contract,
                    subtitles,
                    expanded,
                    current_node_id,
                    observed_signatures,
                    ledger,
                    require_observation=True,
                )

            if action.kind in {"expand", "zoom_out", "shift"} and (
                navigation_actions >= self.config.max_navigation_actions - 1
            ):
                action = self._fallback_action(
                    tree,
                    visible,
                    contract,
                    subtitles,
                    expanded,
                    current_node_id,
                    observed_signatures,
                    ledger,
                    require_observation=True,
                )

            if action.kind == "expand":
                expanded.add(str(action.node_id))
                current_node_id = str(action.node_id)
                navigation_actions += 1
                trace["events"].append({"type": "expand", "action": action.to_dict()})
                continue
            if action.kind == "zoom_out":
                previous = current_node_id
                current_node_id = tree.node(current_node_id).parent_id or tree.root_id
                navigation_actions += 1
                trace["events"].append(
                    {
                        "type": "zoom_out",
                        "from_node_id": previous,
                        "to_node_id": current_node_id,
                        "action": action.to_dict(),
                    }
                )
                continue
            if action.kind == "shift":
                previous = current_node_id
                current_node_id = str(action.node_id)
                navigation_actions += 1
                trace["events"].append(
                    {
                        "type": "shift",
                        "from_node_id": previous,
                        "to_node_id": current_node_id,
                        "action": action.to_dict(),
                    }
                )
                continue

            if navigation_actions >= self.config.max_navigation_actions:
                action = self._fallback_action(
                    tree,
                    visible,
                    contract,
                    subtitles,
                    expanded,
                    current_node_id,
                    observed_signatures,
                    ledger,
                    require_observation=True,
                )
            try:
                added = self._execute_observation(
                    question,
                    options,
                    contract,
                    action,
                    cached,
                    subtitles,
                    tree,
                    ledger,
                    resources,
                    trace,
                )
            except ModelCallLimit:
                trace["evidence_ledger"] = ledger.to_list()
                return (
                    self._best_effort_option(options, ledger, candidate_history),
                    False,
                    "model_call_budget_exhausted_during_observation",
                )
            observed_signatures.append(action.signature())
            search_observations += 1
            if action.node_id is not None:
                current_node_id = action.node_id
            navigation_actions = 0
            if added:
                stagnant_observations = 0
            else:
                stagnant_observations += 1
            if stagnant_observations >= self.config.max_stagnant_observations:
                previous = current_node_id
                current_node_id = tree.node(current_node_id).parent_id or tree.root_id
                repair_directive = "Repeated observations added no new evidence; shift branch."
                trace["events"].append(
                    {
                        "type": "stagnation_backtrack",
                        "from_node_id": previous,
                        "to_node_id": current_node_id,
                    }
                )
                stagnant_observations = 0

    def _compile_contract(
        self,
        question: str,
        subtitles_available: bool,
        resources: ResourceLedger,
        trace: dict[str, Any],
    ) -> TaskContract:
        prompt = build_task_compiler_prompt(
            question,
            subtitles_available=subtitles_available,
        )
        try:
            contract = self._structured_call(
                "task_compiler",
                prompt,
                lambda text: parse_task_contract(
                    text,
                    subtitles_available=subtitles_available,
                ),
                resources,
                trace,
                max_new_tokens=self.config.compiler_max_new_tokens,
            )
            return self._guard_contract(question, contract, subtitles_available)
        except ProtocolError as exc:
            trace["protocol_failures"].append(
                {"role": "task_compiler", "error": f"{type(exc).__name__}: {exc}"}
            )
            return self._heuristic_contract(question, subtitles_available)

    def _breadth_observe(
        self,
        question: str,
        contract: TaskContract,
        cached: CachedVideo,
        subtitles: SubtitleTrack | None,
        tree: SceneTree,
        ledger: EvidenceLedger,
        resources: ResourceLedger,
        trace: dict[str, Any],
    ) -> tuple[int, list[tuple[Any, ...]]]:
        nodes = tree.children(tree.root_id) or [tree.root]
        frames: list[FrameRef] = []
        labels: list[str] = []
        frame_lines: list[str] = []
        signatures: list[tuple[Any, ...]] = []
        for node in nodes:
            node_frames = self.tree_builder.storyboard_frames(cached, node)
            signatures.append(PlannedAction("observe", node.id, "overview").signature())
            for frame in node_frames:
                frames.append(frame)
                label = f"{node.id} {frame.timestamp_seconds:.1f}s"
                labels.append(label)
                frame_lines.append(f"{frame.id} @ {frame.timestamp_seconds:.3f}s -> {node.id}")
        contact_sheet = build_contact_sheet(
            frames,
            labels,
            output_dir=f"{cached.cache_dir}/active_tree_contacts",
            columns=self.config.contact_sheet_columns,
        )
        subtitles_by_node = {
            node.id: self._subtitle_for_nodes(
                subtitles,
                [node],
                max_chars=self.config.breadth_subtitle_chars_per_node,
            )
            for node in nodes
        }
        prompt = build_breadth_prompt(
            question,
            contract,
            nodes,
            frame_lines,
            subtitles_by_node,
        )
        parser = lambda text: parse_observer(
            text,
            valid_node_ids={node.id for node in nodes},
            valid_slot_ids={slot.slot_id for slot in contract.slots},
            valid_option_ids=set(),
            valid_frame_ids={frame.id for frame in frames},
        )
        try:
            decision = self._structured_call(
                "breadth_observer",
                prompt,
                parser,
                resources,
                trace,
                frames=frames,
                images=[contact_sheet],
                frames_as_video=False,
                max_new_tokens=self.config.breadth_max_new_tokens,
            )
        except ProtocolError as exc:
            trace["protocol_failures"].append(
                {"role": "breadth_observer", "error": f"{type(exc).__name__}: {exc}"}
            )
            decision = ObserverDecision((), "breadth observer protocol failed")
        evidence = self._materialize_evidence(
            decision,
            tree,
            contract,
            options=[],
            mode="breadth",
            default_slot_id=None,
            subtitles_available=subtitles is not None,
            allowed_subtitle_text="\n".join(subtitles_by_node.values()),
            allowed_frame_times={frame.id: frame.timestamp_seconds for frame in frames},
            next_index=len(ledger.items) + 1,
            sequence_min_gap_seconds=self.config.temporal_min_gap_seconds,
        )
        added = ledger.add(evidence)
        trace["evidence_ledger"] = ledger.to_list()
        trace["events"].append(
            {
                "type": "root_breadth_observation",
                "node_ids": [node.id for node in nodes],
                "contact_sheet": contact_sheet,
                "frames": [frame.to_dict() for frame in frames],
                "subtitles_by_node": subtitles_by_node,
                "new_evidence_count": added,
                "missing_evidence": decision.missing_evidence,
            }
        )
        return added, signatures

    def _discriminate(
        self,
        question: str,
        options: list[CanonicalOption],
        initial: TaskContract,
        ledger: EvidenceLedger,
        resources: ResourceLedger,
        trace: dict[str, Any],
    ) -> TaskContract:
        prompt = build_discriminator_prompt(question, options, initial, ledger)
        try:
            return self._structured_call(
                "option_discriminator",
                prompt,
                lambda text: parse_discriminator(text, initial=initial, options=options),
                resources,
                trace,
                max_new_tokens=self.config.discriminator_max_new_tokens,
            )
        except ProtocolError as exc:
            trace["protocol_failures"].append(
                {"role": "option_discriminator", "error": f"{type(exc).__name__}: {exc}"}
            )
            return parse_discriminator("{}", initial=initial, options=options)

    def _plan(
        self,
        question: str,
        options: list[CanonicalOption],
        contract: TaskContract,
        ledger: EvidenceLedger,
        *,
        current_node_id: str,
        visible_nodes: list[SceneNode],
        expanded_node_ids: set[str],
        observation_signatures: list[tuple[Any, ...]],
        repair_directive: str,
        resources: ResourceLedger,
        trace: dict[str, Any],
    ) -> list[PlannedAction]:
        prompt = build_planner_prompt(
            question,
            options,
            contract,
            ledger,
            current_node_id=current_node_id,
            visible_nodes=visible_nodes,
            expanded_node_ids=expanded_node_ids,
            observation_signatures=observation_signatures,
            repair_directive=repair_directive,
        )
        return self._structured_call(
            "planner",
            prompt,
            parse_planner,
            resources,
            trace,
            max_new_tokens=self.config.planner_max_new_tokens,
        )

    def _execute_observation(
        self,
        question: str,
        options: list[CanonicalOption],
        contract: TaskContract,
        action: PlannedAction,
        cached: CachedVideo,
        subtitles: SubtitleTrack | None,
        tree: SceneTree,
        ledger: EvidenceLedger,
        resources: ResourceLedger,
        trace: dict[str, Any],
    ) -> int:
        node_ids = (
            list(action.compare_node_ids)
            if action.kind == "compare"
            else [str(action.node_id)]
        )
        nodes = [tree.node(node_id) for node_id in node_ids]
        frames, frame_lines = self._observation_frames(cached, nodes, action.mode or "inspect")
        subtitle_text = self._subtitle_for_nodes(
            subtitles,
            nodes,
            max_chars=self.config.subtitle_max_chars,
        )
        contact_sheet: str | None = None
        images: list[str] | None = None
        frames_as_video = True
        if action.mode == "event_verify" and frames:
            contact_sheet = build_contact_sheet(
                frames,
                [f"{frame.id} {frame.timestamp_seconds:.1f}s" for frame in frames],
                output_dir=f"{cached.cache_dir}/active_tree_contacts",
                columns=min(len(frames), self.config.contact_sheet_columns),
            )
            images = [contact_sheet]
            frames_as_video = False
        prompt = build_observer_prompt(
            question,
            contract,
            action=action,
            nodes=nodes,
            frame_lines=frame_lines,
            subtitles=subtitle_text,
        )
        parser = lambda text: parse_observer(
            text,
            valid_node_ids=set(node_ids),
            valid_slot_ids={slot.slot_id for slot in contract.slots},
            valid_option_ids={item.option_id for item in options},
            valid_frame_ids={frame.id for frame in frames},
            prefer_frame_endpoints=action.mode == "event_verify",
        )
        try:
            decision = self._structured_call(
                "observer",
                prompt,
                parser,
                resources,
                trace,
                frames=frames,
                images=images,
                frames_as_video=frames_as_video,
                max_new_tokens=self.config.observer_max_new_tokens,
            )
        except ProtocolError as exc:
            trace["protocol_failures"].append(
                {"role": "observer", "error": f"{type(exc).__name__}: {exc}"}
            )
            decision = ObserverDecision((), "observer protocol failed")
        evidence = self._materialize_evidence(
            decision,
            tree,
            contract,
            options,
            mode=action.mode or "inspect",
            default_slot_id=action.slot_id,
            subtitles_available=subtitles is not None,
            allowed_subtitle_text=subtitle_text,
            allowed_frame_times={frame.id: frame.timestamp_seconds for frame in frames},
            next_index=len(ledger.items) + 1,
            sequence_min_gap_seconds=self.config.temporal_min_gap_seconds,
        )
        active_before = len(ledger.active_items)
        routing_before = len(ledger.routing_items)
        added = ledger.add(evidence)
        active_added = len(ledger.active_items) - active_before
        routing_added = len(ledger.routing_items) - routing_before
        trace["evidence_ledger"] = ledger.to_list()
        trace["events"].append(
            {
                "type": "active_observation",
                "action": action.to_dict(),
                "node_ids": node_ids,
                "frames": [frame.to_dict() for frame in frames],
                "contact_sheet": contact_sheet,
                "subtitles": subtitle_text,
                "new_evidence_count": added,
                "new_active_evidence_count": active_added,
                "new_routing_clue_count": routing_added,
                "missing_evidence": decision.missing_evidence,
            }
        )
        return added

    def _verify(
        self,
        question: str,
        options: list[CanonicalOption],
        contract: TaskContract,
        ledger: EvidenceLedger,
        cached: CachedVideo,
        resources: ResourceLedger,
        trace: dict[str, Any],
        *,
        attempt: int,
    ) -> tuple[
        VerificationDecision,
        VerificationDecision,
        EvidenceAlignment,
        TemporalComposition,
    ]:
        frames, frame_lines = self._verification_media(
            ledger,
            cached,
        )
        alignment = align_evidence_to_options(
            options,
            ledger.active_items,
            min_phrase_tokens=self.config.alignment_min_phrase_tokens,
            min_margin=self.config.alignment_min_margin,
        )
        temporal = compose_temporal_option(
            options,
            contract,
            ledger,
            min_gap_seconds=self.config.temporal_min_gap_seconds,
        )
        alignment_option = next(
            (item for item in options if item.option_id == alignment.option_id),
            None,
        )
        controller_hints: list[str] = []
        if alignment.confident and alignment_option is not None:
            controller_hints.append(
                f'Candidate text: "{alignment_option.text}"; '
                f"grounding: {alignment.evidence_id}; "
                f'exact option-unique phrase: "{alignment.matched_phrase}". '
                "Accept only if this directly answers the question."
            )
        temporal_option = next(
            (item for item in options if item.option_id == temporal.option_id),
            None,
        )
        if temporal.confident and temporal_option is not None:
            timeline = ", ".join(
                f"{slot_id}@{timestamp:.1f}s"
                for slot_id, timestamp in temporal.timestamps
            )
            controller_hints.append(
                f'Temporal candidate text: "{temporal_option.text}"; '
                f"composition: {temporal.option_pattern}; timeline: {timeline}; "
                f"grounding: {','.join(temporal.evidence_ids)}. "
                "Accept only if every event is visibly/spoken grounded."
            )
        valid_evidence_ids = {item.evidence_id for item in ledger.active_items}
        completeness_prompt = build_completeness_prompt(
            question,
            options,
            contract,
            ledger,
            frame_lines=frame_lines,
            controller_hint="\n".join(controller_hints),
        )
        try:
            completeness = self._structured_call(
                "completeness_verifier",
                completeness_prompt,
                lambda text: parse_verification(
                    text,
                    options=options,
                    valid_evidence_ids=valid_evidence_ids,
                ),
                resources,
                trace,
                frames=frames,
                max_new_tokens=self.config.verifier_max_new_tokens,
            )
        except (ProtocolError, ModelCallLimit) as exc:
            trace["protocol_failures"].append(
                {"role": "completeness_verifier", "error": f"{type(exc).__name__}: {exc}"}
            )
            completeness = VerificationDecision(
                None, False, False, (), "", str(exc), "", None, False
            )

        shuffled = self._shuffled_options(question, options)
        skeptic_prompt = build_skeptic_prompt(
            question,
            shuffled,
            contract,
            ledger,
            frame_lines=frame_lines,
        )
        try:
            skeptic = self._structured_call(
                "blind_skeptic",
                skeptic_prompt,
                lambda text: parse_verification(
                    text,
                    options=options,
                    valid_evidence_ids=valid_evidence_ids,
                ),
                resources,
                trace,
                frames=frames,
                max_new_tokens=self.config.verifier_max_new_tokens,
            )
        except (ProtocolError, ModelCallLimit) as exc:
            trace["protocol_failures"].append(
                {"role": "blind_skeptic", "error": f"{type(exc).__name__}: {exc}"}
            )
            skeptic = VerificationDecision(
                None, False, False, (), "", str(exc), "", None, False
            )

        trace["verification_attempts"].append(
            {
                "attempt": attempt,
                "frame_ids": [frame.id for frame in frames],
                "shuffled_option_order": [item.option_id for item in shuffled],
                "evidence_alignment": alignment.to_dict(),
                "temporal_composition": temporal.to_dict(),
                "completeness": self._verification_dict(completeness),
                "skeptic": self._verification_dict(skeptic),
            }
        )
        return completeness, skeptic, alignment, temporal

    def _structured_call(
        self,
        role: str,
        prompt: str,
        parser: Callable[[str], T],
        resources: ResourceLedger,
        trace: dict[str, Any],
        *,
        frames: list[FrameRef] | None = None,
        images: list[str] | None = None,
        frames_as_video: bool = True,
        max_new_tokens: int,
    ) -> T:
        output = self._model_call(
            role,
            prompt,
            resources,
            frames=frames,
            images=images,
            frames_as_video=frames_as_video,
            max_new_tokens=max_new_tokens,
        )
        try:
            return parser(output.text)
        except Exception as first_error:  # noqa: BLE001 - role parser boundary
            current_error = first_error
            raw_response = output.text
            for attempt in range(self.config.protocol_repair_attempts):
                trace["protocol_failures"].append(
                    {
                        "role": role,
                        "attempt": attempt + 1,
                        "error": f"{type(current_error).__name__}: {current_error}",
                        "raw_response": raw_response,
                    }
                )
                repaired = self._model_call(
                    f"{role}_protocol_repair",
                    repair_prompt(prompt, raw_response, current_error),
                    resources,
                    max_new_tokens=max_new_tokens,
                )
                raw_response = repaired.text
                try:
                    return parser(repaired.text)
                except Exception as exc:  # noqa: BLE001 - role parser boundary
                    current_error = exc
            raise ProtocolError(f"{role} response remained invalid: {current_error}") from current_error

    def _model_call(
        self,
        role: str,
        prompt: str,
        resources: ResourceLedger,
        *,
        frames: list[FrameRef] | None = None,
        images: list[str] | None = None,
        frames_as_video: bool = True,
        max_new_tokens: int,
    ) -> ModelOutput:
        resources.ensure_call_available()
        output = self.model.generate(
            [{"role": "user", "content": prompt}],
            videos=(
                [[frame.path for frame in frames]]
                if frames and frames_as_video
                else None
            ),
            images=images,
            max_new_tokens=max_new_tokens,
            temperature=0.0,
        )
        resources.record_call(
            role=role,
            prompt=prompt,
            raw_response=output.text,
            model_metadata=output.metadata,
            frames=frames,
            images=images,
        )
        return output

    @staticmethod
    def _sequence_breadth_actions(
        contract: TaskContract,
        ledger: EvidenceLedger,
    ) -> list[PlannedAction]:
        if contract.primary_topology != "sequence" or "visual" not in contract.required_modalities:
            return []
        actions: list[PlannedAction] = []
        for slot in contract.slots:
            if not slot.required:
                continue
            candidates = [
                item
                for item in ledger.items
                if item.observation_mode == "breadth"
                and slot.slot_id in item.slot_ids
                and item.modality in {"visual", "ocr"}
                and item.source_frame_ids
            ]
            if not candidates:
                continue
            ranked = sorted(
                (
                    (
                        ActiveTreeVideoAgent._lexical_overlap(slot.description, item.fact),
                        item,
                    )
                    for item in candidates
                ),
                key=lambda value: (value[0], value[1].start_seconds),
                reverse=True,
            )
            score, evidence = ranked[0]
            if score < 0.3:
                continue
            mode = ActiveTreeVideoAgent._slot_observation_mode(
                contract,
                slot.slot_id,
            )
            if mode != "subtitle" and len(set(evidence.source_frame_ids)) < 2:
                continue
            actions.append(
                PlannedAction(
                    "observe",
                    node_id=evidence.node_id,
                    mode=mode,
                    slot_id=slot.slot_id,
                    expected_new_evidence=(
                        f"verify breadth clue for {slot.slot_id}: {slot.description}"
                    ),
                )
            )
        return actions

    @staticmethod
    def _slot_observation_mode(contract: TaskContract, slot_id: str) -> str:
        slot = next((item for item in contract.slots if item.slot_id == slot_id), None)
        if (
            slot is not None
            and "subtitle" in contract.required_modalities
            and re_search(
                r"\b(introduction|introduc\w*|said|says|tell\w*|ask\w*|announc\w*)\b",
                slot.description.casefold(),
            )
        ):
            return "subtitle"
        return "motion"

    @staticmethod
    def _sequence_sweep_action(
        tree: SceneTree,
        contract: TaskContract,
        ledger: EvidenceLedger,
        observed_signatures: list[tuple[Any, ...]],
    ) -> PlannedAction | None:
        if contract.primary_topology != "sequence":
            return None
        missing = ledger.missing_slot_ids(contract)
        if not missing:
            return None
        mode = ActiveTreeVideoAgent._slot_observation_mode(contract, missing[0])
        observed_modes = (
            {"subtitle"}
            if mode == "subtitle"
            else {"inspect", "motion", "event_verify", "detail_ocr"}
        )
        observed_node_ids = {
            str(signature[1])
            for signature in observed_signatures
            if signature
            and signature[0] in {"observe", "compare"}
            and signature[2] in observed_modes
        }
        root_nodes = sorted(
            tree.children(tree.root_id) or [tree.root],
            key=lambda node: node.start_seconds,
        )
        target = next(
            (node for node in root_nodes if node.id not in observed_node_ids),
            None,
        )
        if target is None:
            return None
        return PlannedAction(
            "observe",
            node_id=target.id,
            mode=mode,
            slot_id=missing[0],
            expected_new_evidence=(
                f"temporally sweep uncovered branch for missing slot {missing[0]}"
            ),
        )

    @staticmethod
    def _sequence_refinement_action(
        tree: SceneTree,
        contract: TaskContract,
        ledger: EvidenceLedger,
        observed_signatures: list[tuple[Any, ...]],
    ) -> PlannedAction | None:
        if contract.primary_topology != "sequence":
            return None
        observed_visual_nodes = {
            str(signature[1])
            for signature in observed_signatures
            if signature
            and signature[0] in {"observe", "compare"}
            and signature[2] in {"inspect", "motion", "event_verify", "detail_ocr"}
        }
        for slot_id in ledger.missing_slot_ids(contract):
            for clue in reversed(ledger.routing_items):
                if slot_id not in clue.slot_ids:
                    continue
                children = tree.children(clue.node_id)
                if not children:
                    continue
                midpoint = (clue.start_seconds + clue.end_seconds) / 2
                target = min(
                    children,
                    key=lambda node: (
                        0
                        if node.start_seconds <= midpoint <= node.end_seconds
                        else 1,
                        abs(node.midpoint_seconds - midpoint),
                        -node.boundary_score,
                        node.id,
                    ),
                )
                if target.id in observed_visual_nodes:
                    continue
                return PlannedAction(
                    "observe",
                    node_id=target.id,
                    mode="event_verify",
                    slot_id=slot_id,
                    expected_new_evidence=(
                        f"refine non-decisive clue for missing slot {slot_id}"
                    ),
                )
        return None

    @staticmethod
    def _lexical_overlap(reference: str, candidate: str) -> float:
        import re

        stopwords = {
            "after",
            "away",
            "before",
            "from",
            "into",
            "magician",
            "person",
            "then",
            "video",
            "with",
        }

        def tokens(value: str) -> set[str]:
            return {
                token
                for token in re.findall(r"[a-z0-9]+", value.casefold())
                if len(token) >= 3 and token not in stopwords
            }

        expected = tokens(reference)
        if not expected:
            return 0.0
        return len(expected & tokens(candidate)) / len(expected)

    @staticmethod
    def _sequence_event_compatible(reference: str, candidate: str) -> bool:
        """High-precision predicate gate for target-blind sequence captions."""

        families = (
            (
                (
                    r"\b(remov\w*|uncover\w*|lift\w*|reveal\w*|disappear\w*|"
                    r"no longer (?:visible|cover\w*)|"
                    r"not visible in (?:the )?last|"
                    r"(?:take|takes|took|taken|taking)\b.{0,32}\b(?:away|off)|"
                    r"pull\w*\b.{0,24}\boff)\b"
                ),
                "remove",
            ),
            (
                (
                    r"\b(produc\w*|conjur\w*|materializ\w*|appear\w*|reveal\w*|"
                    r"becom\w* visible|now visible|"
                    r"pull\w*\b.{0,24}\bout)\b"
                ),
                "produce",
            ),
            (r"\b(introduc\w*|self[ -]?introduction|my name is)\b", "introduce"),
            (r"\b(open\w*)\b", "open"),
            (r"\b(clos\w*|shut\w*)\b", "close"),
            (r"\b(enter\w*|arriv\w*|come in)\b", "enter"),
            (r"\b(exit\w*|leav\w*|left|go(?:es|ing)? out)\b", "leave"),
            (r"\b(sit\w*|sat|seated)\b", "sit"),
            (r"\b(stand\w*|stood)\b", "stand"),
            (r"\b(pick\w* up|lift\w* up)\b", "pick_up"),
            (r"\b(put\w* down|set\w* down|plac\w*)\b", "put_down"),
            (r"\b(giv\w*|gave|hand\w* to)\b", "give"),
            (r"\b(receiv\w*|accept\w*)\b", "receive"),
            (r"\b(eat\w*|ate|chew\w*)\b", "eat"),
            (r"\b(drink\w*|drank|sip\w*)\b", "drink"),
            (r"\b(throw\w*|threw|toss\w*)\b", "throw"),
            (r"\b(catch\w*|caught)\b", "catch"),
            (r"\b(walk\w*)\b", "walk"),
            (r"\b(run\w*|ran)\b", "run"),
            (r"\b(danc\w*)\b", "dance"),
        )
        expected = {name for pattern, name in families if re_search(pattern, reference)}
        if not expected:
            return True
        observed = {name for pattern, name in families if re_search(pattern, candidate)}
        negated = re_search(
            r"\b(not|never|no longer|didn't|doesn't|isn't|wasn't)\b",
            candidate,
        )
        grounded_disappearance = re_search(
            r"\b(no longer (?:visible|cover\w*)|"
            r"not visible in (?:the )?last(?: frame)?)\b",
            candidate,
        )
        if negated and not grounded_disappearance:
            return False
        return bool(expected & observed)

    def _subtitle_retrieval_action(
        self,
        options: list[CanonicalOption],
        contract: TaskContract,
        subtitles: SubtitleTrack | None,
        tree: SceneTree,
    ) -> tuple[PlannedAction | None, EvidenceAlignment]:
        routing_evidence: list[AtomicEvidence] = []
        if (
            self.config.enable_subtitle_retrieval_proposal
            and subtitles is not None
            and "subtitle" in contract.required_modalities
        ):
            slot_ids = tuple(sorted(contract.required_slot_ids()))
            for node in tree.children(tree.root_id) or [tree.root]:
                text = " ".join(
                    cue.text for cue in subtitles.cues if cue.overlaps(node.as_window())
                ).strip()
                if not text:
                    continue
                routing_evidence.append(
                    AtomicEvidence(
                        evidence_id=f"ROUTE::{node.id}",
                        node_id=node.id,
                        slot_ids=slot_ids,
                        start_seconds=node.start_seconds,
                        end_seconds=node.end_seconds,
                        modality="subtitle",
                        fact=text,
                        supports_option_ids=(),
                        refutes_option_ids=(),
                        source_frame_ids=(),
                        subtitle_refs=(),
                        observation_mode="retrieval_probe",
                    )
                )
        alignment = align_evidence_to_options(
            options,
            tuple(routing_evidence),
            min_phrase_tokens=self.config.alignment_min_phrase_tokens,
            min_margin=self.config.alignment_min_margin,
        )
        if not alignment.confident or alignment.evidence_id is None:
            return None, alignment
        node_id = alignment.evidence_id.removeprefix("ROUTE::")
        if node_id not in tree.nodes:
            return None, alignment
        slot_id = next(iter(sorted(contract.required_slot_ids())), None)
        return (
            PlannedAction(
                "observe",
                node_id=node_id,
                mode="subtitle",
                slot_id=slot_id,
                expected_new_evidence=(
                    f'ground the option-unique phrase "{alignment.matched_phrase}"'
                ),
            ),
            alignment,
        )

    @staticmethod
    def _required_mode_action(
        action: PlannedAction,
        contract: TaskContract,
        subtitles: SubtitleTrack | None,
    ) -> PlannedAction:
        if action.kind not in {"observe", "compare"}:
            return action
        desired_mode: str | None = None
        if contract.primary_topology == "sequence" and action.slot_id is not None:
            desired_mode = ActiveTreeVideoAgent._slot_observation_mode(
                contract,
                action.slot_id,
            )
            if desired_mode == "subtitle" and subtitles is None:
                desired_mode = "motion"
            if desired_mode == "motion" and action.mode == "event_verify":
                desired_mode = "event_verify"
        elif contract.required_modalities == ["subtitle"] and subtitles is not None:
            desired_mode = "subtitle"
        elif contract.required_modalities == ["visual"] and action.mode in {
            "subtitle",
            "detail_ocr",
        }:
            desired_mode = "motion" if contract.primary_topology == "sequence" else "inspect"
        elif contract.required_modalities == ["ocr"]:
            desired_mode = "detail_ocr"
        if desired_mode is None or action.mode == desired_mode:
            return action
        return PlannedAction(
            action.kind,
            node_id=action.node_id,
            mode=desired_mode,
            slot_id=action.slot_id,
            compare_node_ids=action.compare_node_ids,
            expected_new_evidence=action.expected_new_evidence,
        )

    def _select_action(
        self,
        planned: list[PlannedAction],
        tree: SceneTree,
        visible: list[SceneNode],
        contract: TaskContract,
        subtitles: SubtitleTrack | None,
        expanded: set[str],
        current_node_id: str,
        observed_signatures: list[tuple[Any, ...]],
        search_observations: int,
    ) -> tuple[PlannedAction | None, list[dict[str, Any]]]:
        rejected: list[dict[str, Any]] = []
        for action in planned:
            reason = self._illegal_reason(
                action,
                tree,
                visible,
                contract,
                subtitles,
                expanded,
                current_node_id,
                observed_signatures,
                search_observations,
            )
            if reason is None:
                return action, rejected
            rejected.append({"action": action.to_dict(), "reason": reason})
        return None, rejected

    def _illegal_reason(
        self,
        action: PlannedAction,
        tree: SceneTree,
        visible: list[SceneNode],
        contract: TaskContract,
        subtitles: SubtitleTrack | None,
        expanded: set[str],
        current_node_id: str,
        observed_signatures: list[tuple[Any, ...]],
        search_observations: int,
    ) -> str | None:
        visible_ids = {node.id for node in visible}
        slot_ids = {slot.slot_id for slot in contract.slots}
        if action.kind == "expand":
            if action.node_id not in visible_ids:
                return "expand target is not visible"
            if tree.node(str(action.node_id)).is_leaf:
                return "cannot expand a leaf"
            if action.node_id in expanded:
                return "node is already expanded"
        elif action.kind == "zoom_out":
            if current_node_id == tree.root_id:
                return "already at root"
        elif action.kind == "shift":
            sibling_ids = {node.id for node in tree.siblings(current_node_id)}
            if current_node_id == tree.root_id:
                sibling_ids = visible_ids
            if action.node_id not in sibling_ids:
                return "shift target is not a sibling"
        elif action.kind == "observe":
            if action.node_id not in visible_ids:
                return "observe target is not visible"
            if action.mode is None:
                return "observation mode is missing"
            if action.mode == "subtitle" and subtitles is None:
                return "subtitle observation unavailable"
            if action.slot_id is not None and action.slot_id not in slot_ids:
                return "unknown evidence slot"
            if action.signature() in observed_signatures:
                return "duplicate observation"
        elif action.kind == "compare":
            compare_ids = set(action.compare_node_ids)
            if len(compare_ids) != 2 or not compare_ids <= visible_ids:
                return "compare requires two distinct visible nodes"
            if action.signature() in observed_signatures:
                return "duplicate comparison"
        elif action.kind in {"verify", "answer"}:
            if search_observations < self.config.min_active_observations:
                return "mandatory active observation not completed"
        return None

    def _fallback_action(
        self,
        tree: SceneTree,
        visible: list[SceneNode],
        contract: TaskContract,
        subtitles: SubtitleTrack | None,
        expanded: set[str],
        current_node_id: str,
        observed_signatures: list[tuple[Any, ...]],
        ledger: EvidenceLedger,
        *,
        require_observation: bool = False,
    ) -> PlannedAction:
        missing = ledger.missing_slot_ids(contract)
        slot_id = missing[0] if missing else contract.slots[0].slot_id
        current = tree.node(current_node_id)
        if (
            not require_observation
            and current_node_id != tree.root_id
            and not current.is_leaf
            and current_node_id not in expanded
        ):
            return PlannedAction(
                "expand",
                node_id=current_node_id,
                slot_id=slot_id,
                expected_new_evidence="refine the currently selected branch",
            )
        current_children = tree.children(current_node_id) if current_node_id in expanded else []
        candidates = current_children or visible
        candidates = sorted(
            candidates,
            key=lambda node: (node.boundary_score, node.duration_seconds),
            reverse=True,
        )
        if not require_observation:
            expandable = [
                node for node in candidates if not node.is_leaf and node.id not in expanded
            ]
            if expandable:
                return PlannedAction(
                    "expand",
                    node_id=expandable[0].id,
                    slot_id=slot_id,
                    expected_new_evidence="deterministic expansion of highest-change branch",
                )
        modes = ["inspect", "motion", "detail_ocr", "overview"]
        if subtitles is not None and "subtitle" in contract.required_modalities:
            modes.insert(0, "subtitle")
        for node in candidates:
            for mode in modes:
                action = PlannedAction(
                    "observe",
                    node_id=node.id,
                    mode=mode,
                    slot_id=slot_id,
                    expected_new_evidence="deterministic unresolved-slot observation",
                )
                if action.signature() not in observed_signatures:
                    return action
        raise ProtocolError("controller has no non-duplicate fallback observation")

    def _visible_frontier(
        self,
        tree: SceneTree,
        expanded: set[str],
        current_node_id: str,
    ) -> list[SceneNode]:
        priority_ids: list[str] = []

        def add(node_id: str) -> None:
            if node_id not in priority_ids:
                priority_ids.append(node_id)

        for node in tree.children(current_node_id):
            add(node.id)
        for node in tree.siblings(current_node_id):
            add(node.id)
        for node in tree.children(tree.root_id):
            add(node.id)
        for node_id in sorted(expanded):
            for node in tree.children(node_id):
                add(node.id)
        if current_node_id != tree.root_id:
            add(current_node_id)
        return [tree.node(node_id) for node_id in priority_ids[: self.config.frontier_max_nodes]]

    def _observation_frames(
        self,
        cached: CachedVideo,
        nodes: list[SceneNode],
        mode: str,
    ) -> tuple[list[FrameRef], list[str]]:
        frames: list[FrameRef] = []
        lines: list[str] = []
        for node in nodes:
            if mode == "subtitle":
                node_frames: list[FrameRef] = []
            elif mode == "overview":
                node_frames = self.tree_builder.storyboard_frames(cached, node)
            elif mode == "motion":
                node_frames = cached.uniform_frames(self.config.motion_frames, node.as_window())
            elif mode == "event_verify":
                last_inside = max(
                    node.start_seconds,
                    node.end_seconds - (1.0 / cached.sample_fps),
                )
                node_frames = [
                    cached.nearest_frame(node.start_seconds),
                    cached.nearest_frame(last_inside),
                ]
            elif mode == "detail_ocr":
                node_frames = self.tree_builder.storyboard_frames(cached, node)[
                    : self.config.detail_frames
                ]
            else:
                node_frames = cached.uniform_frames(self.config.inspect_frames, node.as_window())
            for frame in node_frames:
                frames.append(frame)
                lines.append(f"{frame.id} @ {frame.timestamp_seconds:.3f}s -> {node.id}")
        deduplicated: list[FrameRef] = []
        seen: set[str] = set()
        for frame in frames:
            if frame.id in seen:
                continue
            seen.add(frame.id)
            deduplicated.append(frame)
        return deduplicated, lines

    def _verification_media(
        self,
        ledger: EvidenceLedger,
        cached: CachedVideo,
    ) -> tuple[list[FrameRef], list[str]]:
        audit_items = ledger.active_items or ledger.items
        frame_ids = {
            frame_id
            for item in audit_items
            for frame_id in item.source_frame_ids
        }
        by_id = {frame.id: frame for frame in cached.frames}
        frames = [by_id[frame_id] for frame_id in frame_ids if frame_id in by_id]
        if not frames:
            node_windows = list(audit_items[-4:])
            for item in node_windows:
                midpoint = (item.start_seconds + item.end_seconds) / 2
                frames.append(cached.nearest_frame(midpoint))
        frames = sorted(
            {frame.id: frame for frame in frames}.values(),
            key=lambda frame: frame.timestamp_seconds,
        )[:16]
        frame_lines = [f"{frame.id} @ {frame.timestamp_seconds:.3f}s" for frame in frames]
        return frames, frame_lines

    @staticmethod
    def _subtitle_for_nodes(
        subtitles: SubtitleTrack | None,
        nodes: list[SceneNode],
        *,
        max_chars: int,
    ) -> str:
        if subtitles is None or not nodes:
            return ""
        return subtitles.text_for_windows(
            [node.as_window() for node in nodes],
            max_chars=max_chars,
        )

    @staticmethod
    def _materialize_evidence(
        decision: ObserverDecision,
        tree: SceneTree,
        contract: TaskContract,
        options: list[CanonicalOption],
        *,
        mode: str,
        default_slot_id: str | None,
        subtitles_available: bool,
        allowed_subtitle_text: str,
        allowed_frame_times: dict[str, float],
        next_index: int,
        sequence_min_gap_seconds: float = 0.5,
    ) -> list[AtomicEvidence]:
        valid_slots = {slot.slot_id for slot in contract.slots}
        result: list[AtomicEvidence] = []
        for offset, fact in enumerate(decision.facts):
            node = tree.node(fact.node_id)
            if fact.modality == "subtitle" and not subtitles_available:
                continue
            subtitle_refs = fact.subtitle_refs
            fact_text = fact.fact
            source_frame_ids = fact.source_frame_ids
            if (
                contract.primary_topology == "sequence"
                and mode == "event_verify"
                and len(source_frame_ids) < 2
                and ActiveTreeVideoAgent._explicit_frame_transition(fact_text)
                and len(allowed_frame_times) >= 2
            ):
                ordered_frames = sorted(
                    allowed_frame_times,
                    key=allowed_frame_times.__getitem__,
                )
                source_frame_ids = (ordered_frames[0], ordered_frames[-1])
            if fact.modality in {"visual", "ocr"} and not source_frame_ids:
                continue
            if fact.modality == "subtitle":
                relevant_slots = " ".join(
                    slot.description
                    for slot in contract.slots
                    if slot.slot_id in fact.slot_ids
                )
                subtitle_refs = ActiveTreeVideoAgent._select_grounded_subtitle_refs(
                    fact.subtitle_refs,
                    allowed_subtitle_text,
                    relevance_text=f"{fact.fact} {relevant_slots}",
                    limit=2,
                )
                if not subtitle_refs:
                    continue
                fact_text = " | ".join(subtitle_refs)
            cited_times: list[float] = []
            if fact.modality in {"visual", "ocr"}:
                cited_times = [
                    allowed_frame_times[frame_id]
                    for frame_id in source_frame_ids
                    if frame_id in allowed_frame_times
                ]
                if not cited_times:
                    continue
                start = min(cited_times)
                end = max(cited_times)
            else:
                start = min(node.end_seconds, max(node.start_seconds, fact.start_seconds))
                end = min(node.end_seconds, max(start, fact.end_seconds))
                grounded_times = ActiveTreeVideoAgent._subtitle_reference_times(
                    subtitle_refs
                )
                if grounded_times:
                    start = max(node.start_seconds, min(item[0] for item in grounded_times))
                    end = min(node.end_seconds, max(item[1] for item in grounded_times))
            if fact.modality == "subtitle" and (
                fact.end_seconds < node.start_seconds or end <= start
            ):
                start = node.start_seconds
                end = node.end_seconds
            if contract.primary_topology == "sequence" and fact.modality in {
                "visual",
                "ocr",
            }:
                ranked_slots = sorted(
                    (
                        (
                            ActiveTreeVideoAgent._lexical_overlap(
                                slot.description,
                                fact_text,
                            ),
                            slot.slot_id,
                        )
                        for slot in contract.slots
                    ),
                    reverse=True,
                )
                best_score, best_slot_id = ranked_slots[0]
                runner_up = ranked_slots[1][0] if len(ranked_slots) > 1 else 0.0
                best_slot = next(
                    slot for slot in contract.slots if slot.slot_id == best_slot_id
                )
                reliable_match = best_score >= 0.3 and best_score - runner_up >= 0.1
                predicate_matches = ActiveTreeVideoAgent._sequence_event_compatible(
                    best_slot.description,
                    fact_text,
                )
                distinct_times = sorted(set(cited_times))
                transition_grounded = (
                    len(distinct_times) >= 2
                    and distinct_times[-1] - distinct_times[0]
                    >= sequence_min_gap_seconds
                )
                slot_ids = (best_slot_id,) if reliable_match else ()
                routing_only = reliable_match and not (
                    predicate_matches and transition_grounded
                )
            else:
                slot_ids = tuple(item for item in fact.slot_ids if item in valid_slots)
                if not slot_ids and default_slot_id in valid_slots:
                    slot_ids = (str(default_slot_id),)
                if not slot_ids and len(valid_slots) == 1:
                    slot_ids = tuple(valid_slots)
            if contract.primary_topology == "sequence" and not slot_ids:
                continue
            observation_mode = (
                f"{mode}_routing"
                if contract.primary_topology == "sequence"
                and fact.modality in {"visual", "ocr"}
                and mode != "breadth"
                and routing_only
                else mode
            )
            result.append(
                AtomicEvidence(
                    evidence_id=f"EV{next_index + offset:04d}",
                    node_id=node.id,
                    slot_ids=slot_ids,
                    start_seconds=start,
                    end_seconds=end,
                    modality=fact.modality,
                    fact=fact_text,
                    supports_option_ids=(),
                    refutes_option_ids=(),
                    source_frame_ids=source_frame_ids,
                    subtitle_refs=subtitle_refs,
                    observation_mode=observation_mode,
                )
            )
        return result

    @staticmethod
    def _subtitle_reference_times(
        references: tuple[str, ...],
    ) -> tuple[tuple[float, float], ...]:
        import re

        result: list[tuple[float, float]] = []
        for reference in references:
            match = re.search(
                r"\[?(\d+(?:\.\d+)?)s?-(\d+(?:\.\d+)?)s?\]?",
                reference,
            )
            if match is None:
                continue
            start, end = float(match.group(1)), float(match.group(2))
            result.append((min(start, end), max(start, end)))
        return tuple(result)

    @staticmethod
    def _explicit_frame_transition(fact: str) -> bool:
        before = re_search(r"\b(first frame|initially|at first|before)\b", fact)
        after = re_search(
            r"\b(last frame|subsequent frames?|later|afterward|then|no longer|now)\b",
            fact,
        )
        return before and after

    @staticmethod
    def _subtitle_ref_is_grounded(reference: str, subtitle_text: str) -> bool:
        def normalize(value: str) -> str:
            import re

            value = re.sub(r"\[?\d+(?:\.\d+)?s?-\d+(?:\.\d+)?s?\]?", " ", value)
            value = re.sub(r"[^a-z0-9]+", " ", value.casefold())
            return " ".join(value.split())

        needle = normalize(reference)
        haystack = normalize(subtitle_text)
        if not needle or not haystack:
            return False
        if needle in haystack:
            return True
        words = needle.split()
        return len(words) >= 4 and " ".join(words[-4:]) in haystack

    @staticmethod
    def _select_grounded_subtitle_refs(
        references: tuple[str, ...],
        subtitle_text: str,
        *,
        relevance_text: str,
        limit: int,
    ) -> tuple[str, ...]:
        import re

        stopwords = {
            "about",
            "after",
            "before",
            "being",
            "characters",
            "discussing",
            "including",
            "people",
            "person",
            "should",
            "their",
            "there",
            "these",
            "those",
            "video",
            "which",
            "with",
        }

        def tokens(value: str) -> set[str]:
            value = re.sub(r"\[?\d+(?:\.\d+)?s?-\d+(?:\.\d+)?s?\]?", " ", value)
            return {
                token
                for token in re.findall(r"[a-z0-9]+", value.casefold())
                if len(token) >= 3 and token not in stopwords
            }

        grounded: list[tuple[int, str]] = []
        seen: set[str] = set()
        for index, reference in enumerate(references):
            normalized = " ".join(reference.casefold().split())
            if normalized in seen or not ActiveTreeVideoAgent._subtitle_ref_is_grounded(
                reference,
                subtitle_text,
            ):
                continue
            seen.add(normalized)
            grounded.append((index, reference))
        query_tokens = tokens(relevance_text)
        ranked = sorted(
            grounded,
            key=lambda item: (
                len(tokens(item[1]) & query_tokens),
                len(tokens(item[1])),
                -item[0],
            ),
            reverse=True,
        )[:limit]
        return tuple(value for _, value in sorted(ranked))

    @staticmethod
    def _verification_passed(
        completeness: VerificationDecision,
        skeptic: VerificationDecision,
        ledger: EvidenceLedger,
        contract: TaskContract,
        alignment: EvidenceAlignment,
        temporal: TemporalComposition,
    ) -> tuple[bool, str | None]:
        candidate = completeness.candidate_option_id
        passed = (
            candidate is not None
            and candidate == skeptic.candidate_option_id
            and completeness.sufficient
            and skeptic.sufficient
            and completeness.citations_valid
            and skeptic.citations_valid
            and not completeness.missing_slot_ids
            and not skeptic.missing_slot_ids
            and not ledger.missing_slot_ids(contract)
            and (not alignment.confident or candidate == alignment.option_id)
            and (not temporal.confident or candidate == temporal.option_id)
        )
        return passed, candidate if passed else None

    def _alignment_adjudication(
        self,
        completeness: VerificationDecision,
        skeptic: VerificationDecision,
        alignment: EvidenceAlignment,
        ledger: EvidenceLedger,
        contract: TaskContract,
    ) -> tuple[bool, str | None]:
        if (
            not self.config.allow_alignment_adjudication
            or not alignment.confident
            or alignment.option_id is None
            or alignment.evidence_id is None
            or ledger.missing_slot_ids(contract)
        ):
            return False, None
        for role, decision in (
            ("completeness_verifier", completeness),
            ("blind_skeptic", skeptic),
        ):
            if (
                decision.candidate_option_id == alignment.option_id
                and decision.decisive_evidence == alignment.evidence_id
                and decision.citations_valid
                and not decision.missing_slot_ids
            ):
                return True, role
        return False, None

    def _temporal_adjudication(
        self,
        completeness: VerificationDecision,
        skeptic: VerificationDecision,
        temporal: TemporalComposition,
        ledger: EvidenceLedger,
        contract: TaskContract,
    ) -> tuple[bool, str | None]:
        if (
            not self.config.allow_temporal_adjudication
            or not temporal.confident
            or temporal.option_id is None
            or ledger.missing_slot_ids(contract)
        ):
            return False, None
        valid_evidence = set(temporal.evidence_ids)
        for role, decision in (
            ("completeness_verifier", completeness),
            ("blind_skeptic", skeptic),
        ):
            if (
                decision.candidate_option_id == temporal.option_id
                and decision.decisive_evidence in valid_evidence
                and decision.citations_valid
                and not decision.missing_slot_ids
            ):
                return True, role
        return False, None

    @staticmethod
    def _repair_directive(
        verification: tuple[VerificationDecision, VerificationDecision],
        ledger: EvidenceLedger,
        contract: TaskContract,
    ) -> str:
        missing = set(ledger.missing_slot_ids(contract))
        for decision in verification:
            missing.update(decision.missing_slot_ids)
        counter = " | ".join(
            item.counterevidence for item in verification if item.counterevidence
        )
        reasons = " | ".join(item.reason for item in verification if item.reason)
        return (
            f"Re-open a different branch and collect evidence for slots "
            f"{','.join(sorted(missing)) or 'under-supported claims'}. "
            f"Counterevidence: {counter or 'none stated'}. Audit: {reasons}"
        )

    @staticmethod
    def _best_effort_option(
        options: list[CanonicalOption],
        ledger: EvidenceLedger,
        candidate_history: list[str],
    ) -> str:
        if candidate_history:
            counts = {
                option.option_id: candidate_history.count(option.option_id)
                for option in options
            }
            return max(options, key=lambda item: (counts[item.option_id], -int(item.option_id[1:])))\
                .option_id
        scores = {option.option_id: 0 for option in options}
        for item in ledger.items:
            for option_id in item.supports_option_ids:
                if option_id in scores:
                    scores[option_id] += 1
            for option_id in item.refutes_option_ids:
                if option_id in scores:
                    scores[option_id] -= 1
        return max(options, key=lambda item: (scores[item.option_id], -int(item.option_id[1:])))\
            .option_id

    @staticmethod
    def _heuristic_contract(question: str, subtitles_available: bool) -> TaskContract:
        lowered = question.casefold()
        if re_search(r"\b(not|except|never|isn't|is not)\b", lowered):
            topology = "exclusion"
            criterion = "elimination"
        elif re_search(r"\b(before|after|first|then|next|order)\b", lowered):
            topology = "sequence"
            criterion = "mixed"
        elif re_search(r"\b(how many|both|each|all|times)\b", lowered):
            topology = "multi_set"
            criterion = "mixed"
        elif re_search(r"\b(main|overall|throughout|mostly|primary)\b", lowered):
            topology = "global"
            criterion = "coverage"
        else:
            topology = "local"
            criterion = "direct_support"
        modalities = ["visual"]
        if subtitles_available and re_search(
            r"\b(say|says|said|tell|talk|argu|discuss|ask|name|called)\w*\b",
            lowered,
        ):
            modalities = ["subtitle"]
        slots = [EvidenceSlot("S1", "Identify the specific event or subject asked about")]
        if modalities == ["subtitle"]:
            slots = [
                EvidenceSlot(
                    "S1",
                    "Identify the exact spoken subject or statement requested by the question",
                )
            ]
        if topology == "sequence":
            slots = [
                EvidenceSlot("S1", "Find the first event"),
                EvidenceSlot("S2", "Find the later event and verify their order"),
            ]
        elif topology == "multi_set":
            slots = [
                EvidenceSlot("S1", "Find all relevant occurrences"),
                EvidenceSlot("S2", "Verify their count or membership"),
            ]
        elif topology == "global":
            slots = [EvidenceSlot("S1", "Establish representative whole-video coverage")]
        elif topology == "exclusion":
            slots = [EvidenceSlot("S1", "Test the strongest competing options for support")]
        return TaskContract(topology, slots, modalities, answer_criterion=criterion)

    @staticmethod
    def _guard_contract(
        question: str,
        contract: TaskContract,
        subtitles_available: bool,
    ) -> TaskContract:
        generic_descriptions = {
            "observable fact to find",
            "relevant evidence",
            "answer the question",
        }
        heuristic = ActiveTreeVideoAgent._heuristic_contract(question, subtitles_available)
        slots = contract.slots
        if any(slot.description.casefold().strip() in generic_descriptions for slot in slots):
            slots = heuristic.slots
        dialogue_question = subtitles_available and re_search(
            r"\b(say|says|said|tell|talk|argu|discuss|ask|name|called)\w*\b",
            question.casefold(),
        )
        modalities = ["subtitle"] if dialogue_question else contract.required_modalities
        dialogue_subject_question = re_search(
            r"\bwhat\b.{0,80}\b(argu\w*|discuss\w*|talk\w*)\b.{0,30}\babout\b",
            question.casefold(),
        )
        if dialogue_subject_question:
            slots = heuristic.slots
        topology = "local" if dialogue_subject_question else contract.primary_topology
        enumerated_action_sequence = (
            topology == "sequence"
            and "(a)" in question.casefold()
            and "(b)" in question.casefold()
            and not re_search(
                r"\b(introduced|listed|mentioned|explained|said)\b",
                question.casefold(),
            )
        )
        if enumerated_action_sequence:
            has_spoken_slot = subtitles_available and any(
                re_search(
                    r"\b(introduction|introduc\w*|said|says|tell\w*|ask\w*|announc\w*)\b",
                    slot.description.casefold(),
                )
                for slot in slots
            )
            modalities = ["visual", *( ["subtitle"] if has_spoken_slot else [])]
            slots = [
                EvidenceSlot(
                    slot.slot_id,
                    slot.description,
                    slot.required,
                    (
                        "ground the spoken event and its timestamp"
                        if has_spoken_slot
                        and re_search(
                            r"\b(introduction|introduc\w*|said|says|tell\w*|ask\w*|announc\w*)\b",
                            slot.description.casefold(),
                        )
                        else "ground the visible event and its timestamp"
                    ),
                )
                for slot in slots
            ]
        return TaskContract(
            topology,
            slots,
            modalities,
            contract.option_tests,
            contract.answer_criterion,
        )

    @staticmethod
    def _shuffled_options(
        question: str,
        options: list[CanonicalOption],
    ) -> list[CanonicalOption]:
        if len(options) < 2:
            return list(options)
        digest = int(hashlib.sha1(question.encode("utf-8")).hexdigest()[:8], 16)
        offset = digest % len(options)
        rotated = options[offset:] + options[:offset]
        return list(reversed(rotated))

    @staticmethod
    def _verification_dict(decision: VerificationDecision) -> dict[str, Any]:
        return {
            "candidate_option_id": decision.candidate_option_id,
            "sufficient": decision.sufficient,
            "citations_valid": decision.citations_valid,
            "missing_slot_ids": list(decision.missing_slot_ids),
            "counterevidence": decision.counterevidence,
            "reason": decision.reason,
            "decisive_evidence": decision.decisive_evidence,
            "strongest_alternative_option_id": decision.strongest_alternative_option_id,
            "alternative_refuted": decision.alternative_refuted,
        }

    @staticmethod
    def _latest_user_text(messages: Sequence[Mapping[str, Any]]) -> str:
        for message in reversed(messages):
            if message.get("role") != "user":
                continue
            content = message.get("content", "")
            if isinstance(content, str):
                return content.strip()
            if isinstance(content, list):
                texts = [
                    str(part.get("text", ""))
                    for part in content
                    if isinstance(part, Mapping) and part.get("type") == "text"
                ]
                return "\n".join(item for item in texts if item).strip()
        return ""


def re_search(pattern: str, value: str) -> bool:
    import re

    return re.search(pattern, value, flags=re.IGNORECASE) is not None
