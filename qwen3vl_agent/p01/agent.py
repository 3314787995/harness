from __future__ import annotations

import logging
import math
import re
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, TypeVar

from qwen3vl_agent.coarse_to_fine.cache import build_contact_sheet
from qwen3vl_agent.coarse_to_fine.types import FrameRef
from qwen3vl_agent.models.base import BaseVideoModel, ModelOutput
from qwen3vl_agent.p01.config import P01Config
from qwen3vl_agent.p01.control import (
    bounded_automatic_span,
    build_coverage_manifest,
    build_refinement_plan,
    canonicalize_options,
    choose_candidate,
    evenly_spaced_timestamps,
    forced_option,
    interval_chunks,
    merge_evidence,
    missing_required_slots,
    option_label,
    parse_question_interval,
    reconcile_interval,
    resolve_options,
    uniform_timestamps,
    unique_entailed_option,
    unresolved_status,
    validate_fact_provenance,
)
from qwen3vl_agent.p01.media import P01IndexBuilder, P01VideoIndex, SourceFrameStore
from qwen3vl_agent.p01.prompts import (
    ScoutDecision,
    build_answer_composer_prompt,
    build_choice_decision_prompt,
    build_hypothesis_compiler_prompt,
    build_locator_prompt,
    build_observation_compiler_prompt,
    build_protocol_repair_prompt,
    build_refinement_prompt,
    build_rescue_locator_prompt,
    build_scout_prompt,
    build_verifier_prompt,
    compile_decision_spec,
    extract_option_id_from_text,
    parse_choice_decision,
    parse_decision_spec,
    parse_locator,
    parse_observation_spec,
    parse_rescue_locator,
    parse_scout,
    parse_verifier,
)
from qwen3vl_agent.p01.types import (
    CandidateObservation,
    CanonicalOption,
    ChoiceDecision,
    ClaimTest,
    ClaimVerdict,
    ContractViolation,
    DecisionSpec,
    EventFact,
    EvidenceGrade,
    EvidencePacket,
    Fact,
    LocatorCandidate,
    ObservationSlot,
    ObservationSpec,
    OptionAssessment,
    OptionVerdict,
    P01Request,
    P01Result,
    ProtocolError,
    RefinementPlan,
    ResourceExhausted,
    ResourceLedger,
    SourceView,
    StaticFact,
    TextFact,
    TimeSpan,
)

logger = logging.getLogger(__name__)
_ParsedT = TypeVar("_ParsedT")


@dataclass
class _RunState:
    request: P01Request
    ledger: ResourceLedger
    trace: dict[str, Any]
    started: float
    frame_catalog: dict[str, FrameRef] = field(default_factory=dict)
    current_packet: EvidencePacket | None = None
    degraded_stages: list[dict[str, str]] = field(default_factory=list)
    safe_budget_retries: int = 0
    rescue_used: bool = False

    def remember(self, frames: Sequence[FrameRef]) -> None:
        self.frame_catalog.update((frame.id, frame) for frame in frames)


class P01VideoAgent:
    """Training-free, single-span visual evidence executor for known P01 requests."""

    def __init__(
        self,
        model: BaseVideoModel,
        *,
        config: P01Config | Mapping[str, Any] | None = None,
        index_builder: P01IndexBuilder | None = None,
        source_store: SourceFrameStore | None = None,
    ) -> None:
        self.model = model
        self.config = config if isinstance(config, P01Config) else P01Config.from_mapping(config)
        self.config.validate()
        self.index_builder = index_builder or P01IndexBuilder(self.config)
        self.source_store = source_store or SourceFrameStore(self.config)
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
        given_interval: TimeSpan | Sequence[float] | None = None,
        force_choice: bool = False,
        **_: Any,
    ) -> ModelOutput:
        if not self._loaded:
            raise RuntimeError("Agent is not loaded. Call load() first.")
        if images:
            raise ValueError("p01 strategy accepts video input only")
        if subtitle_path:
            raise ValueError("p01 does not accept subtitles")
        if not videos or len(videos) != 1 or not isinstance(videos[0], str):
            raise ValueError("p01 strategy requires exactly one video")
        question = self._latest_user_text(messages)
        if not question:
            raise ValueError("messages must contain a non-empty user question")
        request = P01Request(
            video_path=videos[0],
            question=question,
            choices=tuple(choices or ()),
            given_interval=self._coerce_interval(given_interval),
            force_choice=force_choice,
        )
        result = self.solve(request)
        return ModelOutput(text=result.output_text, metadata={"p01": result.to_dict()})

    def solve(self, request: P01Request) -> P01Result:
        if not self._loaded:
            raise RuntimeError("Agent is not loaded. Call load() first.")
        started = time.perf_counter()
        state = _RunState(
            request=request,
            ledger=ResourceLedger(self.config.max_model_calls),
            trace={
                "strategy": "p01",
                "prompt_version": "p01-v2",
                "question": request.question,
                "choices": list(request.choices),
                "video_path": request.video_path,
                "given_interval": (
                    request.given_interval.to_dict() if request.given_interval is not None else None
                ),
                "protocol_repairs": [],
                "locator_rounds": [],
                "candidate_scouts": [],
                "span_changes": [],
                "bounded_rescue": None,
                "deprecated_force_choice_received": request.force_choice,
            },
            started=started,
        )
        try:
            result = self._run(state)
        except (ContractViolation, FileNotFoundError) as exc:
            result = self._failure(state, "input_contract_violation", exc)
        except ProtocolError as exc:
            result = self._failure(state, "protocol_error", exc)
        except ResourceExhausted as exc:
            result = self._failure(state, "resource_exhausted", exc)
        return self._with_final_resources(result, state)

    def _run(self, state: _RunState) -> P01Result:
        request = state.request
        probe = getattr(self.index_builder, "probe", None)
        prepare_interval = getattr(self.index_builder, "prepare_interval", None)
        metadata = None
        if callable(probe) and callable(prepare_interval):
            try:
                metadata = probe(request.video_path)
            except ValueError:
                metadata = None
        if metadata is not None:
            parsed_interval = parse_question_interval(
                request.question,
                metadata.duration_seconds,
            )
            explicit_span = reconcile_interval(
                request.given_interval,
                parsed_interval,
                metadata.duration_seconds,
            )
            index = (
                prepare_interval(request.video_path, explicit_span, metadata=metadata)
                if explicit_span is not None
                else self.index_builder.prepare(request.video_path)
            )
            state.trace["index_scope"] = (
                "explicit_interval" if explicit_span is not None else "full_video"
            )
        else:
            index = self.index_builder.prepare(request.video_path)
            parsed_interval = parse_question_interval(
                request.question,
                index.duration_seconds,
            )
            explicit_span = reconcile_interval(
                request.given_interval,
                parsed_interval,
                index.duration_seconds,
            )
            state.trace["index_scope"] = "full_video"
        state.trace["video_index"] = index.to_dict()
        state.trace["parsed_interval"] = (
            parsed_interval.to_dict() if parsed_interval is not None else None
        )
        state.trace["resolved_interval"] = (
            explicit_span.to_dict() if explicit_span is not None else None
        )

        answer_mode = "multiple_choice" if request.choices else "free_text"
        spec = self._compile_observation_spec(state, answer_mode)
        state.trace["observation_spec"] = spec.to_dict()
        options = canonicalize_options(request.choices)

        try:
            located = self._initial_candidates(state, index, spec, explicit_span)
        except (ProtocolError, ResourceExhausted) as exc:
            self._degrade(state, "locator", exc)
            located = ()

        observations: list[CandidateObservation] = []
        if explicit_span is not None and located:
            try:
                observations.append(
                    self._scout_explicit_interval(state, index, spec, explicit_span)
                )
            except (ProtocolError, ResourceExhausted) as exc:
                self._degrade(state, "interval_scout", exc)
        else:
            for candidate in located:
                node = index.node(candidate.node_id)
                try:
                    observation = self._scout_candidate(
                        state,
                        index,
                        spec,
                        candidate_id=candidate.node_id,
                        span=node.span,
                        locator_rank=candidate.rank,
                        locator_anchor=candidate.visible_anchor,
                        explicit=False,
                    )
                except (ProtocolError, ResourceExhausted) as exc:
                    self._degrade(state, f"candidate_scout:{candidate.node_id}", exc)
                    continue
                observations.append(observation)

        state.trace["candidate_scouts"] = [item.to_dict() for item in observations]
        selected = (
            observations[0]
            if explicit_span is not None and observations
            else choose_candidate(observations)
        )
        self._record_candidate_ranking(state, spec, observations, selected)

        exempt_from_auto_limit = (
            explicit_span is not None
            or index.duration_seconds <= self.config.short_video_threshold_sec
        )
        packet = (
            self._canonicalize_selected(
                selected,
                index,
                spec,
                explicit_span=explicit_span,
                exempt_from_auto_limit=exempt_from_auto_limit,
                state=state,
            )
            if selected is not None
            else None
        )
        state.current_packet = packet

        decision_spec = self._compile_decision_spec_v2(state, options, spec) if options else None
        if decision_spec is not None:
            state.trace["decision_spec"] = decision_spec.to_dict()

        refinement_used = False
        if packet is not None:
            packet, refinement_used = self._maybe_refine(
                state,
                index,
                spec,
                packet,
                decision_spec,
                explicit_span=explicit_span,
                exempt_from_auto_limit=exempt_from_auto_limit,
            )
            state.current_packet = packet

        if not options:
            rescue_reasons = self._predecision_rescue_reasons(spec, packet, selected)
            if rescue_reasons:
                packet, spec = self._bounded_rescue(
                    state,
                    index,
                    spec,
                    packet,
                    decision_spec=None,
                    observations=observations,
                    reasons=rescue_reasons,
                    explicit_span=explicit_span,
                )
                if packet is not None and not refinement_used:
                    packet, _ = self._maybe_refine(
                        state,
                        index,
                        spec,
                        packet,
                        None,
                        explicit_span=explicit_span,
                        exempt_from_auto_limit=exempt_from_auto_limit,
                    )
            if packet is None:
                packet = self._deterministic_fallback_packet(state, index, spec, explicit_span)
            state.current_packet = packet
            return self._compose_free_text_v2(state, spec, packet)

        assert decision_spec is not None
        if packet is None:
            packet, spec = self._bounded_rescue(
                state,
                index,
                spec,
                None,
                decision_spec=decision_spec,
                observations=observations,
                reasons=("no_local_anchor",),
                explicit_span=explicit_span,
            )
            if packet is None:
                packet = self._deterministic_fallback_packet(state, index, spec, explicit_span)
            state.current_packet = packet
            decision = self._choice_decision(
                state,
                index,
                spec,
                packet,
                options,
                decision_spec,
                role="final_decision",
            )
            return self._mcq_result_v2(
                state,
                spec,
                packet,
                options,
                decision_spec,
                decision,
                decision_source=(
                    "terminal_fallback"
                    if decision.reason.startswith("terminal fallback")
                    else "rescued"
                    if state.rescue_used
                    else "initial"
                ),
            )

        initial_decision = self._choice_decision(
            state,
            index,
            spec,
            packet,
            options,
            decision_spec,
            role="initial_decision",
        )
        state.trace["initial_decision"] = initial_decision.to_dict()
        initial_grade = self._grade_evidence(spec, packet, decision_spec, initial_decision)
        rescue_reasons = self._decision_rescue_reasons(
            spec,
            packet,
            decision_spec,
            initial_decision,
            initial_grade,
            selected,
        )
        state.trace["initial_evidence_grade"] = initial_grade.to_dict()
        if rescue_reasons and self._can_start_rescue(state):
            rescued_packet, rescued_spec = self._bounded_rescue(
                state,
                index,
                spec,
                packet,
                decision_spec=decision_spec,
                observations=observations,
                reasons=rescue_reasons,
                explicit_span=explicit_span,
            )
            if rescued_packet is not None:
                packet, spec = rescued_packet, rescued_spec
                state.current_packet = packet
                final_decision = self._choice_decision(
                    state,
                    index,
                    spec,
                    packet,
                    options,
                    decision_spec,
                    role="final_decision",
                )
                state.trace["final_decision"] = final_decision.to_dict()
                return self._mcq_result_v2(
                    state,
                    spec,
                    packet,
                    options,
                    decision_spec,
                    final_decision,
                    decision_source=(
                        "terminal_fallback"
                        if final_decision.reason.startswith("terminal fallback")
                        else "rescued"
                    ),
                )

        return self._mcq_result_v2(
            state,
            spec,
            packet,
            options,
            decision_spec,
            initial_decision,
            decision_source=(
                "terminal_fallback"
                if initial_decision.reason.startswith("terminal fallback")
                else "initial"
            ),
        )

    def _compile_observation_spec(
        self,
        state: _RunState,
        answer_mode: str,
    ) -> ObservationSpec:
        """Compile the choice-blind observation contract, degrading deterministically."""

        prompt = build_observation_compiler_prompt(
            state.request.question,
            answer_mode=answer_mode,
        )
        try:
            spec = self._structured_call(
                state,
                role="observation_compiler",
                prompt=prompt,
                parser=lambda text: parse_observation_spec(
                    text,
                    expected_answer_mode=answer_mode,
                ),
                schema_hint="ObservationSpec JSON with one to four required slots",
            )
            state.trace["observation_compiler"] = {"kind": "model"}
            return spec
        except (ProtocolError, ResourceExhausted) as exc:
            self._degrade(state, "observation_compiler", exc)
            spec = self._fallback_observation_spec(
                state.request.question,
                answer_mode=answer_mode,
            )
            state.trace["observation_compiler"] = {
                "kind": "deterministic_fallback",
                "reason": str(exc),
            }
            return spec

    @staticmethod
    def _fallback_observation_spec(
        question: str,
        *,
        answer_mode: str,
    ) -> ObservationSpec:
        normalized = " ".join(question.casefold().split())
        ocr_terms = (
            "text",
            "word",
            "written",
            "read",
            "sign",
            "caption",
            "title",
            "number",
            "time",
            "文字",
            "写着",
            "字幕",
            "标牌",
            "数字",
            "时间",
        )
        action_terms = (
            "doing",
            "do next",
            "happen",
            "action",
            "before",
            "after",
            "first",
            "then",
            "how does",
            "what does",
            "动作",
            "做了什么",
            "发生",
            "之前",
            "之后",
            "如何",
        )
        if any(term in normalized for term in ocr_terms):
            mode = "ocr"
            coverage = "text_consensus"
            description = "the exact visible text or number requested by the question"
            detail = ("read the relevant text at original resolution",)
        elif answer_mode == "free_text":
            mode = "subscene_caption"
            coverage = "sequence"
            description = "the complete visible local event, including action and result"
            detail = ("preserve event order and visible result",)
        elif any(term in normalized for term in action_terms):
            mode = "dynamic_action"
            coverage = "sequence"
            description = "the visible action, object, target, order, and result"
            detail = ("inspect action boundaries and outcome",)
        else:
            mode = "static_visual"
            coverage = "point"
            description = "the visible entity, attribute, or relation asked about"
            detail = ()
        return ObservationSpec(
            answer_mode=answer_mode,
            primary_mode=mode,
            required_slots=(ObservationSlot("S1", description),),
            detail_requests=detail,
            coverage_requirement=coverage,
            output_language="same_as_question",
        )

    @staticmethod
    def _degrade(state: _RunState, stage: str, error: Exception | str) -> None:
        item = {
            "stage": stage,
            "error": (error if isinstance(error, str) else f"{type(error).__name__}: {error}"),
        }
        state.degraded_stages.append(item)
        state.trace.setdefault("degraded_stages", []).append(item)

    def _record_candidate_ranking(
        self,
        state: _RunState,
        spec: ObservationSpec,
        observations: Sequence[CandidateObservation],
        selected: CandidateObservation | None,
    ) -> None:
        ranking: list[dict[str, Any]] = []
        for item in observations:
            clear = {
                slot_id
                for fact in item.evidence.facts
                if fact.visibility == "clear"
                for slot_id in fact.slot_ids
            }
            partial = {
                slot_id
                for fact in item.evidence.facts
                if fact.visibility in {"clear", "partial"}
                for slot_id in fact.slot_ids
            } - clear
            ranking.append(
                {
                    "candidate_id": item.candidate_id,
                    "target_visible": item.target_visible,
                    "clear_required_slots": len(clear & spec.required_slot_ids),
                    "partial_required_slots": len(partial & spec.required_slot_ids),
                    "conflicts": len(item.evidence.conflicts),
                    "missing_required_slots": len(item.evidence.missing_slot_ids),
                    "locator_rank": item.locator_rank,
                    "span_seconds": item.supporting_span.duration_seconds,
                    "selected": selected is not None and item.candidate_id == selected.candidate_id,
                }
            )
        state.trace["candidate_ranking"] = ranking

    def _compile_decision_spec_v2(
        self,
        state: _RunState,
        options: Sequence[CanonicalOption],
        spec: ObservationSpec,
    ) -> DecisionSpec:
        prompt = build_hypothesis_compiler_prompt(
            state.request.question,
            options,
            spec,
        )
        try:
            compiled = self._structured_call(
                state,
                role="hypothesis_compiler",
                prompt=prompt,
                parser=lambda text: parse_decision_spec(
                    text,
                    options=options,
                    valid_slot_ids=spec.required_slot_ids,
                ),
                schema_hint="DecisionSpec JSON with atomic claim_tests and every option_rule",
            )
            state.trace["hypothesis_compiler"] = {"kind": "model_label_free_discriminants"}
            return compiled
        except (ProtocolError, ResourceExhausted) as exc:
            self._degrade(state, "hypothesis_compiler", exc)
            compiled = compile_decision_spec(state.request.question, options, spec)
            state.trace["hypothesis_compiler"] = {
                "kind": "deterministic_option_wrapper",
                "reason": str(exc),
            }
            return compiled

    def _maybe_refine(
        self,
        state: _RunState,
        index: P01VideoIndex,
        spec: ObservationSpec,
        packet: EvidencePacket,
        decision_spec: DecisionSpec | None,
        *,
        explicit_span: TimeSpan | None,
        exempt_from_auto_limit: bool,
    ) -> tuple[EvidencePacket, bool]:
        plan = self._sufficiency_plan(spec, packet, decision_spec)
        state.trace["refinement_plan"] = plan.to_dict()
        if not plan.actionable:
            return packet, False
        try:
            refined = self._refine(
                state,
                index,
                spec,
                packet,
                plan,
                decision_spec,
                explicit_span=explicit_span,
                exempt_from_auto_limit=exempt_from_auto_limit,
            )
        except (ProtocolError, ResourceExhausted) as exc:
            self._degrade(state, "refinement_extractor", exc)
            return packet, True
        return refined, True

    def _predecision_rescue_reasons(
        self,
        spec: ObservationSpec,
        packet: EvidencePacket | None,
        selected: CandidateObservation | None,
    ) -> tuple[str, ...]:
        reasons: list[str] = []
        if packet is None:
            reasons.append("no_local_anchor")
            if selected is None:
                reasons.append("all_initial_candidates_invisible")
            return tuple(reasons)
        if packet.missing_slot_ids or missing_required_slots(spec, packet.facts):
            reasons.append("missing_required_slots")
        if packet.conflicts:
            reasons.append("conflicting_observations")
        if validate_fact_provenance(packet):
            reasons.append("invalid_fact_provenance")
        if spec.coverage_requirement == "full_span" and not (
            packet.coverage_manifest.full_span_coverage
        ):
            reasons.append("incomplete_required_coverage")
        if spec.primary_mode == "ocr" and any(
            isinstance(fact, TextFact) and not self._has_ocr_consensus(fact, packet)
            for fact in packet.facts
        ):
            reasons.append("weak_ocr_consensus")
        return tuple(dict.fromkeys(reasons))

    def _can_start_rescue(self, state: _RunState) -> bool:
        return (
            not state.rescue_used
            and self.config.max_rescue_rounds > 0
            and state.ledger.remaining_model_calls > self.config.terminal_call_reserve
        )

    def _bounded_rescue(
        self,
        state: _RunState,
        index: P01VideoIndex,
        spec: ObservationSpec,
        packet: EvidencePacket | None,
        *,
        decision_spec: DecisionSpec | None,
        observations: Sequence[CandidateObservation],
        reasons: Sequence[str],
        explicit_span: TimeSpan | None,
    ) -> tuple[EvidencePacket | None, ObservationSpec]:
        """Run one bounded rescue episode; never widen into a direct full-video answer pass."""

        if state.rescue_used or self.config.max_rescue_rounds < 1:
            return packet, spec
        state.rescue_used = True
        discriminants = decision_spec.claim_tests if decision_spec is not None else ()
        rescue_mode = self._infer_rescue_mode(state.request.question, spec, discriminants)
        rescue_spec = replace(spec, primary_mode=rescue_mode)
        rescue_trace: dict[str, Any] = {
            "reasons": list(dict.fromkeys(reasons)),
            "initial_mode": spec.primary_mode,
            "rescue_mode": rescue_mode,
            "explicit_interval": explicit_span is not None,
            "global_overview_frame_ids": [],
            "locator_anchors": [],
            "candidate_scouts": [],
        }
        state.trace["bounded_rescue"] = rescue_trace

        rescue_observations: list[CandidateObservation] = []
        if explicit_span is not None:
            chunks = interval_chunks(explicit_span, self.config)
            rescue_chunks = self._select_explicit_rescue_chunks(
                index,
                chunks,
                rescue_mode,
            )
            rescue_trace["explicit_rescue_chunks"] = [chunk.to_dict() for chunk in rescue_chunks]
            chunk_packets: list[EvidencePacket] = []
            visible = False
            anchors: list[str] = []
            for chunk_index, chunk in enumerate(rescue_chunks, start=1):
                if state.ledger.remaining_model_calls <= self.config.terminal_call_reserve:
                    break
                try:
                    item = self._scout_candidate(
                        state,
                        index,
                        rescue_spec,
                        candidate_id=f"RESCUE-G42-C{chunk_index}",
                        span=chunk,
                        locator_rank=chunk_index - 1,
                        locator_anchor="bounded explicit-interval rescue",
                        explicit=True,
                        role="rescue_scout",
                        discriminants=discriminants,
                    )
                except (ProtocolError, ResourceExhausted) as exc:
                    self._degrade(state, f"rescue_scout:{chunk_index}", exc)
                    continue
                chunk_packets.append(item.evidence)
                visible = visible or item.target_visible
                anchors.append(item.visible_anchor)
            if chunk_packets:
                combined = self._combine_interval_packets(
                    index,
                    rescue_spec,
                    explicit_span,
                    chunk_packets,
                )
                if len(rescue_chunks) < len(chunks):
                    combined = replace(
                        combined,
                        coverage_manifest=replace(
                            combined.coverage_manifest,
                            full_span_coverage=False,
                        ),
                    )
                rescue_observations.append(
                    CandidateObservation(
                        candidate_id="RESCUE-EXPLICIT",
                        locator_rank=0,
                        target_visible=visible,
                        supporting_span=explicit_span,
                        evidence=combined,
                        visible_anchor=" | ".join(item for item in anchors if item),
                    )
                )
        else:
            overview = self._global_rescue_frames(index)
            state.remember(overview)
            rescue_trace["global_overview_frame_ids"] = [frame.id for frame in overview]
            anchors: tuple[tuple[FrameRef, str], ...] = ()
            if overview and state.ledger.remaining_model_calls > self.config.terminal_call_reserve:
                prompt = build_rescue_locator_prompt(
                    state.request.question,
                    rescue_spec,
                    discriminants,
                    overview,
                    max_candidates=self.config.rescue_candidate_spans,
                )
                try:
                    decision = self._structured_call(
                        state,
                        role="rescue_locator",
                        prompt=prompt,
                        parser=lambda text: parse_rescue_locator(
                            text,
                            frames=overview,
                            max_candidates=self.config.rescue_candidate_spans,
                        ),
                        schema_hint="{candidates:[{frame_id,visible_anchor}]} with one or two candidates",
                        frames=overview,
                        media_paths=tuple(frame.path for frame in overview),
                        media_kind="video",
                        sample_fps=self._effective_fps(
                            [frame.timestamp_seconds for frame in overview],
                            self.config.index_fps,
                        ),
                        sampling={
                            "rule": "bounded_global_rescue_overview",
                            "uniform_fraction": self.config.global_rescue_uniform_fraction,
                            "frame_count": len(overview),
                            "answering_allowed": False,
                        },
                    )
                    by_id = {frame.id: frame for frame in overview}
                    anchors = tuple(
                        (by_id[frame_id], anchor)
                        for frame_id, anchor in zip(
                            decision.frame_ids,
                            decision.visible_anchors,
                        )
                        if frame_id in by_id
                    )
                except (ProtocolError, ResourceExhausted) as exc:
                    self._degrade(state, "rescue_locator", exc)
            if not anchors:
                anchors = self._deterministic_rescue_anchors(index, overview)
                rescue_trace["locator_fallback"] = "controller_salience_ranking"
            original_anchor_ids = [frame.id for frame, _anchor in anchors]
            anchors = self._diversify_rescue_anchors(
                index,
                overview,
                anchors,
                rescue_mode,
            )
            diversified_anchor_ids = [frame.id for frame, _anchor in anchors]
            if diversified_anchor_ids != original_anchor_ids:
                rescue_trace["anchor_diversification"] = {
                    "before": original_anchor_ids,
                    "after": diversified_anchor_ids,
                }
            rescue_trace["locator_anchors"] = [
                {
                    "frame_id": frame.id,
                    "timestamp_seconds": frame.timestamp_seconds,
                    "anchor": anchor,
                }
                for frame, anchor in anchors
            ]
            for rank, (anchor_frame, anchor) in enumerate(
                anchors[: self.config.rescue_candidate_spans]
            ):
                if state.ledger.remaining_model_calls <= self.config.terminal_call_reserve:
                    break
                candidate_span = self._rescue_span_for_anchor(
                    index,
                    anchor_frame.timestamp_seconds,
                    rescue_mode,
                )
                try:
                    item = self._scout_candidate(
                        state,
                        index,
                        rescue_spec,
                        candidate_id=f"RESCUE-{rank + 1}",
                        span=candidate_span,
                        locator_rank=rank,
                        locator_anchor=anchor,
                        explicit=False,
                        role="rescue_scout",
                        discriminants=discriminants,
                    )
                except (ProtocolError, ResourceExhausted) as exc:
                    self._degrade(state, f"rescue_scout:{rank + 1}", exc)
                    continue
                rescue_observations.append(item)

        rescue_trace["candidate_scouts"] = [item.to_dict() for item in rescue_observations]
        selected_rescue = choose_candidate(rescue_observations)
        if selected_rescue is None and rescue_observations:
            selected_rescue = min(
                rescue_observations,
                key=lambda item: (
                    len(item.evidence.missing_slot_ids),
                    len(item.evidence.conflicts),
                    item.locator_rank,
                ),
            )
            rescue_trace["weak_candidate_forced"] = selected_rescue.candidate_id
        if selected_rescue is None:
            rescue_trace["outcome"] = "no_rescue_packet"
            return packet, rescue_spec

        rescued = self._canonicalize_selected(
            selected_rescue,
            index,
            rescue_spec,
            explicit_span=explicit_span,
            exempt_from_auto_limit=(
                explicit_span is not None
                or index.duration_seconds <= self.config.short_video_threshold_sec
            ),
            state=state,
        )
        if packet is not None and self._spans_overlap(
            packet.canonical_span,
            rescued.canonical_span,
        ):
            union = TimeSpan(
                min(packet.canonical_span.start_seconds, rescued.canonical_span.start_seconds),
                max(packet.canonical_span.end_seconds, rescued.canonical_span.end_seconds),
                source="overlapping_rescue_union",
            )
            if explicit_span is not None or union.duration_seconds <= self.config.span_limit(
                rescue_mode
            ):
                rescued = merge_evidence(
                    packet,
                    rescued,
                    canonical_span=union,
                    spec=rescue_spec,
                )
                union_frames = tuple(
                    frame
                    for frame_id in rescued.coverage_manifest.frame_ids
                    if (frame := state.frame_catalog.get(frame_id)) is not None
                )
                union_coverage = build_coverage_manifest(
                    rescue_mode,
                    union,
                    union_frames,
                    sample_fps=rescued.coverage_manifest.sample_fps,
                    shot_ids=index.shot_ids_for_span(union),
                    roi_ids=rescued.coverage_manifest.roi_ids,
                )
                rescued = replace(
                    rescued,
                    coverage_manifest=union_coverage,
                    facts=self._valid_decisive_facts(
                        rescued.facts,
                        union_coverage,
                        union,
                    ),
                )
                rescue_trace["continuous_span_action"] = "merged_overlapping_union"
            else:
                rescue_trace["continuous_span_action"] = "rescued_span_won_without_merge"
        elif packet is not None:
            rescue_trace["continuous_span_action"] = "disjoint_candidates_competed_no_merge"
            if self._packet_rank(spec, packet) <= self._packet_rank(rescue_spec, rescued):
                rescued = packet
                rescue_spec = spec
                rescue_trace["disjoint_winner"] = "initial"
            else:
                rescue_trace["disjoint_winner"] = "rescue"
        rescued = replace(
            rescued,
            missing_slot_ids=missing_required_slots(rescue_spec, rescued.facts),
        )
        rescue_trace["outcome"] = "packet_selected"
        rescue_trace["canonical_span"] = rescued.canonical_span.to_dict()
        return rescued, rescue_spec

    def _select_explicit_rescue_chunks(
        self,
        index: P01VideoIndex,
        chunks: Sequence[TimeSpan],
        mode: str,
    ) -> tuple[TimeSpan, ...]:
        limit = min(self.config.rescue_candidate_spans, len(chunks))
        if len(chunks) <= limit:
            return tuple(chunks)

        def score(chunk: TimeSpan) -> tuple[float, float, float]:
            metrics = [
                item
                for item in index.metrics
                if chunk.start_seconds <= item.timestamp_seconds <= chunk.end_seconds
            ]
            if not metrics:
                return (0.0, 0.0, -chunk.start_seconds)
            if mode == "ocr":
                primary = max(item.text_score for item in metrics)
            elif mode == "static_visual":
                primary = max(item.clarity_score for item in metrics)
            else:
                primary = max(max(item.motion_score, item.change_score) for item in metrics)
            clarity = max(item.clarity_score for item in metrics)
            return (primary, clarity, -chunk.start_seconds)

        selected = sorted(chunks, key=score, reverse=True)[:limit]
        return tuple(sorted(selected, key=lambda chunk: chunk.start_seconds))

    def _global_rescue_frames(self, index: P01VideoIndex) -> tuple[FrameRef, ...]:
        frames = tuple(index.cached_video.frames)
        if len(frames) <= self.config.global_rescue_max_frames:
            return frames
        uniform_count = max(
            1,
            round(
                self.config.global_rescue_max_frames * self.config.global_rescue_uniform_fraction
            ),
        )
        uniform_indices = {
            round(index_value * (len(frames) - 1) / max(1, uniform_count - 1))
            for index_value in range(uniform_count)
        }
        by_id = {frame.id: frame for frame in frames}
        salient = sorted(
            index.metrics,
            key=lambda item: (
                max(item.change_score, item.motion_score, item.text_score),
                item.clarity_score,
                -item.timestamp_seconds,
            ),
            reverse=True,
        )
        selected = [frames[item] for item in sorted(uniform_indices)]
        selected.extend(
            by_id[item.frame_id]
            for item in salient
            if item.frame_id in by_id and item.frame_id not in {frame.id for frame in selected}
        )
        return tuple(
            sorted(
                self._deduplicate_frames(selected)[: self.config.global_rescue_max_frames],
                key=lambda frame: frame.timestamp_seconds,
            )
        )

    def _deterministic_rescue_anchors(
        self,
        index: P01VideoIndex,
        overview: Sequence[FrameRef],
    ) -> tuple[tuple[FrameRef, str], ...]:
        valid = {frame.id: frame for frame in overview}
        ranked = sorted(
            index.metrics,
            key=lambda item: (
                max(item.motion_score, item.change_score, item.text_score),
                item.clarity_score,
                -item.timestamp_seconds,
            ),
            reverse=True,
        )
        result: list[tuple[FrameRef, str]] = []
        minimum_separation = max(1.0, index.duration_seconds / 12.0)
        for metric in ranked:
            frame = valid.get(metric.frame_id)
            if frame is None:
                continue
            if any(
                abs(frame.timestamp_seconds - prior.timestamp_seconds) < minimum_separation
                for prior, _ in result
            ):
                continue
            result.append((frame, "controller-ranked visual salience"))
            if len(result) >= self.config.rescue_candidate_spans:
                break
        if not result and overview:
            result.append((overview[len(overview) // 2], "uniform midpoint fallback"))
        return tuple(result)

    def _diversify_rescue_anchors(
        self,
        index: P01VideoIndex,
        overview: Sequence[FrameRef],
        preferred: Sequence[tuple[FrameRef, str]],
        mode: str,
    ) -> tuple[tuple[FrameRef, str], ...]:
        """Keep rescue scouts in distinct temporal neighborhoods and fill an unused slot."""
        limit = self.config.rescue_candidate_spans
        if limit <= 0:
            return ()
        by_id = {frame.id: frame for frame in overview}
        ranked_metrics = sorted(
            index.metrics,
            key=lambda item: (
                max(item.motion_score, item.change_score, item.text_score),
                item.clarity_score,
                -item.timestamp_seconds,
            ),
            reverse=True,
        )
        controller_candidates = (
            (by_id[item.frame_id], "controller-ranked diverse rescue anchor")
            for item in ranked_metrics
            if item.frame_id in by_id
        )
        selected: list[tuple[FrameRef, str]] = []
        selected_spans: list[TimeSpan] = []
        for frame, anchor in (*preferred, *controller_candidates):
            if frame.id not in by_id or any(frame.id == prior.id for prior, _ in selected):
                continue
            span = self._rescue_span_for_anchor(index, frame.timestamp_seconds, mode)
            if any(self._spans_overlap(span, prior) for prior in selected_spans):
                continue
            selected.append((frame, anchor))
            selected_spans.append(span)
            if len(selected) >= limit:
                break
        if not selected and overview:
            middle = overview[len(overview) // 2]
            selected.append((middle, "uniform midpoint fallback"))
        return tuple(selected)

    def _rescue_span_for_anchor(
        self,
        index: P01VideoIndex,
        timestamp: float,
        mode: str,
    ) -> TimeSpan:
        matching = next(
            (
                shot.span
                for shot in index.shots
                if shot.span.start_seconds - 1e-6 <= timestamp <= shot.span.end_seconds + 1e-6
            ),
            TimeSpan(
                max(0.0, timestamp - 1.0),
                min(index.duration_seconds, max(timestamp + 1.0, 0.001)),
                source="rescue_anchor",
            ),
        )
        return bounded_automatic_span(
            matching,
            mode=mode,
            duration_seconds=index.duration_seconds,
            config=self.config,
        )

    @staticmethod
    def _infer_rescue_mode(
        question: str,
        spec: ObservationSpec,
        discriminants: Sequence[ClaimTest],
    ) -> str:
        text = " ".join([question, *[claim.statement for claim in discriminants]]).casefold()
        if re.search(
            r"\b(text|word|written|read|sign|caption|title|number|clock|time|score|date)\b",
            text,
        ) or any(term in text for term in ("文字", "写着", "数字", "几点", "时间")):
            return "ocr"
        if re.search(
            r"\b(action|doing|does|did|happen|before|after|first|then|result|causes?)\b",
            text,
        ) or any(term in text for term in ("动作", "做了", "发生", "之前", "之后")):
            return "dynamic_action"
        return spec.primary_mode

    @staticmethod
    def _spans_overlap(first: TimeSpan, second: TimeSpan) -> bool:
        return (
            first.end_seconds >= second.start_seconds and second.end_seconds >= first.start_seconds
        )

    @staticmethod
    def _packet_rank(
        spec: ObservationSpec,
        packet: EvidencePacket,
    ) -> tuple[int, int, int, int, float]:
        clear = {
            slot_id
            for fact in packet.facts
            if fact.visibility == "clear"
            for slot_id in fact.slot_ids
        }
        partial = {
            slot_id
            for fact in packet.facts
            if fact.visibility in {"clear", "partial"}
            for slot_id in fact.slot_ids
        } - clear
        return (
            -len(clear & spec.required_slot_ids),
            -len(partial & spec.required_slot_ids),
            len(packet.conflicts),
            len(packet.missing_slot_ids),
            packet.canonical_span.duration_seconds,
        )

    def _deterministic_fallback_packet(
        self,
        state: _RunState,
        index: P01VideoIndex,
        spec: ObservationSpec,
        explicit_span: TimeSpan | None,
    ) -> EvidencePacket:
        if explicit_span is not None:
            span = explicit_span
        else:
            anchors = self._deterministic_rescue_anchors(
                index,
                self._global_rescue_frames(index),
            )
            timestamp = anchors[0][0].timestamp_seconds if anchors else index.duration_seconds / 2
            span = self._rescue_span_for_anchor(index, timestamp, spec.primary_mode)
        frames, fps, _kind, _sampling = self._sample_frames(
            state,
            index,
            spec.primary_mode,
            span,
            explicit=explicit_span is not None,
        )
        coverage = build_coverage_manifest(
            spec.primary_mode,
            span,
            frames,
            sample_fps=fps,
            shot_ids=index.shot_ids_for_span(span),
        )
        view = SourceView(
            view_id="TERMINAL-FALLBACK-VIEW",
            kind="deterministic_local_media_fallback",
            frame_ids=tuple(frame.id for frame in frames),
            media_paths=tuple(frame.path for frame in frames),
            span=span,
        )
        packet = EvidencePacket(
            canonical_span=span,
            facts=(),
            coverage_manifest=coverage,
            missing_slot_ids=tuple(sorted(spec.required_slot_ids)),
            conflicts=(),
            source_views=(view,),
        )
        self._degrade(state, "terminal_packet", "no parsed scout packet; local media retained")
        state.trace["terminal_fallback_packet"] = packet.to_dict()
        return packet

    def _choice_decision(
        self,
        state: _RunState,
        index: P01VideoIndex,
        spec: ObservationSpec,
        packet: EvidencePacket,
        options: Sequence[CanonicalOption],
        decision_spec: DecisionSpec,
        *,
        role: str,
    ) -> ChoiceDecision:
        frames = self._decision_frames(state, packet, spec)
        detail_ids = {frame.id for frame in frames if frame.id.startswith("CROP-")}
        regular = [frame for frame in frames if frame.id not in detail_ids]
        media_kind = (
            "mixed"
            if detail_ids and regular
            else (
                "images" if spec.primary_mode in {"static_visual", "ocr"} or detail_ids else "video"
            )
        )
        fps = (
            None
            if media_kind == "images"
            else self._effective_fps(
                [frame.timestamp_seconds for frame in regular],
                self._mode_fps(spec.primary_mode),
            )
        )
        prompt = build_choice_decision_prompt(
            state.request.question,
            options,
            decision_spec,
            packet,
            frames,
            stage=role,
        )
        raw_responses: list[str] = []
        parse_errors: list[str] = []
        try:
            output = self._model_call(
                state,
                role=role,
                prompt=prompt,
                frames=frames,
                media_paths=tuple(frame.path for frame in frames),
                media_kind=media_kind,
                sample_fps=fps,
                detail_frame_ids=detail_ids,
                sampling={
                    "rule": "local_option_decision",
                    "canonical_span": packet.canonical_span.to_dict(),
                    "frame_count": len(frames),
                },
                terminal=True,
            )
            raw_responses.append(output.text)
            try:
                return parse_choice_decision(
                    output.text,
                    options=options,
                    decision_spec=decision_spec,
                    packet=packet,
                    frames=frames,
                )
            except (ProtocolError, TypeError, ValueError) as exc:
                parse_errors.append(str(exc))
                if self.config.protocol_repair_attempts > 0:
                    repair_prompt = build_protocol_repair_prompt(
                        role,
                        output.text,
                        "ChoiceDecision JSON; preserve exactly one legal selected_option_id and all option assessments",
                    )
                    try:
                        repaired = self._model_call(
                            state,
                            role="protocol_repair",
                            prompt=repair_prompt,
                            sampling={"target_role": role, "media_replayed": False},
                            terminal=True,
                        )
                        raw_responses.append(repaired.text)
                        state.trace["protocol_repairs"].append(
                            {
                                "role": role,
                                "first_error": str(exc),
                                "source_response": output.text,
                                "repair_response": repaired.text,
                                "media_replayed": False,
                            }
                        )
                        try:
                            return parse_choice_decision(
                                repaired.text,
                                options=options,
                                decision_spec=decision_spec,
                                packet=packet,
                                frames=frames,
                            )
                        except (ProtocolError, TypeError, ValueError) as repair_error:
                            parse_errors.append(str(repair_error))
                    except (ResourceExhausted, RuntimeError) as repair_call_error:
                        parse_errors.append(str(repair_call_error))
        except (ResourceExhausted, RuntimeError) as exc:
            parse_errors.append(str(exc))

        for raw in raw_responses:
            extracted = extract_option_id_from_text(raw, options)
            if extracted is not None:
                decision = self._deterministic_option_decision(
                    options,
                    decision_spec,
                    packet,
                    selected_option_id=extracted,
                    reason="terminal fallback: recovered a legal option label from malformed output",
                )
                state.trace.setdefault("decision_fallbacks", []).append(
                    {
                        "role": role,
                        "kind": "legal_label_extraction",
                        "selected_option_id": extracted,
                        "parse_errors": parse_errors,
                    }
                )
                return decision
        decision = self._deterministic_option_decision(
            options,
            decision_spec,
            packet,
            reason="terminal fallback: deterministic evidence score and stable option order",
        )
        state.trace.setdefault("decision_fallbacks", []).append(
            {
                "role": role,
                "kind": "deterministic_evidence_score",
                "selected_option_id": decision.selected_option_id,
                "parse_errors": parse_errors,
            }
        )
        return decision

    def _decision_frames(
        self,
        state: _RunState,
        packet: EvidencePacket,
        spec: ObservationSpec,
    ) -> tuple[FrameRef, ...]:
        allowed_ids = set(packet.coverage_manifest.frame_ids)
        decisive_ids = [
            frame_id
            for fact in packet.facts
            for frame_id in fact.source_frame_ids
            if frame_id in allowed_ids
        ]
        crop_ids = [frame_id for frame_id in allowed_ids if frame_id.startswith("CROP-")]
        chronological = sorted(
            (
                state.frame_catalog[frame_id]
                for frame_id in allowed_ids
                if frame_id in state.frame_catalog
                and packet.canonical_span.contains_decode(
                    state.frame_catalog[frame_id].timestamp_seconds
                )
            ),
            key=lambda frame: frame.timestamp_seconds,
        )
        limit = self._mode_decision_max_frames(spec.primary_mode)
        priority_ids = list(dict.fromkeys((*decisive_ids, *crop_ids)))
        priority = [
            state.frame_catalog[frame_id]
            for frame_id in priority_ids
            if frame_id in state.frame_catalog
        ]
        remaining = max(0, limit - len(priority))
        if remaining and chronological:
            indices = {
                round(value * (len(chronological) - 1) / max(1, remaining - 1))
                for value in range(remaining)
            }
            priority.extend(chronological[index] for index in sorted(indices))
        selected = self._deduplicate_frames(priority)[:limit]
        return tuple(sorted(selected, key=lambda frame: frame.timestamp_seconds))

    def _mode_decision_max_frames(self, mode: str) -> int:
        return {
            "static_visual": self.config.static_max_frames,
            "dynamic_action": self.config.dynamic_refine_max_frames,
            "ocr": self.config.ocr_max_frames + self.config.max_detail_images,
            "subscene_caption": self.config.caption_refine_max_frames,
        }[mode]

    def _deterministic_option_decision(
        self,
        options: Sequence[CanonicalOption],
        decision_spec: DecisionSpec,
        packet: EvidencePacket,
        *,
        selected_option_id: str | None = None,
        reason: str,
    ) -> ChoiceDecision:
        fact_text = " ".join(
            self._normalized_tokens(
                " ".join(
                    [
                        *[fact.statement for fact in packet.facts],
                        *[
                            getattr(fact, "value", "") or getattr(fact, "exact_text", "")
                            for fact in packet.facts
                        ],
                    ]
                )
            )
        )
        fact_tokens = set(fact_text.split())
        claims = {claim.claim_id: claim for claim in decision_spec.claim_tests}
        rules = {rule.option_id: rule for rule in decision_spec.option_rules}
        assessments: list[OptionAssessment] = []
        for option in options:
            rule = rules.get(option.option_id)
            positive_claims = tuple(rule.all_of) if rule is not None else ()
            negative_claims = tuple(rule.none_of) if rule is not None else ()
            positive_text = (
                " ".join(
                    claims[claim_id].statement for claim_id in positive_claims if claim_id in claims
                )
                or option.text
            )
            negative_text = " ".join(
                claims[claim_id].statement for claim_id in negative_claims if claim_id in claims
            )
            positive_tokens = set(self._normalized_tokens(positive_text))
            negative_tokens = set(self._normalized_tokens(negative_text))
            positive_overlap = len(positive_tokens & fact_tokens)
            negative_overlap = len(negative_tokens & fact_tokens)
            support = min(3, positive_overlap)
            contradiction = min(3, negative_overlap)
            assessments.append(
                OptionAssessment(
                    option_id=option.option_id,
                    support_score=support,
                    contradiction_score=contradiction,
                    discriminant_ids=positive_claims,
                    evidence_fact_ids=tuple(fact.fact_id for fact in packet.facts),
                    source_frame_ids=tuple(
                        dict.fromkeys(
                            frame_id for fact in packet.facts for frame_id in fact.source_frame_ids
                        )
                    ),
                    reason="controller lexical overlap fallback",
                )
            )
        if selected_option_id is None:
            cannot_phrases = (
                "cannot be determined",
                "cannot determine",
                "not enough information",
                "insufficient information",
                "unknown from the video",
            )
            cannot_ids = {
                option.option_id
                for option in options
                if any(
                    phrase in " ".join(option.text.casefold().split()) for phrase in cannot_phrases
                )
            }
            selected = max(
                assessments,
                key=lambda item: (
                    0
                    if item.option_id in cannot_ids and item.support_score == 0 and not packet.facts
                    else 1,
                    item.support_score - item.contradiction_score,
                    item.support_score,
                    -item.contradiction_score,
                    -int(item.option_id.removeprefix("O")),
                ),
            )
            selected_option_id = selected.option_id
        resolved = tuple(
            claim_id
            for claim_id, claim in claims.items()
            if set(self._normalized_tokens(claim.statement)) & fact_tokens
        )
        unresolved = tuple(claim_id for claim_id in claims if claim_id not in resolved)
        return ChoiceDecision(
            selected_option_id=selected_option_id,
            option_assessments=tuple(assessments),
            resolved_discriminant_ids=resolved,
            unresolved_discriminant_ids=unresolved,
            reason=reason,
        )

    @staticmethod
    def _normalized_tokens(text: str) -> tuple[str, ...]:
        return tuple(
            token
            for token in re.findall(r"[\w\u4e00-\u9fff]+", text.casefold())
            if len(token) > 1
            and token
            not in {
                "the",
                "and",
                "that",
                "this",
                "with",
                "from",
                "what",
                "which",
                "does",
                "did",
                "video",
                "option",
            }
        )

    def _grade_evidence(
        self,
        spec: ObservationSpec,
        packet: EvidencePacket,
        decision_spec: DecisionSpec | None,
        decision: ChoiceDecision | None,
    ) -> EvidenceGrade:
        clear = {
            slot_id
            for fact in packet.facts
            if fact.visibility == "clear"
            for slot_id in fact.slot_ids
        } & spec.required_slot_ids
        partial = {
            slot_id
            for fact in packet.facts
            if fact.visibility in {"partial", "occluded", "conflicting"}
            for slot_id in fact.slot_ids
        } & spec.required_slot_ids
        missing = spec.required_slot_ids - clear
        provenance = list(validate_fact_provenance(packet))
        modality: list[str] = []
        expected_types: dict[str, tuple[type[Any], ...]] = {
            "static_visual": (StaticFact,),
            "dynamic_action": (EventFact,),
            "ocr": (TextFact,),
            "subscene_caption": (EventFact, StaticFact),
        }
        useful = [fact for fact in packet.facts if fact.visibility != "not_visible"]
        if useful and not any(
            isinstance(fact, expected_types[spec.primary_mode]) for fact in useful
        ):
            modality.append(f"no {spec.primary_mode} fact type in usable evidence")
        if spec.primary_mode == "ocr" and not any(
            isinstance(fact, TextFact) and self._has_ocr_consensus(fact, packet) for fact in useful
        ):
            modality.append("OCR adjacent-frame consensus absent")
        if spec.coverage_requirement == "full_span" and not (
            packet.coverage_manifest.full_span_coverage
        ):
            modality.append("required full-span coverage absent")
        selected_assessment = None
        if decision is not None:
            selected_assessment = next(
                (
                    item
                    for item in decision.option_assessments
                    if item.option_id == decision.selected_option_id
                ),
                None,
            )
            if selected_assessment is None:
                modality.append("selected option has no assessment")
            elif not selected_assessment.source_frame_ids and packet.facts:
                modality.append("selected option has no valid frame citation")
        unresolved = (
            bool(decision.unresolved_discriminant_ids)
            if decision is not None and decision_spec is not None
            else False
        )
        if (
            not missing
            and not provenance
            and not modality
            and not packet.conflicts
            and not unresolved
            and (selected_assessment is None or selected_assessment.support_score >= 2)
        ):
            level = "strong"
        elif clear and not provenance and not packet.conflicts:
            level = "partial"
        elif (
            useful
            or partial
            or (selected_assessment is not None and selected_assessment.support_score > 0)
        ):
            level = "weak"
        else:
            level = "none"
        return EvidenceGrade(
            level=level,
            clear_slot_ids=tuple(sorted(clear)),
            partial_slot_ids=tuple(sorted(partial - clear)),
            missing_slot_ids=tuple(sorted(missing)),
            provenance_errors=tuple(provenance),
            modality_errors=tuple(modality),
            conflicts=packet.conflicts,
        )

    def _decision_rescue_reasons(
        self,
        spec: ObservationSpec,
        packet: EvidencePacket,
        decision_spec: DecisionSpec,
        decision: ChoiceDecision,
        grade: EvidenceGrade,
        selected: CandidateObservation | None,
    ) -> tuple[str, ...]:
        del spec, packet, decision_spec
        reasons: list[str] = []
        if selected is None:
            reasons.append("no_visible_initial_candidate")
        if grade.missing_slot_ids:
            reasons.append("missing_required_slots")
        if grade.provenance_errors:
            reasons.append("invalid_fact_provenance")
        if grade.modality_errors:
            reasons.append("invalid_or_incomplete_modality_evidence")
        if grade.conflicts:
            reasons.append("conflicting_observations")
        if grade.level in {"weak", "none"}:
            reasons.append(f"{grade.level}_evidence_support")
        if decision.unresolved_discriminant_ids:
            reasons.append("unresolved_option_discriminants")
        ranked = sorted(
            decision.option_assessments,
            key=lambda item: (
                item.support_score - item.contradiction_score,
                item.support_score,
                -item.contradiction_score,
            ),
            reverse=True,
        )
        recovered_label = decision.reason.startswith(
            "terminal fallback: recovered a legal option label"
        )
        if len(ranked) >= 2 and not recovered_label:
            first_score = ranked[0].support_score - ranked[0].contradiction_score
            second_score = ranked[1].support_score - ranked[1].contradiction_score
            if first_score <= second_score:
                reasons.append("option_score_tie")
            elif first_score - second_score <= 1 and ranked[0].support_score < 3:
                reasons.append("narrow_option_margin")
        return tuple(dict.fromkeys(reasons))

    def _mcq_result_v2(
        self,
        state: _RunState,
        spec: ObservationSpec,
        packet: EvidencePacket,
        options: Sequence[CanonicalOption],
        decision_spec: DecisionSpec,
        decision: ChoiceDecision,
        *,
        decision_source: str,
    ) -> P01Result:
        prediction = option_label(options, decision.selected_option_id)
        if prediction is None:
            prediction = options[0].benchmark_label
            decision = replace(
                decision,
                selected_option_id=options[0].option_id,
                reason=(decision.reason + "; controller repaired illegal final option").strip("; "),
            )
            decision_source = "terminal_fallback"
        grade = self._grade_evidence(spec, packet, decision_spec, decision)
        if state.rescue_used and (
            state.degraded_stages or decision_source == "terminal_fallback"
        ):
            pipeline_outcome = "completed_with_rescue_and_degradation"
        elif state.rescue_used:
            pipeline_outcome = "completed_with_rescue"
        elif not state.degraded_stages and decision_source == "initial":
            pipeline_outcome = "completed"
        else:
            pipeline_outcome = "completed_with_degradation"
        state.trace["final_evidence_grade"] = grade.to_dict()
        state.trace["stop_reason"] = "mandatory_mcq_prediction_emitted"
        state.trace["decision_source"] = decision_source
        return self._result(
            status="answered",
            prediction=prediction,
            decision_source=decision_source,
            support_level=grade.level,
            pipeline_outcome=pipeline_outcome,
            canonical_span=packet.canonical_span,
            evidence=packet,
            decision=decision,
            evidence_grade=grade,
            missing_facts=grade.missing_slot_ids,
            state=state,
        )

    def _compose_free_text_v2(
        self,
        state: _RunState,
        spec: ObservationSpec,
        packet: EvidencePacket,
    ) -> P01Result:
        facts = tuple(
            sorted(
                packet.facts,
                key=lambda fact: (
                    fact.start_seconds,
                    fact.end_seconds,
                    getattr(fact, "order", None) or 0,
                ),
            )
        )
        frames = self._decision_frames(state, packet, spec)
        prompt = build_answer_composer_prompt(
            state.request.question,
            facts,
            output_language=spec.output_language,
            missing_slot_ids=missing_required_slots(spec, facts),
            frames=frames,
            span=packet.canonical_span,
        )
        answer = ""
        used_fallback = False
        try:
            output = self._model_call(
                state,
                role="answer_composer",
                prompt=prompt,
                frames=frames,
                media_paths=tuple(frame.path for frame in frames),
                media_kind=("images" if spec.primary_mode in {"static_visual", "ocr"} else "video"),
                sample_fps=(
                    None
                    if spec.primary_mode in {"static_visual", "ocr"}
                    else self._effective_fps(
                        [frame.timestamp_seconds for frame in frames],
                        self._mode_fps(spec.primary_mode),
                    )
                ),
                sampling={"rule": "g39_local_best_effort_composer"},
                terminal=True,
            )
            answer = output.text.strip()
        except (ResourceExhausted, RuntimeError) as exc:
            self._degrade(state, "answer_composer", exc)
        if not answer:
            answer = self._event_fact_fallback(facts, state.request.question)
            used_fallback = True
            self._degrade(state, "answer_composer", "empty output; deterministic fact text used")
        grade = self._grade_evidence(spec, packet, None, None)
        state.trace["final_evidence_grade"] = grade.to_dict()
        state.trace["stop_reason"] = "best_effort_g39_answer_emitted"
        return self._result(
            status="answered",
            prediction=answer,
            decision_source="terminal_fallback" if used_fallback else "composer",
            support_level=grade.level,
            pipeline_outcome=(
                "completed_with_rescue_and_degradation"
                if state.rescue_used and state.degraded_stages
                else "completed_with_rescue"
                if state.rescue_used
                else "completed_with_degradation"
                if state.degraded_stages
                else "completed"
            ),
            canonical_span=packet.canonical_span,
            evidence=packet,
            evidence_grade=grade,
            missing_facts=grade.missing_slot_ids,
            state=state,
        )

    @staticmethod
    def _event_fact_fallback(facts: Sequence[Fact], question: str) -> str:
        statements = [
            fact.statement.strip()
            for fact in facts
            if fact.statement.strip() and fact.visibility != "not_visible"
        ]
        if statements:
            return " ".join(dict.fromkeys(statements))
        if re.search(r"[\u4e00-\u9fff]", question):
            return "无法从选定的局部片段中完整辨认该事件。"
        return "The event cannot be fully identified from the selected local segment."

    def _initial_candidates(
        self,
        state: _RunState,
        index: P01VideoIndex,
        spec: ObservationSpec,
        explicit_span: TimeSpan | None,
    ) -> tuple[LocatorCandidate, ...]:
        if explicit_span is not None:
            state.trace["locator_skipped"] = explicit_span.source
            return (LocatorCandidate(index.root_id, "deterministic time constraint", 0),)
        if index.duration_seconds <= self.config.short_video_threshold_sec:
            state.trace["locator_skipped"] = "short_video"
            return (LocatorCandidate(index.root_id, "whole short video", 0),)
        return self._locate(state, index, spec)

    def _locate(
        self,
        state: _RunState,
        index: P01VideoIndex,
        spec: ObservationSpec,
    ) -> tuple[LocatorCandidate, ...]:
        frontier = [LocatorCandidate(index.root_id, "root", 0)]
        for level in range(1, self.config.locator_max_levels + 1):
            expanded: list[LocatorCandidate] = []
            per_parent_limit = self.config.locator_beam_width if len(frontier) == 1 else 1
            for parent in frontier:
                children = index.children(parent.node_id)
                if not children:
                    expanded.append(parent)
                    continue
                if len(children) > self.config.locator_branch_factor:
                    raise RuntimeError("temporal tree violates locator branch bound")
                primary_frames: list[FrameRef] = []
                auxiliary_frames: list[FrameRef] = []
                labels: list[str] = []
                for child in children:
                    primary, auxiliary = index.representative_frames(
                        child,
                        spec.primary_mode,
                    )
                    primary_frames.append(primary)
                    auxiliary_frames.append(auxiliary)
                    labels.append(
                        f"{child.node_id} {child.span.start_seconds:.1f}-{child.span.end_seconds:.1f}s"
                    )
                sheet_dir = self.config.resolved_cache_dir / "contact_sheets"
                sheet_paths: list[str] = []
                page_manifest: list[dict[str, Any]] = []
                page_size = self.config.locator_tiles_per_page
                for view_name, view_frames in (
                    ("primary", primary_frames),
                    ("auxiliary", auxiliary_frames),
                ):
                    for page_index, offset in enumerate(
                        range(0, len(view_frames), page_size),
                        start=1,
                    ):
                        page_frames = view_frames[offset : offset + page_size]
                        page_labels = labels[offset : offset + page_size]
                        sheet = build_contact_sheet(
                            page_frames,
                            page_labels,
                            output_dir=sheet_dir,
                            columns=min(
                                self.config.contact_sheet_columns,
                                len(page_frames),
                            ),
                            tile_width=self.config.locator_tile_width,
                            image_height=self.config.locator_tile_height,
                            label_height=self.config.locator_label_height,
                        )
                        sheet_paths.append(sheet)
                        page_manifest.append(
                            {
                                "view": view_name,
                                "page": page_index,
                                "sheet": sheet,
                                "frame_ids": [frame.id for frame in page_frames],
                                "cell_labels": list(page_labels),
                            }
                        )
                shown_frames = tuple(primary_frames + auxiliary_frames)
                state.remember(shown_frames)
                prompt = build_locator_prompt(
                    state.request.question,
                    spec,
                    [(child.node_id, child.span) for child in children],
                    max_candidates=per_parent_limit,
                )
                decision = self._structured_call(
                    state,
                    role="locator",
                    prompt=prompt,
                    parser=lambda text, children=children, limit=per_parent_limit: parse_locator(
                        text,
                        valid_node_ids={child.node_id for child in children},
                        max_candidates=limit,
                    ),
                    schema_hint="{candidates:[{node_id,visible_anchor}]}",
                    frames=shown_frames,
                    media_paths=tuple(sheet_paths),
                    media_kind="images",
                    sampling={
                        "mode": spec.primary_mode,
                        "representatives_per_cell": 2,
                        "level": level,
                        "tiles_per_page": page_size,
                        "contact_sheet_pages": page_manifest,
                    },
                )
                state.trace["locator_rounds"].append(
                    {
                        "level": level,
                        "parent_id": parent.node_id,
                        "cells": [child.to_dict() for child in children],
                        "contact_sheets": sheet_paths,
                        "contact_sheet_pages": page_manifest,
                        "primary_tiles": [frame.to_dict() for frame in primary_frames],
                        "auxiliary_tiles": [frame.to_dict() for frame in auxiliary_frames],
                        "selected_node_ids": list(decision.node_ids),
                        "visible_anchors": list(decision.visible_anchors),
                    }
                )
                expanded.extend(
                    LocatorCandidate(node_id, anchor, rank)
                    for rank, (node_id, anchor) in enumerate(
                        zip(decision.node_ids, decision.visible_anchors),
                        start=parent.rank * self.config.locator_beam_width,
                    )
                )
            if not expanded:
                return ()
            frontier = sorted(expanded, key=lambda item: item.rank)[
                : self.config.locator_beam_width
            ]
            if level == self.config.locator_max_levels or all(
                index.node(item.node_id).is_leaf for item in frontier
            ):
                break
        return tuple(
            LocatorCandidate(item.node_id, item.visible_anchor, rank)
            for rank, item in enumerate(frontier[: self.config.max_candidate_spans])
        )

    def _scout_candidate(
        self,
        state: _RunState,
        index: P01VideoIndex,
        spec: ObservationSpec,
        *,
        candidate_id: str,
        span: TimeSpan,
        locator_rank: int,
        locator_anchor: str,
        explicit: bool,
        role: str = "candidate_scout",
        discriminants: Sequence[ClaimTest] = (),
    ) -> CandidateObservation:
        frames, sample_fps, media_kind, sampling = self._sample_frames(
            state,
            index,
            spec.primary_mode,
            span,
            explicit=explicit,
        )
        view_id = f"{role.upper()}-{candidate_id}"
        prompt = build_scout_prompt(
            state.request.question,
            spec,
            candidate_id,
            span,
            frames,
            purpose=role,
            discriminants=discriminants,
        )
        decision = self._structured_call(
            state,
            role=role,
            prompt=prompt,
            parser=lambda text: parse_scout(
                text,
                candidate_span=span,
                valid_slot_ids=spec.required_slot_ids,
                frames=frames,
                view_id=view_id,
                fact_prefix=candidate_id,
            ),
            schema_hint="CandidateScout evidence JSON",
            frames=frames,
            media_paths=tuple(frame.path for frame in frames),
            media_kind=media_kind,
            sample_fps=sample_fps,
            sampling=sampling,
        )
        packet = self._packet_from_scout(
            index,
            spec,
            span,
            frames,
            sample_fps,
            view_id,
            decision,
        )
        return CandidateObservation(
            candidate_id=candidate_id,
            locator_rank=locator_rank,
            target_visible=decision.target_visible,
            supporting_span=decision.supporting_span,
            evidence=packet,
            visible_anchor=decision.visible_anchor or locator_anchor,
        )

    def _scout_explicit_interval(
        self,
        state: _RunState,
        index: P01VideoIndex,
        spec: ObservationSpec,
        span: TimeSpan,
    ) -> CandidateObservation:
        chunks = interval_chunks(span, self.config)
        state.trace["interval_chunks"] = [chunk.to_dict() for chunk in chunks]
        observations: list[CandidateObservation] = []
        for chunk_index, chunk in enumerate(chunks, start=1):
            try:
                observations.append(
                    self._scout_candidate(
                        state,
                        index,
                        spec,
                        candidate_id=f"G42-C{chunk_index}",
                        span=chunk,
                        locator_rank=0,
                        locator_anchor="explicit interval",
                        explicit=True,
                        role="interval_scout",
                    )
                )
            except (ProtocolError, ResourceExhausted) as exc:
                self._degrade(state, f"interval_scout:{chunk_index}", exc)
                if isinstance(exc, ResourceExhausted):
                    break
        if not observations:
            raise ProtocolError("all explicit-interval scout chunks failed")
        packet = self._combine_interval_packets(
            index,
            spec,
            span,
            [item.evidence for item in observations],
        )
        if len(observations) < len(chunks):
            packet = replace(
                packet,
                coverage_manifest=replace(
                    packet.coverage_manifest,
                    full_span_coverage=False,
                ),
            )
        return CandidateObservation(
            candidate_id="EXPLICIT",
            locator_rank=0,
            target_visible=any(item.target_visible for item in observations),
            supporting_span=span,
            evidence=packet,
            visible_anchor=" | ".join(
                item.visible_anchor for item in observations if item.visible_anchor
            ),
        )

    def _sample_frames(
        self,
        state: _RunState,
        index: P01VideoIndex,
        mode: str,
        span: TimeSpan,
        *,
        explicit: bool,
    ) -> tuple[tuple[FrameRef, ...], float | None, str, dict[str, Any]]:
        ocr_search_frames: tuple[FrameRef, ...] = ()
        if explicit:
            timestamps, fps, rule = self._interval_timestamps(index, span, mode)
        elif mode == "static_visual":
            timestamps = evenly_spaced_timestamps(span, self.config.static_frames)
            fps, rule = None, "static_even"
        elif mode == "dynamic_action":
            timestamps = uniform_timestamps(
                span,
                fps=self.config.dynamic_fps,
                max_frames=self.config.dynamic_max_frames,
            )
            fps, rule = self.config.dynamic_fps, "dynamic_uniform"
        elif mode == "ocr":
            search = uniform_timestamps(
                span,
                fps=self.config.ocr_search_fps,
                max_frames=min(
                    256,
                    max(
                        self.config.ocr_max_frames,
                        math.ceil(span.duration_seconds * self.config.ocr_search_fps),
                    ),
                ),
            )
            timestamps = search
            fps, rule = None, "ocr_on_demand_ranked"
        elif mode == "subscene_caption":
            timestamps = uniform_timestamps(
                span,
                fps=self.config.caption_fps,
                max_frames=self.config.caption_max_frames,
            )
            fps, rule = self.config.caption_fps, "caption_uniform"
        else:
            raise ValueError(f"unsupported observation mode: {mode}")
        if mode == "ocr":
            ocr_search_frames = self.source_store.extract(
                state.request.video_path,
                timestamps,
                purpose="ocr_search_preview",
                max_side=self.config.index_max_side,
            )
            timestamps = self._rank_ocr_frames(ocr_search_frames)
            fps = None
        frames = self.source_store.extract(
            state.request.video_path,
            timestamps,
            purpose=rule,
        )
        state.remember(frames)
        effective_fps = self._effective_fps(
            [frame.timestamp_seconds for frame in frames],
            fps,
        )
        media_kind = (
            "video"
            if effective_fps is not None
            and (explicit or mode in {"dynamic_action", "subscene_caption"})
            else "images"
        )
        return (
            frames,
            effective_fps,
            media_kind,
            {
                "rule": rule,
                "mode": mode,
                "explicit_interval": explicit,
                "requested_timestamps": list(timestamps),
                "requested_fps": fps,
                "sample_fps": effective_fps,
                "frame_count": len(frames),
                "ocr_preview_frame_count": len(ocr_search_frames),
            },
        )

    def _interval_timestamps(
        self,
        index: P01VideoIndex,
        span: TimeSpan,
        mode: str,
    ) -> tuple[tuple[float, ...], float | None, str]:
        strict_span = TimeSpan(
            span.start_seconds,
            span.end_seconds,
            source=span.source,
        )
        duration = strict_span.duration_seconds
        if duration <= 12:
            fps = self.config.interval_short_fps
            maximum = self.config.interval_short_max_frames
            rule = "interval_le_12"
        elif duration <= 30:
            fps = self.config.interval_medium_fps
            maximum = self.config.interval_medium_max_frames
            rule = "interval_12_30"
        else:
            fps = self.config.interval_long_fps
            maximum = self.config.interval_long_max_frames
            rule = "interval_gt_30"
        timestamps = uniform_timestamps(strict_span, fps=fps, max_frames=maximum)
        if duration > 30:
            keyframes = self._pattern_keyframes(
                index,
                strict_span,
                count=self.config.interval_pattern_keyframes,
            )
            timestamps = self._prioritized_timestamps(
                strict_span,
                keyframes,
                timestamps,
                maximum,
            )
        if mode == "ocr":
            return timestamps, None, f"{rule}_ocr_search"
        return timestamps, fps, rule

    def _rank_ocr_frames(
        self,
        frames: Sequence[FrameRef],
    ) -> tuple[float, ...]:
        from PIL import Image, ImageFilter, ImageStat

        scored: list[tuple[float, float, float]] = []
        for frame in frames:
            try:
                with Image.open(frame.path) as source:
                    gray = source.convert("L")
                    gray.thumbnail((384, 216), Image.Resampling.BILINEAR)
                    edges = gray.filter(ImageFilter.FIND_EDGES)
                    stats = ImageStat.Stat(edges)
                    clarity = min(1.0, math.sqrt(stats.var[0]) / 64.0)
                    text_score = min(1.0, stats.mean[0] / 32.0 + clarity * 0.35)
            except OSError:
                text_score, clarity = 0.0, 0.0
            scored.append((text_score, clarity, frame.timestamp_seconds))
        ranked = sorted(
            scored,
            key=lambda item: (item[0], item[1], -item[2]),
            reverse=True,
        )[: self.config.ocr_max_frames]
        return tuple(sorted(item[2] for item in ranked))

    def _rank_ocr_timestamps(
        self,
        index: P01VideoIndex,
        timestamps: Sequence[float],
    ) -> tuple[float, ...]:
        metrics = {item.frame_id: item for item in index.metrics}
        ranked = sorted(
            timestamps,
            key=lambda timestamp: (
                metrics[index.cached_video.nearest_frame(timestamp).id].text_score,
                metrics[index.cached_video.nearest_frame(timestamp).id].clarity_score,
                -timestamp,
            ),
            reverse=True,
        )[: self.config.ocr_max_frames]
        return tuple(sorted(ranked))

    @staticmethod
    def _pattern_keyframes(
        index: P01VideoIndex,
        span: TimeSpan,
        *,
        count: int,
    ) -> tuple[float, ...]:
        metrics = [
            item
            for item in index.metrics
            if span.start_seconds <= item.timestamp_seconds <= span.end_seconds
        ]
        ranked = sorted(
            metrics,
            key=lambda item: (
                max(item.change_score, item.motion_score, item.text_score),
                item.clarity_score,
            ),
            reverse=True,
        )[:count]
        return tuple(sorted(item.timestamp_seconds for item in ranked))

    def _packet_from_scout(
        self,
        index: P01VideoIndex,
        spec: ObservationSpec,
        span: TimeSpan,
        frames: Sequence[FrameRef],
        sample_fps: float | None,
        view_id: str,
        decision: ScoutDecision,
    ) -> EvidencePacket:
        coverage = build_coverage_manifest(
            spec.primary_mode,
            span,
            frames,
            sample_fps=sample_fps,
            shot_ids=index.shot_ids_for_span(span),
        )
        facts = self._valid_decisive_facts(decision.facts, coverage, span)
        missing = tuple(
            sorted(set(decision.missing_slot_ids) | set(missing_required_slots(spec, facts)))
        )
        view = SourceView(
            view_id=view_id,
            kind="initial_observation",
            frame_ids=tuple(frame.id for frame in frames),
            media_paths=tuple(frame.path for frame in frames),
            span=span,
        )
        return EvidencePacket(
            canonical_span=span,
            facts=facts,
            coverage_manifest=coverage,
            missing_slot_ids=missing,
            conflicts=decision.conflicts,
            source_views=(view,),
        )

    def _combine_interval_packets(
        self,
        index: P01VideoIndex,
        spec: ObservationSpec,
        span: TimeSpan,
        packets: Sequence[EvidencePacket],
    ) -> EvidencePacket:
        if not packets:
            raise ValueError("interval scout produced no packets")
        facts: list[Fact] = []
        conflicts: list[str] = []
        views: list[SourceView] = []
        frame_ids: list[str] = []
        context_ids: list[str] = []
        for packet in packets:
            facts.extend(packet.facts)
            conflicts.extend(packet.conflicts)
            views.extend(packet.source_views)
            frame_ids.extend(packet.coverage_manifest.frame_ids)
            context_ids.extend(packet.coverage_manifest.context_only_frame_ids)
        unique_frame_ids = tuple(dict.fromkeys(frame_ids))
        coverage = replace(
            packets[-1].coverage_manifest,
            observed_start_seconds=span.start_seconds,
            observed_end_seconds=span.end_seconds,
            frame_ids=unique_frame_ids,
            shot_ids=index.shot_ids_for_span(span),
            context_only_frame_ids=tuple(dict.fromkeys(context_ids)),
            full_span_coverage=all(
                packet.coverage_manifest.full_span_coverage for packet in packets
            ),
        )
        return EvidencePacket(
            canonical_span=span,
            facts=tuple(facts),
            coverage_manifest=coverage,
            missing_slot_ids=missing_required_slots(spec, facts),
            conflicts=tuple(dict.fromkeys(conflicts)),
            source_views=tuple(views),
        )

    def _canonicalize_selected(
        self,
        selected: CandidateObservation,
        index: P01VideoIndex,
        spec: ObservationSpec,
        *,
        explicit_span: TimeSpan | None,
        exempt_from_auto_limit: bool,
        state: _RunState,
    ) -> EvidencePacket:
        before = selected.evidence.canonical_span
        if explicit_span is not None:
            canonical = explicit_span
        elif exempt_from_auto_limit:
            canonical = before
        else:
            canonical = bounded_automatic_span(
                selected.supporting_span,
                mode=spec.primary_mode,
                duration_seconds=index.duration_seconds,
                config=self.config,
            )
        frames = tuple(
            frame
            for frame_id in selected.evidence.coverage_manifest.frame_ids
            if (frame := state.frame_catalog.get(frame_id)) is not None
        )
        coverage = build_coverage_manifest(
            spec.primary_mode,
            canonical,
            frames,
            sample_fps=selected.evidence.coverage_manifest.sample_fps,
            shot_ids=index.shot_ids_for_span(canonical),
        )
        facts = self._valid_decisive_facts(
            selected.evidence.facts,
            coverage,
            canonical,
        )
        packet = EvidencePacket(
            canonical_span=canonical,
            facts=facts,
            coverage_manifest=coverage,
            missing_slot_ids=missing_required_slots(spec, facts),
            conflicts=selected.evidence.conflicts,
            source_views=selected.evidence.source_views,
        )
        state.trace["span_changes"].append(
            {
                "reason": "candidate_selection",
                "before": before.to_dict(),
                "after": canonical.to_dict(),
            }
        )
        return packet

    def _sufficiency_plan(
        self,
        spec: ObservationSpec,
        packet: EvidencePacket,
        decision_spec: DecisionSpec | None,
    ) -> RefinementPlan:
        plan = build_refinement_plan(spec, packet, decision_spec, self.config)
        target_slots = set(plan.target_slot_ids)
        reasons = [plan.reason] if plan.reason else []
        densify = plan.densify_fps
        if spec.primary_mode == "ocr":
            weak_ocr_slots = {
                slot_id
                for fact in packet.facts
                if isinstance(fact, TextFact) and not self._has_ocr_consensus(fact, packet)
                for slot_id in fact.slot_ids
            }
            if weak_ocr_slots:
                target_slots.update(weak_ocr_slots)
                densify = self.config.ocr_search_fps
                reasons.append("OCR lacks adjacent-frame consensus")
        if spec.coverage_requirement == "full_span" and not (
            packet.coverage_manifest.full_span_coverage
        ):
            target_slots.update(spec.required_slot_ids)
            densify = densify or self._mode_fps(spec.primary_mode)
            reasons.append("full-span coverage incomplete")
        if decision_spec is not None:
            clear_slots = spec.required_slot_ids - set(missing_required_slots(spec, packet.facts))
            slot_unobserved_claims = [
                claim.claim_id
                for claim in decision_spec.claim_tests
                if claim.slot_ids and not set(claim.slot_ids).issubset(clear_slots)
            ]
            detail_unobserved_claims = self._unresolved_claim_details(decision_spec, packet)
            unobserved_claims = list(
                dict.fromkeys((*slot_unobserved_claims, *detail_unobserved_claims))
            )
            if detail_unobserved_claims:
                reasons.append("option discriminants are not grounded in observed facts")
        else:
            unobserved_claims = []
        if (target_slots or unobserved_claims) and densify is None:
            densify = self._mode_fps(spec.primary_mode)
            reasons.append("unresolved observable fields")
        return replace(
            plan,
            densify_fps=densify,
            target_slot_ids=tuple(sorted(target_slots)),
            target_claim_ids=tuple(unobserved_claims or plan.target_claim_ids),
            reason="; ".join(item for item in reasons if item),
        )

    def _unresolved_claim_details(
        self,
        decision_spec: DecisionSpec,
        packet: EvidencePacket,
    ) -> tuple[str, ...]:
        """Find option axes for which clear facts do not favor one observable value."""
        clear_facts = [fact for fact in packet.facts if fact.visibility == "clear"]
        unresolved: list[str] = []
        slot_ids = {
            slot_id for claim in decision_spec.claim_tests for slot_id in claim.slot_ids
        }
        for slot_id in sorted(slot_ids):
            claims = [
                claim for claim in decision_spec.claim_tests if slot_id in claim.slot_ids
            ]
            if len(claims) < 2:
                continue
            claim_tokens = [
                set(self._normalized_tokens(claim.expected_value or claim.statement))
                for claim in claims
            ]
            if len({frozenset(tokens) for tokens in claim_tokens}) < 2:
                continue
            common_tokens = set.intersection(*claim_tokens) if claim_tokens else set()
            observed_text = " ".join(
                " ".join(
                    item
                    for item in (
                        fact.statement,
                        getattr(fact, "entity", ""),
                        getattr(fact, "attribute", ""),
                        getattr(fact, "relation", ""),
                        getattr(fact, "value", ""),
                        getattr(fact, "subject", ""),
                        getattr(fact, "action", ""),
                        getattr(fact, "object", ""),
                        getattr(fact, "target", ""),
                        getattr(fact, "result", ""),
                        getattr(fact, "exact_text", ""),
                    )
                    if item
                )
                for fact in clear_facts
                if slot_id in fact.slot_ids
            )
            observed_tokens = set(self._normalized_tokens(observed_text))
            scores = [len((tokens - common_tokens) & observed_tokens) for tokens in claim_tokens]
            if not scores:
                continue
            best = max(scores)
            if best == 0 or scores.count(best) > 1:
                unresolved.extend(claim.claim_id for claim in claims)
        return tuple(dict.fromkeys(unresolved))

    def _refine(
        self,
        state: _RunState,
        index: P01VideoIndex,
        spec: ObservationSpec,
        packet: EvidencePacket,
        plan: RefinementPlan,
        decision_spec: DecisionSpec | None,
        *,
        explicit_span: TimeSpan | None,
        exempt_from_auto_limit: bool,
    ) -> EvidencePacket:
        span = self._refinement_span(
            packet.canonical_span,
            plan,
            index.duration_seconds,
            spec.primary_mode,
            explicit_span=explicit_span,
            exempt_from_auto_limit=exempt_from_auto_limit,
        )
        fps = plan.densify_fps or self._mode_fps(spec.primary_mode)
        timestamps = uniform_timestamps(
            span,
            fps=fps,
            max_frames=self._mode_refine_max_frames(spec.primary_mode),
            offset_fraction=0.5,
        )
        frames = list(
            self.source_store.extract(
                state.request.video_path,
                timestamps,
                purpose="refinement",
            )
        )
        crops: list[FrameRef] = []
        for bbox in plan.crop_boxes[: self.config.max_detail_images]:
            original = state.frame_catalog.get(bbox.frame_id)
            if original is None:
                continue
            if original.id not in {frame.id for frame in frames}:
                frames.append(original)
            crops.append(self.source_store.crop(original, bbox, padding_fraction=0.1))
        combined = tuple(self._deduplicate_frames([*frames, *crops]))
        state.remember(combined)
        target_claims = (
            tuple(
                claim
                for claim in decision_spec.claim_tests
                if not plan.target_claim_ids or claim.claim_id in plan.target_claim_ids
            )
            if decision_spec is not None
            else ()
        )
        prompt = build_refinement_prompt(
            state.request.question,
            spec,
            span,
            combined,
            packet.facts,
            target_slot_ids=plan.target_slot_ids,
            target_claims=target_claims,
        )
        view_id = "REFINEMENT-1"
        decision = self._structured_call(
            state,
            role="refinement_extractor",
            prompt=prompt,
            parser=lambda text: parse_scout(
                text,
                candidate_span=span,
                valid_slot_ids=spec.required_slot_ids,
                frames=combined,
                view_id=view_id,
                fact_prefix="REFINE",
            ),
            schema_hint="CandidateScout evidence JSON",
            frames=combined,
            media_paths=tuple(frame.path for frame in combined),
            media_kind=(
                "mixed"
                if crops
                else ("images" if spec.primary_mode in {"static_visual", "ocr"} else "video")
            ),
            sample_fps=(
                fps if crops or spec.primary_mode not in {"static_visual", "ocr"} else None
            ),
            detail_frame_ids={frame.id for frame in crops},
            sampling={
                "rule": "single_refinement",
                "sample_fps": fps,
                "crop_count": len(crops),
                "requested_timestamps": list(timestamps),
            },
        )
        coverage = build_coverage_manifest(
            spec.primary_mode,
            span,
            combined,
            sample_fps=fps,
            shot_ids=index.shot_ids_for_span(span),
            roi_ids=tuple(frame.id for frame in crops),
        )
        facts = self._valid_decisive_facts(decision.facts, coverage, span)
        refined = EvidencePacket(
            canonical_span=span,
            facts=facts,
            coverage_manifest=coverage,
            missing_slot_ids=missing_required_slots(spec, facts),
            conflicts=decision.conflicts,
            source_views=(
                SourceView(
                    view_id=view_id,
                    kind="single_refinement",
                    frame_ids=tuple(frame.id for frame in combined),
                    media_paths=tuple(frame.path for frame in combined),
                    span=span,
                ),
            ),
        )
        merged = merge_evidence(packet, refined, canonical_span=span, spec=spec)
        valid_facts = self._valid_decisive_facts(
            merged.facts,
            merged.coverage_manifest,
            span,
        )
        merged = replace(
            merged,
            facts=valid_facts,
            missing_slot_ids=missing_required_slots(spec, valid_facts),
        )
        state.trace["span_changes"].append(
            {
                "reason": "single_refinement",
                "before": packet.canonical_span.to_dict(),
                "after": span.to_dict(),
            }
        )
        return merged

    def _verify(
        self,
        state: _RunState,
        index: P01VideoIndex,
        packet: EvidencePacket,
        claims: Sequence[ClaimTest],
        spec: ObservationSpec,
    ) -> tuple[ClaimVerdict, ...]:
        fps = self._mode_fps(spec.primary_mode)
        base = uniform_timestamps(
            packet.canonical_span,
            fps=fps,
            max_frames=self.config.verification_max_frames,
            offset_fraction=0.5,
        )
        decisive = [
            timestamp
            for fact in packet.facts
            if fact.visibility in {"clear", "partial", "conflicting"}
            for timestamp in (fact.start_seconds, fact.end_seconds)
        ]
        if spec.primary_mode == "ocr":
            neighbors = [
                max(packet.canonical_span.start_seconds, timestamp - 1 / fps)
                for timestamp in decisive
            ] + [
                min(packet.canonical_span.end_seconds, timestamp + 1 / fps)
                for timestamp in decisive
            ]
        else:
            neighbors = []
        timestamps = self._prioritized_timestamps(
            packet.canonical_span,
            decisive + neighbors,
            base,
            self.config.verification_max_frames,
        )
        frames = list(
            self.source_store.extract(
                state.request.video_path,
                timestamps,
                purpose="verification",
            )
        )
        crops: list[FrameRef] = []
        if spec.primary_mode == "ocr":
            for fact in packet.facts:
                if not isinstance(fact, TextFact) or fact.bbox is None:
                    continue
                nearest = min(
                    frames,
                    key=lambda frame: abs(frame.timestamp_seconds - fact.start_seconds),
                )
                rebased = replace(fact.bbox, frame_id=nearest.id)
                crops.append(self.source_store.crop(nearest, rebased, padding_fraction=0.1))
                if len(crops) >= self.config.max_detail_images:
                    break
        combined = tuple(self._deduplicate_frames([*frames, *crops]))
        state.remember(combined)
        prompt = build_verifier_prompt(
            state.request.question,
            claims,
            packet.facts,
            combined,
            packet.canonical_span,
        )
        verdicts = self._structured_call(
            state,
            role="verifier",
            prompt=prompt,
            parser=lambda text: parse_verifier(
                text,
                valid_claim_ids={claim.claim_id for claim in claims},
                valid_fact_ids={fact.fact_id for fact in packet.facts},
                frames=combined,
            ),
            schema_hint="{verdicts:[{claim_id,verdict,evidence_fact_ids,verification_frame_ids,reason}]}",
            frames=combined,
            media_paths=tuple(frame.path for frame in combined),
            media_kind=(
                "images" if crops or spec.primary_mode in {"static_visual", "ocr"} else "video"
            ),
            sample_fps=None if crops else fps,
            detail_frame_ids={frame.id for frame in crops},
            sampling={
                "rule": "independent_half_step_verification",
                "sample_fps": fps,
                "requested_timestamps": list(timestamps),
                "crop_count": len(crops),
            },
        )
        if spec.primary_mode == "ocr":
            consensus_slots = {
                slot_id
                for fact in packet.facts
                if isinstance(fact, TextFact) and self._has_ocr_consensus(fact, packet)
                for slot_id in fact.slot_ids
            }
            verdicts = tuple(
                replace(
                    verdict,
                    verdict="not_established",
                    reason=(verdict.reason + "; OCR consensus requirement not met").strip("; "),
                )
                if verdict.verdict == "entailed"
                and (
                    len(set(verdict.verification_frame_ids)) < 2
                    or not (
                        next(
                            (
                                set(claim.slot_ids)
                                for claim in claims
                                if claim.claim_id == verdict.claim_id
                            ),
                            set(),
                        )
                        <= consensus_slots
                    )
                )
                else verdict
                for verdict in verdicts
            )
        view = SourceView(
            view_id="VERIFICATION",
            kind="independent_verification",
            frame_ids=tuple(frame.id for frame in combined),
            media_paths=tuple(frame.path for frame in combined),
            span=packet.canonical_span,
        )
        state.trace["verification_view"] = view.to_dict()
        return verdicts

    def _resolve_mcq(
        self,
        state: _RunState,
        spec: ObservationSpec,
        packet: EvidencePacket,
        options: Sequence[CanonicalOption],
        decision_spec: DecisionSpec,
        claim_verdicts: Sequence[ClaimVerdict],
    ) -> P01Result:
        option_verdicts = resolve_options(decision_spec, claim_verdicts)
        sufficient = self._packet_sufficiency(spec, packet)["sufficient"]
        selected_id = unique_entailed_option(option_verdicts) if sufficient else None
        if selected_id is not None:
            answer = option_label(options, selected_id)
            state.trace["stop_reason"] = "unique_entailed_option"
            return self._result(
                status="answered",
                verified_answer=answer,
                prediction_kind="verified",
                canonical_span=packet.canonical_span,
                evidence=packet,
                claim_verdicts=tuple(claim_verdicts),
                option_verdicts=option_verdicts,
                missing_facts=missing_required_slots(spec, packet.facts),
                state=state,
            )
        status = unresolved_status(packet, spec)
        forced = None
        prediction_kind = "none"
        if state.request.force_choice:
            forced = option_label(options, forced_option(option_verdicts))
            prediction_kind = "forced" if forced is not None else "none"
        state.trace["stop_reason"] = "no_unique_entailed_option"
        return self._result(
            status=status,
            forced_prediction=forced,
            prediction_kind=prediction_kind,
            canonical_span=packet.canonical_span,
            evidence=packet,
            claim_verdicts=tuple(claim_verdicts),
            option_verdicts=option_verdicts,
            missing_facts=missing_required_slots(spec, packet.facts),
            state=state,
        )

    def _compose_free_text(
        self,
        state: _RunState,
        spec: ObservationSpec,
        packet: EvidencePacket,
        claims: Sequence[ClaimTest],
        verdicts: Sequence[ClaimVerdict],
    ) -> P01Result:
        eligible_facts = [fact for fact in packet.facts if isinstance(fact, EventFact)]
        fact_by_claim = {claim.claim_id: fact for claim, fact in zip(claims, eligible_facts)}
        verified_facts = tuple(
            fact_by_claim[verdict.claim_id]
            for verdict in verdicts
            if verdict.verdict == "entailed" and verdict.claim_id in fact_by_claim
        )
        missing = missing_required_slots(spec, verified_facts)
        if (
            not verified_facts
            or missing
            or not self._packet_sufficiency(spec, packet)["sufficient"]
        ):
            state.trace["stop_reason"] = "free_text_facts_not_verified"
            return self._result(
                status=unresolved_status(packet, spec),
                prediction_kind="none",
                canonical_span=packet.canonical_span,
                evidence=packet,
                claim_verdicts=tuple(verdicts),
                missing_facts=missing,
                state=state,
            )
        prompt = build_answer_composer_prompt(
            state.request.question,
            verified_facts,
            output_language=spec.output_language,
        )
        output = self._model_call(
            state,
            role="answer_composer",
            prompt=prompt,
        )
        answer = output.text.strip()
        if not answer:
            raise ProtocolError("answer composer returned empty text")
        state.trace["stop_reason"] = "verified_event_facts_composed"
        return self._result(
            status="answered",
            verified_answer=answer,
            prediction_kind="verified",
            canonical_span=packet.canonical_span,
            evidence=packet,
            claim_verdicts=tuple(verdicts),
            missing_facts=(),
            state=state,
        )

    def _structured_call(
        self,
        state: _RunState,
        *,
        role: str,
        prompt: str,
        parser: Callable[[str], _ParsedT],
        schema_hint: str,
        frames: Sequence[FrameRef] = (),
        media_paths: Sequence[str] = (),
        media_kind: str = "images",
        sample_fps: float | None = None,
        detail_frame_ids: set[str] | None = None,
        sampling: Mapping[str, Any] | None = None,
    ) -> _ParsedT:
        output = self._model_call(
            state,
            role=role,
            prompt=prompt,
            frames=frames,
            media_paths=media_paths,
            media_kind=media_kind,
            sample_fps=sample_fps,
            detail_frame_ids=detail_frame_ids,
            sampling=sampling,
        )
        try:
            return parser(output.text)
        except (ProtocolError, TypeError, ValueError) as first_error:
            if self.config.protocol_repair_attempts < 1:
                raise ProtocolError(f"{role}: {first_error}") from first_error
            repair_prompt = build_protocol_repair_prompt(
                role,
                output.text,
                f"{schema_hint}; validation constraint: {first_error}",
            )
            repaired = self._model_call(
                state,
                role="protocol_repair",
                prompt=repair_prompt,
                sampling={"target_role": role, "media_replayed": False},
            )
            state.trace["protocol_repairs"].append(
                {
                    "role": role,
                    "first_error": str(first_error),
                    "source_response": output.text,
                    "repair_response": repaired.text,
                    "media_replayed": False,
                }
            )
            try:
                return parser(repaired.text)
            except (ProtocolError, TypeError, ValueError) as second_error:
                raise ProtocolError(
                    f"{role} protocol repair failed: {second_error}"
                ) from second_error

    def _model_call(
        self,
        state: _RunState,
        *,
        role: str,
        prompt: str,
        frames: Sequence[FrameRef] = (),
        media_paths: Sequence[str] = (),
        media_kind: str = "images",
        sample_fps: float | None = None,
        detail_frame_ids: set[str] | None = None,
        sampling: Mapping[str, Any] | None = None,
        terminal: bool = False,
    ) -> ModelOutput:
        frame_tuple = tuple(frames)
        state.remember(frame_tuple)
        detail_frame_ids = detail_frame_ids or set()
        paths = tuple(media_paths or tuple(frame.path for frame in frame_tuple))
        safe_budget = False
        retry_index = 0
        while True:
            state.ledger.ensure_model_call(
                reserve=0 if terminal else self.config.terminal_call_reserve
            )
            media_config = self._resolved_media_config(
                role,
                media_kind if paths else "none",
                sample_fps,
                safe_budget=safe_budget,
            )
            messages = self._build_media_messages(
                prompt,
                frame_tuple,
                paths,
                media_kind,
                sample_fps,
                detail_frame_ids,
                media_config,
            )
            started = time.perf_counter()
            try:
                output = self.model.generate(
                    messages,
                    max_new_tokens=self.config.role_max_new_tokens(role),
                    temperature=0.0,
                )
            except Exception as exc:
                elapsed = time.perf_counter() - started
                is_oom = self._is_cuda_oom(exc)
                state.ledger.record_model_call(
                    role=role,
                    prompt=prompt,
                    raw_response="",
                    model_metadata={
                        "latency_seconds": elapsed,
                        "failed": True,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "is_cuda_oom": is_oom,
                    },
                    frames=frame_tuple,
                    media_paths=paths,
                    prompt_version="p01-v2",
                    media_config=media_config,
                    sampling={
                        **dict(sampling or {}),
                        "terminal": terminal,
                        "safe_budget_retry_index": retry_index,
                    },
                )
                if is_oom and retry_index < self.config.oom_retry_attempts:
                    retry_index += 1
                    safe_budget = True
                    state.safe_budget_retries += 1
                    state.trace.setdefault("oom_retries", []).append(
                        {
                            "role": role,
                            "failed_call_index": len(state.ledger.model_calls),
                            "retry_index": retry_index,
                            "fallback_profile": "recorded_safe_visual_budget",
                        }
                    )
                    self._clear_cuda_cache()
                    continue
                raise
            elapsed = time.perf_counter() - started
            metadata = dict(output.metadata)
            metadata.setdefault("latency_seconds", elapsed)
            state.ledger.record_model_call(
                role=role,
                prompt=prompt,
                raw_response=output.text,
                model_metadata=metadata,
                frames=frame_tuple,
                media_paths=paths,
                prompt_version="p01-v2",
                media_config=media_config,
                sampling={
                    **dict(sampling or {}),
                    "terminal": terminal,
                    "safe_budget_retry_index": retry_index,
                },
            )
            return output

    def _resolved_media_config(
        self,
        role: str,
        kind: str,
        sample_fps: float | None,
        *,
        safe_budget: bool,
    ) -> dict[str, Any]:
        if safe_budget:
            normal_min = self.config.safe_normal_min_pixels
            normal_max = self.config.safe_normal_max_pixels
            normal_total = self.config.safe_normal_total_pixels
            image_min = self.config.safe_image_min_pixels
            image_max = self.config.safe_image_max_pixels
            detail_max = self.config.safe_detail_max_pixels
            locator_max = self.config.safe_locator_sheet_max_pixels
        else:
            normal_min = self.config.normal_min_pixels
            normal_max = self.config.normal_max_pixels
            normal_total = self.config.normal_total_pixels
            image_min = self.config.image_min_pixels
            image_max = self.config.image_max_pixels
            detail_max = self.config.detail_max_pixels
            locator_max = self.config.locator_sheet_max_pixels
        return {
            "kind": kind,
            "safe_budget": safe_budget,
            "normal_min_pixels": normal_min,
            "normal_max_pixels": normal_max,
            "normal_total_pixels": normal_total,
            "image_min_pixels": image_min,
            "image_max_pixels": image_max,
            "detail_max_pixels": detail_max,
            "locator_sheet_max_pixels": locator_max,
            "effective_image_max_pixels": locator_max if role == "locator" else image_max,
            "sample_fps": sample_fps,
        }

    def _build_media_messages(
        self,
        prompt: str,
        frames: Sequence[FrameRef],
        paths: Sequence[str],
        media_kind: str,
        sample_fps: float | None,
        detail_frame_ids: set[str],
        media_config: Mapping[str, Any],
    ) -> list[dict[str, Any]]:
        if not paths:
            return [{"role": "user", "content": prompt}]
        if media_kind not in {"images", "video", "mixed"}:
            raise ValueError(f"unsupported P01 media kind: {media_kind}")
        paired: list[tuple[str, FrameRef | None]] = []
        if len(paths) == len(frames):
            paired = list(zip(paths, frames))
        else:
            paired = [(path, None) for path in paths]
        parts: list[dict[str, Any]] = []

        def video_part(video_paths: Sequence[str]) -> dict[str, Any]:
            part: dict[str, Any] = {
                "type": "video",
                "video": [self._media_uri(path) for path in video_paths],
                "min_pixels": media_config["normal_min_pixels"],
                "max_pixels": media_config["normal_max_pixels"],
                "total_pixels": media_config["normal_total_pixels"],
            }
            if sample_fps is not None:
                part["fps"] = sample_fps
            return part

        if media_kind == "video":
            parts.append(video_part(paths))
        elif media_kind == "mixed":
            regular_paths = [
                path for path, frame in paired if frame is None or frame.id not in detail_frame_ids
            ]
            detail_paths = [
                path for path, frame in paired if frame is not None and frame.id in detail_frame_ids
            ]
            if regular_paths:
                parts.append(video_part(regular_paths))
            for path in detail_paths:
                parts.append(
                    {
                        "type": "image",
                        "image": self._media_uri(path),
                        "min_pixels": media_config["image_min_pixels"],
                        "max_pixels": media_config["detail_max_pixels"],
                    }
                )
        else:
            for path, frame in paired:
                is_detail = frame is not None and frame.id in detail_frame_ids
                parts.append(
                    {
                        "type": "image",
                        "image": self._media_uri(path),
                        "min_pixels": media_config["image_min_pixels"],
                        "max_pixels": (
                            media_config["detail_max_pixels"]
                            if is_detail
                            else media_config["effective_image_max_pixels"]
                        ),
                    }
                )
        parts.append({"type": "text", "text": prompt})
        return [{"role": "user", "content": parts}]

    @staticmethod
    def _is_cuda_oom(error: Exception) -> bool:
        return type(error).__name__ == "OutOfMemoryError" or (
            "out of memory" in str(error).casefold()
            and ("cuda" in str(error).casefold() or "gpu" in str(error).casefold())
        )

    @staticmethod
    def _clear_cuda_cache() -> None:
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001 - cleanup must not mask the original OOM
            return

    @staticmethod
    def _valid_decisive_facts(
        facts: Sequence[Fact],
        coverage: Any,
        span: TimeSpan,
    ) -> tuple[Fact, ...]:
        frame_ids = set(coverage.frame_ids)
        context_ids = set(coverage.context_only_frame_ids)
        decisive_frame_ids = frame_ids - context_ids
        result: list[Fact] = []
        for fact in facts:
            if fact.visibility == "not_visible":
                result.append(fact)
                continue
            source_frame_ids = tuple(
                frame_id for frame_id in fact.source_frame_ids if frame_id in decisive_frame_ids
            )
            if not source_frame_ids:
                continue
            start_seconds = max(fact.start_seconds, span.start_seconds)
            end_seconds = min(fact.end_seconds, span.end_seconds)
            if end_seconds < start_seconds:
                continue
            updates: dict[str, Any] = {
                "source_frame_ids": source_frame_ids,
                "start_seconds": start_seconds,
                "end_seconds": end_seconds,
            }
            if isinstance(fact, TextFact):
                updates["consensus_frame_ids"] = tuple(
                    frame_id
                    for frame_id in fact.consensus_frame_ids
                    if frame_id in decisive_frame_ids
                )
                if fact.bbox is not None and fact.bbox.frame_id not in decisive_frame_ids:
                    updates["bbox"] = None
            result.append(replace(fact, **updates))
        return tuple(result)

    def _has_ocr_consensus(self, fact: TextFact, packet: EvidencePacket) -> bool:
        ids = tuple(dict.fromkeys(fact.consensus_frame_ids))
        return (
            len(ids) >= 2
            and bool(fact.exact_text)
            and not fact.uncertain_characters
            and fact.bbox is not None
            and set(ids).issubset(packet.coverage_manifest.frame_ids)
        )

    def _packet_sufficiency(
        self,
        spec: ObservationSpec,
        packet: EvidencePacket,
    ) -> dict[str, Any]:
        missing = missing_required_slots(spec, packet.facts)
        provenance = validate_fact_provenance(packet)
        full_span_missing = (
            spec.coverage_requirement == "full_span"
            and not packet.coverage_manifest.full_span_coverage
        )
        weak_ocr = spec.primary_mode == "ocr" and any(
            not any(
                isinstance(fact, TextFact)
                and fact.visibility == "clear"
                and slot_id in fact.slot_ids
                and self._has_ocr_consensus(fact, packet)
                for fact in packet.facts
            )
            for slot_id in spec.required_slot_ids
        )
        sufficient = not (
            missing or provenance or packet.conflicts or full_span_missing or weak_ocr
        )
        return {
            "sufficient": sufficient,
            "missing_slot_ids": list(missing),
            "provenance_errors": list(provenance),
            "conflicts": list(packet.conflicts),
            "full_span_coverage_required_but_missing": full_span_missing,
            "weak_ocr_consensus": weak_ocr,
        }

    def _refinement_span(
        self,
        span: TimeSpan,
        plan: RefinementPlan,
        duration_seconds: float,
        mode: str,
        *,
        explicit_span: TimeSpan | None,
        exempt_from_auto_limit: bool,
    ) -> TimeSpan:
        if explicit_span is not None:
            return TimeSpan(
                explicit_span.start_seconds,
                explicit_span.end_seconds,
                source=explicit_span.source,
            )
        start = max(0.0, span.start_seconds - plan.extend_before_seconds)
        end = min(duration_seconds, span.end_seconds + plan.extend_after_seconds)
        candidate = TimeSpan(start, end, source="refinement_span")
        if exempt_from_auto_limit or candidate.duration_seconds <= self.config.span_limit(mode):
            return candidate
        return bounded_automatic_span(
            span,
            mode=mode,
            duration_seconds=duration_seconds,
            config=self.config,
        )

    def _mode_fps(self, mode: str) -> float:
        return {
            "static_visual": max(1.0, self.config.index_fps),
            "dynamic_action": self.config.dynamic_refine_fps,
            "ocr": self.config.ocr_search_fps,
            "subscene_caption": self.config.caption_refine_fps,
        }[mode]

    def _mode_max_frames(self, mode: str) -> int:
        return {
            "static_visual": self.config.static_max_frames,
            "dynamic_action": self.config.dynamic_max_frames,
            "ocr": self.config.ocr_max_frames,
            "subscene_caption": self.config.caption_max_frames,
        }[mode]

    def _mode_refine_max_frames(self, mode: str) -> int:
        return {
            "static_visual": self.config.static_max_frames,
            "dynamic_action": self.config.dynamic_refine_max_frames,
            "ocr": self.config.ocr_max_frames,
            "subscene_caption": self.config.caption_refine_max_frames,
        }[mode]

    @staticmethod
    def _effective_fps(
        timestamps: Sequence[float],
        requested_fps: float | None,
    ) -> float | None:
        if requested_fps is None or len(timestamps) < 2:
            return requested_fps
        elapsed = timestamps[-1] - timestamps[0]
        if elapsed <= 0:
            return requested_fps
        return min(requested_fps, (len(timestamps) - 1) / elapsed)

    @staticmethod
    def _prioritized_timestamps(
        span: TimeSpan,
        priority: Sequence[float],
        fallback: Sequence[float],
        limit: int,
    ) -> tuple[float, ...]:
        result: list[float] = []
        seen: set[int] = set()
        for timestamp in (*priority, *fallback):
            value = min(span.decode_end_seconds, max(span.decode_start_seconds, timestamp))
            key = round(value * 1_000_000)
            if key in seen:
                continue
            seen.add(key)
            result.append(value)
            if len(result) >= limit:
                break
        return tuple(sorted(result))

    @staticmethod
    def _deduplicate_frames(frames: Sequence[FrameRef]) -> list[FrameRef]:
        result: list[FrameRef] = []
        seen: set[str] = set()
        for frame in frames:
            if frame.id in seen:
                continue
            seen.add(frame.id)
            result.append(frame)
        return result

    def _failure(
        self,
        state: _RunState,
        status: str,
        error: Exception,
    ) -> P01Result:
        logger.warning("P01 stopped with %s: %s", status, error)
        state.trace["stop_reason"] = status
        state.trace["error"] = f"{type(error).__name__}: {error}"
        return self._result(
            status=status,
            prediction_kind="none",
            canonical_span=(
                state.current_packet.canonical_span if state.current_packet is not None else None
            ),
            evidence=state.current_packet,
            missing_facts=(
                state.current_packet.missing_slot_ids if state.current_packet is not None else ()
            ),
            state=state,
        )

    @staticmethod
    def _result(
        *,
        status: str,
        canonical_span: TimeSpan | None,
        evidence: EvidencePacket | None,
        state: _RunState,
        prediction: str | None = None,
        decision_source: str = "none",
        support_level: str = "none",
        pipeline_outcome: str | None = None,
        decision: ChoiceDecision | None = None,
        evidence_grade: EvidenceGrade | None = None,
        prediction_kind: str | None = None,
        verified_answer: str | None = None,
        forced_prediction: str | None = None,
        claim_verdicts: tuple[ClaimVerdict, ...] = (),
        option_verdicts: tuple[OptionVerdict, ...] = (),
        missing_facts: tuple[str, ...] = (),
    ) -> P01Result:
        del prediction_kind
        prediction = prediction or verified_answer or forced_prediction
        if decision_source == "none" and prediction is not None:
            decision_source = "initial" if verified_answer is not None else "terminal_fallback"
        if support_level == "none" and verified_answer is not None:
            support_level = "strong"
        return P01Result(
            status=status,
            prediction=prediction,
            decision_source=decision_source,
            support_level=support_level,
            pipeline_outcome=pipeline_outcome or status,
            canonical_span=canonical_span,
            evidence=evidence,
            decision=decision,
            evidence_grade=evidence_grade,
            claim_verdicts=claim_verdicts,
            option_verdicts=option_verdicts,
            missing_facts=missing_facts,
            trace=state.trace,
            resources=state.ledger.to_dict(),
        )

    @staticmethod
    def _with_final_resources(result: P01Result, state: _RunState) -> P01Result:
        trace = dict(result.trace)
        trace["wall_seconds"] = time.perf_counter() - state.started
        resources = state.ledger.to_dict()
        return replace(result, trace=trace, resources=resources)

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

    @staticmethod
    def _coerce_interval(value: TimeSpan | Sequence[float] | None) -> TimeSpan | None:
        if value is None or isinstance(value, TimeSpan):
            return value
        if len(value) != 2:
            raise ValueError("given_interval requires START and END")
        return TimeSpan(float(value[0]), float(value[1]), source="given_interval")

    @staticmethod
    def _media_uri(path: str) -> str:
        value = Path(path).expanduser().resolve()
        return value.as_uri()
