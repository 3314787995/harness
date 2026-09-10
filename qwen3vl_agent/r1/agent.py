from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from typing import Any

from qwen3vl_agent.coarse_to_fine.types import FrameRef
from qwen3vl_agent.models.base import BaseVideoModel, ModelOutput
from qwen3vl_agent.p01.media import P01IndexBuilder, P01VideoIndex, SourceFrameStore
from qwen3vl_agent.p01.types import TimeSpan
from qwen3vl_agent.r1.config import R1Config
from qwen3vl_agent.r1.control import (
    BudgetExhausted,
    ProtocolError,
    audit_bundle,
    combined_query,
    contains,
    fallback_query,
    intersection,
    parse_discriminants,
    parse_query,
    refuted_fact_ids,
    resolve_scopes,
    strings,
)
from qwen3vl_agent.r1.media import MediaBatch, R1Media
from qwen3vl_agent.r1.observation import apply_observation, observation_parser
from qwen3vl_agent.r1.providers import (
    ExternalEvidenceProvider,
    ExternalSegment,
    NullEvidenceProvider,
    ProviderResult,
)
from qwen3vl_agent.r1.runtime import ModelSession, RunContext
from qwen3vl_agent.r1.types import (
    BindingRecord,
    CoverageRecord,
    DiscriminantSpec,
    EvidenceBundle,
    EvidencePacket,
    QueryField,
    QuerySpec,
    R1Request,
    R1Result,
    SearchCandidate,
    SearchState,
)


@dataclass
class _State:
    request: R1Request
    context: RunContext
    session: ModelSession
    allowed: TimeSpan
    query_scope: TimeSpan | None
    query: QuerySpec
    discriminants: DiscriminantSpec = field(default_factory=DiscriminantSpec)
    index: P01VideoIndex | None = None
    metadata: Any = None
    frames: dict[str, FrameRef] = field(default_factory=dict)
    searches: dict[str, SearchState] = field(default_factory=dict)
    bundles: list[EvidenceBundle] = field(default_factory=list)
    trace: dict[str, Any] = field(default_factory=dict)
    provider_hints: tuple[ExternalSegment, ...] = ()
    provider_cache: dict[tuple[float, float], tuple[ExternalSegment, ...]] = field(
        default_factory=dict
    )
    packet_counter: int = 0
    action_signatures: set[tuple[Any, ...]] = field(default_factory=set)


class R1VideoAgent:
    """Independent R1 controller for preselected questions; P01 is never invoked to solve."""

    config_type = R1Config
    media_type = R1Media
    session_type = ModelSession
    policy_id = "r1-local-evidence/1.0"
    result_namespace = "r1"

    def __init__(
        self,
        model: BaseVideoModel,
        *,
        config: R1Config | Mapping[str, Any] | None = None,
        provider: ExternalEvidenceProvider | None = None,
        index_builder: P01IndexBuilder | None = None,
        source_store: SourceFrameStore | None = None,
    ) -> None:
        self.model = model
        self.config = config if isinstance(config, R1Config) else self.config_type.from_mapping(config)
        self.config.validate()
        self.media = self.media_type(self.config, index_builder, source_store)
        self.provider = provider if provider is not None else NullEvidenceProvider()
        self._loaded = False

    def load(self) -> None:
        self.model.load()
        self._loaded = True

    def unload(self) -> None:
        self.model.unload()
        self._loaded = False

    def generate(
        self,
        messages: list[dict[str, Any]],
        *,
        videos: Sequence[str] | None = None,
        images: Sequence[str] | None = None,
        choices: Sequence[Any] | None = None,
        allowed_scope: Any = None,
        query_scope: Any = None,
        given_interval: Any = None,
        **kwargs: Any,
    ) -> ModelOutput:
        if not self._loaded:
            raise RuntimeError("load the agent before generate")
        if images or not videos or len(videos) != 1:
            raise ValueError("R1 requires one video and no independent image input")
        if kwargs.get("subtitle_path"):
            raise ValueError(
                "inject an ExternalEvidenceProvider; R1 does not generate/parse subtitles"
            )
        query = next((m["content"] for m in reversed(messages) if m.get("role") == "user"), "")
        if not isinstance(query, str):
            raise TypeError("R1 generate requires a plain-text question")
        # The legacy CLI interval intentionally retains its former hard observation boundary.
        request = R1Request(
            str(videos[0]),
            query,
            choices=choices or (),
            allowed_scope=allowed_scope if allowed_scope is not None else given_interval,
            query_scope=query_scope if query_scope is not None else given_interval,
            **{
                k: v
                for k, v in kwargs.items()
                if k
                in {
                    "request_id",
                    "video_id",
                    "group_id",
                    "available_modalities",
                    "output_protocol",
                    "failure_text",
                    "budget",
                }
            },
        )
        result = self.solve(request)
        return ModelOutput(result.output_text, {self.result_namespace: result.to_dict()})

    def solve(self, request: R1Request) -> R1Result:
        context = RunContext(request.budget)
        session = self.session_type(self.model, self.media, self.config, context)
        try:
            metadata = self.media.probe(request.video_path)
            allowed, query_scope = resolve_scopes(request, metadata.duration_seconds)
        except (ValueError, OSError, ImportError) as exc:
            return self._empty_result(request, context, "input_error", str(exc))
        state = _State(
            request,
            context,
            session,
            allowed,
            query_scope,
            fallback_query(request.question),
            metadata=metadata,
        )
        state.trace.update(
            policy_id=self.policy_id,
            request_id=request.request_id,
            video_id=request.video_id,
            group_id=request.group_id,
            allowed_scope=asdict(allowed),
            query_scope=asdict(query_scope) if query_scope else None,
        )
        try:
            self._compile(state)
            self._execute(state)
        except BudgetExhausted as exc:
            context.issues.append(str(exc))
        except (ProtocolError, RuntimeError, OSError, ValueError) as exc:
            context.issues.append(f"execution_error:{type(exc).__name__}:{exc}")
        return self._finish(state)

    def _compile(self, s: _State) -> None:
        payload = {
            "question": s.request.question,
            "allowed_scope": asdict(s.allowed),
            "query_scope": asdict(s.query_scope) if s.query_scope else None,
            "available_modalities": s.request.available_modalities,
        }
        try:
            s.query = s.session.call("query", payload, parser=lambda d, _: parse_query(d)).value
        except (ProtocolError, RuntimeError) as exc:
            if isinstance(exc, BudgetExhausted):
                raise
            s.context.issues.append("query_compiler_unresolved")
        if s.request.choices:
            # Sorted texts erase original labels/order even from the compiler's input.
            payload = {
                "question": s.request.question,
                "query": asdict(s.query),
                "candidate_texts": sorted({c.text for c in s.request.choices}),
            }
            try:
                s.discriminants = s.session.call(
                    "discriminants",
                    payload,
                    parser=lambda d, _: parse_discriminants(
                        d, ocr="ocr" in s.query.observation_modes
                    ),
                ).value
            except (ProtocolError, RuntimeError) as exc:
                if isinstance(exc, BudgetExhausted):
                    raise
                s.context.issues.append("discriminant_compiler_unresolved")
            s.query = combined_query(s.query, s.discriminants)
        s.trace["query_spec"] = asdict(s.query)
        s.trace["discriminant_spec"] = asdict(s.discriminants)
        if set(s.request.available_modalities) & {"subtitle", "asr"}:
            s.provider_hints = self._provider_call(s, "search", s.allowed)

    def _provider_call(
        self, s: _State, operation: str, scope: TimeSpan
    ) -> tuple[ExternalSegment, ...]:
        if not (set(s.request.available_modalities) & {"subtitle", "asr"}):
            return ()
        if len(s.context.provider_calls) >= s.request.budget.max_provider_calls:
            s.context.issues.append("provider_call_budget")
            return ()
        record: dict[str, Any] = {
            "operation": operation,
            "scope": asdict(scope),
            "status": "started",
        }
        s.context.provider_calls.append(record)
        try:
            response = (
                self.provider.search(s.request.question, s.request.video_id, scope)
                if operation == "search"
                else self.provider.read(s.request.video_id, scope)
            )
            if not isinstance(response, ProviderResult):
                raise TypeError("provider must return ProviderResult")
            record.update(
                available=response.available,
                truncated=response.truncated,
                cost=response.cost,
                error=response.error,
            )
            if not response.available or response.error:
                record["status"] = "unavailable"
                return ()
            accepted, rejected, used = [], [], 0
            for item in response.items:
                if not isinstance(item, ExternalSegment):
                    raise TypeError("provider items must be ExternalSegment records")
                valid = (
                    item.source_id == s.request.video_id
                    and item.kind in s.request.available_modalities
                    and contains(scope, TimeSpan(item.start_sec, item.end_sec))
                )
                if not valid:
                    rejected.append(item.segment_id)
                    continue
                if (
                    len(accepted) >= self.config.max_provider_segments
                    or used + len(item.text) > self.config.max_provider_text_chars
                ):
                    record["truncated"] = True
                    break
                if item.segment_id in {v.segment_id for v in accepted}:
                    rejected.append(item.segment_id)
                    continue
                accepted.append(item)
                used += len(item.text)
            record.update(
                status="returned",
                rejected_segment_ids=rejected,
                segment_ids=[v.segment_id for v in accepted],
            )
            return tuple(accepted)
        except Exception as exc:  # noqa: BLE001 - optional provider failures must not abort visual QA
            record.update(status="error", error=f"{type(exc).__name__}: {exc}")
            s.context.issues.append("external_provider_error")
            return ()

    def _read_segments(self, s: _State, scope: TimeSpan) -> tuple[ExternalSegment, ...]:
        key = (scope.start_seconds, scope.end_seconds)
        if key not in s.provider_cache:
            s.provider_cache[key] = self._provider_call(s, "read", scope)
        return s.provider_cache[key]

    def _ensure_index(self, s: _State) -> P01VideoIndex:
        if s.index is None:
            s.index = self.media.navigation(s.request.video_path, s.allowed, s.metadata)
            frames = [
                f
                for f in s.index.cached_video.frames
                if s.allowed.contains_evidence(f.timestamp_seconds, tolerance=1e-6)
            ]
            s.frames.update((f.id, f) for f in frames)
            s.context.decoded_frames += len(frames)
        return s.index

    def _locate(
        self,
        s: _State,
        scope: TimeSpan,
        *,
        reference: bool = False,
        relocation: bool = False,
        reference_packet: EvidencePacket | None = None,
    ) -> list[SearchCandidate]:
        index = self._ensure_index(s)
        key = "reference" if reference else "answer"
        search = s.searches.setdefault(key, SearchState())
        root = index.node(index.root_id)
        pending = list(search.frontier) if relocation else list(root.child_ids or (root.node_id,))
        if relocation:
            search.frontier.clear()
        leaves = []
        for _ in range(self.config.media.locator_max_levels):
            if not pending and search.frontier:
                pending, search.frontier = search.frontier, []
            pending = list(dict.fromkeys(pending))
            pending = [
                n
                for n in pending
                if n not in search.observed_candidates and intersection(index.node(n).span, scope)
            ]
            if not pending or not s.context.can_observe():
                break
            text = (s.request.question + " " + s.query.semantic_hint).casefold()
            reverse = bool(re.search(r"at the end|near the end|ending|结尾|片尾", text))
            pending.sort(key=lambda n: index.node(n).span.start_seconds, reverse=reverse)
            selected_nodes = pending[: self.config.media.locator_branch_factor]
            search.frontier.extend(pending[self.config.media.locator_branch_factor :])
            descriptions, frames = [], []
            node_frames: dict[str, set[str]] = {}
            for node_id in selected_nodes:
                window = intersection(index.node(node_id).span, scope)
                available = [
                    f
                    for f in index.cached_video.frames
                    if window
                    and window.contains_evidence(f.timestamp_seconds, tolerance=1e-6)
                    and f.id not in search.navigation_frame_ids
                ]
                if not available or window is None:
                    continue
                reps = [available[0], available[len(available) // 2], available[-1]]
                reps = list({f.id: f for f in reps}.values())
                node_frames[node_id] = {f.id for f in reps}
                frames.extend(reps)
                descriptions.append(
                    {
                        "node_id": node_id,
                        "span": asdict(window),
                        "frames": [f.to_dict() for f in reps],
                    }
                )
            if not descriptions:
                pending = []
                continue
            frames = list({f.id: f for f in frames}.values())
            batch = MediaBatch(scope, tuple(frames))
            payload = {
                "question": s.request.question,
                "search_role": "reference" if reference else "answer",
                "reference_description": s.query.reference_description if reference else "",
                "semantic_hint": s.query.semantic_hint,
                "allowed_scope": asdict(scope),
                "nodes": descriptions,
                "reference_facts": [asdict(f) for f in reference_packet.facts]
                if reference_packet
                else [],
            }
            # Provider search hits are localization hints, never answer facts.
            payload["text_hints"] = [
                asdict(t)
                for t in s.provider_hints
                if contains(scope, TimeSpan(t.start_sec, t.end_sec))
            ]

            def parse(
                data: dict[str, Any], _: bool, node_frames: dict[str, set[str]] = node_frames
            ) -> list[SearchCandidate]:
                items = data.get("candidates")
                if not isinstance(items, list) or len(items) > self.config.max_candidates:
                    raise ProtocolError("invalid locator candidate count")
                result, seen = [], set()
                for item in items:
                    node_id = item.get("node_id") if isinstance(item, dict) else None
                    if node_id not in node_frames or node_id in seen:
                        raise ProtocolError("locator references an unshown or duplicate node")
                    refs = strings(item.get("anchor_frame_ids", []), "anchor_frame_ids")
                    if not refs or set(refs) - node_frames[node_id]:
                        raise ProtocolError(
                            "locator must cite actual frames from the selected node"
                        )
                    seen.add(node_id)
                    window = intersection(index.node(node_id).span, scope)
                    result.append(
                        SearchCandidate(
                            node_id,
                            window,
                            refs,
                            strings(
                                item.get("matched_anchor_conditions", []), "matched conditions"
                            ),
                            strings(
                                item.get("unresolved_anchor_conditions", []),
                                "unresolved conditions",
                            ),
                        )
                    )
                return result

            result = s.session.call("locator", payload, batch=batch, parser=parse).value
            search.checked_nodes.extend(selected_nodes)
            search.navigation_frame_ids.extend(f.id for f in frames)
            chosen = {c.candidate_id for c in result}
            # Retain unexamined children or other actual frames, never replay identical navigation.
            for node_id in selected_nodes:
                if node_id not in chosen:
                    children = index.node(node_id).child_ids
                    if children:
                        search.frontier.extend(children)
                    elif any(
                        index.node(node_id).span.contains_evidence(f.timestamp_seconds)
                        and f.id not in search.navigation_frame_ids
                        for f in index.cached_video.frames
                    ):
                        search.frontier.append(node_id)
            pending = []
            for candidate in result:
                children = index.node(candidate.candidate_id).child_ids
                if children:
                    pending.extend(children)
                else:
                    leaves.append(candidate)
            if leaves:
                search.frontier.extend(pending)
                break
        if not leaves and pending:
            # Depth-limited nodes remain bounded candidates only if actually localized above.
            leaves = result if "result" in locals() else []
        candidates = []
        for candidate in leaves:
            if candidate.candidate_id in search.observed_candidates:
                continue
            window = self._candidate_window(s, candidate, scope)
            candidates.append(replace(candidate, span=window))
        search.candidates.extend(candidates)
        s.trace["searches"] = {k: asdict(v) for k, v in s.searches.items()}
        return candidates[: self.config.max_candidates]

    def _candidate_window(self, s: _State, c: SearchCandidate, scope: TimeSpan) -> TimeSpan:
        modes, config = set(s.query.observation_modes), self.config.media
        anchor = s.frames[c.anchor_frame_ids[0]].timestamp_seconds
        if "caption" in modes:
            start, end = (
                c.span.start_seconds - config.caption_padding_sec,
                c.span.end_seconds + config.caption_padding_sec,
            )
        elif "ordered" in modes:
            start, end = anchor - config.dynamic_padding_sec, anchor + config.dynamic_padding_sec
        elif "ocr" in modes:
            start, end = anchor - config.ocr_padding_sec, anchor + config.ocr_padding_sec
        else:
            start, end = c.span.start_seconds, c.span.end_seconds
        # Every locator anchor must remain inside the precision observation window.
        times = [s.frames[r].timestamp_seconds for r in c.anchor_frame_ids]
        start, end = min(start, min(times)), max(end, max(times))
        return TimeSpan(max(scope.start_seconds, start), min(scope.end_seconds, end))

    def _execute(self, s: _State) -> None:
        reference_packet = None
        scope = s.query_scope or s.allowed
        if s.query.requires_reference:
            candidates = self._locate(s, s.allowed, reference=True)
            references = []
            for candidate in candidates:
                packet = self._new_packet(s, candidate, "reference")
                self._observe_packet(s, packet)
                s.trace.setdefault("reference_packets", []).append(asdict(packet))
                if packet.anchor_match == "matched" and packet.target_binding == "confirmed":
                    references.append(packet)
            if len(references) != 1:
                s.context.issues.append("reference_anchor_ambiguous_or_missing")
                return
            reference_packet = references[0]
            times = [
                s.frames[r].timestamp_seconds
                for r in reference_packet.anchor_source_ids
                if r in s.frames
            ]
            if not times:
                s.context.issues.append("reference_has_no_visual_anchor")
                return
            if s.query.reference_relation == "before":
                boundary = reference_packet.span.start_seconds
                if boundary <= scope.start_seconds:
                    s.context.issues.append("no_permitted_region_before_reference")
                    return
                scope = TimeSpan(scope.start_seconds, boundary)
            elif s.query.reference_relation == "after":
                boundary = reference_packet.span.end_seconds
                if boundary >= scope.end_seconds:
                    s.context.issues.append("no_permitted_region_after_reference")
                    return
                scope = TimeSpan(boundary, scope.end_seconds)
        self._search_answer(s, scope, reference_packet)

    def _search_answer(self, s: _State, scope: TimeSpan, reference_packet: EvidencePacket | None) -> None:
        if s.query_scope and not s.query.requires_reference:
            candidates = [SearchCandidate("given_interval", scope)]
        else:
            candidates = self._locate(s, scope, reference_packet=reference_packet)
        self._inspect_candidates(s, candidates, reference_packet)
        if self._decisive(s):
            return
        while s.context.refinements < self.config.max_refinements and s.context.can_observe():
            unresolved = [
                b
                for b in s.bundles
                if (not audit_bundle(b, s.query).sufficient or self._competitors(s, b))
                and next(p for p in b.packets if p.role == "answer").anchor_match != "mismatched"
            ]
            if not unresolved:
                break
            bundle = min(
                unresolved, key=lambda b: next(p.rechecks for p in b.packets if p.role == "answer")
            )
            packet = next(p for p in bundle.packets if p.role == "answer")
            self._refine(s, bundle, packet)
        if self._decisive(s):
            return
        if (
            not s.query_scope
            and s.context.can_observe()
            and s.context.relocations < self.config.max_relocations
        ):
            s.context.relocations += 1
            candidates = self._locate(s, scope, relocation=True, reference_packet=reference_packet)
            self._inspect_candidates(s, candidates, reference_packet)
        if (
            s.query.coverage == "existence"
            and not s.query_scope
            and s.context.can_observe()
            and not any(audit_bundle(b, s.query).sufficient for b in s.bundles)
        ):
            # Search misses cannot prove absence. Observe the bounded scope under the remaining budget.
            self._inspect_candidates(
                s, [SearchCandidate("existence_scan", scope)], reference_packet
            )

    def _new_packet(self, s: _State, candidate: SearchCandidate, role: str) -> EvidencePacket:
        s.packet_counter += 1
        packet = EvidencePacket(
            f"packet_{s.packet_counter:03d}", candidate.candidate_id, role, candidate.span
        )
        packet.anchor_source_ids = list(candidate.anchor_frame_ids)
        packet.locator_anchor_ids = candidate.anchor_frame_ids
        return packet

    @staticmethod
    def _competitors(s: _State, chosen: EvidenceBundle) -> list[str]:
        if s.query.coverage == "existence" and any(
            p.existence == "present" for p in chosen.packets
        ):
            return []
        unresolved = []
        for bundle in s.bundles:
            if bundle is chosen:
                continue
            packet = next(p for p in bundle.packets if p.role == "answer")
            excluded = packet.anchor_match == "mismatched" or (
                bool(bundle.bindings) and bundle.bindings[-1].relation == "different"
            )
            if not excluded:
                unresolved.append(bundle.bundle_id)
        search = s.searches.get("answer")
        if search:
            unresolved.extend(
                c.candidate_id
                for c in search.candidates
                if c.candidate_id not in search.observed_candidates
            )
        return list(dict.fromkeys(unresolved))

    def _decisive(self, s: _State) -> bool:
        return any(
            audit_bundle(b, s.query).sufficient and not self._competitors(s, b) for b in s.bundles
        )

    def _inspect_candidates(
        self, s: _State, candidates: list[SearchCandidate], reference: EvidencePacket | None
    ) -> None:
        for candidate in candidates:
            if not s.context.can_observe():
                break
            packet = self._new_packet(s, candidate, "answer")
            required = (
                [s.allowed]
                if s.query.coverage == "existence" and not s.query_scope
                else [candidate.span]
            )
            bundle = EvidenceBundle(
                f"bundle_{len(s.bundles) + 1:03d}",
                ([reference] if reference else []) + [packet],
                required_spans=required,
            )
            s.bundles.append(bundle)
            self._observe_packet(s, packet)
            if "answer" in s.searches:
                s.searches["answer"].observed_candidates.append(candidate.candidate_id)
            if reference and packet.anchor_match == "matched" and s.context.can_observe():
                self._bind(s, bundle, reference, packet)
            # Observe both initial competitors; finding a filled slot in one is not sufficient.

    def _observe_packet(self, s: _State, packet: EvidencePacket) -> None:
        query = s.query
        if packet.role == "reference":
            query = replace(
                query,
                fields=(QueryField("Q1", "Distinctive reference identity features"),),
                anchor_description=query.reference_description,
                coverage="point",
                observation_modes=("static",),
                requires_reference=False,
                requires_speaker_binding=False,
            )
        original_anchors = [s.frames[r] for r in packet.locator_anchor_ids if r in s.frames]
        for span, timestamps, fps in self.media.plan(packet.span, query):
            if not s.context.can_observe():
                s.context.issues.append("model_call_budget")
                break
            try:
                batch = self.media.extract(
                    s.request.video_path,
                    span,
                    timestamps,
                    s.allowed,
                    fps=fps,
                    anchors=original_anchors,
                    ordered=bool({"ordered", "caption"} & set(query.observation_modes)),
                )
                s.context.decoded_frames += len(batch.frames)
                s.frames.update((f.id, f) for f in batch.frames)
                self._observe_batch(s, packet, batch, query)
            except BudgetExhausted:
                raise
            except (ProtocolError, RuntimeError, OSError) as exc:
                packet.coverage.append(CoverageRecord(span, decode_failures=[str(exc)]))
                packet.unresolved = [f"observation_failed:{type(exc).__name__}"]
                s.context.issues.append(f"observation_error:{type(exc).__name__}")
            if packet.anchor_match == "mismatched":
                break
            temporary = EvidenceBundle("audit", [packet], required_spans=[packet.span])
            point_query = replace(query, requires_reference=False)
            if (
                query.coverage == "point"
                or query.coverage == "existence"
                and packet.existence == "present"
            ) and audit_bundle(temporary, point_query).sufficient:
                break

    def _observe_batch(
        self, s: _State, packet: EvidencePacket, batch: MediaBatch, query: QuerySpec
    ) -> None:
        segments = self._read_segments(s, batch.span)
        payload = {
            "question": s.request.question,
            "query": asdict(query),
            "packet_id": packet.packet_id,
            "packet_role": packet.role,
            "packet_span": asdict(packet.span),
            "batch_span": asdict(batch.span),
            "inspection_needs": s.discriminants.inspection_needs,
            "target_union": s.discriminants.target_union,
            "frames": [f.to_dict() for f in batch.frames],
            "crop_transforms": batch.crops,
            "external_segments": [asdict(t) for t in segments],
            "prior_facts": [asdict(f) for f in packet.facts],
            "missing_anchor_ids": batch.missing_anchor_ids,
        }
        payload["available_modalities"] = s.request.available_modalities
        payload["external_service_status"] = [
            {
                key: record.get(key)
                for key in ("operation", "available", "truncated", "error", "status")
            }
            for record in s.context.provider_calls
            if record.get("scope") == asdict(batch.span)
        ]
        signature = (
            packet.packet_id,
            tuple(f.id for f in batch.frames),
            tuple(t.segment_id for t in segments),
            tuple(packet.unresolved),
        )
        if signature in s.action_signatures:
            packet.unresolved = list(
                dict.fromkeys([*packet.unresolved, "unchanged_observation_skipped"])
            )
            return
        s.action_signatures.add(signature)
        result = s.session.call(
            "observe",
            payload,
            batch=batch,
            parser=observation_parser(
                packet, batch, query, s.request.video_id, segments, s.request.available_modalities
            ),
        )
        apply_observation(packet, batch, result)

    def _refine(self, s: _State, bundle: EvidenceBundle, packet: EvidencePacket) -> None:
        s.context.refinements += 1
        packet.rechecks += 1
        if bundle.conflicts:
            fact_ids = {r for item in bundle.conflicts for r in item["fact_ids"]}
            selected = [f for f in packet.facts if f.fact_id in fact_ids]
            source_ids = list(
                dict.fromkeys(
                    r for f in selected for r in (*f.source_frame_ids, *f.original_frame_ids)
                )
            )
            frames = tuple(s.frames[r] for r in source_ids if r in s.frames)
            transforms = {r: value for f in selected for r, value in f.crop_transforms.items()}
            if frames:
                self._observe_batch(
                    s,
                    packet,
                    MediaBatch(packet.span, frames, crops=transforms, coverage_kind="detail"),
                    s.query,
                )
                return
        if packet.crop_requests:
            requests = packet.crop_requests[: self.config.media.max_detail_images]
            frames, transforms = [], {}
            for item in requests:
                source = s.frames.get(item["frame_id"])
                if source is None or not s.allowed.contains_evidence(
                    source.timestamp_seconds, tolerance=1e-6
                ):
                    continue
                crop, transform = self.media.crop(source, item["bbox_xyxy_1000"])
                frames.extend([source, crop])
                transforms[crop.id] = transform
                s.frames[crop.id] = crop
            if frames:
                frames = list({f.id: f for f in frames}.values())
                batch = MediaBatch(
                    packet.span, tuple(frames), crops=transforms, coverage_kind="detail"
                )
                self._observe_batch(s, packet, batch, s.query)
                return
        if s.query.requires_reference and (
            packet.review_request.get("kind") == "identity"
            or not audit_bundle(bundle, s.query).missing_fields
        ):
            reference = next((p for p in bundle.packets if p.role == "reference"), None)
            if reference:
                self._bind(s, bundle, reference, packet)
                return
        # A process may need context beyond the candidate's initial local window. Explicit query
        # intervals remain fixed; expansion is one of the two charged additional observations.
        window = packet.span
        direction = packet.review_request.get("kind", "denser")
        if s.query_scope is None and direction in {"before", "after"}:
            padding = self.config.media.dynamic_padding_sec
            original_window = window
            window = TimeSpan(
                max(
                    s.allowed.start_seconds,
                    window.start_seconds - (padding if direction == "before" else 0),
                ),
                min(
                    s.allowed.end_seconds,
                    window.end_seconds + (padding if direction == "after" else 0),
                ),
            )
            packet.span = window
            bundle.required_spans = [
                window if r == original_window else r for r in bundle.required_spans
            ]
        plan = self.media.plan(window, s.query, refine=True)
        incomplete = [
            r.planned_span
            for r in packet.coverage
            if not r.required_resolution_met
            or r.truncated
            or r.unresolved
            or not r.observation_completed
        ]
        chosen = (
            plan[-1]
            if direction == "after"
            else plan[0]
            if direction == "before"
            else next(
                (item for item in plan if any(intersection(item[0], r) for r in incomplete)),
                plan[0],
            )
        )
        span, times, fps = chosen
        anchors = [s.frames[r] for r in packet.anchor_source_ids if r in s.frames]
        batch = self.media.extract(
            s.request.video_path,
            span,
            times,
            s.allowed,
            fps=fps,
            anchors=anchors,
            ordered=bool({"ordered", "caption"} & set(s.query.observation_modes)),
        )
        s.context.decoded_frames += len(batch.frames)
        s.frames.update((f.id, f) for f in batch.frames)
        self._observe_batch(s, packet, batch, s.query)

    def _bind(
        self, s: _State, bundle: EvidenceBundle, left: EvidencePacket, right: EvidencePacket
    ) -> None:
        left_ids = {r for f in left.facts for r in f.source_frame_ids}
        right_ids = {r for f in right.facts for r in f.source_frame_ids}
        shown_left = sorted(left_ids)[:8]
        shown_right = sorted(right_ids)[:8]
        frames = tuple(
            s.frames[r] for r in dict.fromkeys([*shown_left, *shown_right]) if r in s.frames
        )
        if not shown_left or not shown_right:
            return
        # Independent labelled groups, never a continuous synthetic video over the intervening gap.
        batch = MediaBatch(s.allowed, frames)
        batch.crops = {
            r: transform
            for f in (*left.facts, *right.facts)
            for r, transform in f.crop_transforms.items()
            if r in {f.id for f in frames}
        }
        payload = {
            "question": s.request.question,
            "left": {
                "packet_id": left.packet_id,
                "facts": [asdict(f) for f in left.facts],
                "shown_frame_ids": shown_left,
            },
            "right": {
                "packet_id": right.packet_id,
                "facts": [asdict(f) for f in right.facts],
                "shown_frame_ids": shown_right,
            },
            "frames": [f.to_dict() for f in frames],
        }

        def parse(data: dict[str, Any], limited: bool) -> BindingRecord:
            relation = data.get("relation")
            if relation not in {"same", "different", "unresolved"}:
                raise ProtocolError("unknown binding relation")
            refs = strings(data.get("source_frame_ids", []), "source_frame_ids")
            features = strings(
                data.get("discriminating_features", []),
                "discriminating_features",
                deduplicate=False,
            )
            kinds = strings(data.get("feature_kinds", []), "feature_kinds", deduplicate=False)
            if kinds and len(kinds) != len(features):
                raise ProtocolError("feature_kinds must correspond to identity features")
            if set(refs) - {f.id for f in frames}:
                raise ProtocolError("binding references unseen source")
            generic = {
                "red",
                "blue",
                "green",
                "same colour",
                "same color",
                "same class",
                "airplane",
                "same clothing",
                "same category",
                "person",
                "红色",
                "同类",
                "相同颜色",
            }
            typed = list(zip(features, kinds))
            distinctive = tuple(
                f
                for f, kind in typed
                if kind in {"marking", "distinctive_geometry", "unique_configuration"}
                and f.casefold() not in generic
                and (len(f.split()) > 1 or bool(re.search(r"[\u4e00-\u9fff]{4,}", f)))
            )
            if (
                limited
                or not set(refs) & set(shown_left)
                or not set(refs) & set(shown_right)
                or set(shown_left) & set(shown_right)
                or intersection(left.span, right.span)
                or not distinctive
                or not str(data.get("basis", "")).strip()
            ):
                relation = "unresolved"
            return BindingRecord(
                left.packet_id,
                right.packet_id,
                relation,
                refs,
                str(data.get("basis", "")),
                distinctive,
                tuple(kind for feature, kind in typed if feature in distinctive),
            )

        binding = s.session.call("binding", payload, batch=batch, parser=parse).value
        bundle.bindings.append(binding)

    def _terminal_response(self, s, payload, batch, parser):
        """Override terminal orchestration without changing the default R1 protocol."""
        return s.session.call("final", payload, batch=batch, parser=parser, terminal=True).value

    def _finish(self, s: _State) -> R1Result:
        audits = [(b, audit_bundle(b, s.query)) for b in s.bundles]
        audits.sort(
            key=lambda item: (
                not item[1].sufficient,
                len(item[1].missing_fields),
                len(item[1].unresolved),
                -len(item[1].clear_fact_ids),
            )
        )
        bundle, audit = audits[0] if audits else (None, None)
        s.trace["candidate_bundles"] = [b.to_dict() for b in s.bundles]
        s.trace["searches"] = {key: asdict(value) for key, value in s.searches.items()}
        s.trace["candidate_audits"] = {b.bundle_id: asdict(a) for b, a in audits}
        s.trace["issues"] = list(dict.fromkeys(s.context.issues))
        if bundle is None or not bundle.facts:
            missing_modalities = self._missing_modalities(s, bundle or EvidenceBundle("empty"))
            s.context.issues.extend(missing_modalities)
            completion = (
                "modality_unavailable"
                if missing_modalities
                else "budget_limited"
                if any("budget" in reason for reason in s.context.issues)
                else "protocol_error"
                if any("observation_error" in reason for reason in s.context.issues)
                else "evidence_unresolved"
            )
            result = self._empty_result(s.request, s.context, completion, "no_valid_evidence")
            result.trace = s.trace
            result.evidence_bundle = bundle
            return result
        facts = {f.fact_id: f for f in bundle.facts}
        refuted = refuted_fact_ids(bundle)
        frame_ids = list(dict.fromkeys(r for f in bundle.facts for r in f.source_frame_ids))
        limit = self.config.media.decision_max_frames
        # Share the cap across packets so a long first packet cannot hide its complement.
        groups = [
            list(dict.fromkeys(r for f in p.facts for r in f.source_frame_ids))
            for p in bundle.packets
        ]
        chosen = []
        while any(groups) and len(chosen) < limit:
            for group in groups:
                if group and len(chosen) < limit:
                    item = group.pop(0)
                    if item not in chosen:
                        chosen.append(item)
        s.trace["terminal_previously_observed_omitted_frame_ids"] = [
            r for r in frame_ids if r not in chosen
        ]
        frames = tuple(s.frames[r] for r in chosen if r in s.frames)
        transforms = {
            key: value
            for p in bundle.packets
            for view in p.source_views
            for key, value in view.get("crop_transforms", {}).items()
            if key in chosen
        }
        batch = MediaBatch(s.allowed, frames, crops=transforms) if frames else None
        frozen = {
            "bundle_id": bundle.bundle_id,
            "bindings": [asdict(b) for b in bundle.bindings],
            "conflicts": bundle.conflicts,
            "refuted_fact_ids": sorted(refuted),
            "packets": [
                {
                    "packet_id": p.packet_id,
                    "role": p.role,
                    "span": asdict(p.span),
                    "facts": [asdict(f) for f in p.facts],
                    "fact_reviews": p.fact_reviews,
                    "anchor_match": p.anchor_match,
                    "target_binding": p.target_binding,
                    "unresolved": p.unresolved,
                }
                for p in bundle.packets
            ],
        }
        payload = {
            "question": s.request.question,
            "choices": [asdict(c) for c in s.request.choices],
            "output_protocol": s.request.output_protocol,
            "frozen_bundle": frozen,
            "evidence_audit": asdict(audit),
            "frames": [f.to_dict() for f in frames],
            "competing_bundles_unresolved": self._competitors(s, bundle),
        }
        if batch is not None:
            try:
                prepared = self.media.prepare(batch)
            except (OSError, ValueError, BudgetExhausted) as exc:
                s.trace["terminal_media_omitted_reason"] = f"media_unavailable:{type(exc).__name__}"
                s.context.issues.append("terminal_media_unavailable")
                payload["frames"] = []
                batch, prepared = None, None
            remaining_frames = s.request.budget.max_frame_exposures - s.context.frame_exposures
            remaining_pixels = s.request.budget.max_media_pixels - s.context.media_pixels
            remaining_tokens = (
                None
                if s.request.budget.max_visual_tokens is None
                else s.request.budget.max_visual_tokens - s.context.visual_tokens_estimated
            )
            if prepared is not None and (
                len(frames) > remaining_frames
                or prepared.pixels > remaining_pixels
                or remaining_tokens is not None
                and (prepared.pixels + 1023) // 1024 > remaining_tokens
            ):
                # Facts have already been observed and audited; the reserved terminal calls can
                # still compose from those records when another media exposure cannot fit.
                s.trace["terminal_media_omitted_reason"] = "remaining_media_budget"
                payload["frames"] = []
                batch = None
        claims: list[dict[str, Any]] = []
        prediction = None
        supported = False
        basis = "best_effort"
        terminal_failed = False

        def parse(data: dict[str, Any], limited: bool) -> dict[str, Any]:
            value = data.get("prediction")
            if not isinstance(value, str) or not value.strip():
                raise ProtocolError("terminal prediction is empty")
            value = value.strip()
            if s.request.choices and value not in {c.label for c in s.request.choices}:
                raise ProtocolError("prediction must be an original choice label")
            if s.request.output_protocol == "numeric":
                readings = {
                    f.structured_value.strip()
                    for f in facts.values()
                    if f.observation_status == "clear"
                    and not f.uncertain_characters
                    and f.fact_id not in refuted
                }
                if value not in readings or not re.search(r"\d", value):
                    raise ProtocolError(
                        "numeric output must preserve an observed readout and units"
                    )
            refs = strings(data.get("evidence_fact_ids", []), "evidence_fact_ids", limit=64)
            if set(refs) - set(facts):
                raise ProtocolError("terminal referenced unfrozen facts")
            raw_claims = data.get("claims", [])
            if not isinstance(raw_claims, list):
                raise ProtocolError("claims must be an array")
            parsed_claims = []
            for item in raw_claims:
                if not isinstance(item, dict):
                    raise ProtocolError("invalid claim record")
                ids = strings(item.get("fact_ids", []), "claim fact_ids", limit=64)
                if set(ids) - set(facts):
                    raise ProtocolError("claim references unknown facts")
                statement = str(item.get("statement", ""))
                if statement not in {facts[r].statement for r in ids}:
                    raise ProtocolError(
                        "terminal claim introduces an assertion outside frozen facts"
                    )
                parsed_claims.append({"statement": statement, "fact_ids": list(ids)})
            all_refs = set(refs) | {r for c in parsed_claims for r in c["fact_ids"]}
            grounded = (
                bool(refs)
                and all(facts[r].observation_status == "clear" for r in refs)
                and bool(parsed_claims)
                and all(c["fact_ids"] for c in parsed_claims)
                and all(
                    facts[r].observation_status == "clear"
                    and not facts[r].uncertain_characters
                    and r not in refuted
                    for r in all_refs
                )
            )
            judged = data.get("answer_supported") is True
            assessments = data.get("choice_assessments", [])
            if not isinstance(assessments, list):
                raise ProtocolError("choice_assessments must be an array")
            mapped = {}
            for item in assessments:
                if not isinstance(item, dict) or item.get("label") in mapped:
                    raise ProtocolError("invalid or duplicate choice assessment")
                ids = strings(item.get("fact_ids", []), "choice fact_ids", limit=64)
                if set(ids) - set(facts):
                    raise ProtocolError("choice assessment references unfrozen facts")
                mapped[item.get("label")] = (item.get("status"), ids)
            discriminated = not s.request.choices or (
                set(mapped) == {c.label for c in s.request.choices}
                and all(
                    status == ("supported" if label == value else "rejected")
                    and ids
                    and all(
                        facts[r].observation_status == "clear" and r not in refuted for r in ids
                    )
                    for label, (status, ids) in mapped.items()
                )
            )
            return {
                "prediction": value,
                "claims": parsed_claims,
                "choice_assessments": assessments,
                "supported": grounded and judged and discriminated and not limited,
            }

        try:
            final = self._terminal_response(s, payload, batch, parse)
            prediction, claims = final["prediction"], final["claims"]
            s.trace["choice_assessments"] = final["choice_assessments"]
            supported = bool(final["supported"] and audit and audit.sufficient)
        except (ProtocolError, RuntimeError, OSError, TypeError) as exc:
            terminal_failed = True
            s.context.issues.append(f"terminal_error:{type(exc).__name__}:{exc}")
            # Only terminal responses may supply the fallback label, never locator/observer prose.
            for call in reversed(s.context.calls):
                if call["role"] != "final" and call.get("parsed_role") != "final":
                    continue
                raw = str(call.get("raw_response", ""))
                for choice in s.request.choices:
                    if raw.strip().strip('`"') == choice.label or re.search(
                        rf'"prediction"\s*:\s*"{re.escape(choice.label)}"', raw
                    ):
                        prediction = choice.label
                        break
                if s.request.output_protocol == "numeric":
                    reading = raw.strip().strip('`"')
                    if reading in {
                        f.structured_value
                        for f in facts.values()
                        if f.observation_status == "clear"
                        and not f.uncertain_characters
                        and f.fact_id not in refuted
                    } and re.search(r"\d", reading):
                        prediction = reading
                if prediction:
                    break
        if prediction is None:
            if s.request.choices:
                prediction, basis = s.request.choices[0].label, "unbacked_fallback"
            elif s.request.output_protocol == "numeric":
                prediction = None
            else:
                prediction = " ".join(
                    f.statement
                    for p in bundle.packets
                    if p.role == "answer"
                    for f in p.facts
                    if f.observation_status == "clear" and f.fact_id not in refuted
                )
                prediction = prediction or s.request.failure_text
        if prediction is None or (
            not s.request.choices
            and prediction == s.request.failure_text
            and not any(f.observation_status == "clear" for f in facts.values())
        ):
            basis = "unbacked_fallback"
        if self._competitors(s, bundle):
            supported = False
            s.context.issues.append("competing_candidates_unresolved")
        if (
            "query_compiler_unresolved" in s.context.issues
            or "discriminant_compiler_unresolved" in s.context.issues
            or "terminal_media_unavailable" in s.context.issues
        ):
            supported = False
        missing_modality = self._missing_modalities(s, bundle)
        if missing_modality:
            supported = False
        if supported:
            basis, level, completion = "evidence", "supported", "complete"
        else:
            level = "none" if basis == "unbacked_fallback" else "partial"
            completion = (
                "modality_unavailable"
                if missing_modality
                else "budget_limited"
                if any("budget" in issue for issue in s.context.issues)
                else "protocol_error"
                if terminal_failed
                else "evidence_unresolved"
            )
        unresolved = list(
            dict.fromkeys(
                [
                    *(audit.unresolved if audit else []),
                    *(audit.missing_fields if audit else []),
                    *missing_modality,
                    *s.context.issues,
                ]
            )
        )
        if supported:
            unresolved = []
        s.trace["issues"] = list(dict.fromkeys(s.context.issues))
        return R1Result(
            prediction,
            basis,
            level,
            completion,
            bundle,
            claims,
            unresolved,
            s.context.summary(),
            s.trace,
        )

    @staticmethod
    def _missing_modalities(s: _State, bundle: EvidenceBundle) -> list[str]:
        used = {
            r for f in bundle.facts if f.observation_status == "clear" for r in f.source_segment_ids
        }
        actual = {
            t.kind for values in s.provider_cache.values() for t in values if t.segment_id in used
        }
        actual.update(f.source_kind for f in bundle.facts if f.observation_status == "clear")
        if "visual" in actual:
            actual.add("video")
        return [
            f"required_modality_unavailable:{m}"
            for m in s.query.required_modalities
            if m not in actual or m not in s.request.available_modalities
        ]

    @staticmethod
    def _empty_result(request: R1Request, context: RunContext, state: str, reason: str) -> R1Result:
        prediction = (
            request.choices[0].label
            if request.choices
            else None
            if request.output_protocol == "numeric"
            else request.failure_text
        )
        return R1Result(
            prediction,
            "unbacked_fallback",
            "none",
            state,
            unresolved_reasons=[reason, *context.issues],
            resources=context.summary(),
        )
