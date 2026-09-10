"""R1 V3: bounded visual recovery and source-linked, append-only evidence."""

import copy
from dataclasses import asdict, replace
from itertools import combinations

from qwen3vl_agent.p01.types import TimeSpan
from qwen3vl_agent.r1.agent import R1VideoAgent
from qwen3vl_agent.r1.control import (
    BudgetExhausted,
    ProtocolError,
    audit_bundle,
    intersection,
    strings,
)
from qwen3vl_agent.r1.media import MediaBatch, coverage_record
from qwen3vl_agent.r1.types import EvidenceBundle, QueryField, SearchCandidate
from qwen3vl_agent.r1_v2.agent import R1V2VideoAgent
from qwen3vl_agent.r1_v2.config import R1V2Config
from qwen3vl_agent.r1_v3.config import R1V3Config
from qwen3vl_agent.r1_v3.evidence import modality_state, refresh_binding_tasks
from qwen3vl_agent.r1_v3.observation import originals, parse_observation, rebuild
from qwen3vl_agent.r1_v3.runtime import R1V3ModelSession
from qwen3vl_agent.r1_v3.types import CandidateLink, ObservationRecord, V3Packet
from qwen3vl_agent.r1_v3.version import POLICY_ID


class R1V3VideoAgent(R1V2VideoAgent):
    config_type = R1V3Config
    session_type = R1V3ModelSession
    policy_id = POLICY_ID
    result_namespace = "r1_v3"

    def solve(self, request):
        if set(request.available_modalities) & {"subtitle", "asr"}:
            compat = R1V2VideoAgent(
                self.model,
                config=R1V2Config.from_mapping(asdict(self.config)),
                provider=self.provider,
                index_builder=self.media.index_builder,
                source_store=self.media.source_store,
            )
            result = compat.solve(request)
            result.trace.update(
                execution_branch="v2_external_text_compat", requested_policy_id=self.policy_id
            )
            return result
        result = super().solve(request)
        result.trace["execution_branch"] = "v3_visual"
        return result

    def _new_packet(self, s, candidate, role):
        s.packet_counter += 1
        return V3Packet(
            f"packet_{s.packet_counter:03d}",
            candidate.candidate_id,
            role,
            candidate.span,
            locator_anchor_ids=candidate.anchor_frame_ids,
        )

    @staticmethod
    def _query(s, packet):
        if packet.role != "reference":
            return s.query
        return replace(
            s.query,
            fields=(QueryField("Q1", "Distinctive reference identity features"),),
            anchor_description=s.query.reference_description,
            coverage="point",
            observation_modes=("static",),
            requires_reference=False,
        )

    @staticmethod
    def _review_payload(records):
        return [
            {
                "record_id": r.record_id,
                "target": r.target,
                "facts": [asdict(f) for f in r.facts],
                "gaps": r.gaps,
                "errors": r.errors,
                "original_source_ids": sorted(originals(r.batch, [f.id for f in r.batch.frames])),
            }
            for r in records
        ]

    def _observe_batch(
        self,
        s,
        packet,
        batch,
        query,
        *,
        purpose="initial",
        review_records=(),
        refinement=False,
        binding_task=None,
    ):
        signature = (
            packet.packet_id,
            purpose,
            tuple(f.id for f in batch.frames),
            tuple(r.record_id for r in review_records),
        )
        decisions = s.trace.setdefault("action_decisions", [])
        if signature in s.action_signatures:
            decisions.append(
                {
                    "packet_id": packet.packet_id,
                    "purpose": purpose,
                    "status": "skipped_duplicate",
                    "charged": False,
                }
            )
            return False
        if refinement and s.context.refinements >= self.config.max_refinements:
            return False
        if not s.context.can_observe() or not batch.frames:
            return False
        s.action_signatures.add(signature)
        action = {
            "packet_id": packet.packet_id,
            "purpose": purpose,
            "span": asdict(batch.span),
            "source_frame_ids": [f.id for f in batch.frames],
            "review_record_ids": [r.record_id for r in review_records],
            "charged": False,
        }
        decisions.append(action)
        record_id = f"{packet.packet_id}.o{len(packet.observations) + 1:03d}"
        payload = {
            "question": s.request.question,
            "query": asdict(query),
            "packet_id": packet.packet_id,
            "packet_role": packet.role,
            "packet_span": asdict(packet.span),
            "batch_span": asdict(batch.span),
            "purpose": purpose,
            "inspection_needs": s.discriminants.inspection_needs,
            "target_union": s.discriminants.target_union,
            "frames": [f.to_dict() for f in batch.frames],
            "crop_transforms": batch.crops,
            "review_records": self._review_payload(review_records),
        }
        if binding_task is not None:
            payload["query_binding_task"] = asdict(binding_task)
            payload["missing_fields"] = [
                asdict(f) for f in query.fields if f.field_id in binding_task.field_ids
            ]
            action["task_id"] = binding_task.task_id
        start = len(s.context.calls)
        result, parsed, raw, call_id, coverage = None, None, "", "", None
        try:
            result = s.session.call(
                "observe",
                payload,
                batch=batch,
                parser=lambda d, limited: parse_observation(
                    d,
                    limited,
                    packet=packet,
                    batch=batch,
                    query=query,
                    source_id=s.request.video_id,
                    record_id=record_id,
                    review_ids=[r.record_id for r in review_records],
                ),
            )
            parsed, raw, call_id = result.value, result.raw, result.call_id
            coverage = coverage_record(
                batch, result.prepared, completed=True, truncated=parsed["truncated"]
            )
            coverage.unresolved.extend(parsed["coverage_gaps"])
            action["status"] = "partial_units" if parsed["errors"] else "accepted"
        except BudgetExhausted:
            action["status"] = "budget_stopped"
            raise
        except (ProtocolError, RuntimeError, OSError, ValueError) as exc:
            calls = s.context.calls[start:]
            if calls:
                raw, call_id = calls[-1].get("raw_response", ""), calls[-1]["call_id"]
            parsed = {
                "target": {
                    "status": "unresolved",
                    "description": "",
                    "source_frame_ids": [],
                    "unresolved_conditions": [],
                },
                "facts": (),
                "gaps": (),
                "errors": ({"unit": "protocol", "reason": str(exc)},),
                "reviews": (),
                "existence": "unknown",
                "absence_basis": "",
            }
            action.update(status="failed", error=f"{type(exc).__name__}:{exc}")
        finally:
            issued = any(c["role"] == "observe" for c in s.context.calls[start:])
            if issued and refinement:
                s.context.refinements += 1
                packet.rechecks += 1
                action["charged"] = True
            action["model_calls"] = len(s.context.calls) - start
        if parsed is not None:
            packet.observations.append(
                ObservationRecord(
                    record_id,
                    call_id,
                    purpose,
                    raw,
                    copy.deepcopy(batch),
                    parsed["target"],
                    parsed["facts"],
                    parsed["gaps"],
                    parsed["errors"],
                    parsed["reviews"],
                    coverage,
                    parsed["existence"],
                    parsed["absence_basis"],
                    binding_task.task_id if binding_task else "",
                )
            )
            rebuild(packet)
            refresh_binding_tasks(packet, query)
            self._record_attempt(s, packet)
        return issued

    def _observe_packet(self, s, packet):
        query = self._query(s, packet)
        anchors = [s.frames[r] for r in packet.locator_anchor_ids if r in s.frames]
        for span, times, fps in self.media.plan(packet.span, query):
            if not s.context.can_observe():
                break
            batch = self.media.extract(
                s.request.video_path,
                span,
                times,
                s.allowed,
                fps=fps,
                anchors=anchors,
                ordered=bool({"ordered", "caption"} & set(query.observation_modes)),
            )
            s.context.decoded_frames += len(batch.frames)
            s.frames.update((f.id, f) for f in batch.frames)
            self._observe_batch(s, packet, batch, query)
            if packet.anchor_match == "mismatched":
                break
            if (
                query.coverage == "point"
                and audit_bundle(
                    EvidenceBundle("local", [packet], required_spans=[packet.span]), query
                ).sufficient
            ):
                break
        if packet.role == "reference" and self._urgent(packet):
            self._recheck(s, packet, self._urgent(packet))

    @staticmethod
    def _retired(packet):
        if packet.merged_into or packet.anchor_match == "mismatched":
            return True
        relation_evidence = any(r.target["source_frame_ids"] for r in packet.observations)
        temporal = any(g["kind"] == "temporal_selection" for g in packet.active_gaps)
        return bool(
            packet.observations
            and not relation_evidence
            and not packet.active_errors
            and not temporal
        )

    def _record_attempt(self, s, packet):
        if packet.role != "answer":
            return
        search = s.searches.get("answer")
        if search is None:
            return
        retired = self._retired(packet)
        if retired:
            search.retired_candidates[packet.candidate_id] = (
                "target_mismatched"
                if (packet.anchor_match == "mismatched")
                else "no_target_relation_evidence"
            )
        else:
            search.retired_candidates.pop(packet.candidate_id, None)
        search.attempted_windows.append(
            {
                "candidate_id": packet.candidate_id,
                "node_id": search.candidate_nodes.get(packet.candidate_id),
                "packet_id": packet.packet_id,
                "span": asdict(packet.span),
                "anchor_frame_ids": list(packet.locator_anchor_ids),
                "anchor_match": packet.anchor_match,
                "target_binding": packet.target_binding,
                "fact_count": len(packet.facts),
                "bound_fact_ids": list(packet.bound_fact_ids),
                "retired_to_search": retired,
                "reason": search.retired_candidates.get(packet.candidate_id),
            }
        )

    @staticmethod
    def _answer(bundle):
        return next(p for p in bundle.packets if p.role == "answer")

    @staticmethod
    def _competitors(s, chosen):
        if s.query.coverage == "existence" and any(
            p.existence == "present" for p in chosen.packets
        ):
            return []
        result = []
        for bundle in s.bundles:
            if bundle is chosen:
                continue
            packet = R1V3VideoAgent._answer(bundle)
            excluded = bool(bundle.bindings and bundle.bindings[-1].relation == "different")
            if not excluded and not R1V3VideoAgent._retired(packet):
                result.append(bundle.bundle_id)
        search = s.searches.get("answer")
        if search:
            result.extend(
                c.candidate_id
                for c in search.candidates
                if c.candidate_id not in search.observed_candidates
            )
        return result

    def _decisive(self, s):
        return any(
            not self._retired(self._answer(b))
            and audit_bundle(b, s.query).sufficient
            and not self._competitors(s, b)
            for b in s.bundles
        )

    @staticmethod
    def _urgent(packet):
        if "target_observation_conflict" in packet.unresolved:
            return "conflict"
        if any(g["kind"] == "conflict" for g in packet.active_gaps):
            return "conflict"
        if packet.active_errors:
            return "output_binding"
        return None

    def _recheck(self, s, packet, purpose):
        if not packet.observations or s.context.refinements >= self.config.max_refinements:
            return False
        ids = (
            {e["record_id"] for e in packet.active_errors}
            if purpose == "output_binding"
            else {g["record_id"] for g in packet.active_gaps if g["kind"] == "conflict"}
        )
        records = [r for r in packet.observations if not ids or r.record_id in ids]
        # Preserve the entire original batch for binding repair; partial source replay cannot
        # establish that an unobserved part of an ordered batch was checked.
        selected = records if purpose == "conflict" else records[:1]
        refs = list(dict.fromkeys(f.id for r in selected for f in r.batch.frames))
        if len(refs) > self.config.media.caption_refine_max_frames:
            s.trace.setdefault("action_decisions", []).append(
                {
                    "packet_id": packet.packet_id,
                    "purpose": purpose,
                    "status": "source_set_exceeds_cap",
                    "charged": False,
                }
            )
            return False
        frames = tuple(s.frames[r] for r in refs if r in s.frames)
        span = TimeSpan(
            min(r.batch.span.start_seconds for r in selected),
            max(r.batch.span.end_seconds for r in selected),
        )
        crops = {k: v for r in selected for k, v in r.batch.crops.items()}
        original = selected[0].batch
        # Complete a failed reading of the exact original span, without adding a new span.
        full_replay = len(selected) == 1 and purpose == "output_binding"
        batch = MediaBatch(
            span,
            frames,
            crops=crops,
            requested_fps=original.requested_fps if full_replay else None,
            ordered=original.ordered if full_replay else False,
            coverage_kind=original.coverage_kind if full_replay else "detail",
        )
        return self._observe_batch(
            s,
            packet,
            batch,
            self._query(s, packet),
            purpose=purpose,
            review_records=selected,
            refinement=True,
        )

    def _detail(self, s, bundle, packet):
        if s.context.refinements >= self.config.max_refinements:
            return False
        if self._urgent(packet) or bundle.conflicts:
            return False
        query = self._query(s, packet)
        gaps = [g for g in packet.active_gaps if g["kind"] != "conflict"]
        if gaps:
            gap = gaps[0]
        elif any(
            f.target_confirmed and not f.observation_clear and not f.refuted
            for f in packet.fact_eligibility
        ):
            gap = {"kind": "detail", "reason": "original_observation_quality_insufficient"}
        elif "required_observation_incomplete" in audit_bundle(bundle, query).unresolved:
            gap = {"kind": "detail", "reason": "actual_coverage_incomplete"}
        else:
            # Missing field association is not a visual-resolution problem.
            return False
        purpose = gap["kind"]
        if purpose == "target_identity":
            return self._recheck(s, packet, "target_identity")
        window = packet.span
        crop = gap.get("crop")
        if crop:
            source = s.frames[crop["frame_id"]]
            detail, transform = self.media.crop(source, crop["bbox_xyxy_1000"])
            s.frames[detail.id] = detail
            frames = {r: s.frames[r] for r in packet.anchor_source_ids if r in s.frames}
            frames.update({source.id: source, detail.id: detail})
            batch = MediaBatch(
                window, tuple(frames.values()), crops={detail.id: transform}, coverage_kind="detail"
            )
        else:
            direction = gap.get("direction")
            if purpose == "temporal_context" and s.query_scope is None:
                padding = self.config.media.dynamic_padding_sec
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
            plans = self.media.plan(window, query, refine=True)
            incomplete = [
                r.planned_span
                for r in packet.coverage
                if not r.required_resolution_met or r.unresolved or r.truncated
            ]
            chosen = (
                plans[-1]
                if direction == "after"
                else next(
                    (p for p in plans if any(intersection(p[0], r) for r in incomplete)), plans[0]
                )
            )
            span, times, fps = chosen
            anchors = list(dict.fromkeys([*packet.locator_anchor_ids, *packet.anchor_source_ids]))
            batch = self.media.extract(
                s.request.video_path,
                span,
                times,
                s.allowed,
                fps=fps,
                anchors=[s.frames[r] for r in anchors if r in s.frames],
                ordered=bool({"ordered", "caption"} & set(query.observation_modes)),
            )
            s.context.decoded_frames += len(batch.frames)
            s.frames.update((f.id, f) for f in batch.frames)
        records = [r for r in packet.observations if r.record_id == gap.get("record_id")]
        issued = self._observe_batch(
            s, packet, batch, query, purpose=purpose, review_records=records, refinement=True
        )
        if issued and window != packet.span:
            original = packet.span
            packet.span = window
            bundle.required_spans = [window if x == original else x for x in bundle.required_spans]
        return issued

    def _query_binding_recheck(self, s, packet):
        query = self._query(s, packet)
        tasks = refresh_binding_tasks(packet, query)
        if not tasks or s.context.refinements >= self.config.max_refinements:
            return False
        task = tasks[0]
        records = [r for r in packet.observations if r.record_id in task.origin_record_ids]
        refs = task.source_frame_ids
        if len(refs) > self.config.media.caption_refine_max_frames or any(
            r not in s.frames for r in refs
        ):
            task.status, task.reason = "blocked", "original_source_set_unavailable_or_exceeds_cap"
            s.trace.setdefault("action_decisions", []).append(
                {
                    "purpose": "query_binding",
                    "task_id": task.task_id,
                    "status": task.reason,
                    "charged": False,
                }
            )
            return False
        # Use original target/fact support frames. Re-reading contributes no new time coverage.
        batch = MediaBatch(
            TimeSpan(
                min(r.batch.span.start_seconds for r in records),
                max(r.batch.span.end_seconds for r in records),
            ),
            tuple(sorted((s.frames[r] for r in refs), key=lambda f: f.timestamp_seconds)),
            ordered=any(r.batch.ordered for r in records),
            coverage_kind="detail",
        )
        start = len(s.context.calls)
        before = len(packet.observations)
        try:
            return self._observe_batch(
                s,
                packet,
                batch,
                query,
                purpose="query_binding",
                review_records=records,
                refinement=True,
                binding_task=task,
            )
        finally:
            if any(c["role"] == "observe" for c in s.context.calls[start:]):
                task.attempted = True
                task.result_record_ids.extend(r.record_id for r in packet.observations[before:])
            refresh_binding_tasks(packet, query)

    @staticmethod
    def _comparison_sources(packet):
        crops = {k: v for r in packet.observations for k, v in r.batch.crops.items()}
        priority = list(
            dict.fromkeys(
                crops.get(x, {}).get("source_frame_id", x)
                for x in (
                    packet.anchor_source_ids
                    or [x for r in packet.observations for x in r.target["source_frame_ids"]]
                )
            )
        )
        fields = list(
            dict.fromkeys(
                x
                for f in packet.facts
                if f.supports_query_fields
                for x in (f.original_frame_ids or f.source_frame_ids)
            )
        )
        return list(dict.fromkeys([*priority, *fields]))[:8]

    def _compare(self, s, left_bundle, right_bundle):
        if s.context.refinements >= self.config.max_refinements:
            return False
        left, right = self._answer(left_bundle), self._answer(right_bundle)
        if left.merged_into or right.merged_into or not s.context.can_observe():
            return False
        a, b = self._comparison_sources(left), self._comparison_sources(right)
        if not a or not b:
            return False
        left_records = tuple(r.record_id for r in left.observations)
        right_records = tuple(r.record_id for r in right.observations)
        signature = (
            "candidate_review",
            left.packet_id,
            right.packet_id,
            tuple(a),
            tuple(b),
            left_records,
            right_records,
        )
        if signature in s.action_signatures:
            return False
        s.action_signatures.add(signature)
        batch = MediaBatch(
            s.allowed, tuple(s.frames[r] for r in dict.fromkeys([*a, *b])), coverage_kind="detail"
        )
        payload = {
            "question": s.request.question,
            "left": {
                "candidate_id": left.candidate_id,
                "source_frame_ids": a,
                "record_ids": left_records,
                "targets": [r.target for r in left.observations],
            },
            "right": {
                "candidate_id": right.candidate_id,
                "source_frame_ids": b,
                "record_ids": right_records,
                "targets": [r.target for r in right.observations],
            },
            "frames": [f.to_dict() for f in batch.frames],
        }

        def parse(data, limited):
            relation = data.get("relation")
            if relation not in {"same", "different", "unresolved"}:
                raise ProtocolError("invalid_candidate_relation")
            ar = strings(data.get("left_source_ids", []), "left_source_ids")
            br = strings(data.get("right_source_ids", []), "right_source_ids")
            if set(ar) - set(a) or set(br) - set(b):
                raise ProtocolError("candidate_link_references_unshown_source")
            kind, basis = data.get("basis_kind", "unresolved"), str(data.get("basis", ""))
            if (
                limited
                or not ar
                or not br
                or not basis.strip()
                or kind not in {"same_source_target", "discriminating_features"}
                or kind == "same_source_target"
                and not set(ar) & set(br)
                or relation == "same"
                and data.get("same_occurrence") is not True
            ):
                relation = "unresolved"
            return relation, ar, br, basis, kind, data.get("same_occurrence") is True

        start = len(s.context.calls)
        action = {
            "purpose": "candidate_review",
            "left_id": left.candidate_id,
            "right_id": right.candidate_id,
            "left_source_ids": a,
            "right_source_ids": b,
            "left_record_ids": left_records,
            "right_record_ids": right_records,
            "charged": False,
        }
        s.trace.setdefault("action_decisions", []).append(action)
        try:
            result = s.session.call("candidate_review", payload, batch=batch, parser=parse)
            relation, ar, br, basis, kind, same_occurrence = result.value
            link = CandidateLink(
                result.call_id,
                left.candidate_id,
                right.candidate_id,
                relation,
                ar,
                br,
                basis,
                kind,
                same_occurrence,
            )
            s.trace.setdefault("candidate_links", []).append(asdict(link))
            action["status"] = relation
            if relation == "same":
                left.observations.extend(right.observations)
                left.aliases.extend([right.candidate_id, *right.aliases])
                left.span = TimeSpan(
                    min(left.span.start_seconds, right.span.start_seconds),
                    max(left.span.end_seconds, right.span.end_seconds),
                )
                left.locator_anchor_ids = tuple(
                    dict.fromkeys([*left.locator_anchor_ids, *right.locator_anchor_ids])
                )
                left_bundle.required_spans.extend(
                    x for x in right_bundle.required_spans if x not in left_bundle.required_spans
                )
                right.merged_into = left.packet_id
                left.query_binding_tasks.extend(right.query_binding_tasks)
                rebuild(left)
                refresh_binding_tasks(left, self._query(s, left))
                self._record_attempt(s, right)
        except BudgetExhausted:
            raise
        except (ProtocolError, RuntimeError, OSError, ValueError) as exc:
            action.update(status="failed", error=f"{type(exc).__name__}:{exc}")
            calls = s.context.calls[start:]
            s.trace.setdefault("candidate_links", []).append(
                asdict(
                    CandidateLink(
                        calls[-1]["call_id"] if calls else "",
                        left.candidate_id,
                        right.candidate_id,
                        "unresolved",
                        (),
                        (),
                        str(exc),
                        "unresolved",
                    )
                )
            )
        finally:
            issued = any(c["role"] == "candidate_review" for c in s.context.calls[start:])
            if issued:
                s.context.refinements += 1
                action["charged"] = True
            action["model_calls"] = len(s.context.calls) - start
        return issued

    def _search_answer(self, s, scope, reference_packet):
        candidates = (
            [SearchCandidate("given_interval", scope)]
            if s.query_scope and not s.query.requires_reference
            else self._locate(s, scope, reference_packet=reference_packet)
        )
        self._inspect_candidates(s, candidates, reference_packet)

        def relocate(reason):
            if (
                s.query_scope
                or s.context.relocations >= self.config.max_relocations
                or not s.context.can_observe()
            ):
                return False
            s.context.relocations += 1
            s.trace.setdefault("relocation_decisions", []).append(
                {
                    "reason": reason,
                    "scope": asdict(scope),
                    "model_calls_before": len(s.context.calls),
                }
            )
            found = self._locate(s, scope, relocation=True, reference_packet=reference_packet)
            self._inspect_candidates(s, found, reference_packet)
            return True

        while s.context.can_observe():
            if self._decisive(s):
                s.trace["controller_stop_reason"] = "evidence_complete"
                return
            live = [b for b in s.bundles if not self._retired(self._answer(b))]
            for b in live:
                p = self._answer(b)
                if b.conflicts and "conflicting_facts" not in p.unresolved:
                    p.unresolved.append("conflicting_facts")
            urgent = [
                (b, self._urgent(self._answer(b)) or ("conflict" if b.conflicts else None))
                for b in live
            ]
            if any(reason and self._recheck(s, self._answer(b), reason) for b, reason in urgent):
                continue
            if any(
                self._query_binding_recheck(s, self._answer(b))
                for b in live
                if not self._urgent(self._answer(b)) and not b.conflicts
            ):
                continue
            lost = not live or any(
                self._retired(self._answer(b)) and not self._answer(b).merged_into
                for b in s.bundles
            )
            if lost and relocate("target_not_located"):
                continue
            compared = False
            for left, right in combinations(live, 2):
                a, b = self._answer(left), self._answer(right)
                if (
                    intersection(a.span, b.span)
                    or set(self._comparison_sources(a)) & set(self._comparison_sources(b))
                ) and self._compare(s, left, right):
                    compared = True
                    break
            if compared:
                continue
            if any(
                self._detail(s, b, self._answer(b))
                for b in live
                if not audit_bundle(b, s.query).sufficient
            ):
                continue
            unresolved_target = any(
                self._answer(b).anchor_match != "matched" and not self._answer(b).active_errors
                for b in live
            )
            temporal = any(
                g["kind"] == "temporal_selection" for b in live for g in self._answer(b).active_gaps
            )
            if (unresolved_target or temporal) and relocate("evidence_unresolved"):
                continue
            s.trace["controller_stop_reason"] = (
                "refinement_limit"
                if s.context.refinements >= self.config.max_refinements
                else "no_progress"
            )
            break
        else:
            s.trace["controller_stop_reason"] = "model_call_budget"
        if (
            s.query.coverage == "existence"
            and not s.query_scope
            and s.context.can_observe()
            and not any(audit_bundle(b, s.query).sufficient for b in s.bundles)
        ):
            self._inspect_candidates(
                s, [SearchCandidate("existence_scan", scope)], reference_packet
            )

    def _finish(self, s):
        all_bundles = s.bundles
        s.trace["all_candidate_bundles"] = [b.to_dict() for b in all_bundles]
        s.trace["observation_records"] = [
            asdict(r)
            for b in all_bundles
            for p in b.packets
            if isinstance(p, V3Packet) and not p.merged_into
            for r in p.observations
        ]
        s.trace["evidence_state"] = [
            {
                "packet_id": p.packet_id,
                "merged_into": p.merged_into,
                "aliases": p.aliases,
                "bound_fact_ids": p.bound_fact_ids,
                "target_record_ids": p.target_record_ids,
                "active_errors": p.active_errors,
                "active_gaps": p.active_gaps,
                "resolutions": p.resolutions,
                "fact_eligibility": [asdict(f) for f in p.fact_eligibility],
                "query_binding_tasks": [asdict(t) for t in p.query_binding_tasks],
            }
            for b in all_bundles
            for p in b.packets
            if isinstance(p, V3Packet)
        ]
        live = [b for b in all_bundles if not self._retired(self._answer(b))]
        # Retired/background facts remain in the trace, never in a frozen answer bundle.
        s.bundles = live
        try:
            result = R1VideoAgent._finish(self, s)
        finally:
            s.bundles = all_bundles
        if result.support_level == "partial" and not result.unresolved_reasons:
            # Defensive diagnostic; normal paths supply the specific unit/field/terminal failure.
            result.unresolved_reasons.append(
                "evidence_unresolved:"
                + s.trace.get("controller_stop_reason", "terminal_not_supported")
            )
        return result

    @staticmethod
    def _missing_modalities(s, bundle):
        states = modality_state(s, bundle)
        s.trace["modality_state"] = states
        return [
            f"required_modality_unavailable:{m}"
            for m in s.query.required_modalities
            if not states[m]["allowed"] or not states[m]["input_present"]
        ]

    def _terminal_response(self, s, payload, batch, parser):
        from qwen3vl_agent.r1_v3.terminal import terminal_response

        bundle = next(b for b in s.bundles if b.bundle_id == payload["frozen_bundle"]["bundle_id"])
        unavailable = self._missing_modalities(s, bundle)
        states = s.trace["modality_state"]
        missing_evidence = [
            f"required_modality_evidence_unresolved:{m}"
            for m in s.query.required_modalities
            if states[m]["allowed"]
            and states[m]["input_present"]
            and not states[m]["answer_fact_ids"]
        ]
        # Availability is not evidence eligibility. Preserve that independent terminal gate
        # while using evidence_unresolved, rather than modality_unavailable, for failed readings.
        if unavailable or missing_evidence:
            payload = copy.deepcopy(payload)
            payload["evidence_audit"]["sufficient"] = False
            payload["evidence_audit"]["unresolved"].extend([*unavailable, *missing_evidence])
            s.context.issues.extend([*unavailable, *missing_evidence])
        s.trace["terminal_evidence_audit"] = copy.deepcopy(payload["evidence_audit"])
        return terminal_response(self, s, payload, batch, parser)
