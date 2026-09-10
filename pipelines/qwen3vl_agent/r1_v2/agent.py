"""R1 V2 changes search, while reusing R1's evidence and final-answer contracts."""

import hashlib
import re
from dataclasses import asdict, dataclass, field, replace

from qwen3vl_agent.r1.agent import R1VideoAgent
from qwen3vl_agent.r1.control import ProtocolError, audit_bundle, intersection, strings
from qwen3vl_agent.r1.media import MediaBatch
from qwen3vl_agent.r1.types import SearchCandidate, SearchState
from qwen3vl_agent.r1_v2.config import R1V2Config
from qwen3vl_agent.r1_v2.media import R1V2Media, index_report, navigation_frames
from qwen3vl_agent.r1_v2.runtime import R1V2ModelSession


@dataclass
class V2SearchState(SearchState):
    candidate_nodes: dict[str, str] = field(default_factory=dict)
    attempted_windows: list[dict] = field(default_factory=list)
    retired_candidates: dict[str, str] = field(default_factory=dict)
    rounds: list[dict] = field(default_factory=list)
    stop_events: list[dict] = field(default_factory=list)


class R1V2VideoAgent(R1VideoAgent):
    config_type = R1V2Config
    media_type = R1V2Media
    session_type = R1V2ModelSession
    policy_id = "r1-local-evidence/2.0"
    result_namespace = "r1_v2"

    def solve(self, request):
        result = super().solve(request)
        # Input failures return before the shared controller creates its normal trace.
        result.trace.setdefault("policy_id", self.policy_id)
        return result

    def _ensure_index(self, s):
        index = super()._ensure_index(s)
        if "navigation_index" not in s.trace:
            s.trace["navigation_index"] = index_report(index)
        return index

    def _locate(self, s, scope, *, reference=False, relocation=False, reference_packet=None):
        index = self._ensure_index(s)
        key = "reference" if reference else "answer"
        search = s.searches.setdefault(key, V2SearchState())
        root = index.node(index.root_id)
        queue = list(search.frontier) if relocation else list(root.child_ids or (root.node_id,))
        search.frontier.clear()
        text = (s.request.question + " " + s.query.semantic_hint).casefold()
        reverse = bool(re.search(r"at the end|near the end|ending|结尾|片尾", text))

        def available(node_id):
            window = intersection(index.node(node_id).span, scope)
            return [
                f
                for f in index.cached_video.frames
                if window and window.contains_evidence(f.timestamp_seconds, tolerance=1e-6)
            ]

        def live(node_id):
            shown = set(search.navigation_frame_ids)
            return any(f.id not in shown for f in available(node_id))

        def order(nodes):
            checked = set(search.checked_nodes)
            return sorted(
                dict.fromkeys(nodes),
                key=lambda n: (
                    n in checked,
                    (-1 if reverse else 1) * index.node(n).span.start_seconds,
                ),
            )

        queue = order(queue)
        candidates = []
        stop_reason = "navigation_frames_exhausted"
        for _ in range(self.config.media.locator_max_levels):
            queue = [n for n in dict.fromkeys(queue) if live(n)]
            if not queue:
                break
            if not s.context.can_observe():
                stop_reason = "model_call_budget"
                break
            selected, remainder = (
                queue[: self.config.media.locator_branch_factor],
                queue[self.config.media.locator_branch_factor :],
            )
            descriptions, frames, node_frames = [], [], {}
            for node_id in selected:
                reps = navigation_frames(available(node_id), set(search.navigation_frame_ids))
                node_frames[node_id] = {f.id for f in reps}
                frames.extend(reps)
                descriptions.append(
                    {
                        "node_id": node_id,
                        "span": asdict(intersection(index.node(node_id).span, scope)),
                        "frames": [f.to_dict() for f in reps],
                    }
                )
            frames = list({f.id: f for f in frames}.values())
            payload = {
                "question": s.request.question,
                "search_role": key,
                "reference_description": s.query.reference_description if reference else "",
                "semantic_hint": s.query.semantic_hint,
                "allowed_scope": asdict(scope),
                "nodes": descriptions,
                "reference_facts": [asdict(f) for f in reference_packet.facts]
                if reference_packet
                else [],
                "text_hints": [
                    asdict(t)
                    for t in s.provider_hints
                    if scope.start_seconds <= t.start_sec <= t.end_sec <= scope.end_seconds
                ],
            }

            def parse(data, _limited, node_frames=node_frames):
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
                    result.append(
                        SearchCandidate(
                            node_id,
                            intersection(index.node(node_id).span, scope),
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

            result = s.session.call(
                "locator", payload, batch=MediaBatch(scope, tuple(frames)), parser=parse
            ).value
            search.checked_nodes.extend(selected)
            search.navigation_frame_ids.extend(f.id for f in frames)
            search.rounds.append(
                {
                    "relocation": relocation,
                    "node_ids": selected,
                    "shown_frames": [
                        {"id": f.id, "timestamp_seconds": f.timestamp_seconds} for f in frames
                    ],
                    "selected_node_ids": [c.candidate_id for c in result],
                    "no_candidates": not result,
                }
            )
            chosen = {c.candidate_id for c in result}
            preferred, others = [], list(remainder)
            for node_id in selected:
                children = index.node(node_id).child_ids
                if children:
                    (preferred if node_id in chosen else others).extend(children)
                elif live(node_id):
                    # An observed local candidate never exhausts this node's other navigation frames.
                    others.append(node_id)
            queue = list(dict.fromkeys(preferred + order(others)))
            for candidate in result:
                node_id = candidate.candidate_id
                if index.node(node_id).child_ids:
                    continue
                window = self._candidate_window(s, candidate, scope)
                identity = repr(
                    (
                        node_id,
                        sorted(candidate.anchor_frame_ids),
                        window.start_seconds,
                        window.end_seconds,
                    )
                )
                candidate_id = (
                    f"{key}-{node_id}-{hashlib.sha256(identity.encode()).hexdigest()[:12]}"
                )
                if candidate_id in search.candidate_nodes:
                    continue
                search.candidate_nodes[candidate_id] = node_id
                candidates.append(replace(candidate, candidate_id=candidate_id, span=window))
            if candidates:
                stop_reason = "candidate_found"
                break
            stop_reason = (
                "search_round_limit"
                if any(live(n) for n in queue)
                else "navigation_frames_exhausted"
            )
        search.frontier = [n for n in dict.fromkeys(queue) if live(n)]
        search.candidates.extend(candidates)
        search.stop_events.append(
            {
                "relocation": relocation,
                "reason": stop_reason,
                "no_candidates": not candidates,
                "remaining_node_ids": list(search.frontier),
            }
        )
        s.trace["searches"] = {k: asdict(v) for k, v in s.searches.items()}
        return candidates[: self.config.max_candidates]

    @staticmethod
    def _relocalization_reason(packet):
        if packet.anchor_match == "mismatched":
            return "anchor_mismatched"
        if (
            packet.anchor_match != "matched"
            and packet.target_binding != "confirmed"
            and not packet.facts
            and packet.review_request.get("kind") == "identity"
        ):
            return "target_unconfirmed_identity"
        return None

    def _record_attempt(self, s, packet):
        search = s.searches.get("answer")
        if not isinstance(search, V2SearchState):
            return
        reason = self._relocalization_reason(packet)
        retired = bool(
            not s.query_scope
            and reason
            and packet.anchor_match != "matched"
            and packet.target_binding != "confirmed"
            and not packet.facts
        )
        if retired:
            search.retired_candidates[packet.candidate_id] = reason
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
                "retired_to_search": retired,
                "reason": reason,
            }
        )

    def _inspect_candidates(self, s, candidates, reference):
        previous = len(s.bundles)
        super()._inspect_candidates(s, candidates, reference)
        for bundle in s.bundles[previous:]:
            self._record_attempt(s, next(p for p in bundle.packets if p.role == "answer"))

    def _refine(self, s, bundle, packet):
        super()._refine(s, bundle, packet)
        self._record_attempt(s, packet)

    @staticmethod
    def _competitors(s, chosen):
        competitors = R1VideoAgent._competitors(s, chosen)
        search = s.searches.get("answer")
        retired = set(search.retired_candidates) if isinstance(search, V2SearchState) else set()
        excluded = retired | {
            b.bundle_id
            for b in s.bundles
            if any(p.role == "answer" and p.candidate_id in retired for p in b.packets)
        }
        return [c for c in competitors if c not in excluded]

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
                or not s.context.can_observe()
                or s.context.relocations >= self.config.max_relocations
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

        def lost_target():
            return next(
                (
                    reason
                    for b in s.bundles
                    for p in b.packets
                    if p.role == "answer" and (reason := self._relocalization_reason(p))
                ),
                None,
            )

        if self._decisive(s):
            return
        if reason := lost_target():
            relocate(reason)
        if self._decisive(s):
            return
        while s.context.refinements < self.config.max_refinements and s.context.can_observe():
            search = s.searches.get("answer")
            retired = set(search.retired_candidates) if isinstance(search, V2SearchState) else set()
            unresolved = [
                b
                for b in s.bundles
                if (not audit_bundle(b, s.query).sufficient or self._competitors(s, b))
                and next(p for p in b.packets if p.role == "answer").anchor_match != "mismatched"
                and next(p for p in b.packets if p.role == "answer").candidate_id not in retired
            ]
            if not unresolved:
                break
            bundle = min(
                unresolved, key=lambda b: next(p.rechecks for p in b.packets if p.role == "answer")
            )
            packet = next(p for p in bundle.packets if p.role == "answer")
            self._refine(s, bundle, packet)
            if reason := self._relocalization_reason(packet):
                relocate(reason)
            if self._decisive(s):
                return
        relocate("no_candidate" if not s.bundles else "evidence_unresolved")
        if (
            s.query.coverage == "existence"
            and not s.query_scope
            and s.context.can_observe()
            and not any(audit_bundle(b, s.query).sufficient for b in s.bundles)
        ):
            self._inspect_candidates(
                s, [SearchCandidate("existence_scan", scope)], reference_packet
            )
