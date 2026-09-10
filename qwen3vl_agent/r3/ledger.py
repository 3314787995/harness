"""Append-only observations and conservative canonical event relations."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
from itertools import combinations
from typing import Any

from qwen3vl_agent.r3.planning import covered, overlaps
from qwen3vl_agent.r3.types import (
    Bracket,
    CoverageTile,
    EventQuery,
    EventRecord,
    EventRelation,
    ObservationRecord,
)


class EventLedger:
    def __init__(self, query: EventQuery, tiles: list[CoverageTile]) -> None:
        self.query, self.tiles = query, tiles
        self.observations: list[ObservationRecord] = []
        self.proposals: dict[str, EventRecord] = {}
        self.relations: dict[str, EventRelation] = {}
        self.relation_history: list[dict[str, Any]] = []
        self.sources: dict[str, dict[str, Any]] = {}
        self.conflicts: list[str] = []

    @staticmethod
    def pair_key(a: str, b: str) -> str:
        return "|".join(sorted((a, b)))

    def ingest(self, obs: ObservationRecord) -> None:
        if any(o.observation_id == obs.observation_id for o in self.observations):
            return
        self.validate_sources(obs.source_refs)
        # Construct everything before committing this batch.
        events = []
        for i, data in enumerate(obs.proposals):
            event = EventRecord.from_dict(data)
            event.event_id = f"{obs.observation_id}:e{i:03d}"
            event.member_ids = [event.event_id]
            event.observation_ids = [obs.observation_id]
            events.append(event)
        self.sources.update(deepcopy(obs.source_refs))
        self.observations.append(deepcopy(obs))
        self.proposals.update((e.event_id, e) for e in events)
        self._update_tile(obs)
        self.reconcile([e.event_id for e in events])

    def validate_sources(self, sources: dict[str, dict[str, Any]]) -> None:
        for key, value in sources.items():
            previous = self.sources.get(key)
            if previous and any(
                previous.get(f) != value.get(f)
                for f in (
                    "source_id",
                    "kind",
                    "start_sec",
                    "end_sec",
                    "text",
                    "crop_transform",
                    "alignment_status",
                    "alignment_error_sec",
                    "speaker_id",
                )
            ):
                raise ValueError("source ID changed identity across calls")

    def _update_tile(self, obs: ObservationRecord) -> None:
        tile = next(t for t in self.tiles if t.tile_id == obs.tile_id)
        tile.observation_ids.append(obs.observation_id)
        certificate = {"kind": obs.kind, "core": obs.core, "resolution_met": obs.resolution_met,
            "truncated": obs.truncated, "unresolved": obs.unresolved,
            "actual_max_gap_sec": obs.actual_max_gap_sec, "external_coverage": obs.external_coverage,
            "source_frame_ids": [k for k,v in obs.source_refs.items() if v["kind"] == "frame"],
            "target_assessments": obs.target_assessments,
            "two_stage": bool(obs.visual and len(obs.stage_call_ids) == 2)}
        tile.certificates.append(certificate)
        certs = tile.certificates
        valid = [tuple(c["core"]) for c in certs if c["resolution_met"] and not c["truncated"]]
        valid += [tuple(c["core"]) for c in tile.media_observations if c["resolution_met"]]
        tile.observed = covered(tile.core, valid)
        tile.resolution_met = tile.observed
        if obs.external_coverage == "not_required":
            tile.external_coverage = "not_required"
        else:
            text_spans = [tuple(c["core"]) for c in certs if c["external_coverage"] == "complete" and not c["truncated"]]
            tile.external_coverage = "complete" if covered(tile.core, text_spans) else "unknown"
        def clean(c):
            return c["two_stage"] and c["resolution_met"] and not c["truncated"] and not c["unresolved"]
        statuses = {}
        for target in self.query.targets:
            tid = target.target_id
            def assessment(c):
                return next((a["status"] for a in c["target_assessments"] if a["target_id"] == tid), "uncertain")
            last_bad = max((i for i,c in enumerate(certs) if not clean(c) or assessment(c) == "uncertain"), default=-1)
            usable = certs[last_bad+1:]
            positive = [tuple(c["core"]) for c in usable if assessment(c) == "observed"]
            base_negative = [tuple(c["core"]) for c in usable if c["kind"] != "audit" and assessment(c) == "absent"]
            audited_negative = [tuple(c["core"]) for c in usable if c["kind"] == "audit" and assessment(c) == "absent"
                                and covered(tuple(c["core"]), base_negative)]
            contradictory = any(overlaps(a,b) for a in positive for b in audited_negative)
            if contradictory:
                statuses[tid] = "uncertain"
            elif covered(tile.core, positive + audited_negative):
                statuses[tid] = "observed" if positive else "absent_confirmed"
            else:
                statuses[tid] = "uncertain"
        tile.target_coverage = statuses
        tile.audit_needed = any(v != "observed" for v in statuses.values())
        tile.audit_done = bool(statuses) and all(v in {"observed", "absent_confirmed"} for v in statuses.values())
        last_bad = max((i for i,c in enumerate(certs) if not clean(c)), default=-1)
        clean_spans = [tuple(c["core"]) for c in certs[last_bad+1:] if clean(c)]
        if covered(tile.core, clean_spans):
            tile.unresolved = []
        else:
            tile.unresolved = list(dict.fromkeys([*tile.unresolved, *obs.unresolved]))
        if obs.truncated:
            tile.unresolved = list(dict.fromkeys([*tile.unresolved, "output_truncated"]))

    def apply_relation(
        self,
        a: str,
        b: str,
        relation: str,
        evidence: list[str],
        reason: str,
        *,
        attempted: bool = False,
    ) -> None:
        if a == b or a not in self.proposals or b not in self.proposals:
            raise ValueError("invalid relation endpoints")
        key = self.pair_key(a, b)
        old = self.relations.get(key)
        attempts = (old.attempts if old else 0) + int(attempted)
        if old and old.relation in {"same_occurrence", "distinct_occurrences", "conflict"}:
            if relation not in {"unresolved", old.relation}:
                relation = "conflict"
                self.conflicts.append("contradictory_relation:" + key)
            elif relation == "unresolved":
                relation = old.relation
        self.relations[key] = EventRelation(a, b, relation, evidence, reason, attempts)
        if attempted:
            self.relation_history.append(asdict(self.relations[key]))

    def reconcile(self, new_ids: list[str] | None = None) -> None:
        targets = {t.target_id: t for t in self.query.targets}
        values = [e for e in self.proposals.values() if e.status != "rejected"]
        pairs = (
            combinations(values, 2)
            if new_ids is None
            else (
                (self.proposals[key], other)
                for key in new_ids
                for other in values
                if other.event_id != key and (other.event_id not in new_ids or other.event_id < key)
            )
        )
        for a, b in pairs:
            if (
                a.target_id != b.target_id
                or self.pair_key(a.event_id, b.event_id) in self.relations
            ):
                continue
            same_subject = bool(
                a.actor_ref
                and b.actor_ref
                and a.actor_ref == b.actor_ref
                and a.object_ref == b.object_ref
            )
            common_completion = set(a.completion_evidence_refs) & set(b.completion_evidence_refs)
            same_read_segment = a.fact_kind == "utterance" and a.evidence_refs == b.evidence_refs
            relation, reason, evidence = "unresolved", "needs_continuity_or_reset_evidence", []
            common_onset = set(a.start_evidence_refs) & set(b.start_evidence_refs)
            if same_subject and common_completion and common_onset or same_read_segment:
                relation, reason = "same_occurrence", "same_bound_completion_or_read_segment"
                evidence = sorted(common_completion) or list(a.evidence_refs)
            else:
                separate = (
                    a.offset_bracket.hi is not None
                    and b.onset_bracket.lo is not None
                    and a.offset_bracket.hi < b.onset_bracket.lo
                ) or (
                    b.offset_bracket.hi is not None
                    and a.onset_bracket.lo is not None
                    and b.offset_bracket.hi < a.onset_bracket.lo
                )
                same_window = bool(set(a.observation_ids) & set(b.observation_ids))
                independent_completions = (
                    same_window
                    and a.completion_evidence_refs
                    and b.completion_evidence_refs
                    and not common_completion
                    and (
                        a.reset_evidence_refs or b.reset_evidence_refs or a.actor_ref != b.actor_ref
                    )
                )
                if separate or independent_completions:
                    if targets[a.target_id].repeat_policy != "world" or (
                        a.replay_status == b.replay_status == "original"
                    ):
                        if separate:
                            # This relation can be recomputed from canonical brackets; do not
                            # materialize the quadratic set of unrelated, distant events.
                            continue
                        relation, reason = (
                            "distinct_occurrences",
                            "separate_observed_boundaries_or_reset",
                        )
                        evidence = list(dict.fromkeys(a.evidence_refs + b.evidence_refs))
                elif same_window and a.actor_ref and b.actor_ref and a.actor_ref != b.actor_ref:
                    relation, reason = (
                        "distinct_occurrences",
                        "separately_bound_simultaneous_actors",
                    )
                    evidence = list(dict.fromkeys(a.evidence_refs + b.evidence_refs))
            self.apply_relation(a.event_id, b.event_id, relation, evidence, reason)

    def _groups(self) -> tuple[dict[str, list[EventRecord]], list[str]]:
        parent = {key: key for key, value in self.proposals.items() if value.status != "rejected"}

        def root(key: str) -> str:
            while parent[key] != key:
                key = parent[key]
            return key

        conflicts = []
        distinct = [
            r for r in self.relations.values() if r.relation in {"distinct_occurrences", "conflict"}
        ]
        for r in self.relations.values():
            if (
                r.relation != "same_occurrence"
                or r.left_id not in parent
                or r.right_id not in parent
            ):
                continue
            left, right = root(r.left_id), root(r.right_id)
            if left == right:
                continue
            if any(
                {root(d.left_id), root(d.right_id)} == {left, right}
                for d in distinct
                if d.left_id in parent and d.right_id in parent
            ):
                conflicts.append("same_distinct_transitive_conflict:" + self.pair_key(left, right))
                continue
            parent[max(left, right)] = min(left, right)
        groups: dict[str, list[EventRecord]] = {}
        for key in parent:
            groups.setdefault(root(key), []).append(self.proposals[key])
        return groups, conflicts

    def canonical_events(self) -> list[EventRecord]:
        groups, conflicts = self._groups()
        self.conflicts = list(dict.fromkeys([*self.conflicts, *conflicts]))
        targets = {t.target_id: t for t in self.query.targets}
        result = []
        for key, rows in groups.items():
            rows = sorted(rows, key=lambda e: (e.visible_span[0], e.event_id))
            event = deepcopy(rows[0])
            event.unresolved_reasons = list(
                dict.fromkeys(reason for row in rows for reason in row.unresolved_reasons)
            )
            event.event_id = key
            event.member_ids = [r.event_id for r in rows]
            event.evidence_refs = list(dict.fromkeys(ref for r in rows for ref in r.evidence_refs))
            event.observation_ids = list(
                dict.fromkeys(ref for r in rows for ref in r.observation_ids)
            )
            for field in ("start_evidence_refs", "completion_evidence_refs", "reset_evidence_refs"):
                setattr(
                    event,
                    field,
                    list(dict.fromkeys(ref for r in rows for ref in getattr(r, field))),
                )
            boundary_rows = rows
            if targets[event.target_id].repeat_policy == "world":
                originals = [r for r in rows if r.replay_status == "original"]
                if originals:
                    first = min(originals, key=lambda r: r.visible_span[0])
                    boundary_rows = [
                        r for r in originals if r.visible_span[0] <= first.visible_span[1]
                    ]
                else:
                    event.unresolved_reasons.append("world_event_primary_time_unknown")
                    boundary_rows = []
            for field in ("onset_bracket", "offset_bracket"):
                los = [
                    getattr(r, field).lo for r in boundary_rows if getattr(r, field).lo is not None
                ]
                his = [
                    getattr(r, field).hi for r in boundary_rows if getattr(r, field).hi is not None
                ]
                try:
                    setattr(
                        event, field, Bracket(max(los) if los else None, min(his) if his else None)
                    )
                except ValueError:
                    event.unresolved_reasons.append("contradictory_event_boundaries")
                    setattr(event, field, Bracket())
            event.visible_span = (
                min(r.visible_span[0] for r in (boundary_rows or rows)),
                max(r.visible_span[1] for r in (boundary_rows or rows)),
            )
            event.completed = any(r.completed and r.completion_evidence_refs for r in rows)
            event.left_censored, event.right_censored = (
                event.onset_bracket.lo is None,
                event.offset_bracket.hi is None,
            )
            for attr in {attr for r in rows for attr in r.attributes}:
                values = [r.attributes[attr] for r in rows if attr in r.attributes]
                if all(v == values[0] for v in values):
                    event.attributes[attr] = values[0]
                else:
                    event.attributes.pop(attr, None)
                    event.unresolved_reasons.append("conflicting_attribute:" + attr)
            for name in {n for r in rows for n in r.cooccurrence}:
                statuses = {r.cooccurrence.get(name, "unknown") for r in rows}
                # Local absence in one part does not contradict presence elsewhere in
                # the same episode. A main episode contributes at most one presence.
                if "present" in statuses:
                    event.cooccurrence[name] = "present"
                elif (
                    statuses == {"absent"} and not event.left_censored and not event.right_censored
                ):
                    event.cooccurrence[name] = "absent"
                else:
                    event.cooccurrence[name] = "unknown"
            clear = any(r.status in {"accepted", "boundary_pending"} for r in rows)
            event.status = "accepted" if clear else "proposed"
            if event.unit_kind in {"action_cycle", "state_transition"} and not event.completed:
                event.status = "boundary_pending"
            event.unresolved_reasons = list(dict.fromkeys(event.unresolved_reasons))
            if event.unresolved_reasons:
                event.status = "proposed"
            anchor = (
                event.offset_bracket
                if targets[event.target_id].inclusion_rule == "completes_inside"
                else event.onset_bracket
            )
            for tile in self.tiles:
                if (
                    anchor.lo is not None
                    and anchor.hi is not None
                    and (tile.core[0] <= anchor.lo <= anchor.hi < tile.core[1])
                ):
                    event.owner_tile_id = tile.tile_id
                    break
            result.append(event)
        return result

    def pending_relations(self, events: list[EventRecord] | None = None) -> list[EventRelation]:
        events = events if events is not None else self.canonical_events()
        owner = {m: e for e in events for m in e.member_ids}
        definite = set()
        for relation in self.relations.values():
            if (
                relation.relation == "distinct_occurrences"
                and relation.left_id in owner
                and relation.right_id in owner
            ):
                definite.add(
                    self.pair_key(
                        owner[relation.left_id].event_id, owner[relation.right_id].event_id
                    )
                )
        pending: dict[str, EventRelation] = {}
        for relation in self.relations.values():
            if (
                relation.relation not in {"unresolved", "conflict"}
                or relation.left_id not in owner
                or relation.right_id not in owner
            ):
                continue
            a, b = owner[relation.left_id], owner[relation.right_id]
            if a.event_id == b.event_id:
                continue
            key = self.pair_key(a.event_id, b.event_id)
            if relation.relation != "conflict" and (
                key in definite or self.certainly_distinct(a, b)
            ):
                continue
            previous = pending.get(key)
            if (
                previous is None
                or relation.relation == "conflict"
                or relation.attempts > previous.attempts
            ):
                pending[key] = relation
        return list(pending.values())

    def certainly_distinct(self, a: EventRecord, b: EventRecord) -> bool:
        spec = next(t for t in self.query.targets if t.target_id == a.target_id)
        if spec.repeat_policy == "world" and not (a.replay_status == b.replay_status == "original"):
            return False
        return (
            a.offset_bracket.hi is not None
            and b.onset_bracket.lo is not None
            and a.offset_bracket.hi < b.onset_bracket.lo
        ) or (
            b.offset_bracket.hi is not None
            and a.onset_bracket.lo is not None
            and b.offset_bracket.hi < a.onset_bracket.lo
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "observations": [asdict(o) for o in self.observations],
            "proposals": [asdict(p) for p in self.proposals.values()],
            "events": [asdict(e) for e in self.canonical_events()],
            "relations": [asdict(r) for r in self.relations.values()],
            "relation_history": self.relation_history,
            "sources": self.sources,
            "conflicts": self.conflicts,
            "tiles": [asdict(t) for t in self.tiles],
        }

    @classmethod
    def restore(cls, query: EventQuery, data: dict[str, Any]) -> EventLedger:
        result = cls(
            query,
            [
                CoverageTile(**{**t, "core": tuple(t["core"]), "context": tuple(t["context"])})
                for t in data["tiles"]
            ],
        )
        result.observations = [ObservationRecord(**o) for o in data["observations"]]
        result.proposals = {e["event_id"]: EventRecord.from_dict(e) for e in data["proposals"]}
        result.relations = {
            result.pair_key(r["left_id"], r["right_id"]): EventRelation(**r)
            for r in data["relations"]
        }
        result.relation_history = data.get("relation_history", [])
        result.sources, result.conflicts = data["sources"], data["conflicts"]
        return result
