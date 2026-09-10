"""Append-only observations, alternative identity mappings and versioned derivations."""

from __future__ import annotations

import copy
import itertools
import math

from .media import source_point
from .types import ProtocolError


class StateStore:
    def __init__(self, data=None):
        data = copy.deepcopy(data or {})
        self.entities = data.get("entities", {})
        self.observations = data.get("observations", [])
        self.superseded_observations = data.get("superseded_observations", [])
        self.associations = data.get("associations", [])
        self.containments = data.get("containments", [])
        self.anchor_states = data.get("anchor_states", [])
        self.derived = data.get("derived", [])
        self.gaps = data.get("gaps", [])
        self.history = data.get("history", [])
        self.windows = data.get("windows", {})
        self.revision = data.get("revision", 0)

    def to_dict(self):
        return copy.deepcopy(self.__dict__)

    def ingest(self, window_id, value, frames, query, coverage, call_id):
        if window_id in self.windows:
            return  # A resumed committed observation is not ingested twice.
        slots = {s["id"]: s for s in query["slots"]}
        targets = {e["id"]: e["target_id"] for e in value["entities"]}
        for record in value["records"]:
            if slots[record["slot_id"]]["target_id"] != targets[record["entity_id"]]:
                raise ProtocolError("entity does not satisfy the observation slot's target binding")
        active = {g["id"] for g in self.associations if not g.get("retired")}
        superseded = {key for g in value["associations"] for key in g["supersedes"]}
        if superseded - active:
            raise ProtocolError("cannot supersede an unknown/inactive association")
        corrections = value.get("superseded_observation_ids", [])
        old_records = {r["id"]: r for r in self.records()}
        for oid in corrections:
            old = old_records.get(oid)
            replacements = (
                []
                if old is None
                else [
                    r
                    for r in value["records"]
                    if frames[r["frame_id"]]["source_frame_id"] == old["source_frame_id"]
                    and r["slot_id"] == old["slot_id"]
                    and r["basis"] == "visual_observation"
                    and r["visibility"] in {"visible", "absent"}
                    and any(
                        all(
                            any(
                                link["from_node"] == old["entity_node"]
                                and link["to_entity"] == r["entity_id"]
                                and link["kind"] == "same_entity"
                                for link in alternative["links"]
                            )
                            for alternative in group["alternatives"]
                        )
                        and not group["unresolved_extra"]
                        for group in value["associations"]
                    )
                ]
            )
            if not replacements:
                raise ProtocolError(
                    "correction requires the same source frame, slot and unambiguous physical instance"
                )
        node = lambda local: f"{window_id}/{local}"
        self.revision += 1
        for entity in value["entities"]:
            nid = node(entity["id"])
            self.entities[nid] = {
                **entity,
                "node_id": nid,
                "persistent_id": f"entity_{len(self.entities) + 1:05d}",
                "window_id": window_id,
                "part_of": entity.get("part_of", ""),
            }
        added = []
        for i, record in enumerate(value["records"]):
            meta = frames[record["frame_id"]]
            oid = f"{window_id}/O{i + 1:03d}"
            entry = {
                **copy.deepcopy(record),
                "id": oid,
                "entity_node": node(record["entity_id"]),
                "frame_id": meta["id"],
                "source_frame_id": meta["source_frame_id"],
                "timestamp": meta["timestamp_seconds"],
                "window_id": window_id,
                "call_id": call_id,
                "slot": copy.deepcopy(slots[record["slot_id"]]),
                "reference_status": value["reference_status"],
                "source_size": meta["source_size"],
                "revision": self.revision,
            }
            for name in ("point", "reference_point"):
                if name in record:
                    entry["source_" + name] = source_point(record[name], meta)
            if "scale" in record:
                box = meta["view_box"]
                entry["source_scale"] = (
                    record["scale"] * max(box[2] - box[0], box[3] - box[1]) / 1000
                )
            self.observations.append(entry)
            added.append(oid)
        for g in self.associations:
            if g["id"] in superseded:
                g["retired"] = True
        for group in value["associations"]:
            alternatives = []
            for alternative in group["alternatives"]:
                links = [
                    {"from": link["from_node"], "to": node(link["to_entity"]), "kind": link["kind"]}
                    for link in alternative["links"]
                ]
                alternatives.append(
                    {
                        "links": links,
                        "evidence_frames": [
                            frames[f]["id"] for f in alternative["evidence_frames"]
                        ],
                    }
                )
            self.associations.append(
                {
                    **copy.deepcopy(group),
                    "id": f"{window_id}/{group['group_id']}",
                    "alternatives": alternatives,
                    "retired": False,
                    "revision": self.revision,
                    "window_id": window_id,
                }
            )
        for i, containment in enumerate(value["containments"]):
            meta = frames[containment["frame_id"]]
            self.containments.append(
                {
                    **containment,
                    "id": f"{window_id}/C{i + 1:03d}",
                    "carrier_node": node(containment["carrier_entity_id"]),
                    "frame_id": meta["id"],
                    "timestamp": meta["timestamp_seconds"],
                    "window_id": window_id,
                }
            )
        for anchor in value.get("anchor_states", []):
            meta = frames[anchor["frame_id"]]
            self.anchor_states.append(
                {
                    **anchor,
                    "frame_id": meta["id"],
                    "timestamp": meta["timestamp_seconds"],
                    "window_id": window_id,
                }
            )
        for gap in value["gaps"]:
            self.add_gap({**gap, "window_id": window_id, "source": gap.get("source", "observer")})
        for derived in self.derived:
            if derived.get("valid", True):
                derived["valid"] = False
        self.superseded_observations.extend(corrections)
        self.windows[window_id] = {
            "coverage": coverage,
            "complete": value["complete"],
            "candidate_coverage_complete": value["candidate_coverage_complete"],
            "observation_ids": added,
        }
        self.history.append(
            {
                "event": "observation_committed",
                "window_id": window_id,
                "revision": self.revision,
                "superseded": sorted(superseded),
                "superseded_observation_ids": list(corrections),
                "invalidated_derived": True,
            }
        )

    def add_gap(self, gap):
        value = {**gap, "id": f"gap_{len(self.gaps) + 1:05d}", "resolved": False}
        self.gaps.append(value)
        return value

    def close_gaps(self, ids, window_id):
        for gap in self.gaps:
            if gap["id"] in ids:
                gap["resolved"] = True
                gap["resolved_by"] = window_id
                self.history.append(
                    {"event": "gap_reobserved", "gap_id": gap["id"], "window_id": window_id}
                )

    def records(self, slot_ids=None):
        items = [
            r
            for r in self.observations
            if self.windows.get(r["window_id"], {}).get("query_relevant", True)
            and r["id"] not in self.superseded_observations
            and (slot_ids is None or r["slot_id"] in slot_ids)
        ]
        return sorted(items, key=lambda r: (r["timestamp"], r["id"]))

    def observation_conflicts(self, records, tolerance, limit=8):
        hypotheses, overflow, _ = self.hypotheses({r["entity_node"] for r in records}, limit)
        if overflow or not hypotheses:
            return []  # The operation's identity checks retain this unresolved branch.
        by_frame = {}
        for r in records:
            by_frame.setdefault((r["source_frame_id"], r["slot_id"]), []).append(r)
        conflicts = []
        for rows in by_frame.values():
            for a, b in itertools.combinations(rows, 2):
                if not all(h[a["entity_node"]] == h[b["entity_node"]] for h in hypotheses):
                    continue
                differs = any(
                    a.get(k) is not None and b.get(k) is not None and a[k] != b[k]
                    for k in ("value", "phase", "rank", "visibility")
                )
                if "source_point" in a and "source_point" in b:
                    differs |= math.dist(
                        a["source_point"], b["source_point"]
                    ) > tolerance * math.hypot(*a["source_size"])
                if differs:
                    conflicts.append(
                        {
                            "kind": "conflict",
                            "description": "Unreconciled observations disagree on the same physical frame/slot",
                            "span": [max(0, a["timestamp"] - 0.25), a["timestamp"] + 0.25],
                            "slot_id": a["slot_id"],
                            "evidence_ids": [a["id"], b["id"]],
                        }
                    )
        return conflicts

    def hypotheses(self, nodes, limit=8, *, allow_role=False, preserve_relation=False):
        groups = [g for g in self.associations if not g.get("retired")]
        kinds = {"same_entity", "same_role"} if allow_role else {"same_entity"}
        if preserve_relation:
            groups = [g for g in groups if g["relation_preserved"]]
        reachable = set(nodes)
        while True:
            before = set(reachable)
            for g in groups:
                for a in g["alternatives"]:
                    for link in a["links"]:
                        if link["kind"] in kinds and {link["from"], link["to"]} & reachable:
                            reachable.update((link["from"], link["to"]))
            if before == reachable:
                break
        groups = [
            g
            for g in groups
            if any(
                l["kind"] in kinds and l["from"] in reachable
                for a in g["alternatives"]
                for l in a["links"]
            )
        ]
        overflow = any(g["unresolved_extra"] for g in groups)
        combinations = math.prod(len(g["alternatives"]) for g in groups)
        overflow |= combinations > limit
        result = []
        for chosen in itertools.islice(
            itertools.product(*(g["alternatives"] for g in groups)), limit
        ):
            parent = {n: n for n in self.entities}

            def root(n, parent=parent):
                while parent[n] != n:
                    n = parent[n]
                return n

            for alternative in chosen:
                for link in alternative["links"]:
                    if link["kind"] == "same_entity":
                        a, b = root(link["from"]), root(link["to"])
                        parent[b] = a
            mapping = {n: root(n) for n in parent}
            # A physical identity cannot denote two distinct local entities in one view.
            grouped = [(self.entities[n]["window_id"], mapping[n]) for n in reachable]
            if len(set(grouped)) != len(grouped):
                continue
            if allow_role:
                # Corresponding roles can occur together in a view without being the
                # same physical object. Apply role equivalence after physical bijection checks.
                for alternative in chosen:
                    for link in alternative["links"]:
                        if link["kind"] == "same_role":
                            parent[root(link["to"])] = root(link["from"])
                mapping = {n: root(n) for n in parent}
            result.append(mapping)
        return result, overflow, [g["id"] for g in groups]

    def identity_domain(self, node, limit=8):
        hypotheses, overflow, dependencies = self.hypotheses({node}, limit)
        identities = sorted({self.entities[h[node]]["persistent_id"] for h in hypotheses})
        return {
            "possible_persistent_ids": identities,
            "resolved": len(identities) == 1 and not overflow,
            "unexpanded_ambiguity": overflow,
            "dependencies": dependencies,
        }

    def bindable(self, records, limit=8, allow_role=False):
        nodes = {r["entity_node"] for r in records}
        hypotheses, overflow, _ = self.hypotheses(nodes, limit, allow_role=allow_role)
        return (
            bool(hypotheses)
            and not overflow
            and all(len({h[n] for n in nodes}) <= 1 for h in hypotheses)
        )

    def save_derived(self, result):
        self.derived.append(
            {
                "revision": self.revision,
                "valid": True,
                "result": copy.deepcopy(result),
                "dependencies": [r["id"] for r in self.observations]
                + [g["id"] for g in self.associations if not g.get("retired")],
            }
        )

    def handoff(self, frame_ids, limit=16):
        matching = [r for r in self.records() if r["source_frame_id"] in frame_ids]
        nodes = list(dict.fromkeys(r["entity_node"] for r in matching))[-limit:]
        # A reveal anchor remains useful even if it is several windows away.
        nodes += [
            c["carrier_node"] for c in self.containments[-4:] if c["carrier_node"] not in nodes
        ]
        return {
            "entities": [self.entities[n] for n in nodes],
            "previous_observations": [r for r in matching if r["entity_node"] in nodes][-32:],
            "association_groups": [g for g in self.associations if not g.get("retired")][-12:],
            "containment_anchors": self.containments[-4:],
        }
