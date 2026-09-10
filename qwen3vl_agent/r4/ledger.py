"""Append-only observations and reversible, evidence-constrained identity components."""

from __future__ import annotations

import itertools
import unicodedata
from bisect import insort
from dataclasses import asdict
from typing import Any

from qwen3vl_agent.r4.types import InventorySpec, ProtocolError


def normalize(value: str, policy: dict[str, Any]) -> str:
    result = " ".join(unicodedata.normalize("NFC", value).split())
    if policy.get("casefold"):
        result = result.casefold()
    if policy.get("strip_punctuation"):
        result = " ".join(
            "".join(c for c in result if not unicodedata.category(c).startswith("P")).split()
        )
    aliases = policy.get("aliases", {})
    visited = set()
    while result in aliases:
        if result in visited:
            raise ProtocolError("normalization alias cycle")
        visited.add(result)
        result = aliases[result]
        if not isinstance(result, str):
            raise ProtocolError("alias target must be text")
    return result


def components(nodes: list[str], relations: list[dict[str, Any]]) -> dict[str, Any]:
    superseded = {key for r in relations for key in r.get("supersedes", [])}
    active = [r for r in relations if r["relation_id"] not in superseded]
    parent = {n: n for n in nodes}

    def root(n: str) -> str:
        while parent[n] != n:
            n = parent[n]
        return n

    def join(a: str, b: str) -> None:
        a, b = root(a), root(b)
        if a != b:
            parent[max(a, b)] = min(a, b)

    for relation in active:
        if relation["relation"] == "same":
            join(relation["left"], relation["right"])
    bad = {
        root(r["left"])
        for r in active
        if r["relation"] == "different" and root(r["left"]) == root(r["right"])
    }
    disputed_nodes = {n for n in nodes if root(n) in bad}
    conflicts = [
        r["relation_id"]
        for r in active
        if r["left"] in disputed_nodes and r["right"] in disputed_nodes
    ]
    # Quarantine both directions of a contradiction, then rebuild. No latest-wins union.
    parent = {n: n for n in nodes}
    accepted = [r for r in active if r["relation_id"] not in conflicts]
    for relation in accepted:
        if relation["relation"] == "same":
            join(relation["left"], relation["right"])
    groups: dict[str, list[str]] = {}
    for node in nodes:
        groups.setdefault(root(node), []).append(node)
    different = {
        tuple(sorted((root(r["left"]), root(r["right"]))))
        for r in accepted
        if r["relation"] == "different"
    }
    return {
        "groups": groups,
        "roots": {n: root(n) for n in nodes},
        "different": different,
        "conflicts": conflicts,
        "active": accepted,
    }


def clique_lower(nodes: list[str], different: set[tuple[str, str]]) -> int:
    neighbors = {n: set() for n in nodes}
    for a, b in different:
        if a in neighbors and b in neighbors:
            neighbors[a].add(b)
            neighbors[b].add(a)
    order = sorted(nodes, key=lambda n: (-len(neighbors[n]), n))
    if not nodes or not any(neighbors.values()):
        return int(bool(nodes))
    best = 0
    for first in order[:32]:
        clique = [first]
        for candidate in order:
            if candidate != first and all(candidate in neighbors[n] for n in clique):
                clique.append(candidate)
        best = max(best, len(clique))
        if best == len(nodes):
            break
    return best


class CanonicalInventory:
    def __init__(self, spec: InventorySpec, state: dict[str, Any] | None = None) -> None:
        self.spec = spec
        self.state = (
            state
            if state is not None
            else {
                "observations": {},
                "relations": [],
                "task_updates": {},
                "update_relations": [],
                "tracklets": [],
                "committed_calls": [],
                "pair_attempts": {},
                "issues": [],
            }
        )

    def commit(self, batch: dict[str, Any], call_id: str) -> None:
        if call_id in self.state["committed_calls"]:
            return
        local_map = {}
        for record in batch["observations"]:
            key = f"obs_{len(self.state['observations']):07d}"
            values = asdict(record)
            values["observation_id"] = key
            values["call_id"] = call_id
            local_map[record.local_id] = key
            self.state["observations"][key] = values
            if len({d["source_frame_id"] for d in record.detections}) > 1:
                self.state["tracklets"].append(
                    {
                        "tracklet_id": f"track_{len(self.state['tracklets']):07d}",
                        "observation_id": key,
                        "detections": record.detections,
                        "predicate_status": record.predicate_status,
                    }
                )
        for relation in batch["relations"]:
            self.add_relation(local_map[relation["left"]], local_map[relation["right"]], relation)
        update_map = {
            update["local_id"]: f"update_{len(self.state['task_updates']) + i:07d}"
            for i, update in enumerate(batch["updates"])
        }
        for update in batch["updates"]:
            values = dict(update)
            key = update_map[update["local_id"]]
            values["refers_to"] = update_map.get(values.get("refers_to"), values.get("refers_to"))
            values.update(update_id=key, call_id=call_id)
            self.state["task_updates"][key] = values
        self.state["committed_calls"].append(call_id)
        self._same_frame_constraints()

    def add_relation(
        self, left: str, right: str, value: dict[str, Any], *, updates: bool = False
    ) -> None:
        records = self.state["task_updates" if updates else "observations"]
        target = self.state["update_relations" if updates else "relations"]
        if left not in records or right not in records or left == right:
            raise ProtocolError("relation references unknown/dentical members")
        if value["relation"] not in {"same", "different", "unknown"}:
            raise ProtocolError("invalid relation")
        supplied = set(value.get("evidence_refs", []))
        if value["relation"] != "unknown" and any(
            not supplied.intersection(records[n]["evidence_refs"]) for n in (left, right)
        ):
            raise ProtocolError("relation must cite both observations")
        known = {r["relation_id"] for r in target}
        if set(value.get("supersedes", [])) - known:
            raise ProtocolError("relation supersedes an unknown decision")
        for old in target:
            if old["relation_id"] in value.get("supersedes", []) and (
                {old["left"], old["right"]} != {left, right} or value["relation"] == "unknown"
            ):
                raise ProtocolError("supersession requires a supported recheck of the same pair")
        target.append(
            {
                "relation_id": f"{'ur' if updates else 'rel'}_{len(target):07d}",
                "left": left,
                "right": right,
                "relation": value["relation"],
                "reason": value.get("reason", ""),
                "evidence_refs": list(supplied),
                "supersedes": value.get("supersedes", []),
            }
        )

    def _same_frame_constraints(self) -> None:
        specs = {s.set_id: s for s in self.spec.sets}
        frame_index: dict[str, list[Any]] = {}
        for key, record in self.state["observations"].items():
            if (
                specs[record["set_id"]].namespace != "physical_instance"
                or record["population_status"] != "included"
            ):
                continue
            for detection in record["detections"]:
                frame_index.setdefault(
                    record["entry_id"] + ":" + detection["source_frame_id"], []
                ).append((key, detection))
        known = {tuple(sorted((r["left"], r["right"]))) for r in self.state["relations"]}
        for entries in frame_index.values():
            for (a, da), (b, db) in itertools.combinations(entries, 2):
                pair = tuple(sorted((a, b)))
                if a == b or pair in known:
                    continue
                x, y = da["bbox"], db["bbox"]
                disjoint = x[2] <= y[0] or y[2] <= x[0] or x[3] <= y[1] or y[3] <= x[1]
                if disjoint:
                    self.add_relation(
                        a,
                        b,
                        {
                            "relation": "different",
                            "evidence_refs": [da["ref"], db["ref"]],
                            "reason": "distinct simultaneous source-frame detections",
                        },
                    )
                    known.add(pair)

    def graph(self, *, updates: bool = False) -> dict[str, Any]:
        if updates:
            return components(list(self.state["task_updates"]), self.state["update_relations"])
        specs = {s.set_id: s for s in self.spec.sets}
        nodes = [
            k
            for k, v in self.state["observations"].items()
            if specs[v["set_id"]].namespace == "physical_instance"
        ]
        relations = [
            r for r in self.state["relations"] if r["left"] in nodes and r["right"] in nodes
        ]
        return components(nodes, relations)

    def pending_pairs(
        self, *, updates: bool = False, max_rounds: int = 2, limit: int | None = None
    ) -> list[tuple[str, str]]:
        graph = self.graph(updates=updates)
        records = self.state["task_updates" if updates else "observations"]
        candidates = []
        representatives = sorted(graph["groups"])
        for a, b in itertools.combinations(representatives, 2):
            if tuple(sorted((a, b))) in graph["different"]:
                continue
            left, right = records[a], records[b]
            if updates:
                if any(
                    left.get(k) != right.get(k)
                    for k in ("owner", "task_id", "item_key", "kind", "unit")
                ):
                    continue
            elif (
                left["population_status"] == "excluded" or right["population_status"] == "excluded"
            ):
                continue
            key = ("update:" if updates else "identity:") + a + ":" + b
            if self.state["pair_attempts"].get(key, 0) >= max_rounds:
                continue
            priority = 0 if updates or left["category"] == right["category"] else 1
            if left["entry_id"] == right["entry_id"]:
                priority -= 1
            if limit is None:
                candidates.append((priority, a, b))
            else:
                insort(candidates, (priority, a, b))
                if len(candidates) > limit:
                    candidates.pop()
        return [(a, b) for _, a, b in sorted(candidates)[:limit]]

    def snapshot(self) -> dict[str, Any]:
        graph = self.graph()
        return {
            **self.state,
            "unresolved_identity_semantics": "Every unlinked, non-DIFFERENT component pair remains UNKNOWN.",
            "canonical_entities": [
                {"entity_id": k, "observation_ids": v} for k, v in graph["groups"].items()
            ],
            "identity_conflicts": graph["conflicts"],
        }
