"""Append-only factual versions and isolated, dependency-aware hypothesis worlds."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, field

from .types import ProtocolError, digest, plain


@dataclass
class Cell:
    key: str
    value: object = None
    kind: str = "unknown"
    sources: list[str] = field(default_factory=list)
    dependencies: list[str] = field(default_factory=list)
    valid: bool = True
    grounded: bool = False
    unit: str = ""
    time: float | None = None
    complete: bool = False
    reason: str = ""

    @property
    def known(self):
        return self.valid and self.value is not None and self.kind != "unknown"

    def to_dict(self):
        return plain(asdict(self))


class FactStore:
    def __init__(self, state=None):
        state = deepcopy(state or {})
        self.versions = state.get(
            "versions", [{"version": 0, "cells": {}, "semantic_hash": digest({})}]
        )
        self.entities = state.get("entities", [])
        self.observations = state.get("observations", [])

    @property
    def current(self):
        return deepcopy(self.versions[-1]["cells"])

    @property
    def version(self):
        return self.versions[-1]["version"]

    def ingest(self, observation, evidence, call_id):
        cells = self.current
        entities = {x["id"]: x for x in self.entities}
        entities.update({x["id"]: deepcopy(x) for x in observation["entities"]})
        self.entities = list(entities.values())
        for item in observation["facts"]:
            sources = sorted({evidence[e]["id"] for e in item["evidence_ids"]})
            key = item["key"]
            old = Cell(**cells[key]) if key in cells else None
            cell = Cell(
                key,
                deepcopy(item["value"]),
                item["kind"],
                sources,
                grounded=item["kind"] == "observed",
                unit=item["unit"],
                time=item["time"],
                complete=item["complete"],
            )
            if cell.kind == "reported_claim":
                cell.grounded = False
            if old and old.known and cell.kind == "unknown":
                continue
            is_events = lambda v: (
                isinstance(v, list)
                and all(
                    isinstance(e, dict)
                    and {"entity_id", "predicate", "start", "end", "occurrence_id"} <= e.keys()
                    for e in v
                )
            )
            if old and old.known and cell.known and is_events(old.value) and is_events(cell.value):
                # Preserve event windows for the deduplicating COUNT_EVENTS operator.
                merged = {digest(e): e for e in [*old.value, *cell.value]}
                cell.value = list(merged.values())
                cell.time = None
                cell.complete = old.complete and cell.complete
                cell.sources = sorted(set(old.sources + cell.sources))
                old = None
            if old and old.time is not None and cell.time is not None and cell.time < old.time:
                # Historical facts must have their own time-bound key, not replace latest state.
                key = f"{key}@{cell.time:g}"
                cell.key = key
                old = Cell(**cells[key]) if key in cells else None
            if old and old.known and cell.known and old.time == cell.time:
                if old.value == cell.value and old.unit == cell.unit:
                    cell.sources = sorted(set(old.sources + cell.sources))
                    cell.complete = old.complete or cell.complete
                else:
                    cell.value, cell.kind, cell.grounded = None, "unknown", False
                    cell.sources = sorted(set(old.sources + cell.sources))
                    cell.reason = "conflicting_measurements"
            cells[key] = cell.to_dict()
        semantic = {k: {p: v for p, v in c.items() if p != "sources"} for k, c in cells.items()}
        semantic_hash = digest(semantic)
        progress = semantic_hash != self.versions[-1]["semantic_hash"]
        self.versions.append(
            {"version": self.version + 1, "cells": cells, "semantic_hash": semantic_hash}
        )
        self.observations.append(
            {
                "call_id": call_id,
                "value": deepcopy(observation),
                "evidence": deepcopy(evidence),
                "progress": progress,
            }
        )
        return progress

    def to_dict(self):
        return deepcopy(
            {
                "versions": self.versions,
                "entities": self.entities,
                "observations": self.observations,
            }
        )


class World:
    def __init__(self, identifier, fact_version, cells, invariants=()):
        self.id, self.fact_version = identifier, fact_version
        self.cells = {
            k: Cell(**deepcopy(v)) if isinstance(v, dict) else deepcopy(v) for k, v in cells.items()
        }
        self.original = deepcopy(self.cells)
        self.invariants = set(invariants)
        self.transactions = []
        self.invalidated = []
        self.execution = []
        self.issues = []
        self.intervention_keys = set()
        self.intervention_time = None

    def get(self, key):
        return deepcopy(self.cells.get(key, Cell(key, reason="missing_variable")))

    def fork(self, identifier):
        other = World(identifier, self.fact_version, self.cells, self.invariants)
        other.original = deepcopy(self.cells)
        other.intervention_time = self.intervention_time
        return other

    def invalidate(self, changed):
        pending, affected = set(changed), set()
        while pending:
            parents = pending
            pending = set()
            for key, cell in self.cells.items():
                if key not in changed and key not in affected and set(cell.dependencies) & parents:
                    affected.add(key)
                    pending.add(key)
                    cell.valid = False
                    cell.reason = "parent_changed"
        self.invalidated.extend(sorted(affected))
        return affected

    def write(self, cell):
        old = self.cells.get(cell.key)
        if cell.key in self.intervention_keys and old and old.value != cell.value:
            raise ProtocolError(f"program overwrites direct intervention: {cell.key}")
        if cell.key in self.invariants and old and old.value != cell.value:
            raise ProtocolError(f"write violates invariant: {cell.key}")
        if old and (old.value != cell.value or old.valid != cell.valid):
            self.invalidate({cell.key})
        self.cells[cell.key] = deepcopy(cell)

    def transact(self, interventions, *, allowed_sources=None):
        """Evaluate every simultaneous RHS before commit; explicit sequential steps are barriers."""
        allowed_sources = allowed_sources or {}
        group = []
        for item in interventions:
            if item["sequential"]:
                if group:
                    self._commit(group, allowed_sources)
                    group = []
                self._commit([item], allowed_sources)
            else:
                group.append(item)
        if group:
            self._commit(group, allowed_sources)

    def _commit(self, items, allowed_sources):
        before = deepcopy(self.cells)
        writes = {}
        for item in items:
            reference = before if item["read_world"] == "current" else self.original
            target, op, value = item["target"], item["op"], deepcopy(item["value"])
            source = "I:" + item["id"]
            source_ids = [source]
            dependencies = []
            grounded = True
            if op == "swap":
                if not isinstance(value, str):
                    raise ProtocolError("swap value must name the second key")
                pairs = [
                    (target, reference.get(value, Cell(value))),
                    (value, reference.get(target, Cell(target))),
                ]
            else:
                if isinstance(value, dict) and set(value) == {"ref"}:
                    rhs = reference.get(value["ref"], Cell(value["ref"]))
                    value, grounded = deepcopy(rhs.value) if rhs.known else None, rhs.grounded
                    dependencies = (
                        ["factual:" + rhs.key] if reference is self.original else [rhs.key]
                    )
                    source_ids += rhs.sources
                if op == "remove":
                    if not target.endswith(".exists"):
                        raise ProtocolError("remove targets ENTITY.exists")
                    value = False
                elif op == "delay":
                    from .operators import bounds

                    if bounds(value) is None or bounds(value)[0] < 0:
                        raise ProtocolError("delay requires nonnegative seconds")
                    if not target.endswith(".delay"):
                        raise ProtocolError(
                            "delay targets ENTITY.delay; other actors keep their clock"
                        )
                elif op == "replace_event":
                    if value is not None and not isinstance(value, (dict, list)):
                        raise ProtocolError("replace_event requires structured event data")
                elif op == "continue_trend":
                    # Stipulates method/interval; the trend executor computes its consequence.
                    if not isinstance(value, dict):
                        raise ProtocolError("continue_trend requires a trend specification")
                pairs = [
                    (
                        target,
                        Cell(
                            target,
                            value,
                            "stipulated" if value is not None else "unknown",
                            source_ids,
                            dependencies,
                            grounded=grounded,
                        ),
                    )
                ]
            for key, cell in pairs:
                if key in writes:
                    raise ProtocolError("simultaneous transaction writes a key twice")
                if key in self.invariants:
                    raise ProtocolError(f"intervention violates invariant: {key}")
                copied = deepcopy(cell)
                copied.key = key
                copied.kind = "stipulated" if copied.known else "unknown"
                copied.sources = sorted(set(copied.sources + [source]))
                # Snapshot dependencies must not point back into this mutable world.
                if op == "swap":
                    copied.dependencies = ["factual:" + cell.key]
                writes[key] = copied
        changed = {k for k, c in writes.items() if k not in before or before[k].value != c.value}
        invalid = self.invalidate(changed)
        # Unmodelled observed post-intervention effects cannot be inherited as valid outcomes.
        if changed:
            changed_entities = {k.rsplit(".", 1)[0] for k in changed}
            for key, cell in self.cells.items():
                if (
                    cell.kind in {"observed", "reported_claim"}
                    and key not in writes
                    and key not in self.invariants
                ):
                    suffix = key.rsplit(".", 1)[-1]
                    if suffix in {
                        "collision",
                        "collisions",
                        "outcome",
                        "future",
                        "trajectory",
                        "events",
                        "position_after",
                    } or (suffix == "supported" and changed_entities):
                        cell.valid = False
                        cell.reason = "observed_effect_requires_counterfactual_recomputation"
                        invalid.add(key)
        self.cells.update(writes)
        self.intervention_keys.update(writes)
        times = [i.get("at_time") if i.get("at_time") is not None else 0.0 for i in items]
        self.intervention_time = min(
            times + ([self.intervention_time] if self.intervention_time is not None else [])
        )
        self.transactions.append(
            {
                "intervention_ids": [i["id"] for i in items],
                "read_snapshot": digest({k: c.to_dict() for k, c in before.items()}),
                "writes": {k: c.to_dict() for k, c in writes.items()},
                "invalidated": sorted(invalid),
            }
        )

    def to_dict(self):
        return {
            "id": self.id,
            "fact_version": self.fact_version,
            "cells": {k: c.to_dict() for k, c in self.cells.items()},
            "transactions": deepcopy(self.transactions),
            "invalidated": sorted(set(self.invalidated)),
            "execution": deepcopy(self.execution),
            "issues": deepcopy(self.issues),
        }
