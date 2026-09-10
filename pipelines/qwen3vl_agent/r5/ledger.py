"""Append-only facts and summary lineage; active versions never erase old observations."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from dataclasses import asdict
from typing import Any

from qwen3vl_agent.r5.types import FACT_KINDS, ProtocolError, SegmentCard
from qwen3vl_agent.r5.observation import evidence_refs


def identity(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
    ).hexdigest()[:24]


def strings(value: Any) -> list[str]:
    if not isinstance(value, list) or any(not isinstance(v, str) for v in value):
        raise ProtocolError("expected a string list")
    return list(dict.fromkeys(value))


def statement(value: Any, config: Any) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > config.max_statement_chars:
        raise ProtocolError("empty or oversized statement")
    return value.strip()


def parse_observation(
    data: dict, catalog: dict, tile: dict, config: Any, prior_ids: set[str],
    *, aliases: dict | None = None, fact_limit: int | None = None,
    permitted_modalities: set[str] | None = None,
) -> dict:
    rows = data.get("facts")
    limit = fact_limit or config.max_facts_per_card
    facts, hypotheses, rejected, warnings, local_names = [], [], [], [], {}
    lost = False
    if not isinstance(rows, list):
        rejected.append({"index": None, "reason": "facts must be an array", "raw": rows})
        rows, lost = [], True
    for index, row in enumerate(rows):
        local = f"f{index + 1}"
        try:
            if index >= limit:
                raise ProtocolError("fact limit exceeded")
            if not isinstance(row, dict):
                raise ProtocolError("fact must be an object")
            text = statement(row.get("statement"), config)
            refs = evidence_refs(row, catalog, aliases)
            kind = row.get("kind")
            if not isinstance(kind, str) or kind not in FACT_KINDS:
                raise ProtocolError("unknown fact kind")
            kinds = {catalog[r]["kind"] for r in refs}
            if kind in {"visual_observation", "screen_text"} and "frame" not in kinds:
                raise ProtocolError("transcript cannot prove a visual action or screen text")
            if kind in {"utterance", "reported_event"} and not kinds & {"subtitle", "asr"}:
                raise ProtocolError("reported content requires read text")
            required = {"visual_observation": "video", "screen_text": "screen_text"}.get(kind)
            if permitted_modalities is not None and required and required not in permitted_modalities:
                raise ProtocolError("fact modality is not permitted")
            raw_role = row.get("role", "other")
            role = raw_role.strip().lower() if isinstance(raw_role, str) else "other"
            inferred = role in {"inference", "inferred", "hypothesis", "interpretation"} or (
                str(row.get("epistemic", "observed")).lower()
                not in {"observation", "observed", "direct", "visual_observation"}
            ) or row.get("inferred") is True
            if not isinstance(raw_role, str) or role not in {"setting", "action", "outcome", "transition", "other"}:
                warnings.append({"index": index, "field": "role", "raw": raw_role, "normalized": "other"})
                role = "other"
            a, b = tile["core"]
            in_core = any(a <= catalog[r]["start_sec"] < b if catalog[r]["kind"] == "frame"
                          else catalog[r]["start_sec"] < b and catalog[r]["end_sec"] > a for r in refs)
            fact = {"local_id": local, "statement": text, "kind": kind, "role": role,
                    "raw_role": raw_role, "evidence_refs": refs, "in_core": in_core,
                    "protected": row.get("protected") is True or role in {"setting", "outcome", "transition"},
                    "supersedes": [], "correction_reason": None,
                    "support_status": "model_reported", "reference_status": "validated",
                    "epistemic": "inferred" if inferred else "observed"}
            if inferred:
                hypotheses.append(fact)
                continue
            try:
                supersedes = strings(row.get("supersedes", []))
                reason = row.get("correction_reason")
                if set(supersedes) - prior_ids or (supersedes and (not isinstance(reason, str) or not reason.strip())):
                    raise ProtocolError("correction needs supplied old facts and a reason")
                fact.update(supersedes=supersedes, correction_reason=reason)
            except ProtocolError as exc:
                warnings.append({"index": index, "field": "supersedes", "reason": str(exc)})
                lost = True
            old_local = row.get("local_id", local)
            if isinstance(old_local, str):
                if old_local in local_names:
                    local_names[old_local] = None
                else:
                    local_names[old_local] = local
            facts.append(fact)
        except (ProtocolError, ValueError, TypeError, KeyError) as exc:
            rejected.append({"index": index, "reason": str(exc), "raw": row})
            lost = True
    transitions = []
    edges = data.get("local_transitions", [])
    if not isinstance(edges, list):
        warnings.append({"field": "local_transitions", "reason": "expected array"})
        lost = True
        edges = []
    for edge in edges:
        if (isinstance(edge, dict) and isinstance(edge.get("from"), str) and isinstance(edge.get("to"), str)
                and local_names.get(edge["from"]) and local_names.get(edge["to"])
                and edge.get("relation") == "shown_before"):
            transitions.append({"from": local_names[edge["from"]], "to": local_names[edge["to"]], "relation": "shown_before"})
        else:
            warnings.append({"field": "local_transitions", "raw": edge, "reason": "invalid auxiliary relation"})
            lost = True
    entities = data.get("local_entities", [])
    if not isinstance(entities, list):
        entities = []
    entities = [e for e in entities if isinstance(e, dict)]
    try:
        unresolved = strings(data.get("unresolved", []))
    except ProtocolError:
        unresolved, lost = ["invalid_unresolved_field"], True
    flag = data.get("truncated", False)
    if not isinstance(flag, bool):
        flag, lost = True, True
    if rejected:
        unresolved.append("isolated_observation_facts")
    if lost:
        unresolved.append("observation_content_incomplete")
    return {
        "facts": facts,
        "hypotheses": hypotheses,
        "rejected": rejected,
        "normalizations": warnings,
        "local_entities": entities,
        "local_transitions": transitions,
        "unresolved": unresolved,
        "truncated": flag or lost,
    }


class FactCardStore:
    def __init__(self, state: dict | None = None):
        self.state = (
            state
            if state is not None
            else {
                "facts": {},
                "cards": {},
                "active_cards": {},
                "nodes": {},
                "claims": {},
                "corrections": [],
                "catalog": {},
                "hypotheses": {},
                "observations": {},
            }
        )

    def commit(
        self,
        tile: dict,
        value: dict,
        catalog: dict,
        call_id: str,
        *,
        target: str | None = None,
        coverage_ok: bool = True,
        resolve_coverage: bool = False,
    ) -> str:
        target = target or tile["segment_id"]
        previous = self.state["active_cards"].get(target)
        previous_card = self.state["cards"].get(previous)
        self.state["observations"][call_id] = {
            "segment_id": target, "rejected": value.get("rejected", []),
            "normalizations": value.get("normalizations", []),
            "truncated": value["truncated"], "unresolved": value["unresolved"],
        }
        for row in value.get("hypotheses", []):
            key = f"hypothesis:{call_id}:{row['local_id']}"
            self.state["hypotheses"][key] = {**row, "id": key, "segment_id": target, "call_id": call_id}
        keep = list(previous_card["fact_ids"]) if previous_card else []
        mapping = {f["local_id"]: f"fact:{call_id}:{f['local_id']}" for f in value["facts"]}
        for row in value["facts"]:
            fact_id = mapping[row["local_id"]]
            self.state["facts"][fact_id] = {
                **row,
                "id": fact_id,
                "segment_id": target,
                "call_id": call_id,
                "core_interval": list(tile["core"]),
            }
            if row["supersedes"]:
                self.state["corrections"].append(
                    {
                        "new_fact_id": fact_id,
                        "supersedes": row["supersedes"],
                        "reason": row["correction_reason"],
                    }
                )
                keep = [f for f in keep if f not in row["supersedes"]]
            keep.append(fact_id)
        card_id = f"card:{target}:{call_id}"
        card = SegmentCard(
            card_id,
            target,
            previous_card["core_interval"] if previous_card else list(tile["core"]),
            list(dict.fromkeys(keep)),
            value["local_entities"],
            [
                {"from": mapping[e["from"]], "to": mapping[e["to"]], "relation": e["relation"]}
                for e in value["local_transitions"]
            ],
            call_id,
            (
                "observed_under_policy"
                if coverage_ok and not value["truncated"] and (
                    not previous_card or resolve_coverage or previous_card["coverage_status"] == "observed_under_policy"
                )
                else "incomplete"
            ),
            value["unresolved"],
            value["truncated"],
            previous,
        )
        self.state["cards"][card_id] = asdict(card)
        self.state["active_cards"][target] = card_id
        self.state["catalog"].update(catalog)
        return card_id

    def candidate_explanations(self, limit: int = 8) -> list[dict]:
        result, seen = [], set()
        for row in self.state["hypotheses"].values():
            key = " ".join(row["statement"].lower().split())
            if key not in seen:
                seen.add(key)
                result.append({"statement": row["statement"], "segment_id": row["segment_id"],
                               "epistemic": "inferred", "citable": False})
                if len(result) == limit:
                    break
        return result

    def deactivate(self, segment_id: str) -> None:
        self.state["active_cards"].pop(segment_id, None)

    def cards(self) -> list[dict]:
        return sorted(
            (self.state["cards"][v] for v in self.state["active_cards"].values()),
            key=lambda c: (c["core_interval"], c["segment_id"]),
        )

    def pages(self, limit: int) -> list[dict]:
        pages = []
        for card in self.cards():
            facts = self.card_facts(card)
            for i in range(0, max(1, len(facts)), limit):
                pages.append(
                    {
                        **card,
                        "card_id": f"{card['card_id']}:page{i // limit}",
                        "fact_ids": facts[i : i + limit],
                    }
                )
        return pages

    def public_fact(self, fact_id: str) -> dict:
        fact = self.state["facts"][fact_id]
        return {
            key: fact[key]
            for key in (
                "id",
                "statement",
                "kind",
                "role",
                "protected",
                "segment_id",
                "core_interval",
            )
        }

    def card_facts(self, card: dict) -> list[str]:
        return [f for f in card["fact_ids"] if self.state["facts"][f]["in_core"]]

    def leaf_ids(self, ref: str) -> list[str]:
        if ref in self.state["facts"]:
            return [ref]
        if ref not in self.state["claims"]:
            raise ProtocolError("unknown summary support reference")
        return list(self.state["claims"][ref]["leaf_fact_ids"])

    def leaf_unit(self, card: dict, *, boundary: bool) -> dict:
        facts = self.card_facts(card)
        return {
            "id": card["card_id"],
            "units": [self.public_fact(f) for f in facts],
            "ranges": [card["core_interval"]],
            "protected": [f for f in facts if boundary or self.state["facts"][f]["protected"]],
            "conflicts": card["unresolved"],
        }

    def commit_node(self, key: str, children: list[dict], data: dict, call_id: str) -> dict:
        inputs = {}
        for child in children:
            for unit in child["units"]:
                inputs.setdefault(unit["id"], unit)
        passthrough_refs = data.get("passthrough_refs", [])
        passthrough = [deepcopy(inputs[ref]) for ref in passthrough_refs]
        passthrough_lineage = {ref: self.leaf_ids(ref) for ref in passthrough_refs}
        claims = []
        for i, row in enumerate(data["claims"]):
            claim_id = f"claim:{key}:{i}"
            leaves = sorted({f for ref in row["support_refs"] for f in self.leaf_ids(ref)})
            claim = {"id": claim_id, **row, "leaf_fact_ids": leaves, "call_id": call_id}
            self.state["claims"][claim_id] = claim
            claims.append({"id": claim_id, "statement": row["statement"], "kind": "synthesis"})
        node = {
            "id": f"node:{key}",
            "children": [c["id"] for c in children],
            "units": [*claims, *passthrough],
            "ranges": [r for c in children for r in c["ranges"]],
            "protected": sorted({f for c in children for f in c["protected"]}),
            "conflicts": [*data["conflicts"], *(x for c in children for x in c["conflicts"])],
            "omitted_refs": data["omitted_refs"],
            "passthrough_refs": list(passthrough_refs),
            "passthrough_lineage": passthrough_lineage,
            "call_id": call_id,
        }
        self.state["nodes"][key] = node
        return node
