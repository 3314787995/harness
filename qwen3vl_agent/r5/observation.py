"""Compact observation wire format; source identities never depend on model text."""

from __future__ import annotations

import json
from typing import Any

from qwen3vl_agent.r5.types import ProtocolError

PROTOCOL_VERSION = "r5-direct-v3.1"


def compact_catalog(catalog: dict) -> tuple[dict, dict]:
    public, aliases, counters = {}, {}, {"F": 0, "T": 0}
    for ref, item in sorted(catalog.items(), key=lambda kv: (kv[1]["start_sec"], kv[0])):
        prefix = "F" if item["kind"] == "frame" else "T"
        counters[prefix] += 1
        alias = f"{prefix}{counters[prefix]}"
        aliases[alias] = ref
        public[alias] = {
            "id": alias, "kind": item["kind"],
            "start_sec": item["start_sec"], "end_sec": item["end_sec"],
            **({"text": item["text"]} if "text" in item else {}),
        }
    return public, aliases


def evidence_refs(row: dict, catalog: dict, aliases: dict | None) -> list[str]:
    values = row.get("evidence", row.get("evidence_refs"))
    if isinstance(values, str):
        values = [values]
    if not isinstance(values, list) or not values or any(not isinstance(v, str) for v in values):
        raise ProtocolError("evidence must contain a shown point or two frame endpoints")
    if aliases is None:  # Direct parser callers may supply canonical recorded references.
        if any(v not in catalog for v in values):
            raise ProtocolError("fact references unshown or unread evidence")
        return list(dict.fromkeys(values))
    if len(values) > 2 or any(v not in aliases for v in values):
        raise ProtocolError("evidence needs one known alias or two known frame endpoints")
    refs = [aliases[v] for v in values]
    if len(refs) == 1 or refs[0] == refs[-1]:
        return refs[:1]
    first, last = (catalog[r] for r in refs)
    if first["kind"] != "frame" or last["kind"] != "frame":
        raise ProtocolError("only frame evidence accepts an interval")
    a, b = first["start_sec"], last["start_sec"]
    if a > b:
        raise ProtocolError("evidence endpoints are reversed")
    return [r for r, item in sorted(catalog.items(), key=lambda kv: kv[1]["start_sec"])
            if item["kind"] == "frame" and a <= item["start_sec"] <= b]


def decode_observation(raw: str, *, truncated: bool = False) -> dict[str, Any]:
    """Salvage only fully decoded entries of the TOP LEVEL facts array."""
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1].strip()
        if text.endswith("```"):
            text = text[:-3].rstrip()
    decoder = json.JSONDecoder(parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
    try:
        obj, end = decoder.raw_decode(text)
        if not isinstance(obj, dict):
            raise ValueError("observation must be an object")
        if truncated:
            obj["truncated"] = True
        return obj
    except (ValueError, TypeError):
        pass
    facts: list[Any] = []
    pos = 1

    def skip(index: int) -> int:
        while index < len(text) and text[index].isspace():
            index += 1
        return index

    try:
        if not text.startswith("{"):
            raise ValueError("missing object")
        while pos < len(text):
            key, pos = decoder.raw_decode(text, skip(pos))
            pos = skip(pos)
            if not isinstance(key, str) or text[pos] != ":":
                break
            pos = skip(pos + 1)
            if key == "facts":
                if text[pos] != "[":
                    break
                pos = skip(pos + 1)
                while pos < len(text) and text[pos] != "]":
                    value, pos = decoder.raw_decode(text, pos)
                    facts.append(value)
                    pos = skip(pos)
                    if pos >= len(text) or text[pos] != ",":
                        break
                    pos = skip(pos + 1)
                break
            _, pos = decoder.raw_decode(text, pos)
            pos = skip(pos)
            if text[pos] != ",":
                break
            pos = skip(pos + 1)
    except (ValueError, TypeError, IndexError):
        pass
    return {"facts": facts, "truncated": True,
            "unresolved": ["observation_json_incomplete"], "recovered_json": True}
