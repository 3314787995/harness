"""Append-only readings with source checks and dependency-local invalidation."""

from copy import deepcopy
from dataclasses import asdict

from .types import ProtocolError, VariableRecord, digest
from .units import Unit, number, numeric_tokens

# These are mathematical definitions, not guessed observations or world knowledge.
DEFINITIONS = {
    "zero": ("0", "1"),
    "one": ("1", "1"),
    "two": ("2", "1"),
    "hundred": ("100", "1"),
    "sixty": ("60", "1"),
    "triangle_degrees": ("180", "degree"),
    "right_degrees": ("90", "degree"),
}


def numeric_in_span(value, text):
    """Conservative digit grounding. Word-number givens remain unresolved for explicit review."""
    if value is None:
        return True
    if isinstance(value, bool):
        return str(value).lower() in text.lower()
    if isinstance(value, (dict, list)):
        return False
    try:
        parsed = number(value)
    except ValueError:
        return str(value) in text
    tokens = numeric_tokens(text)
    return any(number(token) == parsed for token in tokens)


class EvidenceStore:
    def __init__(self, question, contract, state=None):
        self.question, self.contract = question, contract
        self.state = state if state is not None else {}
        for key, default in (
            ("evidence", {}),
            ("variables", {}),
            ("current", {}),
            ("entities", {}),
            ("relations", {}),
            ("relation_versions", {}),
            ("derived", {}),
            ("revisions", []),
            ("applied", []),
        ):
            self.state.setdefault(key, deepcopy(default))

    def present(self, evidence):
        for e in evidence.values():
            if e["scope_hash"] != self.contract.fingerprint or not self.contract.permits(
                e["timestamp_seconds"]
            ):
                raise ProtocolError("evidence outside permission")
            if e.get("modality") not in self.contract.available_modalities:
                raise ProtocolError("unpermitted evidence modality")
            previous = self.state["evidence"].get(e["id"])
            if previous and previous != e:
                raise ProtocolError("evidence ID substitution")
            self.state["evidence"][e["id"]] = deepcopy(e)

    def sources(self, refs, *, packet=None):
        if not refs:
            raise ProtocolError("observed record requires actual presented evidence")
        output = []
        permitted = (
            {e["id"] for e in (packet or {}).values()}
            if packet is not None
            else set(self.state["evidence"])
        )
        for ref in refs:
            ref = packet[ref]["id"] if packet is not None and ref in packet else ref
            if ref not in permitted or ref not in self.state["evidence"]:
                raise ProtocolError("unknown/unpresented evidence reference")
            output.append(ref)
        return sorted(set(output))

    def get(self, ref):
        if "@" not in ref:
            ref = self.state["current"].get(ref, ref)
        value = self.state["variables"].get(ref)
        if not value or not value["valid"]:
            raise ProtocolError(f"unknown or stale variable: {ref}")
        return value

    def invalidate(self, parent):
        queue, invalidated = [parent], []
        while queue:
            old = queue.pop(0)
            for key, value in {**self.state["variables"], **self.state["derived"]}.items():
                if value.get("valid", True) and old in value.get("parents", []):
                    value["valid"] = False
                    invalidated.append(key)
                    queue.append(key)
        return invalidated

    def append(self, record, *, origin="observed", packet=None, support="unreviewed"):
        row = deepcopy(record)
        row.pop("origin", None)
        Unit.parse(row["unit"])
        if not row["id"] or "@" in row["id"]:
            raise ProtocolError("invalid variable identity")
        if row["entity_id"] not in self.state["entities"] and origin == "observed":
            raise ProtocolError("variable entity missing")
        if origin == "observed":
            row["evidence_refs"] = self.sources(row["evidence_refs"], packet=packet)
            if not numeric_in_span(row["value"], row["raw_text"]):
                raise ProtocolError("observed value absent from raw transcription")
            for key in ("event_time",):
                if row.get(key) is not None and not self.contract.permits(row[key]):
                    raise ProtocolError("event time outside permission")
            if row.get("valid_interval") and not self.contract.permits_span(row["valid_interval"]):
                raise ProtocolError("variable interval outside permission")
        elif origin in {"given", "hypothetical"}:
            text = row.get("question_span", "")
            if not text or text not in self.question or not numeric_in_span(row["value"], text):
                raise ProtocolError("given/hypothetical value must occur in exact question span")
            row.update(raw_text=text, evidence_refs=[])
        else:
            raise ProtocolError("only host execution may add derived or definition values")
        old_ref = self.state["current"].get(row["id"])
        old = self.state["variables"].get(old_ref)
        if old:
            if old["origin"] != origin:
                raise ProtocolError("given/hypothetical and observed values require separate IDs")
            key = ("entity_id", "attribute", "role", "scope", "snapshot", "unit_basis")
            if any(old[k] != row[k] for k in key):
                raise ProtocolError("variable ID rebound to another entity/role/scope/snapshot")
            if all(old.get(k) == v for k, v in row.items()) and old["origin"] == origin:
                return old_ref
        version = old["version"] + 1 if old else 1
        value = VariableRecord(**row, version=version, origin=origin, visual_support_status=support)
        ref = value.ref
        invalidated = self.invalidate(old_ref) if old_ref else []
        if old:
            old["valid"] = False
        self.state["variables"][ref] = asdict(value)
        self.state["current"][value.id] = ref
        self.state["revisions"].append({"old": old_ref, "new": ref, "invalidated": invalidated})
        return ref

    def ingest(self, value, packet, key, *, reread=False):
        if key in self.state["applied"]:
            return []
        # Validate on a copy so a malformed last record cannot partially mutate state.
        candidate = EvidenceStore(self.question, self.contract, deepcopy(self.state))
        refs = []
        for entity in value["entities"]:
            old = candidate.state["entities"].get(entity["id"])
            if old and any(old[k] != entity[k] for k in ("scope", "snapshot")):
                raise ProtocolError("entity ID crosses scope/snapshot")
            candidate.state["entities"][entity["id"]] = deepcopy(entity)
        for row in value["observations"]:
            old = candidate.state["current"].get(row["id"])
            previous = candidate.state["variables"].get(old, {})
            ref = candidate.append(row, packet=packet)
            if reread:
                current = candidate.state["variables"][ref]
                keys = ("value", "unit", "entity_id", "attribute", "role", "snapshot", "scope")
                current["visual_support_status"] = (
                    "consistent_reread"
                    if previous and all(previous.get(k) == current[k] for k in keys)
                    else "revised_on_reread"
                )
            refs.append(ref)
        for relation in value["relations"]:
            relation = deepcopy(relation)
            relation["evidence_refs"] = candidate.sources(relation["evidence_refs"], packet=packet)
            if relation["source_kind"] == "given" and (
                not relation["question_span"] or relation["question_span"] not in self.question
            ):
                raise ProtocolError("relation given absent from question")
            if any(e not in candidate.state["entities"] for e in relation["objects"].values()):
                raise ProtocolError("relation names unknown object")
            if any(
                candidate.state["entities"][e]["snapshot"] != relation["snapshot"]
                for e in relation["objects"].values()
            ):
                raise ProtocolError("relation crosses snapshots")
            old = candidate.state["relations"].get(relation["id"])
            if old and any(old[k] != relation[k] for k in ("scope", "snapshot", "objects")):
                raise ProtocolError("relation ID rebound to other objects or snapshot")
            if old != relation:
                versions = candidate.state["relation_versions"].setdefault(relation["id"], [])
                if versions:
                    previous = f"relation:{relation['id']}@{len(versions)}"
                    candidate.invalidate(previous)
                    candidate.state["derived"][previous]["valid"] = False
                versions.append(deepcopy(relation))
                candidate.state["derived"][f"relation:{relation['id']}@{len(versions)}"] = {
                    "value": deepcopy(relation),
                    "parents": [],
                    "valid": True,
                    "origin": "observed_relation",
                }
            candidate.state["relations"][relation["id"]] = relation
        candidate.state["applied"].append(key)
        self.state.clear()
        self.state.update(candidate.state)
        return refs

    def add_derived(self, key, value, parents, *, kind="execution"):
        for parent in parents:
            if parent in self.state["derived"]:
                if not self.state["derived"][parent]["valid"]:
                    raise ProtocolError("stale derived parent")
            else:
                self.get(parent)
        self.state["derived"][key] = {
            "value": value,
            "parents": list(parents),
            "valid": True,
            "origin": "derived",
            "kind": kind,
        }

    def current(self):
        return {
            key: deepcopy(self.get(ref))
            for key, ref in self.state["current"].items()
            if self.state["variables"][ref]["valid"]
        }

    def fingerprint(self):
        return digest({"variables": self.current(), "relations": self.state["relations"]})
