"""Evidence cards, isolated row commits and revisable identity assertions."""
from __future__ import annotations

from .convergence import generic_category
import copy
import hashlib
import itertools
import json
import re
from dataclasses import asdict

from .collection_contracts import (ContractError, conditions, record_schema, check_schema, task_schema,
                                   RELATION_SCHEMA, validate, snapshot_schema)
from .contracts import issue
from .ledger import CanonicalInventory, components, normalize
from .observation import parse_batch, source_bbox


def fail(path, code, message, expected=None, actual=None):
    raise ContractError([issue(path, code, message, expected=expected, actual=actual)])


def restore_refs(refs, aliases, catalog, path, *, required=False):
    if required and not refs:
        fail(path, "missing_positive_evidence", "Positive evidence needs a supplied reference")
    result = []
    for i, ref in enumerate(refs):
        if ref not in aliases or aliases[ref] not in catalog:
            fail(f"{path}[{i}]", "unknown_reference", "Only actually supplied evidence references are admissible", list(aliases), ref)
        result.append(aliases[ref])
    return list(dict.fromkeys(result))


def in_core(meta, tile, set_id):
    if meta.get("entry_id") != tile.get("entry_id"):
        return False
    temporal = tile["core"][0] <= meta["start_sec"] < tile["core"][1]
    # Complete text is owned by its start window. Query/cutoff containment is checked
    # separately by membership_sets and the provider, not by chopping utterances at grid boundaries.
    return temporal and ("membership_sets" not in meta or set_id in meta["membership_sets"])


def modality_ok(target, refs, catalog, raw_text=None):
    kinds = {catalog[r]["kind"] for r in refs}
    if target.evidence_relation in {"mentioned", "planned", "completed"}:
        return bool(kinds & (set(target.required_modalities) & {"subtitle", "asr"})) or (
            "screen_text" in target.required_modalities and "frame" in kinds and bool(raw_text))
    if target.evidence_relation == "text_present":
        return ("frame" in kinds and bool(set(target.required_modalities) & {"video", "screen_text"})) or (
            "subtitle" in kinds and "subtitle" in target.required_modalities)
    return "frame" in kinds


def parse_card(row, target, tile, aliases, catalog, *, inspection=False):
    row = validate(row, record_schema(target, inspection=inspection))
    evidence = restore_refs(row.get("refs", []), aliases, catalog, "refs")
    detections = []
    for i, d in enumerate(row.get("boxes", [])):
        if any(type(v) is not int for v in d["xyxy"]):
            fail(f"boxes[{i}].xyxy", "coordinate_type", "Coordinates must be JSON integers, not decimal values", "four integers", d["xyxy"])
        ref = restore_refs([d["ref"]], aliases, catalog, f"boxes[{i}].ref")[0]
        a, b, c, e = d["xyxy"]
        if not (a < c and b < e):
            fail(f"boxes[{i}].xyxy", "invalid_bbox", "Box must have positive area", "left<right, top<bottom", d["xyxy"])
        if catalog[ref]["kind"] != "frame":
            fail(f"boxes[{i}].ref", "box_not_frame", "Boxes require an actual image")
        detections.append({"ref": ref, "bbox": source_bbox(d["xyxy"], catalog[ref]),
                           "source_frame_id": catalog[ref]["source_frame_id"], "wire_bbox": d["xyxy"],
                           "in_core":in_core(catalog[ref], tile, target.set_id)})
    if target.namespace == "physical_instance":
        derived = list(dict.fromkeys(d["ref"] for d in detections))
        if evidence and set(evidence) != set(derived):
            fail("refs", "conflicting_entity_refs", "Explicit refs conflict with detections", derived, evidence)
        evidence = derived
    if not modality_ok(target, evidence, catalog, row.get("raw_text")):
        fail("refs", "wrong_modality", "The claim requires the requested modality", target.required_modalities, [catalog[r]["kind"] for r in evidence])
    states = set(row["conditions"].values())
    membership = "excluded" if "no" in states else "accepted" if states == {"yes"} else "unknown"
    issues = list(row.get("uncertainties", []))
    if not any(in_core(catalog[r], tile, target.set_id) for r in evidence):
        membership = "unknown"
        issues.append("context_only_member")
    visibility = row.get("visibility", "clear")
    if visibility in {"occluded", "unreadable", "unknown"} or issues:
        membership = "unknown" if membership != "excluded" else membership
        issues.append("visibility_or_membership_uncertain")
    if target.predicate_kind in {"moving", "enters", "exits"}:
        motion = row["motion"]
        mr = restore_refs(motion["refs"], aliases, catalog, "motion.refs")
        witnesses = restore_refs(motion["witness_refs"], aliases, catalog, "motion.witness_refs")
        times = {(catalog[r].get("source_id"), catalog[r]["start_sec"]) for r in mr if catalog[r]["kind"] == "frame"}
        justified = (len(times) >= 2 and motion["object_motion"] and motion["camera_motion_accounted"]
                     and any(in_core(catalog[r], tile, target.set_id) for r in witnesses))
        if target.namespace == "physical_instance":
            justified &= len({catalog[d["ref"]]["start_sec"] for d in detections}) >= 2
        if target.predicate_kind in {"enters", "exits"}:
            justified &= motion.get("boundary_crossing") is True and motion.get("identity_continuity") is True
        if not justified and membership == "accepted":
            membership = "unknown"
            issues.append("motion_unproven")
    if tile.get("crop_core") and detections:
        x1, y1, x2, y2 = tile["crop_core"]
        b = detections[0]["bbox"]
        cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
        owns = x1 <= cx < x2 and y1 <= cy < y2
        if not owns:
            membership = "excluded"
            issues.append("neighbor_crop_owns_center")
        wire = detections[0]["wire_bbox"]
        crop_box = catalog[detections[0]["ref"]].get("crop_transform", {}).get("bbox_xyxy_1000", [0,0,1000,1000])
        cut_by_internal_edge = ((wire[0] == 0 and crop_box[0] > 0) or (wire[1] == 0 and crop_box[1] > 0)
                                or (wire[2] == 1000 and crop_box[2] < 1000) or (wire[3] == 1000 and crop_box[3] < 1000))
        if owns and cut_by_internal_edge:
            membership = "unknown"
            issues.append("crop_boundary_unresolved")
    for attribute in target.attribute_keys:
        if row.get("attributes", {}).get(attribute) is None:
            issues.append("required_attribute_missing:" + attribute)
    query_value = row.get("query_value")
    mapping = row.get("mapping_evidence")
    if target.namespace == "text_value":
        if not row.get("raw_text"):
            membership = "unknown"
            issues.append("literal_text_missing")
        query_value = normalize(row["raw_text"], target.normalization) if row.get("raw_text") else None
        mapping = "literal transcription with public normalization" if target.normalization else "literal transcription"
    elif target.equivalence == "combination":
        attrs = row.get("attributes", {})
        if any(attrs.get(k) is None for k in target.attribute_keys):
            query_value = None
            issues.append("combination_attributes_missing")
        else:
            query_value = json.dumps([attrs[k] for k in target.attribute_keys], ensure_ascii=False, separators=(",", ":"))
            mapping = "ordered task attribute dimensions"
    elif target.namespace == "semantic_category" and (not query_value or not mapping):
        query_value = None
        issues.append("category_mapping_unresolved")
    elif generic_category(query_value, target):
        # This is a copied unit label, not a query-granularity value. Keep its
        # provenance for targeted review; never invent the intended category.
        generic_issue = "query_value_is_count_unit" if " ".join(query_value.casefold().split()) == " ".join(target.count_unit.casefold().split()) else "query_value_is_generic_target"
        query_value = None
        issues.append(generic_issue)
    # All detections have already passed geometry, modality and provenance checks.
    # Presentation selection does not discard evidence or infer identity.
    preferred = sorted(range(len(detections)), key=lambda i: (not detections[i]["in_core"], i))[:3]
    return {"set_id": target.set_id, "entry_id": tile["entry_id"], "window_id": tile["tile_id"],
            "local_id": row.get("id"), "raw_value": row["name"], "actual_class": row["class"],
            "membership": membership, "conditions": row["conditions"], "facts": row["facts"],
            "evidence_refs": evidence, "detections": detections, "attributes": row.get("attributes", {}),
            "representative_detections": [detections[i] for i in preferred],
            "query_value": query_value, "reported_query_value": row.get("query_value"),
            "mapping_evidence": mapping, "visibility": visibility,
            "issues": sorted(set(issues)), "population": target.population, "count_unit": target.count_unit,
            "namespace": target.namespace, "equivalence": target.equivalence}


class EvidenceStore:
    def __init__(self, spec, state=None):
        self.spec = spec
        self.targets = {s.set_id: s for s in spec.sets}
        self.state = state if state is not None else {
            "cards": {}, "checks": {}, "relations": [], "revisions": [], "slots": {},
            "history": None, "relation_attempts": [], "quarantined": {}, "events": []}
        self.history = CanonicalInventory(spec, self.state.get("history"))
        self.state["history"] = self.history.state
        self.state.setdefault("snapshots", {})

    @property
    def cards(self):
        return self.state["cards"]

    def commit_card(self, card, slot, call_id, *, candidate_id=None):
        event = call_id + ":" + slot
        if event in self.state["events"]:
            return self.state["slots"].get(slot)
        key = candidate_id or self.state["slots"].get(slot) or "C" + hashlib.sha256(slot.encode()).hexdigest()[:16]
        if candidate_id and candidate_id not in self.cards:
            fail("candidate_id", "unknown_candidate", "Inspection cannot create a candidate")
        if key in self.cards:
            old = self.cards[key]
            if card["set_id"] != old["set_id"]:
                fail("set", "inspection_set_change", "An inspection cannot change the candidate's set")
            self.state["revisions"].append({"candidate_id": key, "before": copy.deepcopy(old), "call_id": call_id})
            card = {**card, "raw_value": old["raw_value"], "window_id": old["window_id"],
                    "evidence_refs": sorted(set(old["evidence_refs"] + card["evidence_refs"])),
                    "detections": old["detections"] + [d for d in card["detections"] if d not in old["detections"]]}
        self.cards[key] = {**card, "candidate_id": key, "call_id": call_id}
        self.state["slots"][slot] = key
        self.state["quarantined"].pop(slot, None)
        self.state["events"].append(event)
        # Shared slots and explicit inspection IDs are idempotent. Separate records do
        # not acquire SAME merely from equal names, source-frame coordinates or colors.
        return key

    def commit_check(self, row, target, tile, aliases, catalog, slot, call_id, *, review_key=None):
        row = validate(row, check_schema(target))
        refs = restore_refs(row["refs"], aliases, catalog, "refs", required=row["state"] == "seen")
        status = row["state"]
        if row.get("uncertainties"):
            status = "unreadable"
        if status == "seen":
            if not modality_ok(target, refs, catalog, row.get("raw_text")):
                fail("refs", "wrong_modality", "Candidate positive evidence requires the requested modality")
            if not any(in_core(catalog[r], tile, target.set_id) for r in refs):
                status = "unreadable"
            if row.get("support") != "direct" or row.get("uncertainties"):
                status = "unreadable"
        key = json.dumps([target.set_id, row["candidate"], tile["tile_id"]], ensure_ascii=False)
        if review_key is not None and (review_key != key or key not in self.state["checks"]):
            fail("candidate", "invalid_check_review", "Review must replace exactly the requested existing check")
        previous = self.state["checks"].get(key)
        if previous and previous.get("call_id") == call_id:
            return key
        updated = {**row, "reported_state": row["state"], "state": status, "refs": refs, "window_id": tile["tile_id"],
                   "call_id": call_id, "input_refs": list(aliases.values()), "reviewed": review_key is not None,
                   "needs_review": False}
        if previous:
            self.state.setdefault("check_revisions", []).append(
                {"key": key, "before": copy.deepcopy(previous), "after": copy.deepcopy(updated), "call_id": call_id})
        self.state["checks"][key] = updated
        self.state["quarantined"].pop(slot, None)
        return key

    def commit_task(self, row, target, tile, source, aliases, catalog, slot, call_id, local_update_ids=None):
        row = validate(row, task_schema(target))
        row["evidence_refs"] = restore_refs(row["evidence_refs"], aliases, catalog, "evidence_refs", required=True)
        if not any(in_core(catalog[r], tile, target.set_id) for r in row["evidence_refs"]):
            row["binding_supported"] = False
        parsed = parse_batch({"observation_status": "valid", "observations": [], "task_updates": [row]},
                             self.spec, catalog, tile, source, 12)
        for update in parsed["updates"]:
            reference = update.get("refers_to")
            if reference and local_update_ids and reference in local_update_ids:
                update["refers_to"] = local_update_ids[reference]
            elif reference and reference not in self.history.state["task_updates"]:
                fail("refers_to", "unknown_task_reference", "A task correction must reference an accepted update")
        before = set(self.history.state["task_updates"])
        candidate = parsed["updates"][0]
        identity_fields = ("kind", "owner", "task_id", "item_key", "quantity", "unit", "evidence_refs", "effective_time",
                           "replacement_item", "replacement_quantity", "completion_predicate", "refers_to")
        same = next((u["update_id"] for u in self.history.state["task_updates"].values()
                     if all(u.get(k) == candidate.get(k) for k in identity_fields)), None)
        if same:
            self.state["quarantined"].pop(slot, None)
            return same
        self.history.commit(parsed, slot)
        added = set(self.history.state["task_updates"]) - before
        self.state["quarantined"].pop(slot, None)
        return next(iter(added), None)

    def commit_snapshot(self, row, target, tile, aliases, catalog, slot, call_id, local_ids):
        row=validate(row,snapshot_schema(target))
        if target.namespace != "physical_instance":
            fail("set","snapshot_unit","Simultaneous physical inventories require physical instances")
        frames={}
        for short,groups in row["frames"].items():
            ref=restore_refs([short],aliases,catalog,"frames."+short,required=True)[0]
            meta=catalog[ref]
            if not in_core(meta,tile,target.set_id):
                fail("frames."+short,"snapshot_outside_core","A census must belong to the current query/core frame")
            used=set();members={}
            for carrier,ids in groups.items():
                if not carrier.strip():fail("frames."+short,"carrier_missing","A carrier needs a grounded local name")
                resolved=[]
                for ident in ids:
                    key=local_ids.get(ident)
                    if key not in self.cards or self.cards[key]["set_id"] != target.set_id:
                        fail("frames."+short,"snapshot_dependency","Census entries must reference accepted records in this set")
                    if key in used:fail("frames."+short,"duplicate_carrier_membership","One instance cannot occupy two carrier slots at the same instant")
                    used.add(key);resolved.append(key)
                members[carrier]=resolved
            frames[ref]={"groups":members,"source_frame_id":meta["source_frame_id"],"full_frame":not bool(meta.get("crop_transform"))}
        # A census explicitly asserts simultaneous independent instances; missing or
        # uncertain frames are omitted, never filled with empty inventories by the host.
        for ref,frame in frames.items():
            ids=[k for group in frame["groups"].values() for k in group]
            for a,b in itertools.combinations(ids,2):
                self.add_relation(a,b,"different","explicit_frame_census",[ref],"independent simultaneous instances explicitly listed for one supplied frame",call_id=call_id)
        self.state["snapshots"][slot]={"set_id":target.set_id,"window_id":tile["tile_id"],"frames":frames,"call_id":call_id}
        self.state["quarantined"].pop(slot,None)
        return slot

    def add_relation(self, left, right, relation, basis, refs, facts, *, supersedes=(), call_id=None):
        if left == right or left not in self.cards or right not in self.cards:
            fail("relation", "unknown_candidate", "Identity relation requires two accepted candidate IDs")
        if any(self.cards[x]["namespace"] != "physical_instance" for x in (left, right)):
            fail("relation", "nonphysical_identity", "Category and text records do not use physical identity")
        if (self.cards[left]["population"], self.cards[left]["count_unit"]) != (self.cards[right]["population"], self.cards[right]["count_unit"]):
            fail("relation", "identity_unit_mismatch", "Identity needs compatible units/population")
        existing = self.state["relations"]
        by_id = {r["relation_id"]: r for r in existing}
        for ident in supersedes:
            if ident not in by_id or {by_id[ident]["left"], by_id[ident]["right"]} != {left, right}:
                fail("supersedes", "invalid_supersession", "Only the same pair can be explicitly revised")
        signature = (left, right, relation, basis, sorted(set(refs)))
        if any((r["left"], r["right"], r["relation"], r["basis"], r["evidence_refs"]) == signature for r in existing if r["active"] or basis == "shared_observation"):
            return
        for ident in supersedes:
            by_id[ident]["active"] = False
        rec = {"relation_id": f"R{len(existing)+1}", "left": left, "right": right, "relation": relation,
               "basis": basis, "evidence_refs": sorted(set(refs)), "reason": facts, "active": True,
               "supersedes": list(supersedes), "call_id": call_id}
        existing.append(rec)
        graph = self.graph()
        if graph["conflicts"]:
            # Quarantine the SAME assertions participating in an inconsistent component.
            for edge in existing:
                if edge["active"] and edge["relation"] == "same" and edge["relation_id"] in graph["conflicts"]:
                    edge.update(active=False, reopened_reason="contradiction_with_independent_objects")

    def graph(self):
        nodes = [k for k, c in self.cards.items() if c["namespace"] == "physical_instance" and c["membership"] != "excluded"]
        return components(nodes, [r for r in self.state["relations"] if r["active"] and r["left"] in nodes and r["right"] in nodes])

    def accept_relation(self, row, aliases, catalog, call_id, allowed_ids):
        row = validate(row, RELATION_SCHEMA)
        left, right, relation = row["left"], row["right"], row["relation"]
        if left not in allowed_ids or right not in allowed_ids:
            fail("relation", "unshown_candidate", "Comparison can only update the supplied candidate shortlist")
        refs = restore_refs(row["refs"], aliases, catalog, "refs", required=relation != "UNKNOWN")
        parents = {catalog[r].get("parent_ref", r) for r in refs}
        if relation != "UNKNOWN":
            for key in (left, right):
                evidence = {catalog.get(r, {}).get("parent_ref", r) for r in self.cards[key]["evidence_refs"]}
                if not parents.intersection(evidence):
                    fail("refs", "both_candidates_required", "Identity needs displayed evidence for both candidates")
        basis = row["basis"]
        if relation == "SAME":
            if basis not in {"shared_observation", "continuous_track", "reidentification"}:
                fail("basis", "insufficient_identity_basis", "SAME needs identity evidence")
            if basis == "shared_observation":
                if not any(a["source_frame_id"] == b["source_frame_id"] and a["bbox"] == b["bbox"]
                           for a in self.cards[left]["detections"] for b in self.cards[right]["detections"]):
                    fail("basis", "not_shared_observation", "Different localizations are not exact source reuse")
            if basis == "continuous_track" and (not row.get("continuous_identity") or len({catalog[r]["start_sec"] for r in refs}) < 2):
                fail("basis", "continuity_unproven", "Continuity needs multiple source times and an explicit continuity judgement")
            if basis == "reidentification" and (not row.get("features") or len(refs) < 2):
                fail("features", "reidentification_unproven", "Reidentification needs stable features and both representatives")
        if relation == "DIFFERENT":
            if basis not in {"coexistence", "distinct_tracks", "stable_difference"}:
                fail("basis", "insufficient_identity_basis", "Box differences or failed matching do not establish DIFFERENT")
            if not row.get("independent_objects"):
                fail("independent_objects", "independence_unproven", "DIFFERENT requires independent physical objects")
            if basis == "coexistence" and not any(a["source_frame_id"] == b["source_frame_id"] and a["bbox"] != b["bbox"]
                    and a["source_frame_id"] in {catalog[r].get("source_frame_id") for r in refs}
                    for a in self.cards[left]["detections"] for b in self.cards[right]["detections"]):
                fail("basis", "coexistence_unproven", "Coexistence needs two localizations on the same source frame")
            if basis in {"distinct_tracks", "stable_difference"} and not row.get("features"):
                fail("features", "distinction_unproven", "State reliable distinguishing evidence, not coordinate differences")
            if basis in {"distinct_tracks", "stable_difference"}:
                weak = re.compile(r"bbox|bounding.?box|box.?width|crop|unmatched|match(?:ing)? fail|coordinates|裁剪|框宽|坐标|匹配失败", re.I)
                if all(weak.search(feature) for feature in row["features"]):
                    fail("features", "insufficient_identity_basis", "Coordinate/crop changes and failed matching do not prove distinct entities", "independent physical evidence", row["features"])
        self.add_relation(left, right, relation.lower(), basis, refs, row["facts"], supersedes=row.get("supersedes", []), call_id=call_id)
        self.state["relation_attempts"].append(sorted([left, right]))

    def snapshot(self):
        return copy.deepcopy(self.state)
