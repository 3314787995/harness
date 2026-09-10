"""Program-controlled R4 collection execution; each question owns all reasoning state."""
from __future__ import annotations

import copy
import hashlib
import itertools
import json
import math
import time
from dataclasses import asdict, fields, replace

from .checkpoint import Checkpoint, file_digest, implementation_digest, model_signature
from .collection_contracts import (ContractError, conditions, diagnostics, envelope_schema,
                                   parse_compile, compile_diagnostics, validate, DISTINCT_SCHEMA, COEXISTING_SCHEMA)
from .collection_reduce import (map_answer, reduce_inventory, coverage_for, simple_choice)
from .contracts import issue, error_details
from .inventory import EvidenceStore, fail, in_core, parse_card, restore_refs
from .planning import resolve_sources, query_scope, scope_spans, make_plan, history_time
from .providers import ProviderAdapter
from .session import CollectionSession, StageFailure, InputNeedsSplit
from .types import BudgetExhausted, CoverageTile, InventorySpec, ProtocolError, R4Result, SetSpec, timestamp
from .prompts import VERSION
from .interaction import bind_interaction, decode_observation, decode_pair
from .convergence import semantic_review, progress_snapshot, obligations


def executor(target, scope, operations):
    if target.namespace != "task_item" and target.candidates and any(op.op in {"membership", "missing_members"} and target.set_id in op.inputs for op in operations):
        return "candidate_presence"
    if target.namespace == "physical_instance":
        return "local_instances" if scope.get("kind") == "frame" else "cross_segment_entities"
    return "category_text_combination"


def public_target(target):
    return {"set_id": target.set_id, "namespace": target.namespace, "target": target.target,
            "count_unit": target.count_unit, "requirements": conditions(target), "equivalence": target.equivalence,
            "attribute_keys": list(target.attribute_keys), "evidence_relation": target.evidence_relation,
            "required_modalities": list(target.required_modalities), "population": target.population,
            **({"owner": target.owner, "task_id": target.task_id, "task_projection": target.task_projection}
               if target.namespace == "task_item" else {})}


class CollectionController:
    def __init__(self, agent, request):
        self.agent, self.request, self.config, self.media = agent, request, agent.config, agent.media
        self.sources, source_issues = resolve_sources(request, self.media)
        if not any(s["allowed"] for s in self.sources.values()):
            raise StageFailure("input", "no_permitted_media", "No permitted source range")
        if request.resume and agent.provider is not None and getattr(agent.provider, "version", None) is None:
            raise ValueError("Resuming an injected provider requires a stable version")
        request_key = asdict(request)
        request_key.pop("resume"); request_key.pop("checkpoint_path")
        fingerprint = {"protocol": VERSION, "request": request_key, "config": asdict(self.config),
            "implementation": implementation_digest(), "sources": {k: {"media": s["source_id"],
                "external": [{"path": f["path"], "digest": file_digest(f["path"])} for f in s["external_files"]]} for k, s in self.sources.items()},
            "model": model_signature(agent.model), "model_type": type(agent.model).__qualname__,
            "model_config": {k: getattr(agent.model, k, None) for k in ("dtype", "device", "device_map", "max_memory", "attn_implementation", "revision", "video_config", "generation")},
            "provider_version": getattr(agent.provider, "version", None)}
        self.checkpoint = Checkpoint(request.checkpoint_path, fingerprint, resume=request.resume)
        self.work = self.checkpoint.restored or {"spec": None, "inventory": None, "windows": {}, "scope_windows": {},
            "bindings": {}, "scope_done": False, "catalog": {}, "provider_cache": {}, "provider_reads": {},
            "scope_issues": source_issues, "issues": [], "transactions": {}, "failures": [], "executors": {},
            "review_round": 0, "result": None, "started_at": time.time()}
        self.spec, self.store = None, None
        self.session = CollectionSession(agent.model, self.media, self.config, request, self.work, self.save, request.checkpoint_path)
        self.adapters = {k: ProviderAdapter(agent.provider, s, self.config, self.session, self.work["provider_cache"]) for k, s in self.sources.items()}
        duration = sum(b-a for s in self.sources.values() for a, b in s["allowed"])
        if not self.session.state["limits"]:
            self.session.configure(max(1, math.ceil(duration/8)), True, duration)

    def save(self):
        if self.store is not None:
            self.work["inventory"] = self.store.state
        self.checkpoint.save(self.work)

    def compile(self):
        if self.work["spec"]:
            self.spec = InventorySpec.from_dict(self.work["spec"])
            return
        req = self.request
        payload = {"question": req.question, "choices": [{"label": c.label, "text": c.text} for c in req.choices],
                   "public_policy": req.benchmark_policy, "query_scope": req.query_scope,
                   "execution_subtype": req.execution_subtype,
                   "sources": [{"entry_id": s["entry_id"], "allowed": s["allowed"], "available_modalities": s["available_modalities"],
                                "task_context": s["task_context"], "actor_bindings": s["actor_bindings"]} for s in self.sources.values()],
                   "history_cutoff": req.query_time}
        if hasattr(payload["query_scope"], "start_seconds"):
            payload["query_scope"] = [req.query_scope.start_seconds, req.query_scope.end_seconds]
        feedback, errors, calls = None, [], []
        for attempt in range(2):
            key = "compile:" + str(attempt)
            try:
                rec = self.session.call(key, "compile", payload, pool="recovery" if attempt else "compile",
                                        root_id="compile", recovery=bool(attempt), feedback=feedback)
                calls.append(key)
                rec["default_assignments"] = []
                spec = parse_compile(self.session.parse(rec), req, assignments=rec["default_assignments"])
                # Apply only public/default constraints; semantic evidence mappings are kept elsewhere.
                targets = []
                operations = {o.operation_id: o for o in spec.operations}
                required_attributes = {s.set_id: set(s.attribute_keys) for s in spec.sets}
                def require(key, attribute):
                    if key in operations:
                        for child in operations[key].inputs:
                            require(child, attribute)
                    else:
                        required_attributes[key].add(attribute)
                for op in spec.operations:
                    attr = op.group_by.removeprefix("attributes.") if op.op in {"group_count", "argmax_count", "argmin_count"} and op.group_by not in {"category", "value"} else None
                    if attr:
                        for child in op.inputs:
                            require(child, attr)
                for target in spec.sets:
                    candidates = target.candidates
                    relevant = [o for o in spec.operations if target.set_id in o.inputs and o.op in {"membership", "missing_members"}]
                    if relevant and not candidates:
                        candidates = tuple(dict.fromkeys(v for o in relevant for v in o.candidates))
                        if not candidates:
                            candidates = tuple(sorted({c.text for c in req.choices}))
                    targets.append(replace(target, candidates=candidates,
                        attribute_keys=tuple(sorted(required_attributes[target.set_id])) if target.equivalence != "combination" else target.attribute_keys,
                        normalization=req.benchmark_policy.get("normalization", target.normalization),
                        population=req.benchmark_policy.get("population", target.population)))
                spec = replace(spec, sets=tuple(targets), scope=query_scope(req, spec.scope))
                self.spec = InventorySpec.from_dict(spec.to_dict())
                self.work["spec"] = self.spec.to_dict()
                rec["accepted_task"] = self.work["spec"]
                self.session.validation(rec)
                self.save()
                return
            except (ContractError, ProtocolError, StageFailure) as exc:
                if isinstance(exc, StageFailure) and exc.failure["code"] == "prompt_contract_error":
                    raise
                detail = getattr(exc, "errors", error_details(exc))
                rec = self.session.state["calls"].get(key)
                if rec:
                    first = self.session.state["calls"].get("compile:0", {})
                    if attempt and rec.get("raw_response") is not None and rec["raw_response"] == first.get("raw_response"):
                        rec["repeated_invalid_output"] = True
                    feedback = self.session.validation(rec, detail)
                    errors.extend(rec["validation_errors"])
                else:
                    precise = [{**e, "stage": "compile", "call_id": key} for e in detail]
                    errors.extend(precise)
                    feedback = compile_diagnostics(precise)
                feedback = {"errors": feedback, "previous_output": rec.get("raw_response") if rec else None,
                            "original_task": payload, "instruction": "Return the corrected complete task, not an error wrapper."}
        failure = StageFailure("compile", "compile_recovery_failed", "Task compilation did not produce an executable task", errors=errors, call_ids=calls)
        if self.session.state["calls"].get("compile:1", {}).get("repeated_invalid_output"):
            failure.failure["repeated_invalid_output"] = True
        raise failure

    def load_window(self, window):
        source = self.sources[window["entry_id"]]
        cache_key = window["tile_id"]
        if window["kind"] == "video":
            kwargs = {f.name: window[f.name] for f in fields(CoverageTile) if f.name in window}
            batch, archive = self.media.observe_tile(source, CoverageTile(**kwargs))
            self.work["catalog"].update(archive)
            catalog = {f.id: archive[f.id] for f in batch.frames}
            errors = list(batch.errors)
            times = sorted({f.timestamp_seconds for f in batch.frames})
            if not any(window["core"][0] <= t < window["core"][1] for t in times):
                errors.append("no_core_frame")
            max_gap = max((b-a for a, b in itertools.pairwise([window["context"][0], *times, window["context"][1]])), default=0)
            if max_gap > 1.6/window["fps"] + 1e-3:
                errors.append("sampling_density_unmet")
            window["max_gap_sec"] = max_gap
            complete = not errors
        else:
            batch, catalog = None, {}
            if cache_key not in self.work["provider_reads"]:
                self.work["provider_reads"][cache_key] = self.adapters[source["entry_id"]].fetch(tuple(window["context"]), window["kind"])
                self.save()
            read = self.work["provider_reads"][cache_key]
            complete, errors = read["complete"], list(read["issues"])
            for segment in read["items"]:
                ref = "text-" + hashlib.sha256((source["entry_id"] + ":" + segment["segment_id"]).encode()).hexdigest()[:20]
                catalog[ref] = {**segment, "id": ref, "entry_id": source["entry_id"],
                    "history_start": history_time(source, segment["start_sec"]), "alignment_error_sec": read["alignment_error_sec"]}
        for ref, item in catalog.items():
            item["membership_sets"] = []
            for s in self.spec.sets:
                spans = self.work.get("set_spans", {}).get(s.set_id, {}).get(source["entry_id"], source["allowed"])
                if any(a <= item["start_sec"] and item["end_sec"] <= b for a, b in spans):
                    item["membership_sets"].append(s.set_id)
                elif item["kind"] != "frame" and any(a < item["end_sec"] and item["start_sec"] < b for a,b in spans):
                    errors.append("text_query_boundary_gap")
                    complete = False
        self.work["catalog"].update(catalog)
        window["snapshot_required"]={s.set_id:[r for r,m in catalog.items() if m["kind"]=="frame" and in_core(m,window,s.set_id)]
                                     for s in self.spec.sets if s.set_id in window["set_ids"]}
        window.update(input_complete=complete, input_errors=errors, source_refs=list(catalog))
        return batch, catalog

    def wire_input(self, role, targets, window, catalog, *, cards=None, candidate_filter=True):
        aliases, public, counters = {}, {}, {"F": 0, "T": 0}
        temporal = role in {"scope", "identity"} or any(s.predicate_kind != "static" or s.namespace == "task_item" for s in targets)
        for ref, meta in sorted(catalog.items(), key=lambda kv: (kv[1]["start_sec"], kv[0])):
            kind = "F" if meta["kind"] == "frame" else "T"
            counters[kind] += 1
            short = kind + str(counters[kind])
            aliases[short] = ref
            item = {"region": "core" if window["core"][0] <= meta["start_sec"] < window["core"][1] else "context",
                    "sets": meta.get("membership_sets", []), "order": len(public)}
            if meta["kind"] != "frame":
                item.update(kind=meta["kind"], text=meta["text"])
                for name in ("speaker_id", "alignment_status", "alignment_error_sec"):
                    if name in meta:
                        item[name] = meta[name]
            if temporal:
                item.update(start_sec=meta["start_sec"], end_sec=meta["end_sec"], entry_id=meta["entry_id"])
                if meta.get("history_start") is not None:
                    item["history_time"] = meta["history_start"]
            if meta.get("candidate_id"):
                item["candidate_id"] = meta["candidate_id"]
            public[short] = item
        payload = {"question": self.request.question, "sets": [public_target(s) for s in targets], "catalog": public}
        by_op={o.operation_id:o for o in self.spec.operations}
        def leaves(key):
            return set().union(*(leaves(k) for k in by_op[key].inputs)) if key in by_op else {key}
        simultaneous=set().union(*(leaves(o.inputs[0]) for o in self.spec.operations if o.op=="max_simultaneous_count"))
        if role == "discover_candidates" and simultaneous & {s.set_id for s in targets}:
            payload["simultaneous_sets"]=sorted(simultaneous & {s.set_id for s in targets})
            payload["simultaneous_carriers"]={sid: sorted({o.group_by for o in self.spec.operations
                if o.op=="max_simultaneous_count" and sid in leaves(o.inputs[0])}) for sid in payload["simultaneous_sets"]}
            if any(len(axes)>1 for axes in payload["simultaneous_carriers"].values()):
                raise StageFailure("compile","ambiguous_carrier_axes","A simultaneous set needs one carrier definition; use separate sets for distinct carrier units")
        for s in targets:
            if s.candidates and self.work["executors"].get(s.set_id) == "candidate_presence":
                present = {r["candidate"] for r in self.store.state["checks"].values() if r["set"] == s.set_id and r["state"] == "seen" and not r.get("needs_review")} if self.store else set()
                candidate_names = [c for c in s.candidates if not candidate_filter or c not in present]
                payload.setdefault("candidates", {})[s.set_id] = candidate_names
        if temporal:
            payload.update(core=window["core"], context=window["context"])
        if cards is not None:
            payload["existing"] = [{"candidate_id": c["candidate_id"], "set": c["set_id"], "name": c["raw_value"],
                "class": c["actual_class"], "query_value": c.get("query_value"),
                "reported_query_value": c.get("reported_query_value"), "issues": c["issues"],
                "conditions": c["conditions"], "facts": c["facts"]} for c in cards]
            if role=='inspect_existing' and not window.get('source_window_review'):
                # A batched category review has multiple home windows; evidence must
                # not inherit the first candidate's global core/context labels.
                for short, meta in public.items():
                    full=catalog[aliases[short]]
                    meta['core_for']=[c['candidate_id'] for c in cards
                                      if in_core(full,self.work['windows'][c['window_id']],c['set_id'])]
                    meta['region']='member_evidence'
        if any(s.namespace == "task_item" for s in targets) and self.store:
            payload["task_registry"] = [{k: u.get(k) for k in ("update_id", "owner", "task_id", "kind", "item_key", "quantity", "unit")}
                                        for u in list(self.store.history.state["task_updates"].values())[-12:]]
        return payload, aliases

    def row_errors(self, exc, path):
        details = getattr(exc, "errors", error_details(exc))
        return [{**e, "path": path + ("." + e["path"].lstrip("$.") if e.get("path") not in {None, "$"} else "")} for e in details]

    def accept_observation(self, data, role, targets, window, aliases, catalog, transaction, call_id, *, existing=(), recovery=False):
        txn = self.work["transactions"][transaction]
        candidate_only = bool(txn["payload"].get("candidates")) and all(
            self.work["executors"].get(s.set_id) == "candidate_presence" for s in targets)
        regions = []
        if txn['payload'].get('interaction') and isinstance(data,dict) and ('members' in data or 'answers' in data or txn['payload'].get('coverage_review')):
            data = decode_observation(data,txn['payload'],targets,role)
            regions = data.pop('_regions', [])
            # Wire refs are validated before any slot is committed. Keep full provenance.
            regions = [{**r, 'refs':restore_refs(r['refs'], aliases, catalog, 'regions.refs')} for r in regions]
            txn.setdefault('wire_decodes',{})[call_id]={'wire_version':2,'canonical_response':copy.deepcopy(data), 'localized_gaps':regions,
                'rule':'Explicit Q/J judgments and labels mapped to internal fields; no inferred visual facts'}
        data = validate(data, envelope_schema(role, candidate_only=candidate_only))
        if candidate_only:
            if data.get("gaps") or (data.get("coverage", "complete") != "complete" and "input_gaps" not in data):
                fail("coverage", "ambiguous_candidate_coverage",
                     "Candidate coverage is derived from checks and structured input_gaps; free-text absence is not a coverage gap")
            for gap in data.get("input_gaps", []):
                if gap["candidate"] not in txn["payload"]["candidates"].get(gap["set"], []):
                    fail("input_gaps", "unrequested_check", "Input gaps must refer to an explicitly requested candidate")
            data["coverage"] = "complete"
        elif data.get("input_gaps"):
            fail("input_gaps", "wrong_executor", "Structured candidate input gaps belong only to candidate presence checks")
        if role == "inspect_existing" and data.get("records"):
            fail("records", "inspection_creates_duplicates", "Use updates with existing candidate IDs, or explicit new_candidates")
        by_set, errors, committed = {s.set_id: s for s in targets}, [], []
        txn = self.work["transactions"][transaction]
        committed_locals = txn.setdefault("local_ids", {})
        allowed_ids = {c["candidate_id"] for c in existing}
        # Recovery uses original host-assigned slots. Unchanged accepted rows are never re-created.
        invalid_slots = set(txn.get("invalid_slots", []))
        new_invalid = []
        collections = ("records", "updates", "new_candidates", "checks", "task_updates", "snapshots")
        duplicate_ids, seen = set(), set()
        for collection in ("records", "new_candidates"):
            for row in data.get(collection, []):
                ident = row.get("id") if isinstance(row, dict) else None
                if ident in seen:
                    duplicate_ids.add(ident)
                seen.add(ident)
        if sum(len(data.get(k, [])) for k in ("records", "updates", "new_candidates", "task_updates")) > 12:
            fail("records", "candidate_limit", "At most twelve records total; report overflow instead")
        # Reserve IDs for task updates; references to an invalid sibling cannot be committed.
        for collection in collections:
            for index, row in enumerate(data.get(collection, [])):
                slot = f"{transaction}/{collection}/{index}"
                path = f"{collection}[{index}]"
                if recovery and invalid_slots and slot not in invalid_slots:
                    continue
                if slot in txn.get("accepted_slots", []) and not recovery:
                    continue
                try:
                    rejected = self.store.state["quarantined"].get(slot, {})
                    if recovery and collection == "checks" and row is None and any(
                            e.get("code") == "unrequested_check" for e in rejected.get("errors", [])):
                        # Explicit tombstone only for a host-proven inapplicable check. Never absence evidence.
                        self.work.setdefault("discarded_protocol_slots", {})[slot] = rejected
                        self.store.state["quarantined"].pop(slot, None)
                        txn.setdefault("accepted_slots", []).append(slot)
                        committed.append(slot)
                        self.save()
                        continue
                    if not isinstance(row, dict):
                        fail(path, "schema_type", "A row must be an object", "object", type(row).__name__)
                    sid = row.get("set_id" if collection == "task_updates" else "set")
                    if sid not in by_set:
                        fail("set", "unrequested_set", "Use a requested collection ID", list(by_set), sid)
                    target = by_set[sid]
                    if target.namespace == "task_item" and collection != "task_updates":
                        fail("set", "task_update_required", "History membership requires structured plan/completion updates; generic object/check rows cannot prove an empty or completed plan")
                    if collection == "snapshots":
                        if sid not in txn["payload"].get("simultaneous_sets",[]):
                            fail("set","unrequested_snapshot","Only a simultaneous-count input requests a frame census")
                        staged=EvidenceStore(self.spec,self.store.snapshot())
                        key=staged.commit_snapshot(row,target,window,aliases,catalog,slot,call_id,committed_locals)
                        self.store=staged
                    elif collection == "checks":
                        requested = txn["payload"].get("candidates", {}).get(sid, [])
                        if row.get("candidate") not in requested:
                            fail("candidate", "unrequested_check", "Only checks explicitly requested in this input are applicable", requested, row.get("candidate"))
                        if any(g["set"] == sid and g["candidate"] == row["candidate"] for g in data.get("input_gaps", [])) and row.get("state") != "unreadable":
                            fail("state", "candidate_gap_conflict", "A check with an explicit input gap cannot establish presence or absence")
                        review = next((r for r in txn["payload"].get("check_review", [])
                                       if r["set"] == sid and r["candidate"] == row.get("candidate")), None)
                        home = self.work["windows"][review["window_id"]] if review else window
                        key = self.store.commit_check(row, target, home, aliases, catalog, slot, call_id,
                                                      review_key=review["check_key"] if review else None)
                    elif collection == "task_updates":
                        key = self.store.commit_task(row, target, window, self.sources[window["entry_id"]], aliases, catalog, slot, call_id, committed_locals)
                        if key:
                            committed_locals[row["local_id"]] = key
                    else:
                        if txn["payload"].get("check_review"):
                            fail("set", "check_review_only", "Candidate review accepts only the requested checks")
                        if self.work["executors"].get(sid) == "candidate_presence":
                            fail("set", "candidate_checks_required", "Candidate presence sets use checks, not category records")
                        inspection = collection == "updates"
                        key = row.get("candidate_id") if inspection else None
                        if inspection and key not in allowed_ids:
                            fail("candidate_id", "unrequested_candidate", "Inspect only the supplied IDs")
                        if not inspection and row.get("id") in duplicate_ids:
                            fail("id", "duplicate_local_id", "Local object IDs must be unique")
                        home = self.work["windows"].get(self.store.cards[key]["window_id"], window) if inspection else window
                        card = parse_card(row, target, home, aliases, catalog, inspection=inspection)
                        key = self.store.commit_card(card, slot, call_id, candidate_id=key)
                        if not inspection:
                            committed_locals[row["id"]] = key
                    txn.setdefault("accepted_slots", []).append(slot)
                    committed.append(slot)
                    self.save()
                except (ContractError, ProtocolError, ValueError, KeyError) as exc:
                    detail = self.row_errors(exc, path)
                    errors.extend(detail)
                    new_invalid.append(slot)
                    self.store.state["quarantined"][slot] = {"row": row, "errors": detail, "window_id": window["tile_id"], "call_id": call_id}
        for collection, shape in (("distinct_pairs", DISTINCT_SCHEMA), ("coexisting", COEXISTING_SCHEMA)):
            for i, row in enumerate(data.get(collection, [])):
                path, slot = f"{collection}[{i}]", f"{transaction}/{collection}/{i}"
                if slot in txn.get("accepted_slots", []) or (recovery and invalid_slots and slot not in invalid_slots):
                    continue
                try:
                    row = validate(row, shape)
                    pairs = [row] if collection == "distinct_pairs" else [
                        {"left":a,"right":b,"refs":[row["ref"]],"independent_objects":True}
                        for a,b in itertools.combinations(row["ids"],2)]
                    staged = EvidenceStore(self.spec, self.store.snapshot())
                    for pair in pairs:
                        left, right = committed_locals.get(pair["left"]), committed_locals.get(pair["right"])
                        if left is None or right is None:
                            fail(path, "relation_dependency_invalid", "Both local records must be accepted before a relation")
                        staged.accept_relation({"left": left, "right": right, "relation": "DIFFERENT", "basis": "coexistence",
                            "refs": pair["refs"], "facts": "independent objects coexisting in a supplied frame", "independent_objects": True},
                            aliases, catalog, call_id, {left, right})
                    self.store = staged
                    self.store.state["quarantined"].pop(slot, None)
                    txn.setdefault("accepted_slots", []).append(slot)
                    committed.append(slot)
                    self.save()
                except (ContractError, KeyError) as exc:
                    detail = self.row_errors(exc, path)
                    errors.extend(detail)
                    new_invalid.append(slot)
                    self.store.state["quarantined"][slot] = {"row": row, "errors": detail, "window_id": window["tile_id"], "call_id": call_id}
        if recovery and invalid_slots:
            omitted = invalid_slots - set(committed) - set(new_invalid)
            for slot in omitted:
                errors.append(issue(slot, "recovery_slot_missing", "Recovery omitted an invalid slot; do not erase it to imply zero"))
            new_invalid.extend(omitted)
        txn["invalid_slots"] = new_invalid
        txn["response_coverage"] = data["coverage"]
        txn["response_gaps"] = data.get("gaps", [])
        txn['localized_gaps'] = regions
        txn['response_gaps'] += [r['reason']+': '+r['detail'] for r in regions]
        txn["input_gaps"] = data.get("input_gaps", [])
        txn["overflow"] = data.get("overflow", False)
        window["snapshot_gaps"]=[]
        for sid in txn["payload"].get("simultaneous_sets",[]):
            required=set(window.get("snapshot_required",{}).get(sid,[]))
            present={ref for r in self.store.state["snapshots"].values() if r["set_id"]==sid and r["window_id"]==window["tile_id"] for ref in r["frames"]}
            if required-present:
                window["snapshot_gaps"].append({"set_id":sid,"uninspected_frames":sorted(required-present)})
        if data.get("overflow") and not data.get("gaps"):
            txn["response_gaps"] = ["candidate_overflow"]
        if role == "discover_candidates":
            for s in targets:
                if self.work["executors"].get(s.set_id) != "candidate_presence":
                    continue
                for candidate in txn["payload"].get("candidates", {}).get(s.set_id, []):
                    if not any(r["set"] == s.set_id and r["candidate"] == candidate and r["window_id"] == window["tile_id"] for r in self.store.state["checks"].values()):
                        txn["response_gaps"].append("candidate_uninspected:" + candidate)
        for review in txn["payload"].get("check_review", []):
            current = self.store.state["checks"].get(review["check_key"], {})
            if current.get("call_id") not in txn["call_ids"] or not current.get("reviewed"):
                errors.append(issue("checks", "check_review_missing", "Return every requested existing candidate check; empty output cannot retract an earlier judgment"))
        return errors, committed

    def observe(self, window, *, role="discover_candidates", pool="base", cards=None, check_reviews=None):
        transaction = window["tile_id"] + ":" + role
        if transaction in self.work["transactions"] and self.work["transactions"][transaction].get("done"):
            return
        targets = [s for s in self.spec.sets if s.set_id in window["set_ids"]]
        # A presence executor with no remaining Q tasks has nothing to ask the model.
        # Keep the dependency on positive checks; refresh_gaps reopens it after retraction.
        if role=='discover_candidates' and all(self.work['executors'].get(s.set_id)=='candidate_presence' for s in targets):
            empty_payload, _ = self.wire_input(role,targets,window,{})
            if not any(empty_payload.get('candidates',{}).values()):
                window.update(status='complete',input_complete=True,input_errors=[],source_refs=[],gaps=[],overflow=False,
                              host_skipped='positive_evidence_resolves_all_candidates')
                self.work['transactions'][transaction]={'payload':empty_payload,'aliases':{},'catalog':{},
                    'window_id':window['tile_id'],'root_id':window.get('root_id',window['tile_id']),
                    'accepted_slots':[],'invalid_slots':[],'call_ids':[],'done':True,'successful_call':'host:no_pending_tasks',
                    'response_coverage':'complete','response_gaps':[],'overflow':False,'input_gaps':[]}
                self.refresh_gaps(); self.save(); return
        if cards is None or window.get('source_window_review'):
            batch, catalog = self.load_window(window)
        else:
            batch, catalog = self.media.representatives(cards, self.work["catalog"])
            self.work["catalog"].update(catalog)
            window.setdefault("input_complete", bool(catalog))
            window.setdefault("input_errors", [])
        prepared = self.media.prepare(batch) if batch and batch.frames else None
        payload, aliases = self.wire_input(role, targets, window, catalog, cards=cards)
        if check_reviews is not None:
            payload["check_review"] = check_reviews
            payload["candidates"] = {s.set_id: [r["candidate"] for r in check_reviews if r["set"] == s.set_id] for s in targets}
        bind_interaction(payload,targets,role)
        if window.get('review_goal'):
            payload['review_goal'] = window['review_goal']
        if window.get('coverage_review'):
            payload['coverage_review'] = window['coverage_review']
        txn = self.work["transactions"].setdefault(transaction, {"payload": payload, "aliases": aliases,
            "catalog": catalog, "window_id": window["tile_id"], "root_id": window.get("root_id", window["tile_id"]),
            "accepted_slots": [], "invalid_slots": [], "call_ids": []})
        # Frozen original payload/media reference numbering is reused on recovery/resume.
        payload, aliases, catalog = txn["payload"], txn["aliases"], txn["catalog"]
        feedback, all_errors = None, []
        for attempt in range(2):
            key = transaction + ":" + str(attempt)
            try:
                if attempt:
                    feedback = {"errors": diagnostics(all_errors), "invalid_slots": txn["invalid_slots"],
                                "invalid_records": self.recovery_records(txn),
                                "instruction": "Keep original array positions and IDs. Repair every reported slot using this same media. Do not erase invalid slots with empty arrays. Existing accepted slots are ignored on replay; they are not new discoveries. Only a host-proven unrequested checks slot may be null."}
                rec = self.session.call(key, role, payload, pool="recovery" if attempt else pool, targets=targets,
                    prepared=prepared, aliases=aliases, catalog=catalog, root_id=txn["root_id"], recovery=bool(attempt), feedback=feedback)
                if key not in txn["call_ids"]:
                    txn["call_ids"].append(key)
                data = self.session.parse(rec)
                errors, committed = self.accept_observation(data, role, targets, window, aliases, catalog, transaction, key,
                                                            existing=cards or (), recovery=bool(attempt))
                self.session.validation(rec, errors, committed=committed)
                if errors:
                    raise ContractError(errors)
                txn.update(done=True, successful_call=key)
                status = "complete" if window["input_complete"] and txn["response_coverage"] == "complete" and not txn["response_gaps"] and not txn["overflow"] else "partial"
                window.update(status=status, gaps=txn["response_gaps"], overflow=txn["overflow"], failure=None)
                window['localized_gaps'] = txn.get('localized_gaps', [])
                self.refresh_gaps()
                self.save()
                return
            except (ContractError, StageFailure) as exc:
                if isinstance(exc, StageFailure) and exc.failure["code"] == "prompt_contract_error":
                    raise
                errors = getattr(exc, "errors", error_details(exc))
                all_errors.extend(errors)
                rec = self.session.state["calls"].get(key)
                if rec:
                    self.session.validation(rec, errors)
        txn["done"] = True
        failure = StageFailure(role, "observation_recovery_failed", "Window output remains invalid after bounded recovery",
                               errors=all_errors, call_ids=txn["call_ids"], window_id=window["tile_id"])
        if len(txn["call_ids"]) == 2:
            a, b = (self.session.state["calls"][key].get("raw_response") for key in txn["call_ids"])
            if a is not None and a == b:
                failure.failure["repeated_invalid_output"] = True
        window.update(status="failed", failure=failure.failure)
        self.work["failures"].append(failure.failure)
        self.save()

    def recovery_records(self, txn):
        context = []
        for slot in txn["invalid_slots"][:12]:
            row = self.store.state["quarantined"].get(slot, {}).get("row")
            if not isinstance(row, dict):
                continue
            value = {k: row[k] for k in ("id", "candidate_id", "set", "candidate", "name", "class", "conditions", "state", "support", "query_value") if k in row}
            if isinstance(row.get("boxes"), list):
                value["box_count"] = len(row["boxes"])
            context.append({"slot": slot, "previous_fields_for_diagnosis": value})
        return context

    def refresh_gaps(self):
        # Recompute candidate obligations from current evidence, not sticky response flags.
        # Original response and protocol failures remain untouched in transactions.
        positive = {(r["set"], r["candidate"]) for r in self.store.state["checks"].values()
                    if r["state"] == "seen" and not r.get("needs_review")}
        for window in self.work["windows"].values():
            txn = self.work["transactions"].get(window["tile_id"] + ":discover_candidates")
            if txn and txn.get("successful_call") and not window.get("children"):
                audit_id = window.get('coverage_audit_id')
                audit = self.work['transactions'].get(str(audit_id)+':inspect_existing', {})
                cover = audit if audit.get('successful_call') else txn
                window['coverage_source_call'] = cover.get('successful_call')
                window['localized_gaps'] = cover.get('localized_gaps', [])
                gaps = [g for g in cover.get("response_gaps", []) if not g.startswith("candidate_uninspected:")]
                obligations = []
                for sid, candidates in txn["payload"].get("candidates", {}).items():
                    # Include filtered candidates again if an earlier positive is later retracted.
                    target = next(s for s in self.spec.sets if s.set_id == sid)
                    for candidate in target.candidates:
                        if (sid, candidate) in positive:
                            continue
                        local = [r for r in self.store.state["checks"].values()
                                 if r["set"] == sid and r["candidate"] == candidate and r["window_id"] == window["tile_id"]]
                        explicit_gap = any(g["set"] == sid and g["candidate"] == candidate for g in txn.get("input_gaps", []))
                        # A later successful review of this exact input can resolve its old gap.
                        resolved = any(r.get("reviewed") and r["state"] == "not_seen" for r in local)
                        if (explicit_gap and not resolved) or not any(r["state"] == "not_seen" for r in local):
                            obligations.append({"set": sid, "candidate": candidate})
                gaps += ["candidate_uninspected:" + r["candidate"] for r in obligations]
                window["candidate_gaps"] = obligations
                window["gaps"] = list(dict.fromkeys(gaps))
                # Pure candidate checks can resolve a previous partial candidate obligation.
                candidate_only = bool(txn["payload"].get("candidates")) and all(
                    self.work["executors"].get(sid) == "candidate_presence" for sid in window["set_ids"])
                coverage_ok = cover.get("response_coverage") == "complete" or (
                    candidate_only and cover.get("response_coverage") == "partial" and not gaps)
                window["status"] = "complete" if window.get("input_complete") and coverage_ok and not gaps and not txn.get("overflow") else "partial"
            window["membership_gaps"] = [c["candidate_id"] for c in self.store.cards.values() if c["window_id"] == window["tile_id"] and c["membership"] == "unknown"]
            window["category_gaps"] = [c["candidate_id"] for c in self.store.cards.values() if c["window_id"] == window["tile_id"] and c["namespace"] == "semantic_category" and c["query_value"] is None and c["membership"] != "excluded"]
        # Local crop regions partition one source frame. Entire objects strictly inside
        # disjoint core regions cannot be the same spatial instance; no video identity call.
        local = [c for c in self.store.cards.values() if c["membership"] == "accepted"
                 and self.work["executors"].get(c["set_id"]) == "local_instances"]
        for a, b in itertools.combinations(local, 2):
            wa, wb = (self.work["windows"][c["window_id"]] for c in (a, b))
            if wa.get("root_id") != wb.get("root_id") or not wa.get("crop_core") or not wb.get("crop_core") or wa["crop_core"] == wb["crop_core"]:
                continue
            da, db = a["detections"][0], b["detections"][0]
            if da["source_frame_id"] != db["source_frame_id"]:
                continue
            def interior(d, w):
                x,y,r,t = d["bbox"]; l,u,h,v = w["crop_core"]
                return l < x < r < h and u < y < t < v
            if interior(da, wa) and interior(db, wb):
                self.store.add_relation(a["candidate_id"], b["candidate_id"], "different", "disjoint_core_ownership",
                    [da["ref"], db["ref"]], "complete local objects inside disjoint core regions of the same source frame", call_id="host:crop_partition")

    def bind_scopes(self):
        if self.work["scope_done"]:
            return
        scopes = {}
        for s in self.spec.sets:
            scope = s.scope or self.spec.scope
            if scope.get("kind") == "semantic":
                scopes[s.set_id if s.scope else "global"] = scope
        if not scopes:
            self.work["scope_done"] = True
            return
        if not self.work["scope_windows"]:
            locating = replace(self.spec, scope={"kind": "full"}, sets=tuple(replace(s, scope={}) for s in self.spec.sets))
            for tile in make_plan(locating, self.sources, self.config, {}):
                w = asdict(tile)
                w["tile_id"] = "scope_" + w["tile_id"]
                w["scope_only"] = True
                self.work["scope_windows"][w["tile_id"]] = w
            self.save()
        while True:
            w = next((w for w in self.work["scope_windows"].values() if w["status"] == "pending"), None)
            if w is None:
                break
            batch, catalog = self.load_window(w)
            prepared = self.media.prepare(batch) if batch and batch.frames else None
            payload, aliases = self.wire_input("scope", self.spec.sets, w, catalog)
            payload = {k: v for k, v in payload.items() if k in {"catalog", "core", "context"}}
            payload["scopes"] = scopes
            failures, feedback = [], None
            for attempt in range(2):
                key = w["tile_id"] + ":scope:" + str(attempt)
                try:
                    rec = self.session.call(key, "scope", payload, pool="recovery" if attempt else w.get("pool", "base"),
                        targets=self.spec.sets, prepared=prepared, aliases=aliases, catalog=catalog,
                        recovery=bool(attempt), root_id=w.get("root_id", w["tile_id"]), feedback=feedback)
                    data = validate(self.session.parse(rec), envelope_schema("scope"))
                    bindings = []
                    for row in data["bindings"]:
                        if row["scope_id"] not in scopes:
                            fail("scope_id", "unknown_scope", "Only requested semantic scopes may be bound")
                        refs = restore_refs(row["refs"], aliases, catalog, "refs", required=True)
                        a, b = row["interval"]
                        if not (w["core"][0] <= a < b <= w["core"][1]):
                            fail("interval", "scope_outside_input", "Binding must stay in this core; adjacent evidence can extend it in a later window")
                        if not any(a <= catalog[r]["start_sec"] < b for r in refs):
                            fail("refs", "scope_witness_missing", "Binding needs evidence inside its interval")
                        scope = scopes[row["scope_id"]]
                        if scope.get("result_kind") == "frame":
                            t = min(catalog[r]["start_sec"] for r in refs if a <= catalog[r]["start_sec"] < b)
                            a, b = t, min(b, t + 1 / (self.sources[w["entry_id"]]["source_fps"] or 24))
                        bindings.append({"scope_id": row["scope_id"], "entry_id": w["entry_id"], "interval": [a, b],
                                         "evidence_refs": refs, "facts": row["facts"], "call_id": key})
                    # Commit the entire binding response only after all ranges/references pass.
                    for binding in bindings:
                        values = self.work["bindings"].setdefault(binding.pop("scope_id"), [])
                        if binding not in values:
                            values.append(binding)
                    w["status"] = "complete" if data["coverage"] == "complete" and not data.get("gaps") and w["input_complete"] else "partial"
                    self.session.validation(rec)
                    break
                except InputNeedsSplit:
                    a,b = w["core"]
                    remaining = self.config.max_focused_calls - self.session.usage()["calls_by_purpose"]["focused"]
                    if w.get("depth",0) < 2 and b-a > self.config.min_split_sec and remaining >= 2:
                        mid=(a+b)/2
                        children=[]
                        for i,(lo,hi) in enumerate(((a,mid),(mid,b))):
                            key=w["tile_id"]+f".input{i}"
                            self.work["scope_windows"][key]={**copy.deepcopy(w),"tile_id":key,"core":[lo,hi],"context":[lo,hi],
                                "status":"pending","depth":w.get("depth",0)+1,"root_id":w.get("root_id",w["tile_id"]),"pool":"focused"}
                            children.append(key)
                        w.update(status="superseded",children=children)
                    else:
                        w.update(status="partial",gaps=["scope_input_limit"])
                    break
                except (ContractError, StageFailure) as exc:
                    if isinstance(exc, StageFailure) and exc.failure["code"] == "prompt_contract_error":
                        raise
                    errors = getattr(exc, "errors", error_details(exc))
                    failures.extend(errors)
                    record = self.session.state["calls"].get(key)
                    if record:
                        self.session.validation(record, errors)
                    feedback = {"errors": diagnostics(errors)}
                    if attempt:
                        w["status"] = "failed"
                        self.work["failures"].append(StageFailure("scope", "scope_recovery_failed", "Semantic range output invalid", errors=failures, window_id=w["tile_id"]).failure)
            self.save()
            if all(scope.get("selection") == "first" and self.work["bindings"].get(key) for key, scope in scopes.items()):
                break
        for key, scope in scopes.items():
            bindings = sorted(self.work["bindings"].get(key, []), key=lambda b: (b["entry_id"], b["interval"][0]))
            if not bindings:
                self.work["scope_issues"].append("semantic_scope_unbound:" + key)
                continue
            selection = scope.get("selection", "all")
            if selection in {"first", "last"} and len({w["entry_id"] for w in self.work["scope_windows"].values()}) > 1:
                self.work["scope_issues"].append("semantic_order_across_sources_unbound:" + key)
            if selection in {"first", "last"}:
                bindings = [bindings[0 if selection == "first" else -1]]
            self.work["bindings"][key] = bindings
            required = [w for w in self.work["scope_windows"].values() if not w.get("children") and (selection != "first" or any(w["entry_id"] == b["entry_id"] and w["core"][0] <= b["interval"][0] for b in bindings))]
            if any(w["status"] != "complete" for w in required):
                self.work["scope_issues"].append("semantic_scope_search_incomplete:" + key)
        self.work["scope_done"] = True
        self.save()

    def plan(self):
        if self.work["windows"]:
            return
        self.bind_scopes()
        self.work["set_spans"] = {}
        sets = []
        for s in self.spec.sets:
            scope = s.scope or self.spec.scope
            key = s.set_id if s.scope else "global"
            self.work["set_spans"][s.set_id] = {entry: scope_spans(scope, source, self.work["bindings"], key) for entry, source in self.sources.items()}
            for entry, spans in self.work["set_spans"][s.set_id].items():
                if spans and not (set(s.required_modalities) & set(self.sources[entry]["available_modalities"])):
                    self.work["scope_issues"].append("required_modality_unavailable:" + s.set_id + ":" + entry)
            selected = scope
            if scope.get("kind") == "semantic" and scope.get("result_kind") == "frame":
                binding = self.work["bindings"].get(key, [])
                if len(binding) == 1:
                    selected = {"kind": "frame", "timestamp_sec": binding[0]["interval"][0], "entry_ids": [binding[0]["entry_id"]]}
                    s = replace(s, scope=selected)
            self.work["executors"][s.set_id] = executor(s, selected, self.spec.operations)
            sets.append(s)
        # Bound frame scopes only affect the host execution plan, not the frozen compiled task.
        bound = replace(self.spec, sets=tuple(sets))
        for tile in make_plan(bound, self.sources, self.config, self.work["bindings"]):
            row = asdict(tile)
            row.update(root_id=tile.tile_id, pool="base", generation="initial")
            self.work["windows"][tile.tile_id] = row
        self.store = EvidenceStore(self.spec, self.work["inventory"])
        n = len(self.work["windows"]) + sum(w.get("pool", "base") == "base" for w in self.work["scope_windows"].values())
        duration = sum(b-a for s in self.sources.values() for a,b in s["allowed"])
        entity = any(v == "cross_segment_entities" for v in self.work["executors"].values())
        known = not self.work["scope_windows"] and all(v == "local_instances" for v in self.work["executors"].values())
        self.session.configure(max(1, n), entity, duration, known_frame=known)
        self.save()

    def split(self, window, reason, *, dense=False):
        if window.get("children") or window.get("depth", 0) >= 2 or not self.session.can_call("focused"):
            window.update(status="partial", gaps=[reason])
            return False
        remaining = self.config.max_focused_calls - self.session.usage()["calls_by_purpose"]["focused"]
        children = []
        is_frame = all(self.work["executors"].get(k) == "local_instances" for k in window["set_ids"])
        if is_frame or dense:
            if remaining < 4:
                window.update(status="partial", gaps=["spatial_split_budget"])
                return False
            pad = 1000 * self.config.crop_overlap
            regions = [(0,0,500,500),(500,0,1000,500),(0,500,500,1000),(500,500,1000,1000)]
            for i, region in enumerate(regions):
                a,b,c,d=region
                child = {**copy.deepcopy(window), "tile_id": window["tile_id"]+f".crop{i}", "crop_core": list(region),
                         "bbox": [max(0,a-pad), max(0,b-pad), min(1000,c+pad), min(1000,d+pad)]}
                children.append(child)
        else:
            a,b = window["core"]
            if remaining < 2 or b-a <= self.config.min_split_sec:
                window.update(status="partial", gaps=["temporal_split_budget"])
                return False
            mid = (a+b)/2
            for i,(lo,hi) in enumerate(((a,mid),(mid,b))):
                rate = window["fps"] if reason in {"generation_oom", "visual_token_limit", "input_split"} else (6.0 if window.get("depth",0)==0 else 12.0)
                if (hi-lo)*rate > 24 + 1e-6:
                    rate = window["fps"]  # Partition the existing schedule; never reduce its density.
                child = {**copy.deepcopy(window), "tile_id": window["tile_id"]+f".time{i}", "core": [lo,hi], "context": [lo,hi], "fps": rate}
                children.append(child)
        for child in children:
            child.update(status="pending", children=[], failure=None, depth=window.get("depth",0)+1,
                         pool="focused", generation="refinement", root_id=window.get("root_id",window["tile_id"]))
            self.work["windows"][child["tile_id"]] = child
        window.update(children=[c["tile_id"] for c in children], status="superseded", split_reason=reason)
        # Discovery refinement replaces a window inventory; old cards remain in the audit trail.
        for card in self.store.cards.values():
            if card["window_id"] == window["tile_id"]:
                card["superseded_window"] = True
                self.store.state["revisions"].append({"candidate_id": card["candidate_id"], "before": copy.deepcopy(card), "reason": "window_partition"})
                card["membership"] = "excluded"
        self.save()
        return True

    def state(self):
        return reduce_inventory(self.spec, self.store, self.work["windows"], self.request, self.sources, self.work["scope_issues"])

    def sensitive(self, cards, state):
        # Compare conservative outputs under both membership extremes. Do not spend on irrelevant sets.
        final_id = self.spec.output_id or self.spec.operations[-1].operation_id
        by_op = {o.operation_id: o for o in self.spec.operations}
        relevant = set()
        def walk(key):
            if key in by_op:
                for item in by_op[key].inputs:
                    walk(item)
            else:
                relevant.add(key)
        walk(final_id)
        result = []
        for c in cards:
            if c["set_id"] not in relevant or c["membership"] == "excluded":
                continue
            # Incomplete semantic values/core witnesses must be resolved before using
            # current coarse bounds to judge answer sensitivity (otherwise they mask themselves).
            if semantic_review(c):
                result.append(c)
                continue
            if c["membership"] == "unknown" and not any(i.startswith("required_attribute_missing:") for i in c["issues"]):
                outcomes = []
                for membership in ("accepted", "excluded"):
                    imagined = EvidenceStore(self.spec, self.store.snapshot())
                    imagined.cards[c["candidate_id"]]["membership"] = membership
                    projection = reduce_inventory(self.spec, imagined, self.work["windows"], self.request, self.sources, self.work["scope_issues"])
                    final = projection["final"]
                    outcomes.append(json.dumps({k: final.get(k) for k in ("value", "bounds", "groups", "candidate_states", "possible_values", "supported")}, sort_keys=True))
                if outcomes[0] == outcomes[1] and c.get("query_value") is not None:
                    continue
            result.append(c)
        return result

    def choose_review(self, state):
        unresolved = [c for c in self.store.cards.values() if c["membership"] == "unknown" or
                      (c["namespace"] == "semantic_category" and c["query_value"] is None) or
                      any(i.startswith("required_attribute_missing:") for i in c["issues"])]
        # Different accepted canonical names for the same recognized class are a query-mapping conflict.
        categories = [c for c in self.store.cards.values() if c["namespace"] == "semantic_category" and c["membership"] != "excluded"]
        for a,b in itertools.combinations(categories,2):
            if a["set_id"] == b["set_id"] and a["actual_class"].casefold() == b["actual_class"].casefold() and a["query_value"] != b["query_value"]:
                for c in (a,b):
                    if c not in unresolved:
                        unresolved.append(c)
        if map_answer(state, self.spec, self.request)[1] == "option_mapping_conflict":
            for card in categories:
                if card not in unresolved:
                    unresolved.append(card)
            return unresolved  # A poisoned canonical value can look insensitive under the same wrong mapping.
        return self.sensitive(unresolved, state)

    def choose_check_review(self, state):
        """Review candidate assertions, not just member cards. No labels/gold reach the observer."""
        conflict = map_answer(state, self.spec, self.request)[1] == "option_mapping_conflict"
        attempts = self.work.get("check_review_attempts", [])
        selected = []
        for key, row in reversed(list(self.store.state["checks"].items())):
            if key in attempts:
                continue
            # Another trusted positive already settles this ordinary existence candidate.
            # Rechecking a redundant/weak row cannot change its presence result.
            if row["state"] != "seen" and any(other_key != key and other["set"] == row["set"] and other["candidate"] == row["candidate"]
                   and other["state"] == "seen" and not other.get("needs_review")
                   for other_key, other in self.store.state["checks"].items()):
                continue
            uncertain = row["state"] == "unreadable" and (
                row.get("reported_state") == "seen" or row.get("support") in {"related", "uncertain"} or row.get("uncertainties"))
            if uncertain or (conflict and row["state"] == "seen" and not row.get("reviewed")):
                selected.append((key, row))
        # Ambiguous positive claims first; then the latest assertion that created a conflict.
        selected.sort(key=lambda pair: pair[1]["state"] != "unreadable")
        return selected

    def inspect_checks(self, selected):
        home_id = selected[0][1]["window_id"]
        chosen = [(key, row) for key, row in selected if row["window_id"] == home_id][:12]
        reviews = []
        for key, row in chosen:
            self.work.setdefault("check_review_attempts", []).append(key)
            row["needs_review"] = True
            reviews.append({"check_key": key, "set": row["set"], "candidate": row["candidate"],
                            "window_id": home_id, "previous_state": row["state"],
                            "previous_facts": row["facts"], "previous_support": row.get("support"),
                            "reason": "Challenge this claim, not confirm its old status. State the actual visible action/object, then distinguish it from the exact requested candidate. Preparation, associated objects and results are not the action. Identify the decisive visible cue or retract the positive claim."})
        self.work["review_round"] += 1
        window = {**copy.deepcopy(self.work["windows"][home_id]), "tile_id": f"check_review_{self.work['review_round']}",
                  "set_ids": sorted({r["set"] for r in reviews}), "children": [], "status": "pending",
                  "check_reviews": reviews}
        self.work.setdefault("review_windows", {})[window["tile_id"]] = window
        self.save()
        try:
            self.observe(window, role="inspect_existing", pool="qualification", check_reviews=reviews)
        except InputNeedsSplit:
            window.update(status="partial", gaps=["check_review_input_limit"])
            self.save()

    def inspect(self, cards, *, allow_repack=True):
        self.work["review_round"] += 1
        n = self.work["review_round"]
        original = self.work["windows"][cards[0]["window_id"]]
        window = {**copy.deepcopy(original), "tile_id": f"review_{n}", "set_ids": sorted({c["set_id"] for c in cards}),
                  "root_id": original.get("root_id", original["tile_id"]), "children": [], "status": "pending",
                  "candidate_ids": [c["candidate_id"] for c in cards]}
        window['source_window_review'] = any('context_only_member' in c.get('issues', []) for c in cards)
        window['review_goal'] = [{'candidate_id':c['candidate_id'], 'issues':c['issues'],
            'question':'Re-identify the value on the compiled count_unit axis. Separate any scene title or raw caption from the represented kind. Check whether the old value is an umbrella label or alias; correct it from the image and explain the mapping. Verify the original question conditions.'
                       if c['namespace']=='semantic_category' else
                       'Verify this existing candidate in its original core; a context witness alone does not prove membership.'}
            for c in cards]
        self.work.setdefault("review_windows", {})[window["tile_id"]] = window
        self.save()
        try:
            self.observe(window, role="inspect_existing", pool="qualification", cards=cards)
        except InputNeedsSplit:
            window.update(status="partial", gaps=["review_input_limit"])
            self.save()
            if allow_repack and len(cards) > 1:
                window["status"]="repacked"
                midpoint=(len(cards)+1)//2
                for part in (cards[:midpoint],cards[midpoint:]):
                    if self.session.can_call("qualification"):
                        self.inspect(part,allow_repack=False)

    def coverage_action(self):
        choices = []
        for w in self.work['windows'].values():
            if w.get('children') or w['status'] != 'partial' or w.get('split_attempted'):
                continue
            targets = [s for s in self.spec.sets if s.set_id in w['set_ids']]
            short_members = all(s.namespace != 'task_item' and s.predicate_kind == 'static' and
                                self.work['executors'][s.set_id] != 'candidate_presence' for s in targets)
            if short_members and not w.get('localized_gaps') and not w.get('overflow') and not w.get('snapshot_gaps'):
                if w.get('coverage_audit_id'):
                    continue  # Repeating an unlocalized inability claim is not another actionable audit.
                action, cost = 'coverage_audit', 1
            else:
                spatial = bool(w.get('overflow')) or all(self.work['executors'][s.set_id]=='local_instances' for s in targets)
                if w.get('depth',0) >= 2 or (not spatial and w['core'][1]-w['core'][0] <= self.config.min_split_sec):
                    continue
                action, cost = 'localized_refinement', 4 if spatial else 2
            score = (not bool(w.get('localized_gaps')), -len(w.get('category_gaps',[])), w['core'][0], w['tile_id'])
            choices.append((score, w, action, cost))
        if not choices:
            return None
        _,w,action,cost = min(choices,key=lambda x:x[0])
        return w,action,cost

    def audit_coverage(self, original):
        self.work['review_round'] += 1
        window = {**copy.deepcopy(original), 'tile_id':f"coverage_review_{self.work['review_round']}",
                  'status':'pending', 'children':[], 'pool':'focused',
                  'coverage_review':{'window_id':original['tile_id'], 'prior_gaps':original.get('gaps',[]),
                     'known_findings':[{k:c.get(k) for k in ('candidate_id','raw_value','membership','query_value')}
                                       for c in self.store.cards.values() if c['window_id']==original['tile_id'] and c['membership']!='excluded']}}
        original['coverage_audit_id'] = window['tile_id']
        self.work.setdefault('review_windows',{})[window['tile_id']] = window
        self.save()  # Started audit is never granted a fresh allowance on resume.
        try:
            self.observe(window,role='inspect_existing',pool='focused')
        except InputNeedsSplit:
            window.update(status='partial',gaps=['coverage_audit_input_limit'])
        self.refresh_gaps()
        self.save()

    def action(self, kind, targets, operation):
        before = progress_snapshot(self.store,self.work['windows'])
        record = {'kind':kind,'targets':targets,'state':'started','before':before,
                  'calls_before':self.session.usage()['model_calls']}
        self.work.setdefault('resolution_actions',[]).append(record)
        self.save()
        completed = False
        try:
            value = operation()
            completed = True
            return value
        finally:
            after=progress_snapshot(self.store,self.work['windows'])
            record.update(state='returned' if completed else 'interrupted',changed=before!=after,after=after,
                          calls_after=self.session.usage()['model_calls'])
            self.save()

    def remaining_identity_pairs(self):
        graph=self.store.graph()
        attempted={tuple(p) for p in self.store.state['relation_attempts']}
        cards=self.sensitive([c for c in self.store.cards.values() if c['namespace']=='physical_instance' and c['membership']=='accepted'
               and self.work['executors'].get(c['set_id'])=='cross_segment_entities'],self.state())
        result=[]
        for a,b in itertools.combinations(cards,2):
            x,y=(graph['roots'][c['candidate_id']] for c in (a,b))
            pair=tuple(sorted((a['candidate_id'],b['candidate_id'])))
            if x!=y and tuple(sorted((x,y))) not in graph['different'] and pair not in attempted and (a['population'],a['count_unit'])==(b['population'],b['count_unit']):
                result.append(pair)
        return result

    def identity(self, state, shortlist_limit=1):
        graph = self.store.graph()
        candidates = self.sensitive([c for c in self.store.cards.values() if c["namespace"] == "physical_instance" and c["membership"] == "accepted"
                                    and self.work["executors"].get(c["set_id"]) == "cross_segment_entities"], state)
        attempted = {tuple(p) for p in self.store.state["relation_attempts"]}
        pairs = []
        for a,b in itertools.combinations(candidates,2):
            x,y = graph["roots"].get(a["candidate_id"]), graph["roots"].get(b["candidate_id"])
            pair = tuple(sorted([a["candidate_id"],b["candidate_id"]]))
            if x == y or tuple(sorted((x,y))) in graph["different"] or pair in attempted:
                continue
            if (a["population"],a["count_unit"]) != (b["population"],b["count_unit"]):
                continue
            ta = min(self.work["catalog"][r]["start_sec"] for r in a["evidence_refs"])
            tb = min(self.work["catalog"][r]["start_sec"] for r in b["evidence_refs"])
            common={d['source_frame_id'] for d in a['detections']} & {d['source_frame_id'] for d in b['detections']}
            pairs.append(((not bool(common),abs(ta-tb)),pair,a,b))
        if not pairs:
            return False
        pairs.sort(key=lambda v:(v[0],v[1]))
        a,b = pairs[0][2:]
        left, right = [a], [b]
        for _,_,x,y in pairs[1:]:
            if len(right) < shortlist_limit and x["candidate_id"] == a["candidate_id"] and y not in right:
                right.append(y)
            elif len(left) < shortlist_limit and y["candidate_id"] == b["candidate_id"] and x not in left and x not in right:
                left.append(x)
        cards = left+right
        batch,catalog = self.media.representatives(cards,self.work["catalog"])
        self.work["catalog"].update(catalog)
        prepared = self.media.prepare(batch) if batch else None
        original = self.work["windows"][a["window_id"]]
        targets = [s for s in self.spec.sets if s.set_id in {c["set_id"] for c in cards}]
        payload,aliases = self.wire_input("identity",targets,original,catalog,cards=cards)
        payload.update(left=[c["candidate_id"] for c in left],right=[c["candidate_id"] for c in right],
                       previous_relations=[r for r in self.store.state["relations"] if r["left"] in {c["candidate_id"] for c in cards} and r["right"] in {c["candidate_id"] for c in cards}])
        payload["evidence_by_object"] = {
            c["candidate_id"]: [short for short, full in aliases.items()
                               if catalog[full].get("candidate_id") == c["candidate_id"]
                               or (catalog[full].get("source_frame_id") is not None and catalog[full].get("source_frame_id") in {
                                   self.work["catalog"][r].get("source_frame_id") for r in c["evidence_refs"]})]
            for c in cards}
        common_frames={d['source_frame_id'] for d in a['detections']} & {d['source_frame_id'] for d in b['detections']}
        displayed_frames={m.get('source_frame_id') for m in catalog.values()}
        payload['pair_task']={'input_type':'isolated_representatives',
             'available_bases':['uncertain','reidentification','stable_difference']+(['coexistence'] if common_frames & displayed_frames else [])}
        for meta in payload['catalog'].values():
            meta['region']='comparison' # not inherited from the first object's discovery window
        key = "identity_" + hashlib.sha256(json.dumps([payload["left"],payload["right"]]).encode()).hexdigest()[:16]
        root=original.get("root_id", original["tile_id"])
        feedback, errors=[],[]
        for attempt in range(2):
            identifier = key+":"+str(attempt)
            try:
                rec=self.session.call(identifier,"identity",payload,pool="recovery" if attempt else "identity",targets=targets,
                    prepared=prepared,aliases=aliases,catalog=catalog,root_id=root,recovery=bool(attempt),feedback=feedback)
                raw_data = self.session.parse(rec)
                if isinstance(raw_data,dict) and 'decision' in raw_data:
                    raw_data=decode_pair(raw_data,payload)
                if isinstance(raw_data, dict) and isinstance(raw_data.get("relations"), list):
                    for row in raw_data["relations"]:
                        if isinstance(row, dict) and isinstance(row.get("refs"), list) and any(
                                isinstance(ref, str) and ref in payload["left"] + payload["right"] for ref in row["refs"]):
                            fail("relations.refs", "identity_reference_is_object_id",
                                 "left/right identify objects; refs identify supplied image/text evidence",
                                 list(aliases), row["refs"])
                data=validate(raw_data,envelope_schema("identity"))
                # Relation response is staged separately; invalid relations cannot partly change the graph.
                staged=EvidenceStore(self.spec,self.store.snapshot())
                for relation in data["relations"]:
                    if any(ref in {c["candidate_id"] for c in cards} for ref in relation.get("refs", [])):
                        fail("relations.refs", "identity_reference_is_object_id", "Object IDs cannot serve as image/text evidence",
                             list(aliases), relation["refs"])
                    if not ((relation["left"] in payload["left"] and relation["right"] in payload["right"]) or
                            (relation["right"] in payload["left"] and relation["left"] in payload["right"])):
                        fail("relations", "comparison_pair_outside_shortlist", "Only compare the requested left and right groups")
                    staged.accept_relation(relation,aliases,catalog,identifier,{c["candidate_id"] for c in cards})
                for x,y in itertools.product(payload["left"],payload["right"]):
                    staged.state["relation_attempts"].append(sorted([x,y]))
                self.store=staged
                self.session.validation(rec)
                self.save()
                return True
            except InputNeedsSplit:
                if shortlist_limit > 1:
                    return self.identity(state,shortlist_limit=1)
                raise
            except (ContractError, StageFailure) as exc:
                if isinstance(exc, StageFailure) and exc.failure["code"] == "prompt_contract_error":
                    raise
                errors.extend(getattr(exc,"errors",error_details(exc)))
                feedback={"errors":diagnostics(errors)}
                rec=self.session.state["calls"].get(identifier)
                if rec:
                    self.session.validation(rec,errors)
        for x,y in itertools.product(payload["left"],payload["right"]):
            self.store.state["relation_attempts"].append(sorted([x,y]))
        call_ids = [key + ":" + str(i) for i in range(2) if key + ":" + str(i) in self.session.state["calls"]]
        self.work["failures"].append(StageFailure("identity","identity_recovery_failed","Identity evidence remains invalid",errors=errors,
                                               call_ids=call_ids,window_id=original["tile_id"]).failure)
        self.save()
        return True

    def finish(self, state=None, failure=None, budget=False):
        state = state if state is not None else self.state() if self.store else {"results":[],"sets":{},"issues":[]}
        prediction,basis=map_answer(state,self.spec,self.request) if self.spec else (None,"unresolved")
        if failure:
            prediction=None
        status="supported" if prediction is not None else "execution_failed" if failure else "budget_exhausted" if budget else "unresolved"
        if prediction is None and not failure and self.work["failures"]:
            blocking=[f for f in self.work["failures"] if f.get("window_id") not in self.work["windows"] or self.work["windows"][f["window_id"]].get("status") == "failed"]
            if blocking:
                failure=blocking[0]
                status="execution_failed"
        evidence_status = status
        answer_mode = "determined" if prediction is not None else "none"
        answer_detail = None
        if prediction is None and self.spec and self.request.choices:
            from .answering import best_effort
            try:
                prediction, answer_detail = best_effort(self, state)
                if prediction is not None:
                    answer_mode, basis = "best_effort", "best_effort"
            except (BudgetExhausted, StageFailure, InputNeedsSplit) as exc:
                answer_detail = {"status":"unavailable", "reason":str(exc), "failure":getattr(exc,"failure",None)}
        windows=list(self.work["windows"].values())
        card_values=list(self.store.cards.values()) if self.store else []
        trace={"protocol":VERSION,"compiled_task":self.spec.to_dict() if self.spec else None,
               "executors":self.work["executors"],"bindings":self.work["bindings"],"scope_windows":self.work["scope_windows"],
               "transactions":self.work["transactions"],"failures":self.work["failures"],
               "sampling_schedule_completed":state.get("sampling_schedule_completed",False) and not any(w.get('host_skipped') for w in windows),
               "host_skipped_windows":[w['tile_id'] for w in windows if w.get('host_skipped')],
               "uninspected_ranges":[w["tile_id"] for w in windows if w["status"] == "pending"],
               "unreadable_regions":[{"window_id":w["tile_id"],"gaps":w.get("gaps",w.get("input_errors",[]))} for w in windows if w["status"] in {"partial","failed"}],
               "membership_ambiguities":[c["candidate_id"] for c in card_values if c["membership"] == "unknown"],
               "category_ambiguities":[c["candidate_id"] for c in card_values if c["namespace"] == "semantic_category" and c["membership"] != "excluded" and c["query_value"] is None],
               "category_mapping_conflicts":state.get("mapping_conflicts", []),
               "review_windows":self.work.get("review_windows", {}),
               "check_review_attempts":self.work.get("check_review_attempts", []),
               "candidate_check_ambiguities":[{"set":r["set"],"candidate":r["candidate"],"window_id":r["window_id"]}
                    for r in self.store.state["checks"].values() if r.get("needs_review") or r["state"] == "unreadable"] if self.store else [],
               "resolution_obligations":obligations(self.store,self.work["windows"]) if self.store else [],
               "resolution_actions":self.work.get("resolution_actions",[]),
               "stop_detail":self.work.get("stop_detail"),
               "answer_mapping_status":basis, "answer_mode":answer_mode, "evidence_status":evidence_status,
                "best_effort":answer_detail,
               "coverage_policy":"8s/2fps with bounded focused reviews; completion is sampling-policy support, not exhaustive perception"}
        if self.store:
            graph = self.store.graph()
            roots = sorted(graph["groups"])
            trace["identity_ambiguities"] = [[a,b] for a,b in itertools.combinations(roots,2) if (a,b) not in graph["different"]]
        support="supported" if status == "supported" else "unsupported" if status == "execution_failed" or not (card_values or (self.store and self.store.state["checks"])) else "partial"
        unresolved = list(dict.fromkeys(self.work["issues"] + state.get("issues", [])))
        if status != "supported":
            if map_answer(state,self.spec,self.request)[1] == "option_mapping_conflict":
                unresolved.append(basis)
            if trace["membership_ambiguities"]:
                unresolved.append("membership_unresolved")
            if trace["category_ambiguities"]:
                unresolved.append("query_value_unresolved")
            if trace.get("identity_ambiguities"):
                unresolved.append("identity_unresolved")
            if not trace["sampling_schedule_completed"]:
                unresolved.append("sampling_or_coverage_incomplete")
        result=R4Result(prediction,state,{"supported":"complete","unresolved":"evidence_incomplete","budget_exhausted":"budget_limited","execution_failed":"execution_error"}[status],
            support,basis if prediction is not None else "pipeline_failure" if failure else "unresolved",
            self.store.snapshot() if self.store else {},windows,unresolved,
            self.work["catalog"],self.session.usage(),trace,failure,status,answer_mode,evidence_status)
        self.work["result"]=asdict(result)
        self.save()
        return result

    def run(self):
        if self.work["result"] is not None:
            return R4Result(**self.work["result"])
        try:
            self.compile()
            self.store=EvidenceStore(self.spec,self.work["inventory"])
            self.plan()
            for window in self.work.get("review_windows", {}).values():
                if window["status"] == "pending":
                    try:
                        self.observe(window, role="inspect_existing", pool=window.get("pool", "qualification"),
                                     cards=[self.store.cards[k] for k in window["candidate_ids"]] if "candidate_ids" in window else None,
                                     check_reviews=window.get("check_reviews"))
                    except InputNeedsSplit:
                        window.update(status="partial",gaps=["review_input_limit"])
                        self.save()
            while True:
                pending=next((w for w in self.work["windows"].values() if w["status"] == "pending" and not w.get("children")),None)
                if pending is None:
                    break
                try:
                    self.observe(pending,pool=pending.get("pool","base"))
                except InputNeedsSplit as exc:
                    self.split(pending,"input_split")
                if pending.get("overflow") and not pending.get("children"):
                    self.split(pending,"member_overflow",dense=True)
                state=self.state()
                # Basic coverage is only skippable if the operator already has a proof (e.g. positive existence).
                if map_answer(state,self.spec,self.request)[0] is not None:
                    return self.finish(state)
            reviewed={key for w in self.work.get("review_windows", {}).values() for key in w.get("candidate_ids", [])}
            while True:
                state=self.state()
                if map_answer(state,self.spec,self.request)[0] is not None:
                    return self.finish(state)
                check_candidates = self.choose_check_review(state)
                if check_candidates and self.session.can_call("qualification"):
                    self.action("candidate_check",[key for key,_ in check_candidates],lambda:self.inspect_checks(check_candidates))
                    continue
                candidates=[c for c in self.choose_review(state) if c["candidate_id"] not in reviewed]
                if candidates and self.session.can_call("qualification"):
                    # At most six visual representatives; text-only mapping can batch twelve expressions.
                    candidates.sort(key=lambda c:('context_only_member' in c['issues'],not semantic_review(c),c["window_id"],c["candidate_id"]))
                    first=candidates[0]
                    if 'context_only_member' in first['issues']:
                        chosen=[c for c in candidates if c["window_id"]==first['window_id']][:6]
                    else:
                        chosen=[c for c in candidates if 'context_only_member' not in c['issues']][:6]
                    reviewed.update(c["candidate_id"] for c in chosen)
                    self.action("member_identification",[c["candidate_id"] for c in chosen],lambda:self.inspect(chosen))
                    continue
                coverage_action=self.coverage_action()
                remaining=self.session.state['limits']['focused']-self.session.usage()['calls_by_purpose']['focused']
                if coverage_action:
                    gap,kind,cost=coverage_action
                    if remaining>=cost and self.session.can_call('focused'):
                        if kind=='coverage_audit':
                            self.action(kind,[gap['tile_id']],lambda:self.audit_coverage(gap))
                        else:
                            gap['split_attempted']=True
                            def refine():
                                if self.split(gap,'localized_coverage_gap',dense=bool(gap.get('overflow'))):
                                    for child_id in gap['children']:
                                        try:
                                            self.observe(self.work['windows'][child_id],pool='focused')
                                        except InputNeedsSplit:
                                            self.work['windows'][child_id].update(status='partial',gaps=['input_limit_at_refinement'])
                            self.action(kind,[gap['tile_id']],refine)
                        continue
                pairs=self.remaining_identity_pairs()
                if pairs and self.session.can_call('identity'):
                    try:
                        if self.action('identity',[list(p) for p in pairs[:1]],lambda:self.identity(state)):
                            continue
                    except InputNeedsSplit:
                        self.work['issues'].append('identity_input_limit')
                blocked=[]
                if (check_candidates or candidates) and not self.session.can_call('qualification'):
                    blocked.append({'action':'member_or_check_review','pool':'qualification','remaining_targets':len(check_candidates)+len(candidates)})
                if coverage_action and (remaining<coverage_action[2] or not self.session.can_call('focused')):
                    blocked.append({'action':coverage_action[1],'pool':'focused','window_id':coverage_action[0]['tile_id'],'required_calls':coverage_action[2]})
                if pairs and not self.session.can_call('identity'):
                    blocked.append({'action':'identity','pool':'identity','remaining_pairs':len(pairs)})
                self.work['stop_detail']={'code':'action_budget_blocked' if blocked else 'no_actionable_evidence_progress',
                    'blocked_actions':blocked,'total_calls_used':self.session.usage()['model_calls'],
                    'total_calls_limit':self.session.state['limits']['total'],
                    'note':'Only concrete unperformed actions blocked by an allowance count as budget exhaustion; repeated UNKNOWN or unlocalized gaps alone do not.'}
                return self.finish(state,budget=bool(blocked))
        except BudgetExhausted as exc:
            self.work["issues"].append(str(exc))
            self.work["stop_detail"]={"code":"resource_budget_blocked","resource":str(exc)}
            return self.finish(budget=True)
        except StageFailure as exc:
            self.work["failures"].append(exc.failure)
            return self.finish(failure=exc.failure)
        except (ValueError, TypeError, KeyError, ProtocolError) as exc:
            failure=StageFailure("execution",type(exc).__name__,str(exc)).failure
            self.work["failures"].append(failure)
            return self.finish(failure=failure)
