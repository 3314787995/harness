"""Deterministic R7 controller; the same model handles every bounded role."""

from __future__ import annotations

import json
import re
import time
from copy import deepcopy
from dataclasses import asdict

from qwen3vl_agent.models.base import ModelOutput

from .config import R7Config
from .contracts import (
    SCHEMAS,
    validate_candidates,
    validate_execution_choice,
    validate_observation,
    validate_reason,
    validate_task,
    validate_verdict,
)
from .mechanisms import BUILTIN_RULE, audit_candidates, execute_worlds, unique_entailed
from .media import ScopedMedia, sampling, uniform, windows
from .runtime import ModelSession, checkpoint_for, software_fingerprint
from .state import FactStore, World
from .types import BudgetExhausted, ModelFailure, ProtocolError, R7Request, R7Result, digest, plain

DSL_GUIDE = {
    "READ_FACT/SET": "one arg copy; SET params.pack=list/map builds a composite from sourced args, map needs params.keys",
    "ADD": "two or more compatible-unit scalar/interval arguments",
    "COUNT_EVENTS": "complete event-list args, overlap deduplicated; optional params entity_id/predicate/time_window (source-backed expression [start,end]); edge-crossing occurrence is unresolved",
    "SWAP": "two refs; simultaneous in-place exchange",
    "TABLE_LOOKUP": "table ref and key expression",
    "RULE_TRANSITION": "nested transition table, state, event",
    "SORT": "entity->numeric map OR scalar args+params.keys; descending boolean; returns order/intervals/ambiguous",
    "SELECT": "map/list/sort-result and key/index expression; params.top_k=true selects first K of a resolved order",
    "COMPARE": "two arguments and params.relation eq/ne/lt/le/gt/ge/contains",
}


def compact_cells(cells, relevant=(), limit=160):
    ordered = sorted(cells, key=lambda k: (k not in relevant, k))
    selected = {}
    for key in ordered[:limit]:
        cell = deepcopy(cells[key])
        cell["sources"] = cell.get("sources", [])[:8]
        selected[key] = cell
    return {
        "cells": selected,
        "omitted_keys": ordered[limit : limit + 16],
        "omitted_count": max(0, len(ordered) - limit),
    }


def merge_audits(executed, verdict):
    model = {a["label"]: {x["id"]: x for x in a["atoms"]} for a in verdict["assessments"]}
    result = deepcopy(executed)
    for candidate in result:
        for atom in candidate["atoms"]:
            proposed = deepcopy(model[candidate["label"]][atom["id"]])
            if atom["status"] in {"entailed_by_execution", "contradicted"}:
                continue
            # Model language is conditional support, never program entailment/hard elimination.
            if proposed["status"] == "entailed_by_execution":
                proposed["status"] = "supported"
            if proposed["status"] == "contradicted":
                proposed["status"] = "unknown"
            if not proposed["evidence_ids"]:
                proposed["status"] = "unknown"
            atom.update(proposed)
    return result


def choose_gap(gaps, candidates, contract, attempted, mode):
    choices = []
    labels = {c["label"] for c in candidates}
    for gap in gaps:
        if (
            gap["kind"] not in {"perceptual", "binding"}
            or gap["action"] == "none"
            or gap["resolvability"] <= 0
        ):
            continue
        if not set(gap["affects_candidates"]) <= labels or not gap["affects_candidates"]:
            continue
        query = gap["neutral_query"]
        if re.search(
            r"\b(option|candidate|answer|confirm)\b|选项|答案|确认.*正确", query, re.IGNORECASE
        ):
            continue
        if any(c["text"] in query for c in candidates):
            continue
        if gap["window"] is None or not contract.permits_span(gap["window"]):
            continue
        key = digest(gap)
        if key in attempted:
            continue
        if gap["action"] == "crop" and (gap["frame_id"] is None or gap["bbox"] is None):
            continue
        cost = gap["window"][1] - gap["window"][0]
        choices.append((-gap["impact"], -gap["resolvability"], cost, key, gap))
    if not choices:
        return None
    selected = deepcopy(min(choices, key=lambda v: v[:4])[-1])
    return selected


class R7VideoAgent:
    def __init__(self, model, *, config=None):
        self.model = model
        self.config = R7Config.from_mapping(config)

    def load(self):
        self.model.load()

    def unload(self):
        self.model.unload()

    def generate(
        self, messages, *, videos=None, images=None, choices=(), given_interval=None, **kwargs
    ):
        users = [m for m in messages if m.get("role") == "user"]
        if (
            len(videos or []) != 1
            or images
            or not users
            or not isinstance(users[-1]["content"], str)
        ):
            raise ValueError("R7 requires one video, a text question and original choices")
        if given_interval is not None:
            kwargs.setdefault("allowed_scope", given_interval)
        result = self.solve(R7Request(videos[0], users[-1]["content"], choices, **kwargs))
        return ModelOutput(result.text, {"r7": result.to_dict()})

    def solve(self, request):
        start = time.perf_counter()
        cpu_start = time.process_time()
        software = software_fingerprint(self.model, self.config)
        try:
            media = ScopedMedia(request, self.config, software)
            checkpoint = checkpoint_for(request, self.config, media, software)
        except (ValueError, OSError, RuntimeError) as exc:
            return R7Result(
                None,
                "input_error",
                "none",
                str(exc),
                trace={"error_type": type(exc).__name__, "mode": request.mode},
            )
        state = checkpoint.restored or {
            "coverage": [],
            "completed_observations": [],
            "round": 0,
            "actions": [],
            "attempted": [],
            "stage": "compile",
        }
        if "result" in state:
            return R7Result(**state["result"])
        store = FactStore(state.get("facts"))
        try:
            for observation in store.observations:
                media.restore_evidence(observation["evidence"])
        except (ValueError, OSError) as exc:
            return R7Result(
                None,
                "input_error",
                "none",
                "checkpoint_evidence_mismatch",
                trace={"error": str(exc)},
            )

        def save():
            state["facts"] = store.to_dict()
            checkpoint.save(state)

        limits = self.config.budget(media.contract, request)
        session = ModelSession(self.model, self.config, limits, state, save)
        spec, candidates, worlds, assessments, gaps = {}, [], {}, [], []

        def finish(prediction, completion, support, reason, error=None):
            result = R7Result(
                prediction,
                completion,
                support,
                reason,
                spec,
                candidates,
                store.to_dict(),
                [w.to_dict() for w in worlds.values()],
                assessments,
                gaps,
                state["coverage"],
                session.resources(),
                {
                    "mode": request.mode,
                    "software": software,
                    "input_contract": asdict(media.contract),
                    "source_sha256": media.source_hash,
                    "actions": state["actions"],
                    "protocol_visibility": request.target_visibility,
                    "semantic_visibility": spec.get("target_visibility", "unresolved"),
                    "single_candidate": len(request.choices) == 1,
                    "fact_replay": bool(request.facts_input),
                    "elapsed_seconds": state.get("elapsed_seconds", 0)
                    + time.perf_counter()
                    - start,
                    "cpu_seconds": time.process_time() - cpu_start,
                    "error": error,
                },
            )
            state["result"] = asdict(result)
            save()
            return result

        def observe(key, window, *, tail=False, dense=False, gap=None, frame_count=None):
            if key in state["completed_observations"]:
                return state.get("progress", {}).get(key, False)
            times, fps = sampling(
                window, self.config, tail=tail, dense=dense, source_fps=media.metadata.source_fps
            )
            if frame_count is not None:
                times, fps = uniform(*window, frame_count), None
            batch = media.extract(window, times, fps=fps)
            if gap and gap["action"] == "crop":
                frame = media.frame(gap["frame_id"])
                if not window[0] <= frame.timestamp_seconds <= window[1]:
                    raise ProtocolError("crop anchor outside selected window")
                # One registered full frame and its crop retain spatial reference.
                batch.frames = (frame, media.crop(frame, gap["bbox"]))
                batch.crops = {
                    batch.frames[1].id: media.catalog[batch.frames[1].id]["crop_transform"]
                }
                batch.ordered = False
            prepared = media.prepare(batch)
            evidence = media.evidence(prepared, window)
            # Only neutral task slots and existing identities reach the observer, never options/worlds.
            payload = {
                "slots": spec["slots"],
                "targets": spec["targets"],
                "entities": store.entities,
                "window": window,
                "neutral_query": gap["neutral_query"] if gap else "Read the neutral fact slots.",
                "frames": [
                    {
                        "frame_id": k,
                        "source_seconds": e["timestamp_seconds"],
                        **({"text": e["text"]} if "text" in e else {}),
                    }
                    for k, e in evidence.items()
                ],
            }
            value, call_id = session.call(
                key,
                "observe",
                payload,
                lambda v: validate_observation(v, spec, evidence, store.entities),
                prepared,
                evidence,
            )
            progress = store.ingest(value, evidence, call_id)
            state["coverage"].append(media.coverage(batch, True))
            state["completed_observations"].append(key)
            state.setdefault("progress", {})[key] = progress
            save()
            return progress

        try:
            if len(request.choices) == 1:
                return finish(
                    request.choices[0].label,
                    "single_candidate",
                    "noninformative",
                    "official_single_candidate",
                )
            if request.mode == "B0":
                value = self._direct(request, media, session, state, save, None)
                return finish(value["prediction"], "complete", "conditional", "direct_baseline")
            context = {
                "question": request.question,
                "protocol": asdict(media.contract),
                "query_time": request.query_time,
                "query_scope": request.query_scope,
                "operation_hint": request.execution_subtype,
                "declared_target_visibility": request.target_visibility,
            }
            if request.facts_input and "replay_loaded" not in state:
                spec, candidates = self._replay(request, media, store)
                state.update(spec=spec, candidates=candidates, replay_loaded=True)
                save()
            if state.get("replay_loaded"):
                spec, candidates = state["spec"], state["candidates"]
            else:
                spec, _ = session.call(
                    "compile", "compile", context, lambda v: validate_task(v, request)
                )
                if request.mode == "B1":
                    value = self._direct(request, media, session, state, save, spec)
                    return finish(
                        value["prediction"], "complete", "conditional", "focused_direct_baseline"
                    )
                compiled, _ = session.call(
                    "candidates",
                    "candidates",
                    {**context, "task": spec, "options": [asdict(c) for c in request.choices]},
                    lambda v: validate_candidates(v, request),
                )
                candidates = compiled["candidates"]
            if not request.facts_input:
                initial = windows(media.contract, self.config)
                for i, window in enumerate(initial):
                    if session.remaining <= 3:
                        state["initial_coverage_limited"] = True
                        break
                    observe(
                        f"initial:{i}",
                        window,
                        tail="S1" in spec["mechanisms"] and i == len(initial) - 1,
                    )
            while True:
                pending = state.get("pending_refinement")
                if pending and pending["key"] in state["completed_observations"]:
                    if state["progress"][pending["key"]]:
                        state["round"] = pending["round"] + 1
                        state.pop("pending_refinement")
                        save()
                    else:
                        worlds = {}
                        for snapshot in state["last_worlds"]:
                            w = World(snapshot["id"], snapshot["fact_version"], snapshot["cells"])
                            for name in ("transactions", "invalidated", "execution", "issues"):
                                setattr(w, name, snapshot[name])
                            worlds[w.id] = w
                        assessments, gaps = state["last_assessments"], state["last_verdict"]["gaps"]
                        return finish(
                            state["last_verdict"]["prediction"],
                            "complete",
                            "conditional",
                            "no_new_information",
                        )
                round_id = state["round"]
                relevant = {s["id"] for s in spec["slots"]} | {
                    a["key"] for c in candidates for a in c["atoms"]
                }
                facts = compact_cells(store.current, relevant)
                proposal, _ = session.call(
                    f"reason:{round_id}",
                    "reason",
                    {
                        "task": spec,
                        "candidates": candidates,
                        "facts": facts,
                        "builtin_rules": [BUILTIN_RULE],
                        "operators": DSL_GUIDE,
                        "mode": request.mode,
                    },
                    lambda v: validate_reason(
                        v,
                        spec,
                        candidates,
                        self.config.max_program_steps,
                        self.config.max_latent_branches,
                    ),
                )
                worlds = execute_worlds(
                    spec, candidates, proposal, store, self.config, operators=request.mode != "B2"
                )
                assessments = audit_candidates(candidates, worlds)
                certain = unique_entailed(assessments, spec["query_operator"])
                if certain:
                    return finish(
                        certain, "complete", "entailed_by_execution", "unique_execution_result"
                    )
                payload, valid_ids = self._verification_payload(
                    spec, candidates, worlds, assessments, state["coverage"], request.question
                )
                terminal = session.remaining <= 1

                def validate(v, ids=valid_ids, checks=assessments):
                    validate_verdict(v, candidates, ids)
                    validate_execution_choice(v, checks, spec["query_operator"])

                verdict, _ = session.call(
                    f"verify:{round_id}",
                    "final" if terminal else "verify",
                    payload,
                    validate,
                    terminal=terminal,
                )
                assessments = merge_audits(assessments, verdict)
                gaps = verdict["gaps"]
                state["last_verdict"] = verdict
                state["last_worlds"] = [w.to_dict() for w in worlds.values()]
                state["last_assessments"] = assessments
                save()
                gap = choose_gap(gaps, candidates, media.contract, state["attempted"], request.mode)
                if (
                    request.mode in {"B2", "B3"}
                    or gap is None
                    or round_id >= self.config.max_refinements
                    or session.remaining <= 3
                ):
                    reason = (
                        "no_useful_legal_observation"
                        if gap is None
                        else "refinement_cap"
                        if round_id >= self.config.max_refinements
                        else "budget_stop"
                        if session.remaining <= 3
                        else "ablation_stop"
                    )
                    return finish(
                        verdict["prediction"],
                        "budget_limited" if reason == "budget_stop" else "complete",
                        "conditional",
                        reason,
                    )
                selected = deepcopy(gap)
                if gap["action"] == "observe":
                    a, b = selected["window"]
                    max_duration = (self.config.max_frames_per_call - 1) / min(
                        self.config.fine_motion_fps,
                        media.metadata.source_fps or self.config.fine_motion_fps,
                    )
                    if b - a > max_duration:
                        center = (a + b) / 2
                        selected["window"] = [center - max_duration / 2, center + max_duration / 2]
                matched_count = None
                if request.mode == "B4-uniform":
                    target_times, _ = sampling(
                        selected["window"],
                        self.config,
                        dense=True,
                        source_fps=media.metadata.source_fps,
                    )
                    matched_count = (
                        2
                        if gap["action"] == "crop"
                        else len(media.extract(selected["window"], target_times).frames)
                    )
                    all_windows = windows(media.contract, self.config)
                    selected.update(
                        window=list(all_windows[round_id % len(all_windows)]),
                        action="observe",
                        frame_id=None,
                        bbox=None,
                        neutral_query="Read the original neutral slots across the allowed interval.",
                    )
                key = f"refine:{round_id}"
                if not any(a["round"] == round_id for a in state["actions"]):
                    state["actions"].append(
                        {"round": round_id, "gap": selected, "matched_frame_count": matched_count}
                    )
                state["pending_refinement"] = {"round": round_id, "key": key}
                save()
                progress = observe(
                    key,
                    selected["window"],
                    dense=request.mode != "B4-uniform",
                    gap=selected,
                    frame_count=matched_count,
                )
                state["attempted"].append(digest(gap))
                if not progress:
                    return finish(
                        verdict["prediction"], "complete", "conditional", "no_new_information"
                    )
                state["round"] += 1
                state.pop("pending_refinement", None)
                save()
        except BudgetExhausted as exc:
            if candidates:
                if not worlds:
                    worlds = {"factual": World("factual", store.version, store.current)}
                    assessments = audit_candidates(candidates, worlds)
                payload, valid_ids = self._verification_payload(
                    spec, candidates, worlds, assessments, state["coverage"], request.question
                )
                try:
                    verdict, _ = session.call(
                        "terminal",
                        "final",
                        payload,
                        lambda v: validate_verdict(v, candidates, valid_ids),
                        terminal=True,
                    )
                    gaps = verdict["gaps"]
                    validate_execution_choice(verdict, assessments, spec["query_operator"])
                    assessments = merge_audits(assessments, verdict)
                    return finish(
                        verdict["prediction"], "budget_limited", "forced_choice", str(exc)
                    )
                except (BudgetExhausted, ModelFailure, ProtocolError) as terminal_error:
                    return finish(
                        None, "execution_error", "none", "terminal_unavailable", str(terminal_error)
                    )
            return finish(
                None, "execution_error", "none", "budget_before_candidate_compilation", str(exc)
            )
        except (ModelFailure, ProtocolError, ValueError, OSError, KeyError, TypeError) as exc:
            return finish(None, "execution_error", "none", type(exc).__name__, str(exc))

    def _verification_payload(self, spec, candidates, worlds, assessments, coverage, question):
        valid_ids = set()
        packed = []
        relevant = {a["key"] for c in candidates for a in c["atoms"]}
        factual = worlds.get("factual")
        for world in worlds.values():
            cells = {k: v.to_dict() for k, v in world.cells.items()}
            if factual is not None and world.id != "factual":
                cells = {
                    k: v for k, v in cells.items() if k in relevant or factual.get(k).to_dict() != v
                }
            limit = 96 if world.id == "factual" else max(12, 96 // max(1, len(worlds) - 1))
            compact = compact_cells(cells, relevant, limit=limit)
            for key, cell in compact["cells"].items():
                valid_ids.add(f"{world.id}:{key}")
                valid_ids.update(cell["sources"])
            packed.append(
                {
                    "id": world.id,
                    **compact,
                    "inherits_factual": world.id != "factual",
                    "transactions": [
                        {
                            "intervention_ids": t["intervention_ids"],
                            "writes": {k: c["value"] for k, c in t["writes"].items()},
                            "invalidated": t["invalidated"][:32],
                        }
                        for t in world.transactions
                    ],
                    "execution": world.execution[-64:],
                }
            )
        # Budget may expire before program compilation; still supply grounded observed facts below.
        return {
            "question": question,
            "atom_status_semantics": "truth of option propositions; apply the question's negation/least operator when selecting",
            "task": spec,
            "candidates": candidates,
            "worlds": packed,
            "program_assessments": assessments,
            "coverage": coverage[-16:],
        }, valid_ids

    def _direct(self, request, media, session, state, save, spec):
        history = []
        intervals = media.contract.allowed_time_intervals
        # A direct per-window baseline for long/disjoint inputs, explicitly reported as hierarchical.
        for i, window in enumerate(windows(media.contract, self.config)):
            if session.remaining <= 1 and history:
                break
            times, fps = sampling(
                window,
                self.config,
                tail=bool(
                    spec
                    and "S1" in spec["mechanisms"]
                    and i == len(windows(media.contract, self.config)) - 1
                ),
                source_fps=media.metadata.source_fps,
            )
            if spec and spec["anchors"]:
                anchors = [
                    a["time_span"]
                    for a in spec["anchors"]
                    if a["time_span"] is not None and media.contract.permits_span(a["time_span"])
                ]
                if anchors:
                    a, b = anchors[0]
                    if window[0] <= a < b <= window[1]:
                        times = sorted(
                            set(
                                times[: self.config.initial_frames // 2]
                                + sampling((a, b), self.config)[0][
                                    : self.config.initial_frames // 2
                                ]
                            )
                        )
            batch = media.extract(window, times, fps=fps)
            prepared = media.prepare(batch)
            evidence = media.evidence(prepared, window)
            payload = {
                "question": request.question,
                "options": [asdict(c) for c in request.choices],
                "protocol": asdict(media.contract),
                "subtitles": media.read_subtitles(window),
            }

            def validate(v):
                if v["prediction"] not in {c.label for c in request.choices}:
                    raise ProtocolError("invalid direct label")

            value, _ = session.call(
                f"direct:{i}",
                "direct",
                payload,
                validate,
                prepared,
                evidence,
                terminal=len(windows(media.contract, self.config)) == 1,
            )
            history.append(value)
            entry = media.coverage(batch, True)
            if entry not in state["coverage"]:
                state["coverage"].append(entry)
            save()
        if len(history) == 1:
            return history[0]
        value, _ = session.call(
            "direct:combine",
            "direct",
            {
                "question": request.question,
                "options": [asdict(c) for c in request.choices],
                "permitted_intervals": intervals,
                "window_reports": history,
            },
            validate,
            terminal=True,
        )
        return value

    def _replay(self, request, media, store):
        from pathlib import Path

        import jsonschema

        artifact = json.loads(Path(request.facts_input).read_text(encoding="utf-8"))
        if artifact.get("software") != media.model_fingerprint:
            raise ProtocolError("fact replay model/processor/software fingerprint mismatch")
        expected = {
            "question": request.question,
            "options": [asdict(c) for c in request.choices],
            "scope": asdict(media.contract),
            "source_sha256": media.source_hash,
        }
        if artifact.get("identity") != plain(expected):
            raise ProtocolError("fact replay question/options/scope/source mismatch")
        spec, candidates = artifact["task_spec"], artifact["candidates"]
        jsonschema.validate(spec, SCHEMAS["compile"])
        jsonschema.validate({"candidates": candidates}, SCHEMAS["candidates"])
        validate_task(spec, request)
        validate_candidates({"candidates": candidates}, request)
        for observation in artifact["facts"]["observations"]:
            evidence = observation["evidence"]
            for e in evidence.values():
                if (
                    not media.contract.permits(e["timestamp_seconds"])
                    or e["scope_hash"] != media.contract.fingerprint
                ):
                    raise ProtocolError("fact replay evidence outside scope")
                if e.get("modality") == "video":
                    t = e["timestamp_seconds"]
                    allowed = next(
                        s for s in media.contract.allowed_time_intervals if s[0] <= t <= s[1]
                    )
                    batch = media.extract(allowed, [t])
                    original = batch.frames[0]
                    box = e["view_box"]
                    width, height = e["source_size"]
                    if box != [0, 0, width, height]:
                        original = media.crop(
                            original,
                            [box[0] / width, box[1] / height, box[2] / width, box[3] / height],
                        )
                    if (
                        media.catalog[original.id]["pixel_sha256"] != e["pixel_sha256"]
                        or original.id != e["id"]
                    ):
                        raise ProtocolError("fact replay pixels/PTS differ from permitted source")
                elif e.get("modality") == "subtitle":
                    if not any(
                        s["id"] == e["id"] and s["text"] == e["text"] for s in media.subtitles
                    ):
                        raise ProtocolError("fact replay subtitle mismatch")
                else:
                    raise ProtocolError("unsupported replay modality")
            jsonschema.validate(observation["value"], SCHEMAS["observe"])
            validate_observation(observation["value"], spec, evidence, store.entities)
            store.ingest(observation["value"], evidence, observation["call_id"])
        if store.to_dict() != artifact["facts"]:
            raise ProtocolError("fact replay derived store differs from observation history")
        return spec, candidates
