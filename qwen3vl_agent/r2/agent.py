"""One query-driven controller for all eight R2 subtypes."""

from __future__ import annotations

import copy
import json
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

from qwen3vl_agent.models.base import ModelOutput
from qwen3vl_agent.r1.media import MediaBatch

from .config import R2Config
from .context import fit_final_payload, spread
from .contracts import certify_final, validate_observation, validate_query
from .evidence import assessment_policy
from .media import R2Media
from .observation import (
    ObservationSpec,
    identity_requirements,
    normalize_identity_handoff,
    validate_identity_handoff,
)
from .planning import initial_spans, make_windows, sample_times, select_action
from .reduce import gap, reduce_operation, reduce_query
from .runtime import ModelSession, new_checkpoint, stable_key
from .state import StateStore
from .types import BudgetExhausted, InputContract, ProtocolError, R2Request, R2Result


def external_navigation(request, contract):
    """Existing transcripts only. Drop entire cues crossing an access boundary."""
    from qwen3vl_agent.r4.providers import read_external_file

    output = []
    for kind in ("subtitle", "asr"):
        path = getattr(request, kind + "_path")
        if not path:
            continue
        if Path(path).suffix.lower() == ".jsonl":
            rows = [
                json.loads(line)
                for line in Path(path).read_text(encoding="utf-8-sig").splitlines()
                if line.strip()
            ]
            segments = [
                {
                    "start_sec": r["start_sec"],
                    "end_sec": r["end_sec"],
                    "text": r["text"],
                    "kind": kind,
                    "segment_id": str(r.get("segment_id", i)),
                }
                for i, r in enumerate(rows)
            ]
        else:
            file = SimpleNamespace(path=path, kind=kind, alignment_error_sec=None)
            segments = [asdict(s) for s in read_external_file(file, request.video_id)]
        for s in segments:
            if contract.permits_span((s["start_sec"], s["end_sec"])):
                output.append(s)
    return output


class R2VideoAgent:
    def __init__(self, model, *, config=None, media=None):
        self.model = model
        self.config = R2Config.from_mapping(config)
        self.media = media or R2Media(self.config)

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
            raise ValueError("R2 requires one video and a text question")
        if given_interval is not None:
            kwargs.setdefault("allowed_scope", given_interval)
        result = self.solve(R2Request(videos[0], users[-1]["content"], choices=choices, **kwargs))
        return ModelOutput(result.text, {"r2": result.to_dict()})

    def solve(self, request: R2Request):
        metadata = self.media.probe(request.video_path)
        contract = InputContract.resolve(request, metadata.duration_seconds)
        checkpoint = new_checkpoint(
            request, self.config, self.model, self.media.source_digest(request.video_path)
        )
        state = checkpoint.restored or {
            "coverage": [],
            "notes": [],
            "actions": [],
            "jobs": {},
            "completed": [],
            "required": [],
            "resolved": {},
            "final_round": 0,
        }
        if "result" in state:
            return R2Result(**state["result"])
        store = StateStore(state.get("store"))
        self.media.catalog.update(state.get("media_catalog", {}))

        def save():
            state["store"] = store.to_dict()
            state["media_catalog"] = self.media.catalog
            checkpoint.save(state)

        session = ModelSession(
            self.model, self.config, request.budget.capped(self.config.budget), state, save
        )
        context = {
            "question": request.question,
            "input_contract": asdict(contract),
            "query_scope": request.query_scope,
            "operation_hint": request.execution_subtype,
        }
        query = state.get("query")
        final, derived, failure = state.get("final_response"), {}, None
        try:
            if query is None:
                intent, _ = session.call(
                    "compile_intent",
                    "compile_intent",
                    context,
                    validator=lambda v: validate_query(
                        v, hint=request.execution_subtype, question=request.question
                    ),
                )
                if request.choices:
                    query, _ = session.call(
                        "compile_discriminants",
                        "compile_discriminants",
                        {
                            **context,
                            "intent": intent,
                            "options": [asdict(c) for c in request.choices],
                        },
                        validator=lambda v: validate_query(
                            v,
                            previous=intent,
                            hint=request.execution_subtype,
                            question=request.question,
                        ),
                    )
                else:
                    query = intent
                state["query"] = query
                save()
            state["observation_spec"] = ObservationSpec.from_query(query).to_dict()
            if "windows" not in state:
                self._plan(request, query, contract, metadata, state, store, session, save)
            for window in state["windows"]:
                if state.get("identity_resolved_early"):
                    break
                if window["id"] not in state["completed"]:
                    self._observe(
                        window, request, query, contract, metadata, state, store, session, save
                    )
                if self._identity_complete(request, query, state, store):
                    save()
                    break
            derived = reduce_query(
                query,
                store,
                self.config,
                state["required"],
                state["coverage"],
                state.get("query_frame_time", request.query_time),
            )
            save()
            if state.get("pending_final_gap"):
                derived["gaps"].append(state["pending_final_gap"])
                derived["sufficient"] = False
            while (
                not derived["sufficient"] and len(state["actions"]) < self.config.max_refinements
            ) or any(not a.get("complete") for a in state["actions"]):
                if not self._repair(
                    derived["gaps"], request, query, contract, metadata, state, store, session, save
                ):
                    break
                derived = reduce_query(
                    query,
                    store,
                    self.config,
                    state["required"],
                    state["coverage"],
                    state.get("query_frame_time", request.query_time),
                )
                save()
        except (BudgetExhausted, ProtocolError, RuntimeError, ValueError, OSError) as exc:
            failure = f"{type(exc).__name__}: {exc}"
            state["notes"].append(failure)
            if query:
                derived = reduce_query(
                    query,
                    store,
                    self.config,
                    state["required"],
                    state["coverage"],
                    state.get("query_frame_time", request.query_time),
                )
        try:
            while state["final_round"] < 2:
                final = self._finish(request, query, derived, contract, state, store, session)
                state["final_response"] = final
                state["final_round"] += 1
                state["pending_final_gap"] = final["recheck"]
                save()
                if (
                    not final["recheck"]
                    or len(state["actions"]) >= self.config.max_refinements
                    or session.budget.terminal_call_reserve - session.summary()["terminal_calls"]
                    < 2
                    or session.budget.max_model_calls - session.summary()["model_calls"] < 2
                    or query is None
                    or state.get("observer_protocol_failed")
                ):
                    break
                if not self._repair(
                    [final["recheck"]],
                    request,
                    query,
                    contract,
                    metadata,
                    state,
                    store,
                    session,
                    save,
                ):
                    break
                derived = reduce_query(
                    query,
                    store,
                    self.config,
                    state["required"],
                    state["coverage"],
                    state.get("query_frame_time", request.query_time),
                )
                save()
        except (BudgetExhausted, ProtocolError, RuntimeError, ValueError, OSError) as exc:
            state["notes"].append(f"terminal: {type(exc).__name__}: {exc}")
            terminal_receipts = [c for c in state["calls"] if c["role"] == "final"]
            if not isinstance(exc, BudgetExhausted) or (
                terminal_receipts and terminal_receipts[-1].get("validation_error")
            ):
                # A rejected terminal response cannot inherit an earlier provisional answer.
                final = None
        unresolved = [*derived.get("gaps", []), *state["notes"]]
        if final:
            unresolved.extend(final["unresolved"])
            if final["recheck"]:
                unresolved.append(final["recheck"])
        supported = bool(final and derived.get("sufficient") and not unresolved)
        if request.choices and final:
            assessment = next(a for a in final["assessments"] if a["label"] == final["prediction"])
            supported &= assessment["status"] == "supported" and bool(assessment["evidence_ids"])
        result = R2Result(
            final["prediction"] if final else None,
            "complete" if supported else "partial" if final else "failed",
            "supported" if supported else "partial" if store.observations else "unsupported",
            value_state=derived,
            state_store=store.to_dict(),
            coverage_manifest=state["coverage"],
            unresolved_items=unresolved,
            evidence_refs=final["evidence_ids"] if final else [],
            option_assessments=final["assessments"] if final else [],
            resources=session.summary(),
            trace={
                "query": query,
                "contract": asdict(contract),
                "calls": state["calls"],
                "actions": state["actions"],
                "media_catalog": self.media.catalog,
                "failure": failure,
                "observation_spec": state.get("observation_spec"),
                "preflight_failures": state.get("preflight_failures", []),
                "final_context": state.get("final_context", []),
            },
        )
        state["result"] = asdict(result)
        save()
        return result

    def _identity_complete(self, request, query, state, store):
        if not all(o["op"] == "identity_at_time" for o in query["operations"]):
            return False
        query_time = state.get("query_frame_time", request.query_time)
        operations = [
            reduce_operation(o, query, store, self.config, query_time) for o in query["operations"]
        ]
        if any(o["status"] != "supported" for o in operations):
            return False
        paths = [
            sorted((o["value"]["query_time"], o["value"]["reveal"]["timestamp"]))
            for o in operations
        ]
        # Do not clip these paths to permissions: an unreadable gap must prevent propagation.
        result = reduce_query(query, store, self.config, paths, state["coverage"], query_time)
        if result["sufficient"]:
            state.update(required=paths, identity_resolved_early=True)
            return True
        return False

    def _locate(self, key, description, request, contract, state, session, save, spans=None):
        if key in state["resolved"]:
            return state["resolved"][key]
        frontier = list(spans or contract.allowed_time_intervals)
        external = external_navigation(request, contract)
        for level in range(self.config.locator_levels):
            proposed = []
            for i, span in enumerate(frontier[:4]):
                times = [
                    span[0] + (span[1] - span[0]) * j / (self.config.locator_frames - 1)
                    for j in range(self.config.locator_frames)
                ]
                batch = self.media.extract(request.video_path, span, times, contract)
                batch.ordered = False
                prepared = self.media.prepare(batch)
                frames = {
                    f"F{j + 1:02d}": self.media.catalog[f.id] for j, f in enumerate(prepared.frames)
                }
                payload = {
                    "description": description,
                    "allowed_span": span,
                    "frames": self._frame_descriptors(frames),
                    "external_navigation": [
                        s for s in external if span[0] <= s["start_sec"] and s["end_sec"] <= span[1]
                    ][:64],
                }

                def validate(value, frames=frames):
                    for c in value["candidates"]:
                        if c["start_frame"] not in frames or c["end_frame"] not in frames:
                            raise ProtocolError("locator referenced an unshown frame")
                        if (
                            frames[c["start_frame"]]["timestamp_seconds"]
                            > frames[c["end_frame"]]["timestamp_seconds"]
                        ):
                            raise ProtocolError("locator reversed the interval")
                    return value

                save()
                result, _ = session.call(
                    f"locate:{key}:{level}:{i}", "locate", payload, prepared, validate
                )
                for candidate in result["candidates"]:
                    a = frames[candidate["start_frame"]]["timestamp_seconds"]
                    b = frames[candidate["end_frame"]]["timestamp_seconds"]
                    padding = max((span[1] - span[0]) / (self.config.locator_frames - 1), 0.5)
                    proposed.extend(
                        contract.intersect((max(span[0], a - padding), min(span[1], b + padding)))
                    )
            if not proposed:
                state["resolved"][key] = []
                save()
                return []
            frontier = list(dict.fromkeys(tuple(s) for s in proposed))[:4]
            if all(
                b - a <= self.config.core_sec + 2 * self.config.context_sec for a, b in frontier
            ):
                break
        state["resolved"][key] = frontier
        save()
        return frontier

    def _plan(self, request, query, contract, metadata, state, store, session, save):
        if query["scope"]["kind"] == "semantic":
            self._locate(
                "scope", query["scope"]["description"], request, contract, state, session, save
            )
        for anchor in query["anchors"]:
            if anchor["kind"] == "semantic":
                self._locate(
                    anchor["id"], anchor["description"], request, contract, state, session, save
                )
        spans = initial_spans(query, request, contract, state["resolved"])
        state["query_scope_intervals"] = spans
        if all(o["op"] == "identity_at_time" for o in query["operations"]):
            # A historical query may use a later reveal within the immutable read contract.
            spans = list(contract.allowed_time_intervals)
        endpoint = all(op["op"] == "endpoint_delta" for op in query["operations"])
        if endpoint:
            anchor_spans = []
            for anchor in query["anchors"]:
                if anchor["kind"] == "semantic":
                    anchor_spans += state["resolved"].get(anchor["id"], [])
                else:
                    t = (
                        anchor["time"]
                        if anchor["kind"] == "time"
                        else spans[0][0]
                        if anchor["kind"] == "start"
                        else spans[-1][1]
                    )
                    anchor_spans += contract.intersect((max(0, t - 0.5), t + 0.5))
            if not query["anchors"] and spans:
                anchor_spans = [
                    (spans[0][0], min(spans[0][1], spans[0][0] + 1)),
                    (max(spans[-1][0], spans[-1][1] - 1), spans[-1][1]),
                ]
            spans = anchor_spans
        if not spans:
            store.add_gap(gap("localization", "Required semantic scope/anchor was not found"))
        for key, candidates in state["resolved"].items():
            if len(candidates) != 1:
                description = (
                    query["scope"]["description"]
                    if key == "scope"
                    else next((a["description"] for a in query["anchors"] if a["id"] == key), key)
                )
                store.add_gap(
                    gap(
                        "localization",
                        description or "Required query target is not located",
                        anchor_key=key,
                        candidate_count=len(candidates),
                    )
                )
        if endpoint:
            windows = [
                {
                    "id": f"endpoint_{i:05d}",
                    "core": list(s),
                    "span": list(s),
                    "fps": self.config.fps,
                    "endpoint": self.config.endpoint_frames,
                    "kind": "observation",
                }
                for i, s in enumerate(spans)
            ]
        else:
            windows = make_windows(spans, contract, self.config, fast=query["fast_motion"])
        state.update(windows=windows, required=spans)
        save()

    @staticmethod
    def _frame_descriptors(frames):
        return [
            {
                "frame_id": alias,
                "source_seconds": m["timestamp_seconds"],
                "source_size": m["source_size"],
                "view_box": m["view_box"],
            }
            for alias, m in frames.items()
        ]

    def _observe(
        self, window, request, query, contract, metadata, state, store, session, save, feedback=None
    ):
        times, fps = sample_times(window, metadata.source_fps)
        shared = [
            self.media.frame(self.media.catalog[fid]["source_frame_id"])
            for c in state["coverage"][-2:]
            if c["window_id"] != window["id"] and c.get("processed", c["completed"])
            for fid in c["frame_ids"]
            if fid in self.media.catalog
            and window["span"][0]
            <= self.media.catalog[fid]["timestamp_seconds"]
            <= window["span"][1]
        ]
        shared = sorted({f.id: f for f in shared}.values(), key=lambda f: f.timestamp_seconds)
        shared = shared if len(shared) <= 2 else [shared[0], shared[-1]]
        if (
            request.query_time is not None
            and window["span"][0] <= request.query_time <= window["span"][1]
        ):
            times.append(request.query_time)
        batch = self.media.extract(
            request.video_path, window["span"], times, contract, fps=fps, anchors=shared
        )
        if (
            request.query_time is not None
            and batch.frames
            and window["span"][0] <= request.query_time <= window["span"][1]
        ):
            nearest = min(batch.frames, key=lambda f: abs(f.timestamp_seconds - request.query_time))
            if abs(nearest.timestamp_seconds - request.query_time) <= 1 / (
                metadata.source_fps or 1
            ):
                state["query_frame_time"] = nearest.timestamp_seconds
        # A fixed crop is derived from the original full-frame sequence, never recentered per frame.
        if window.get("bbox"):
            batch = self.media.fixed_crop(batch, window["bbox"])
        coverage = self.media.coverage(batch, window["id"], False)
        coverage.update(
            core=window["core"], kind="observation", overlap_frame_ids=[f.id for f in shared]
        )
        if window.get("endpoint") and state["coverage"]:
            old_ids = state["coverage"][-1]["frame_ids"]
            connection = [
                self.media.frame(fid)
                for fid in old_ids[:2]
                if contract.permits(self.media.catalog[fid]["timestamp_seconds"])
            ]
            frames = {f.id: f for f in (*connection, *batch.frames)}
            batch = MediaBatch(
                batch.span,
                tuple(sorted(frames.values(), key=lambda f: f.timestamp_seconds)),
                batch.requested_fps,
                False,
                batch.crops,
            )
        prepared = self.media.prepare(batch)
        if not prepared.frames:
            raise ProtocolError("observer has no permitted frames")
        frames = {f"F{i + 1:02d}": self.media.catalog[f.id] for i, f in enumerate(prepared.frames)}
        handoff = store.handoff({m["source_frame_id"] for m in frames.values()})
        # If a fixed query anchor is between windows, the latest entity descriptions are context only.
        if not handoff["entities"]:
            handoff["entities"] = list(store.entities.values())[-16:]
        known_nodes = {e["node_id"] for e in handoff["entities"]}
        spec = ObservationSpec.from_query(query)
        payload = {
            "window_id": window["id"],
            "core": window["core"],
            "span": window["span"],
            "frames": self._frame_descriptors(frames),
            "targets": query["targets"],
            "slots": query["slots"],
            "anchors": query["anchors"],
            "query_time": request.query_time,
            "handoff": handoff,
            "observation_spec": spec.to_dict(),
            "observation_task": "Read the specified measurements in order, including core boundaries and changes. Explain every unreadable slot.",
            "max_records": self.config.max_records,
        }
        if not any(handoff.values()):
            payload.pop("handoff")
        requirements = identity_requirements(query, handoff, frames)
        if requirements:
            payload["identity_requirements"] = requirements
        if feedback:
            payload["recheck_context"] = feedback
        # Preserve attempted media for terminal review and interrupted-call recovery.
        coverage.update(
            processed=False, completed=False, protocol_status="pending", media_kind=prepared.kind
        )
        previous_coverage = next(
            (c for c in state["coverage"] if c["window_id"] == window["id"]), None
        )
        if previous_coverage is None:
            state["coverage"].append(coverage)
        else:
            previous_coverage.update(coverage)
            coverage = previous_coverage
        save()

        def validate(value):
            value, _ = spec.normalize(value, window["span"], frames)
            value = normalize_identity_handoff(value, requirements)
            validate_observation(
                value,
                query,
                frames,
                known_nodes,
                self.config.max_records,
                handoff["previous_observations"],
                spec,
            )
            validate_identity_handoff(value, requirements)
            # Validate correction/association transactions before committing any state,
            # so reference failures use the same-media format repair budget.
            trial = StateStore(store.to_dict())
            trial.ingest(window["id"], value, frames, query, coverage, "validation")
            return value

        try:
            value, call_id = session.call(
                "observe:" + window["id"],
                "observe",
                payload,
                prepared,
                validate,
            )
        except ProtocolError as exc:
            coverage.update(protocol_status="failed", protocol_error=str(exc))
            state["observer_protocol_failed"] = True
            save()
            raise
        usable_slots = {
            r["slot_id"]
            for r in value["records"]
            if r["visibility"] == "visible"
            and r["basis"] == "visual_observation"
            and not spec.record_missing(r)
        }
        coverage.update(
            processed=True,
            protocol_status="accepted",
            completed=bool(
                value["complete"]
                and not value["gaps"]
                and usable_slots == {t["slot_id"] for t in spec.tasks}
            ),
            model_reported_complete=value["complete"],
            usable_slot_ids=sorted(usable_slots),
            validation_basis="program_contract_checks",
        )
        store.ingest(window["id"], value, frames, query, coverage, call_id)
        state["completed"].append(window["id"])
        save()
        return value

    def _repair(self, gaps, request, query, contract, metadata, state, store, session, save):
        if state.get("observer_protocol_failed"):
            return False
        # Resume the uncommitted tail of a scheduled repair before choosing any new work.
        pending = next((a for a in state["actions"] if not a.get("complete")), None)
        # Observation IDs/revision counters alone are not semantic progress. Identical
        # evidence values and candidate outcomes must not unlock the same reread forever.
        latest = store.derived[-1]["result"] if store.derived else {}
        semantic = [
            {
                "op": o["op"],
                "status": o["status"],
                "value": {
                    k: v
                    for k, v in o["value"].items()
                    if k not in {"reveal", "association_dependencies"}
                },
            }
            for o in latest.get("operations", [])
        ]
        revision = stable_key(
            {
                "operations": semantic,
                "explanations": sorted(
                    {
                        (g["kind"], g.get("slot_id", ""), g["description"])
                        for g in gaps
                        if g["kind"] not in {"coverage", "protocol"}
                    }
                ),
            }
        )
        action = pending or select_action(
            gaps,
            state["required"],
            contract,
            [a["signature"] for a in state["actions"]],
            revision,
            self.config,
        )
        if not action:
            state["notes"].append("no_progress: no new permitted repair action")
            return False
        if pending is None:
            state["actions"].append(action)
            save()
        if action["action"] == "relocate":
            key = "relocation:" + stable_key(action["signature"])
            spans = self._locate(
                key,
                action["gap"]["description"],
                request,
                contract,
                state,
                session,
                save,
                action["spans"],
            )
            if not spans:
                action["complete"] = True
                save()
                return False
            action["location_resolved"] = len(spans) == 1
            action["windows"] = make_windows(spans, contract, self.config, prefix=key)
            if action["gap"].get("anchor_key") == "scope" and len(spans) == 1:
                state["required"] = spans
                for previous in store.windows.values():
                    a, b = previous["coverage"]["span"]
                    previous["query_relevant"] = any(a < y and x < b for x, y in spans)
            elif not state["required"]:
                state["required"] = spans
        new_gaps = []
        all_complete = True
        for window in action["windows"]:
            if window["id"] in state["completed"]:
                continue
            window["bbox"] = action.get("bbox")
            value = self._observe(
                window,
                request,
                query,
                contract,
                metadata,
                state,
                store,
                session,
                save,
                feedback={
                    "gap": {
                        k: v
                        for k, v in action["gap"].items()
                        if k in {"kind", "description", "span", "slot_id", "bbox"}
                    },
                    "previous_observations": [
                        {
                            k: r[k]
                            for k in (
                                "id",
                                "slot_id",
                                "timestamp",
                                "visibility",
                                "value",
                                "description",
                                "source_point",
                                "source_reference_point",
                                "rotation_type",
                                "orientation_angle",
                            )
                            if k in r
                        }
                        for r in store.records(
                            [action["gap"]["slot_id"]] if action["gap"].get("slot_id") else None
                        )
                    ][-8:],
                    "needed_measurements": [
                        t
                        for t in ObservationSpec.from_query(query).to_dict()["tasks"]
                        if not action["gap"].get("slot_id")
                        or t["slot_id"] == action["gap"]["slot_id"]
                    ],
                    "observation_change": {
                        "action": action["action"],
                        "span": window["span"],
                        "fps": window["fps"],
                        "bbox": window.get("bbox"),
                        "phase_shifted": window.get("shifted", False),
                    },
                },
            )
            new_gaps += value["gaps"]
            all_complete &= value["complete"]
        if (
            all_complete
            and action.get("location_resolved", True)
            and not any(g["kind"] == action["gap"]["kind"] for g in new_gaps)
        ):
            ids = [
                g["id"]
                for g in store.gaps
                if not g["resolved"]
                and g["kind"] == action["gap"]["kind"]
                and (
                    g.get("id") == action["gap"].get("id")
                    or (
                        (
                            gap_span := g.get("span")
                            or store.windows.get(g.get("window_id"), {})
                            .get("coverage", {})
                            .get("core")
                        )
                        and any(a <= gap_span[0] and gap_span[1] <= b for a, b in action["spans"])
                    )
                )
            ]
            store.close_gaps(ids, action["windows"][-1]["id"])
        action["complete"] = True
        save()
        return True

    def _finish(self, request, query, derived, contract, state, store, session):
        # Supply raw frames from the most fragile gap plus the start/end and transitions.
        available = [c for c in state["coverage"] if c["frame_ids"]]
        critical_times = []
        for operation in derived.get("operations", []):
            value = operation["value"]
            for sequence in value.get("sequences", {}).values():
                critical_times += [
                    x for item in sequence for x in item["change_bracket"] if x is not None
                ]
            if "points" in value and "raw_deltas" in value:
                ds = value["raw_deltas"]
                critical_times += [
                    value["points"][i][0] for i in range(1, len(ds)) if ds[i] * ds[i - 1] < 0
                ]
            if value.get("query_time") is not None:
                critical_times.append(value["query_time"])
        critical_times += [sum(g["span"]) / 2 for g in derived.get("gaps", []) if g.get("span")]
        priorities = []
        for t in critical_times:
            matching = [c for c in reversed(available) if c["span"][0] <= t <= c["span"][1]]
            if matching:
                priorities.append((matching[0], t))
        if available:
            priorities += [
                (available[0], available[0]["span"][0]),
                (available[-1], available[-1]["span"][1]),
            ]
        cited, raw_windows = [], []
        for coverage, center in priorities:
            if len(cited) >= self.config.max_frames_per_call:
                break
            ids = [
                fid
                for fid in coverage["frame_ids"]
                if fid in self.media.catalog
                and contract.permits(self.media.catalog[fid]["timestamp_seconds"])
            ]
            if not ids:
                continue
            nearest = min(
                range(len(ids)),
                key=lambda i: abs(self.media.catalog[ids[i]]["timestamp_seconds"] - center),
            )
            count = min(16, self.config.max_frames_per_call - len(cited))
            start = max(0, min(nearest - count // 2, len(ids) - count))
            local = ids[start : start + count]
            # Repeated gaps often point at exactly the same snippet. Do not
            # append its metadata again when no new raw frame will be shown.
            if not any(fid not in cited for fid in local):
                continue
            raw_windows.append(
                {
                    "window_id": coverage["window_id"],
                    "frame_ids": local,
                    "source_frame_ids": [
                        self.media.catalog[fid]["source_frame_id"] for fid in local
                    ],
                    "continuous_local_sampling": True,
                }
            )
            cited += [fid for fid in local if fid not in cited]
        prepared = None
        if cited:
            from qwen3vl_agent.p01.types import TimeSpan

            frames = tuple(
                sorted((self.media.frame(fid) for fid in cited), key=lambda f: f.timestamp_seconds)
            )
            lo, hi = frames[0].timestamp_seconds, frames[-1].timestamp_seconds
            prepared = self.media.prepare(
                MediaBatch(TimeSpan(lo, max(lo + 1e-6, hi)), frames, ordered=False)
            )
            try:
                session.check(prepared, terminal=True)
            except BudgetExhausted:
                state["notes"].append("terminal_raw_media_unavailable_due_to_budget")
                prepared = None
        source_ids = {self.media.catalog[fid]["source_frame_id"] for fid in cited}
        focused = [r for r in store.observations if r["source_frame_id"] in source_ids]
        evidence = spread(
            list(
                {
                    r["id"]: r
                    for r in [*focused, *store.observations[:8], *store.observations[-8:]]
                }.values()
            ),
            64,
        )
        # Full raw coordinates stay in trace; the final model sees compact derived features.
        compact = copy.deepcopy(derived)
        for operation in compact.get("operations", []):
            for key in (
                "points",
                "ordered_path",
                "raw_deltas",
                "filtered_changes",
                "local_displacement_rates",
            ):
                operation["value"].pop(key, None)
            operation["evidence_ids"] = operation["evidence_ids"][-64:]
        payload = {
            "question": request.question,
            "options": [asdict(c) for c in request.choices],
            "derived": compact,
            "observations": evidence,
            "unresolved": state["notes"],
            "raw_windows": raw_windows,
            "raw_frames": [
                {
                    "frame_id": f"F{i + 1:02d}",
                    "source_frame_id": self.media.catalog[f.id]["source_frame_id"],
                    "view_frame_id": f.id,
                    "view_box": self.media.catalog[f.id]["view_box"],
                    "source_seconds": f.timestamp_seconds,
                }
                for i, f in enumerate(prepared.frames if prepared else ())
            ],
        }

        payload["assessment_policy"] = assessment_policy(query, payload)
        try:
            payload, diagnostic = fit_final_payload(payload, session.budget.max_text_chars_per_call)
        except BudgetExhausted as exc:
            state.setdefault("preflight_failures", []).append(
                {
                    "role": "final",
                    "reason": str(exc),
                    "model_invoked": False,
                }
            )
            session.save()
            raise
        state.setdefault("final_context", []).append(diagnostic)
        if diagnostic.get("mode") == "best_effort":
            note = "terminal_best_effort: compact evidence view; certification unavailable"
            if note not in state["notes"]:
                state["notes"].append(note)
        session.save()

        def validate(value):
            return certify_final(value, payload)

        result, _ = session.call(
            f"final:{state['final_round']}", "final", payload, prepared, validate
        )
        return result
