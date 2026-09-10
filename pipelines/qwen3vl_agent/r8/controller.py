"""One controller, one frozen model instance, deterministic execution and bounded repairs."""

from __future__ import annotations

import itertools
import json
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

from qwen3vl_agent.models.base import ModelOutput
from qwen3vl_agent.p01.types import TimeSpan
from qwen3vl_agent.r1.media import MediaBatch

from .adapters import Adapters
from .config import MODEL, REVISION, R8Config
from .contracts import validate
from .evidence import EvidenceStore, numeric_in_span
from .matcher import match
from .media import ScopedMedia, coverage_record, sampling, windows
from .query_ir import Executor
from .runtime import ModelSession, checkpoint_for, software_fingerprint
from .solver import solve_constraints
from .types import (
    BudgetExhausted,
    Defect,
    ModelFailure,
    ModelingError,
    ProtocolError,
    R8Request,
    R8Result,
    digest,
    plain,
)
from .units import IntervalQuantity, Quantity, encode, number

MODE_SETTINGS = {
    "A": {
        "direct": True,
        "model_math": False,
        "constraints": False,
        "repair": "none",
        "reread": False,
    },
    "B": {
        "direct": False,
        "model_math": True,
        "constraints": False,
        "repair": "none",
        "reread": False,
    },
    "C": {
        "direct": False,
        "model_math": False,
        "constraints": False,
        "repair": "none",
        "reread": False,
    },
    "D": {
        "direct": False,
        "model_math": False,
        "constraints": True,
        "repair": "none",
        "reread": False,
    },
    "E": {
        "direct": False,
        "model_math": False,
        "constraints": True,
        "repair": "uniform",
        "reread": False,
    },
    "F": {
        "direct": False,
        "model_math": False,
        "constraints": True,
        "repair": "directed",
        "reread": False,
    },
    "G": {
        "direct": False,
        "model_math": False,
        "constraints": True,
        "repair": "directed",
        "reread": True,
    },
}


def defect(kind, detail, **kwargs):
    return asdict(Defect(kind, detail, **kwargs))


def validate_task(task, request, store):
    validate(task, "compile")
    ids = [s["id"] for s in task["slots"]]
    if len(ids) != len(set(ids)):
        raise ProtocolError("duplicate task slot")
    precision = task["precision"]
    if precision["kind"] != "exact":
        text = precision["question_span"]
        if not text or text not in request.question:
            raise ProtocolError("approximation requires an exact question span")
        import re

        if not re.search(
            r"closest|nearest|approximately|approximate|round|最接近|近似|四舍五入|保留",
            text,
            re.IGNORECASE,
        ):
            raise ProtocolError("approximation not requested in source span")
    test = EvidenceStore(request.question, store.contract, deepcopy(store.state))
    for given in task["givens"]:
        test.append(given, origin=given["origin"], support="question_given")


def refresh_refs(program, store):
    """A reading revision retains identity/role/snapshot; only that exact reference is advanced."""

    def visit(v):
        if isinstance(v, list):
            return [visit(x) for x in v]
        if isinstance(v, dict):
            return {k: visit(x) for k, x in v.items()}
        if isinstance(v, str) and v in store.state["variables"]:
            return store.state["current"][store.state["variables"][v]["id"]]
        return v

    return visit(program)


class R8VideoAgent:
    def __init__(self, model, config=None):
        self.model, self.config = model, R8Config.from_mapping(config)
        path = str(getattr(model, "model_path", MODEL))
        if path != MODEL and not Path(path).is_dir():
            raise ProtocolError(
                "R8 fixes Qwen3-VL-8B-Instruct; a local pinned snapshot directory is also accepted"
            )
        revision = getattr(model, "revision", None)
        if revision is not None and revision != REVISION:
            raise ProtocolError("R8 model revision mismatch")
        if type(model).__module__ == "qwen3vl_agent.models.qwen3vl" and revision != REVISION:
            raise ProtocolError("the real Qwen wrapper must explicitly pin the R8 revision")

    def load(self):
        self.model.load()

    def unload(self):
        self.model.unload()

    def generate(self, messages, *, videos=None, images=None, choices=(), **kwargs):
        if images or not videos or len(videos) != 1 or not isinstance(videos[0], str):
            raise ProtocolError("R8 requires exactly one source video path")
        if (
            len(messages) != 1
            or messages[0].get("role") != "user"
            or not isinstance(messages[0].get("content"), str)
        ):
            raise ProtocolError("R8 generate accepts one original user question string")
        result = self.solve(R8Request(videos[0], messages[0]["content"], choices=choices, **kwargs))
        return ModelOutput(result.text, {"r8": result.to_dict()})

    def solve(self, request):
        if isinstance(request, dict):
            request = R8Request(**request)
        run = Controller(self.model, self.config, request)
        return run.run()


class Controller:
    def __init__(self, model, config, request):
        self.model, self.config, self.request = model, config, request
        self.settings = MODE_SETTINGS[request.mode]
        self.software = software_fingerprint(model, config)
        self.media = ScopedMedia(request, config, self.software)
        self.checkpoint = checkpoint_for(request, config, self.media, self.software)
        self.state = self.checkpoint.restored or {}
        for key, default in (
            ("store", {}),
            ("adapters", {}),
            ("coverage", []),
            ("observations", {}),
            ("actions", []),
            ("executions", []),
            ("reread_keys", []),
            ("extra_rounds", 0),
            ("recompiles", 0),
        ):
            self.state.setdefault(key, deepcopy(default))
        self.store = EvidenceStore(request.question, self.media.contract, self.state["store"])
        self.adapters = Adapters(self.store, self.state["adapters"])
        limits = {
            key: min(getattr(config, key), getattr(request, key) or getattr(config, key))
            for key in ("max_model_calls", "max_unique_frames", "max_seconds")
        }
        limits["reserve_fallback"] = request.require_choice
        self.session = ModelSession(model, config, limits, self.state, self.save)
        self.task = self.state.get("task")
        if self.store.state["evidence"]:
            self.media.restore_evidence(self.store.state["evidence"])

    def save(self):
        self.checkpoint.save(self.state)

    def observation_payload(self, packet, query=None):
        # This whitelist deliberately omits question answers, options, old values and past predictions.
        identities = [
            {k: row[k] for k in ("id", "actor_id", "scope", "round_id", "start_time", "end_time")}
            for row in self.adapters.state["attempts"].values()
        ]
        return {
            "slots": self.task["slots"],
            "scope": self.task["scope"],
            "snapshot": self.task["snapshot"],
            "target_description": self.task["target"],
            "coverage_need": self.task["coverage_need"],
            "neutral_query": query
            or "Locate and transcribe the requested variables; preserve unreadable candidates.",
            "known_attempt_identities": identities,
            "evidence": {k: {p: v for p, v in e.items() if p != "path"} for k, e in packet.items()},
        }

    def observe(self, key, batch, *, core=None, query=None, reread=False):
        if key in self.state["observations"]:
            return self.state["observations"][key]
        if self.session.elapsed() >= self.session.limits["max_seconds"]:
            raise BudgetExhausted("global wall time cap")
        prepared = self.media.prepare(batch)
        window = (batch.span.start_seconds, batch.span.end_seconds)
        packet = self.media.evidence(prepared, window)
        self.store.present(packet)
        # Save the exact pre-observation payload: resumed calls must not see newer event identities.
        pending = self.state.setdefault("observation_inputs", {})
        pending.setdefault(key, self.observation_payload(packet, query))
        self.save()

        def validator(value):
            test = EvidenceStore(
                self.request.question, self.media.contract, deepcopy(self.store.state)
            )
            test.ingest(value, packet, key, reread=reread)
            adapter = Adapters(test, deepcopy(self.adapters.state))
            adapter.ingest(value, packet, key)
            for context in value["requested_context"]:
                if context["frame_id"]:
                    test.sources([context["frame_id"]], packet=packet)
                if context["window"] and not self.media.contract.permits_span(context["window"]):
                    raise ProtocolError("requested context exceeds permission")

        value, _call_id = self.session.call(
            key, "reread" if reread else "observe", pending[key], validator, prepared, packet
        )
        self.store.ingest(value, packet, key, reread=reread)
        self.adapters.ingest(value, packet, key)
        for context in value["requested_context"]:
            if context["frame_id"] in packet:
                context["frame_id"] = packet[context["frame_id"]]["id"]
        self.state["observations"][key] = value
        if core:
            self.state["coverage"].append(coverage_record(batch, core, window, value["discovery"]))
        self.save()
        return value

    def base_plans(self):
        if "base_plans" in self.state:
            return self.state["base_plans"]
        plans = []
        contract = self.media.contract
        if self.request.fixed_frames:
            for window in contract.allowed_time_intervals:
                times = sorted(
                    {t for t in self.request.fixed_frames if window[0] <= t <= window[1]}
                )
                for start in range(0, len(times), self.config.initial_frames):
                    plans.append(
                        {
                            "window": window,
                            "core": None,
                            "times": times[start : start + self.config.initial_frames],
                            "fps": None,
                        }
                    )
        elif self.task["plan"] == "range" and not self.settings["direct"]:
            for plan in windows(contract, self.config, self.request.query_scope):
                # Split over-large windows rather than reducing the requested sampling density.
                a, b = plan["context"]
                maximum_span = (self.config.max_frames_per_call - 1) / self.config.scan_fps
                if b - a > maximum_span:
                    raise ProtocolError(
                        "core/context configuration exceeds per-call density; reduce core_seconds"
                    )
                times, fps = sampling(plan["context"], self.config, fps=self.config.scan_fps)
                plans.append(
                    {"window": plan["context"], "core": plan["core"], "times": times, "fps": fps}
                )
        else:
            for window in (
                contract.intersect(self.request.query_scope)
                if self.request.query_scope
                else contract.allowed_time_intervals
            ):
                times, fps = sampling(window, self.config)
                anchors = [
                    a["time"]
                    for a in self.task["anchors"]
                    if a["time"] is not None and window[0] <= a["time"] <= window[1]
                ]
                if (
                    self.request.query_time is not None
                    and window[0] <= self.request.query_time <= window[1]
                ):
                    anchors.append(self.request.query_time)
                if anchors:
                    times = sorted({window[0], window[1], *anchors, *times})
                for start in range(0, len(times), self.config.initial_frames):
                    plans.append(
                        {
                            "window": window,
                            "core": None,
                            "times": times[start : start + self.config.initial_frames],
                            "fps": fps,
                        }
                    )
        self.state["base_plans"] = plain(plans)
        self.save()
        return plans

    def base_observe(self):
        for i, plan in enumerate(self.base_plans()):
            key = f"base:{i}"
            if key in self.state["observations"]:
                continue
            batch = self.media.extract(plan["window"], plan["times"], fps=plan["fps"])
            self.observe(key, batch, core=plan["core"])

    def coverage_complete(self):
        if self.task["coverage_need"] == "local":
            return True
        rows = self.state["coverage"]
        if not rows:
            return False

        def closed(row):
            attempts = self.adapters.state["attempts"]
            items = self.adapters.state["items"]
            return (
                all(
                    v in attempts and attempts[v]["end_time"] is not None
                    for v in row["open_event_boundaries"]
                )
                and all(
                    v in attempts
                    and (attempts[v].get("replay_of") or attempts[v].get("duplicate_of"))
                    for v in row["possible_replays"]
                )
                and all(v in items and items[v]["readable"] for v in row["unreadable_items"])
            )

        rows = [
            r
            for r in rows
            if r["scan_completed"] and r["resolution_met"] and r["discovery_complete"] and closed(r)
        ]
        needed = (
            self.media.contract.intersect(self.request.query_scope)
            if self.request.query_scope
            else self.media.contract.allowed_time_intervals
        )
        for a, b in needed:
            cursor = a
            for row in sorted(rows, key=lambda r: r["core"][0]):
                x, y = row["core"]
                if x <= cursor + 1e-6:
                    cursor = max(cursor, y)
            if cursor < b - 1e-6:
                return False
        return True

    def state_fingerprint(self):
        coverage = {
            digest({k: v for k, v in r.items() if k not in {"discovery_rationale"}})
            for r in self.state["coverage"]
        }
        return digest(
            {
                "store": self.store.fingerprint(),
                "adapters": {
                    k: self.adapters.state[k] for k in ("attempts", "items", "transactions")
                },
                "coverage": sorted(coverage),
            }
        )

    def formal_payload(self, defects=None):
        return {
            "question": self.request.question,
            "task": self.task,
            "variables": self.store.current(),
            "entities": self.store.state["entities"],
            "relations": self.store.state["relations"],
            "adapter_records": {
                k: self.adapters.state[k] for k in ("attempts", "items", "transactions")
            },
            "coverage": self.state["coverage"],
            "defects": defects or [],
            "previous_program": self.state.get("program") if defects else None,
        }

    def formalize(self, defects=None):
        key = "formalize:" + str(self.state["recompiles"])
        payloads = self.state.setdefault("formal_inputs", {})
        payloads.setdefault(key, self.formal_payload(defects))
        program, _ = self.session.call(key, "formalize", payloads[key])
        self.state["program"] = program
        self.save()
        return program

    def execution(self):
        program = refresh_refs(self.state["program"], self.store)
        if program["unresolved"]:
            raise ModelingError("bad_ir: " + "; ".join(program["unresolved"]))
        if program["backend"] == "constraints" and not program["query"]["nodes"]:
            program["geometry"]["output_unit"] = self.task["output_unit"]
        alternatives = [row for row in self.store.current().values() if row["alternatives"]]
        if alternatives:
            return self.joint_execution(program, alternatives)
        imported, adapter_trace = self.adapters.execute(
            program["adapters"], coverage_complete=self.coverage_complete()
        )
        if program["backend"] == "constraints":
            if not self.settings["constraints"]:
                raise ModelingError("constraints disabled in mode C")
            result = solve_constraints(program["geometry"], self.store, self.config)
            data = {
                "backend": "constraints",
                "solver": result,
                "parents": result.get("parents", []),
                "adapters": encode(adapter_trace),
            }
            if result["status"] != "solved_target":
                self.state["executions"].append(plain(data))
                return None, data
            value, post, parents = self.postprocess_solution(program, result, self.store, imported)
            if post is not None:
                data["post_processing"] = post.to_dict()
            data["parents"] = parents
        else:
            executed = Executor(self.store, self.config, imported).run(program["query"])
            if any(executed.nodes.get(key) is not True for key in program["checks"]):
                raise ModelingError("declared check is missing or did not pass")
            value = executed.value
            data = {"backend": "direct", **executed.to_dict(), "adapters": encode(adapter_trace)}
        if isinstance(value, (Quantity, IntervalQuantity)):
            value = value.convert(self.task["output_unit"])
        self.state["executions"].append(plain(data))
        self.save()
        return value, data

    def postprocess_solution(self, program, result, store, imported):
        value = (
            Quantity.make(result["value"]["exact"], result["unit"])
            if result["value"]["kind"] == "rational"
            else result
        )
        if not program["query"]["nodes"]:
            if program["checks"]:
                raise ModelingError("declared check is missing from geometry post-processing")
            return value, None, result["parents"]
        if not isinstance(value, Quantity):
            raise ModelingError(
                "algebraic post-processing requires a supported exact AST in the constraint target"
            )
        imported = {
            **imported,
            "solver_target": {
                "value": {"status": "solved_target", "quantity": value},
                "parents": result["parents"],
            },
        }
        post = Executor(store, self.config, imported).run(program["query"])
        if any(post.nodes.get(key) is not True for key in program["checks"]):
            raise ModelingError("declared check is missing or did not pass after geometry solve")
        return post.value, post, sorted(set(result["parents"] + post.parents))

    def joint_execution(self, program, alternatives):
        choices = []
        for row in alternatives:
            values = list(dict.fromkeys([row["value"], *row["alternatives"]]))
            if None in values or any(not numeric_in_span(v, row["raw_text"]) for v in values):
                raise ModelingError(
                    "ambiguous variable candidates are not an exhaustive sourced reading set"
                )
            choices.append(values)
        count = 1
        for values in choices:
            count *= len(values)
        branch_values, trace, parents = [], [], set()
        for assignment in itertools.islice(
            itertools.product(*choices), self.config.max_joint_candidates
        ):
            branch = EvidenceStore(
                self.request.question, self.media.contract, deepcopy(self.store.state)
            )
            assumptions = {}
            for row, value in zip(alternatives, assignment):
                ref = branch.state["current"][row["id"]]
                branch.state["variables"][ref].update(value=value, alternatives=[], unresolved=[])
                assumptions[ref] = value
            imported, adapter_trace = Adapters(branch, deepcopy(self.adapters.state)).execute(
                program["adapters"], coverage_complete=self.coverage_complete()
            )
            if program["backend"] == "constraints":
                if not self.settings["constraints"]:
                    raise ModelingError("constraints disabled in mode C")
                result = solve_constraints(program["geometry"], branch, self.config)
                if result["status"] != "solved_target":
                    raise ModelingError(
                        "a joint reading candidate does not have a uniquely solved target"
                    )
                value, post, branch_parents = self.postprocess_solution(
                    program, result, branch, imported
                )
                if post is not None:
                    result = {"solver": result, "post_processing": post.to_dict()}
            else:
                executed = Executor(branch, self.config, imported).run(program["query"])
                if any(executed.nodes.get(key) is not True for key in program["checks"]):
                    raise ModelingError("candidate declared check is missing or failed")
                value, branch_parents, result = executed.value, executed.parents, executed.to_dict()
            if isinstance(value, (Quantity, IntervalQuantity)):
                value = value.convert(self.task["output_unit"])
            branch_values.append(value)
            parents.update(branch_parents)
            trace.append(
                {
                    "assumptions": assumptions,
                    "execution": result,
                    "adapter_trace": encode(adapter_trace),
                }
            )
        data = {
            "backend": program["backend"],
            "joint_candidates": plain(trace),
            "joint_total": count,
            "joint_complete": count <= self.config.max_joint_candidates
            and all(r["alternatives_exhaustive"] for r in alternatives),
            "parents": sorted(parents),
        }
        self.state["executions"].append(data)
        self.save()
        return {"candidate_values": branch_values, "complete": data["joint_complete"]}, data

    def key_frames(self, parents=None):
        refs = []
        variables = []
        for ref in parents or self.store.state["current"].values():
            if ref in self.store.state["derived"]:
                refs.extend(self.store.state["derived"][ref]["value"].get("evidence_refs", []))
            else:
                variables.append(self.store.get(ref))
        for row in variables:
            refs.extend(row["evidence_refs"])
        for row in self.store.state["relations"].values():
            refs.extend(row["evidence_refs"])
        for kind in ("attempts", "items", "transactions"):
            for row in self.adapters.state[kind].values():
                refs.extend(row["evidence_refs"])
        return list(
            dict.fromkeys(
                ref for ref in refs if self.store.state["evidence"][ref]["modality"] == "video"
            )
        )

    def blind_reread(self, parents):
        changed = False
        for frame in self.key_frames(parents):
            source = self.media.catalog[frame]["source_frame_id"]
            key = "source_reread:" + source
            if key in self.state["reread_keys"]:
                continue
            before = self.store.fingerprint()
            boxes = []
            if self.request.comparison != "fixed_evidence":
                for observation in self.state["observations"].values():
                    boxes.extend(
                        c["bbox"]
                        for c in observation["requested_context"]
                        if c["bbox"] and c["frame_id"] == source
                    )
            batch = self.media.local_batch(source, boxes)
            self.observe(
                key,
                batch,
                query="Blindly re-transcribe the requested labels and geometric premises from source context.",
                reread=True,
            )
            self.state["reread_keys"].append(key)
            changed |= before != self.store.fingerprint()
            self.save()
        return changed

    def audit(self, data):
        frame_ids = self.key_frames(data.get("parents"))
        # Audit each permitted interval separately; discontinuous retrieved frames remain images.
        batches = []
        for allowed in self.media.contract.allowed_time_intervals:
            frames = [
                self.media.frame(f)
                for f in frame_ids
                if allowed[0] <= self.media.catalog[f]["timestamp_seconds"] <= allowed[1]
            ]
            for i in range(0, len(frames), self.config.max_frames_per_call):
                batches.append(
                    MediaBatch(
                        TimeSpan(*allowed),
                        tuple(frames[i : i + self.config.max_frames_per_call]),
                        None,
                        False,
                    )
                )
        if not batches:
            batches = [None]
        audits = []
        for batch in batches:
            prepared = self.media.prepare(batch) if batch else None
            packet = (
                self.media.evidence(prepared, (batch.span.start_seconds, batch.span.end_seconds))
                if batch
                else {}
            )
            if packet:
                self.store.present(packet)
            payload = {
                **self.formal_payload(),
                "program": refresh_refs(self.state["program"], self.store),
                "execution": data,
                "evidence": {
                    k: {p: v for p, v in e.items() if p != "path"} for k, e in packet.items()
                },
                "source_check_scope": [e["id"] for e in packet.values()],
            }
            key = "audit:" + digest(payload)
            result, _ = self.session.call(key, "audit", payload, prepared=prepared, evidence=packet)
            for gap in result["defects"]:
                if gap["evidence_refs"]:
                    gap["evidence_refs"] = self.store.sources(gap["evidence_refs"])
                for v in gap["variable_refs"]:
                    self.store.get(v)
                if gap["window"] and not self.media.contract.permits_span(gap["window"]):
                    raise ProtocolError("audit repair exceeds permission")
            audits.append(result)
        self.state["audits"] = audits
        self.save()
        return {
            "semantics_ok": all(a["semantics_ok"] for a in audits),
            "sources_ok": all(a["sources_ok"] for a in audits),
            "coverage_ok": self.coverage_complete() and all(a["coverage_ok"] for a in audits),
            "defects": [d for a in audits for d in a["defects"]],
        }

    def repair(self, defects):
        if self.settings["repair"] == "none" or not defects:
            return False
        ranked = sorted(
            defects,
            key=lambda d: (
                not d["blocking"],
                not d["answer_sensitive"],
                -d["dependent_nodes"],
                d["bbox"] is None,
            ),
        )
        gap = ranked[0]
        modeling = gap["kind"] == "bad_ir"
        if modeling and self.state["recompiles"] < self.config.semantic_recompiles:
            before = digest(self.state.get("program"))
            self.state["recompiles"] += 1
            self.formalize([gap])
            changed = before != digest(self.state["program"])
            self.state["actions"].append(
                {"kind": "semantic_recompile", "defect": gap, "changed": changed}
            )
            self.save()
            return changed
        if (
            self.state["extra_rounds"] >= self.config.repair_rounds
            or self.request.comparison == "fixed_evidence"
        ):
            return False
        fingerprint = self.state_fingerprint()
        action_id = digest({"gap": gap, "state": fingerprint, "policy": self.settings["repair"]})
        if any(a.get("id") == action_id for a in self.state["actions"]):
            return False
        self.state["extra_rounds"] += 1
        key = "repair:" + action_id
        if self.settings["repair"] == "uniform":
            allowed = self.media.contract.allowed_time_intervals
            window = allowed[(self.state["extra_rounds"] - 1) % len(allowed)]
            # Equal call/frame allowance: different temporal phase, not target-specific selection.
            n = min(self.config.initial_frames, self.config.max_frames_per_call)
            times = [window[0] + (window[1] - window[0]) * (i + 0.5) / n for i in range(n)]
            batch = self.media.extract(window, times)
            self.observe(
                key,
                batch,
                query="Uniformly inspect all required slots in this extra sampling batch.",
            )
        else:
            if gap["kind"] in {"missing_denominator", "incomplete_extrema_set"}:
                # Revisit the actual under-covered interval in sequential dense pieces. This
                # is one extra evidence round, with every constituent call charged separately.
                needed = gap["window"]
                if not needed:
                    bad = next(
                        (
                            r
                            for r in self.state["coverage"]
                            if not r["discovery_complete"]
                            or not r["resolution_met"]
                            or r["open_event_boundaries"]
                            or r["possible_replays"]
                            or r["unreadable_items"]
                        ),
                        None,
                    )
                    needed = (
                        bad["core"]
                        if bad
                        else self.request.query_scope
                        or self.media.contract.allowed_time_intervals[0]
                    )
                for interval in self.media.contract.intersect(needed):
                    fps = self.config.scan_fps * 2
                    maximum = (self.config.max_frames_per_call - 1) / fps
                    start, end = interval
                    part = 0
                    while start < end - 1e-9:
                        stop = min(end, start + maximum)
                        times, _ = sampling((start, stop), self.config, fps=fps)
                        batch = self.media.extract((start, stop), times, fps=fps)
                        self.observe(
                            key + f":scan:{part}", batch, core=(start, stop), query=gap["detail"]
                        )
                        start = stop
                        part += 1
                changed = fingerprint != self.state_fingerprint()
                self.state["actions"].append(
                    {"id": action_id, "kind": "directed_scan", "defect": gap, "changed": changed}
                )
                self.save()
                return changed
            frame_ids = gap["evidence_refs"]
            if not frame_ids and gap["variable_refs"]:
                frame_ids = self.store.get(gap["variable_refs"][0])["evidence_refs"]
            contexts = [
                c
                for observation in self.state["observations"].values()
                for c in observation["requested_context"]
            ]
            context = next((c for c in contexts if c["frame_id"] and c["bbox"]), None)
            if not frame_ids and context:
                frame_ids = [context["frame_id"]]
            if (
                frame_ids
                and (gap["bbox"] or context)
                and gap["kind"] not in {"missing_denominator", "incomplete_extrema_set"}
            ):
                fid = next((f for f in frame_ids if f in self.media.catalog), None)
                if fid:
                    box = gap["bbox"] or context["bbox"]
                    batch = self.media.local_batch(fid, [box])
                    self.observe(key, batch, query=gap["detail"], reread=True)
                else:
                    return False
            else:
                window = gap["window"]
                if not window:
                    t = (
                        self.media.catalog[frame_ids[0]]["timestamp_seconds"]
                        if frame_ids and frame_ids[0] in self.media.catalog
                        else None
                    )
                    permitted = next(
                        (
                            w
                            for w in self.media.contract.allowed_time_intervals
                            if t is not None and w[0] <= t <= w[1]
                        ),
                        self.media.contract.allowed_time_intervals[0],
                    )
                    if t is None:
                        t = (permitted[0] + permitted[1]) / 2
                    window = (max(permitted[0], t - 1), min(permitted[1], t + 1))
                fps = self.config.scan_fps * 2
                length = (self.config.max_frames_per_call - 1) / fps
                window = (window[0], min(window[1], window[0] + length))
                times, _ = sampling(window, self.config, fps=fps)
                batch = self.media.extract(window, times, fps=fps)
                self.observe(
                    key,
                    batch,
                    core=window if self.task["coverage_need"] != "local" else None,
                    query=gap["detail"],
                    reread=True,
                )
        changed = fingerprint != self.state_fingerprint()
        self.state["actions"].append(
            {"id": action_id, "kind": self.settings["repair"], "defect": gap, "changed": changed}
        )
        self.save()
        return changed

    def finish(
        self,
        prediction,
        status,
        reason,
        *,
        value=None,
        data=None,
        defects=None,
        matching="not_run",
        audit=None,
    ):
        data, audit = data or {}, audit or {}
        verified = status in {"verified_exact", "verified_at_option_precision", "observed_answer"}
        solver_status = data.get("solver", {}).get(
            "status", "not_used" if data.get("backend") == "direct" else "not_run"
        )
        result = R8Result(
            prediction,
            status,
            verified,
            reason,
            value=encode(value),
            evidence_status="checked" if audit.get("sources_ok") else "unresolved",
            modeling_status="checked" if audit.get("semantics_ok") else "unresolved",
            solver_status=solver_status,
            matching_status=matching,
            task_spec=self.task or {},
            variables=self.store.state["variables"],
            evidence=self.store.state["evidence"],
            coverage=self.state["coverage"],
            defects=defects or [],
            execution=data,
            resources=self.session.resources(),
            trace={
                "mode": self.request.mode,
                "mode_settings": self.settings,
                "comparison": self.request.comparison,
                "diagnostic": self.request.diagnostic,
                "software": self.software,
                "actions": self.state["actions"],
                "revisions": self.store.state["revisions"],
                "executions": self.state["executions"],
                "audits": self.state.get("audits", []),
                "request_id": self.request.request_id,
                "video_id": self.request.video_id,
                "group_id": self.request.group_id,
                "question_sha256": digest(self.request.question),
                "options": plain([asdict(c) for c in self.request.choices]),
                "geometry_relations": self.store.state["relations"],
                "adapter_record_versions": self.adapters.state["record_versions"],
            },
        )
        result.trace["entities"] = self.store.state["entities"]
        result.trace["adapter_records"] = {
            k: self.adapters.state[k] for k in ("attempts", "items", "transactions")
        }
        result.trace["video_sha256"] = self.media.source_hash
        result.trace["scope_hash"] = self.media.contract.fingerprint
        result.trace["fallback_from"] = self.state.get("fallback_from")
        result.trace["relation_versions"] = self.store.state["relation_versions"]
        self.state["result"] = result.to_dict()
        self.session.persist()
        return result

    def unresolved(self, status, reason, *, data=None, defects=None):
        if self.request.require_choice and self.request.choices:
            payload = {
                "question": self.request.question,
                "choices": [asdict(c) for c in self.request.choices],
                "variables": self.store.current(),
                "unresolved_reason": reason,
            }

            def validator(v):
                if v["prediction"] not in {c.label for c in self.request.choices}:
                    raise ProtocolError("fallback must choose an original label")

            try:
                result, _ = self.session.call(
                    "fallback", "fallback", payload, validator, terminal=True
                )
            except BudgetExhausted:
                return self.finish(
                    None,
                    status,
                    reason + "; fallback budget unavailable",
                    data=data,
                    defects=defects,
                )
            self.state["fallback_from"] = status
            return self.finish(
                result["prediction"],
                "forced_guess",
                reason,
                data=data,
                defects=defects,
                matching="forced",
            )
        return self.finish(None, status, reason, data=data, defects=defects)

    def baseline(self, role):
        payload = {
            "question": self.request.question,
            "choices": [asdict(c) for c in self.request.choices],
        }
        prepared, packet = None, {}
        if role == "direct":
            # A uses one bounded packet; fixed_evidence may explicitly supply several packets.
            frames = []
            for plan in self.base_plans():
                batch = self.media.extract(plan["window"], plan["times"], fps=plan["fps"])
                frames.extend(batch.frames)
            if len(frames) > self.config.max_frames_per_call:
                raise ProtocolError(
                    "baseline A input exceeds per-call frame cap; provide a bounded fixed frame set"
                )
            # Permission-separated packets cannot be represented as one continuous video.
            if len(self.media.contract.allowed_time_intervals) != 1:
                raise ProtocolError("baseline A requires one permitted interval per request")
            window = self.media.contract.allowed_time_intervals[0]
            batch = MediaBatch(TimeSpan(*window), tuple(frames), None, False)
            prepared = self.media.prepare(batch)
            packet = self.media.evidence(prepared, window)
            self.store.present(packet)
            payload["evidence"] = {
                k: {p: v for p, v in e.items() if p != "path"} for k, e in packet.items()
            }
        else:
            payload["variables"] = self.store.current()

        def validator(v):
            if (
                self.request.choices
                and v["prediction"] is not None
                and v["prediction"] not in {c.label for c in self.request.choices}
            ):
                raise ProtocolError("baseline output label is not an original choice")
            if not self.request.choices and v["value"] is not None:
                number(v["value"])

        value, _ = self.session.call(role, role, payload, validator, prepared, packet)
        prediction = value["prediction"] if self.request.choices else value["value"]
        return self.finish(
            prediction,
            "unresolved_modeling",
            "experimental model answer without deterministic/source verification",
            value=value["value"],
            matching="model_prediction",
        )

    def load_variables(self):
        if self.state.get("variables_loaded") or not self.request.variables_input:
            return
        artifact = json.loads(Path(self.request.variables_input).read_text(encoding="utf-8"))
        allowed = {
            "format",
            "kind",
            "question_sha256",
            "video_sha256",
            "scope_hash",
            "entities",
            "evidence",
            "variables",
            "relations",
            "adapters",
        }
        if set(artifact) != allowed or artifact["format"] != "r8-variable-replay/1":
            raise ProtocolError(
                "invalid variable replay schema; answers/extra metadata are not accepted"
            )
        if artifact["kind"] not in {"observed_replay", "oracle_diagnostic"} or (
            artifact["kind"] == "oracle_diagnostic" and not self.request.diagnostic
        ):
            raise ProtocolError("oracle variables require explicit diagnostic=True")
        if (
            artifact["question_sha256"] != digest(self.request.question)
            or artifact["video_sha256"] != self.media.source_hash
            or artifact["scope_hash"] != self.media.contract.fingerprint
        ):
            raise ProtocolError("variable replay question/media/scope mismatch")
        self.media.restore_evidence(artifact["evidence"])
        packet = {
            fid: self.media.get_evidence(fid)
            for fid in artifact["evidence"]
            if fid in self.media.catalog
        }
        self.store.present(packet)
        observation = {
            "entities": artifact["entities"],
            "observations": artifact["variables"],
            "relations": artifact["relations"],
            **artifact["adapters"],
            "unresolved": [],
            "requested_context": [],
            "discovery": {
                "complete": False,
                "rationale": "replay does not itself establish video coverage",
                "open_event_boundaries": [],
                "possible_replays": [],
                "unreadable_items": [],
            },
        }
        validate(observation, "observe")
        self.store.ingest(observation, packet, "variable_replay")
        self.adapters.ingest(observation, packet, "variable_replay")
        self.state["variables_loaded"] = True
        self.save()

    def run(self):
        if self.state.get("result"):
            stored = {
                k: v for k, v in self.state["result"].items() if k not in {"pipeline_id", "version"}
            }
            return R8Result(**stored)
        try:
            if self.task is None:
                payload = {
                    "question": self.request.question,
                    "public_protocol": {
                        "scope": asdict(self.media.contract),
                        "query_scope": self.request.query_scope,
                        "query_time": self.request.query_time,
                        "output_protocol": self.request.output_protocol,
                    },
                    "mechanism_hint": self.request.execution_subtype,
                }
                self.task, _ = self.session.call(
                    "compile",
                    "compile",
                    payload,
                    lambda v: validate_task(v, self.request, self.store),
                )
                self.state["task"] = self.task
                for given in self.task["givens"]:
                    self.store.append(given, origin=given["origin"], support="question_given")
                self.save()
            if self.settings["direct"]:
                return self.baseline("direct")
            self.load_variables()
            if not self.request.variables_input:
                self.base_observe()
            if self.settings["model_math"]:
                return self.baseline("model_math")
            if "program" not in self.state:
                self.formalize()
            while True:
                try:
                    value, data = self.execution()
                except (
                    ModelingError,
                    ProtocolError,
                    ZeroDivisionError,
                    KeyError,
                    TypeError,
                ) as exc:
                    reason = str(exc)
                    kind = (
                        "missing_denominator"
                        if "denominator" in reason
                        else "incomplete_extrema_set"
                        if "extrema" in reason
                        else "unreadable_value"
                        if any(
                            t in reason
                            for t in ("unreadable", "missing input", "ambiguous variable")
                        )
                        else "bad_ir"
                    )
                    gaps = [defect(kind, reason)]
                    if self.repair(gaps):
                        continue
                    return self.unresolved(
                        "unresolved_evidence" if kind != "bad_ir" else "unresolved_modeling",
                        reason,
                        defects=gaps,
                    )
                if value is None:
                    status = data["solver"]["status"]
                    kind = (
                        "solver_unknown"
                        if status == "timeout_or_unknown"
                        else "unsupported_relation"
                        if status == "unsupported_theory"
                        else "snapshot_conflict"
                        if status == "inconsistent_constraints"
                        else "unreadable_value"
                    )
                    gaps = [defect(kind, data["solver"].get("reason", status))]
                    if self.repair(gaps):
                        continue
                    return self.unresolved(
                        "solver_unknown"
                        if status == "timeout_or_unknown"
                        else "unresolved_modeling",
                        status,
                        data=data,
                        defects=gaps,
                    )
                if self.settings["reread"] and self.blind_reread(data.get("parents")):
                    continue
                audit = self.audit(data)
                gaps = audit["defects"]
                if not audit["semantics_ok"] and not gaps:
                    gaps.append(defect("bad_ir", "semantic audit did not pass"))
                if not audit["sources_ok"] and not gaps:
                    gaps.append(defect("unreadable_value", "source audit did not pass"))
                if not audit["coverage_ok"] and not gaps:
                    gaps.append(
                        defect(
                            "missing_denominator"
                            if self.task["coverage_need"] == "all_attempts"
                            else "incomplete_extrema_set",
                            "coverage/discovery obligations remain",
                        )
                    )
                if gaps:
                    if self.repair(gaps):
                        continue
                    return self.unresolved(
                        "unresolved_evidence"
                        if not audit["sources_ok"] or not audit["coverage_ok"]
                        else "unresolved_modeling",
                        "layered verification incomplete",
                        data=data,
                        defects=gaps,
                    )
                matched = match(value, self.request.choices, self.task)
                data["option_match"] = asdict(matched)
                if matched.status == "annotation_anomaly":
                    if self.request.require_choice:
                        return self.unresolved(
                            "annotation_anomaly", "question/option annotation mismatch", data=data
                        )
                    return self.finish(
                        None,
                        "annotation_anomaly",
                        "question/option annotation mismatch",
                        value=value,
                        data=data,
                        matching=matched.status,
                        audit=audit,
                    )
                if matched.prediction is not None:
                    observed = (
                        data["backend"] == "direct"
                        and not self.state["program"]["query"]["nodes"]
                        and bool(data.get("parents"))
                        and all(self.store.get(p)["origin"] == "observed" for p in data["parents"])
                    )
                    status = (
                        "observed_answer"
                        if observed
                        else "verified_exact"
                        if matched.status == "exact"
                        else "verified_at_option_precision"
                    )
                    return self.finish(
                        matched.prediction,
                        status,
                        "all required system checks passed",
                        value=value,
                        data=data,
                        matching=matched.status,
                        audit=audit,
                    )
                gaps = [
                    defect(
                        "option_sensitive_uncertainty",
                        "known computation does not determine an original option",
                    )
                ]
                if self.repair(gaps):
                    continue
                return self.unresolved(
                    "unresolved_modeling", "option matching unresolved", data=data, defects=gaps
                )
        except BudgetExhausted as exc:
            return self.unresolved(
                "unresolved_evidence",
                str(exc),
                defects=[defect("unreadable_value", "budget exhausted")],
            )
        except ModelFailure:
            # Model/backend/contract engineering failures remain visible; never convert them to A.
            self.session.persist()
            raise
