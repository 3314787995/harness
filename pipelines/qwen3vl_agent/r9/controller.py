"""One bounded controller for S1-S6, with checkpointed phase transitions."""

import time
from copy import deepcopy
from dataclasses import asdict, replace

from qwen3vl_agent.models.base import ModelOutput
from qwen3vl_agent.models.qwen3vl import InputContextExceeded
from qwen3vl_agent.p01.types import TimeSpan
from qwen3vl_agent.r1.media import MediaBatch

from . import acquisition
from .audit import deterministic_audit, validate_visual_audit
from .compiler import observation_task, option_domain, validate_spec
from .config import R9Config
from .identity import update_entity_links
from .media import ScopedMedia
from .observer import observation_payload
from .operators import run_query
from .protocol import format_prediction, match_choice, validate_answer
from .runtime import Session, checkpoint_for, software_fingerprint
from .spatial_state import SpatialState
from .types import (
    BudgetExhausted,
    Gap,
    ProtocolError,
    QueryResult,
    R9Request,
    R9Result,
    digest,
    plain,
)


class R9VideoAgent:
    def __init__(self, model, config=None):
        self.model, self.config = model, R9Config.from_mapping(config)

    def load(self):
        self.model.load()

    def unload(self):
        self.model.unload()

    def solve(self, request):
        if isinstance(request, dict):
            try:
                request = R9Request(**request)
            except (ValueError, TypeError) as exc:
                return R9Result(
                    "",
                    None,
                    "invalid_input",
                    False,
                    None,
                    None,
                    [str(exc)],
                    [],
                    [],
                    {"model_calls": 0, "validation_stage": "public_request"},
                )
        return Controller(self.model, self.config, request).run()

    def generate(self, messages, *, videos=None, images=None, choices=(), **kwargs):
        if images or not videos or len(videos) != 1:
            raise ProtocolError("R9 requires exactly one video")
        users = [m for m in messages if m.get("role") == "user"]
        if len(users) != 1 or not isinstance(users[0].get("content"), str):
            raise ProtocolError("R9 generate requires one plain-text user question")
        result = self.solve(R9Request(videos[0], users[0]["content"], choices=choices, **kwargs))
        return ModelOutput(result.text, {"r9": result.to_dict()})


class Controller:
    def __init__(self, model, config, request):
        started = time.perf_counter()
        self.model, self.config, self.request = model, config, request
        self.media = ScopedMedia(request, config)
        self.software = software_fingerprint(model, config)
        self.checkpoint = checkpoint_for(request, config, self.media, self.software)
        self.data = self.checkpoint.restored or {
            "phase": "compile",
            "spec": None,
            "spatial": None,
            "catalog": {},
            "queue": [],
            "history": [],
            "refinements": 0,
            "round": 0,
            "gaps": [],
            "visual_audits": [],
            "stage_errors": [],
            "result": None,
        }
        self.state = (
            SpatialState(self.data["spec"], self.data["spatial"]) if self.data["spec"] else None
        )
        self.media.restore(self.data["catalog"])
        self.session = Session(model, config, request, self.data, self.save)
        self.session.started = started

    def save(self):
        if self.state:
            self.data["spatial"] = self.state.data
        self.data["catalog"] = self.media.catalog
        self.checkpoint.save(self.data)

    def persist(self):
        self.session.persist()

    def compile(self):
        payload = {
            "question": self.request.question,
            "option_domain": option_domain(self.request.choices),
            "public_protocol": {
                "output_protocol": self.request.protocol,
                "output_unit": self.request.output_unit,
                "allowed_intervals": self.media.contract.allowed_time_intervals,
                "query_scope": self.request.query_scope,
                "query_time": self.request.query_time,
            },
        }
        spec = self.session.call(
            "compile",
            "compile",
            payload,
            validator=lambda s: validate_spec(s, self.request, self.media.contract, self.config),
        )
        self.data["spec"] = spec
        self.state = SpatialState(spec)
        self.data["queue"] = acquisition.overview(self.request, self.media.contract, self.config)
        self.data["phase"] = "baseline" if self.request.mode == "B2" else "acquire"
        self.persist()

    def execute_action(self, action):
        if action["kind"] == "crop":
            return self.media.local_batch(action["frame_id"], [action["bbox"]])
        return self.media.extract(
            action["time_interval"],
            action["times"],
            fps=self.config.scan_fps if action["sequential"] else None,
        )

    def acquire(self):
        if not self.data["queue"]:
            self.data["phase"] = "relations"
            self.persist()
            return
        action = self.data["queue"][0]
        batch = self.execute_action(action)
        prepared = self.media.prepare(batch)
        evidence = self.media.evidence(prepared)
        namespace = f"obs{len(self.data['history']):03d}"
        previously_shown = {
            e["id"] for receipt in self.data["receipts"] for e in receipt["evidence"].values()
        }
        if (
            action["named_gap"] != "initial_overview"
            and {f.id for f in batch.frames} <= previously_shown
        ):
            self.data["gaps"].append(
                asdict(Gap("coverage", "sampling produced no new source view"))
            )
            self.data["phase"] = "finish"
            self.persist()
            return
        payload = observation_payload(self.data["spec"], self.state, action, evidence)

        def checked(value):
            self.check_media_times(value)
            trial = SpatialState(self.data["spec"], self.state.data)
            update_entity_links(trial, value, evidence, namespace, self.config)

        try:
            value = self.session.call(namespace, "observe", payload, prepared, evidence, checked)
        except InputContextExceeded:
            if action["kind"] != "sample" or len(action["times"]) <= 2:
                raise ProtocolError("minimal observation cannot fit the context budget") from None
            mid = len(action["times"]) // 2
            splits = [
                dict(action, times=action["times"][:mid]),
                dict(action, times=action["times"][mid:]),
            ]
            if action.get("sequential"):
                boundary = (action["times"][mid - 1] + action["times"][mid]) / 2
                splits[0]["time_interval"] = [action["time_interval"][0], boundary]
                splits[1]["time_interval"] = [boundary, action["time_interval"][1]]
            # New job IDs; the failed preparation remains charged.
            self.data["history"].append({**action, "context_split": True})
            self.data["queue"][:1] = splits
            self.persist()
            return
        update_entity_links(self.state, value, evidence, namespace, self.config)
        self.state.data["coverage"].append(
            self.media.coverage(batch, sequential=action.get("sequential", False))
        )
        self.data["gaps"] = self.map_gaps(value["gaps"], evidence)
        self.data["history"].append(action)
        self.data["queue"].pop(0)
        self.persist()

    def check_media_times(self, value):
        for record in [*value.get("records", []), *value.get("links", [])]:
            if not self.media.contract.permits_span(record["valid_time"]):
                raise ProtocolError("model record widens the permitted video interval")

    def map_gaps(self, values, evidence):
        gaps = deepcopy(values)
        for gap in gaps:
            if gap["frame_id"] in evidence:
                gap["frame_id"] = evidence[gap["frame_id"]]["id"]
            if gap["frame_id"] and gap["frame_id"] not in self.media.catalog:
                gap["frame_id"], gap["bbox"] = None, None
            if gap["time_interval"] and not self.media.contract.permits_span(gap["time_interval"]):
                gap["time_interval"] = None
        return gaps

    def evidence_batch(self, record_ids=(), *, maximum=None):
        maximum = maximum or self.config.preferred_frames_per_call
        observations = self.state.data["observations"] if self.state else {}
        needed = []
        for rid in record_ids:
            needed += self.state.data["records"][rid]["source_observation_ids"]
        if not needed and self.state:
            # One source per requested entity first, then recent sources/bridge frames.
            for entity in self.data["spec"]["entities"]:
                for link in reversed(self.state.data["entity_links"].get(entity["id"], [])):
                    needed += link["source_observation_ids"]
            needed += list(reversed(observations))
        fids = list(dict.fromkeys(observations[i]["frame_id"] for i in needed if i in observations))
        if not fids:
            fids = list(self.media.catalog)
        selected, crops = [], 0
        for fid in fids:
            if len(selected) >= maximum:
                break
            crop = self.media.catalog[fid]["source_frame_id"] != fid
            if crop and crops >= self.config.max_detail_images_per_call:
                continue
            selected.append(fid)
            crops += int(crop)
        if not selected:
            return None
        frames = tuple(
            sorted((self.media.frame(fid) for fid in selected), key=lambda f: f.timestamp_seconds)
        )
        a, b = min(f.timestamp_seconds for f in frames), max(f.timestamp_seconds for f in frames)
        if a == b:
            a, b = next(s for s in self.media.contract.allowed_time_intervals if s[0] <= a <= s[1])
        return MediaBatch(
            TimeSpan(a, b),
            frames,
            ordered=crops == 0,
            crops={
                f.id: self.media.catalog[f.id]["view_box"]
                for f in frames
                if self.media.catalog[f.id]["source_frame_id"] != f.id
            },
        )

    def relations(self):
        batch = self.evidence_batch()
        if batch is None:
            self.data["phase"] = "refine"
            self.data["gaps"] = [asdict(Gap("relation", "no observed frames"))]
            self.persist()
            return
        prepared = self.media.prepare(batch)
        evidence = self.media.evidence(prepared)
        presented = {e["id"] for e in evidence.values()}
        context = self.state.context()
        context["observations"] = [
            o for o in self.state.data["observations"].values() if o["frame_id"] in presented
        ][-24:]
        payload = {
            "task": observation_task(self.data["spec"]),
            "operation": self.data["spec"]["operation"],
            "operations": [
                self.data["spec"]["operation"],
                *[n["operation"] for n in self.data["spec"]["nodes"]],
            ],
            "direction_rule": self.data["spec"]["direction_rule"],
            "route": self.data["spec"]["route"],
            "state": context,
            "named_gaps": self.data["gaps"],
        }
        namespace = f"rel{self.data['round']:03d}"

        def validator(value):
            self.check_media_times(value)
            SpatialState(self.data["spec"], self.state.data).relations(value, presented, namespace)

        value = self.session.call(namespace, "relations", payload, prepared, evidence, validator)
        self.state.relations(value, presented, namespace)
        self.data["gaps"] = self.map_gaps(value["gaps"], evidence)
        self.data["phase"] = "query"
        self.persist()

    def query(self):
        result = run_query(
            self.data["spec"],
            self.state,
            self.media.contract,
            self.request.query_time,
            self.request.query_scope,
        )
        self.data["query_result"] = asdict(result)
        if result.value is None:
            self.data["gaps"] = [asdict(g) for g in result.gaps] + self.data["gaps"]
            self.data["phase"] = "refine"
        else:
            self.data["phase"] = "audit"
        self.persist()

    def audit(self):
        result = self.query_result()
        missing = [
            rid
            for rid in result.record_ids
            if self.state.data["audit_verdicts"].get(rid) != "supported"
        ]
        if missing:
            batch = self.evidence_batch(missing, maximum=self.config.max_video_frames_per_call)
            prepared = self.media.prepare(batch)
            evidence = self.media.evidence(prepared)
            presented = {e["id"] for e in evidence.values()}
            eligible = [
                rid
                for rid in missing
                if all(
                    self.state.data["observations"][oid]["frame_id"] in presented
                    for oid in self.state.data["records"][rid]["source_observation_ids"]
                )
            ]
            if eligible:
                payload = {
                    "atomic_claims": [self.state.data["records"][rid] for rid in eligible],
                    "frames": {a: e["id"] for a, e in evidence.items()},
                }
                value = self.session.call(
                    f"audit{self.data['round']:03d}",
                    "audit",
                    payload,
                    prepared,
                    evidence,
                    lambda v: validate_visual_audit(v, eligible, evidence),
                )
                self.data["visual_audits"].append(value)
                for check in value["checks"]:
                    self.state.data["audit_verdicts"][check["record_id"]] = check["verdict"]
                    if check["verdict"] == "contradicted":
                        self.state.invalidate([check["record_id"]])
                self.data["gaps"] = self.map_gaps(value["gaps"], evidence)
        verification = deterministic_audit(self.data["spec"], self.state, result)
        self.data["verification"] = asdict(verification)
        self.data["gaps"] = [asdict(g) for g in verification.gaps] + self.data["gaps"]
        self.data["phase"] = "finish" if verification.can_stop else "refine"
        self.persist()

    def refine(self):
        if (
            self.request.mode == "B3"
            or self.data["refinements"] >= self.config.max_refinement_rounds
        ):
            self.data["phase"] = "finish"
        else:
            gaps = [Gap(**g) for g in self.data["gaps"]]
            action = acquisition.choose(
                gaps, self.state, self.media, self.config, self.data["history"], self.request
            )
            if action is None:
                self.data["phase"] = "finish"
            else:
                self.data["queue"] = [action]
                self.data["refinements"] += 1
                self.data["round"] += 1
                self.data["phase"] = "acquire"
        self.persist()

    def query_result(self):
        value = dict(self.data.get("query_result") or {})
        value["gaps"] = [Gap(**g) for g in value.get("gaps", [])]
        return QueryResult(**value)

    def answer_call(self, *, baseline=False, semantic=None):
        batch = self.evidence_batch(maximum=self.config.max_video_frames_per_call)
        if not baseline and batch:
            remaining = (
                self.session.limits["max_visual_exposures"]
                - self.session.resources()["visual_exposures"]
            )
            exposed = {i for r in self.data["receipts"] for i in r["source_frame_ids"]}
            # Terminal calls can work from the sourced state if all visual exposure is spent.
            retained = tuple(
                f for f in batch.frames if self.media.catalog[f.id]["source_frame_id"] in exposed
            )[: max(0, remaining)]
            batch = (
                replace(
                    batch,
                    frames=retained,
                    crops={k: v for k, v in batch.crops.items() if k in {f.id for f in retained}},
                )
                if retained
                else None
            )
        prepared = self.media.prepare(batch) if batch else None
        evidence = self.media.evidence(prepared) if prepared else {}
        source_ids = set(self.state.data["observations"]) if self.state else set()
        source_ids |= {e["id"] for e in evidence.values()}
        unit = self.request.output_unit or (
            self.data["spec"]["measurement"]["unit"] if self.data["spec"] else None
        )
        payload = {
            "question": self.request.question,
            "original_option_texts": [c.text for c in self.request.choices],
            "output_protocol": self.request.protocol,
            "output_unit": unit,
            "query_scope": self.request.query_scope,
            "query_time": self.request.query_time,
            "force_answer": self.request.force_answer,
            "frames": {alias: e["id"] for alias, e in evidence.items()},
            "available_observation_ids": sorted(source_ids),
            "baseline_mode": self.request.mode if baseline else None,
        }
        if self.state:
            payload["spatial_evidence"] = self.state.context()
        if self.request.mode == "B2":
            payload["explicit_reference_frame"] = self.data["spec"]["query_frame"]
        if semantic is not None:
            payload["computed_semantic_answer"] = semantic
        validator = lambda v: validate_answer(v, self.request, source_ids, unit)
        try:
            key = (
                "terminal_answer-"
                + digest(
                    {
                        "input": payload,
                        "frames": [f.id for f in prepared.frames] if prepared else [],
                    }
                )[:12]
            )
            result = self.session.call(
                key, "answer", payload, prepared, evidence, validator, terminal=True
            )
        except InputContextExceeded:
            compact = {k: v for k, v in payload.items() if k not in {"spatial_evidence", "frames"}}
            compact["sourced_observations"] = (
                self.state.context(8)["observations"] if self.state else []
            )
            result = self.session.call(
                "terminal_answer_compact", "answer", compact, validator=validator, terminal=True
            )
        return result

    def baseline(self):
        if self.request.mode != "B0":
            actions = acquisition.overview(self.request, self.media.contract, self.config)
            for action in actions:
                self.execute_action(action)
            self.persist()
        return self.finish(baseline=True)

    def finish(self, baseline=False):
        result = self.query_result()
        verified = bool(self.data.get("verification", {}).get("can_stop")) and not baseline
        semantic = result.value if verified else None
        sources, scale = result.source_ids, result.scale_source
        status = result.status if verified else "unresolved"
        forced = False
        unit = self.request.output_unit or (
            self.data["spec"]["measurement"]["unit"] if self.data["spec"] else None
        )
        mapping_missing = (
            semantic is not None
            and self.request.choices
            and match_choice(semantic, self.request.choices) is None
        )
        if baseline or mapping_missing or (semantic is None and self.request.force_answer):
            answer = self.answer_call(baseline=baseline, semantic=semantic)
            semantic = answer["semantic_answer"]
            sources = answer["source_ids"] or sources
            unit = answer["unit"]
            forced = not verified and not baseline
            status = result.status if verified and not mapping_missing else "estimated"
            self.data["terminal_reason"] = answer["reason"]
        prediction, text = format_prediction(semantic, self.request)
        if self.request.protocol == "numeric" and not scale:
            scale = ["unknown"]
        reasons = list(dict.fromkeys(g["detail"] for g in self.data["gaps"]))
        if baseline:
            reasons = [
                "text-only baseline has no visual evidence"
                if self.request.mode == "B0"
                else "direct model estimate; no structured spatial audit"
            ]
        trace = {
            "request_id": self.request.request_id,
            "mode": self.request.mode,
            "comparison": self.request.comparison,
            "software": self.software,
            "config": asdict(self.config),
            "question_spec": self.data["spec"],
            "spatial_state": self.state.data if self.state else None,
            "query_result": asdict(result),
            "verification": self.data.get("verification"),
            "resources": self.session.resources(),
            "receipts": self.data["receipts"],
            "actions": self.data["history"],
            "stage_errors": self.data["stage_errors"],
            "terminal_reason": self.data.get("terminal_reason"),
            "semantic_accuracy_measured": False,
        }
        output = R9Result(
            text, prediction, status, forced, semantic, unit, reasons, sources, scale, plain(trace)
        )
        self.data["result"] = output.to_dict()
        self.data["phase"] = "done"
        self.persist()
        return output

    def run(self):
        if self.data["result"]:
            return R9Result(**self.data["result"])
        if self.request.mode in {"B0", "B1"}:
            return self.baseline()
        while self.data["phase"] != "done":
            phase = self.data["phase"]
            if phase == "finish":
                return self.finish()
            if phase == "baseline":
                return self.baseline()
            try:
                getattr(self, phase)()
            except (BudgetExhausted, ProtocolError, InputContextExceeded) as exc:
                self.data["stage_errors"].append(
                    {"phase": phase, "error": str(exc), "kind": type(exc).__name__}
                )
                self.data["gaps"].append(
                    asdict(
                        Gap("protocol" if isinstance(exc, ProtocolError) else "coverage", str(exc))
                    )
                )
                self.data["phase"] = "finish"
                self.persist()
        return R9Result(**self.data["result"])
