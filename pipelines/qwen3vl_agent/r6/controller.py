"""One evidence-driven controller for all six discussion groups."""

import time
from copy import deepcopy
from dataclasses import asdict, replace

from .acquisition import (
    action_identity,
    clip_batches,
    coverage_complete,
    fallback_actions,
    initial_actions,
    select_action,
    uniform_times,
    validate_action,
    windows,
)
from .config import R6Config
from .media import ScopedMedia
from .prompts import SUBTYPES
from .providers import TextSources, UnavailableAudioProvider
from .runtime import Session, checkpoint_for, software_fingerprint
from .schema import validate_observation, validate_query
from .state import (
    add_observation,
    evaluate_assessment,
    new_state,
    obligations_met,
    relation_sources,
    verification_checks,
)
from .types import (
    BudgetExhausted,
    ContextOverflow,
    ModelFailure,
    ProtocolError,
    R6Request,
    R6Result,
    digest,
)


def gap(kind, predicate, *, modality="video", labels=(), span=None):
    return {
        "kind": kind,
        "predicate": predicate,
        "candidate_labels": list(labels),
        "entity_ids": [],
        "span": span,
        "modality": modality,
        "desired_observation": predicate,
        "blocks_answer": True,
    }


class R6VideoAgent:
    def __init__(self, model, config=None, *, media_factory=ScopedMedia):
        self.model = model
        self.config = R6Config.from_mapping(config)
        self.media_factory = media_factory

    def load(self):
        self.model.load()

    def unload(self):
        self.model.unload()

    def solve(self, request: R6Request) -> R6Result:
        if not isinstance(request, R6Request):
            raise TypeError("solve requires an answer-blind R6Request")
        started = time.perf_counter()
        try:
            controller = Controller(self, request)
        except ProtocolError:
            raise
        except (OSError, RuntimeError, ValueError) as exc:
            forced = request.answer_protocol == "forced_choice"
            return R6Result(
                request.request_id,
                request.choices[0].label if forced else None,
                "insufficient",
                forced,
                True,
                "TOOL_FAILURE",
                pending_gaps=[gap("local_fact", "Media/text initialization failed")],
                costs={
                    "model_calls": 0,
                    "tool_calls": 1,
                    "failed_calls": 0,
                    "audio_seconds": 0,
                    "end_to_end_seconds": time.perf_counter() - started,
                },
                trace={
                    "events": [
                        {
                            "event": "initialization_failure",
                            "reason": f"{type(exc).__name__}: {exc}",
                        }
                    ]
                },
            )
        controller.session.started = started
        return controller.run()


class Controller:
    def __init__(self, agent, request):
        self.request, self.config, self.model = request, agent.config, agent.model
        self.media = agent.media_factory(request, self.config)
        self.texts = TextSources(request, self.media.contract)
        self.audio = UnavailableAudioProvider()
        self.software = software_fingerprint(self.model, self.config)
        try:
            self.checkpoint = checkpoint_for(
                request, self.config, self.media, self.texts, self.software
            )
        except ValueError as exc:
            raise ProtocolError(str(exc)) from exc
        self.state = self.checkpoint.restored or new_state()
        self.session = Session(
            self.model, self.config, self.state, lambda: self.checkpoint.save(self.state)
        )
        if self.checkpoint.restored:
            self.media.restore(self.state.get("media_catalog", {}))
        self.state["protocol"] = asdict(self.media.contract)
        self.state["software"] = self.software
        self.state["text_issues"] = self.texts.issues

    def save(self):
        self.state["media_catalog"] = deepcopy(self.media.catalog)
        self.session.persist()

    @property
    def query(self):
        return self.state["query_spec"]

    def run(self):
        if self.state.get("result"):
            return R6Result(**self.state["result"])
        try:
            if self.request.mode != "pipeline":
                return self.baseline()
            while self.state["stop_reason"] is None:
                phase = self.state["phase"]
                if phase == "compile":
                    self.compile()
                elif phase == "initial":
                    queue = self.state["initial_actions"]
                    index = self.state.setdefault("initial_index", 0)
                    if index < len(queue):
                        self.execute(queue[index], f"initial-{index}")
                        self.state["initial_index"] += 1
                    else:
                        self.state["phase"] = "assess"
                elif phase == "assess":
                    assessment = self.assess()
                    self.state["assessments"].append(assessment)
                    self.state["relations"].update({r["id"]: r for r in assessment["relations"]})
                    self.state["gaps"] = assessment["gaps"]
                    self.add_program_gaps(assessment)
                    self.state["phase"] = "decide"
                elif phase == "decide":
                    self.decide()
                elif phase == "refine":
                    self.execute(
                        self.state["pending_action"], f"refine-{self.state['refinements']}"
                    )
                    self.state["refinements"] += 1
                    self.state["phase"] = "assess"
                self.save()
        except BudgetExhausted as exc:
            self.state["stop_reason"] = "BUDGET_EXHAUSTED"
            self.state["events"].append({"event": "budget_stop", "reason": str(exc)})
        except (ProtocolError, ModelFailure, OSError, RuntimeError, ValueError, KeyError) as exc:
            self.state["stop_reason"] = "TOOL_FAILURE"
            self.state["events"].append(
                {"event": "failure", "reason": f"{type(exc).__name__}: {exc}"}
            )
        return self.finish()

    def compile(self):
        payload = {
            "question": self.request.question,
            "choices": [asdict(c) for c in self.request.choices],
            "protocol": asdict(self.media.contract),
            "subtype_hint": self.request.subtype,
        }
        value = self.session.call(
            "compiler",
            "compiler",
            payload,
            validator=lambda v: validate_query(v, self.request, self.media.contract),
        )
        query = deepcopy(value)
        for option, original in zip(query["option_claims"], self.request.choices, strict=True):
            option["text"] = original.text
        if query["reference_interval"] is not None:
            for atom in query["atoms"]:
                if atom["story_time"] is None:
                    atom["story_time"] = list(query["reference_interval"])
        self.state["query_spec"] = query
        self.state["initial_actions"] = initial_actions(
            self.request, query, self.media, self.texts, self.config
        )
        self.state["phase"] = "initial"

    def observer_payload(self, target, *, overview=False):
        payload = {
            "question": self.request.question,
            "entities": self.query["entities"],
            "reference_scope": self.query["reference_scope"],
            "overview_only": overview,
        }
        if self.config.observer_options == "all":
            payload["choices"] = [asdict(c) for c in self.request.choices]
        elif self.config.observer_options == "neutral":
            payload["neutral_targets"] = self.query["discriminators"]
            # The checker may propose a biased query; never forward its answer-bearing prose.
            payload["requested_observation"] = self.query["discriminators"]
        return payload

    def observe(self, key, *, batch=None, text_sources=(), target="", overview=False, safe=False):
        if key in self.state.get("completed_observations", []):
            return
        prepared = self.media.prepare(batch, safe=safe) if batch else None
        sources = self.media.evidence(prepared) if prepared else {}
        sources.update({f"T{i + 1:02d}": s for i, s in enumerate(text_sources)})
        payload = self.observer_payload(target, overview=overview)
        try:
            value = self.session.call(
                key,
                "observer",
                payload,
                prepared=prepared,
                sources=sources,
                validator=lambda v: validate_observation(
                    v, sources, self.query, self.config.max_relevant_facts_per_call
                ),
            )
        except ContextOverflow:
            self.state["events"].append(
                {"event": "observation_split", "key": key, "reason": "measured_input_limit"}
            )
            if batch and len(batch.frames) > 1:
                middle = len(batch.frames) // 2
                for i, frames in enumerate((batch.frames[:middle], batch.frames[middle:])):
                    self.observe(
                        f"{key}/split{i}",
                        batch=replace(batch, frames=frames, ordered=False),
                        text_sources=text_sources,
                        target=target,
                        overview=overview,
                        safe=True,
                    )
            elif len(text_sources) > 1:
                middle = len(text_sources) // 2
                for i, rows in enumerate((text_sources[:middle], text_sources[middle:])):
                    self.observe(f"{key}/split{i}", batch=batch, text_sources=rows, target=target)
            elif batch and not safe:
                self.observe(
                    f"{key}/safe",
                    batch=batch,
                    text_sources=text_sources,
                    target=target,
                    overview=overview,
                    safe=True,
                )
            else:
                raise
            self.state.setdefault("completed_observations", []).append(key)
            self.save()
            return
        already_applied = key in self.state.get("applied_observations", [])
        add_observation(self.state, value, sources, job_key=key)
        if batch and not already_applied:
            self.state["coverage"].append(self.media.coverage(batch, overview=overview))
        if value["overflow"]:
            self.state["gaps"].append(
                gap("coverage", "Observation overflow: additional facts remain")
            )
            if batch and len(batch.frames) > 1:
                middle = len(batch.frames) // 2
                for i, frames in enumerate((batch.frames[:middle], batch.frames[middle:])):
                    self.observe(
                        f"{key}/facts{i}",
                        batch=replace(batch, frames=frames, ordered=False),
                        text_sources=text_sources,
                        target=target,
                        overview=overview,
                    )
        self.state.setdefault("completed_observations", []).append(key)
        self.save()

    def execute(self, proposed, key):
        validate_action(proposed, self.state, self.media.contract)
        record = next((a for a in self.state["actions"] if a["key"] == key), None)
        if record and record["status"] == "completed":
            return
        if record is None:
            identity = action_identity(proposed)
            if any(a["identity"] == identity for a in self.state["actions"]):
                raise ProtocolError("identical observation request blocked")
            record = {
                "key": key,
                "identity": identity,
                "request": proposed,
                "status": "started",
                "sources_before": sorted(self.state["sources"]),
            }
            self.state["actions"].append(record)
            self.save()
        kind, target = proposed["kind"], proposed["query"]
        if kind == "search_allowed_text":
            rows = self.session.tool(
                kind, proposed, lambda: self.texts.search(target, span=proposed["span"])
            )
            # Explicit bounded search result, with omitted IDs retained for later acquisition.
            unseen = [r for r in rows if r["id"] not in self.state["sources"]]
            rows = unseen or rows
            chosen = rows[:8]
            record["omitted_text_source_ids"] = [r["id"] for r in rows[8:]]
            if chosen:
                self.observe(key, text_sources=chosen, target=target)
            else:
                self.state["gaps"].append(
                    gap("modality_missing", "No usable aligned text", modality="subtitle")
                )
        elif kind == "analyze_audio_if_available":
            self.session.tool(kind, proposed, lambda: self.audio.observe())
            self.state["gaps"].append(
                gap("modality_missing", "Actual audio observer unavailable", modality="audio")
            )
        elif kind == "reduce_by_code":
            self.session.tool(kind, proposed, lambda: {"status": "already_reduced_in_assessment"})
        elif kind == "inspect_source_frame_or_crop":
            ids = proposed["source_ids"]
            if len(ids) > self.config.crop_frames_per_request:
                raise ProtocolError("crop request exceeds two source frames")
            frames = []
            for sid in ids:
                parent = self.state["sources"][sid]["source_frame_id"]
                frame = (
                    self.session.tool(
                        kind, proposed, lambda p=parent: self.media.crop(p, proposed["bbox"])
                    )
                    if proposed["bbox"]
                    else self.media.frame(parent)
                )
                frames.append(frame)
            batch = self.media.batch_for_sources([f.id for f in frames])
            self.observe(key, batch=batch, target=target)
        else:
            span = proposed["span"]
            if proposed.get("overview"):
                chunks = [(span, uniform_times(span, proposed["overview_count"]))]
            else:
                chunks = list(clip_batches(span, proposed["fps"], self.config.max_frames_per_call))
            for index, (window, times) in enumerate(chunks):
                batch = self.session.tool(
                    kind,
                    {"span": window, "times": times},
                    lambda w=window, ts=times: self.media.extract(
                        w, ts, fps=None if proposed.get("overview") else proposed["fps"]
                    ),
                )
                self.observe(
                    f"{key}/{index}",
                    batch=batch,
                    text_sources=self.texts.search("", span=window)[:8],
                    target=target,
                    overview=proposed.get("overview", False),
                )
        record["status"] = "completed"
        record["new_source_ids"] = sorted(
            set(self.state["sources"]) - set(record["sources_before"])
        )
        self.save()

    def assess(self):
        facts = self.state["facts"]
        relations = self.state["relations"]
        key = f"assessment-{len(self.state['assessments'])}"
        while True:
            payload = {
                "query_spec": self.query,
                "subtype_obligations": SUBTYPES[self.query["subtype"]],
                "facts": list(facts.values()),
                "relations": list(relations.values()),
                "permitted_intervals": self.media.contract.allowed_intervals,
                "coverage": self.state["coverage"],
                "pending_gaps": self.state["gaps"],
                "verification_feedback": [v["result"] for v in self.state["verifications"][-2:]],
                "omitted_fact_ids": sorted(set(self.state["facts"]) - facts.keys()),
                "omitted_relation_ids": sorted(set(self.state["relations"]) - relations.keys()),
            }

            def validate(value, facts=facts, relations=relations):
                return evaluate_assessment(
                    value,
                    self.query,
                    facts,
                    relations,
                    self.state["sources"],
                    coverage_complete=coverage_complete(self.state, self.media.contract),
                    relation_offset=len(self.state["relations"]),
                )

            try:
                value = self.session.call(key, "relation_checker", payload, validator=validate)
                return validate(value)
            except ContextOverflow:
                if len(facts) <= 1 and not relations:
                    raise
                # State is intact; only this view is reduced, with explicit omission metadata.
                facts = dict(list(facts.items())[-max(1, len(facts) // 2) :])
                relations = {}
                self.state["events"].append(
                    {"event": "assessment_context_reduced", "retained_facts": len(facts)}
                )
                key += "/reduced"

    def add_program_gaps(self, assessment):
        for ambiguity in self.query["ambiguities"]:
            self.state["gaps"].append(gap("input_ambiguity", ambiguity))
        available = {s["modality"] for s in self.state["sources"].values()}
        for atom in self.query["atoms"]:
            relevant = [
                c
                for c in assessment["candidates"]
                if c["selection_status"] != "contradicted"
                and c["answer_target_fit"] != "off_target"
                and any(
                    a["atom_id"] == atom["id"] and a["status"] == "unknown"
                    for a in c["atom_assessments"]
                )
            ]
            if relevant and "audio" in atom["required_modalities"] and "audio" not in available:
                self.state["gaps"].append(
                    gap(
                        "modality_missing",
                        atom["claim"],
                        modality="audio",
                        labels=[c["label"] for c in relevant],
                    )
                )
        if self.query["coverage"] != "local" and not coverage_complete(
            self.state, self.media.contract
        ):
            self.state["gaps"].append(
                gap("coverage", "Relevant global/occasion coverage remains incomplete")
            )
        for warning in assessment["normalizations"]:
            audio = warning.get("missing_modalities") == ["audio"]
            self.state["gaps"].append(
                gap(
                    "modality_missing" if audio else "local_fact",
                    "Unmet grounding for " + warning["atom_id"],
                    modality="audio" if audio else "video",
                )
            )

    def decide(self):
        assessment = self.state["assessments"][-1]
        ready = obligations_met(assessment, self.query) and not any(
            g["blocks_answer"] for g in self.state["gaps"]
        )
        winner = next(
            c for c in assessment["candidates"] if c["label"] == assessment["preferred_label"]
        )
        if (
            ready
            and winner["direct"]
            and not self.config.direct_channel
            and self.state["refinements"] == 0
        ):
            ready = False
        if ready:
            if not self.config.relation_verification:
                self.state["events"].append({"event": "verification_ablated"})
                self.state["stop_reason"] = "NO_PROGRESS"
                return
            if self.verify(assessment):
                self.state["stop_reason"] = "EVIDENCE_SUFFICIENT"
                return
        if self.query["ambiguities"]:
            self.state["stop_reason"] = "INPUT_AMBIGUITY"
            return
        if self.state["gaps"] and all(g["modality"] == "audio" for g in self.state["gaps"]):
            self.state["stop_reason"] = "MODALITY_UNAVAILABLE"
            return
        if not self.session.remaining():
            raise BudgetExhausted("no remaining acquisition/assessment budget")
        if self.state["refinements"] >= self.config.max_refinement_rounds:
            self.state["stop_reason"] = (
                "MODALITY_UNAVAILABLE"
                if any(g["modality"] == "audio" for g in self.state["gaps"])
                else "NO_PROGRESS"
            )
            return
        fallback = fallback_actions(self.state, self.query, self.media.contract, self.config)
        proposals = (
            assessment["actions"] + fallback
            if self.config.refinement_policy == "targeted"
            else fallback
        )
        chosen = select_action(proposals, self.state, self.media.contract, self.config)
        if chosen is None:
            self.state["stop_reason"] = (
                "MODALITY_UNAVAILABLE"
                if any(g["kind"] == "modality_missing" for g in self.state["gaps"])
                else "NO_PROGRESS"
            )
        else:
            self.state["pending_action"] = chosen
            self.state["phase"] = "refine"

    def verify(self, assessment):
        checks = verification_checks(
            assessment, self.query, self.state["facts"], self.state["relations"]
        )
        ids = list(dict.fromkeys(i for c in checks for i in c["source_ids"]))
        for competitor in assessment["competitors"]:
            for fid in competitor["fact_ids"]:
                ids.extend(self.state["facts"][fid]["source_ids"])
            for rid in competitor["relation_ids"]:
                ids.extend(relation_sources(rid, self.state["facts"], self.state["relations"]))
        ids = list(dict.fromkeys(ids))
        signature = digest({"checks": checks, "raw_sources": sorted(ids)})
        if any(v.get("signature") == signature for v in self.state["verifications"]):
            self.state["events"].append({"event": "identical_verification_blocked"})
            self.state["gaps"].append(
                gap("relation_bridge", "Obtain new evidence for unresolved verification")
            )
            return False
        visual = [i for i in ids if self.state["sources"][i]["modality"] == "video"]
        if len(visual) > self.config.max_frames_per_call:
            self.state["gaps"].append(
                gap("coverage", "Critical verification packet exceeds frame cap")
            )
            return False
        batch = self.media.batch_for_sources(visual)
        prepared = self.media.prepare(batch) if batch else None
        sources = self.media.evidence(prepared) if prepared else {}
        text = [self.state["sources"][i] for i in ids if i not in visual]
        sources.update({f"T{i + 1:02d}": s for i, s in enumerate(text)})
        aliases = {s["id"]: alias for alias, s in sources.items()}
        neutral_checks = [
            {**c, "source_ids": [aliases[i] for i in c["source_ids"]]} for c in checks
        ]
        payload = {
            "checks": neutral_checks,
            "original_question": self.request.question,
            "reference_scope": self.query["reference_scope"],
            "comparison_context": [o["text"] for o in self.query["option_claims"]],
        }

        def validate(value):
            if len(value["checks"]) != len(checks) or {c["check_id"] for c in value["checks"]} != {
                c["check_id"] for c in checks
            }:
                raise ProtocolError("verifier omitted or invented checks")
            for check in value["checks"]:
                if set(check["source_ids"]) - sources.keys():
                    raise ProtocolError("verifier references an unshown source")
                if check["state"] == "supported" and (not check["source_ids"] or check["missing"]):
                    raise ProtocolError(
                        "supported verification needs sources and no missing premise"
                    )

        try:
            value = self.session.call(
                f"verify-{len(self.state['assessments'])}",
                "verifier",
                payload,
                prepared=prepared,
                sources=sources,
                validator=validate,
                terminal=True,
            )
        except ContextOverflow:
            self.state["gaps"].append(
                gap("coverage", "Verification context exceeds measured token cap")
            )
            return False
        self.state["verifications"].append(
            {
                "checks_requested": checks,
                "result": value,
                "signature": signature,
                "source_aliases": aliases,
            }
        )
        self.state["gaps"].extend(value["gaps"])
        passed = all(c["state"] == "supported" for c in value["checks"]) and not value["gaps"]
        if not passed:
            self.state["gaps"].append(
                gap("relation_bridge", "Critical verification remains unresolved")
            )
        return passed

    def baseline(self):
        neutral = {
            "entities": [],
            "reference_scope": "Original question",
            "discriminators": [],
            "target_description": self.request.question,
            "initial_actions": [],
        }
        self.state.setdefault("query_spec", neutral)
        sources, prepared = {}, None
        if self.request.mode == "captions":
            from .acquisition import action

            for index, span in enumerate(windows(self.media.contract, self.config)):
                if not self.session.remaining():
                    break
                self.execute(
                    action("observe_clip", span=span, query=self.request.question),
                    f"caption-{index}",
                )
            self.state["events"].append(
                {
                    "event": "caption_coverage",
                    "complete": coverage_complete(self.state, self.media.contract),
                }
            )
        elif self.request.mode == "direct":
            frames = []
            actions = initial_actions(
                self.request,
                neutral,
                self.media,
                TextSources(
                    replace(self.request, subtitle_path=None, asr_path=None), self.media.contract
                ),
                self.config,
            )
            for proposed in actions:
                times = uniform_times(proposed["span"], proposed.get("overview_count", 8))
                batch = self.session.tool(
                    "direct_decode",
                    proposed,
                    lambda p=proposed, t=times: self.media.extract(p["span"], t),
                )
                frames.extend(batch.frames)
            if frames:
                prepared = self.media.prepare(self.media.batch_for_sources([f.id for f in frames]))
                sources = self.media.evidence(prepared)
            text = self.texts.search(self.request.question)[:8]
            sources.update({f"T{i + 1:02d}": s for i, s in enumerate(text)})
        payload = {
            "mode": self.request.mode,
            "question": self.request.question,
            "choices": [asdict(c) for c in self.request.choices],
            "captions": list(self.state["facts"].values()),
        }

        def validate(value):
            if value["preferred_label"] not in {c.label for c in self.request.choices}:
                raise ProtocolError("unknown answer label")
            if set(value["source_ids"]) - sources.keys():
                raise ProtocolError("baseline cited unshown raw source")

        value = self.session.call(
            "baseline-answer",
            "answer",
            payload,
            prepared=prepared,
            sources=sources,
            validator=validate,
            terminal=True,
        )
        self.state["baseline_prediction"] = value["preferred_label"]
        self.state["baseline_source_ids"] = [sources[i]["id"] for i in value["source_ids"]]
        self.state["sources"].update({s["id"]: s for s in sources.values()})
        self.state["stop_reason"] = "NO_PROGRESS"
        self.state["events"].append(
            {"event": "unverified_baseline_complete", "limitations": value["limitations"]}
        )
        return self.finish()

    def finish(self):
        assessment = self.state["assessments"][-1] if self.state["assessments"] else None
        preferred = (
            assessment["preferred_label"] if assessment else self.state.get("baseline_prediction")
        )
        technical = preferred is None
        sufficient = self.state["stop_reason"] == "EVIDENCE_SUFFICIENT"
        prediction = preferred or self.request.choices[0].label
        if not sufficient and self.request.answer_protocol == "allow_abstention":
            prediction = None
        self.save()
        trace = {
            k: deepcopy(v)
            for k, v in self.state.items()
            if k not in {"result", "jobs", "media_catalog"}
        }
        cited = (
            next(c["source_ids"] for c in assessment["candidates"] if c["label"] == preferred)
            if assessment
            else self.state.get("baseline_source_ids", [])
        )
        result = R6Result(
            self.request.request_id,
            prediction,
            "sufficient" if sufficient else "insufficient",
            bool(prediction is not None and not sufficient),
            technical,
            self.state["stop_reason"] or "NO_PROGRESS",
            deepcopy(self.state["gaps"]),
            sorted(cited),
            self.session.resources(),
            trace,
        )
        self.state["result"] = result.to_dict()
        self.save()
        return result
