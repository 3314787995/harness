"""Coverage-first factual synthesis with directly retained Composer answers."""

from __future__ import annotations

from dataclasses import asdict
import time
from typing import Any

from qwen3vl_agent.models.base import BaseVideoModel, ModelOutput
from qwen3vl_agent.p01.types import TimeSpan
from qwen3vl_agent.r1.config import R1Config
from qwen3vl_agent.r3.media import R3SourceFrameStore
from qwen3vl_agent.r5.checkpoint import Checkpoint, file_digest, implementation_digest
from qwen3vl_agent.r5.config import R5Config
from qwen3vl_agent.r5.ledger import FactCardStore, parse_observation
from qwen3vl_agent.r5.media import R5Media
from qwen3vl_agent.r5.observation import PROTOCOL_VERSION, compact_catalog
from qwen3vl_agent.r5.planning import (
    estimate,
    make_plan,
    merge_calls,
    resolve_source,
    split_tile,
    union_duration,
)
from qwen3vl_agent.r5.providers import ProviderAdapter
from qwen3vl_agent.r5.runtime import ModelSession, NeedsSplit, RunContext
from qwen3vl_agent.r5.synthesis import build_tree, compose, fallback_root
from qwen3vl_agent.r5.types import (
    BudgetExhausted,
    ProtocolError,
    R5Budget,
    R5Request,
    R5Result,
    SummarySpec,
)


def pending_receipt(ctx: RunContext, tile: dict, phase: str) -> bool:
    return any(c["role"] == "observe" and c.get("segment_id") == tile["segment_id"]
               and c.get("attempt_phase") == phase and c["status"] == "returned"
               and not c.get("observation_committed") for c in ctx.calls)


class R5VideoAgent:
    def __init__(
        self,
        model: BaseVideoModel,
        config: R5Config | dict | None = None,
        provider: Any = None,
        *,
        index_builder: Any = None,
        source_store: Any = None,
    ):
        self.model, self.provider = model, provider
        self.config = config if isinstance(config, R5Config) else R5Config.from_mapping(config)
        self.config.validate()
        self.media = R5Media(
            R1Config(media=self.config.media),
            index_builder,
            source_store or R3SourceFrameStore(self.config.media),
        )

    def load(self) -> None:
        if not self.model.is_loaded:
            self.model.load()

    def unload(self) -> None:
        self.model.unload()

    def generate(
        self,
        messages: list[dict],
        *,
        videos: list[str] | None = None,
        images: list[str] | None = None,
        choices: Any = None,
        given_interval: Any = None,
        **kwargs: Any,
    ) -> ModelOutput:
        if not videos or len(videos) != 1 or images:
            raise ValueError("R5 requires exactly one video and no independent images")
        users = [m for m in messages if m.get("role") == "user"]
        if not users or not isinstance(users[-1].get("content"), str):
            raise ValueError("R5 requires a text question")
        if given_interval is not None:
            if kwargs.get("query_scope") is not None and list(kwargs["query_scope"]) != list(
                given_interval
            ):
                raise ValueError("given_interval conflicts with query_scope")
            kwargs["query_scope"] = given_interval
        files = list(kwargs.pop("external_files", ()))
        modalities = set(kwargs.pop("available_modalities", ("video", "screen_text")))
        for key, kind in (("subtitle_path", "subtitle"), ("asr_path", "asr")):
            path = kwargs.pop(key, None)
            if path:
                files.append({"path": path, "kind": kind})
                modalities.add(kind)
        kwargs.setdefault("budget", self.config.budget)
        result = self.solve(
            R5Request(
                video_path=videos[0],
                question=users[-1]["content"],
                choices=choices or (),
                external_files=tuple(files),
                available_modalities=tuple(sorted(modalities)),
                **kwargs,
            )
        )
        return ModelOutput(result.prediction, {"r5": result.to_dict()})

    def solve(self, request: R5Request) -> R5Result:
        caps = {}
        for key, value in asdict(request.budget).items():
            values = [v for v in (value, getattr(self.config.budget, key)) if v is not None]
            caps[key] = (
                max(values) if key == "terminal_call_reserve" else min(values) if values else None
            )
        ctx = RunContext(R5Budget(**caps))
        session = ModelSession(self.model, self.media, self.config, ctx)
        work = {
            "protocol_version": PROTOCOL_VERSION,
            "recovery_queue": [],
            "base_pass_done": False,
            "spec": None,
            "tiles": {},
            "store": FactCardStore().state,
            "provider_cache": {},
            "issues": [],
            "draft": None,
            "synthesis_done": False,
            "result": None,
            "tree_depth": 0,
            "hierarchy_complete": True,
        }
        try:
            source, scope = resolve_source(request, self.media)
        except (OSError, ValueError, RuntimeError, ImportError) as exc:
            return R5Result(
                "",
                request.execution_subtype or "factual_video_summary",
                "input_error",
                "none",
                "no_answer",
                {"complete": False},
                [],
                [str(exc)],
                ctx.summary(),
                {},
            )
        fingerprint_request = asdict(request)
        for key in ("checkpoint_path", "resume"):
            fingerprint_request.pop(key)
        external = []
        for file in request.external_files:
            try:
                external.append(file_digest(file.path))
            except OSError:
                external.append({"unavailable": file.path})
        if (
            request.resume
            and self.provider is not None
            and getattr(self.provider, "version", None) is None
        ):
            raise ValueError("resuming an injected provider requires a stable provider.version")
        checkpoint = Checkpoint(
            request.checkpoint_path,
            {
                "request": fingerprint_request,
                "media": source["source_id"],
                "external": external,
                "config": asdict(self.config),
                "implementation": implementation_digest(),
                "model_path": self.model.model_path,
                "model_type": type(self.model).__qualname__,
                "model_configuration": {
                    k: getattr(self.model, k, None)
                    for k in (
                        "device",
                        "dtype",
                        "attn_implementation",
                        "device_map",
                        "max_memory",
                        "generation",
                        "video",
                        "video_config",
                        "generation_config",
                        "revision",
                    )
                },
                "provider_type": type(self.provider).__qualname__,
                "provider_version": getattr(self.provider, "version", None),
            },
            resume=request.resume,
        )
        if checkpoint.restored:
            work = checkpoint.restored["work"]
            if work.get("protocol_version") != PROTOCOL_VERSION:
                raise ValueError("R5 protocol changed; use a new result/checkpoint directory")
            for key, value in checkpoint.restored["context"].items():
                setattr(ctx, key, value)
            ctx.clock_mark = time.monotonic()
            # A hard-killed in-flight call has unknown duration. Charge conservatively;
            # ordinary interrupts have already saved their measured elapsed time.
            for call in ctx.calls:
                if call["status"] == "started":
                    ctx.elapsed_sec += min(ctx.budget.max_elapsed_sec, max(0, time.time() - call["started_at_utc"]))
                    call.update(status="interrupted", error="unfinished_call_on_resume")
            for tile in work["tiles"].values():
                if tile["status"] == "observing":
                    replayable = any(c["role"] == "observe" and c.get("segment_id") == tile["segment_id"]
                                     and c["status"] == "returned" and not c.get("observation_committed") for c in ctx.calls)
                    tile.update(status="pending" if replayable else "incomplete", error="interrupted_observation")
            for entry in work["recovery_queue"]:
                if entry["status"] == "running":
                    root = work["tiles"][entry["segment_id"]]
                    entry["status"] = ("pending" if entry["mode"] == "split" or root["status"] == "pending"
                                       else "observed" if root["status"] == "observed" else "incomplete")
            if work["result"]:
                return R5Result(**work["result"])
        store = FactCardStore(work["store"])

        def save() -> None:
            checkpoint.save({"work": work, "context": ctx.snapshot()})

        ctx.on_change = save
        adapter = ProviderAdapter(self.provider, source, self.config, ctx, work["provider_cache"])
        if work.get("scope_interval"):
            scope = TimeSpan(*work["scope_interval"])
        if not work["tiles"]:
            plan = make_plan(scope, self.config, source["source_fps"])
            work["tiles"] = {t["segment_id"]: t for t in plan}
            work["estimate"] = estimate(plan, self.config, request.available_modalities)
            work["estimate"]["within_model_cap"] = (
                work["estimate"]["planned_model_calls"] <= ctx.limit()
            )
            save()
        try:
            if work["spec"] is None:
                ctx.required_reserve = 2
                result = session.call(
                    "compile",
                    {
                        "question": request.question,
                        "execution_subtype": request.execution_subtype,
                        "query_scope": [scope.start_seconds, scope.end_seconds],
                        "available_modalities": request.available_modalities,
                        "output_language": request.output_language,
                    },
                    parser=lambda d: SummarySpec.parse(d, request),
                )
                work["spec"] = asdict(result.value)
                save()
            if not work.get("scope_interval"):
                compiled_scope = work["spec"].get("scope_interval")
                if request.query_scope is None and compiled_scope is not None:
                    candidate = TimeSpan(*compiled_scope)
                    if (
                        candidate.start_seconds < scope.start_seconds
                        or candidate.end_seconds > scope.end_seconds
                    ):
                        raise ProtocolError("compiled query interval exceeds allowed scope/cutoff")
                    scope = candidate
                    plan = make_plan(scope, self.config, source["source_fps"])
                    work["tiles"] = {t["segment_id"]: t for t in plan}
                    work["estimate"] = estimate(plan, self.config, request.available_modalities)
                    work["estimate"]["within_model_cap"] = (
                        work["estimate"]["planned_model_calls"] <= ctx.limit()
                    )
                work["scope_interval"] = [scope.start_seconds, scope.end_seconds]
                save()
            if not work["synthesis_done"]:
                self._map(request, source, scope, work, store, session, adapter)
                work["draft"] = self._synthesize(request, work, store, session)
                work["synthesis_done"] = True
                save()
        except (
            BudgetExhausted,
            NeedsSplit,
            ProtocolError,
            ValueError,
            TypeError,
            KeyError,
        ) as exc:
            work["issues"].append(f"execution_incomplete:{exc}")
        except BaseException as exc:
            work["fatal_error"] = {"type": type(exc).__name__, "message": str(exc)}
            save()
            raise
        result = self._finish(request, scope, work, store, ctx)
        work["result"] = result.to_dict()
        save()
        return result

    def _observe(
        self,
        request: R5Request,
        source: dict,
        scope: TimeSpan,
        tile: dict,
        work: dict,
        store: FactCardStore,
        session: ModelSession,
        adapter: ProviderAdapter,
        *,
        reason: str | None = None,
        phase: str = "base",
        compact: bool = False,
    ) -> dict:
        if not pending_receipt(session.context, tile, phase) and not session.context.can_observe():
            raise BudgetExhausted("visual_time_or_call_budget")
        tile.update(status="observing", attempt_phase=phase)
        session.context.changed()
        catalog, batch = {}, None
        quality = {"required_resolution_met": True, "errors": [], "source_frame_ids": []}
        if set(request.available_modalities) & {"video", "screen_text"}:
            batch, catalog, quality = self.media.observe_tile(source, tile)
            session.context.decoded_frames += len(batch.frames)
            if not batch.frames:
                raise NeedsSplit("decoder_returned_no_permitted_frames")
        text_state, provider_issues = {}, []
        for kind in sorted(set(request.available_modalities) & {"asr", "subtitle"}):
            reading = adapter.fetch(tuple(tile["context"]), kind)
            text_state[kind] = {k: v for k, v in reading.items() if k != "items"}
            provider_issues.extend(reading["issues"])
            if "provider_truncated" in reading["issues"]:
                raise NeedsSplit("provider_truncated")
            for item in reading["items"]:
                if item["start_sec"] < scope.start_seconds or item["end_sec"] > scope.end_seconds:
                    provider_issues.append("text_crosses_query_boundary")
                    text_state[kind]["complete"] = False
                    continue
                key = f"text:{source['source_id'][:12]}:{kind}:{item['segment_id']}"
                catalog[key] = {
                    **item,
                    "id": key,
                    "alignment_error_sec": reading["alignment_error_sec"],
                    "independence_group": f"speech:{source['source_id']}",
                }
        previous = store.state["active_cards"].get(tile["segment_id"])
        prior = store.state["cards"][previous]["fact_ids"] if previous and compact else []
        prior = [
            f
            for f in prior
            if any(
                (tile["context"][0] <= store.state["catalog"][r]["start_sec"] < tile["context"][1])
                if store.state["catalog"][r]["kind"] == "frame"
                else (
                    store.state["catalog"][r]["start_sec"] < tile["context"][1]
                    and store.state["catalog"][r]["end_sec"] > tile["context"][0]
                )
                for r in store.state["facts"][f]["evidence_refs"]
            )
        ][-self.config.max_facts_per_card :]
        visible_catalog, aliases = compact_catalog(catalog)
        fact_limit = self.config.recovery_facts if compact else self.config.max_facts_per_card
        payload = {
            "question": request.question,
            "spec": work["spec"],
            "segment_id": tile["segment_id"],
            "core": tile["core"],
            "context": tile["context"],
            "catalog": visible_catalog,
            "fact_limit": fact_limit,
            "attempt_phase": phase,
            "output_language": request.output_language,
            "recovery_question": reason,
            "prior_facts": [store.public_fact(f) for f in prior],
            "input_issues": provider_issues,
        }

        def parser(data: dict) -> dict:
            return parse_observation(data, catalog, tile, self.config, set(prior),
                                     aliases=aliases, fact_limit=fact_limit,
                                     permitted_modalities=set(request.available_modalities))

        if not catalog:
            raise ProtocolError("no_available_segment_evidence")
        result = session.call("observe", payload, batch=batch, parser=parser, aliases=aliases,
                              tokens_override=self.config.recovery_tokens if compact else None)
        required = set(work["spec"].get("required_modalities", []))
        required_missing = required - set(request.available_modalities)
        required_text_missing = any(
            not text_state.get(k, {}).get("complete") for k in required & {"asr", "subtitle"}
        )
        if batch is None:
            quality["required_resolution_met"] = bool(text_state) and all(
                v["complete"] for v in text_state.values()
            )
        coverage_ok = bool(
            quality["required_resolution_met"]
            and not quality["errors"]
            and not required_missing
            and not required_text_missing
        )
        store.commit(
            tile, result.value, catalog, result.call_id, coverage_ok=coverage_ok,
            resolve_coverage=compact,
        )
        tile["quality"] = quality
        tile["text_state"] = text_state
        tile["input_issues"] = provider_issues
        tile["status"] = (
            "observed" if coverage_ok and not result.value["truncated"] else "incomplete"
        )
        next(c for c in session.context.calls if c["call_id"] == result.call_id)["observation_committed"] = True
        session.context.changed()
        return result.value

    def _map(
        self,
        request: R5Request,
        source: dict,
        scope: TimeSpan,
        work: dict,
        store: FactCardStore,
        session: ModelSession,
        adapter: ProviderAdapter,
    ) -> None:
        ctx = session.context
        roots = sorted((t for t in work["tiles"].values() if t["depth"] == 0), key=lambda t: t["core"])

        def reserve(extra: int = 1) -> None:
            pages = len(store.pages(self.config.max_facts_per_card)) + extra
            ctx.required_reserve = merge_calls(pages, self.config.merge_fan_in) + 2

        def attempt(tile: dict, *, phase: str, compact: bool = False) -> None:
            try:
                self._observe(request, source, scope, tile, work, store, session, adapter,
                              phase=phase, compact=compact,
                              reason="Fill remaining major-content gaps using prior_facts." if compact else None)
            except NeedsSplit as exc:
                if tile["depth"] >= 1 and str(exc) == "oom_split_required":
                    raise RuntimeError("R5 OOM persisted after the allowed input split") from exc
                tile.update(status="incomplete", error=str(exc))
            except (ProtocolError, BudgetExhausted) as exc:
                tile.update(status="incomplete", error=str(exc))
            ctx.changed()

        if not work["base_pass_done"]:
            for tile in roots:
                if tile["status"] != "pending":
                    continue
                reserve()
                if not pending_receipt(ctx, tile, "base") and not ctx.can_observe():
                    tile.update(status="unobserved", error="base_coverage_budget_limited")
                    work["issues"].append("base_coverage_budget_limited")
                    ctx.changed()
                    continue
                attempt(tile, phase="base")
            work["base_pass_done"] = True
            ctx.changed()

        queued = {entry["segment_id"] for entry in work["recovery_queue"]}
        for tile in roots:
            if tile["status"] != "incomplete" or tile["segment_id"] in queued:
                continue
            mode = "split" if tile.get("error") in {
                "oom_split_required", "per_call_frame_cap", "text_context_budget"
            } else "compact"
            work["recovery_queue"].append({"segment_id": tile["segment_id"], "mode": mode,
                                            "status": "pending", "reason": tile.get("error", "observation_incomplete")})

        def priority(entry: dict) -> tuple:
            tile = work["tiles"][entry["segment_id"]]
            card_id = store.state["active_cards"].get(entry["segment_id"])
            has_facts = bool(card_id and store.card_facts(store.state["cards"][card_id]))
            boundary = tile["core"][0] == scope.start_seconds or tile["core"][1] == scope.end_seconds
            return has_facts, not boundary, tile["core"]

        work["recovery_queue"].sort(key=priority)
        ctx.changed()
        for entry in work["recovery_queue"]:
            if entry["status"] != "pending":
                continue
            root = work["tiles"][entry["segment_id"]]
            if entry["mode"] == "split" and root.get("children"):
                existing_children = [work["tiles"][key] for key in root["children"]]
                if not any(t["status"] == "pending" for t in existing_children):
                    entry["status"] = "observed" if all(t["status"] == "observed" for t in existing_children) else "incomplete"
                    ctx.changed()
                    continue
            reserve(2 if entry["mode"] == "split" else 1)
            replay_tiles = [work["tiles"][key] for key in root.get("children", [])] or [root]
            replayable = any(pending_receipt(ctx, t, "recovery") for t in replay_tiles)
            if not replayable and (ctx.phase_calls("recovery") >= self.config.max_recovery_calls or not ctx.can_observe()):
                entry["status"] = "budget_limited"
                work["issues"].append("base_recovery_budget_limited")
                ctx.changed()
                continue
            entry["status"] = "running"
            ctx.changed()
            if entry["mode"] == "compact":
                attempt(root, phase="recovery", compact=True)
                entry["status"] = "observed" if root["status"] == "observed" else "incomplete"
            else:
                children = ([work["tiles"][key] for key in root["children"]]
                            if root.get("children") else split_tile(root, self.config))
                if not children:
                    if entry["reason"] == "oom_split_required":
                        raise RuntimeError("R5 OOM cannot be recovered within the input split limit")
                    entry["status"] = "incomplete"
                    continue
                root.update(status="split", children=[t["segment_id"] for t in children], split_reason=entry["reason"])
                work["tiles"].update({t["segment_id"]: t for t in children})
                ctx.changed()
                for child in children:
                    if child["status"] != "pending":
                        continue
                    reserve()
                    if not pending_receipt(ctx, child, "recovery") and (
                            ctx.phase_calls("recovery") >= self.config.max_recovery_calls or not ctx.can_observe()):
                        child.update(status="unobserved", error="base_recovery_budget_limited")
                        work["issues"].append("base_recovery_budget_limited")
                        continue
                    attempt(child, phase="recovery")
                entry["status"] = "observed" if all(t["status"] == "observed" for t in children) else "incomplete"
            ctx.changed()

    def _synthesize(
        self, request: R5Request, work: dict, store: FactCardStore, session: ModelSession
    ) -> dict | None:
        if not any(store.card_facts(c) for c in store.cards()):
            work["issues"].append("no_observed_facts")
            return None
        try:
            root = build_tree(store, session, work, request)
        except (BudgetExhausted, NeedsSplit, ProtocolError, ValueError, TypeError, KeyError) as exc:
            work["issues"].append(f"merge_incomplete:{exc}")
            work["hierarchy_complete"] = False
            root = fallback_root(store, self.config)
        work["root"] = root["id"]
        try:
            return compose(store, root, session, work, request)
        except (BudgetExhausted, NeedsSplit, ProtocolError, ValueError, TypeError, KeyError) as exc:
            work["issues"].append(f"composer_incomplete:{exc}")
            return None

    def _finish(
        self, request: R5Request, scope: TimeSpan, work: dict, store: FactCardStore, ctx: RunContext
    ) -> R5Result:
        tiles = [t for t in work["tiles"].values() if t["status"] != "split"]
        observed = [t["core"] for t in tiles if t["status"] == "observed"]
        visited = [t["core"] for t in tiles if "quality" in t]
        coverage = {
            "query_scope": [scope.start_seconds, scope.end_seconds],
            "planned_core_count": len(tiles),
            "visited_core_fraction": union_duration(visited) / scope.duration_seconds,
            "valid_core_fraction": union_duration(observed) / scope.duration_seconds,
            "complete": bool(tiles) and all(t["status"] == "observed" for t in tiles)
                        and all(c["coverage_status"] == "observed_under_policy" for c in store.cards()),
            "missing_intervals": [t["core"] for t in tiles if t["status"] != "observed"],
            "actual_max_frame_gap_sec": max(
                (t.get("quality", {}).get("actual_max_frame_gap_sec", 0) for t in tiles), default=0
            ),
            "policy_only": True,
        }
        draft = work.get("draft")
        issues = list(work["issues"])
        issues.extend((work.get("spec") or {}).get("unresolved", []))
        issues.extend(x for c in store.cards() for x in c["unresolved"])
        prediction, used_facts = "", set()
        if draft is not None:
            issues.extend(draft.get("unresolved", []))
            prediction = (draft["prediction"] if request.choices else
                          " ".join(c["statement"] for c in draft["claims"]))
            for claim in draft["claims"]:
                for ref in claim["support_refs"]:
                    used_facts.update(store.leaf_ids(ref))
        if not prediction:
            issues.append("no_valid_answer")
        evidence = sorted({ref for fact in used_facts
                           for ref in store.state["facts"][fact]["evidence_refs"]})
        support = "unverified" if evidence else "none"
        basis = "composer" if prediction else "no_answer"
        complete = bool(prediction and coverage["complete"] and work["hierarchy_complete"] and not issues)
        completion = ("complete" if complete else "budget_limited"
                      if not prediction and any("budget" in i for i in issues) else "partial")
        trace = {
            "protocol_version": PROTOCOL_VERSION,
            "recovery_queue": work["recovery_queue"],
            "base_pass_done": work["base_pass_done"],
            "recovery_calls": ctx.phase_calls("recovery"),
            "request_id": request.request_id, "video_id": request.video_id,
            "group_id": request.group_id, "native_labels": list(request.native_labels),
            "spec": work["spec"], "tiles": work["tiles"], "fact_store": store.state,
            "composer": draft, "tree_depth": work["tree_depth"],
            "estimate": work.get("estimate"), "root": work.get("root"),
            "reference_validation": "deterministic", "semantic_support": "not_verified",
        }
        return R5Result(
            prediction,
            (work["spec"] or {}).get("operation", request.execution_subtype or "factual_video_summary"),
            completion, support, basis, coverage, evidence, list(dict.fromkeys(issues)), ctx.summary(), trace,
        )
