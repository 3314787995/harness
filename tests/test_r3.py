from __future__ import annotations

import copy
import itertools
import json
from dataclasses import replace

import pytest
from PIL import Image
from r3_observation_fakes import event_report, visual_report

from qwen3vl_agent.coarse_to_fine.types import FrameRef
from qwen3vl_agent.models.base import BaseVideoModel, ModelOutput
from qwen3vl_agent.p01 import P01Config, TimeSpan
from qwen3vl_agent.p01.media import SourceFrameStore, VideoMetadata
from qwen3vl_agent.r3 import (
    ExternalSegment,
    ProviderResult,
    R3Budget,
    R3Config,
    R3Request,
    R3VideoAgent,
    TemporalProviderResult,
)
from qwen3vl_agent.r3.checkpoint import Checkpoint
from qwen3vl_agent.r3.ledger import EventLedger
from qwen3vl_agent.r3.observation import parse_batch
from qwen3vl_agent.r3.planning import make_tiles, relative_scope, sample_times
from qwen3vl_agent.r3.reduce import duration_bounds, ordered_select, temporal_reduce, union_length
from qwen3vl_agent.r3.types import (
    Bracket,
    CoverageTile,
    EventQuery,
    EventRecord,
    Operation,
    ProtocolError,
)


def query(op="count_occurrences", *, unit="action_cycle", scope=None, targets=None, **operation):
    return {
        "targets": targets
        or [
            {
                "target_id": "target",
                "description": "place a cup and release it",
                "unit_kind": unit,
                "completion_criterion": "the hand releases the cup",
                "reset_criterion": "a new placement begins",
                "inclusion_rule": "completes_inside" if unit == "action_cycle" else "intersects",
            }
        ],
        "operations": [{"operation_id": "answer", "op": op, "target_ids": ["target"], **operation}],
        "scope": scope or {"kind": "full"},
        "unresolved": [],
        "version": 2,
    }


def truth_observer(truth):
    """Synthetic phase detector; truth is confined to the test fake, never runtime inputs."""

    def observe(payload):
        frames = [f for f in payload["frames"] if not f.get("crop_transform")]
        events = []
        for i, occurrence in enumerate(truth):
            start, end = occurrence["span"]
            active = [f for f in frames if start <= f["timestamp_sec"] < end]
            if not active:
                continue
            before = [f for f in frames if f["timestamp_sec"] < start]
            after = [f for f in frames if f["timestamp_sec"] >= end]
            target = occurrence.get("target", "target")
            if target not in {t["target_id"] for t in payload["query"]["targets"]}:
                continue
            evidence = [active[0]["id"], active[-1]["id"]]
            events.append(
                {
                    "local_id": f"local_{i}",
                    "target_id": target,
                    "actor_ref": occurrence.get("actor", "foreground person"),
                    "object_ref": "cup",
                    "description": occurrence.get("description", f"event {i}"),
                    "category": occurrence.get("category", "cup placement"),
                    "fact_kind": "visual_event",
                    "evidence_refs": evidence,
                    "before_start_refs": [before[-1]["id"]] if before else [],
                    "start_refs": [active[0]["id"]],
                    "last_active_refs": [active[-1]["id"]],
                    "completion_refs": [after[0]["id"]] if after else [],
                    "after_end_refs": [after[0]["id"]] if after else [],
                    "reset_refs": [after[0]["id"]] if after else [],
                    "completed": bool(after),
                    "match": "clear",
                    "replay_status": occurrence.get("replay_status", "original"),
                    "attributes": {
                        k: {"value": v, "evidence_refs": evidence}
                        for k, v in occurrence.get("attributes", {}).items()
                    },
                    "cooccurrence": {
                        k: {"status": v, "evidence_refs": evidence if v != "unknown" else []}
                        for k, v in occurrence.get("cooccurrence", {}).items()
                    },
                    "unresolved_reasons": [],
                }
            )
        return {
            "events": events,
            "unresolved": [],
            "truncated": False,
            "observation_status": "valid",
            "crop_requests": [],
        }

    return observe


def relation(payload):
    a, b = payload["left"], payload["right"]
    same = a["description"] == b["description"] and a["actor_ref"] == b["actor_ref"]
    return {
        "relation": "same_occurrence" if same else "distinct_occurrences",
        "evidence_refs": [payload["frames"][0]["id"]],
        "reason": "synthetic continuity" if same else "synthetic independent completions",
    }


def final(payload):
    return {
        "prediction": payload["choices"][0]["label"] if payload["choices"] else "Observed event.",
        "evidence_refs": payload["value_state"].get("evidence_refs", [])[:2],
    }


class FakeModel(BaseVideoModel):
    def __init__(self, compiled=None, truth=(), **handlers):
        super().__init__("Qwen/Qwen3-VL-8B-Instruct")
        sample = compiled or query()
        if isinstance(sample, list):
            sample = next((item for item in sample if isinstance(item, dict) and item.get("targets")), query())
        intent = {
            "version": 2,
            "targets": [{k: t.get(k, "appearance_episode" if k == "unit_kind" else "target")
                         for k in ("target_id", "description", "unit_kind")} for t in sample["targets"]],
            "tasks": copy.deepcopy(sample["operations"]), "scope": copy.deepcopy(sample["scope"]),
            "needs_candidate_union": False, "unresolved": [],
        }
        self.handlers = {
            "compile_intent": intent,
            "compile": compiled or query(),
            "observe": truth_observer(truth),
            "observe_visual": visual_report,
            "relation": relation,
            "repair": "{}",
            "final": final,
            **handlers,
        }
        self.calls = []

    def load(self):
        self._loaded = True

    def unload(self):
        self._loaded = False

    def generate(self, messages, **kwargs):
        content = messages[-1]["content"]
        text = next(p["text"] for p in content if p.get("text", "").startswith("R3:"))
        role = text.splitlines()[0].split(":")[1]
        payload = json.loads(text.split("INPUT_JSON:\n")[1])
        self.calls.append({"role": role, "payload": payload, "content": content, "kwargs": kwargs})
        handler = self.handlers.get(role, self.handlers["observe"] if role == "observe_events" else None)
        value = handler.pop(0) if isinstance(handler, list) else handler
        if isinstance(value, BaseException):
            raise value
        value = value(payload) if callable(value) else value
        if role == "observe_events" and "observe_events" not in self.handlers:
            if isinstance(value, dict):
                value = event_report(payload, value)
            elif isinstance(value, ModelOutput):
                try:
                    value = ModelOutput(json.dumps(event_report(payload, json.loads(value.text))), value.metadata)
                except (ValueError, KeyError):
                    pass
        if isinstance(value, ModelOutput):
            return value
        return ModelOutput(
            value if isinstance(value, str) else json.dumps(value),
            {"output_tokens": 64, "input_tokens": 256},
        )


class Probe:
    def __init__(self, duration):
        self.duration = duration

    def probe(self, path):
        return VideoMetadata(str(path), self.duration, 24, 320, 192)


class Store(SourceFrameStore):
    def __init__(self, config, image):
        super().__init__(config)
        self.image = image
        self.requests = []
        self.extra = ()

    def extract(self, path, timestamps, *, purpose, max_side=None):
        times = list(timestamps)
        self.requests.append(times)
        return (
            tuple(
                FrameRef(f"SRC-{round(t * 1000000):012d}", round(t, 6), str(self.image))
                for t in times
            )
            + self.extra
        )


@pytest.fixture
def setup(tmp_path):
    image = tmp_path / "source.png"
    Image.new("RGB", (320, 192), "red").save(image)
    video = tmp_path / "video.mp4"
    video.write_bytes(b"synthetic media identity; fake decoder injected")

    def build(model=None, *, duration=16, provider=None, **kwargs):
        media = P01Config(cache_dir=str(tmp_path / "cache"))
        config = R3Config(media=media, **kwargs)
        store = Store(media, image)
        model = model or FakeModel()
        agent = R3VideoAgent(
            model,
            config=config,
            index_builder=Probe(duration),
            source_store=store,
            provider=provider,
        )
        return agent, model, store, str(video)

    return build


def run(setup, model, *, duration=16, **kwargs):
    agent, model, store, path = setup(model, duration=duration)
    result = agent.solve(R3Request(path, "Answer the temporal question.", **kwargs))
    return result, model, store


































def event(
    key,
    start,
    end,
    *,
    actor="person",
    category="target",
    status="accepted",
    co=None,
    target="target",
):
    return EventRecord(
        key,
        target,
        actor,
        "cup",
        key,
        category,
        "appearance_episode",
        "visual_event",
        Bracket.parse(start),
        Bracket.parse(end),
        (start[1] or 0, end[0] or 0),
        [key + "-ref"],
        completion_evidence_refs=[key + "-completion"],
        reset_evidence_refs=[key + "-reset"],
        observation_ids=["one_synthetic_observation"],
        member_ids=[key],
        status=status,
        completed=True,
        cooccurrence=co or {},
        left_censored=start[0] is None,
        right_censored=end[1] is None,
        replay_status="original",
    )


def synthetic_ledger(events, operation=None, *, scope=(0, 30), targets=None):
    compiled = EventQuery.from_dict(query(unit="appearance_episode", targets=targets))
    if operation:
        compiled = replace(compiled, operations=(operation,))
    tile = CoverageTile(
        "tile",
        scope,
        scope,
        4,
        observed=True,
        resolution_met=True,
        audit_needed=False,
        audit_done=True,
    )
    ledger = EventLedger(compiled, [tile])
    ledger.proposals = {e.event_id: e for e in events}
    ledger.reconcile()
    return ledger
