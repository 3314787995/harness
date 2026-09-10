from __future__ import annotations

import json
from dataclasses import replace
from itertools import pairwise

import pytest
from PIL import Image

from qwen3vl_agent.coarse_to_fine.types import FrameRef
from qwen3vl_agent.models.base import BaseVideoModel, ModelOutput
from qwen3vl_agent.p01 import P01Config, TimeSpan
from qwen3vl_agent.p01.media import SourceFrameStore, VideoMetadata
from qwen3vl_agent.r5 import (
    ExternalFile,
    ExternalSegment,
    R5Budget,
    R5Config,
    R5Request,
    R5VideoAgent,
    SummaryProviderResult,
)
from qwen3vl_agent.r5.evaluate import preflight, read_manifest, run_manifest
from qwen3vl_agent.r5.ledger import parse_observation
from qwen3vl_agent.r5.planning import make_plan, merge_calls, times, union_duration
from qwen3vl_agent.r5.synthesis import parse_merge
from qwen3vl_agent.r5.types import ProtocolError


def observation(events=None):
    def observe(payload):
        a, b = payload["core"]
        frames = [
            v
            for v in payload["catalog"].values()
            if v["kind"] == "frame" and a <= v["start_sec"] < b
        ]
        facts = []
        candidates = events if events is not None else [(a, b, f"Activity at {a:g} seconds.")]
        for i, (start, end, text) in enumerate(candidates):
            visible = [f for f in frames if start <= f["start_sec"] < end]
            if visible:
                facts.append(
                    {
                        "local_id": f"f{i}",
                        "statement": text,
                        "kind": "visual_observation",
                        "role": "outcome" if "ending" in text else "action",
                        "evidence_refs": [visible[0]["id"]],
                    }
                )
        return {
            "facts": facts,
            "local_entities": [],
            "local_transitions": [],
            "unresolved": [],
            "truncated": False,
        }

    return observe


def merge(payload):
    return {
        "claims": [
            {"statement": u["statement"], "support_refs": [u["id"]]} for u in payload["units"]
        ],
        "omitted_refs": [],
        "conflicts": [],
    }


def composer(payload):
    unique = {}
    for unit in [*payload["units"], *payload["facts"]]:
        unique.setdefault(unit["statement"], []).append(unit["id"])
    claims = [
        {"id": f"a{i}", "statement": text, "support_refs": refs}
        for i, (text, refs) in enumerate(unique.items())
    ]
    choices = payload["choices"]
    return {
        "prediction": choices[0]["label"] if choices else "",
        "claims": claims,
        "options": {c["label"]: [r["id"] for r in claims] for c in choices},
    }


class FakeModel(BaseVideoModel):
    def __init__(self, events=None, **handlers):
        super().__init__("Qwen/Qwen3-VL-8B-Instruct")
        self.handlers = {
            "compile": {
                "operation": "factual_video_summary",
                "focus": "main content",
                "required_modalities": [],
                "unresolved": [],
            },
            "observe": observation(events),
            "merge": merge,
            "compose": composer,
            "repair": "{}",
            **handlers,
        }
        self.calls = []

    def load(self):
        self._loaded = True

    def unload(self):
        self._loaded = False

    def generate(self, messages, **kwargs):
        parts = messages[-1]["content"]
        text = next(p["text"] for p in parts if p.get("text", "").startswith("R5:"))
        role = text.splitlines()[0].split(":")[1]
        payload = json.loads(text.split("INPUT_JSON:\n")[1])
        self.calls.append({"role": role, "payload": payload, "parts": parts, "kwargs": kwargs})
        value = self.handlers[role]
        value = value.pop(0) if isinstance(value, list) else value
        if isinstance(value, BaseException):
            raise value
        value = value(payload) if callable(value) else value
        if isinstance(value, ModelOutput):
            return value
        return ModelOutput(
            value if isinstance(value, str) else json.dumps(value),
            {"input_tokens": 128, "output_tokens": 64},
        )


class Probe:
    def __init__(self, duration, fps=24):
        self.duration, self.fps = duration, fps

    def probe(self, path):
        return VideoMetadata(str(path), self.duration, self.fps, 160, 96)


class Store(SourceFrameStore):
    def __init__(self, config, image):
        super().__init__(config)
        self.image, self.requests = image, []

    def extract(self, path, timestamps, *, purpose, max_side=None):
        values = list(timestamps)
        self.requests.append(values)
        return tuple(
            FrameRef(f"SRC-{round(t * 1e6)}", round(t, 6), str(self.image)) for t in values
        )


@pytest.fixture
def setup(tmp_path):
    image = tmp_path / "frame.png"
    Image.new("RGB", (160, 96), "white").save(image)
    video = tmp_path / "source.mp4"
    video.write_bytes(b"synthetic source for an injected decoder")

    def build(model=None, *, duration=6, fps=24, provider=None, **kwargs):
        media = P01Config(cache_dir=str(tmp_path / "cache"))
        config = R5Config(
            media=media, core_frames=8, context_frames=2, max_frames_per_call=12, **kwargs
        )
        store = Store(media, image)
        model = model or FakeModel()
        agent = R5VideoAgent(
            model, config, provider, index_builder=Probe(duration, fps), source_store=store
        )
        return agent, model, store, str(video)

    return build


def run(setup, model=None, *, duration=6, config=None, **kwargs):
    agent, model, store, video = setup(model, duration=duration, **(config or {}))
    return (
        agent.solve(R5Request(video_path=video, question="Summarize the main content.", **kwargs)),
        model,
        store,
    )


























class Provider:
    version = "fixed-fixture-v1"

    def __init__(self, rows=(), *, complete=True, error=None):
        self.rows, self.complete, self.error, self.calls = rows, complete, error, []

    def read(self, source_id, span):
        self.calls.append((source_id, span))
        rows = tuple(ExternalSegment(source_id=source_id, **r) for r in self.rows)
        return SummaryProviderResult(
            items=rows,
            error=self.error,
            coverage_status="complete" if self.complete else "unknown",
            covered_intervals=((span.start_seconds, span.end_seconds),),
            alignment_error_sec=None,
            provider_version=self.version,
        )


def text_observer(payload):
    facts = [
        {
            "local_id": f"t{i}",
            "statement": "Narrator says: " + s["text"],
            "kind": "reported_event",
            "role": "other",
            "evidence_refs": [key],
        }
        for i, (key, s) in enumerate(payload["catalog"].items())
        if s["kind"] in {"asr", "subtitle"}
    ]
    return {
        "facts": facts,
        "local_entities": [],
        "local_transitions": [],
        "unresolved": [],
        "truncated": False,
    }
