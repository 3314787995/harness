from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
from PIL import Image

from qwen3vl_agent.coarse_to_fine.cache import CachedVideo
from qwen3vl_agent.coarse_to_fine.types import FrameRef
from qwen3vl_agent.models.base import BaseVideoModel, ModelOutput
from qwen3vl_agent.p01 import P01Config, TimeSpan
from qwen3vl_agent.p01.media import P01VideoIndex, SourceFrameStore, TemporalNode, VideoMetadata
from qwen3vl_agent.r1 import (
    ExternalSegment,
    ProviderResult,
    R1Budget,
    R1Config,
    R1Request,
    R1VideoAgent,
)
from qwen3vl_agent.r1.control import audit_bundle, covered, resolve_scopes
from qwen3vl_agent.r1.media import MediaBatch, coverage_record, time_batches
from qwen3vl_agent.r1.types import BindingRecord, EvidenceBundle, QueryField, QuerySpec


def query(**kwargs):
    return {
        "fields": [{"description": "target clothing colour"}],
        "anchor_description": "target person",
        "observation_modes": ["static"],
        "coverage": "point",
        **kwargs,
    }


def observe(payload, *, value="red", status="clear", **kwargs):
    frames = payload["frames"]
    refs = [frames[len(frames) // 2]["id"]] if frames else []
    facts = (
        [
            {
                "statement": f"The target clothing is {value}.",
                "structured_value": value,
                "subject_or_local_entity": "person_1",
                "attribute": "clothing_colour",
                "source_frame_ids": refs,
                "source_segment_ids": [],
                "observation_status": status,
                "supports_query_fields": [f["field_id"] for f in payload["query"]["fields"]],
            }
        ]
        if refs
        else []
    )
    return {
        "anchor_match": "matched",
        "anchor_source_ids": refs,
        "target_binding": "confirmed",
        "target_source_ids": refs,
        "facts": facts,
        "unresolved": [],
        "truncated": False,
        "crop_requests": [],
        **kwargs,
    }


def locate(payload):
    nodes = payload["nodes"][:1]
    return {
        "candidates": [
            {
                "node_id": n["node_id"],
                "anchor_frame_ids": [n["frames"][0]["id"]],
                "matched_anchor_conditions": ["target"],
                "unresolved_anchor_conditions": [],
            }
            for n in nodes
        ]
    }


def final(payload):
    facts = [
        f for p in payload["frozen_bundle"]["packets"] if p["role"] == "answer" for f in p["facts"]
    ]
    selected = facts[-1]
    choices = payload["choices"]
    label = next(
        (c["label"] for c in choices if c["text"] == selected["structured_value"]),
        choices[0]["label"] if choices else selected["structured_value"],
    )
    return {
        "prediction": label,
        "evidence_fact_ids": [selected["fact_id"]],
        "claims": [{"statement": selected["statement"], "fact_ids": [selected["fact_id"]]}],
        "answer_supported": True,
        "alternatives_excluded": True,
        "choice_assessments": [
            {
                "label": c["label"],
                "status": "supported" if c["label"] == label else "rejected",
                "fact_ids": [selected["fact_id"]],
            }
            for c in choices
        ],
    }


class FakeModel(BaseVideoModel):
    def __init__(self, **handlers):
        super().__init__("Qwen/Qwen3-VL-8B-Instruct")
        self.handlers = {
            "query": query(),
            "discriminants": {
                "inspection_needs": ["Read clothing colour"],
                "target_union": [],
                "observation_modes": [],
            },
            "locator": locate,
            "observe": observe,
            "final": final,
            "repair": "{}",
            **handlers,
        }
        self.calls = []

    def load(self):
        self._loaded = True

    def unload(self):
        self._loaded = False

    def generate(self, messages, **kwargs):
        content = messages[-1]["content"]
        text = next(p["text"] for p in content if p.get("text", "").startswith("R1:"))
        role = text.splitlines()[0].split(":")[1]
        payload = json.loads(text.split("INPUT_JSON:\n")[1]) if "INPUT_JSON:\n" in text else {}
        self.calls.append({"role": role, "payload": payload, "content": content, "kwargs": kwargs})
        handler = self.handlers[role]
        response = handler.pop(0) if isinstance(handler, list) else handler
        if isinstance(response, Exception):
            raise response
        response = response(payload) if callable(response) else response
        return ModelOutput(
            response if isinstance(response, str) else json.dumps(response), {"output_tokens": 32}
        )


class Index:
    def __init__(self, path, duration, spans):
        self.calls = 0
        self.duration = duration
        self.frames = tuple(
            FrameRef(f"SRC-{t * 1000:09d}", float(t), str(path)) for t in range(duration)
        )
        nodes = {
            f"n{i}": TemporalNode(f"n{i}", TimeSpan(*span), 0, (), ())
            for i, span in enumerate(spans)
        }
        nodes["root"] = TemporalNode("root", TimeSpan(0, duration), 1, tuple(nodes), ())
        cache = CachedVideo(
            "video.mp4", str(path.parent), duration, 24, 640, 360, 1, self.frames, False
        )
        self.index = P01VideoIndex(cache, (), (), nodes, "root")

    def probe(self, path):
        return VideoMetadata(str(path), self.duration, 24, 640, 360)

    def prepare_interval(self, path, span, *, metadata):
        self.calls += 1
        return self.index


class Store(SourceFrameStore):
    def __init__(self, config, image):
        super().__init__(config)
        self.image = image
        self.calls = []
        self.exclude = set()

    def extract(self, path, timestamps, *, purpose, max_side=None):
        self.calls.append(list(timestamps))
        return tuple(
            FrameRef(f"SRC-{round(t * 1000):09d}", float(t), str(self.image))
            for t in timestamps
            if round(t, 3) not in self.exclude
        )


@pytest.fixture
def setup(tmp_path):
    image = tmp_path / "source.png"
    Image.new("RGB", (640, 360), "red").save(image)

    def build(
        model=None,
        *,
        duration=20,
        spans=((0, 5), (10, 15)),
        provider=None,
        media_changes=None,
        **config,
    ):
        media = P01Config(cache_dir=str(tmp_path / "cache"), **(media_changes or {}))
        settings = R1Config(media=media, **config)
        index = Index(image, duration, spans)
        store = Store(media, image)
        model = model or FakeModel()
        agent = R1VideoAgent(
            model, config=settings, index_builder=index, source_store=store, provider=provider
        )
        return agent, model, index, store

    return build


def request(**kwargs):
    return R1Request(
        "video.mp4",
        "What colour is the target person's clothing?",
        choices=("red", "blue"),
        **kwargs,
    )


























class Provider:
    def __init__(self, *, available=True, empty=False):
        self.available, self.empty = available, empty
        self.calls = []

    def search(self, question, source_id, span):
        self.calls.append("search")
        return ProviderResult(
            (ExternalSegment(source_id, "hint", 0.5, 1, "search hint only", "asr"),),
            available=self.available,
        )

    def read(self, source_id, span):
        self.calls.append("read")
        items = (
            ()
            if self.empty
            else (
                ExternalSegment(
                    source_id,
                    "s1",
                    0.5,
                    1,
                    "I left because it rained.",
                    "asr",
                    speaker_id="voice1",
                    alignment_status="aligned",
                ),
            )
        )
        return ProviderResult(items, available=self.available, cost={"seconds": 0.5})
