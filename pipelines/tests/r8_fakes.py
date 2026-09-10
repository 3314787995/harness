"""Scripted CPU fixtures, never presented as measurements of Qwen accuracy."""

import json
from copy import deepcopy
from fractions import Fraction

import av
import numpy as np

from qwen3vl_agent.models.base import BaseVideoModel, ModelOutput
from qwen3vl_agent.r8.config import MODEL


def make_video(path, frames=40, rate=8, *, vfr=False):
    with av.open(str(path), "w") as output:
        stream = output.add_stream("mpeg4", rate=rate)
        stream.width, stream.height = 96, 64
        stream.pix_fmt = "yuv420p"
        for i in range(frames):
            pixels = np.zeros((64, 96, 3), dtype=np.uint8)
            pixels[:, :, 0] = (i * 5) % 255
            pixels[:, 30:60, 1] = 190
            frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
            frame.pts = i * 2 + (i // 7 if vfr else 0)
            frame.time_base = Fraction(1, rate * 2)
            for packet in stream.encode(frame):
                output.mux(packet)
        for packet in stream.encode():
            output.mux(packet)
    return path


def task(question="Which price is closest per yogurt?"):
    return {
        "target": "price_per_cup",
        "target_entity": "yogurt",
        "answer_type": "number",
        "output_unit": "CNY/cup",
        "coverage_need": "local",
        "plan": "local",
        "slots": [
            {
                "id": "price",
                "entity_id": "yogurt",
                "attribute": "price",
                "role": "total_price",
                "scope": "shopping",
                "snapshot": "receipt",
                "description": "total yogurt price",
            },
            {
                "id": "count",
                "entity_id": "yogurt",
                "attribute": "count",
                "role": "quantity",
                "scope": "shopping",
                "snapshot": "receipt",
                "description": "cup count",
            },
        ],
        "givens": [],
        "anchors": [],
        "semantic_constraints": [],
        "precision": {"kind": "closest", "places": None, "question_span": "closest"},
        "scope": "shopping",
        "snapshot": "receipt",
    }


def variable(key="price", value="19.9", unit="CNY", refs=None):
    return {
        "id": key,
        "entity_id": "yogurt",
        "attribute": key,
        "role": "total_price" if key == "price" else "quantity",
        "scope": "shopping",
        "snapshot": "receipt",
        "raw_text": str(value) if value is not None else "unreadable",
        "value": value,
        "unit": unit,
        "unit_basis": "",
        "evidence_refs": refs or ["F01"],
        "event_time": None,
        "content_time": None,
        "valid_interval": None,
        "alternatives": [],
        "alternatives_exhaustive": False,
        "unresolved": [],
    }


def observation(payload):
    return {
        "entities": [
            {
                "id": "yogurt",
                "description": "yogurt pack",
                "scope": "shopping",
                "snapshot": "receipt",
            }
        ],
        "observations": [variable(), variable("count", "6", "cup")],
        "relations": [],
        "attempts": [],
        "items": [],
        "transactions": [],
        "unresolved": [],
        "requested_context": [],
        "discovery": {
            "complete": True,
            "rationale": "synthetic visible receipt",
            "open_event_boundaries": [],
            "possible_replays": [],
            "unreadable_items": [],
        },
    }


def program():
    return {
        "backend": "direct",
        "query": {
            "nodes": [{"id": "per_cup", "op": "divide", "args": ["price", "count"], "params": {}}],
            "target_node": "per_cup",
            "output_unit": "CNY/cup",
        },
        "geometry": {
            "symbols": [],
            "constraints": [],
            "rules": [],
            "target": {"constant": "zero"},
            "output_unit": "1",
        },
        "adapters": [],
        "checks": [],
        "unresolved": [],
    }


class FakeModel(BaseVideoModel):
    def __init__(self, handlers=None):
        super().__init__(MODEL)
        self.calls, self.handlers = [], handlers or {}

    def load(self):
        self._loaded = True

    def unload(self):
        self._loaded = False

    def generate(self, messages, **kwargs):
        content = messages[-1]["content"]
        body = json.loads(
            next(
                p["text"].split("R8_JSON\n", 1)[1]
                for p in content
                if p.get("type") == "text" and "R8_JSON\n" in p["text"]
            )
        )
        self.calls.append((deepcopy(body), deepcopy(kwargs), deepcopy(messages)))
        role, payload = body["stage"], body["input"]
        if role in self.handlers:
            value = self.handlers[role](payload)
            if isinstance(value, Exception):
                raise value
        elif role == "compile":
            value = task()
        elif role in {"observe", "reread"}:
            value = observation(payload)
        elif role == "formalize":
            value = program()
        elif role == "audit":
            value = {
                "semantics_ok": True,
                "sources_ok": True,
                "coverage_ok": True,
                "defects": [],
                "explanation": "scripted test audit",
            }
        else:
            value = {
                "prediction": "B",
                "value": "199/60",
                "unit": "CNY/cup",
                "explanation": "scripted test",
            }
        return ModelOutput(
            value if isinstance(value, str) else json.dumps(value),
            {"input_tokens": 7, "output_tokens": 8},
        )
