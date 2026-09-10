"""Synthetic scripted responses, unrelated to any benchmark video or its gold answers."""

import copy
import json
from fractions import Fraction

from qwen3vl_agent.models.base import BaseVideoModel, ModelOutput

QUESTION = "Set right to original left. What is the right count?"


def make_video(path, *, suffix_color="blue", variable_pts=False):
    import av
    from PIL import Image

    with av.open(str(path), "w") as output:
        stream = output.add_stream("mpeg4", rate=8)
        stream.width, stream.height, stream.pix_fmt = 96, 64, "yuv420p"
        for index in range(40):
            image = Image.new("RGB", (96, 64), "red" if index < 24 else suffix_color)
            frame = av.VideoFrame.from_image(image)
            frame.pts = index if not variable_pts else index + (index // 3)
            frame.time_base = Fraction(1, 8)
            for packet in stream.encode(frame):
                output.mux(packet)
        for packet in stream.encode():
            output.mux(packet)
    return path


def intervention(target="R.count", value=None, *, op="set", identifier="set-right"):
    return {
        "id": identifier,
        "op": op,
        "target": target,
        "value": {"ref": "L.count"} if value is None else value,
        "read_world": "factual",
        "sequential": False,
        "source_span": "Set right to original left.",
    }


def task(mechanism="S4"):
    return {
        "mechanisms": [mechanism],
        "target_visibility": "counterfactual",
        "query_operator": "value",
        "targets": [
            {"id": "L", "description": "left object"},
            {"id": "R", "description": "right object"},
        ],
        "slots": [
            {
                "id": "L.count",
                "entity_id": "L",
                "predicate": "count",
                "description": "Read left count",
            },
            {
                "id": "R.count",
                "entity_id": "R",
                "predicate": "count",
                "description": "Read right count",
            },
        ],
        "anchors": [],
        "target_time": None,
        "interventions": [intervention()],
        "invariants": [],
        "stipulations": [],
        "unresolved": [],
    }


def candidates(options=None, key="R.count", expected=None, scenario="main"):
    options = options or [{"label": "A", "text": "2"}, {"label": "B", "text": "9"}]
    expected = expected or {"2": 2, "9": 9}
    return {
        "candidates": [
            {
                "label": o["label"],
                "text": o["text"],
                "interventions": [],
                "atoms": [
                    {
                        "id": "a1",
                        "text_span": o["text"],
                        "scenario_id": scenario,
                        "key": key,
                        "relation": "eq",
                        "expected": expected[o["text"]],
                        "polarity": "positive",
                        "modality": "fact",
                    }
                ],
            }
            for o in options
        ]
    }


def scenario(identifier="main", **kwargs):
    return {
        "id": identifier,
        "candidate_label": None,
        "programs": [],
        "hypotheses": [],
        "physics": [],
        "trends": [],
        **kwargs,
    }


def reason(**kwargs):
    return {
        "rules": [],
        "factual_program": [],
        "scenarios": [scenario()],
        "unresolved": [],
        **kwargs,
    }


def fact(key, value, *, time=0.0, unit="", complete=True, refs=None, entity=None, kind="observed"):
    return {
        "key": key,
        "entity_id": entity or key.split(".")[0],
        "predicate": key.rsplit(".", 1)[-1],
        "value": value,
        "kind": kind,
        "evidence_ids": refs or ["F01"],
        "unit": unit,
        "time": time,
        "complete": complete,
    }


class ScriptedQwen(BaseVideoModel):
    def __init__(
        self,
        *,
        spec=None,
        proposal=None,
        observe_hook=None,
        candidate_hook=None,
        verify_hook=None,
        invalid_role=None,
        interrupt_role=None,
        error_role=None,
    ):
        super().__init__("synthetic-qwen")
        self.spec = spec or task()
        self.proposal = proposal or reason()
        self.observe_hook, self.candidate_hook, self.verify_hook = (
            observe_hook,
            candidate_hook,
            verify_hook,
        )
        self.invalid_role, self.interrupt_role, self.error_role = (
            invalid_role,
            interrupt_role,
            error_role,
        )
        self.payloads, self.messages = [], []

    def load(self):
        self._loaded = True

    def unload(self):
        self._loaded = False

    def generate(self, messages, **kwargs):
        body = json.loads(messages[-1]["content"][-1]["text"].split("\n", 1)[1])
        role, payload = body["stage"], body["input"]
        self.payloads.append((role, copy.deepcopy(payload)))
        self.messages.append(copy.deepcopy(messages))
        if self.interrupt_role == role:
            self.interrupt_role = None
            raise KeyboardInterrupt("synthetic interruption")
        if self.error_role == role:
            raise RuntimeError("synthetic model timeout")
        if self.invalid_role == role:
            return ModelOutput("invalid JSON", {"input_tokens": 11, "output_tokens": 2})
        if role == "compile":
            value = self.spec
        elif role == "candidates":
            value = (
                self.candidate_hook(payload)
                if self.candidate_hook
                else candidates(payload["options"])
            )
        elif role == "observe":
            t = payload["frames"][0]["source_seconds"]
            value = (
                self.observe_hook(payload)
                if self.observe_hook
                else {
                    "entities": [],
                    "facts": [fact("L.count", 2, time=t), fact("R.count", 9, time=t)],
                    "gaps": [],
                }
            )
        elif role == "reason":
            value = self.proposal(payload) if callable(self.proposal) else self.proposal
        elif role in {"verify", "final"}:
            executed = {c["label"]: c for c in payload["program_assessments"]}
            assessments = []
            for c in payload["candidates"]:
                atoms = executed.get(c["label"], {}).get("atoms", [])
                assessments.append(
                    {
                        "label": c["label"],
                        "atoms": [
                            {
                                "id": a["id"],
                                "status": next(
                                    (e["status"] for e in atoms if e["id"] == a["id"]), "unknown"
                                ),
                                "evidence_ids": next(
                                    (e["evidence_ids"] for e in atoms if e["id"] == a["id"]), []
                                ),
                                "reason": "synthetic comparison",
                            }
                            for a in c["atoms"]
                        ],
                    }
                )
            supported = [
                c["label"]
                for c in assessments
                if all(a["status"] in {"supported", "entailed_by_execution"} for a in c["atoms"])
            ]
            value = {
                "prediction": supported[0] if supported else payload["candidates"][0]["label"],
                "assessments": assessments,
                "gaps": [],
                "unresolved": [],
            }
            if self.verify_hook:
                value = self.verify_hook(value, payload)
        else:
            value = {
                "prediction": payload["options"][0]["label"],
                "reason": "synthetic baseline",
                "unresolved": [],
            }
        return ModelOutput(
            json.dumps(value), {"input_tokens": 11, "output_tokens": 7, "visual_tokens": 3}
        )
