"""Scripted role fixtures are software tests, not Qwen accuracy measurements."""

import json
from copy import deepcopy
from dataclasses import replace

from qwen3vl_agent.models.base import BaseVideoModel, ModelOutput
from qwen3vl_agent.r9.config import MODEL, R9Config
from qwen3vl_agent.r9.spatial_state import SpatialState
from tests.r8_fakes import make_video


def config(tmp_path, **kwargs):
    c = R9Config(initial_overview_frames=4, preferred_frames_per_call=4, max_refinement_rounds=1)
    return replace(c, media=replace(c.media, cache_dir=str(tmp_path / "cache")), **kwargs)


def spec(question="Where is the target?", operation="bearing"):
    return {
        "operation": operation,
        "entities": [
            {"id": "e1", "role": "origin", "description": "stove"},
            {"id": "e2", "role": "forward_target", "description": "sofa"},
            {"id": "e3", "role": "target", "description": "tv"},
        ],
        "time": {"reference_anchor": None, "query_anchor": None},
        "query_frame": {
            "kind": "query",
            "origin_role": "origin",
            "forward_role": "forward_target",
            "plane": "horizontal",
            "orientation_source": "question_defined",
            "heading_kind": "body",
        },
        "direction_rule": {"kind": "four_quadrants", "back_threshold_degrees": None},
        "measurement": {"geometry_semantics": "none", "unit": None, "metric_required": False},
        "route": None,
        "required_capabilities": ["entity_binding", "query_frame", "bearing_signs"],
        "source_spans": [{"start": 0, "end": len(question), "text": question}],
        "nodes": [],
        "output_node": None,
    }


def record(predicate="bearing", value=None, arguments=None, frame="query", unit=None):
    return {
        "record_id": "r1",
        "predicate": predicate,
        "arguments": arguments or ["e1", "e2", "e3"],
        "reference_frame_id": frame,
        "valid_time": [0, 10],
        "value": value or {"horizontal": "left", "depth": "front", "angle_interval": None},
        "unit": unit,
        "method": "direct_observation",
        "source_observation_ids": ["o1", "o2", "o3"],
        "parent_record_ids": [],
        "status": "evidence_supported",
        "uncertainty_description": "",
    }


def seeded_state(s=None):
    s = s or spec()
    state = SpatialState(s)
    for n, e in enumerate(s["entities"], 1):
        state.data["observations"][f"o{n}"] = {
            "id": f"o{n}",
            "entity_candidate": f"c{n}",
            "frame_id": f"f{n}",
            "source_frame_id": f"f{n}",
            "timestamp_s": float(n),
            "observation_statement": e["description"],
        }
        state.data["entity_links"][e["id"]] = [
            {
                "entity_id": e["id"],
                "candidate_ids": [f"c{n}"],
                "status": "confirmed",
                "basis": "distinctive_attributes",
                "source_observation_ids": [f"o{n}"],
                "valid_time": [0, 10],
            }
        ]
    return state


class FakeModel(BaseVideoModel):
    def __init__(self, task=None, *, missing=False, malformed_once=False, interrupt_at=None):
        super().__init__(MODEL)
        self.task = task
        self.calls = []
        self.missing = missing
        self.malformed_once = malformed_once
        self.interrupt_at = interrupt_at

    def load(self):
        self._loaded = True

    def unload(self):
        self._loaded = False

    def generate(self, messages, **kwargs):
        body = json.loads(messages[-1]["content"][-1]["text"].split("R9_JSON\n", 1)[1])
        self.calls.append({"messages": messages, "body": body, "kwargs": kwargs})
        if self.interrupt_at == len(self.calls):
            self.interrupt_at = None
            raise KeyboardInterrupt("scripted interruption")
        if self.malformed_once:
            self.malformed_once = False
            return ModelOutput("not json")
        stage, p = body["stage"], body["input"]
        if stage == "compile":
            value = deepcopy(self.task) if self.task else spec(p["question"])
        elif stage == "observe":
            aliases = list(p["frames"])
            times = [v["timestamp_seconds"] for v in p["frames"].values()]
            observations, links = [], []
            if not self.missing:
                for i, e in enumerate(p["task"]["entities"]):
                    oid, cid = f"o{i + 1}", f"c{i + 1}"
                    observations.append(
                        {
                            "id": oid,
                            "frame_id": aliases[min(i, len(aliases) - 1)],
                            "shot_id": "shot1",
                            "entity_candidate": cid,
                            "category": e["description"],
                            "visible_attributes": ["distinct fixture marker"],
                            "facing_cues": [],
                            "bbox_xyxy": [0.1, 0.1, 0.8, 0.8],
                            "coordinate_space": "presented_image_normalized",
                            "occlusion": "none",
                            "visible_extent": "complete",
                            "observation_statement": e["description"],
                        }
                    )
                    links.append(
                        {
                            "entity_id": e["id"],
                            "candidate_ids": [cid],
                            "status": "confirmed",
                            "basis": "distinctive_attributes",
                            "source_observation_ids": [oid],
                            "valid_time": [min(times), max(times)],
                        }
                    )
            value = {"observations": observations, "links": links, "gaps": []}
        elif stage == "relations":
            observations = p["state"]["observations"]
            if not observations:
                value = {"frames": [], "records": [], "gaps": []}
            else:
                r = record()
                r["source_observation_ids"] = [o["id"] for o in observations]
                r["valid_time"] = [
                    min(o["timestamp_s"] for o in observations),
                    max(o["timestamp_s"] for o in observations),
                ]
                value = {"frames": [], "records": [r], "gaps": []}
        elif stage == "audit":
            value = {
                "checks": [
                    {
                        "record_id": r["record_id"],
                        "verdict": "supported",
                        "frame_ids": [next(iter(p["frames"]))],
                        "reason": "visible fixture",
                    }
                    for r in p["atomic_claims"]
                ],
                "gaps": [],
            }
        else:
            semantic = p["original_option_texts"][0] if p["original_option_texts"] else 2.5
            value = {
                "semantic_answer": semantic,
                "unit": p["output_unit"],
                "source_ids": [],
                "reason": "scripted best effort",
            }
        return ModelOutput(
            json.dumps(value),
            {
                "input_tokens": 100,
                "output_tokens": 30,
                "visual_tokens": 20,
                "latency_seconds": 0.01,
            },
        )


__all__ = ["FakeModel", "config", "make_video", "record", "seeded_state", "spec"]
