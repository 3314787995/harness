"""Synthetic protocol fixtures, never observations of the benchmark videos."""

import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

from qwen3vl_agent.coarse_to_fine.types import FrameRef
from qwen3vl_agent.models.base import ModelOutput
from qwen3vl_agent.p01.types import TimeSpan
from qwen3vl_agent.r1.media import MediaBatch
from qwen3vl_agent.r6 import R6Request
from qwen3vl_agent.r6.types import InputContract, digest


def request(tmp_path, **kwargs):
    path = tmp_path / "synthetic.mp4"
    path.write_bytes(b"synthetic-media-for-controller-tests")
    return R6Request(
        str(path),
        "Which explanation matches the observed event?",
        ["A. The cat scratched them.", "B. The dog scratched them."],
        **kwargs,
    )


def query(choices=None, *, relation_type="direct", required_modalities=None):
    choices = choices or [
        {"label": "A", "text": "The cat scratched them."},
        {"label": "B", "text": "The dog scratched them."},
    ]
    return {
        "answer_operator": "best_explanation",
        "target_description": "Explain the event",
        "reference_scope": "the relevant story period",
        "reference_interval": None,
        "subtype": "S4",
        "coverage": "local",
        "entities": [{"id": "p", "description": "person"}],
        "atoms": [
            {
                "id": f"a{i}",
                "claim": c["text"],
                "entity_ids": ["p"],
                "story_time": None,
                "required_modalities": required_modalities or ["video"],
                "relation_type": relation_type,
            }
            for i, c in enumerate(choices)
        ],
        "option_claims": [
            {
                "label": c["label"],
                "logic": {"op": "atom", "id": f"a{i}"},
                "selection_polarity": "positive",
                "selection_set": None,
                "none_of_context": "",
            }
            for i, c in enumerate(choices)
        ],
        "discriminators": ["Identify the animal and its action"],
        "initial_actions": [],
        "ambiguities": [],
    }


def fact(alias="F01", *, story_time=None, quality="clear"):
    return {
        "source_ids": [alias],
        "entity_ids": ["p"],
        "story_time": story_time,
        "kind": "observation",
        "predicate": "A cat scratches the person",
        "speaker": None,
        "referred_entity": None,
        "quote_or_paraphrase": "paraphrase",
        "quality": quality,
        "coverage_notes": "Synthetic test statement",
    }


def relation(key="r1", *, parents=None, facts=None, story_time=None):
    return {
        "key": key,
        "claim": "The observation supports the explanation",
        "relation_type": "reveals",
        "entity_ids": ["p"],
        "story_time": story_time,
        "premise_fact_ids": facts or ["F000001"],
        "premise_relation_ids": parents or [],
        "bridge": "Synthetic inferential bridge.",
        "support_state": "supported",
        "missing_premises": [],
        "strong_alternatives": [],
        "conflict_flag": False,
    }


def assessment(spec, facts, *, relation_rows=None, both_true=False):
    fid = next(iter(facts), "F000001")
    relations = relation_rows or []
    candidates = []
    for i, option in enumerate(spec["option_claims"]):
        candidates.append(
            {
                "label": option["label"],
                "atom_assessments": [
                    {
                        "atom_id": f"a{i}",
                        "status": "supported" if i == 0 or both_true else "contradicted",
                        "fact_ids": [fid],
                        "relation_ids": [relations[0]["key"]] if relations else [],
                    }
                ],
                "answer_target_fit": "complete" if i == 0 else "partial",
                "fit_reason": "Target comparison",
                "fit_fact_ids": [fid],
                "fit_relation_ids": [],
                "missing_premises": [],
                "direct": not relations,
            }
        )
    return {
        "relations": relations,
        "candidates": candidates,
        "preferred_label": candidates[0]["label"],
        "competitors": [
            {
                "label": c["label"],
                "addressed": True,
                "reason": "Less explanatory",
                "fact_ids": [fid],
                "relation_ids": [],
            }
            for c in candidates[1:]
        ],
        "gaps": [],
        "actions": [],
        "occasions": [],
        "universe_complete": False,
        "coverage_fact_ids": [],
    }


class FakeModel:
    def __init__(self, handlers=None):
        self.handlers = handlers or {}
        self.calls = []
        self.loaded = False

    def load(self):
        self.loaded = True

    def unload(self):
        self.loaded = False

    def generate(self, messages, **kwargs):
        body = json.loads(messages[1]["content"][0]["text"])
        role, data, sources = body["role"], body["input"], body["source_manifest"]
        self.calls.append(
            {"role": role, "body": deepcopy(body), "messages": deepcopy(messages), "kwargs": kwargs}
        )
        handler = self.handlers.get(role)
        if handler:
            value = handler(body)
        elif role == "compiler":
            value = query(data["choices"])
            value["reference_interval"] = data["protocol"]["reference_scope"]
        elif role == "observer":
            record = fact(next(iter(sources)))
            record["entity_ids"] = [e["id"] for e in data["entities"]]
            if all(s["modality"] in {"subtitle", "asr"} for s in sources.values()):
                record["kind"] = "attributed_statement"
                record["speaker"] = "speaker p"
            value = {"records": [record], "gaps": [], "overflow": False}
        elif role == "relation_checker":
            value = assessment(data["query_spec"], {f["id"]: f for f in data["facts"]})
        elif role == "verifier":
            value = {
                "checks": [
                    {
                        "check_id": c["check_id"],
                        "state": "supported",
                        "source_ids": [next(iter(sources))],
                        "missing": [],
                    }
                    for c in data["checks"]
                ],
                "gaps": [],
            }
        else:
            value = {
                "preferred_label": data["choices"][0]["label"],
                "source_ids": [],
                "limitations": [],
            }
        return ModelOutput(
            value if isinstance(value, str) else json.dumps(value),
            {
                "input_tokens": 100,
                "output_tokens": 30,
                "visual_tokens": 20,
                "processed_pixels": 20480,
                "latency_seconds": 0.01,
            },
        )


class FakeMedia:
    def __init__(self, request, config):
        self.request, self.config = request, config
        self.source_hash = digest(Path(request.video_path).read_bytes().hex())
        self.metadata = SimpleNamespace(duration_seconds=4)
        self.contract = InputContract.resolve(request, 4, self.source_hash)
        self.catalog = {}

    def extract(self, span, times, *, fps=None):
        assert self.contract.permits_span(span)
        frames = []
        for t in times:
            assert self.contract.permits(t)
            fid = "FS-" + digest([t, self.contract.fingerprint])[:12]
            path = self.request.video_path
            self.catalog[fid] = {
                "id": fid,
                "path": path,
                "source_frame_id": fid,
                "modality": "video",
                "source_time": [t, t],
                "timestamp_seconds": t,
                "scope_hash": self.contract.fingerprint,
                "pixel_sha256": self.source_hash,
                "view_box": [0, 0, 32, 32],
                "source_size": [32, 32],
            }
            frames.append(FrameRef(fid, t, path))
        return MediaBatch(TimeSpan(*span), tuple(frames), requested_fps=fps, ordered=True)

    def frame(self, fid):
        row = self.catalog[fid]
        return FrameRef(fid, row["timestamp_seconds"], row["path"])

    def crop(self, fid, bbox):
        parent = self.catalog[fid]
        sid = fid + "crop"
        self.catalog[sid] = {**parent, "id": sid, "source_frame_id": fid, "view_box": list(bbox)}
        return self.frame(sid)

    def batch_for_sources(self, ids):
        return MediaBatch(TimeSpan(0, 4), tuple(self.frame(i) for i in ids)) if ids else None

    def prepare(self, batch, *, safe=False):
        return SimpleNamespace(
            frames=batch.frames,
            parts=[{"type": "image", "image": f.path} for f in batch.frames],
            pixels=1024 * len(batch.frames),
            sizes=[(32, 32)] * len(batch.frames),
            video_frame_metadata=[],
        )

    def evidence(self, prepared):
        return {
            f"F{i + 1:02d}": deepcopy(self.catalog[f.id]) for i, f in enumerate(prepared.frames)
        }

    def coverage(self, batch, *, overview=False):
        return {
            "span": [batch.span.start_seconds, batch.span.end_seconds],
            "overview": overview,
            "completed": True,
            "resolution_met": True,
            "finite_sampling_only": True,
        }

    def restore(self, sources):
        self.catalog = deepcopy(sources)
