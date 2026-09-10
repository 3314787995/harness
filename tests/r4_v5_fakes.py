"""Deterministic input fixtures; these do not measure Qwen's visual accuracy."""
import copy
import json
import re
from pathlib import Path
from types import SimpleNamespace
from PIL import Image
from qwen3vl_agent.models.base import BaseVideoModel, ModelOutput
from qwen3vl_agent.coarse_to_fine.types import FrameRef
from qwen3vl_agent.p01.media import SourceFrameStore
from qwen3vl_agent.r4 import R4VideoAgent, R4Config, R4Request


def task(namespace="physical_instance", op="count_unique", *, scope=None, **extra):
    target = {"physical_instance": "tools", "semantic_category": "fruit types", "text_value": "printed words", "task_item": "shopping items"}[namespace]
    equivalence = {"physical_instance": "entity", "semantic_category": "category", "text_value": "literal", "task_item": "task"}[namespace]
    s = {"set_id": "items", "namespace": namespace, "target": target, "count_unit": namespace,
         "predicate": "visible in the query scope", "equivalence": equivalence, **extra}
    if namespace == "task_item":
        s.update(evidence_relation="planned", required_modalities=["subtitle"], predicate_kind="task")
    if namespace == "text_value":
        s["evidence_relation"] = "text_present"
    return {"sets": [s], "operations": [{"operation_id": "answer", "op": op, "inputs": ["items"]}],
            "scope": scope or {"kind": "full"}, "version": 5}


def card(payload, i=1, *, name="wrench", cls=None, query_value=None, status="yes", ref=None, bbox=None, set_id=None, **extra):
    s = next(s for s in payload["sets"] if set_id is None or s["set_id"] == set_id)
    ref = ref or next((r for r, meta in payload["catalog"].items() if meta["region"] == "core"), next(iter(payload["catalog"])))
    row = {"id": f"O{i}", "set": s["set_id"], "name": name, "class": cls or name,
           "facts": "fixture visual evidence", "conditions": {k: status for k in s.get("requirements", s.get("conditions", {}))}}
    if s["namespace"] == "physical_instance":
        row["boxes"] = [{"ref": ref, "xyxy": bbox or [50+i*100,100,140+i*100,600]}]
    else:
        row["refs"] = [ref]
        if query_value is not None:
            row.update(query_value=query_value, mapping_evidence="fixture query-granularity mapping")
    return {**row, **extra}


class Model(BaseVideoModel):
    def __init__(self, compiled, observe=None, inspect=None, identity=None, scope=None, compile_outputs=None):
        super().__init__("test/frozen-model")
        self.compiled=compiled
        self.handlers={"discover_candidates":observe,"inspect_existing":inspect,"identity":identity,"scope":scope}
        self.calls=[]
        self.compile_outputs=list(compile_outputs or [])
        self.loads=0
    def load(self):
        self.loads+=1; self._loaded=True
    def unload(self):
        self._loaded=False
    def generate(self,messages,**kwargs):
        text=messages[-1]["content"][-1]["text"]
        role=re.search(r"R4:([a-z_]+)",text)[1]
        payload=json.loads(text.split("INPUT\n",1)[1])
        self.calls.append({"role":role,"payload":payload,"messages":copy.deepcopy(messages),"kwargs":kwargs})
        if role == "compile":
            result=self.compile_outputs.pop(0) if self.compile_outputs else copy.deepcopy(self.compiled)
        elif self.handlers[role]:
            result=self.handlers[role](payload,self)
        elif role == "discover_candidates":
            result={"records":[],"checks":[],"coverage":"complete"}
        elif role == "identity":
            result={"relations":[{"left":a,"right":b,"relation":"UNKNOWN","basis":"uncertain","refs":[],"facts":"fixture ambiguity"}
                                 for a in payload["left"] for b in payload["right"]]}
        elif role == "inspect_existing":
            result={"updates":[],"coverage":"partial","gaps":["fixture ambiguity"]}
        else:
            result={"bindings":[],"coverage":"complete"}
        if isinstance(result,BaseException):
            raise result
        if isinstance(result,ModelOutput):
            return result
        if isinstance(result,str):
            raw=result
        else:
            raw=json.dumps(result)
        parts=messages[-1]["content"]
        pixels=sum(p["max_pixels"] for p in parts if p["type"] == "image")
        return ModelOutput(raw,{"output_tokens":100,"visual_tokens":pixels//1024,"processed_pixels":pixels*2,"finish_reason":"stop"})


def setup(tmp_path,model,duration=8, *, request_kwargs=None, config=None):
    tmp_path.mkdir(parents=True,exist_ok=True)
    video=tmp_path/'fixture.mp4'
    video.write_bytes(b'synthetic source; no real inference')
    image=tmp_path/'frame.png'
    Image.new('RGB',(256,256),'white').save(image)
    config=config or R4Config.from_mapping({"media":{"cache_dir":str(tmp_path/'cache'),"normal_max_pixels":393216,"normal_total_pixels":8388608,"detail_max_pixels":1048576}})
    class Store(SourceFrameStore):
        def extract(self,path,timestamps,**kwargs):
            return tuple(FrameRef(f"t{t:.6f}",t,str(image)) for t in timestamps)
    class Probe:
        def probe(self,path):
            return SimpleNamespace(duration_seconds=duration,source_fps=24)
    agent=R4VideoAgent(model,config,index_builder=Probe(),source_store=Store(config.media))
    request=R4Request(str(model.compiled["sets"][0]["target"])+" inventory?", video_path=str(video),
                      choices={"A":"1","B":"2","C":"3","D":"4"}, **(request_kwargs or {}))
    return agent,request
