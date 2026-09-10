"""Targeted v5 engineering checks: synthetic observations and CPU media only."""
import copy
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import pytest
from PIL import Image
from qwen3vl_agent.r3 import R3Request, R3Config, R3Budget, R3VideoAgent, QuerySpec
from qwen3vl_agent.r3.query import validate_query, rule_query, candidate_vocabulary, VERSION
from qwen3vl_agent.r3.lines import parse_lines, PROMPTS
from qwen3vl_agent.r3.candidates import ingest, events_from_rows
from qwen3vl_agent.r3.query_reduce import reduce_events, reduce_query
from qwen3vl_agent.r3.query_engine import exact_choice, deterministic_mapping
from qwen3vl_agent.r3.query_media import QueryMedia, Access, base_tasks
from qwen3vl_agent.r3.query_runtime import QueryRuntime
from qwen3vl_agent.r3.checkpoint import Checkpoint
from qwen3vl_agent.r3.types import ProtocolError, BudgetExhausted
from qwen3vl_agent.models.base import BaseVideoModel, ModelOutput
from qwen3vl_agent.models.qwen3vl import enforce_visual_budget, VisualBudgetExceeded
from qwen3vl_agent.coarse_to_fine.types import FrameRef
from qwen3vl_agent.p01.types import TimeSpan
from qwen3vl_agent.r1.media import MediaBatch


def event(i,a,b,target="task",value=None):
    return {"id":str(i),"target":target,"onset":[a,a],"offset":[b,b],"extent":[a,b],
            "complete":True,"value":value or target,"source_refs":[f"image{i}"],"rows":[str(i)],"companions":{}}

def scan(span=(0,10),fps=8,complete=True):
    return [{"span":list(span),"complete":complete,"sampling":{"requested_fps":fps}}]

def spec(op="count_occurrences",**kw):
    return QuerySpec(op,"task","action_cycle",**kw)

CASES=[("count_occurrences",{},3), ("first_occurrence",{},"v1"), ("last_occurrence",{},"v3"),
       ("first_k",{"k":2},["v1","v2"]), ("last_k",{"k":2},["v2","v3"]),
       ("nth_occurrence",{"k":2},"v2"), ("order_events",{"targets":("task",)},["task"]*3),
       ("next_after_anchor",{"anchor":"anchor","anchor_selection":"unique"},"v1"),
       ("previous_before_anchor",{"anchor":"anchor","anchor_selection":"unique"},"v3"),
       ("localize_event",{},None), ("event_duration",{"aggregation":"sum"},{"min_sec":3,"max_sec":3}),
       ("cooccurrence_frequency",{"targets":("companion",)},None)]

@pytest.mark.parametrize("op,kw,want",CASES)
def test_all_twelve_operations_and_required_coverage(op,kw,want):
    es=[event(i,2*i,2*i+1,value=f"v{i}") for i in (1,2,3)]
    if op=="next_after_anchor": es.append(event(9,.1,1,"anchor"))
    if op=="previous_before_anchor": es.append(event(9,8,9,"anchor"))
    if op=="cooccurrence_frequency":
        for e in es: e["companions"]={"companion":True}
    q=spec(op,**kw)
    result=reduce_events(q,es,scan(),(0,10))
    assert result["closed"], result["gaps"]
    if want is not None: assert result["value"]==want
    if op=="cooccurrence_frequency": assert result["value"]["companion"]["denominator"]==3
    assert not reduce_events(q,es,[],(0,10))["closed"]


def test_ordinal_conflict_missing_events_and_unknown_boundaries():
    a,b=event(1,1,5),event(2,2,4)
    q=spec("nth_occurrence",k=2)
    assert any(g["kind"]=="order" for g in reduce_events(q,[a,b],scan(),(0,10))["gaps"])
    assert not reduce_events(q,[a],scan(),(0,10))["closed"]
    b["offset"]=None
    assert not reduce_events(q,[a,b],scan(),(0,10))["closed"]
    # Counting needs the complete unit, not precise timing brackets.
    a["onset"]=a["offset"]=None
    assert reduce_events(spec(),[a],scan(),(0,10))["closed"]


def test_duration_intervals_union_difference_and_comparison_conflict():
    a,b=event(1,1,5,"a"),event(2,3,7,"b")
    q=spec("event_duration",targets=("a","b"),aggregation="union")
    assert reduce_events(q,[a,b],scan(),(0,10))["value"]=={"min_sec":6,"max_sec":6}
    q=replace(q,aggregation="compare",comparison="longest")
    assert not reduce_events(q,[a,b],scan(),(0,10))["closed"]
    b=event(2,7,9,"b")
    q=replace(q,aggregation="difference",comparison=None)
    assert reduce_events(q,[a,b],scan(),(0,10))["value"]=={"min_sec":2,"max_sec":2}
    a["offset"]=None
    assert not reduce_events(q,[a,b],scan(),(0,10))["closed"]


def test_neighbor_multiple_anchors_intermediate_coverage_and_after_not_next():
    q=spec("next_after_anchor",anchor="anchor")
    es=[event(1,1,2,"anchor"),event(2,3,4),event(3,5,6)]
    assert not reduce_events(q,es,scan(fps=1),(0,10))["closed"]
    assert not reduce_events(q,es+[event(4,7,8,"anchor")],scan(),(0,10))["closed"]
    q=replace(q,relation="after")
    assert reduce_events(q,es,scan(),(0,10))["value"]==["task","task"]


def test_cooccurrence_keeps_unknown_and_denominator():
    q=spec("cooccurrence_frequency",targets=("music","speech"))
    es=[event(1,1,2),event(2,3,4)]
    es[0]["companions"]={"music":True,"speech":False}
    es[1]["companions"]={"music":None,"speech":False}
    r=reduce_events(q,es,scan(),(0,10))
    assert not r["closed"] and r["value"]["music"]=={"present":1,"absent":0,"unknown":1,"denominator":2}


def refs():
    return {f"F{i+1:02d}":{"id":f"source{i}","timestamp_seconds":i,"path":"frame.png"} for i in range(8)}

def lines(*rows,status="complete"):
    return "\n".join(json.dumps(r) for r in (*rows,{"at":["F01","F08"],"decision":"end","status":status,"note":"Clearly visible sampled workspace."}))

def row(d="occurrence",at=("F03",),**kw):
    if d in {"begin","continue","complete","appearance","attribute"}: kw.setdefault("target","task")
    return {"at":list(at),"decision":d,"note":"Visible action evidence.",**kw}

def call(key="one",recipe="cycles",**task):
    return {"id":key,"task":{"key":key,"core":[0,8],"recipe":recipe,**task},"refs":refs(),"sampling":{"resolution_met":True,"requested_fps":8}}


def test_bad_lines_preserve_good_lines_and_empty_does_not_prove_zero():
    raw=json.dumps(row())+'\n'+json.dumps(row(at=("F99",)))+'\n'+lines(row("not_target",at=("F01",)))
    parsed=parse_lines(raw,"cycles",refs(),["task"])
    assert len(parsed["rows"])==2 and parsed["errors"] and not parsed["complete"]
    for raw in ('', '{"events":[]}', '[]', lines(status="complete")):
        assert not parse_lines(raw,"cycles",refs(),["task"])["negative"]
    assert parse_lines(lines(status="absent"),"cycles",refs(),["task"])["negative"]
    assert not parse_lines(lines(row()),"cycles",refs(),["task"],truncated=True)["complete"]


@pytest.mark.parametrize("bad",[row(at=("F99",)),row(at=(1,)),row(at=("1.5",)),row(accepted=True),row(target_id="T1")])
def test_references_and_internal_fields_rejected(bad):
    p=parse_lines(lines(bad),"cycles",refs(),["task"])
    assert not p["rows"] and p["errors"]


def test_replay_prepare_retract_split_and_idempotent_local_replacement():
    state={}
    ingest(state,call(),parse_lines(lines(row()),"cycles",refs(),["task"]))
    assert reduce_query(spec(),state,(0,8))["candidate_estimate"]==1
    correction=call("two",replace_span=[0,8])
    p=parse_lines(lines(row("not_target",("F01","F08")),status="absent"),"cycles",refs(),["task"])
    ingest(state,correction,p);ingest(state,correction,p)
    assert len(state["committed"])==2 and state["candidates"][0]["active"] is False
    result=reduce_query(spec(),state,(0,8))
    assert result["candidate_estimate"]==0 and not result["closed"]  # dedicated zero review still required
    p=parse_lines(lines(row(at=("F02",)),row(at=("F05",))),"cycles",refs(),["task"])
    ingest(state,call("three",replace_span=[0,8]),p)
    assert reduce_query(spec(),state,(0,8))["candidate_estimate"]==2


def test_cross_window_production_steps_merge_with_actual_visual_tail_only():
    q=QuerySpec("nth_occurrence","task","production_instance",k=1,project="shape")
    state={}
    p=parse_lines(lines(row("begin",("F01","F02"))),"instances",refs(),["task"])
    ingest(state,call("one","instances"),p)
    p=parse_lines(lines(row("continue",("F02","F03")),row("complete",("F05","F06"),value="Vase")),"instances",refs(),["task"])
    ingest(state,call("two","instances",tail_refs=["source1"]),p)
    es,problems=events_from_rows(state["candidates"],q)
    assert len(es)==1 and es[0]["complete"] and es[0]["value"]=="Vase" and not problems
    assert reduce_query(q,state,(0,8))["candidate_estimate"]=="Vase"
    broken=copy.deepcopy(state)
    for r in broken["candidates"]: r["tail_refs"]=[]
    assert not reduce_query(q,broken,(0,8))["closed"]


@pytest.mark.parametrize("field,value",[("k",None),("k",0),("k",True),("k","2"),("mystery",1)])
def test_query_schema_explicit_parameters(field,value):
    q={"op":"nth_occurrence","target":"making a vessel","unit":"production_instance","k":2,field:value}
    with pytest.raises(ProtocolError): validate_query(q,R3Request("video","Question?"))


@pytest.mark.parametrize("op,kw,_",CASES)
def test_public_queryspec_objects_use_the_same_boundary_validator(op,kw,_):
    q=spec(op,**kw)
    result=validate_query(q,R3Request("v","q"))
    assert result.op==op and result.targets==q.targets


def test_query_scope_choice_isolation_and_no_id_special_cases():
    req=R3Request("video","How many times did someone ring the bell?",choices={"A":"3","B":"4"})
    assert rule_query(req)["op"]=="count_occurrences"
    assert candidate_vocabulary(req)==[]
    q={"op":"count_occurrences","target":"ringing a bell","unit":"action_cycle","k":None}
    assert validate_query(q,req).op=="count_occurrences"
    with pytest.raises(ProtocolError,match="source"):
        validate_query({**q,"scope":{"kind":"interval","interval":[0,15],"source":"first 15 seconds"}},req)
    with pytest.raises(ProtocolError,match="anchor"):
        validate_query({"op":"next_after_anchor","target":"activity","unit":"activity"},req)
    assert "3" not in json.dumps(validate_query(q,req).observer_view())


def test_final_exact_mapping_and_nullable_no_default():
    req=R3Request("v","q",choices={"A":"A pig.","B":"A cat."})
    assert exact_choice(" a PIG. ",req.choices)=="A"
    assert exact_choice("pig",req.choices) is None
    assert exact_choice("A pig.",replace(req,choices={"A":"A pig.","B":"A pig."}).choices) is None
    assert deterministic_mapping(req,{"closed":False,"value":"A pig."}) is None


class MemoryMedia(QueryMedia):
    def __init__(self,config,tmp_path,duration=8):
        super().__init__(config)
        self.duration=duration
        self.image=tmp_path/"source.png"
        Image.new("RGB",(64,64),"gray").save(self.image)
    def probe(self,path): return SimpleNamespace(duration_seconds=self.duration,source_fps=32)
    def source_digest(self,path): return "synthetic_media"
    def extract(self,path,span,timestamps,contract,*,fps=None,anchors=()):
        assert contract.permits_span(span)
        fs=[]
        for t in sorted(set(timestamps)|{f.timestamp_seconds for f in anchors}):
            fid=f"pts_{t:.6f}"
            self.catalog[fid]={"id":fid,"path":str(self.image),"source_frame_id":fid,"timestamp_seconds":t,
                "source_frame_index":None,"source_fps":32,"source_sha256":"synthetic_media","source_size":[64,64],
                "view_box":[0,0,64,64],"pts":round(t*32),"time_base":[1,32],"origin_pts":0}
            fs.append(FrameRef(fid,t,str(self.image)))
        return MediaBatch(TimeSpan(*span),tuple(fs),fps,True)

class FakeModel(BaseVideoModel):
    def __init__(self,mode="count"):
        super().__init__("fake")
        self.mode,self.calls=mode,[]
    def load(self): self._loaded=True
    def unload(self): self._loaded=False
    def generate(self,messages,**kwargs):
        self.calls.append((messages,kwargs))
        sys=messages[0]["content"]
        if "R3:final_visual" in sys:
            return ModelOutput('{"prediction":"invalid"}' if self.mode=="bad_final" else '{"prediction":"B"}')
        if "R3:query_spec" in sys: return ModelOutput(json.dumps({"op":"count_occurrences","target":"task","unit":"action_cycle"}))
        data=json.loads(messages[1]["content"][0]["text"])
        if "checks" in data:
            if self.mode in {"empty","bad_final"}: return ModelOutput('{}')
            answers=[]
            for c in data["checks"]:
                groups=c["reference_groups"]
                at=list(dict.fromkeys(x for g in groups for x in (g[0],g[-1]))) if len(groups)<=2 else [g[len(g)//2] for g in groups]
                answers.append({"check":c["check"],"verdict":"confirmed","at":at,
                    "note":"Specified visual matter is visible across every supplied segment."})
            return ModelOutput("\n".join(json.dumps(r) for r in answers))
        marks=data["core_frame_markers"]
        if self.mode in {"empty","bad_final"}: return ModelOutput('{}')
        if self.mode=="interrupt" and len(self.calls)==1: raise KeyboardInterrupt("interrupted model")
        r=[] if self.mode=="negative" else [{"at":marks[len(marks)//2:len(marks)//2+2],"decision":"occurrence","note":"One complete visible cycle."}]
        r.append({"at":[marks[0],marks[-1]],"decision":"end","status":"absent" if self.mode=="negative" else "complete","note":"Readable full sample; no target action." if self.mode=="negative" else "One complete cycle is visible."})
        return ModelOutput("\n".join(json.dumps(x) for x in r),{"output_tokens":100})

def setup(tmp_path,mode="count",**config):
    cfg=R3Config.from_mapping({"media":{"cache_dir":str(tmp_path/"cache")},**config})
    model=FakeModel(mode)
    agent=R3VideoAgent(model,config=cfg)
    agent.media=MemoryMedia(cfg,tmp_path)
    req=R3Request("fake.mp4","How many times did the task happen?",choices={"A":"2","B":"3"},checkpoint_path=str(tmp_path/"check.jsonl"))
    return agent,model,req


def test_full_executor_two_required_calls_no_compiler_final_or_gold(tmp_path):
    agent,model,req=setup(tmp_path)
    result=agent.solve(req)
    assert result.prediction=="A" and result.answer_basis=="visual_reduction"
    assert len(model.calls)==3 and result.resources["model_calls"]==3
    assert all('"choices"' not in json.dumps(m) for m,k in model.calls)
    assert all(k["visual_token_limit"]==12288 for m,k in model.calls)
    result2=agent.solve(replace(req,resume=True))
    assert result2.to_dict()==result.to_dict() and len(model.calls)==3


def test_empty_observations_bounded_reviews_and_single_visual_final(tmp_path):
    agent,model,req=setup(tmp_path,"bad_final")
    result=agent.solve(req)
    assert result.prediction is None and result.answer_basis=="protocol_failure"
    assert sum("R3:final_visual" in m[0]["content"] for m,k in model.calls)==1
    assert len(result.trace["critical_reviews"])<=3
    assert not result.trace["default_choice_used"]
    final=next(m for m,k in model.calls if "R3:final_visual" in m[0]["content"])
    assert any(p["type"] in {"image","video"} for p in final[1]["content"])
    assert len(json.loads(final[1]["content"][0]["text"])["choices"])==2


def test_zero_requires_scan_and_one_dedicated_reread(tmp_path):
    agent,model,req=setup(tmp_path,"negative")
    result=agent.solve(replace(req,choices={"A":"0","B":"1"}))
    assert result.prediction=="A" and result.semantic_result["value"]==0
    assert len(model.calls)==5  # two necessary cores + three bounded pieces of one zero check
    assert result.trace["critical_reviews"][0]["task"]["checks"][0]["kind"]=="zero_confirmation"


def test_returned_response_resume_does_not_infer_again(tmp_path,monkeypatch):
    agent,model,req=setup(tmp_path)
    original=QueryRuntime.changed
    tripped=[]
    def stop_after_return(runtime):
        original(runtime)
        if not tripped and any(c["status"]=="returned" for c in runtime.state["calls"].values()):
            tripped.append(True)
            raise KeyboardInterrupt("after durable raw return")
    monkeypatch.setattr(QueryRuntime,"changed",stop_after_return)
    with pytest.raises(KeyboardInterrupt): agent.solve(req)
    result=agent.solve(replace(req,resume=True))
    assert result.semantic_result["value"]==2 and len(model.calls)==3
    assert len(result.candidate_state["rows"])==2


def test_inflight_recovery_keeps_lost_charge_and_old_version_rejected(tmp_path):
    agent,model,req=setup(tmp_path,"interrupt")
    with pytest.raises(KeyboardInterrupt): agent.solve(req)
    result=agent.solve(replace(req,resume=True))
    assert result.resources["model_calls"]==4 and len(model.calls)==4
    old=tmp_path/"old.jsonl"
    old.write_text(json.dumps({"kind":"header","fingerprint":{"version":"r3-4.0"}})+'\n')
    with pytest.raises(ValueError,match="mismatch"):
        agent.solve(replace(req,checkpoint_path=str(old),resume=True))


def test_optional_denial_does_not_remove_required_tasks_and_visual_gate(tmp_path):
    agent,model,req=setup(tmp_path,max_refinements=0)
    req=replace(req,budget=R3Budget(max_model_calls=3))
    result=agent.solve(req)
    assert result.semantic_result["value"] is None and result.candidate_state["estimate"]==2
    assert len(model.calls)==3
    with pytest.raises(VisualBudgetExceeded): enforce_visual_budget(12289,10,12288,100)
    with pytest.raises(VisualBudgetExceeded): enforce_visual_budget(10,101,12288,100)
    enforce_visual_budget(8192,100,12288,100)


def test_default_sampling_caps_scope_and_source_rate(tmp_path):
    agent,model,req=setup(tmp_path)
    tasks=base_tasks(spec(),(2,10),agent.config)
    assert len(tasks)==2 and tasks[0]["span"]==[2,7]
    batch=agent.media.task_batch("v",tasks[0],Access((2,10)),source_fps=2)
    assert batch.requested_fps==2 and len({f.id for f in batch.frames})==len(batch.frames)
    assert all(2<=f.timestamp_seconds<=10 for f in batch.frames)
    with pytest.raises(AssertionError): agent.media.extract("v",(0,3),[1],Access((2,10)))


def test_four_recipes_contain_only_own_protocol_and_neutral_examples():
    assert set(PROMPTS)=={"cycles","instances","anchors","time"}
    for prompt in PROMPTS.values():
        assert VERSION in prompt and '"decision":"end"' in prompt
        assert "video_6480" not in prompt and "paper animal" not in prompt
    assert "companions maps" not in PROMPTS["cycles"]


def test_real_old_empty_outputs_cannot_supply_negative_coverage():
    saved=json.loads((Path(__file__).parent/"fixtures/r3_observation_failures.json").read_text(encoding="utf-8"))
    checked=0
    for record in saved:
        if record["role"]=="R3:observe":
            result=parse_lines(record["output"],"cycles",refs(),["task"])
            assert not result["complete"] and not result["negative"]
            checked+=1
        elif 'A pig.' in record["output"]:
            req=R3Request("v","q",choices={"A":"A pig.","B":"A cat."})
            assert exact_choice(json.loads(record["output"])["prediction"],req.choices)=="A"
    assert checked>=2


def test_replay_policy_and_late_unrelated_event_do_not_block_first():
    q=spec("first_occurrence",basis="onset")
    first,later=event(1,1,2),event(2,7,8)
    later["onset"]=None
    assert reduce_events(q,[first,later],scan((0,3)),(0,10))["closed"]
    state={}
    p=parse_lines(lines(row("replay")),"cycles",refs(),["task"])
    ingest(state,call(),p)
    assert not reduce_query(spec(),state,(0,8))["closed"]
    assert reduce_query(spec(count_replays=True),state,(0,8))["candidate_estimate"]==1
    assert reduce_query(spec(count_replays=False),state,(0,8))["candidate_estimate"]==0


def test_global_budget_rejects_optional_task_but_keeps_affordable_required_call(tmp_path):
    agent,model,req=setup(tmp_path)
    req=replace(req,budget=R3Budget(max_frame_exposures=2))
    runtime=QueryRuntime(model,agent.media,agent.config,req,Checkpoint(None,{},resume=False))
    prepared=SimpleNamespace(frames=[1,2,3],pixels=100,kind="images",sizes=[(32,32)]*3,video_frame_metadata=[])
    with pytest.raises(BudgetExhausted,match="frame"):
        runtime.call({"key":"optional"},[],prepared=prepared)
    # No pool refusal can poison a subsequent affordable required stage.
    out=runtime.call({"key":"query","kind":"query"},[{"role":"system","content":"R3:query_spec"}])
    assert out["status"]=="returned" and runtime.usage()["model_calls"]==1


def test_query_scope_intersection_and_invalid_upstream_has_no_parser_retry(tmp_path):
    from qwen3vl_agent.r3.query_engine import QueryEngine
    agent,model,req=setup(tmp_path)
    req=replace(req,allowed_scope=(2,8),query_scope=(1,5),query_spec={"op":"count_occurrences","target":"task","unit":"action_cycle"})
    engine=QueryEngine(agent,req)
    assert engine.scope(engine.determine_query())==(2,5)
    other=replace(req,query_spec={},checkpoint_path=str(tmp_path/"invalid.jsonl"))
    result=agent.solve(other)
    assert not any("R3:query_spec" in m[0]["content"] for m,k in model.calls)
    assert any(g["kind"]=="query" for g in result.evidence_gaps)


def test_runner_null_is_completed_and_fallback_does_not_become_supported_accuracy(tmp_path):
    from test_r1345_debug_runner import case, Model
    from qwen3vl_agent.debug12 import run_cases, PIPELINES
    from qwen3vl_agent.r3.types import R3Result
    cases=[case("R3",i) for i in (1,2)]
    def factory(pipeline,model,config):
        class Agent:
            def solve(self,request):
                prediction=None if request.request_id.endswith("1") else "A"
                basis="unresolved" if prediction is None else "forced_choice"
                return R3Result(prediction,{},"incomplete","unsupported",basis,trace={"answer_basis":basis})
        return Agent()
    configs={p:{"model":{},p.lower():{}} for p in PIPELINES}
    summary=run_cases(cases,cases,configs,tmp_path,{"signature":"r3-v5-null"},model_factory=lambda c:Model(),agent_factory=factory,gpu_check=None)
    assert summary["completed"]==2 and summary["failed"]==0 and summary["no_prediction"]==1
    assert summary["answer_match_rate_planned"]==.5
    assert summary["r3_supported_accuracy"]==0 and summary["r3_fallback_ratio"]==.5
    assert summary["unbacked_fallbacks"]==1 and summary["unbacked_fallback_correct"]==1


def test_actual_processor_visual_gate_runs_before_gpu_and_saves_receipt(tmp_path,monkeypatch):
    import sys
    import numpy as np
    from qwen3vl_agent.models.qwen3vl import Qwen3VLModel
    class Inputs(dict):
        input_ids=SimpleNamespace(shape=(1,10))
        def to(self,*args): raise AssertionError("GPU transfer must not happen")
    inputs=Inputs(image_grid_thw=np.array([[1,256,256]]))  # actual 16384 visual tokens
    class Processor:
        def apply_chat_template(self,*a,**kw): return "prompt"
        def __call__(self,**kw):
            assert kw["do_resize"] is False
            return inputs
    def vision(*a,**kw):
        assert kw["image_patch_size"]==16 and kw["return_video_metadata"]
        return None,None,{}
    monkeypatch.setitem(sys.modules,"torch",SimpleNamespace())
    monkeypatch.setitem(sys.modules,"qwen_vl_utils",SimpleNamespace(process_vision_info=vision))
    model=Qwen3VLModel("fake")
    model._loaded,model.model,model.processor=True,object(),Processor()
    path=tmp_path/"processor.json"
    with pytest.raises(VisualBudgetExceeded):
        model.generate([{"role":"user","content":"q"}],visual_token_limit=12288,preparation_receipt_path=str(path))
    receipt=json.loads(path.read_text())
    assert receipt["visual_tokens"]==16384 and receipt["processed_visual_grids"]["image_grid_thw"]==[[1,256,256]]


def test_measure_does_not_require_unqueried_boundaries_and_single_frame_is_unknown():
    q=spec("event_duration",targets=("a","b"),aggregation="difference")
    a,b=event(1,1,2,"a"),event(2,4,5,"b")
    a["onset"]=None;b["offset"]=None
    r=reduce_events(q,[a,b],scan(),(0,10))
    assert r["closed"] and r["value"]=={"min_sec":2,"max_sec":2}
    state={}
    p=parse_lines(lines(row("begin"),row("complete",at=("F05",))),"instances",refs(),["task"])
    ingest(state,call("one","instances"),p)
    es,_=events_from_rows(state["candidates"],spec("localize_event"))
    assert es[0]["onset"] is None and es[0]["offset"] is None


def test_selects_in_program_before_required_attribute_read(tmp_path):
    agent,_,req=setup(tmp_path)
    class SelectModel(FakeModel):
        def generate(self,messages,**kwargs):
            if "R3:confirm_candidates" in messages[0]["content"] or "R3:final_visual" in messages[0]["content"]:
                return super().generate(messages,**kwargs)
            self.calls.append((messages,kwargs))
            data=json.loads(messages[1]["content"][0]["text"])
            marks=data["core_frame_markers"]
            if data["query"]["attribute"] is None:
                rows=[row("begin",marks[:2],target="T1"),row("complete",marks[-2:],target="T1")]
            else:
                assert "program-selected instance" in data["task"]
                rows=[row("attribute",[marks[0],marks[-1]],target="T1",value="A vase.")]
            rows.append({"at":[marks[0],marks[-1]],"decision":"end","status":"complete","note":"Visible work and result."})
            return ModelOutput("\n".join(json.dumps(r) for r in rows))
    model=SelectModel()
    agent.model=model
    req=replace(req,question="Which vessel is made first?",choices={"A":"A vase.","B":"A bowl."},
                query_spec={"op":"first_occurrence","target":"task","unit":"production_instance","project":"shape"})
    result=agent.solve(req)
    assert result.prediction=="A" and result.answer_basis=="visual_direct"
    assert 3<=len(model.calls)<=5 and result.trace["critical_reviews"]


def test_truncated_local_retraction_cannot_reuse_old_coverage_to_close_count():
    state={}
    p=parse_lines(lines(row(at=("F02",)),row(at=("F05",))),"cycles",refs(),["task"])
    ingest(state,call(),p)
    raw=json.dumps(row("not_target",("F01","F03")))+'\n{"at":'
    p=parse_lines(raw,"cycles",refs(),["task"],truncated=True)
    ingest(state,call("review",replace_span=[0,8],kind="review"),p)
    result=reduce_query(spec(),state,(0,8))
    assert result["candidate_estimate"]==1 and not result["closed"]
    assert any(g["kind"]=="protocol" for g in result["gaps"])


def test_semantic_stage_brackets_preserve_membership_and_do_not_supply_target_coverage():
    q=spec(scope={"kind":"semantic","description":"stage","first_sec":4})
    stage=event(9,2,11,"stage")
    stage.update(onset=[2,3],offset=[10,11],purpose="scope")
    e=event(1,4,4)
    scans=scan((0,16))
    scans[0].update(targets=["stage"],purpose="scope")
    assert not reduce_events(q,[stage,e],scans,(0,16))["closed"]
    scans+=scan((2,7))
    good=reduce_events(q,[stage,e],scans,(0,16))
    assert good["closed"] and good["value"]==1 and good["required_scope"]==[2,7]
    edge=event(2,6.5,6.5)
    bad=reduce_events(q,[stage,e,edge],scans,(0,16))
    assert not bad["closed"] and any(g["kind"]=="scope_membership" for g in bad["gaps"])


def test_semantic_binding_is_necessary_before_counting_and_range_is_relative(tmp_path):
    import re
    agent,_,req=setup(tmp_path)
    agent.media.duration=16
    class StageModel(FakeModel):
        def generate(self,messages,**kwargs):
            if "R3:confirm_candidates" in messages[0]["content"] or "R3:final_visual" in messages[0]["content"]:
                return super().generate(messages,**kwargs)
            self.calls.append((messages,kwargs))
            data=json.loads(messages[1]["content"][0]["text"])
            marks=data["core_frame_markers"]
            if data["query"]["target"]=="stage":
                rows=[row("begin",("F03","F04"),target="T1"),row("complete",("F11","F12"),target="T1")]
            else:
                text=" ".join(p.get("text","") for p in messages[1]["content"][1:])
                pairs=re.findall(r"(F\d+) at ([\d.]+)s",text)
                mark=next((f for f,t in pairs if abs(float(t)-4)<1e-6),None)
                rows=[row(at=(mark,))] if mark else []
            rows.append({"at":[marks[0],marks[-1]],"decision":"end","status":"complete" if rows else "absent","note":"Readable sample with all requested targets reported."})
            return ModelOutput("\n".join(json.dumps(r) for r in rows))
    model=StageModel();agent.model=model
    req=replace(req,question="Count task cycles during the first 4 seconds of the stage.",choices={"A":"1","B":"2"},
        query_spec={"op":"count_occurrences","target":"task","unit":"action_cycle", "scope":{
            "kind":"semantic","description":"stage","first_sec":4,"source":"first 4 seconds of the stage"}})
    result=agent.solve(req)
    assert result.prediction=="A" and result.semantic_result["state"]=="supported"
    assert len(model.calls)<=6
    tasks=[c["task"] for c in result.trace["calls"]]
    assert tasks[0]["key"].startswith("scope:") and tasks[1]["core"]==[2,6]


def test_reread_that_resolves_unit_can_schedule_newly_necessary_attribute(tmp_path):
    agent,_,req=setup(tmp_path)
    agent.media.duration=4
    class LaterAttribute(FakeModel):
        def generate(self,messages,**kwargs):
            if "R3:final_visual" in messages[0]["content"]: return super().generate(messages,**kwargs)
            data=json.loads(messages[1]["content"][0]["text"])
            if "checks" in data:
                if any(any(d["decision"]=="continue" for d in c["candidate_descriptions"]) for c in data["checks"]):
                    self.calls.append((messages,kwargs))
                    c=data["checks"][0];marks=c["allowed_frames"]
                    return ModelOutput(json.dumps({"check":c["check"],"verdict":"corrected","at":[marks[0],marks[-1]],
                        "note":"Same vessel is completed in these images.","rows":[
                            row("begin",marks[:2],target="T1"),row("complete",marks[-2:],target="T1")]}))
                return super().generate(messages,**kwargs)
            self.calls.append((messages,kwargs))
            marks=data["core_frame_markers"]
            if data["query"]["attribute"] is not None:
                rows=[row("attribute",[marks[0],marks[-1]],target="T1",value="A vase.")]
            else:
                rows=[row("begin",marks[:2],target="T1"),row("continue",marks[-2:],target="T1")]
            rows.append({"at":[marks[0],marks[-1]],"decision":"end","status":"complete","note":"Readable work on one vessel."})
            return ModelOutput("\n".join(json.dumps(r) for r in rows))
    model=LaterAttribute();agent.model=model
    req=replace(req,question="Which vessel is made first?",choices={"A":"A vase.","B":"A bowl."},
        query_spec={"op":"first_occurrence","target":"task","unit":"production_instance","project":"shape"})
    result=agent.solve(req)
    assert result.prediction=="A" and result.support_level=="supported"
    assert result.trace["review_calls_used"]<=3 and result.candidate_state["revisions"]
