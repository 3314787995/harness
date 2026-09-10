"""Only v5 collection, recovery, composition and runner contracts."""
import copy
import json
import re
from dataclasses import replace
from pathlib import Path

import pytest

from qwen3vl_agent.r4.collection_contracts import (ContractError, parse_compile, parse_json, validate,
    record_schema, envelope_schema, compile_schema)
from qwen3vl_agent.r4.collection_prompts import build_prompt, example_cases, validate_example
from qwen3vl_agent.r4.collection_reduce import (reduce_inventory, map_answer, physical_worlds, View, bounds)
from qwen3vl_agent.r4.inventory import EvidenceStore, parse_card
from qwen3vl_agent.r4.types import InventorySpec, R4Request, R4Budget, SetSpec
from qwen3vl_agent.r4.controller import executor
from r4_v5_fakes import Model, task, card, setup


def refs():
    catalog={"source-F1":{"id":"source-F1","kind":"frame","entry_id":"source-1","source_id":"v","source_frame_id":"frame1",
                           "start_sec":0,"end_sec":0,"membership_sets":["items"],"path":"unused"},
             "source-F2":{"id":"source-F2","kind":"frame","entry_id":"source-1","source_id":"v","source_frame_id":"frame2",
                           "start_sec":1,"end_sec":1,"membership_sets":["items"],"path":"unused"}}
    return catalog,{"F1":"source-F1","F2":"source-F2"},{"tile_id":"w","entry_id":"source-1","set_ids":["items"],"core":[0,2],"context":[0,2],"status":"complete"}


def direct_card(name="wrench", *, i=1, ref="F1", **kwargs):
    s=SetSpec("items","physical_instance","tools")
    payload={"sets":[{"set_id":"items","namespace":"physical_instance","conditions":{"target":"tools","predicate":"visible"}}],"catalog":{"F1":{"region":"core"}}}
    row=card(payload,i,name=name,ref=ref,**kwargs)
    return s,row


@pytest.mark.parametrize("ns",["physical_instance","semantic_category","text_value","task_item"])
def test_final_prompt_examples_are_valid_for_their_hypothetical_tasks(ns):
    s=InventorySpec.from_dict(task(ns)).sets[0]
    prompt=build_prompt("discover_candidates",{"sets":[],"catalog":{}},[s])
    blocks=re.findall(r"```json\n(.*?)\n```",prompt,re.S)
    cases=example_cases("discover_candidates",[s])
    assert len(blocks)==len(cases)
    for text,(name,target,response) in zip(blocks,cases):
        assert json.loads(text)==response
        validate_example("discover_candidates",target,json.loads(text))
    assert "one_record_shape_per_set" not in prompt and '"items":' not in prompt
    assert "triangle" not in prompt


def test_all_routes_and_public_compile_rules():
    for ns,scope,expected in [("physical_instance",{"kind":"frame","timestamp_sec":1},"local_instances"),
                             ("physical_instance",{"kind":"full"},"cross_segment_entities"),
                             ("semantic_category",{"kind":"full"},"category_text_combination"),
                             ("text_value",{"kind":"full"},"category_text_combination")]:
        s=InventorySpec.from_dict(task(ns,scope=scope))
        assert executor(s.sets[0],s.scope,s.operations)==expected
    req=R4Request("which is missing?",video_path="v")
    bad=task("task_item",evidence_relation="visually_present",required_modalities=["video"])
    bad["sets"][0].update(evidence_relation="visually_present",required_modalities=["video"])
    with pytest.raises(ContractError,match="task_item"):
        parse_compile(bad,req)
    for name,value in [("unresolved",True),("sets",[])]:
        bad=task(); bad[name]=value
        with pytest.raises(ContractError): parse_compile(bad,req)
    s=InventorySpec.from_dict(task("semantic_category","missing_members",candidates=["A","B"]))
    assert executor(s.sets[0],s.scope,s.operations)=="candidate_presence"


def test_qualified_local_objects_complete_without_identity(tmp_path):
    compiled=task(scope={"kind":"frame","timestamp_sec":0})
    def observe(p,m):
        a,b=card(p,1),card(p,2)
        return {"records":[a,b],"distinct_pairs":[{"left":"O1","right":"O2","refs":[a["boxes"][0]["ref"]],"independent_objects":True}],"coverage":"complete"}
    model=Model(compiled,observe)
    agent,req=setup(tmp_path,model)
    result=agent.solve(req)
    assert result.result_status=="supported",result.to_dict()
    assert result.prediction=="B"
    assert [c["role"] for c in model.calls]==["compile","discover_candidates"]
    assert result.resources["limits"]["total"]==7
    visual=model.calls[1]
    assert "source_time" not in json.dumps(visual["messages"])
    assert "start_sec" not in json.dumps(visual["payload"])
    assert "choices" not in visual["payload"]


def test_wrong_member_not_counted_and_valid_rows_survive(tmp_path):
    def observe(p,m):
        good=card(p,1)
        wrong=card(p,2,name="bottle",status="no")
        return {"records":[good,wrong],"coverage":"complete"}
    model=Model(task(),observe)
    agent,req=setup(tmp_path,model)
    result=agent.solve(req)
    assert result.prediction=="A",result.to_dict()
    assert [c["membership"] for c in result.inventory["cards"].values()]==["accepted","excluded"]


def test_category_aliases_and_excluded_shapes(tmp_path):
    def observe(p,m):
        return {"records":[card(p,1,name="red apple",cls="apple",query_value="apple"),
                           card(p,2,name="apple",query_value="apple"),
                           card(p,3,name="triangle",status="no")],"coverage":"complete"}
    model=Model(task("semantic_category"),observe)
    agent,req=setup(tmp_path,model)
    result=agent.solve(req)
    assert result.prediction=="A",result.to_dict()
    assert result.resources["calls_by_purpose"]["identity"]==0


def test_missing_candidate_empty_positive_refs_are_valid(tmp_path):
    def observe(p,m):
        f=next(iter(p["catalog"]))
        return {"records":[],"checks":[{"set":"items","candidate":n,"state":"seen" if n=="cooking" else "not_seen",
                "refs":[f] if n=="cooking" else [],"support":"direct","facts":"fixture evidence"} for n in p["candidates"]["items"]],"coverage":"complete"}
    model=Model(task("semantic_category","missing_members",candidates=["cooking","eating"]),observe)
    agent,req=setup(tmp_path,model)
    req=replace(req,choices={"A":"cooking","B":"eating"})
    result=agent.solve(req)
    assert result.prediction=="B",result.to_dict()
    assert len(model.calls)==2


@pytest.mark.parametrize("bad",[True,None,{"items":[]}])
def test_invalid_record_types_are_rejected(bad):
    s,row=direct_card()
    row["boxes"]=bad
    with pytest.raises(ContractError,match="boxes"):
        validate(row,record_schema(s))


def test_refs_integer_coordinates_crop_and_context():
    catalog,aliases,tile=refs()
    s,row=direct_card()
    for b in ([1.1,20,30,40],[0,0,0,10],[137.273]*4):
        row["boxes"][0]["xyxy"]=b
        with pytest.raises(ContractError): parse_card(row,s,tile,aliases,catalog)
    row["boxes"][0].update(ref="F99",xyxy=[0,0,1000,1000])
    with pytest.raises(ContractError,match="reference"): parse_card(row,s,tile,aliases,catalog)
    row["boxes"][0]["ref"]="F1"
    catalog["source-F1"]["crop_transform"]={"bbox_xyxy_1000":[100,200,500,800]}
    parsed=parse_card(row,s,tile,aliases,catalog)
    assert parsed["detections"][0]["bbox"]==[100,200,500,800]
    tile["core"]=[0.5,2]
    parsed=parse_card(row,s,tile,aliases,catalog)
    assert parsed["membership"]=="unknown" and "context_only_member" in parsed["issues"]


def test_recovery_only_replaces_bad_slot_same_media(tmp_path):
    def observe(p,m):
        n=sum(c["role"]=="discover_candidates" for c in m.calls)
        a,b=card(p,1),card(p,2,status="no")
        if n==1: b["boxes"][0]["xyxy"]=[0,0,0,10]
        return {"records":[a,b],"coverage":"complete"}
    model=Model(task(),observe)
    agent,req=setup(tmp_path,model)
    result=agent.solve(req)
    assert result.prediction=="A",result.to_dict()
    assert len(result.inventory["cards"])==2
    assert len(model.calls)==3
    a,b=model.calls[1:]
    assert [x for x in a["messages"][0]["content"] if x["type"]=="image"]==[x for x in b["messages"][0]["content"] if x["type"]=="image"]
    assert "invalid_bbox" in b["messages"][0]["content"][-1]["text"]
    assert not result.inventory["quarantined"]


def test_two_bad_outputs_fail_without_final_or_zero(tmp_path):
    model=Model(task(),lambda p,m:{"records":{"items":[]},"coverage":"complete"})
    agent,req=setup(tmp_path,model)
    result=agent.solve(req)
    assert result.result_status=="execution_failed",result.to_dict()
    assert result.prediction is None and len(model.calls)==3
    assert not result.inventory["cards"]
    assert result.failure["repeated_invalid_output"] is True
    assert result.resources["calls_by_purpose"]["recovery"]==1


def test_format_prompt_fault_does_not_start_observation(tmp_path,monkeypatch):
    import qwen3vl_agent.r4.collection_prompts as p
    def bad(*args): raise AssertionError("invalid program example")
    monkeypatch.setattr(p,"validate_example",bad)
    model=Model(task())
    agent,req=setup(tmp_path,model)
    result=agent.solve(req)
    assert result.failure["stage"]=="observe_prompt"
    assert result.failure["code"]=="prompt_contract_error"
    assert len(model.calls)==1


def test_identity_can_be_revoked_and_box_change_is_not_proof():
    spec=InventorySpec.from_dict(task())
    store=EvidenceStore(spec)
    catalog,aliases,tile=refs()
    ids=[]
    for i,ref in ((1,"F1"),(2,"F2"),(3,"F1")):
        s,row=direct_card(i=i,ref=ref)
        ids.append(store.commit_card(parse_card(row,s,tile,aliases,catalog),str(i),"call"))
    a,b,c=ids
    store.add_relation(a,b,"same","reidentification",list(catalog),"stable shape")
    store.add_relation(b,c,"same","reidentification",list(catalog),"stable shape")
    store.add_relation(a,c,"different","coexistence",["source-F1"],"independent objects")
    assert store.graph()["roots"][a]!=store.graph()["roots"][c]
    assert any(not r["active"] and r.get("reopened_reason") for r in store.state["relations"])
    with pytest.raises(ContractError):
        store.accept_relation({"left":a,"right":b,"relation":"DIFFERENT","basis":"bbox_width_changed", "refs":["F1","F2"],"facts":"width changed"},aliases,catalog,"bad",set(ids))


def test_unknown_identity_is_not_distinct():
    views={"s":View("physical_instance","entity",{"a","b"},{"a","b"},True)}
    graph={"different":set()}
    worlds=physical_worlds(views,graph)
    assert {len(w["s"].definite) for w,g in worlds}=={1,2}
    assert bounds(views["s"],graph)==[1,2]
    views["s"].closed=False
    assert bounds(views["s"],graph)==[1,None]


def test_numeric_result_never_overruled_by_model():
    req=R4Request("count?",video_path="v",choices={"A":"3","B":"5"})
    spec=InventorySpec.from_dict(task())
    assert map_answer({"final":{"supported":True,"value":6},"sampling_schedule_completed":True},spec,req)==(None,"option_mapping_conflict")


def test_checkpoint_return_replayed_and_version_change_rejected(tmp_path,monkeypatch):
    model=Model(task(),lambda p,m:{"records":[card(p)],"coverage":"complete"})
    agent,req=setup(tmp_path,model,request_kwargs={"checkpoint_path":str(tmp_path/'state.jsonl')})
    original=EvidenceStore.commit_card
    once=[True]
    def interrupt(*args,**kwargs):
        if once.pop() if once else False: raise KeyboardInterrupt()
        return original(*args,**kwargs)
    monkeypatch.setattr(EvidenceStore,"commit_card",interrupt)
    with pytest.raises(KeyboardInterrupt): agent.solve(req)
    before=len(model.calls)
    result=agent.solve(replace(req,resume=True))
    assert result.prediction=="A" and len(model.calls)==before
    assert agent.solve(replace(req,resume=True)).prediction=="A" and len(model.calls)==before
    agent.config=replace(agent.config,review_tokens=700)
    with pytest.raises(ValueError,match="mismatch"): agent.solve(replace(req,resume=True))


def test_interrupted_recovery_does_not_regain_allowance(tmp_path):
    n=[0]
    def observe(p,m):
        n[0]+=1
        if n[0]==2: return KeyboardInterrupt()
        return {"records":False,"coverage":"complete"}
    model=Model(task(),observe)
    agent,req=setup(tmp_path,model,request_kwargs={"checkpoint_path":str(tmp_path/'state.jsonl')})
    with pytest.raises(KeyboardInterrupt): agent.solve(req)
    before=len(model.calls)
    result=agent.solve(replace(req,resume=True))
    assert result.prediction is None and len(model.calls)==before
    assert result.result_status=="execution_failed"


def test_budget_detached_from_r3_and_physical_limits(tmp_path):
    assert R4Budget().terminal_call_reserve==0
    model=Model(task(),lambda p,m:{"records":[card(p)],"coverage":"complete"})
    agent,req=setup(tmp_path,model,duration=80)
    result=agent.solve(req)
    assert result.resources["limits"]["total"]==24
    assert result.resources["limits"]["frames"]==640
    assert result.resources["calls_by_purpose"]["identity"]<=4
    assert result.resources["calls_by_purpose"]["recovery"]<=2
