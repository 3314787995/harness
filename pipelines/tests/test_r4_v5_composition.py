"""Mechanism fixtures for composed queries; no benchmark/video inference."""
import json
from dataclasses import replace
import pytest
from r4_v5_fakes import Model, task, card, setup
from qwen3vl_agent.r4 import MediaSource, ExternalSegment, InventoryProviderResult
from qwen3vl_agent.r4.types import InventorySpec
from qwen3vl_agent.r4.collection_contracts import ContractError


@pytest.mark.parametrize("operator,expected",[("UNION",3),("INTERSECTION",1),("DIFFERENCE",1),("ORDERED_COUNTS",[2,2])])
def test_composed_scoped_sets_through_agent(tmp_path,operator,expected):
    compiled=task("semantic_category")
    s=compiled["sets"][0]
    compiled["sets"]=[{**s,"set_id":"day1","scope":{"kind":"interval","interval":[0,8]}},
                      {**s,"set_id":"day3","scope":{"kind":"interval","interval":[8,16]}}]
    if operator == "ORDERED_COUNTS":
        compiled["operations"]=[{"operation_id":"answer","op":operator,"inputs":["day1","day3"]}]
    else:
        compiled["operations"]=[{"operation_id":"combined","op":operator,"inputs":["day1","day3"]},
                                {"operation_id":"answer","op":"COUNT_DISTINCT","inputs":["combined"]}]
    def observe(p,m):
        sid=p["sets"][0]["set_id"]
        fruits=["apple","pear"] if sid=="day1" else ["apple","orange"]
        return {"records":[card(p,i+1,name=n,query_value=n) for i,n in enumerate(fruits)],"coverage":"complete"}
    model=Model(compiled,observe)
    agent,req=setup(tmp_path,model,duration=16)
    req=replace(req,choices=(),output_protocol="auto")
    result=agent.solve(req)
    assert result.result_status=="supported",result.to_dict()
    assert json.loads(result.prediction)==expected
    assert len([c for c in model.calls if c["role"]=="discover_candidates"])==2
    assert not any(c["role"]=="identity" for c in model.calls)


def test_group_counts_feed_argmax(tmp_path):
    compiled=task(scope={"kind":"frame","timestamp_sec":0})
    compiled["operations"]=[{"operation_id":"groups","op":"GROUP_COUNT","inputs":["items"],"group_by":"color"},
                            {"operation_id":"answer","op":"ARGMAX","inputs":["groups"]}]
    compiled["choice_values"]={"A":"red","B":"blue"}
    def observe(p,m):
        rows=[card(p,i+1,attributes={"color":color}) for i,color in enumerate(["red","red","red","blue"])]
        return {"records":rows,"coexisting":[{"ids":[r["id"] for r in rows],"ref":rows[0]["boxes"][0]["ref"],"independent_objects":True}],"coverage":"complete"}
    model=Model(compiled,observe)
    agent,req=setup(tmp_path,model)
    req=replace(req,choices={"A":"red decorations","B":"blue decorations"})
    result=agent.solve(req)
    assert result.prediction=="A",result.to_dict()
    assert result.value_state["final"]["groups"]=={"blue":[1,1],"red":[3,3]}


def test_simultaneous_max_is_not_global_union(tmp_path):
    compiled=task(op="MAX_SIMULTANEOUS_COUNT")
    compiled['operations'][0]['group_by']='person'
    def observe(p,m):
        assert p['simultaneous_carriers']['items']==['person']
        f=list(p["catalog"])
        rows=[card(p,1,ref=f[0],attributes={"carrier":"K1"}),card(p,2,ref=f[0],attributes={"carrier":"K1"}),
              card(p,3,ref=f[1],attributes={"carrier":"K1"})]
        censuses={ref:({'K1':['O1','O2']} if ref==f[0] else {'K1':['O3']} if ref==f[1] else {})
                  for ref,meta in p['catalog'].items() if meta['region']=='core'}
        return {"records":rows,"coexisting":[{"ids":["O1","O2"],"ref":f[0],"independent_objects":True}],
                "snapshots":[{"set":"items","frames":censuses}],"coverage":"complete"}
    model=Model(compiled,observe)
    agent,req=setup(tmp_path,model)
    result=agent.solve(req)
    assert result.prediction=="B",result.to_dict()
    assert result.value_state["final"]["bounds"]==[2,2]
    assert not any(c["role"]=="identity" for c in model.calls)


def test_simultaneous_representatives_cannot_prove_missing_instants(tmp_path):
    compiled=task(op='MAX_SIMULTANEOUS_COUNT',scope={'kind':'interval','interval':[2,4]})
    def observe(p,m):
        context=next(k for k,v in p['catalog'].items() if v['region']=='context')
        core=[k for k,v in p['catalog'].items() if v['region']=='core']
        rows=[card(p,i+1,ref=ref,attributes={'carrier':'K1'}) for i,ref in enumerate((core[0],core[-1]))]
        for i,row in enumerate(rows):row['boxes'].append({'ref':context,'xyxy':[100+200*i,100,200+200*i,500]})
        return {'records':rows,'coexisting':[{'ids':['O1','O2'],'ref':context,'independent_objects':True}],'coverage':'complete'}
    model=Model(compiled,observe);agent,req=setup(tmp_path,model)
    from qwen3vl_agent.r4 import R4Budget
    r=agent.solve(replace(req,budget=R4Budget(max_model_calls=2)))
    assert r.prediction is None,r.to_dict()
    assert r.value_state['final']['bounds']==[1,2]
    assert not r.value_state['final']['per_instant_inventory_complete']
    assert r.coverage_manifest[0]['snapshot_gaps']


def test_group_count_after_difference_filters_nonmembers(tmp_path):
    compiled=task('semantic_category');s=compiled['sets'][0]
    compiled['sets']=[{**s,'set_id':'a','scope':{'kind':'interval','interval':[0,4]}},
                      {**s,'set_id':'b','scope':{'kind':'interval','interval':[4,8]}}]
    compiled['operations']=[{'operation_id':'difference','op':'DIFFERENCE','inputs':['a','b']},
                            {'operation_id':'answer','op':'GROUP_COUNT','inputs':['difference']}]
    def observe(p,m):
        names=['apple','pear'] if p['sets'][0]['set_id']=='a' else ['pear']
        return {'records':[card(p,i+1,name=n,query_value=n) for i,n in enumerate(names)],'coverage':'complete'}
    model=Model(compiled,observe);agent,req=setup(tmp_path,model)
    r=agent.solve(replace(req,choices=(),output_protocol='auto'))
    assert json.loads(r.prediction)=={'apple':1},r.to_dict()


def test_argmax_does_not_ignore_uninspected_groups(tmp_path):
    compiled=task()
    compiled['operations']=[{'operation_id':'groups','op':'GROUP_COUNT','inputs':['items'],'group_by':'color'},
                            {'operation_id':'answer','op':'ARGMAX','inputs':['groups']}]
    compiled['choice_values']={'A':'red','B':'blue'}
    def observe(p,m):
        first=sum(c['role']=='discover_candidates' for c in m.calls)==1
        rows=[card(p,i+1,attributes={'color':'red' if first else 'blue'}) for i in range(1 if first else 2)]
        return {'records':rows,'coexisting':[] if first else [{'ids':['O1','O2'],'ref':rows[0]['boxes'][0]['ref'],'independent_objects':True}], 'coverage':'complete'}
    model=Model(compiled,observe);agent,req=setup(tmp_path,model,duration=16)
    r=agent.solve(replace(req,choices={'A':'red','B':'blue'}))
    assert r.prediction=='B',r.to_dict()
    assert len(model.calls)==3


def test_literal_case_and_attribute_combination_have_different_keys(tmp_path):
    for namespace,extra,expected in [("text_value",{},"B"),("semantic_category",{"equivalence":"combination","attribute_keys":["top","bottom"]},"A")]:
        compiled=task(namespace,**extra)
        def observe(p,m):
            if namespace=="text_value":
                rows=[card(p,1,name="OPEN",raw_text="OPEN"),card(p,2,name="Open",raw_text="Open")]
            else:
                rows=[card(p,1,name="outfit",attributes={"top":"red","bottom":"blue"}),card(p,2,name="same outfit",attributes={"top":"red","bottom":"blue"})]
            return {"records":rows,"coverage":"complete"}
        model=Model(compiled,observe)
        agent,req=setup(tmp_path/namespace,model)
        result=agent.solve(req)
        assert result.prediction==expected,result.to_dict()


def test_first_anchor_binds_frame_and_stops_after_required_prefix(tmp_path):
    compiled=task(scope={"kind":"semantic","description":"first visible tool","selection":"first","result_kind":"frame"})
    def scope(p,m):
        f=next(k for k,v in p["catalog"].items() if v["region"]=="core")
        a=p["catalog"][f]["start_sec"]
        return {"bindings":[{"scope_id":"global","refs":[f],"interval":[a,a+0.04],"facts":"first visible tool in this input"}],"coverage":"complete"}
    model=Model(compiled,lambda p,m:{"records":[card(p)],"coverage":"complete"},scope=scope)
    agent,req=setup(tmp_path,model,duration=16)
    result=agent.solve(req)
    assert result.prediction=="A",result.to_dict()
    assert result.trace["executors"]["items"]=="local_instances"
    assert [c["role"] for c in model.calls]==["compile","scope","discover_candidates"]


def test_history_plan_completion_through_text_adapter(tmp_path):
    compiled=task("task_item",op="remaining_quantity",owner="me",task_id="trip",task_projection="remaining")
    class Provider:
        version="fixture-v1"
        def read(self,source_id,span):
            items=tuple(ExternalSegment(segment_id=k,source_id=source_id,kind="subtitle",start_sec=a,end_sec=a+1,
                text=t,speaker_id="me",alignment_status="aligned") for k,a,t in [("plan",0,"I plan to buy three apples."),("done",2,"I bought two apples.")])
            return InventoryProviderResult(items=items,coverage_status="complete",covered_intervals=((0,8),),alignment_error_sec=0,
                                           provider_version=self.version)
    def observe(p,m):
        refs=list(p["catalog"])
        base={"set_id":"items","item_key":"apples","owner":"me","task_id":"trip","unit":"item","binding_supported":True}
        return {"records":[],"task_updates":[{**base,"local_id":"U1","kind":"create_plan","quantity":3,"evidence_refs":[refs[0]]},
            {**base,"local_id":"U2","kind":"complete_item","quantity":2,"completion_predicate":"bought","evidence_refs":[refs[1]],"refers_to":"U1"}],"coverage":"complete"}
    model=Model(compiled,observe)
    agent,req=setup(tmp_path,model)
    agent.provider=Provider()
    req=replace(req,video_path="",sources=(MediaSource(req.video_path,recorded_at="2026-01-01T00:00:00Z",available_modalities=("subtitle",),actor_bindings={"me":"me"},task_context="trip"),),
                query_time="2026-01-01T00:00:08Z",benchmark_policy={"history_complete":True,"completion_predicate":"bought"})
    result=agent.solve(req)
    assert result.prediction=="A",result.to_dict()
    assert len(result.inventory["history"]["task_updates"])==2
    assert all(not c["payload"].get("catalog") or all(k.startswith("T") for k in c["payload"]["catalog"]) for c in model.calls if c["role"]!="compile")


def test_dependency_cycles_and_wrong_operand_types_rejected():
    compiled=task()
    compiled["operations"]=[{"operation_id":"a","op":"count_unique","inputs":["b"]},{"operation_id":"b","op":"union","inputs":["a"]}]
    with pytest.raises(ValueError,match="forward/cyclic"): InventorySpec.from_dict(compiled)
    compiled["operations"]=[{"operation_id":"a","op":"count_unique","inputs":["items"]},{"operation_id":"b","op":"union","inputs":["a"]}]
    with pytest.raises(ValueError,match="set operands"): InventorySpec.from_dict(compiled)
