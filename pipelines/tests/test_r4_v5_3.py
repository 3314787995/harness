"""5.3 deterministic barriers: no real-model accuracy claims."""
import copy
from dataclasses import replace
import pytest
from r4_v5_fakes import Model, setup, task, card
from test_r4_v5 import refs, direct_card
from test_r4_v5_2 import check
from qwen3vl_agent.r4.collection_contracts import ContractError
from qwen3vl_agent.r4.inventory import EvidenceStore, parse_card
from qwen3vl_agent.r4.controller import CollectionController


def test_extra_valid_detections_retained_not_rejected():
    target, row = direct_card()
    catalog, aliases, window = refs()
    row["boxes"] = [{"ref":"F1" if i%2 else "F2","xyxy":[10+i,20,300,600]} for i in range(8)]
    parsed = parse_card(row,target,window,aliases,catalog)
    assert len(parsed["detections"]) == 8
    assert len(parsed["representative_detections"]) == 3
    assert parsed["membership"] == "accepted" and len(parsed["evidence_refs"]) == 2


@pytest.mark.parametrize("fault", ["reference","coordinate","condition"])
def test_validation_covers_unselected_detections_and_judgment(fault):
    target, row = direct_card()
    catalog, aliases, window = refs()
    row["boxes"] *= 8
    row = copy.deepcopy(row)
    if fault == "reference": row["boxes"][-1] = {"ref":"F99","xyxy":[10,20,30,40]}
    elif fault == "coordinate": row["boxes"][-1] = {"ref":"F1","xyxy":[10,20,10,40]}
    else: row["conditions"]["predicate"] = "visually present"
    with pytest.raises(ContractError):
        parse_card(row,target,window,aliases,catalog)


def presence(tmp_path, observe, inspect=None, duration=8):
    model=Model(task("semantic_category","missing_members",candidates=["writing","reading"]),observe,inspect)
    agent, req=setup(tmp_path,model,duration=duration)
    return model,agent,replace(req,choices={"A":"writing","B":"reading"})


def test_checked_absence_works_without_model_global_coverage(tmp_path):
    def observe(p,m):
        return {"checks":[check(p,"writing"),check(p,"reading","not_seen")],"input_gaps":[]}
    model,agent,req=presence(tmp_path,observe)
    result=agent.solve(req)
    assert result.prediction == "B",result.to_dict()
    assert len(model.calls)==2 and result.trace["sampling_schedule_completed"]


def test_legacy_absence_gap_requires_explicit_correction(tmp_path):
    def observe(p,m):
        out={"checks":[check(p,"writing"),check(p,"reading","not_seen")]}
        if len(m.calls)==2:
            out.update(coverage="partial",gaps=["The requested reading activity is not visually present."])
        else:
            out["input_gaps"]=[]
        return out
    model,agent,req=presence(tmp_path,observe)
    result=agent.solve(req)
    assert result.prediction == "B" and len(model.calls)==3
    assert "ambiguous_candidate_coverage" in str(model.calls[-1]["messages"])


def test_unreadable_input_is_not_absence(tmp_path):
    def observe(p,m):
        return {"checks":[check(p,c,"unreadable") for c in p["candidates"]["items"]],
                "input_gaps":[{"set":"items","candidate":c,"reason":"occluded"} for c in p["candidates"]["items"]]}
    model,agent,req=presence(tmp_path,observe)
    result=agent.solve(req)
    assert result.prediction is None
    assert not result.trace["sampling_schedule_completed"]


def test_contradictory_check_does_not_commit_absence(tmp_path):
    def observe(p,m):
        return {"checks":[check(p,"writing","not_seen")],"input_gaps":[
            {"set":"items","candidate":"writing","reason":"unreadable"}]}
    model,agent,req=presence(tmp_path,observe)
    result=agent.solve(req)
    assert result.prediction is None and not result.inventory["checks"]
    assert any(e["code"]=="candidate_gap_conflict" for e in result.failure["errors"])


def test_redundant_uncertain_check_does_not_spend_review(tmp_path):
    model,agent,req=presence(tmp_path,None)
    c=CollectionController(agent,req); c.compile(); c.store=EvidenceStore(c.spec); c.plan()
    c.store.state["checks"]={
        "weak":{"set":"items","candidate":"writing","state":"unreadable","reported_state":"seen","support":"related"},
        "strong":{"set":"items","candidate":"writing","state":"seen","support":"direct"},
        "decisive":{"set":"items","candidate":"reading","state":"seen","support":"direct"}}
    for row in c.store.state["checks"].values():
        row["window_id"] = next(iter(c.work["windows"]))
    keys=[k for k,r in c.choose_check_review(c.state())]
    assert "weak" not in keys and keys[0]=="decisive"


def test_identity_object_reference_repaired_without_guessing(tmp_path):
    def observe(p,m):
        return {"records":[card(p,1),card(p,2)],"coverage":"complete"}
    def identity(p,m):
        assert p["evidence_by_object"]
        n=sum(call["role"]=="identity" for call in m.calls)
        return {"relations":[{"left":p["left"][0],"right":p["right"][0],
            "relation":"UNKNOWN","basis":"uncertain","refs":[p["left"][0]] if n==1 else [],
            "facts":"Identity cannot be established."}]}
    model=Model(task(),observe,identity=identity)
    agent,req=setup(tmp_path,model)
    result=agent.solve(req)
    assert result.prediction is None
    calls=[c for c in model.calls if c["role"]=="identity"]
    assert len(calls)==2
    assert "identity_reference_is_object_id" in str(calls[-1]["messages"])
    assert all(r["relation"].lower() == "unknown" for r in result.inventory["relations"])
