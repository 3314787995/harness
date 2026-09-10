"""Only v5.1 affected behavior: CPU images, saved raw outputs, and fake models."""
import copy
import hashlib
import json
from dataclasses import replace
from pathlib import Path
import pytest
from PIL import Image
from test_r3_query_v5 import (CASES, event, scan, spec, row, lines, refs, call, MemoryMedia, setup)
from qwen3vl_agent.r3 import R3Request, R3Config, R3VideoAgent, QuerySpec
from qwen3vl_agent.r3.query import rule_query, validate_query
from qwen3vl_agent.r3.lines import parse_lines
from qwen3vl_agent.r3.candidates import ingest, events_from_rows
from qwen3vl_agent.r3.query_reduce import reduce_query, reduce_events
from qwen3vl_agent.r3.evidence import gate, subjects, required_aspects
from qwen3vl_agent.r3.review import checks_for, plan_checks, parse_checks, commit_checks
from qwen3vl_agent.r3.query_engine import QueryEngine
from qwen3vl_agent.r3.query_media import Access, base_tasks
from qwen3vl_agent.r3.query_runtime import QueryRuntime
from qwen3vl_agent.models.base import BaseVideoModel, ModelOutput


def observation(state=None, rows=None, q=None):
    state={} if state is None else state
    q=q or spec()
    rows=rows or [row(at=("F03","F04"))]
    p=parse_lines(lines(*rows),q.recipe,refs(),{"T1":"task"},unit=q.unit,core_markers=list(refs()))
    ingest(state,call(recipe=q.recipe),p)
    return state


def review_record(state, q=None, verdict="confirmed", replacement=None):
    q=q or spec()
    reduced=reduce_query(q,state,(0,8))
    checks=checks_for(q,reduced,state)
    c=next(c for c in checks if c["rows"])
    c.update(part=0,parts=1,segments=[{"span":[0,8],"core":[0,8]}])
    task={"checks":[c],"kind":"review","key":"review:test","recipe":q.recipe,"span":[0,8],"core":[0,8]}
    rrefs={k:{**v,"source_frame_id":v["id"]} for k,v in refs().items()}
    obj={"check":"C1","verdict":verdict,"at":["F01","F08"],"note":"Actual supplied frames support this local judgment."}
    if replacement is not None: obj["rows"]=replacement
    raw=json.dumps(obj)
    parsed=parse_checks(raw,task,rrefs,q,{"T1":"task"})
    return {"id":"review:test","task":task,"refs":rrefs,"sampling":{"resolution_met":True,"requested_fps":16}},parsed


def test_unconfirmed_occurrence_is_only_estimate_no_negation_regex():
    state=observation(rows=[row(at=("F03","F04"),note="Object is placed; no launch occurs.")])
    r=reduce_query(spec(),state,(0,8))
    assert r["value"] is None and r["candidate_estimate"]==1 and not r["closed"]
    assert r["count_bounds"]=={"lower":0,"upper":None}
    record,p=review_record(state,verdict="rejected")
    commit_checks(state,record,p,spec())
    assert not any(e["active"] for e in state["candidates"])
    assert reduce_query(spec(),state,(0,8))["candidate_estimate"]==0
    assert not reduce_query(spec(),state,(0,8))["closed"]  # zero is a separate matter


def test_explicit_target_production_unit_endpoints_and_truncation():
    for r in [row("begin",("F01","F02")),row("appearance",("F01","F02"),target="T1")]:
        r.pop("target",None) if r["decision"]=="begin" else None
        p=parse_lines(lines(r),"instances",refs(),{"T1":"making a vessel"},unit="production_instance")
        assert not p["rows"] and p["errors"]
    good=row("begin",("F01","F02"),target="T1")
    raw=lines(good).replace('"F01", "F08"','"F03", "F08"')
    p=parse_lines(raw,"instances",refs(),{"T1":"task"},unit="production_instance",core_markers=list(refs()))
    assert len(p["rows"])==1 and not p["complete"]
    many=lines(*[row(at=("F02","F03")) for _ in range(9)])
    p=parse_lines(many,"cycles",refs(),{"T1":"task"},truncated=True)
    assert len(p["rows"])==9 and not p["complete"] and not p["negative"]


def test_numbered_display_hash_pts_core_raw_unchanged_and_budget(tmp_path):
    cfg=R3Config.from_mapping({"media":{"cache_dir":str(tmp_path/"cache")},"visual_token_target":320,"visual_token_limit":640})
    media=MemoryMedia(cfg,tmp_path)
    before=hashlib.sha256(media.image.read_bytes()).hexdigest()
    task={"key":"base","span":[0,5],"core":[1,4],"fps":8}
    batch=media.task_batch("v",task,Access((0,8)),source_fps=32)
    rendered=media.prepare(batch)
    assert rendered.kind=="numbered_images" and not rendered.video_frame_metadata
    assert len(rendered.frames)==len(batch.frames)==40
    assert rendered.pixels<=320*1024 and rendered.pixels==sum(w*h for w,h in rendered.sizes)
    assert hashlib.sha256(media.image.read_bytes()).hexdigest()==before
    for i,(frame,original) in enumerate(zip(rendered.frames,batch.frames),1):
        text,img=rendered.parts[2*i-2:2*i]
        d=rendered.display[f"F{i:02d}"]
        assert text["type"]=="text" and img["type"]=="image"
        assert f"{original.timestamp_seconds:.6f}s" in text["text"]
        assert d["header_pixels"]==32 and d["original_path"]==str(media.image)
        assert d["zone"]==("CORE" if 1<=frame.timestamp_seconds<4 else "CONTEXT")
        assert d["sha256"]==hashlib.sha256(Path(frame.path).read_bytes()).hexdigest()
        assert Image.open(frame.path).size==tuple(d["display_size"])
        assert media.catalog[frame.id]["path"]==str(media.image)


def test_confirmation_requires_versioned_multiple_frames_and_is_idempotent():
    state=observation()
    record,p=review_record(state)
    assert not p["errors"]
    commit_checks(state,record,p,spec());commit_checks(state,record,p,spec())
    r=reduce_query(spec(),state,(0,8))
    assert r["closed"] and r["value"]==1 and len(state["confirmations"])==1
    state["candidates"][0]["version"]+=1
    assert not reduce_query(spec(),state,(0,8))["closed"]
    bad=json.dumps({"check":"C1","verdict":"confirmed","at":["F03"],"note":"Still image."})
    assert not parse_checks(bad,record["task"],record["refs"],spec(),{"T1":"task"})["checks"]


def test_incomplete_correction_keeps_old_and_good_other_check_is_independent():
    state=observation()
    record,p=review_record(state,verdict="corrected",replacement=[row(at=("F99",))])
    assert p["errors"] and not p["checks"]
    commit_checks(state,record,p,spec())
    assert state["candidates"][0]["active"]
    record,p=review_record(state)
    record["task"]["checks"].append({**record["task"]["checks"][0],"key":"another"})
    raw=json.dumps({"check":"C1","verdict":"confirmed","at":["F02","F03"],"note":"Full change is visible."})+'\n{"check":"C2",'
    p=parse_checks(raw,record["task"],record["refs"],spec(),{"T1":"task"},truncated=True)
    assert len(p["checks"])==1 and p["errors"]


def test_correction_splits_and_merge_retires_old_rows_and_confirms_only_matters():
    state=observation()
    replacement=[row(at=("F02","F03")),row(at=("F05","F06"))]
    record,p=review_record(state,verdict="corrected",replacement=replacement)
    assert not p["errors"]
    commit_checks(state,record,p,spec())
    r=reduce_query(spec(),state,(0,8))
    assert r["value"]==2 and not state["candidates"][0]["active"]
    assert len(state["revisions"][-1]["added"])==2
    assert all("onset" not in p["aspects"] for p in state["confirmations"])
    assert all(x["lineage"]==["one:1"] for x in state["candidates"] if x["active"])
    # Merge two explicit competing fragments in a single scoped visual reconstruction.
    both={"candidates":copy.deepcopy([x for x in state["candidates"] if x["active"]]),"coverage":state["coverage"]}
    es,_=events_from_rows(both["candidates"],spec())
    check=checks_for(spec(),reduce_query(spec(),both,(0,8)),both)[0]
    check.update(events=[e["id"] for e in es],rows=[x["id"] for x in both["candidates"]],
        versions={x["id"]:x["version"] for x in both["candidates"]},part=0,parts=1,segments=[{"span":[0,8],"core":[0,8]}])
    record["id"]="merge";record["task"]["checks"]=[check]
    raw=json.dumps({"check":"C1","verdict":"corrected","at":["F02","F06"],"note":"One cycle spans these fragments.","rows":[row(at=("F02","F06"))]})
    p=parse_checks(raw,record["task"],record["refs"],spec(),{"T1":"task"})
    commit_checks(both,record,p,spec())
    assert reduce_query(spec(),both,(0,8))["value"]==1


@pytest.mark.parametrize("op,kw,want",CASES)
def test_all_twelve_operations_gate_their_own_required_evidence(op,kw,want):
    q=spec(op,**kw)
    es=[event(i,2*i,2*i+1,value=f"v{i}") for i in (1,2,3)]
    if op=="next_after_anchor": es.append(event(9,.1,1,"anchor"))
    if op=="previous_before_anchor": es.append(event(9,8,9,"anchor"))
    if op=="cooccurrence_frequency":
        for e in es: e["companions"]={"companion":True}
    state={"candidates":[{"id":e["id"],"version":1,"active":True} for e in es]}
    result=reduce_events(q,es,scan(),(0,10))
    closed=gate(q,copy.deepcopy(result),state)
    assert not closed["closed"] and closed["value"] is None
    subjects_=subjects(es,state)
    state["confirmations"]=[{"signature":subjects_[e["id"]]["signature"],"aspects":list(required_aspects(q,e,result["selected"]))} for e in es]
    if q.template=="NEIGHBOR": state["relation_confirmations"]=[{"kind":"adjacency_confirmation","signatures":sorted(subjects_[x]["signature"] for x in result["selected"])}]
    assert gate(q,copy.deepcopy(result),state)["closed"]
    # Removing one relevant proof prevents closure; unrelated events are ignored.
    chosen=result["selected"][0]
    state["confirmations"]=[p for p in state["confirmations"] if p["signature"]!=subjects_[chosen]["signature"]]
    assert not gate(q,copy.deepcopy(result),state)["closed"]


def test_targeted_beginning_and_competitors_not_video_midpoint(tmp_path):
    cfg=R3Config.from_mapping({"media":{"cache_dir":str(tmp_path/"cache")}})
    media=MemoryMedia(cfg,tmp_path)
    media.extract("v",(0,60),[0,.125,.25,29,30],Access((0,60)))
    ids=list(media.catalog)
    check={"key":"intro","kind":"order","span":[0,60],"source_refs":ids[:3],"rows":["x"],"events":["x"],"descriptions":[{"note":"Title candidate"}]}
    tasks=plan_checks([check],spec(),media,cfg,(0,60),32)
    assert tasks and max(t["span"][1] for t in tasks)<1
    check.update(key="competition",source_refs=[ids[0],ids[-1]],rows=["x","y"],events=["x","y"],descriptions=[{"note":"First candidate","bounds":[0,0]},{"note":"Competitor","bounds":[30,30]}])
    tasks=plan_checks([check],spec(),media,cfg,(0,60),32)
    spans=[s["span"] for t in tasks for s in t["segments"]]
    assert any(a<=0<=b for a,b in spans) and any(a<=30<=b for a,b in spans)
    # v5.4 never copies the out-of-window competitor into a local replacement task.
    assert {d['note'] for t in tasks for d in t['checks'][0]['descriptions']}=={'First candidate','Competitor'}
    assert all(t['span'][0]<=d['bounds'][0]<=d['bounds'][1]<=t['span'][1] for t in tasks for d in t['checks'][0]['descriptions'])
    for t in tasks: assert len(media.task_batch("v",t,Access((0,60)),source_fps=32).frames)<=64


def test_long_required_core_split_keeps_all_intervals_independently(tmp_path):
    agent,_,_=setup(tmp_path)
    c={"key":"badcore","kind":"protocol","span":[0,12],"source_refs":[],"rows":[],"events":[],"descriptions":[]}
    ts=plan_checks([c],spec(),agent.media,agent.config,(0,12),32)
    assert len(ts)>3 and ts[0]["span"][0]==0 and ts[-1]["span"][1]==12
    assert all(t["estimated_frames"]<=64 for t in ts)
    assert not ts[-1]["depends_on"] and all(a["span"][1]==b["span"][0] for a,b in zip(ts,ts[1:]))


def test_batching_does_not_expand_another_checks_allowed_segments(tmp_path):
    agent,_,_=setup(tmp_path)
    agent.media.extract("v",(0,8),[2,6],Access((0,8)))
    checks=[{"key":str(i),"kind":"confirmation","span":[t,t],"source_refs":[f"pts_{t:.6f}"],
             "rows":[str(i)],"events":[str(i)],"descriptions":[],"purpose":"query"} for i,t in enumerate([2,6])]
    tasks=plan_checks(checks,spec(),agent.media,agent.config,(0,8),32)
    assert len(tasks)==2 and all(len(t["checks"])==1 for t in tasks)
    assert all(len(t["segments"])==1 for t in tasks)
    assert len(tasks[0]["checks"][0]["segments"])==len(tasks[1]["checks"][0]["segments"])==1
    assert tasks[0]["checks"][0]["segments"][0]["span"][1]<3


def test_stage_confirmation_keeps_end_outside_derived_query_interval(tmp_path):
    agent,_,_=setup(tmp_path)
    agent.media.extract("v",(0,16),[2,3,10,11],Access((0,16)))
    c={"key":"stage","kind":"confirmation","purpose":"scope","span":[2,11],
       "source_refs":list(agent.media.catalog),"rows":["stage"],"events":["stage"],"descriptions":[]}
    tasks=plan_checks([c],spec(),agent.media,agent.config,(2,7),32,legal_scope=(0,16))
    spans=[s["span"] for t in tasks for s in t["segments"]]
    assert any(a<=11<=b for a,b in spans) and all(0<=a<b<=16 for a,b in spans)


@pytest.mark.parametrize("question,op,anchor,k",[
 ("What is the second paper animal made in this video?","nth_occurrence",None,2),
 ("Which wooden toy was created fourth?","nth_occurrence",None,4),
 ("After the woman cleans her floor, what does she do next?","next_after_anchor","the woman cleans her floor",None),
 ("Before opening the gate, what did she do previously?","previous_before_anchor","opening the gate",None)])
def test_general_query_rules(question,op,anchor,k):
    req=R3Request("v",question)
    q=validate_query(rule_query(req),req)
    assert (q.op,q.anchor,q.k)==(op,anchor,k)
    assert q.scope=={"kind":"full"} and (not anchor or q.target!=q.anchor)
    if k: assert q.unit=="production_instance" and q.basis=="ambiguous"
    else: assert q.unit=="activity"


class V51Model(BaseVideoModel):
    def __init__(self,mode="confirm"):
        super().__init__("fake-v51");self.calls=[];self.mode=mode
    def load(self): pass
    def unload(self): pass
    def generate(self,messages,**kwargs):
        self.calls.append((messages,kwargs))
        system=messages[0]["content"]
        if "R3:final_visual" in system: return ModelOutput('{"prediction":"B"}')
        if "R3:query_spec" in system: return ModelOutput('{}')
        data=json.loads(messages[1]["content"][0]["text"])
        if "checks" in data:
            if self.mode=="interrupt" and sum("R3:confirm_candidates" in x[0][0]["content"] for x in self.calls)==1: raise KeyboardInterrupt("review interrupted")
            out=[]
            for c in data["checks"]:
                marks=c["allowed_frames"]
                out.append({"check":c["check"],"verdict":"uncertain" if self.mode=="uncertain" else "confirmed",
                            "at":[marks[0],marks[-1]],"note":"Actual multi-frame movement confirms the specified unit."})
            return ModelOutput("\n".join(json.dumps(r) for r in out),{"output_tokens":80})
        marks=data["core_frame_markers"]
        rows=[] if self.mode=="negative" else [row(at=marks[len(marks)//2:len(marks)//2+2])]
        rows.append({"at":[marks[0],marks[-1]],"decision":"end","status":"absent" if self.mode=="negative" else "complete","note":"Readable core; target absent." if self.mode=="negative" else "All relevant movement reported."})
        return ModelOutput("\n".join(json.dumps(r) for r in rows),{"output_tokens":80})


def fake_agent(tmp_path,mode="confirm",**config):
    tmp_path.mkdir(parents=True,exist_ok=True)
    agent,_,req=setup(tmp_path,**config)
    model=V51Model(mode);agent.model=model
    return agent,model,req


def test_executor_confirmation_shared_load_isolation_and_resume(tmp_path):
    agent,model,req=fake_agent(tmp_path)
    result=agent.solve(req)
    assert result.prediction=="A" and result.semantic_result["value"]==2 and result.support_level=="supported"
    assert result.trace["review_calls_used"] in {1,2} and len(model.calls)==2+result.trace["review_calls_used"]
    for messages,kwargs in model.calls:
        assert '"choices"' not in json.dumps(messages)
        assert not any(x.get("type")=="video" for x in messages[1]["content"])
        assert "video_frame_metadata" not in kwargs
    assert agent.solve(replace(req,resume=True)).to_dict()==result.to_dict()
    assert all(c["display_transforms"] for c in result.trace["calls"])


def test_uncertain_checks_are_not_retried_and_final_does_not_change_value(tmp_path):
    agent,model,req=fake_agent(tmp_path,"uncertain")
    result=agent.solve(req)
    assert result.semantic_result["value"] is None and result.candidate_state["estimate"]==2
    assert result.answer_basis=="partial_choice" and result.prediction=="B"
    assert result.trace["review_calls_used"]<=3
    assert sum("R3:final_visual" in m[0]["content"] for m,k in model.calls)==1


def test_returned_check_raw_resume_and_interrupted_check_share_three_allowance(tmp_path,monkeypatch):
    agent,model,req=fake_agent(tmp_path)
    original=QueryRuntime.changed
    stopped=[]
    def interrupt(runtime):
        original(runtime)
        if not stopped and any(c["status"]=="returned" and c["task"].get("kind")=="review" for c in runtime.state["calls"].values()):
            stopped.append(True);raise KeyboardInterrupt("after raw saved")
    monkeypatch.setattr(QueryRuntime,"changed",interrupt)
    with pytest.raises(KeyboardInterrupt): agent.solve(req)
    before=len(model.calls)
    result=agent.solve(replace(req,resume=True))
    assert result.semantic_result["value"]==2 and len(model.calls)==before+1
    # v5.3 resumes the returned first check without inference, then runs the
    # distinct second check (previous versions batched both in the first call).
    messages=[json.dumps(m,sort_keys=True) for m,k in model.calls]
    assert len(messages)==len(set(messages))


def test_three_budget_including_lost_review_and_no_optional_poison(tmp_path):
    agent,model,req=fake_agent(tmp_path,"interrupt")
    with pytest.raises(KeyboardInterrupt): agent.solve(req)
    result=agent.solve(replace(req,resume=True))
    assert result.trace["review_calls_used"]<=3
    assert any(c["is_recovery"] for c in result.trace["calls"])
    # No review allowance: every necessary base core still runs; final remains partial.
    agent,model,req=fake_agent(tmp_path/"other",max_refinements=0)
    result=agent.solve(req)
    assert result.semantic_result["value"] is None and result.candidate_state["estimate"]==2
    assert sum("R3:observe_cycles" in m[0]["content"] for m,k in model.calls)==2


def test_query_ambiguity_uses_one_parse_no_repair(tmp_path):
    agent,model,req=fake_agent(tmp_path)
    req=replace(req,question="After opening either gate during the first 7 seconds, what happens next?")
    assert rule_query(req) is None
    result=agent.solve(req)
    assert sum("R3:query_spec" in m[0]["content"] for m,k in model.calls)==1
    assert result.answer_basis=="forced_choice" and result.semantic_result["value"] is None


def test_interrupted_query_does_not_run_a_second_semantic_parser(tmp_path):
    agent,model,req=fake_agent(tmp_path)
    generate=model.generate
    def interrupt(messages,**kwargs):
        if "R3:query_spec" in messages[0]["content"]:
            model.calls.append((messages,kwargs));raise KeyboardInterrupt("query interrupted")
        return generate(messages,**kwargs)
    model.generate=interrupt
    req=replace(req,question="Which activity comes later, depending on the relevant scene?")
    with pytest.raises(KeyboardInterrupt): agent.solve(req)
    result=agent.solve(replace(req,resume=True))
    assert sum("R3:query_spec" in m[0]["content"] for m,k in model.calls)==1
    assert result.semantic_result["value"] is None


def test_v50_checkpoint_rejected(tmp_path):
    agent,model,req=fake_agent(tmp_path)
    Path(req.checkpoint_path).write_text(json.dumps({"kind":"header","fingerprint":{"version":"r3-5.0"}})+'\n')
    with pytest.raises(ValueError,match="mismatch"): agent.solve(replace(req,resume=True))


def test_real_v50_failures_replay_cannot_close_counts_or_adopt_title():
    records=json.loads((Path(__file__).parent/"fixtures/r3_v50_failures.json").read_text(encoding="utf-8"))
    state={};kept=0
    for c in records:
        if c["task"].get("recipe")!="cycles": continue
        core=[k for k,v in c["refs"].items() if c["task"]["core"][0]<=v["timestamp_seconds"]<c["task"]["core"][1]]
        p=parse_lines(c["raw"],"cycles",c["refs"],{"T1":"task"},unit="action_cycle",core_markers=core)
        assert not p["complete"]
        kept+=sum(r["decision"]=="occurrence" for r in p["rows"])
        ingest(state,{"id":c["call_id"],"task":c["task"],"refs":c["refs"],"sampling":{"resolution_met":True,"requested_fps":8}},p)
    r=reduce_query(spec(),state,(0,15.384097488722817))
    assert kept>=2 and r["value"] is None and r["count_bounds"]["lower"]==0
    title=[c for c in records if c["case"]=="R3-MME-225-3" and c["task"].get("kind")=="review"][0]
    p=parse_lines(title["raw"],"instances",title["refs"],{"T1":"making a paper animal"},unit="production_instance",truncated=True)
    assert not p["complete"] and not any(r["decision"]=="appearance" for r in p["rows"])
    bad_query=next(c for c in records if c["case"]=="R3-MME-251-3" and c["task"]["kind"]=="query")
    req=R3Request("v",bad_query["question"])
    assert validate_query(rule_query(req),req).anchor is not None


def test_no_information_skips_without_charging_or_rescheduling(tmp_path):
    agent,model,req=fake_agent(tmp_path,refinement_fps=8)
    result=agent.solve(req)
    assert result.trace["review_calls_used"]==0 and result.candidate_state["skipped"]
    assert len(model.calls)==3  # two bases, one visual final
    keys=[x["task"] for x in result.candidate_state["skipped"]]
    assert len(keys)==len(set(keys)) and result.semantic_result["value"] is None


def test_unrelated_later_unverified_candidate_does_not_block_selected_prefix():
    q=QuerySpec("first_occurrence","task","production_instance",basis="onset",project="shape")
    s=observation(q=q,rows=[row("begin",("F01","F02"),target="T1"),row("complete",("F03","F04"),target="T1",value="Vase")])
    record,p=review_record(s,q)
    commit_checks(s,record,p,q)
    later={**s["candidates"][0],"id":"later","call_id":"later","bounds":[7,7.5],"lineage":["later"],"source_refs":["late1","late2"]}
    s["candidates"].append(later)
    assert reduce_query(q,s,(0,8))["closed"]
