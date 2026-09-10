"""v5.2 targeted CPU regressions, including actual v5.1 raw-response replay."""
import copy
import json
from pathlib import Path
from dataclasses import replace
import pytest
from test_r3_query_v51 import observation, review_record, fake_agent
from test_r3_query_v5 import spec, row, setup
from qwen3vl_agent.r3 import R3Config, QuerySpec
from qwen3vl_agent.r3.review import checks_for, plan_checks, parse_checks, commit_checks
from qwen3vl_agent.r3.query_reduce import reduce_query, covers
from qwen3vl_agent.r3.lines import parse_lines, EXAMPLES
from qwen3vl_agent.r3.candidates import ingest


def replay(case):
    data=json.loads((Path(__file__).parent/'fixtures/r3_v51_failures.json').read_text(encoding='utf-8'))
    d=next(x for x in data if case in x['case'])
    q=QuerySpec(**d['query']);state={}
    aliases={f'T{i}':t for i,t in enumerate(dict.fromkeys([q.target,*q.targets,*([q.anchor] if q.anchor else [])]),1)}
    for c in d['calls']:
        if not c['id'].startswith('base:'): continue
        a,b=c['task']['core']
        marks=[k for k,v in c['refs'].items() if a<=v['timestamp_seconds']<b or c['task'].get('last') and v['timestamp_seconds']==b]
        parsed=parse_lines(c['raw'],q.recipe,c['refs'],aliases,unit=q.unit,core_markers=marks)
        ingest(state,c,parsed)
    return q,state,d


def test_budget_default_and_stricter_override():
    assert R3Config().max_refinements==12
    assert R3Config.from_mapping({'max_refinements':3}).max_refinements==3
    with pytest.raises(ValueError): R3Config.from_mapping({'max_refinements':13})


@pytest.mark.parametrize('case',['action-count','225-3','251-3'])
def test_real_observations_still_cannot_become_proven_answers(case):
    q,s,_=replay(case)
    r=reduce_query(q,s,(0, max(c['span'][1] for c in s['coverage'])))
    assert r['value'] is None and not r['closed']


def test_real_paper_uses_semantic_evidence_before_protocol(tmp_path):
    q,s,_=replay('225-3')
    r=reduce_query(q,s,(0,60));checks=checks_for(q,r,s)
    assert checks[0]['kind'] not in {'protocol','coverage'}
    assert any(x.get('value')=='A pig.' for x in s['candidates'])
    a,_,_=setup(tmp_path)
    c=next(g for g in checks if g['kind']=='protocol') if any(g['kind']=='protocol' for g in checks) else {
        'key':'core','kind':'protocol','span':[0,32],'source_refs':[],'rows':[],'descriptions':[]}
    tasks=plan_checks([c],q,a.media,a.config,(0,60),30)
    assert tasks and all(t['fps']==2 for t in tasks)
    assert tasks[0]['span'][0]==c['span'][0] and tasks[-1]['span'][1]==c['span'][1]
    assert all(t['checks'][0]['parts']==1 and not t['depends_on'] for t in tasks)


def test_paper_candidates_are_independent_checks_with_all_actual_sources(tmp_path):
    from types import SimpleNamespace
    q,s,d=replay('225-3');r=reduce_query(q,s,(0,60))
    media=SimpleNamespace(catalog={v['id']:v for c in d['calls'] if c['id'].startswith('base:') for v in c['refs'].values()})
    tasks=plan_checks(checks_for(q,r,s),q,media,R3Config(),(0,60),30)
    # v5.4 scopes every part independently, including instance-unit ambiguities.
    assert len(tasks)<=12 and all(t['checks'][0]['parts']==1 for t in tasks if t['checks'][0]['kind']=='order')
    assert {d['value'] for t in tasks for d in t['checks'][0]['descriptions'] if d.get('value')}=={'A cat.','A pig.','A dog.'}
    for t in tasks:
        c=t['checks'][0]
        group=[other for other in tasks if other['checks'][0]['key']==c['key']]
        assert set(c['source_refs'])<=set(x for other in group for seg in other['segments'] for x in seg['required_refs'])
        assert t['estimated_frames']<=64


def test_joint_long_check_can_cite_more_than_four_legal_segments():
    q=QuerySpec('nth_occurrence','task','production_instance',k=2)
    s=observation(q=q,rows=[row('begin',('F01','F02'),target='T1'),row('complete',('F07','F08'),target='T1',value='Vase')])
    record,_=review_record(s,q)
    c=record['task']['checks'][0]
    marks=list(record['refs'])[:6]
    c['segments']=[{'span':[record['refs'][k]['timestamp_seconds']]*2} for k in marks]
    obj={'check':'C1','verdict':'confirmed','at':marks,'note':'Every supplied segment supports the specified process.'}
    assert not parse_checks(json.dumps(obj),record['task'],record['refs'],q,{'T1':'task'})['errors']


def test_real_cleaning_searches_anchor_before_tail_protocol(tmp_path):
    q,s,_=replay('251-3');r=reduce_query(q,s,(0,81.57))
    checks=checks_for(q,r,s)
    assert checks[0]['kind']=='anchor_search' and checks[0]['span'][0]==0
    assert checks[0]['descriptions']
    assert 'T2=closing a suitcase' in EXAMPLES['anchors']
    assert 'T1=subsequent activity' in EXAMPLES['anchors']


def scan_tasks(tmp_path):
    a,_,_=setup(tmp_path)
    c={'key':'core','kind':'protocol','span':[0,8],'source_refs':[],
       'rows':[],'events':[],'descriptions':[],'versions':{},'lineage':[],'aspects':[]}
    return plan_checks([c],spec(),a.media,a.config,(0,8),32)


def commit_scan(state,task,verdict,ident):
    a,b=task['span'];refs={'F01':{'id':ident+'a','source_frame_id':ident+'a','timestamp_seconds':a},
                         'F02':{'id':ident+'b','source_frame_id':ident+'b','timestamp_seconds':b}}
    obj={'check':'C1','verdict':verdict,'at':['F01','F02'],'note':'Readable local motion.'}
    if verdict=='corrected': obj['rows']=[row(at=('F01','F02'))]
    p=parse_checks(json.dumps(obj),task,refs,spec(),{'T1':'task'})
    assert not p['errors']
    record={'id':ident,'task':task,'refs':refs,'sampling':{'resolution_met':True,'requested_fps':16}}
    commit_checks(state,record,p,spec())
    return record,p


def test_independent_scan_mixed_verdicts_partial_commit_and_resume(tmp_path):
    tasks=scan_tasks(tmp_path);s={}
    first,p=commit_scan(s,tasks[0],'corrected','a')
    assert s['candidates'] and s['candidates'][0]['active']
    commit_checks(s,first,p,spec())
    assert len(s['candidates'])==1
    # First slice is durable before any other slice has returned.
    s=json.loads(json.dumps(s))
    commit_scan(s,tasks[1],'confirmed','b')
    assert len(s['resolved_checks'])==2 and covers(s['coverage'],(0,tasks[1]['span'][1]))
    assert not covers(s['coverage'],(0,8))
    assert not s.get('confirmations')  # scan readability cannot verify an event
    assert not reduce_query(spec(),s,(0,8))['closed']


def test_slice_cannot_retract_a_candidate_crossing_its_edges(tmp_path):
    tasks=scan_tasks(tmp_path)
    c={'key':'cross','kind':'protocol','span':[0,8],'source_refs':[],
       'rows':['cross'],'events':[],'descriptions':[{'bounds':[1,7],'target':'task'}],
       'versions':{'cross':1},'lineage':['cross'],'aspects':[]}
    agent,_,_=setup(tmp_path)
    ts=plan_checks([c],spec(),agent.media,agent.config,(0,8),32)
    assert all(not t['checks'][0]['rows'] for t in ts)
    assert ts[0]['checks'][0]['context_descriptions']


def test_cross_slice_conflict_invalidates_an_earlier_confirmed_result():
    s=observation();record,p=review_record(s);commit_checks(s,record,p,spec())
    assert reduce_query(spec(),s,(0,8))['closed']
    s['local_conflicts']=[{'rows':[s['candidates'][0]['id']],'span':[2,4],'reason':'local contradiction'}]
    result=reduce_query(spec(),s,(0,8))
    assert result['value'] is None and any(g['kind']=='identity' for g in result['gaps'])


def test_correction_can_fix_alias_but_cannot_invent_target():
    q=QuerySpec('next_after_anchor','later activity','activity',anchor='closing door')
    s=observation(q=q,rows=[row('begin',('F01','F02'),target='T1'),row('complete',('F03','F04'),target='T1')])
    record,_=review_record(s,q)
    obj={'check':'C1','verdict':'corrected','at':['F01','F04'],'note':'These pictures show the reference door closing.',
         'rows':[row('begin',('F01','F02'),target='T2'),row('complete',('F03','F04'),target='T2')]}
    aliases={'T1':'task','T2':'closing door'}
    parsed=parse_checks(json.dumps(obj),record['task'],record['refs'],q,aliases)
    assert not parsed['errors']
    commit_checks(s,record,parsed,q)
    assert any(r['active'] and r['target']=='closing door' for r in s['candidates'])
    obj['rows'][0]['target']='T99'
    assert parse_checks(json.dumps(obj),record['task'],record['refs'],q,aliases)['errors']


def test_default_can_run_fourth_review_without_changing_global_caps(tmp_path):
    agent,model,req=fake_agent(tmp_path,max_refinements=12,short_core_sec=1,refinement_fps=32)
    result=agent.solve(req)
    assert result.trace['review_calls_used']>3
    assert result.trace['review_calls_used']<=12
    before=len(model.calls)
    assert agent.solve(replace(req,resume=True)).to_dict()==result.to_dict()
    assert len(model.calls)==before


def test_unaffordable_joint_check_skips_before_call_and_other_check_runs(tmp_path,monkeypatch):
    from qwen3vl_agent.r3 import query_engine
    original=query_engine.plan_checks
    def add_impossible(*args,**kwargs):
        tasks=original(*args,**kwargs)
        if not tasks: return tasks
        impossible=[]
        for i in range(13):
            t=copy.deepcopy(tasks[0]);t['key']=f'impossible:{i}'
            t['checks']=[{**t['checks'][0],'key':'joint-impossible','parts':13,'part':i}]
            impossible.append(t)
        return impossible+tasks
    monkeypatch.setattr(query_engine,'plan_checks',add_impossible)
    a,m,req=fake_agent(tmp_path)
    result=a.solve(req)
    assert result.semantic_result['value']==2
    assert any(s.get('required_calls')==13 for s in result.candidate_state['skipped'])
    assert not any(c['id'].startswith('impossible') for c in result.trace['calls'])


def test_zero_confirmation_needs_all_independent_scan_slices(tmp_path):
    tasks=scan_tasks(tmp_path);s={}
    for t in tasks:
        c=t['checks'][0];c['kind']='zero_confirmation'
    commit_scan(s,tasks[0],'confirmed','first')
    assert not s['zero_reviewed']
    for i,t in enumerate(tasks[1:],1): commit_scan(s,t,'confirmed',str(i))
    assert s['zero_reviewed']


def test_v51_checkpoint_is_not_resumed(tmp_path):
    agent,model,req=fake_agent(tmp_path)
    Path(req.checkpoint_path).write_text(json.dumps({'kind':'header','fingerprint':{'version':'r3-5.1'}})+'\n')
    with pytest.raises(ValueError,match='mismatch'): agent.solve(replace(req,resume=True))
