"""v5.5 host tasks and bounded comparisons; simulated model, no GPU."""
import copy,json,re
from dataclasses import replace
from pathlib import Path
import pytest
from r4_v5_fakes import Model,setup,task
from test_r4_v5 import refs
from test_r4_v5_3 import presence
from qwen3vl_agent.r4.controller import CollectionController
from qwen3vl_agent.r4.inventory import EvidenceStore,parse_card
from qwen3vl_agent.r4.types import SetSpec
from qwen3vl_agent.r4.interaction import bind_interaction,decode_observation,decode_pair,interaction_prompt,pair_prompt
from qwen3vl_agent.r4.collection_contracts import ContractError,conditions


def member(p,i=1,label='wrench',judgment='yes'):
    sid=next(iter(p['interaction']['requirements']))
    t=next(t for t in p['sets'] if t['set_id']==sid)
    ref=next((f for f,m in p['catalog'].items() if m['region']=='core'),next(iter(p['catalog'])))
    r={'set':sid,'object':'O'+str(i),'label':label,
       'judgments':{j:judgment for j in p['interaction']['requirements'][sid]}}
    if t['namespace']=='physical_instance': r['boxes']=[{'ref':ref,'xyxy':[i*100,100,i*100+80,500]}]
    else: r['refs']=[ref]
    if t['namespace']=='text_value': r['text']=label
    return r


def answers(p):
    core=next(f for f,m in p['catalog'].items() if m['region']=='core')
    return [{'task':q,'status':'observed' if t['candidate']=='writing' else 'absent',
             'refs':[core] if t['candidate']=='writing' else [],
             **({'note':'A person is actively writing letters.'} if t['candidate']=='writing' else {})}
            for q,t in p['interaction']['tasks'].items()]


@pytest.mark.parametrize('namespace,label',[('physical_instance','wrench'),('semantic_category','apple'),('text_value','OPEN')])
def test_compact_member_executes_without_prose_boilerplate(tmp_path,namespace,label):
    def observe(p,m): return {'members':[member(p,label=label)],'coverage':'complete'}
    model=Model(task(namespace),observe); agent,req=setup(tmp_path,model)
    result=agent.solve(req)
    assert result.prediction=='A',result.to_dict()
    assert len(model.calls)==2 and len(result.inventory['cards'])==1
    c=next(iter(result.inventory['cards'].values()))
    assert c['actual_class']==label and c['conditions']=={'target':'yes','predicate':'yes'}
    text=model.calls[-1]['messages'][-1]['content'][-1]['text']
    assert 'Q tasks' in text and 'Return one JSON object' in text
    assert 'answer_label' not in text and 'human_review' not in text


def test_fixed_candidates_have_no_support_or_name_copy(tmp_path):
    model,agent,req=presence(tmp_path,lambda p,m:{'answers':answers(p)})
    result=agent.solve(req)
    assert result.prediction=='B' and len(model.calls)==2
    checks=list(result.inventory['checks'].values())
    assert {c['candidate'] for c in checks}=={'writing','reading'}
    assert len(checks)==2


def test_local_instances_need_explicit_same_frame_independence(tmp_path):
    def observe(p,m):
        rows=[member(p,1),member(p,2)]
        return {'members':rows,'coverage':'complete','separate':[{'objects':['O1','O2'],'ref':rows[0]['boxes'][0]['ref']}]}
    model=Model(task(scope={'kind':'frame','timestamp_sec':0}),observe); agent,req=setup(tmp_path,model)
    result=agent.solve(req)
    assert result.prediction=='B',result.to_dict()
    assert len(model.calls)==2


@pytest.mark.parametrize('fault',['new_task','duplicate','omit','illegal_ref','judgment_text'])
def test_bad_protocol_is_not_silently_repaired(tmp_path,fault):
    def observe(p,m):
        rows=answers(p)
        if fault=='new_task': rows[0]['task']='washing dishes'
        if fault=='duplicate': rows.append(copy.deepcopy(rows[0]))
        if fault=='omit': rows.pop()
        if fault=='illegal_ref': rows[0]['refs']=['F999']
        if fault=='judgment_text': rows[0]['status']='visually present'
        return {'answers':rows}
    model,agent,req=presence(tmp_path,observe)
    result=agent.solve(req)
    assert result.prediction is None
    txn=result.trace['transactions']['tile_000000:discover_candidates']
    assert len(txn['call_ids'])==2
    assert not any(c['state']=='seen' for c in result.inventory['checks'].values())


def test_recovery_reasks_same_tasks_without_losing_committed_checks(tmp_path):
    def observe(p,m):
        rows=answers(p)
        if len(m.calls)==2: rows[1]['status']='not_seen'
        return {'answers':list(reversed(rows))}
    model,agent,req=presence(tmp_path,observe)
    result=agent.solve(req)
    assert result.prediction=='B' and len(result.inventory['checks'])==2
    first,recovery=model.calls[1:]
    assert first['payload']['interaction']==recovery['payload']['interaction']
    assert first['messages'][0]['content'][:-1]==recovery['messages'][0]['content'][:-1]
    assert 'REOBSERVE' in recovery['messages'][0]['content'][-1]['text']


def test_host_skips_empty_check_task_and_reopens_after_retraction(tmp_path):
    model,agent,req=presence(tmp_path,None,duration=16)
    c=CollectionController(agent,req); c.compile(); c.store=EvidenceStore(c.spec); c.plan()
    c.store.state['checks']={str(i):{'set':'items','candidate':name,'state':'seen','window_id':'prior'}
                             for i,name in enumerate(('writing','reading'))}
    window=next(iter(c.work['windows'].values()))
    c.observe(window)
    assert len(model.calls)==1 # compile only; no media/observation call
    assert window['host_skipped'] and window['status']=='complete'
    txn=c.work['transactions'][window['tile_id']+':discover_candidates']
    assert txn['call_ids']==[] and txn['catalog']=={}
    c.store.state['checks']['1']['state']='not_seen'; c.refresh_gaps()
    assert window['status']=='partial'
    assert any(g['candidate']=='reading' for g in window['candidate_gaps'])


def test_context_only_compact_record_stays_unknown():
    catalog,aliases,window=refs()
    target=SetSpec('items','physical_instance','tools')
    for meta in catalog.values(): meta['start_sec']=window['core'][1]+1
    p={'sets':[],'catalog':{},'candidates':{}}; bind_interaction(p,[target],'discover_candidates')
    raw={'members':[{'set':'items','object':'O1','label':'wrench','judgments':{'J1':'yes','J2':'yes'},
                     'boxes':[{'ref':'F1','xyxy':[100,100,200,300]}]}],'coverage':'complete'}
    out=decode_observation(raw,p,[target],'discover_candidates')
    assert parse_card(out['records'][0],target,window,aliases,catalog)['membership']=='unknown'


def test_compact_inspection_uses_existing_id(tmp_path):
    def observe(p,m): return {'members':[member(p,judgment='unknown')],'coverage':'complete'}
    def inspect(p,m):
        r=member(p); r['object']=p['existing'][0]['candidate_id']
        return {'members':[r],'coverage':'complete'}
    model=Model(task(),observe,inspect); agent,req=setup(tmp_path,model)
    result=agent.solve(req)
    assert result.prediction=='A' and len(result.inventory['cards'])==1
    assert len(result.inventory['revisions'])==1


def test_pairwise_comparison_is_small_and_unknown_is_accepted(tmp_path):
    def observe(p,m): return {'members':[member(p,1),member(p,2),member(p,3)],'coverage':'complete'}
    def identity(p,m):
        assert len(p['left'])==len(p['right'])==1
        assert p['pair_task']['input_type']=='isolated_representatives'
        assert all(v['region']=='comparison' for v in p['catalog'].values())
        return {'decision':'UNKNOWN','basis':'uncertain','refs':[],'reason':'No reliable identity evidence.'}
    model=Model(task(),observe,identity=identity); agent,req=setup(tmp_path,model)
    result=agent.solve(req)
    assert result.prediction is None and not result.trace['failures']
    assert all(r['relation']=='unknown' for r in result.inventory['relations'])
    assert sum(c['role']=='identity' for c in model.calls)==3


@pytest.mark.parametrize('basis',['continuous_track','distinct_tracks'])
def test_isolated_views_cannot_claim_tracks(basis):
    p={'left':['C1'],'right':['C2'],'pair_task':{'available_bases':['uncertain','reidentification','stable_difference']}}
    with pytest.raises(ContractError): decode_pair({'decision':'DIFFERENT','basis':basis,'refs':['F1'],
                                                    'reason':'different scenes','features':['different scenes']},p)


def test_final_example_uses_actual_schema_and_preserves_unknown():
    t=SetSpec('items','physical_instance','tools')
    p={'sets':[],'catalog':{}}; bind_interaction(p,[t],'discover_candidates')
    text=interaction_prompt(p,[t],'discover_candidates')
    demo=json.loads(re.search(r'```json\n(.*?)\n```',text,re.S)[1])
    decoded=decode_observation(demo,p,[t],'discover_candidates')
    assert set(decoded['records'][0]['conditions'].values())=={'unknown'}


def test_new_template_fault_stops_before_observation(tmp_path,monkeypatch):
    from qwen3vl_agent.r4 import interaction
    def broken(*args,**kwargs): raise ValueError('invalid generated template')
    monkeypatch.setattr(interaction,'member_schema',broken)
    model=Model(task()); agent,req=setup(tmp_path,model)
    result=agent.solve(req)
    assert result.failure['stage']=='observe_prompt'
    assert len(model.calls)==1


def test_specialized_history_motion_and_simultaneous_contracts_remain():
    for t,p in [(SetSpec('s','task_item','plans'),{}),
                (SetSpec('s','physical_instance','tools',predicate_kind='moving'),{}),
                (SetSpec('s','physical_instance','tools'),{'simultaneous_sets':['s']})]:
        bind_interaction(p,[t],'discover_candidates')
        assert 'interaction' not in p


def test_real_failed_record_can_recover_in_compact_protocol(tmp_path):
    logs=json.loads((Path(__file__).parent/'fixtures/r4_v5_4_failures.json').read_text())
    raw=next(r['output'] for r in logs if r['source']=='R4-MME-225-1/00004.json')
    def observe(p,m):
        if len(m.calls)==2: return raw
        return {'members':[member(p,label='apple')],'coverage':'complete'}
    model=Model(task('semantic_category'),observe); agent,req=setup(tmp_path,model)
    result=agent.solve(req)
    assert result.prediction=='A' and len(model.calls)==3
    assert len(result.inventory['cards'])==1
    txn=result.trace['transactions']['tile_000000:discover_candidates']
    assert len(txn['wire_decodes'])==1


def test_actual_graph_and_unrequested_checks_are_not_new_answers():
    logs=json.loads((Path(__file__).parent/'fixtures/r4_v5_4_failures.json').read_text())
    from qwen3vl_agent.r4.collection_contracts import parse_json
    t=SetSpec('activities','semantic_category','daily activity',candidates=('writing',))
    p={'candidates':{'activities':['writing']},'sets':[],'catalog':{}}
    bind_interaction(p,[t],'discover_candidates')
    for log in logs:
        if '077-2' in log['source']:
            with pytest.raises(ContractError):
                decode_pair(parse_json(log['output']),{'left':['A'],'right':['B'],'pair_task':{'available_bases':['uncertain']}})
        if '251-1' in log['source']:
            with pytest.raises(ContractError): decode_observation(parse_json(log['output']),p,[t],'discover_candidates')
