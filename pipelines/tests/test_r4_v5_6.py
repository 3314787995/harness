"""Evidence-obligation regressions. All model responses are simulated, no GPU."""
import copy
import json
import re
from dataclasses import replace
from pathlib import Path
import pytest
from r4_v5_fakes import Model, setup, task
from test_r4_v5_5 import member, answers
from test_r4_v5_3 import presence
from test_r4_v5 import refs
from qwen3vl_agent.r4.controller import CollectionController
from qwen3vl_agent.r4.inventory import EvidenceStore,parse_card
from qwen3vl_agent.r4.interaction import (interaction_prompt,bind_interaction,decode_observation,
                                       pair_prompt,pair_schema)
from qwen3vl_agent.r4.collection_contracts import validate,parse_json,ContractError
from qwen3vl_agent.r4.types import SetSpec


@pytest.mark.parametrize('value',['animal face','Animal FACE.','animal face kind'])
def test_generic_target_and_unit_are_not_finished_categories(value):
    cat,aliases,window=refs()
    target=SetSpec('items','semantic_category','animal face',count_unit='animal face kind',equivalence='category')
    row={'id':'O1','set':'items','name':value,'class':value,'facts':'visible shape',
         'conditions':{'target':'yes','predicate':'yes'},'refs':['F1'],
         'query_value':value,'mapping_evidence':'model label'}
    card=parse_card(row,target,window,aliases,cat)
    assert card['query_value'] is None
    assert any(i.startswith('query_value_is_') for i in card['issues'])


def test_real_generic_label_enters_review_even_with_open_bounds(tmp_path):
    compiled=task('semantic_category'); compiled['sets'][0].update(target='animal face',count_unit='animal face kind')
    raw=parse_json(json.loads((Path(__file__).parent/'fixtures/r4_v5_5_convergence_failures.json').read_text())[0]['output'])
    def observe(p,m):
        r=copy.deepcopy(raw['members'][0]);r['refs']=[next(k for k,v in p['catalog'].items() if v['region']=='core')]
        return {'members':[r],'coverage':'partial','gaps':['unjudgeable region']}
    def inspect(p,m):
        if p.get('coverage_review'): return {'coverage':'complete','gaps':[]}
        r=member(p,label='fox');r['object']=p['existing'][0]['candidate_id']
        assert p['review_goal'] and p['existing'][0]['query_value'] is None
        return {'members':[r],'coverage':'complete'}
    model=Model(compiled,observe,inspect);agent,req=setup(tmp_path,model)
    result=agent.solve(req)
    assert result.prediction=='A',result.to_dict()
    assert result.resources['calls_by_purpose']['qualification']==1
    assert result.resources['calls_by_purpose']['focused']==1
    assert len(result.inventory['cards'])==1
    assert next(iter(result.inventory['cards'].values()))['query_value']=='fox'
    assert [a['kind'] for a in result.trace['resolution_actions']]==['member_identification','coverage_audit']
    assert all(a['changed'] for a in result.trace['resolution_actions'])


def test_context_review_uses_original_core_not_only_old_context(tmp_path):
    def observe(p,m):
        r=member(p,label='pear')
        context=next((f for f,x in p['catalog'].items() if x['region']=='context'),None)
        if context:r['refs']=[context]
        return {'members':[r],'coverage':'complete'}
    def inspect(p,m):
        assert p['review_goal']
        assert sum(x['region']=='core' for x in p['catalog'].values())>1
        r=member(p,label='pear');r['object']=p['existing'][0]['candidate_id']
        return {'members':[r],'coverage':'complete'}
    model=Model(task('semantic_category'),observe,inspect);agent,req=setup(tmp_path,model,duration=16)
    result=agent.solve(req)
    assert result.prediction=='A',result.to_dict()
    assert result.resources['calls_by_purpose']['qualification']==1
    assert len(result.inventory['cards'])==2


def test_unlocalized_gap_is_audited_once_then_unresolved_not_false_budget(tmp_path):
    def observe(p,m):return {'members':[],'coverage':'partial','gaps':['unjudgeable region']}
    def inspect(p,m):
        assert p['coverage_review']
        return {'coverage':'partial','gaps':['Still cannot locate the unreadable region.']}
    model=Model(task('semantic_category'),observe,inspect);agent,req=setup(tmp_path,model)
    result=agent.solve(req)
    assert result.result_status=='unresolved' and result.prediction is None
    assert result.resources['model_calls']==3 and result.resources['calls_by_purpose']['focused']==1
    assert result.trace['stop_detail']['code']=='no_actionable_evidence_progress'
    assert not result.trace['stop_detail']['blocked_actions']
    assert not result.inventory['cards']


def test_localized_gap_gets_refinement_and_clears_without_fake_zero(tmp_path):
    def observe(p,m):
        ref=next(k for k,v in p['catalog'].items() if v['region']=='core')
        if len(m.calls)==2:
            return {'members':[],'coverage':'partial','regions':[{'set':'items','refs':[ref],
                        'reason':'uninspected','detail':'The lower object needs a closer temporal view.'}]}
        return {'members':[member(p,label='pear')],'coverage':'complete'}
    model=Model(task('semantic_category'),observe);agent,req=setup(tmp_path,model)
    result=agent.solve(req)
    assert result.prediction=='A',result.to_dict()
    assert result.resources['calls_by_purpose']['focused']==2
    assert result.trace['resolution_actions'][0]['kind']=='localized_refinement'


def test_audit_cannot_commit_members_or_erase_earlier_cards(tmp_path):
    def observe(p,m):return {'members':[member(p,label='pear')],'coverage':'partial','gaps':['unclear edge']}
    def inspect(p,m):return {'coverage':'complete','members':[]}
    model=Model(task('semantic_category'),observe,inspect);agent,req=setup(tmp_path,model)
    result=agent.solve(req)
    assert result.result_status=='execution_failed' and result.prediction is None
    assert len(result.inventory['cards'])==1
    assert result.coverage_manifest[0]['status']=='partial'


@pytest.mark.parametrize('ref',['F999','context'])
def test_gap_refs_require_displayed_core_and_set(ref):
    t=SetSpec('items','physical_instance','tools')
    p={'sets':[],'catalog':{'F1':{'region':'context','sets':['items']}},'candidates':{}}
    bind_interaction(p,[t],'discover_candidates')
    r={'members':[],'coverage':'partial','regions':[{'set':'items','refs':['F1' if ref=='context' else ref],
         'reason':'blurred','detail':'The image is blurred.'}]}
    with pytest.raises(ContractError):decode_observation(r,p,[t],'discover_candidates')


@pytest.mark.parametrize('namespace',['physical_instance','semantic_category','text_value'])
def test_final_member_examples_pass_real_decoder_and_are_branch_specific(namespace):
    t=SetSpec('items',namespace,'tools' if namespace=='physical_instance' else 'fruit categories' if namespace=='semantic_category' else 'labels')
    p={'sets':[],'catalog':{}};bind_interaction(p,[t],'discover_candidates')
    prompt=interaction_prompt(p,[t],'discover_candidates')
    assert 'Q tasks' not in prompt and 'unjudgeable region' not in prompt
    examples=[json.loads(x) for x in re.findall(r'```json\n(.*?)\n```',prompt,re.S)]
    assert len(examples)>=2
    for e in examples:decode_observation(e,p,[t],'discover_candidates')
    assert examples[0]['coverage']=='complete'
    assert set(examples[-1]['members'][0]['judgments'].values())=={'unknown'}


def test_q_examples_cover_observed_absent_uncertain_without_member_instructions(tmp_path):
    model,agent,req=presence(tmp_path,lambda p,m:{'answers':answers(p)})
    result=agent.solve(req);assert result.prediction=='B'
    prompt=model.calls[1]['messages'][0]['content'][-1]['text']
    assert 'xyxy' not in prompt and 'Member tasks' not in prompt
    examples=[json.loads(x) for x in re.findall(r'```json\n(.*?)\n```',prompt,re.S)]
    assert {e['answers'][0]['status'] for e in examples}=={'observed','absent','uncertain'}
    assert examples[0]['answers'][0]['refs']==['F1'] and examples[0]['answers'][0]['note']


def test_pair_examples_no_longer_only_generic_unknown():
    p={k:{} for k in ('pair_task','existing','catalog','evidence_by_object')};p.update(left=['C1'],right=['C2'])
    prompt=pair_prompt(p)
    examples=[json.loads(x) for x in re.findall(r'```json\n(.*?)\n```',prompt,re.S)]
    for e in examples:validate(e,pair_schema())
    assert {e['decision'] for e in examples}=={'SAME','DIFFERENT','UNKNOWN'}
    assert 'The supplied views do not establish object identity.' not in prompt


def test_budget_stop_names_unperformed_action_and_retains_headroom(tmp_path):
    def observe(p,m):return {'members':[],'coverage':'partial','gaps':['unknown search region']}
    model=Model(task('semantic_category'),observe);agent,req=setup(tmp_path,model,duration=48)
    agent.config=replace(agent.config,max_focused_calls=0)
    result=agent.solve(req)
    assert result.result_status=='budget_exhausted'
    detail=result.trace['stop_detail']
    assert detail['blocked_actions'][0]['action']=='coverage_audit'
    assert detail['total_calls_used']<detail['total_calls_limit']


def test_real_activity_bad_refs_remain_rejected():
    fixture=json.loads((Path(__file__).parent/'fixtures/r4_v5_5_convergence_failures.json').read_text())
    raw=parse_json(fixture[2]['output']);t=SetSpec('activities','semantic_category','activities')
    p={'sets':[],'catalog':{},'candidates':{'activities':['a','b','c','d']}}
    bind_interaction(p,[t],'discover_candidates')
    with pytest.raises(ContractError) as err:decode_observation(raw,p,[t],'discover_candidates')
    assert any('refs' in e['path'] for e in err.value.errors)


def test_terminal_resume_does_not_reaudit(tmp_path):
    def observe(p,m):return {'members':[],'coverage':'partial','gaps':['unclear']}
    def inspect(p,m):return {'coverage':'partial','gaps':['still unclear']}
    model=Model(task('semantic_category'),observe,inspect)
    agent,req=setup(tmp_path,model,request_kwargs={'checkpoint_path':str(tmp_path/'state.json')})
    first=agent.solve(req);n=len(model.calls)
    second=agent.solve(replace(req,resume=True))
    assert second.result_status==first.result_status and len(model.calls)==n


def test_category_batch_uses_each_home_window_witness(tmp_path):
    compiled=task('semantic_category');compiled['sets'][0].update(target='fruit',count_unit='fruit kind')
    def observe(p,m):return {'members':[member(p,label='fruit')],'coverage':'complete'}
    def inspect(p,m):
        assert len(p['existing'])==3
        rows=[]
        for c in p['existing']:
            ref=next(k for k,v in p['catalog'].items() if c['candidate_id'] in v.get('core_for',[]))
            r=member(p,label='pear');r['object']=c['candidate_id'];r['refs']=[ref];rows.append(r)
        assert len({r['refs'][0] for r in rows})==3
        return {'members':rows,'coverage':'complete'}
    model=Model(compiled,observe,inspect);agent,req=setup(tmp_path,model,duration=24)
    result=agent.solve(req)
    assert result.prediction=='A',result.to_dict()
    assert result.resources['calls_by_purpose']['qualification']==1
    assert len(result.inventory['cards'])==3
    assert all(c['membership']=='accepted' for c in result.inventory['cards'].values())


def test_interrupted_coverage_audit_resumes_same_window_and_media(tmp_path):
    def observe(p,m):return {'members':[],'coverage':'partial','gaps':['unspecified edge']}
    def inspect(p,m):
        if len(m.calls)==3:raise KeyboardInterrupt()
        return {'coverage':'complete','gaps':[]}
    model=Model(task('semantic_category'),observe,inspect)
    agent,req=setup(tmp_path,model,request_kwargs={'checkpoint_path':str(tmp_path/'audit.json')})
    with pytest.raises(KeyboardInterrupt):agent.solve(req)
    result=agent.solve(replace(req,resume=True))
    assert len(model.calls)==4
    assert model.calls[2]['messages'][0]['content'][:-1]==model.calls[3]['messages'][0]['content'][:-1]
    assert result.resources['calls_by_purpose']['focused']==1
    assert result.resources['calls_by_purpose']['recovery']==1
    assert result.coverage_manifest[0]['status']=='complete'
    assert len(result.trace['review_windows'])==1


def test_all_pairs_attempted_unknown_is_unresolved_not_budget_label(tmp_path):
    def observe(p,m):return {'members':[member(p,1),member(p,2)],'coverage':'complete'}
    def identity(p,m):return {'decision':'UNKNOWN','basis':'uncertain','refs':[],
                             'reason':'The distinguishing marking is occluded.'}
    model=Model(task(),observe,identity=identity);agent,req=setup(tmp_path,model)
    agent.config=replace(agent.config,max_identity_calls=1)
    result=agent.solve(req)
    assert result.result_status=='unresolved' and result.prediction is None
    assert result.trace['stop_detail']['code']=='no_actionable_evidence_progress'
    assert not result.trace['resolution_actions'][-1]['changed']
