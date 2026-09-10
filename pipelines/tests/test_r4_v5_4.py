"""Targeted 5.4 contracts and real-output replay; no model accuracy claims."""
import copy
import json
import re
from pathlib import Path
from dataclasses import replace
import pytest
from r4_v5_fakes import Model, setup, task, card
from test_r4_v5 import refs
from test_r4_v5_2 import check
from test_r4_v5_3 import presence
from qwen3vl_agent.r4.types import SetSpec
from qwen3vl_agent.r4.inventory import parse_card
from qwen3vl_agent.r4.collection_contracts import (check_schema, validate, parse_json,
    diagnostics, record_schema, ContractError)
from qwen3vl_agent.r4.collection_prompts import build_prompt, example_cases, validate_example


@pytest.mark.parametrize('state',['not_seen','unreadable'])
def test_negative_or_unknown_does_not_require_positive_support(state):
    target=SetSpec('activities','semantic_category','activity',candidates=('writing',))
    row={'set':'activities','candidate':'writing','state':state,'refs':[],'facts':'input inspected'}
    assert validate(row,check_schema(target))==row
    with pytest.raises(ContractError) as caught:
        validate({**row,'support':'not_seen'},check_schema(target),'checks[0]')
    feedback=diagnostics(caught.value.errors)
    assert feedback[0]['actual_value']=='not_seen'
    assert 'For state:not_seen, OMIT support' in feedback[0]['instruction']
    assert 'retain the check' in feedback[0]['instruction']


def test_seen_still_requires_support_and_legal_evidence(tmp_path):
    target=SetSpec('items','semantic_category','activity',candidates=('writing',))
    with pytest.raises(ContractError):
        validate({'set':'items','candidate':'writing','state':'seen','refs':[],'facts':'test'},check_schema(target))
    def observe(p,m):
        row=check(p,'writing'); row['refs']=[]
        return {'checks':[row,check(p,'reading','not_seen')]}
    model,agent,req=presence(tmp_path,observe)
    result=agent.solve(req)
    assert result.prediction is None


def test_actual_check_recovery_messages_are_applicable(tmp_path):
    def observe(p,m):
        row=check(p,'reading','not_seen')
        if len(m.calls)==2: row['support']='not_seen'
        return {'checks':[check(p,'writing'),row]}
    model,agent,req=presence(tmp_path,observe)
    result=agent.solve(req)
    assert result.prediction=='B' and len(model.calls)==3
    first,recovery=model.calls[1:]
    assert first['messages'][0]['content'][:-1]==recovery['messages'][0]['content'][:-1]
    text=recovery['messages'][-1]['content'][-1]['text']
    assert 'FORMAT DEMO empty' not in text
    assert 'conditions unknown' not in text and 'RECORD REPAIR' not in text
    assert 'For not_seen omit support' in text
    assert len(result.inventory['checks'])==2
    for raw in re.findall(r'```json\n(.*?)\n```',text,re.S):
        example=json.loads(raw)
        assert example.get('checks')
    assert 'answer_label' not in text and 'human_review' not in text


def test_empty_recovery_cannot_erase_checks(tmp_path):
    def observe(p,m):
        if len(m.calls)==2:
            return {'checks':[{**check(p,c,'not_seen'),'support':'not_seen'} for c in p['candidates']['items']]}
        return {'checks':[]}
    model,agent,req=presence(tmp_path,observe)
    result=agent.solve(req)
    assert result.prediction is None and not result.inventory['checks']
    assert any(e['code']=='recovery_slot_missing' for e in result.failure['errors'])
    txn=result.trace['transactions']['tile_000000:discover_candidates']
    assert len(txn['call_ids'])==2


@pytest.mark.parametrize('namespace',['physical_instance','semantic_category','text_value','task_item'])
def test_recovery_examples_match_their_schema(namespace):
    target=SetSpec('items',namespace,'tools')
    payload={'sets':[],'catalog':{}}
    feedback={'errors':[],'invalid_slots':['tile/records/0']}
    text=build_prompt('discover_candidates',payload,[target],feedback=feedback)
    assert 'FORMAT DEMO empty' not in text
    embedded=[json.loads(r) for r in re.findall(r'```json\n(.*?)\n```',text,re.S)]
    cases=[(t,out) for name,t,out in example_cases('discover_candidates',[target]) if name!='empty']
    assert embedded==[out for t,out in cases]
    for t,out in cases: validate_example('discover_candidates',t,out)
    if namespace=='task_item': assert 'RECORD REPAIR' not in text
    else: assert 'RECORD REPAIR' in text


def test_record_judgment_recovery_retains_original_id(tmp_path):
    def observe(p,m):
        row=card(p)
        if len(m.calls)==2: row['conditions']['predicate']='visually present'
        return {'records':[row],'coverage':'complete'}
    model=Model(task(),observe); agent,req=setup(tmp_path,model)
    result=agent.solve(req)
    assert result.prediction=='A' and len(result.inventory['cards'])==1
    assert len(model.calls)==3
    text=model.calls[-1]['messages'][-1]['content'][-1]['text']
    assert 'RECORD REPAIR' in text and 'CHECK REPAIR' not in text
    assert 'visually present' in text and 'yes, no, or unknown' in text


def test_unit_label_stays_unresolved_until_targeted_review(tmp_path):
    def observe(p,m):
        return {'records':[card(p,i,name='fruit',cls='fruit',query_value=p['sets'][0]['count_unit']) for i in (1,2)],'coverage':'complete'}
    def inspect(p,m):
        assert p['existing'][0]['query_value'] is None
        assert p['existing'][0]['reported_query_value']=='semantic_category'
        text=m.calls[-1]['messages'][-1]['content'][-1]['text']
        assert 'TARGETED CATEGORY REVIEW' in text
        updates=[]
        for c in p['existing']:
            row=card(p,name='red apple',cls='apple',query_value='apple'); row.pop('id')
            row['candidate_id']=c['candidate_id']; updates.append(row)
        return {'updates':updates,'coverage':'complete'}
    model=Model(task('semantic_category'),observe,inspect); agent,req=setup(tmp_path,model)
    result=agent.solve(req)
    assert result.prediction=='A' and len(result.inventory['cards'])==2
    assert [c['role'] for c in model.calls]==['compile','discover_candidates','inspect_existing']


def test_ordinary_category_and_text_values_not_rewritten():
    catalog,aliases,window=refs()
    for ns,value in [('semantic_category','apple'),('text_value','printed words')]:
        target=SetSpec('items',ns,'printed words' if ns=='text_value' else 'fruit',count_unit='printed words' if ns=='text_value' else 'fruit type')
        row={'id':'O1','set':'items','name':value,'class':value,'conditions':{'target':'yes','predicate':'yes'},
             'facts':'visible','refs':['F1'],'query_value':value,'mapping_evidence':'specific fruit'}
        if ns=='text_value': row['raw_text']=value
        assert parse_card(row,target,window,aliases,catalog)['query_value']==value


def test_identity_feedback_and_unknown_do_not_invent_independence(tmp_path):
    def observe(p,m): return {'records':[card(p,1),card(p,2)],'coverage':'complete'}
    def identity(p,m):
        count=sum(c['role']=='identity' for c in m.calls)
        row={'left':p['left'][0],'right':p['right'][0],'relation':'UNKNOWN','basis':'uncertain','refs':[],
             'facts':'No independent coexistence or continuity established.'}
        if count==1: row.update(relation='DIFFERENT',basis='distinct_tracks',refs=list(p['catalog']),facts='different scenes')
        else:
            text=m.calls[-1]['messages'][-1]['content'][-1]['text']
            assert 'independent_objects:true' in text and 'IDENTITY CORRECTION' in text
            assert 'invalid slots' not in text
        return {'relations':[row]}
    model=Model(task(),observe,identity=identity); agent,req=setup(tmp_path,model)
    result=agent.solve(req)
    assert result.prediction is None
    assert all(r['relation']=='unknown' for r in result.inventory['relations'])
    assert sum(c['role']=='identity' for c in model.calls)==2


def test_real_v53_outputs_remain_rejected_with_precise_feedback():
    logs=json.loads((Path(__file__).parent/'fixtures/r4_v5_3_failures.json').read_text())
    assert len(logs)==7
    for log in logs:
        d=parse_json(log['raw_output'])
        if log['source'].startswith('R4-MME-251'):
            target=SetSpec('activities','semantic_category','activity',candidates=tuple(r['candidate'] for r in d['checks']))
            for row in d['checks']:
                with pytest.raises(ContractError) as caught: validate(row,check_schema(target),'checks[0]')
                assert 'OMIT support' in diagnostics(caught.value.errors)[0]['instruction']
        elif log['role'].endswith('discover_candidates'):
            ns='physical_instance' if '077-2' in log['source'] else 'semantic_category'
            target=SetSpec('items',ns,'object')
            for row in d['records']:
                with pytest.raises(ContractError) as caught: validate(row,record_schema(target),'records[0]')
                assert any(e['path'].endswith('conditions.predicate') for e in caught.value.errors)
        else:
            assert all(r['relation']=='DIFFERENT' and not r.get('independent_objects') for r in d['relations'])
