"""v5.3 protocol/commit fixes; real saved responses and no-model checks only."""
import copy
import json
from pathlib import Path
from dataclasses import replace
import pytest
from test_r3_query_v51 import observation, review_record, fake_agent
from test_r3_query_v5 import spec, row, refs
from qwen3vl_agent.r3 import QuerySpec
from qwen3vl_agent.r3.review import review_prompt, REVIEW_EXAMPLES, parse_checks, commit_checks
from qwen3vl_agent.r3.lines import parse_lines
from qwen3vl_agent.r3.candidates import ingest, events_from_rows
from qwen3vl_agent.r3.query_reduce import reduce_query
from qwen3vl_agent.r3.evidence import structural_aspects


def real_case(fragment):
    d=next(d for d in json.loads((Path(__file__).parent/'fixtures/r3_v52_failures.json').read_text(encoding='utf-8')) if fragment in d['case'])
    q=QuerySpec(**d['query']);s={}
    aliases={f'T{i}':t for i,t in enumerate(dict.fromkeys([q.target,*q.targets,*([q.anchor] if q.anchor else [])]),1)}
    for c in d['calls']:
        if not c['id'].startswith('base:'):continue
        a,b=c['task']['core']
        core=[k for k,v in c['refs'].items() if a<=v['timestamp_seconds']<b or c['task'].get('last') and v['timestamp_seconds']==b]
        p=parse_lines(c['raw'],q.recipe,c['refs'],aliases,unit=q.unit,core_markers=core)
        ingest(s,c,p)
    return q,s,aliases,d['calls']


@pytest.mark.parametrize('recipe',['cycles','instances','anchors','time'])
def test_review_prompt_has_one_applicable_valid_example_and_no_end_requirement(recipe):
    prompt=review_prompt(recipe)
    assert 'Finish with exactly one line' not in prompt
    example=REVIEW_EXAMPLES[recipe]['output']
    assert '"decision":"end"' not in prompt
    assert 'C2 = making' not in prompt
    if recipe=='cycles':
        assert 'Tall vase' not in prompt and all(set(r)=={'at','decision','note'} for r in example['rows'])
    aliases={'T1':'task','T2':'anchor'}
    p=parse_lines('\n'.join(json.dumps(r) for r in example['rows']),recipe,refs(),aliases,
                  unit='production_instance' if recipe=='instances' else 'activity',require_end=False)
    assert not p['errors'] and not p['end']


def test_real_duplicate_c1_cannot_delete_floor_anchor():
    q,s,aliases,calls=real_case('251-3')
    call=next(c for c in calls if c['id'].startswith('review:5db69'))
    original=copy.deepcopy(s['candidates'])
    p=parse_checks(call['raw'],call['task'],call['refs'],q,aliases)
    assert not p['checks'] and any('duplicate' in e['error'] for e in p['errors'])
    commit_checks(s,call,p,q)
    assert s['candidates']==original
    assert any(r['active'] and r['target']==q.anchor for r in s['candidates'])


def test_duplicate_invalidates_only_that_check_and_malformed_tail_is_detected():
    s=observation();record,_=review_record(s)
    record['task']['checks'].append({**record['task']['checks'][0],'key':'independent'})
    good={'check':'C2','verdict':'confirmed','at':['F01','F08'],'note':'Supported independent check.'}
    bad={'check':'C1','verdict':'rejected','at':['F01','F08'],'note':'Not target.'}
    raw=json.dumps(bad)+'\n'+json.dumps(good)+'\n'+ '{"check":"C1","verdict":'
    p=parse_checks(raw,record['task'],record['refs'],spec(),{'T1':'task'})
    assert [r['check'] for r in p['checks']]==['C2']


@pytest.mark.parametrize('suffix,truncated',[('\n{"check":"C99","verdict":"confirmed"}',False),('\n{',False),('',True)])
def test_incomplete_response_never_commits_partial_destructive_replacement(suffix,truncated):
    s=observation();record,_=review_record(s)
    raw=json.dumps({'check':'C1','verdict':'rejected','at':['F01','F08'],'note':'No target.'})+suffix
    parsed=parse_checks(raw,record['task'],record['refs'],spec(),{'T1':'task'},truncated=truncated)
    assert not parsed['checks']
    commit_checks(s,record,parsed,spec())
    assert s['candidates'][0]['active']


def test_real_pig_step_completion_is_rejected_before_replacing_original():
    q,s,aliases,calls=real_case('225-3')
    call=next(c for c in calls if c['id'].startswith('review:b5c251'))
    before=copy.deepcopy(s['candidates'])
    p=parse_checks(call['raw'],call['task'],call['refs'],q,aliases)
    assert not p['checks'] and any('after completion' in e['error'] for e in p['errors'])
    commit_checks(s,call,p,q)
    assert s['candidates']==before and not s.get('confirmations')


def test_valid_two_production_instances_still_allowed():
    q=QuerySpec('nth_occurrence','task','production_instance',k=2,basis='offset',project='shape')
    s=observation(q=q,rows=[row('begin',('F01','F02'),target='T1'),row('complete',('F03','F04'),target='T1',value='Vase')])
    replacement=[row('begin',('F01','F02'),target='T1'),row('complete',('F03','F04'),target='T1',value='Vase'),
                 row('begin',('F05','F06'),target='T1'),row('complete',('F07','F08'),target='T1',value='Bowl')]
    record,p=review_record(s,q,verdict='corrected',replacement=replacement)
    assert not p['errors'];commit_checks(s,record,p,q)
    es,errors=events_from_rows(s['candidates'],q)
    assert len(es)==2 and not errors


def test_base_production_complete_then_continue_is_an_explicit_gap():
    q=QuerySpec('count_occurrences','task','production_instance')
    s=observation(q=q,rows=[row('begin',('F01','F02'),target='T1'),row('complete',('F03','F04'),target='T1'),
        row('continue',('F05','F06'),target='T1'),row('complete',('F07','F08'),target='T1')])
    r=reduce_query(q,s,(0,8))
    assert not r['closed'] and any(g['kind']=='event_unit' for g in r['gaps'])


def test_missing_boundary_and_attribute_not_confirmed_by_a_generic_yes():
    q=QuerySpec('first_occurrence','task','production_instance',basis='onset',project='shape')
    s=observation(q=q,rows=[row('continue',('F01','F02'),target='T1'),row('complete',('F03','F04'),target='T1')])
    record,p=review_record(s,q)
    assert not p['errors'];commit_checks(s,record,p,q)
    proof=s['confirmations'][0]
    assert 'onset' not in proof['aspects'] and 'attribute' not in proof['aspects']
    assert any('withheld' in e.get('reason','') for e in s['check_errors'])
    assert not reduce_query(q,s,(0,8))['closed']


def test_production_unit_conflict_cannot_receive_unit_confirmation():
    e={'id':'e','complete':True,'onset':None,'offset':[1,2],'value':None,'source_refs':['a','b']}
    proof=structural_aspects(e,['onset','offset','attribute','unit','identity','completion'],spec(),
        conflicts=[{'kind':'event_unit','events':['e']}])
    assert proof==['offset']


def test_v52_checkpoint_rejected_and_live_prompt_uses_current_recipe(tmp_path):
    agent,model,req=fake_agent(tmp_path)
    result=agent.solve(req)
    for messages,_ in model.calls:
        if 'R3:confirm_candidates' in messages[0]['content']:
            assert 'Finish with exactly one line' not in messages[0]['content']
            assert 'Tall vase' not in messages[0]['content']
    assert result.semantic_result['value']==2
    Path(req.checkpoint_path).write_text(json.dumps({'kind':'header','fingerprint':{'version':'r3-5.2'}})+'\n')
    with pytest.raises(ValueError,match='mismatch'): agent.solve(replace(req,resume=True))
