"""Evidence-transfer regressions from v5.3, no model or remote execution."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
from test_r3_query_v5 import refs,row,spec,lines,call,setup
from test_r3_query_v51 import observation,review_record,fake_agent
from qwen3vl_agent.r3 import QuerySpec,R3Config
from qwen3vl_agent.r3.lines import parse_lines
from qwen3vl_agent.r3.review import parse_checks,commit_checks,plan_checks,checks_for
from qwen3vl_agent.r3.candidates import ingest,events_from_rows
from qwen3vl_agent.r3.query_reduce import reduce_query
from qwen3vl_agent.r3.answer_evidence import build_packet,packet_payload
from qwen3vl_agent.r3.query_media import Access


def fixture(fragment):
    return next(x for x in json.loads((Path(__file__).parent/'fixtures/r3_v53_failures.json').read_text(encoding='utf-8')) if fragment in x['case'])


@pytest.mark.parametrize('empty',[[],None])
def test_empty_non_mutation_rows_is_normalized_but_nonempty_rows_is_rejected(empty):
    state=observation();c,_=review_record(state)
    obj={'check':'C1','verdict':'confirmed','at':['F01','F08'],'note':'visible motion','rows':empty}
    p=parse_checks(json.dumps(obj),c['task'],c['refs'],spec(),{'T1':'task'})
    assert len(p['checks'])==1 and not p['errors']
    obj['rows']=[row()]
    assert not parse_checks(json.dumps(obj),c['task'],c['refs'],spec(),{'T1':'task'})['checks']


def test_real_empty_rows_confirmations_no_longer_fail_on_empty_field():
    d=fixture('225');q=QuerySpec(**d['query']);n=0
    for c in d['calls']:
        if '"verdict":"confirmed"' not in c['raw'] or '"rows":[]' not in c['raw']:continue
        n+=1;p=parse_checks(c['raw'],c['task'],c['refs'],q,{'T1':q.target})
        assert not any('unexpected or missing check fields' in x['error'] for x in p['errors'])
        # A missing segment is still a substantive evidence gap, not fixed by [] normalization.
    assert n==6


def test_real_count_citations_preserved_without_promoting_candidate_truth():
    d=fixture('action-count');q=QuerySpec(**d['query']);c=d['calls'][0]
    p=parse_lines(c['raw'],q.recipe,c['refs'],{'T1':q.target},unit=q.unit)
    assert len(p['rows'])==4 and len(p['rows'][1]['at'])==9
    assert not p['complete'] and not p['negative']  # visibility remains uncertain
    s={};ingest(s,c,p)
    assert not reduce_query(q,s,(0,15.384))['closed']
    c=d['calls'][1];p=parse_lines(c['raw'],q.recipe,c['refs'],{'T1':q.target},unit=q.unit)
    assert any('places' in r['note'] and r['decision']=='occurrence' for r in p['rows'])
    # No string-based semantic correction of an actual visual claim.


def test_duplicate_citations_normalize_only_same_frame_unknown_still_fails():
    p=parse_lines(lines(row(at=('F02','f2','F03'))),'cycles',refs(),{'T1':'task'})
    assert p['rows'][0]['at']==['F02','F03']
    p=parse_lines(lines(row(at=('F02','F99'))),'cycles',refs(),{'T1':'task'})
    assert not p['rows'] and p['errors']


def test_instance_identity_is_separate_from_unseen_beginning():
    q=QuerySpec('nth_occurrence','task','production_instance',k=2,basis='offset')
    s=observation(q=q,rows=[row('begin',('F01','F02'),target='T1',instance='I1'),
          row('complete',('F03','F04'),target='T1',instance='I1',value='Vase'),
          row('continue',('F05','F06'),target='T1',instance='I2'),
          row('complete',('F07','F08'),target='T1',instance='I2',value='Bowl')])
    es,errors=events_from_rows(s['candidates'],q)
    assert len(es)==2 and not errors and es[1]['onset'] is None and es[1]['value']=='Bowl'
    assert not reduce_query(q,s,(0,8))['closed']  # identity names do not confer visual confirmation


def test_same_local_instance_name_in_another_call_does_not_merge():
    q=QuerySpec('count_occurrences','task','production_instance')
    s=observation(q=q,rows=[row('continue',('F01','F02'),target='T1',instance='I1')])
    c=call(recipe='instances');c['id']='next'
    p=parse_lines(lines(row('continue',('F05','F06'),target='T1',instance='I1')),'instances',refs(),{'T1':'task'},unit=q.unit)
    ingest(s,c,p)
    es,_=events_from_rows(s['candidates'],q)
    assert len(es)==2


@pytest.mark.parametrize('fragment',['225','251','action-count'])
def test_real_review_tasks_are_scoped_independent_and_within_frame_cap(fragment):
    d=fixture(fragment);q=QuerySpec(**d['query']);catalog={v['id']:v for c in d['calls'] for v in c['refs'].values()}
    state={'candidates':d['candidate_state']['rows']}
    checks=checks_for(q,d['reduction'],state)
    tasks=plan_checks(checks,q,SimpleNamespace(catalog=catalog),R3Config(),d['reduction']['required_scope'],30)
    assert tasks
    for t in tasks:
        c=t['checks'][0];a,b=t['span']
        assert c['parts']==1 and c['part']==0 and not t['depends_on'] and t['estimated_frames']<=64
        assert len(t['segments'])==1
        assert all(a<=x['bounds'][0]<=x['bounds'][1]<=b for x in c['descriptions'])
        assert set(c['rows']).isdisjoint(c['context_rows'])


def test_two_local_verdicts_commit_independently_and_returned_commits_are_idempotent():
    s=observation(rows=[row(at=('F02','F03')),row(at=('F06','F07'))])
    c,_=review_record(s);old=copy.deepcopy(c['task']['checks'][0]);ids=[x['id'] for x in s['candidates']]
    for index,verdict in enumerate(['rejected','confirmed']):
        check={**old,'key':f'local:{index}','parts':1,'part':0,'rows':[ids[index]],'events':[ids[index]],
               'versions':{ids[index]:1},'segments':[{'span':[0,8],'core':[0,8]}]}
        rec={**c,'id':f'local:{index}','task':{**c['task'],'checks':[check]}}
        p=parse_checks(json.dumps({'check':'C1','verdict':verdict,'at':['F01','F08'],'note':'Local observation.'}),rec['task'],rec['refs'],spec(),{'T1':'task'})
        commit_checks(s,rec,p,spec());commit_checks(s,rec,p,spec())
    assert not s['candidates'][0]['active'] and s['candidates'][1]['active']
    assert reduce_query(spec(),s,(0,8))['value']==1


def test_local_slice_cannot_confirm_event_using_unpictured_beginning():
    q=QuerySpec('first_occurrence','task','production_instance',basis='offset')
    s=observation(q=q,rows=[row('begin',('F01','F02'),target='T1'),row('complete',('F07','F08'),target='T1')])
    rec,_=review_record(s,q);c=rec['task']['checks'][0]
    c.update(local_only=True,segments=[{'span':[5,8],'core':[5,8]}])
    p=parse_checks(json.dumps({'check':'C1','verdict':'confirmed','at':['F07','F08'],'note':'Finished item.'}),rec['task'],rec['refs'],q,{'T1':'task'})
    commit_checks(s,rec,p,q)
    assert not s.get('confirmations') and not reduce_query(q,s,(0,8))['closed']


def test_real_final_packet_carries_egg_facts_and_their_actual_images():
    d=fixture('251');q=QuerySpec(**d['query']);catalog={v['id']:v for c in d['calls'] for v in c['refs'].values()}
    packet=build_packet(q,d['reduction'],{'candidates':d['candidate_state']['rows']},catalog,Access((0,81.57)))
    eggs=[f for f in packet['facts'] if f.get('value')=='Frying an egg.']
    assert eggs and all(f['source_refs'] for f in eggs)
    assert any(32<=catalog[x]['timestamp_seconds']<=38 for x in packet['frame_ids'])
    assert len(packet['frame_ids'])<=32
    aliases={f'F{i:02d}':catalog[x] for i,x in enumerate(packet['frame_ids'],1)}
    payload=packet_payload(packet,aliases)
    assert all(f['at'] and all(x in aliases for x in f['at']) for f in payload['facts'])
    assert 'choices' not in json.dumps(packet) and 'answer_label' not in json.dumps(packet)


def test_final_cap_reports_omissions_and_never_changes_program_result():
    d=fixture('225');q=QuerySpec(**d['query']);before=copy.deepcopy(d['reduction'])
    catalog={v['id']:v for c in d['calls'] for v in c['refs'].values()}
    p=build_packet(q,d['reduction'],{'candidates':d['candidate_state']['rows']},catalog,Access((0,60)),max_frames=2)
    assert p['omitted'] and len(p['frame_ids'])==2 and d['reduction']==before


def test_production_entry_uses_local_checks_and_persists_answer_packet(tmp_path):
    agent,model,req=fake_agent(tmp_path,'uncertain')
    result=agent.solve(req)
    assert result.semantic_result['value'] is None
    assert result.trace['answer_evidence_packet']['facts']
    for messages,_ in model.calls:
        data=json.loads(messages[1]['content'][0]['text'])
        if 'checks' in data:
            assert all(c['parts_required']==1 and c['local_only'] for c in data['checks'])
            assert 'choices' not in data
        if 'R3:final_visual' in messages[0]['content']:
            assert data['candidate_evidence']['facts']


def test_incomplete_correction_keeps_valid_facts_without_mutating_or_certifying_events():
    s=observation();old=copy.deepcopy(s['candidates'])
    rec,p=review_record(s,verdict='corrected',replacement=[row(at=('F02','F03')),row(at=('F99',))])
    assert not p['checks'] and len(p['retained_facts'])==1
    commit_checks(s,rec,p,spec());commit_checks(s,rec,p,spec())
    assert s['candidates']==old and len(s['uncommitted_facts'])==1 and not s.get('confirmations')
    packet=build_packet(spec(),reduce_query(spec(),s,(0,8)),s,{v['id']:v for v in rec['refs'].values()},Access((0,8)))
    assert any(f['status']=='uncommitted_fact_not_event_evidence' for f in packet['facts'])


def test_local_correction_facts_survive_missing_scan_endpoints_but_coverage_does_not():
    s=observation();rec,_=review_record(s)
    c=rec['task']['checks'][0];c.update(kind='protocol',aspects=[])
    raw=json.dumps({'check':'C1','verdict':'corrected','at':['F02','F03'],'note':'Visible local motion.',
                    'rows':[row(at=('F02','F03'))]})
    p=parse_checks(raw,rec['task'],rec['refs'],spec(),{'T1':'task'})
    n=len(s['coverage']);commit_checks(s,rec,p,spec())
    assert p['checks'] and len(s['coverage'])==n and not s.get('confirmations')
    assert any('scan boundary' in e.get('reason','') for e in s['check_errors'])
