"""Evidence-boundary and recovery tests tied to the v5 redesign and prior failures."""
import copy
import hashlib
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from r4_v5_fakes import Model, task, card, setup
from test_r4_v5 import refs, direct_card
from qwen3vl_agent.models.base import ModelOutput
from qwen3vl_agent.models.qwen3vl import VisualBudgetExceeded
from qwen3vl_agent.r4 import R4Config, R4Request, R4Budget
from qwen3vl_agent.r4.collection_contracts import ContractError, parse_json, check_schema, validate
from qwen3vl_agent.r4.collection_prompts import build_prompt, example_cases, validate_example
from qwen3vl_agent.r4.collection_reduce import map_answer, physical_worlds, View, simple_choice
from qwen3vl_agent.r4.inventory import EvidenceStore, parse_card
from qwen3vl_agent.r4.controller import CollectionController
from qwen3vl_agent.r4.session import CollectionSession, StageFailure
from qwen3vl_agent.r4.types import InventorySpec, SetSpec
from qwen3vl_agent.r4.evaluate import run_manifest


def update_rows(p, *, canonical=None, status="yes"):
    rows=[]
    for old in p["existing"]:
        row=card(p,name=old["name"],cls=old["class"],query_value=canonical,status=status,set_id=old["set"])
        row.pop("id")
        row["candidate_id"]=old["candidate_id"]
        rows.append(row)
    return {"updates":rows,"coverage":"complete"}


def test_real_errors_do_not_silently_become_members_or_final_answers():
    f=json.loads((Path(__file__).parent/'fixtures/r4_v5_previous_failures.json').read_text(encoding='utf-8'))
    assert len(f['source_files'])==5 and all(len(v['sha256'])==64 for v in f['source_files'])
    old=f['cases']['cup']['bottle_records'][0]
    assert 'bottle' in old['category']
    target=SetSpec('items','physical_instance','cups')
    catalog,aliases,tile=refs()
    _,row=direct_card(name=old['value'],cls=old['category'])
    row['facts']='The old record only asserts a visible physical object.'
    row['conditions']={'predicate':'yes'}  # No new visual qualification is invented in replay.
    with pytest.raises(ContractError,match='target'):
        parse_card(row,target,tile,aliases,catalog)
    row['conditions']['target']='unknown'
    assert parse_card(row,target,tile,aliases,catalog)['membership']=='unknown'
    assert 'bounding box dimensions' in f['cases']['cup']['bbox_width_relation'][0]['reason'].lower()
    assert {'triangle','dog','Naughty Dog','pig'} <= set(f['cases']['animal']['raw_categories'])
    assert f['cases']['animal']['shape_record']['predicate_status']=='satisfied'
    target=SetSpec('items','semantic_category','animal faces')
    row={'id':'O1','set':'items','name':'triangle','class':'triangle','conditions':{'target':'unknown','predicate':'yes'},
         'facts':'Old evidence identifies only a geometric shape.','refs':['F1']}
    parsed=parse_card(row,target,tile,aliases,catalog)
    assert parsed['membership']=='unknown' and parsed['query_value'] is None
    absent=parse_json(f['cases']['missing_observe']['raw_output'])
    negative=[r for r in absent['observations'] if not r['evidence_refs']]
    assert negative
    target=SetSpec('items','semantic_category','activities',candidates=('eating',))
    for r in negative:
        validate({'set':'items','candidate':'eating','state':'not_seen' if r['predicate_status']=='not_satisfied' else 'unreadable',
                  'refs':[],'facts':'The old response supplied no positive evidence.'},check_schema(target))
    req=R4Request('How many?',video_path='v',choices={'A':'4','B':'5'})
    assert map_answer({'final':{'value':6,'supported':True}},InventorySpec.from_dict(task()),req)[0] is None


@pytest.mark.parametrize('role',['discover_candidates','inspect_existing'])
def test_final_examples_cover_motion_candidates_and_inspection(role):
    import re
    targets=[InventorySpec.from_dict(task(predicate_kind='moving')).sets[0],
             InventorySpec.from_dict(task('semantic_category',candidates=['apple','pear'])).sets[0]]
    for target in targets:
        prompt=build_prompt(role,{'catalog':{},'sets':[]},[target])
        blocks=re.findall(r'```json\n(.*?)\n```',prompt,re.S)
        for block,(_,s,_) in zip(blocks,example_cases(role,[target])):
            validate_example(role,s,json.loads(block))
            if role=='inspect_existing': assert 'records' not in json.loads(block)


def test_category_mapping_conflict_is_reviewed_before_answer(tmp_path):
    def observe(p,m):
        return {'records':[card(p,1,name='red apple',cls='apple',query_value='red apple'),
                           card(p,2,name='green apple',cls='apple',query_value='green apple')],'coverage':'complete'}
    model=Model(task('semantic_category'),observe,inspect=lambda p,m:update_rows(p,canonical='apple'))
    agent,req=setup(tmp_path,model)
    r=agent.solve(req)
    assert r.prediction=='A',r.to_dict()
    assert [c['role'] for c in model.calls]==['compile','discover_candidates','inspect_existing']
    assert len(r.inventory['cards'])==2 and len(r.inventory['revisions'])==2
    assert not r.value_state['mapping_conflicts'] and not r.trace['category_ambiguities']


def test_membership_review_updates_same_id_and_clears_gap(tmp_path):
    model=Model(task(),lambda p,m:{'records':[card(p,status='unknown')],'coverage':'complete'},
                inspect=lambda p,m:update_rows(p))
    agent,req=setup(tmp_path,model)
    r=agent.solve(req)
    assert r.prediction=='A',r.to_dict()
    assert len(r.inventory['cards'])==1 and len(r.inventory['revisions'])==1
    assert not r.coverage_manifest[0]['membership_gaps']


def test_returned_inspection_is_replayed_without_new_call(tmp_path,monkeypatch):
    model=Model(task(),lambda p,m:{'records':[card(p,status='unknown')],'coverage':'complete'},inspect=lambda p,m:update_rows(p))
    agent,req=setup(tmp_path,model,request_kwargs={'checkpoint_path':str(tmp_path/'state.jsonl')})
    commit=EvidenceStore.commit_card
    hit=[False]
    def interrupt(self,*a,**kw):
        if kw.get('candidate_id') and not hit[0]:
            hit[0]=True;raise KeyboardInterrupt()
        return commit(self,*a,**kw)
    monkeypatch.setattr(EvidenceStore,'commit_card',interrupt)
    with pytest.raises(KeyboardInterrupt): agent.solve(req)
    before=len(model.calls)
    r=agent.solve(replace(req,resume=True))
    assert r.prediction=='A' and len(model.calls)==before
    assert len(r.inventory['cards'])==1


def test_global_recovery_cap_preserves_failures_and_continues_windows(tmp_path):
    def observe(p,m):
        count=sum(c['role']=='discover_candidates' for c in m.calls)
        return {'records':[card(p,query_value='apple')],'coverage':'complete'} if count>=6 else {'records':False,'coverage':'complete'}
    model=Model(task('semantic_category'),observe)
    agent,req=setup(tmp_path,model,duration=32)
    r=agent.solve(req)
    assert len(model.calls)==7,r.to_dict()  # compile + four base + two recoveries
    assert r.resources['calls_by_purpose']['recovery']==2
    assert r.result_status=='execution_failed' and r.prediction is None
    assert len(r.inventory['cards'])==1 and not r.trace['uninspected_ranges']
    assert len(r.trace['failures'])==3


def test_truncated_top_level_never_commits_even_closed_first_row(tmp_path):
    def observe(p,m):
        prefix=json.dumps({'records':[card(p)],'coverage':'complete'})
        return ModelOutput(prefix,{'output_tokens':1024,'finish_reason':'length'})
    model=Model(task(),observe)
    agent,req=setup(tmp_path,model)
    r=agent.solve(req)
    assert r.result_status=='execution_failed' and not r.inventory['cards']
    assert len(model.calls)==3 and r.failure['errors'][0]['code']=='response_truncated'
    with pytest.raises(ContractError):parse_json('{"records":[],"records":[]}')


@pytest.mark.parametrize('bad',[1.0,True])
def test_integer_bbox_does_not_coerce_float_or_bool(bad):
    s,row=direct_card();catalog,aliases,tile=refs()
    row['boxes'][0]['xyxy']=[bad,10,30,40]
    with pytest.raises(ContractError):parse_card(row,s,tile,aliases,catalog)


def test_crop_fragment_at_internal_edge_stays_unknown():
    s,row=direct_card();catalog,aliases,tile=refs()
    catalog['source-F1']['crop_transform']={'bbox_xyxy_1000':[400,0,1000,600]}
    tile['crop_core']=[500,0,1000,500]
    row['boxes'][0]['xyxy']=[0,100,600,600]
    r=parse_card(row,s,tile,aliases,catalog)
    assert r['membership']=='unknown' and 'crop_boundary_unresolved' in r['issues']


def test_local_crop_inventory_uses_one_frame_and_disjoint_regions(tmp_path):
    model=Model(task(scope={'kind':'frame','timestamp_sec':1}))
    agent,req=setup(tmp_path,model)
    controller=CollectionController(agent,req)
    controller.compile();controller.store=EvidenceStore(controller.spec);controller.plan()
    original=next(iter(controller.work['windows'].values()))
    assert original['core']==original['context']
    assert controller.split(original,'member_overflow',dense=True)
    source={'id':'Fsource','kind':'frame','entry_id':original['entry_id'],'start_sec':1,'end_sec':1,
            'source_frame_id':'Fsource','membership_sets':['items']}
    # Coordinates here are original-image positions, strictly inside their unique cores.
    for i,(key,bbox) in enumerate(zip(original['children'],[[100,100,200,200],[600,100,700,200],[100,600,200,700],[600,600,700,700]])):
        w=controller.work['windows'][key];w['status']='complete'
        p={'sets':[{'set_id':'items','namespace':'physical_instance','conditions':{'target':'x','predicate':'x'}}],'catalog':{'F1':{'region':'core'}}}
        c=parse_card(card(p,bbox=bbox),controller.spec.sets[0],w,{'F1':'Fsource'},{'Fsource':source})
        controller.store.commit_card(c,key,'crop')
    controller.refresh_gaps()
    state=controller.state()
    assert state['final']['bounds']==[4,4],state
    assert len(controller.store.graph()['different'])==6


def test_actual_visual_token_gate_repartitions_before_generation(tmp_path):
    class Gate(Model):
        def generate(self,messages,**kwargs):
            assert kwargs['visual_token_limit']==8192
            images=[p for p in messages[0]['content'] if p['type']=='image']
            if len(images)>10:
                raise VisualBudgetExceeded('processor measured visual tokens: 8500 exceeds 8192')
            return super().generate(messages,**kwargs)
    model=Gate(task('semantic_category'),lambda p,m:{'records':[card(p,query_value='apple')],'coverage':'complete'})
    agent,req=setup(tmp_path,model)
    r=agent.solve(req)
    assert r.prediction=='A',r.to_dict()
    blocked=[c for c in r.resources['calls'] if c['status']=='input_blocked']
    assert len(blocked)==1 and not blocked[0]['charged']
    assert r.resources['model_calls']==3 and r.resources['frame_exposures']==16
    leaves=[w for w in r.coverage_manifest if not w.get('children')]
    assert [w['core'] for w in leaves]==[[0,4],[4,8]]
    assert all(w['fps']==2 for w in leaves)
    assert all(c['generation_kwargs']['max_new_tokens']==768 for c in r.resources['calls'] if c['pool']=='focused')


def test_presence_positive_stops_search_but_single_choice_does_not_prove_absence(tmp_path):
    def seen(p,m):
        return {'checks':[{'set':'items','candidate':n,'state':'seen','refs':[next(iter(p['catalog']))],'facts':'visible fruit'}
                          for n in p['candidates']['items']],'coverage':'complete'}
    model=Model(task('semantic_category','membership',candidates=['apple']),seen)
    agent,req=setup(tmp_path,model,duration=32)
    r=agent.solve(replace(req,choices={'A':'yes','B':'no'}))
    assert r.prediction=='A' and len(model.calls)==2
    assert not r.trace['sampling_schedule_completed'] and len(r.trace['uninspected_ranges'])==3
    def partial(p,m):
        return {'checks':[{'set':'items','candidate':n,'state':'seen' if n=='apple' else 'unreadable',
                           'refs':[next(iter(p['catalog']))] if n=='apple' else [],'facts':'limited evidence'}
                          for n in p['candidates']['items']],'coverage':'partial','gaps':['unreadable']}
    for explicit in [False,True]:
        m=Model(task('semantic_category','missing_members',candidates=['apple','pear']),partial)
        a,q=setup(tmp_path/str(explicit),m,duration=8)
        q=replace(q,choices={'A':'apple','B':'pear'},benchmark_policy={'unique_missing':True} if explicit else {})
        result=a.solve(q)
        assert (result.prediction=='B') is explicit


def test_budget_pools_and_resume_root_are_not_reissued(tmp_path):
    model=Model(task());agent,req=setup(tmp_path,model)
    work={};session=CollectionSession(model,agent.media,agent.config,req,work,lambda:None)
    session.configure(1,True,8)
    payload={'sets':[],'catalog':{}}
    session.call('a','discover_candidates',payload,pool='recovery',recovery=True,root_id='root')
    again=CollectionSession(model,agent.media,agent.config,req,work,lambda:None)
    with pytest.raises(StageFailure,match='already spent'):
        again.call('b','discover_candidates',payload,pool='recovery',recovery=True,root_id='root')
    again.call('c','discover_candidates',payload,pool='recovery',recovery=True,root_id='other')
    with pytest.raises(StageFailure,match='two format'):
        again.call('d','discover_candidates',payload,pool='recovery',recovery=True,root_id='third')
    assert len(model.calls)==2
    assert not again.can_call('recovery')
    assert all(c['kwargs']['max_new_tokens']==1024 for c in model.calls)
    assert sum(session.state['limits'][k] for k in ['compile','base','focused','qualification','identity','recovery'])==14
    assert session.state['limits']['total']==15  # the unused forced-choice slot is not spendable
    assert R4Config().compiler_tokens==512 and R4Config().review_tokens==768


def test_independent_entry_skips_terminal_before_load_and_keeps_full_denominator(tmp_path):
    compiled=task(scope={'kind':'frame','timestamp_sec':0})
    model=Model(compiled,lambda p,m:{'records':[card(p)],'coverage':'complete'})
    agent,req=setup(tmp_path/'source',model)
    calls=[]
    original=agent.solve
    def solve(request):
        calls.append(request.request_id)
        if request.request_id=='broken': raise ValueError('question-local fault')
        return original(request)
    agent.solve=solve
    rows=[(replace(req,request_id=k),{'answer_label':'A'}) for k in ['broken','valid']]
    output=tmp_path/'run'
    report=run_manifest(agent,rows,output)
    assert report['failed']==1 and report['correct']==1 and report['accuracy']==.5
    assert report['terminal_count']==2 and report['answered']==1 and model.loads==1
    count=len(model.calls)
    run_manifest(agent,rows,output,resume=True)
    assert model.loads==1 and len(model.calls)==count
    assert len((output/'native_predictions.jsonl').read_text().splitlines())==1
    assert len(json.loads((output/'unanswered.json').read_text()))==1
    with pytest.raises(ValueError,match='identical'):
        agent.config=replace(agent.config,review_tokens=700)
        run_manifest(agent,rows,output,resume=True)


def test_load_failure_is_batch_fatal(tmp_path):
    model=Model(task());agent,req=setup(tmp_path/'source',model)
    def fail():raise RuntimeError('load failed')
    agent.load=fail
    with pytest.raises(RuntimeError,match='load failed'):
        run_manifest(agent,[(req,{})],tmp_path/'run')
    assert not model.calls


def test_boolean_mapping_and_literal_case_cannot_match_numeric_or_other_word():
    spec=InventorySpec.from_dict(task())
    req=R4Request('count?',video_path='v',choices={'A':'yes','B':'2'})
    assert map_answer({'final':{'supported':True,'value':1}},spec,req)[0] is None
    spec=InventorySpec.from_dict(task('text_value',op='list_members'))
    req=replace(req,choices={'A':'Open','B':'OPEN'})
    assert map_answer({'final':{'supported':True,'value':['OPEN']}},spec,req)[0]=='B'
    graph={'different':set()}
    assert physical_worlds({'x':View('physical_instance','entity',set('123456789'),set('123456789'),True)},graph) is None
    assert simple_choice('one or two')=='one or two'
    assert simple_choice('2 red objects and 3 blue objects')=='2 red objects and 3 blue objects'


def test_bad_relation_slot_does_not_discard_valid_objects(tmp_path):
    def observe(p,m):
        n=sum(c['role']=='discover_candidates' for c in m.calls)
        return {'records':[card(p,1),card(p,2)],'coexisting':[{'ids':['O1','O2'],
                 'ref':'F99' if n==1 else next(iter(p['catalog'])),'independent_objects':True}],'coverage':'complete'}
    model=Model(task(scope={'kind':'frame','timestamp_sec':0}),observe)
    agent,req=setup(tmp_path,model)
    r=agent.solve(req)
    assert r.prediction=='B',r.to_dict()
    assert len(r.inventory['cards'])==2 and not r.inventory['quarantined']
    assert len(r.inventory['revisions'])==0 and len(model.calls)==3
    first=r.resources['calls'][1]
    assert len(first['committed_slots'])==2


def test_scope_visual_gate_preserves_first_search_prefix(tmp_path):
    class Gate(Model):
        def generate(self,messages,**kwargs):
            if len([p for p in messages[0]['content'] if p['type']=='image'])>10:
                raise VisualBudgetExceeded('processor measured 8500 > 8192')
            return super().generate(messages,**kwargs)
    def scope(p,m):
        ref=next(k for k,v in p['catalog'].items() if v['region']=='core')
        t=p['catalog'][ref]['start_sec']
        return {'bindings':[{'scope_id':'global','refs':[ref],'interval':[t,t+.04],'facts':'first matching frame in this checked prefix'}],'coverage':'complete'}
    model=Gate(task(scope={'kind':'semantic','description':'first visible tool','selection':'first','result_kind':'frame'}),
               lambda p,m:{'records':[card(p)],'coverage':'complete'},scope=scope)
    agent,req=setup(tmp_path,model)
    r=agent.solve(req)
    assert r.prediction=='A',r.to_dict()
    assert r.resources['calls_by_purpose']['focused']==1
    assert r.trace['bindings']['global'][0]['interval'][0]==0


def test_qualification_visual_gate_rebatches_stable_candidates(tmp_path):
    class Gate(Model):
        def generate(self,messages,**kwargs):
            role='R4:inspect_existing' in messages[0]['content'][-1]['text']
            if role and len([p for p in messages[0]['content'] if p['type']=='image'])>4:
                raise VisualBudgetExceeded('processor measured 8500 > 8192')
            return super().generate(messages,**kwargs)
    def observe(p,m):
        rows=[card(p,i+1,status='unknown') for i in range(6)]
        return {'records':rows,'coexisting':[{'ids':[r['id'] for r in rows],'ref':rows[0]['boxes'][0]['ref'],'independent_objects':True}],'coverage':'complete'}
    def inspect(p,m):
        data=update_rows(p)
        for row in data['updates']:
            ref=next(k for k,v in p['catalog'].items() if v.get('candidate_id')==row['candidate_id'])
            row['boxes']=[{'ref':ref,'xyxy':[0,0,1000,1000]}]
        return data
    model=Gate(task(),observe,inspect=inspect)
    agent,req=setup(tmp_path,model)
    r=agent.solve(replace(req,choices={'A':'6','B':'7'}))
    assert r.prediction=='A',r.to_dict()
    assert len(r.inventory['cards'])==6 and r.resources['calls_by_purpose']['qualification']==2
    assert any(w['status']=='repacked' for w in r.trace['review_windows'].values())


def test_history_never_accepts_generic_object_as_task_update(tmp_path):
    model=Model(task('task_item'))
    agent,req=setup(tmp_path,model)
    controller=CollectionController(agent,req)
    controller.spec=InventorySpec.from_dict(task('task_item'))
    controller.store=EvidenceStore(controller.spec)
    catalog,aliases,tile=refs()
    payload={'sets':[{'set_id':'items','namespace':'task_item','conditions':{'target':'x','predicate':'x'}}],
             'catalog':{'F1':{'region':'core'}}}
    controller.work['transactions']['t']={'payload':payload,'accepted_slots':[],'invalid_slots':[]}
    errors,committed=controller.accept_observation({'records':[card(payload)],'coverage':'complete'},
        'discover_candidates',controller.spec.sets,tile,aliases,catalog,'t','call')
    assert errors[0]['code']=='task_update_required' and not committed
    assert not controller.store.cards and not controller.store.history.state['task_updates']


def test_missing_allowed_modality_does_not_send_video_or_prove_absence(tmp_path):
    model=Model(task('semantic_category','missing_members',candidates=['apple'],required_modalities=['asr'],evidence_relation='mentioned'))
    agent,req=setup(tmp_path,model)
    r=agent.solve(req)
    assert r.result_status=='unresolved' and r.prediction is None
    assert len(model.calls)==1 and not r.coverage_manifest
    assert any('required_modality_unavailable' in x for x in r.unresolved_items)


def test_all_identity_interpretations_same_typed_option_stop_before_comparing(tmp_path):
    compiled=task()
    compiled['choice_values']={'A':{'kind':'one_of','values':[1,2]},'B':3}
    def observe(p,m):
        frames=list(p['catalog'])
        return {'records':[card(p,1,ref=frames[0]),card(p,2,ref=frames[1])],'coverage':'complete'}
    model=Model(compiled,observe)
    agent,req=setup(tmp_path,model)
    r=agent.solve(replace(req,choices={'A':'one or two','B':'three'}))
    assert r.prediction=='A',r.to_dict()
    assert r.value_state['final']['possible_values']==[1,2] and len(model.calls)==2
    assert r.answer_basis=='deterministic_all_interpretations'


def test_unbounded_public_option_can_use_lower_bound_without_inventing_upper(tmp_path):
    compiled=task()
    compiled['choice_values']={'A':{'kind':'interval','low':3,'high':None}}
    def observe(p,m):
        rows=[card(p,i+1) for i in range(3)]
        return {'records':rows,'coexisting':[{'ids':[r['id'] for r in rows],'ref':rows[0]['boxes'][0]['ref'],'independent_objects':True}],'coverage':'complete'}
    model=Model(compiled,observe)
    agent,req=setup(tmp_path,model,duration=32)
    r=agent.solve(replace(req,choices={'A':'at least three','B':'one','C':'two'}))
    assert r.prediction=='A',r.to_dict()
    assert r.value_state['final']['bounds']==[3,None] and len(model.calls)==2
    assert not r.trace['sampling_schedule_completed']


def test_equal_names_and_coordinates_do_not_automatically_create_same():
    spec=InventorySpec.from_dict(task());store=EvidenceStore(spec)
    s,row=direct_card();catalog,aliases,tile=refs()
    parsed=parse_card(row,s,tile,aliases,catalog)
    a=store.commit_card(copy.deepcopy(parsed),'first','call')
    b=store.commit_card(copy.deepcopy(parsed),'second','call')
    assert store.graph()['roots'][a]!=store.graph()['roots'][b] and not store.state['relations']
    with pytest.raises(ContractError,match='Coordinate/crop'):
        store.accept_relation({'left':a,'right':b,'relation':'DIFFERENT','basis':'stable_difference','refs':['F1'],
            'facts':'the bounding box width changed','features':['bounding box width: 730 vs 704'],'independent_objects':True},
            aliases,catalog,'identity',{a,b})


@pytest.mark.parametrize('cap,value,expected_calls',[
    ('max_model_calls',1,1),('max_frame_exposures',1,1),('max_generated_tokens',128,0)])
def test_total_exposure_and_generation_limits_stop_before_extra_model_call(tmp_path,cap,value,expected_calls):
    model=Model(task())
    agent,req=setup(tmp_path,model)
    r=agent.solve(replace(req,budget=R4Budget(**{cap:value})))
    assert r.result_status=='budget_exhausted' and r.prediction is None,r.to_dict()
    assert len(model.calls)==expected_calls and r.resources['model_calls']==expected_calls


def test_real_yaml_shared_model_and_r4_image_budget_remain_separate(tmp_path):
    from qwen3vl_agent.debug12 import configurations, PROJECT
    configs=configurations(PROJECT/'configs/r1345_dual4090d.yaml',tmp_path)
    assert all(c['model']==configs['R1']['model'] for c in configs.values())
    r4=configs['R4']['r4']
    assert r4['media']['normal_max_pixels']==393216 and r4['media']['detail_max_pixels']==1048576
    assert r4['media']['normal_total_pixels']==8388608 and r4['observer_tokens']==1024


def test_simultaneous_final_prompt_example_matches_census_contract():
    import re
    s=InventorySpec.from_dict(task(op='MAX_SIMULTANEOUS_COUNT')).sets[0]
    prompt=build_prompt('discover_candidates',{'simultaneous_sets':['items'],'catalog':{}},[s])
    demo=json.loads(re.findall(r'```json\n(.*?)\n```',prompt,re.S)[-1])
    validate_example('discover_candidates',SetSpec('format_tools','physical_instance','wrench'),demo)
    assert demo['snapshots'][0]['frames']['F2']=={}
