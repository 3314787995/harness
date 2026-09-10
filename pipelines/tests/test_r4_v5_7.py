"""Local contracts and simulated execution only; no accuracy claims."""
import copy
import json
import re
from dataclasses import replace
import pytest

from r4_v5_fakes import Model, setup, task
from test_r4_v5_5 import member
from qwen3vl_agent.r4.interaction import bind_interaction, interaction_prompt, decode_observation
from qwen3vl_agent.r4.collection_contracts import ContractError
from qwen3vl_agent.r4.types import SetSpec
from qwen3vl_agent.r4.convergence import generic_category


class AnswerModel(Model):
    def __init__(self, *args, answer=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.handlers['best_effort'] = answer or (lambda p,m: {
            'prediction':'B','reason':'The supplied views favor two objects, with uncertain identity.',
            'refs':[next(iter(p['catalog']))], 'uncertainties':['Identity remains uncertain.']})


def category(p, i=1, value='pear', label=None):
    return {**member(p,i,label=label or value), 'query_value':value,
            'category_basis':'specific_kind', 'mapping_reason':'The visible individual caption and shape identify this kind.'}


@pytest.mark.parametrize('label',['animal face','origami animal face','paper animal face'])
def test_umbrella_suspicion_is_not_exact_string_only(label):
    assert generic_category(label, SetSpec('items','semantic_category','animal face',count_unit='animal face kind'))


def test_category_raw_name_is_separate_and_missing_mapping_not_invented():
    t=SetSpec('items','semantic_category','fruit',count_unit='fruit kind')
    p={'question':'How many kinds of fruit are displayed?', 'sets':[{'set_id':'items','namespace':'semantic_category'}],
       'catalog':{'F1':{'region':'core'}}}
    bind_interaction(p,[t],'discover_candidates')
    row=category(p,value='pear',label='Sweet Pear')
    out=decode_observation({'members':[row],'coverage':'complete'},p,[t],'discover_candidates')['records'][0]
    assert out['name']=='Sweet Pear' and out['query_value']=='pear'
    del row['mapping_reason']
    with pytest.raises(ContractError):decode_observation({'members':[row],'coverage':'complete'},p,[t],'discover_candidates')


@pytest.mark.parametrize('namespace',['physical_instance','semantic_category','text_value'])
def test_final_prompt_examples_and_public_semantic_axis(namespace):
    t=SetSpec('items',namespace,'fruit',count_unit='fruit kind')
    p={'question':'How many kinds are made, rather than just displayed?', 'sets':[{'set_id':'items','namespace':namespace}], 'catalog':{}}
    bind_interaction(p,[t],'discover_candidates')
    prompt=interaction_prompt(p,[t],'discover_candidates')
    assert p['question'] in prompt
    for raw in re.findall(r'```json\n(.*?)\n```',prompt,re.S):
        decode_observation(json.loads(raw),p,[t],'discover_candidates')


def test_generic_category_enters_real_controller_review(tmp_path):
    compiled=task('semantic_category');compiled['sets'][0].update(target='animal face',count_unit='animal face kind')
    def observe(p,m):return {'members':[category(p,i,value='origami animal face') for i in (1,2)],'coverage':'complete'}
    def inspect(p,m):
        assert 'count_unit axis' in p['review_goal'][0]['question']
        rows=[]
        for c in p['existing']:
            r=category(p,value='fox');r['object']=c['candidate_id'];rows.append(r)
        return {'members':rows,'coverage':'complete'}
    model=AnswerModel(compiled,observe,inspect);agent,req=setup(tmp_path,model)
    result=agent.solve(req)
    assert result.prediction=='A' and result.answer_mode=='determined'
    assert result.resources['calls_by_purpose']['qualification']==1
    assert result.resources['calls_by_purpose']['answer']==0
    assert next(iter(result.inventory['cards'].values()))['query_value']=='fox'


def test_option_conflict_reopens_nonempty_category_mapping(tmp_path):
    def observe(p,m):return {'members':[category(p,1),category(p,2)],'coverage':'complete'}
    def inspect(p,m):
        rows=[]
        for i,c in enumerate(p['existing']):
            r=category(p,value=['pear','apple'][i]);r['object']=c['candidate_id'];rows.append(r)
        return {'members':rows,'coverage':'complete'}
    model=AnswerModel(task('semantic_category'),observe,inspect);agent,req=setup(tmp_path,model)
    result=agent.solve(replace(req,choices={'A':'2','B':'3'}))
    assert result.prediction=='A' and result.resources['calls_by_purpose']['qualification']==1


def test_related_positive_is_retracted_by_targeted_check(tmp_path):
    def response(p,review=False):
        ref=next(k for k,v in p['catalog'].items() if v['region']=='core')
        return {'answers':[{'task':q,'status':'absent','refs':[]} if t['candidate']=='reading' and review else
            {'task':q,'status':'observed','refs':[ref], 'visible_fact':'A hand holds a closed book.' if t['candidate']=='reading' else 'A hand writes letters.',
             'evidence_kind':'preparation' if t['candidate']=='reading' else 'exact', 'note':'This is preparation.' if t['candidate']=='reading' else 'The writing act is visible.'}
            for q,t in p['interaction']['tasks'].items()]}
    def inspect(p,m):
        assert 'Challenge this claim' in p['check_review'][0]['reason']
        return response(p,True)
    compiled=task('semantic_category','missing_members',candidates=['writing','reading'])
    model=AnswerModel(compiled,lambda p,m:response(p),inspect);agent,req=setup(tmp_path,model)
    result=agent.solve(replace(req,choices={'A':'writing','B':'reading'}))
    assert result.prediction=='B' and result.answer_mode=='determined'
    assert result.resources['calls_by_purpose']['qualification']==1
    assert any(r['before']['state']=='unreadable' for r in result.inventory['check_revisions'])


def entity_model(answer=None):
    return AnswerModel(task(),lambda p,m:{'members':[member(p,1),member(p,2)],'coverage':'complete'},
        identity=lambda p,m:{'decision':'UNKNOWN','basis':'uncertain','refs':[], 'reason':'The identifying marks are hidden.'}, answer=answer)


def test_best_effort_keeps_uncertainty_evidence_and_resume(tmp_path):
    model=entity_model();agent,req=setup(tmp_path,model,request_kwargs={'checkpoint_path':str(tmp_path/'state.json')})
    result=agent.solve(req)
    assert result.prediction=='B' and result.answer_mode=='best_effort'
    assert result.result_status==result.evidence_status=='unresolved'
    assert result.support_level!='supported'
    assert result.value_state['final']['supported'] is False
    assert result.trace['best_effort']['ledger_modified'] is False
    assert result.resources['calls_by_purpose']['answer']==1
    call=model.calls[-1];assert call['role']=='best_effort'
    assert any(p['type']=='image' for p in call['messages'][0]['content'])
    text=json.dumps(call['payload']);assert 'answer_label' not in text and 'human_review' not in text
    n=len(model.calls); resumed=agent.solve(replace(req,resume=True))
    assert resumed.prediction=='B' and len(model.calls)==n


@pytest.mark.parametrize('bad',['unknown_ref','invalid_choice','truncated'])
def test_bad_final_answer_not_fabricated_or_retried(tmp_path,bad):
    def answer(p,m):
        if bad=='truncated':return '{"prediction":"B"'
        return {'prediction':'Z' if bad=='invalid_choice' else 'B', 'reason':'Best evidence favors this.',
                'refs':['F999' if bad=='unknown_ref' else next(iter(p['catalog']))], 'uncertainties':['Uncertain.']}
    model=entity_model(answer);agent,req=setup(tmp_path,model)
    result=agent.solve(req)
    assert result.prediction is None and result.trace['best_effort']['status']=='invalid'
    assert result.resources['calls_by_purpose']['answer']==1


def test_answer_reserved_inside_existing_total_budget(tmp_path):
    model=entity_model();agent,req=setup(tmp_path,model)
    req=replace(req,budget=replace(req.budget,max_model_calls=3))
    result=agent.solve(req)
    assert result.prediction=='B' and result.result_status=='budget_exhausted'
    assert result.resources['model_calls']==3
    assert result.resources['calls_by_purpose']['identity']==0
    assert result.resources['calls_by_purpose']['answer']==1


def test_real_responses_cannot_be_accepted_as_semantically_complete():
    from pathlib import Path
    from qwen3vl_agent.r4.collection_contracts import parse_json
    fixture=json.loads((Path(__file__).parent/'fixtures/r4_v5_6_semantic_failures.json').read_text())
    for record in fixture:
        category_case=record['source'].startswith('225')
        t=SetSpec('items','semantic_category','animal face',count_unit='animal face kind')
        p={'sets':[], 'catalog':{}, **({} if category_case else {'candidates':{'items':['Eating a meal.']}})}
        bind_interaction(p,[t],'discover_candidates')
        raw=parse_json(record['output'])
        with pytest.raises(ContractError) as exc:decode_observation(raw,p,[t],'discover_candidates')
        assert any(e['code'].startswith('schema') for e in exc.value.errors)
        if not category_case:
            # Even after fixing shape, the explicit disjunction remains a semantic doubt.
            raw['answers'][0].update(evidence_kind='exact',visible_fact='Preparing or eating a meal.')
            out=decode_observation(raw,p,[t],'discover_candidates')
            assert out['checks'][0]['uncertainties']


def test_returned_answer_replays_without_generation(tmp_path,monkeypatch):
    from qwen3vl_agent.r4 import answering
    validate=answering.validate
    def interrupted(*args,**kwargs):raise KeyboardInterrupt()
    model=entity_model();agent,req=setup(tmp_path,model,request_kwargs={'checkpoint_path':str(tmp_path/'returned.json')})
    monkeypatch.setattr(answering,'validate',interrupted)
    with pytest.raises(KeyboardInterrupt):agent.solve(req)
    n=len(model.calls);monkeypatch.setattr(answering,'validate',validate)
    result=agent.solve(replace(req,resume=True))
    assert result.prediction=='B' and len(model.calls)==n


def test_interrupted_answer_does_not_receive_another_call(tmp_path):
    def interrupted(p,m):raise KeyboardInterrupt()
    model=entity_model(interrupted);agent,req=setup(tmp_path,model,request_kwargs={'checkpoint_path':str(tmp_path/'interrupted.json')})
    with pytest.raises(KeyboardInterrupt):agent.solve(req)
    n=len(model.calls);result=agent.solve(replace(req,resume=True))
    assert result.prediction is None and len(model.calls)==n
    assert result.trace['best_effort']['failure']['code']=='interrupted_call'


@pytest.mark.parametrize('failure',[None, {'stage':'observe','code':'protocol_error','message':'Malformed observation'}])
def test_runner_keeps_best_effort_status_export_and_full_denominator(tmp_path,failure):
    from types import SimpleNamespace
    from test_r1345_debug_runner import case, Model as RunnerModel, PIPELINES
    from qwen3vl_agent.debug12 import run_cases
    cases=[case('R4',i) for i in (1,2)]
    model=RunnerModel()
    def factory(p,m,c):
        def solve(req):
            payload={'prediction':'B', 'failure':failure, 'result_status':'execution_failed' if failure else 'budget_exhausted',
                     'answer_mode':'best_effort', 'evidence_status':'budget_exhausted',
                     'support_level':'partial','completion_state':'budget_limited',
                     'answer_basis':'best_effort', 'resources':{'model_calls':0}}
            return SimpleNamespace(**payload,to_dict=lambda:payload)
        return SimpleNamespace(solve=solve)
    kwargs={'model_factory':lambda c:model,'agent_factory':factory,'gpu_check':None}
    configs={p:{'model':{},p.lower():{}} for p in PIPELINES}
    summary=run_cases(cases,cases,configs,tmp_path,{'signature':'v5.7'},**kwargs)
    assert summary['planned']==summary['answered']==2 and summary['correct']==0
    assert summary['r4_answer_mode_counts']['best_effort']==2
    assert summary['r4_result_status_counts']['execution_failed' if failure else 'budget_exhausted']==2
    assert len((tmp_path/'r4_native_predictions.jsonl').read_text().splitlines())==2
    run_cases(cases,cases,configs,tmp_path,{'signature':'v5.7'},resume=True,**kwargs)
    assert model.loads==1
    with pytest.raises(ValueError):run_cases(cases,cases,configs,tmp_path,{'signature':'changed'},resume=True,**kwargs)


def test_stage_protocol_failure_can_answer_from_retained_images(tmp_path):
    model=AnswerModel(task(),lambda p,m:'{broken response')
    agent,req=setup(tmp_path,model)
    result=agent.solve(req)
    assert result.result_status=='execution_failed' and result.failure
    assert result.prediction=='B' and result.answer_mode=='best_effort'
    assert result.inventory['cards']=={}
    assert result.trace['best_effort']['ledger_modified'] is False


def test_invalid_observe_prompt_spends_no_observation_call(tmp_path,monkeypatch):
    from qwen3vl_agent.r4 import interaction
    def broken(*args,**kwargs):raise ValueError('broken template')
    monkeypatch.setattr(interaction,'interaction_prompt',broken)
    model=entity_model();agent,req=setup(tmp_path,model)
    result=agent.solve(req)
    assert result.failure['code']=='prompt_contract_error'
    assert [c['role'] for c in model.calls]==['compile']
