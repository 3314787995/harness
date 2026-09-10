"""Host-owned questions and compact answers; no inferred visual facts.

The compiler and evidence store retain their contracts. This boundary translates
explicit judgments, never repairs malformed answers or manufactures evidence.
"""
import copy
import json
from .collection_contracts import STR, REFS, TRI, obj, array, box_schema, validate, conditions
from .contracts import ContractError, issue


def bind_interaction(payload, targets, role):
    if role not in {'discover_candidates', 'inspect_existing'}:
        return
    # History, motion and simultaneous censuses keep their specialized contract.
    if any(t.namespace == 'task_item' or t.predicate_kind != 'static' for t in targets) or payload.get('simultaneous_sets'):
        return
    tasks = {}
    requirements = {}
    for t in targets:
        if t.set_id in payload.get('candidates', {}):
            for candidate in payload['candidates'][t.set_id]:
                tasks['Q'+str(len(tasks)+1)] = {'set':t.set_id, 'candidate':candidate}
        else:
            requirements[t.set_id] = {'J'+str(i+1): {'field':k, 'question':v}
                for i,(k,v) in enumerate(conditions(t).items())}
    payload['interaction'] = {'version':1, 'tasks':tasks, 'requirements':requirements,
        'existing_ids':[c['candidate_id'] for c in payload.get('existing', [])],
        'category_questions':{t.set_id:'Which '+(t.count_unit or t.target)+' is represented? Separate the concrete kind from an umbrella title; justify it from the image and original question.' for t in targets if t.namespace=='semantic_category' and t.set_id in requirements and t.equivalence!='combination'}}


def answer_schema(binding):
    row=obj({'task':{'enum':list(binding['tasks'])},'status':{'enum':['observed','absent','uncertain']},
             'refs':REFS,'note':STR,
             'evidence_kind':{'enum':['exact','preparation','result','associated','uncertain']},
             'visible_fact':STR}, ['task','status','refs'])
    row['allOf']=[{'if':{'properties':{'status':{'const':'observed'}}},'then':{'required':['note','evidence_kind','visible_fact']}}]
    return row


def pair_schema():
    schema=obj({'decision':{'enum':['SAME','DIFFERENT','UNKNOWN']},
        'basis':{'enum':['reidentification','stable_difference','coexistence','uncertain']},
        'refs':{**REFS,'maxItems':4},'features':array({**STR,'maxLength':96},3),
        'reason':{**STR,'maxLength':160}},['decision','basis','refs','reason'])
    schema['allOf']=[{'if':{'properties':{'decision':{'enum':['SAME','DIFFERENT']}}},
                      'then':{'required':['features'],'properties':{'features':{'minItems':1},'refs':{'minItems':1}}}}]
    return schema


def decode_pair(data,payload):
    row=validate(data,pair_schema())
    allowed={'SAME':{'reidentification'},'DIFFERENT':{'stable_difference','coexistence'},'UNKNOWN':{'uncertain'}}
    if row['basis'] not in allowed[row['decision']] or row['basis'] not in payload['pair_task']['available_bases']:
        raise ContractError([issue('basis','unavailable_identity_evidence','Only use bases supported by this input type; isolated representatives do not show tracks',
                                   expected=payload['pair_task']['available_bases'],actual=row['basis'])])
    result={'left':payload['left'][0],'right':payload['right'][0],'relation':row['decision'],
            'basis':row['basis'],'refs':row['refs'],'facts':row['reason']}
    if 'features' in row: result['features']=row['features']
    if row['decision']=='DIFFERENT': result['independent_objects']=True # the explicit decision itself asserts independence
    return {'relations':[result]}


def pair_prompt(payload,feedback=None):
    examples=[
        ('Hypothetical tool with matching distinctive damage in both views',
         {'decision':'SAME','basis':'reidentification','refs':['F1','F2'],'features':['same broken upper tooth and engraved serial'],
          'reason':'Matching damage and serial identify this particular tool.'}),
        ('Hypothetical tools with incompatible permanent structures',
         {'decision':'DIFFERENT','basis':'stable_difference','refs':['F1','F2'],'features':['one has a welded closed end; the other has an open fork'],
          'reason':'The permanent end structures distinguish the two tools.'}),
        ('Hypothetical obstructed identifying region',
         {'decision':'UNKNOWN','basis':'uncertain','refs':['F1','F2'],
          'reason':'The handle marking is hidden in F2; that identifying feature cannot be compared.'})]
    for _, example in examples: validate(example,pair_schema())
    text=('Judge ONLY the one supplied pair. Return decision,basis,refs,reason, and features for SAME/DIFFERENT. '
          'No object IDs or relations graph in output; the host binds your answer to this pair. '
          'These are isolated representative images, NOT tracks: continuous_track and distinct_tracks cannot be claimed. '
          'SAME requires distinctive stable matching features and context (reidentification). '
          'DIFFERENT requires stable distinguishing physical features or actual coexistence; scene changes, contents, color alone or unmatched views do not suffice. '
          'The decision explicitly asserts identity/independence; do not guess it. UNKNOWN is a valid completed comparison. '
          'Use at most 3 short features (96 characters each), 4 evidence refs and a 160-character reason. '
          'Inspect the visible structure in both objects before deciding. UNKNOWN must state the specific missing or conflicting cue; '
          'do not copy a generic inability sentence. Do not describe every frame.\n')
    for explanation, example in examples:
        text+='FORMAT ONLY, unrelated imagined objects; never copy these facts or references: '+explanation+'\n```json\n'+json.dumps(example)+'\n```\n'
    if feedback: text+='Rejudge this SAME pair from the SAME images. No invalid-record slots apply here. If evidence is insufficient, choose UNKNOWN. Errors: '+json.dumps(feedback.get('errors',[]))+'\n'
    # Avoid carrying the original discovery window: both objects are evidence for this comparison.
    public={k:payload[k] for k in ('pair_task','left','right','existing','catalog','evidence_by_object')}
    return text+'INPUT\n'+json.dumps(public,ensure_ascii=False)


def member_schema(target, binding, inspection=False):
    props={'set':{'const':target.set_id},'label':STR,
        'judgments':obj({j:TRI for j in binding['requirements'][target.set_id]},binding['requirements'][target.set_id]),
        'note':STR,'visibility':{'enum':['clear','occluded','unreadable','unknown']},
        'uncertainties':array(STR), 'attributes':{'type':'object'}, 'text':{'type':['string','null']}}
    props['object']={'enum':binding['existing_ids']} if inspection else {'type':'string','pattern':r'^O[1-9][0-9]*$'}
    required=['set','object','label','judgments']
    if target.namespace=='semantic_category' and target.equivalence!='combination':
        props.update(query_value={'type':['string','null']},
                     category_basis={'enum':['specific_kind','umbrella','unknown']}, mapping_reason=STR)
        required += ['query_value','category_basis','mapping_reason']
    if target.namespace=='physical_instance':
        props['boxes']={**array(box_schema(),24),'minItems':1}; required.append('boxes')
    else:
        props['refs']={**REFS,'minItems':1}; required.append('refs')
        if target.namespace=='text_value': required.append('text')
    return obj(props,required)


def decode_observation(data, payload, targets, role):
    binding=payload['interaction']
    if payload.get('coverage_review'):
        data=validate(data,obj({'coverage':{'enum':['complete','partial','unreadable']},
                               'gaps':array(STR),'regions':array(region_schema(binding),8)},['coverage']))
        out={'updates':[], 'checks':[], 'coverage':data['coverage'], 'gaps':data.get('gaps',[]), 'overflow':False}
        out['_regions']=validate_regions(data.get('regions',[]),payload)
        if out['_regions'] and out['coverage']=='complete':
            raise ContractError([issue('regions','coverage_conflict','Unresolved regions cannot accompany complete coverage')])
        return out
    schema=obj({'members':array({'type':'object'},12),'answers':array(answer_schema(binding),12),
                'separate':array(obj({'objects':{'type':'array','items':STR,'minItems':2,'maxItems':12,'uniqueItems':True},
                                     'ref':REFS['items']},['objects','ref']),12),
                'coverage':{'enum':['complete','partial','unreadable']},'gaps':array(STR),
                'regions':array(region_schema(binding),8),'overflow':{'type':'boolean'}},
               (['members','coverage'] if binding['requirements'] else []) + (['answers'] if binding['tasks'] else []))
    data=validate(data,schema)
    got=[r['task'] for r in data.get('answers',[])]
    if len(got)!=len(set(got)) or set(got)!=set(binding['tasks']):
        raise ContractError([issue('answers','task_set_mismatch','Answer every requested Q ID exactly once; do not invent or omit tasks',
                                   expected=list(binding['tasks']),actual=got)])
    out={'checks':[], 'coverage':data.get('coverage','complete'),'gaps':data.get('gaps',[]),
         'overflow':data.get('overflow',False)}
    out['_regions']=validate_regions(data.get('regions',[]),payload)
    if out['_regions'] and out['coverage']=='complete':
        raise ContractError([issue('regions','coverage_conflict','Unresolved regions cannot accompany complete coverage')])
    # Sort by host task order so array positions do not depend on model ordering.
    answers={r['task']:r for r in data.get('answers',[])}
    for q,bound in binding['tasks'].items():
        r=answers[q]; state={'observed':'seen','absent':'not_seen','uncertain':'unreadable'}[r['status']]
        check={**bound,'state':state,'refs':r['refs'],
               'facts':r.get('note',f'Model reported {r["status"]} for host task {q} in this input.')}
        if state=='seen':
            # Explicit semantic relation is retained, never inferred from the status alone.
            direct = r['evidence_kind']=='exact' and bool(r['visible_fact'].strip())
            check['support']='direct' if direct else 'related'
            check['facts']=r['visible_fact']+' | '+r['note']
            if not direct:
                check['uncertainties']=['Claimed observed but evidence_kind='+r['evidence_kind']]
            # Disjunctive rationales explicitly leave the requested action unresolved.
            if ' or ' in r['visible_fact'].casefold() or ' or ' in r['note'].casefold():
                check['uncertainties']=['Disjunctive positive claim requires exact-action review']
        out['checks'].append(check)
    by_set={t.set_id:t for t in targets}
    rows=[]; seen=set()
    for i,r in enumerate(data.get('members',[])):
        sid=r.get('set')
        if sid not in binding['requirements']:
            raise ContractError([issue(f'members[{i}].set','unrequested_set','This set has no member discovery task')])
        t=by_set[sid]; validate(r,member_schema(t,binding,role=='inspect_existing'),f'members[{i}]')
        if r['object'] in seen:
            raise ContractError([issue(f'members[{i}].object','duplicate_object','Each object ID occurs once')])
        seen.add(r['object'])
        row={'set':sid,'name':r['label'],'class':r['label'],
             'conditions':{binding['requirements'][sid][j]['field']:v for j,v in r['judgments'].items()},
             'facts':r.get('note','Model supplied label: '+r['label'])}
        row['candidate_id' if role=='inspect_existing' else 'id']=r['object']
        for k in ('boxes','refs','visibility','uncertainties','attributes'):
            if k in r: row[k]=copy.deepcopy(r[k])
        if t.namespace=='semantic_category' and t.equivalence!='combination':
            value = r['query_value'] if r['category_basis']=='specific_kind' else None
            row.update(query_value=value, mapping_evidence=r['mapping_reason'])
            row['class']=value or r['label']
            row['facts'] += ' | category_basis='+r['category_basis']+'; mapping: '+r['mapping_reason']
        if t.namespace=='text_value': row['raw_text']=r['text']
        rows.append(row)
    out['updates' if role=='inspect_existing' else 'records']=rows
    if data.get('separate'):
        if role=='inspect_existing':
            raise ContractError([issue('separate','inspection_relation_not_discovery','Inspection updates stable members; use pair comparison for identity')])
        out['coexisting']=[{'ids':r['objects'],'ref':r['ref'],'independent_objects':True} for r in data['separate']]
    return out


def region_schema(binding):
    return obj({'set':{'enum':list(binding['requirements'])}, 'refs':{**REFS,'minItems':1,'maxItems':3},
                'reason':{'enum':['occluded','blurred','too_small','uninspected']},'detail':STR},
               ['set','refs','reason','detail'])


def validate_regions(regions,payload):
    for i,r in enumerate(regions):
        for j,ref in enumerate(r['refs']):
            meta=payload['catalog'].get(ref)
            if meta is None or r['set'] not in meta.get('sets',[]) or meta.get('region')!='core':
                raise ContractError([issue(f'regions[{i}].refs[{j}]','invalid_gap_reference',
                    'Locate a gap with displayed core evidence belonging to its set',actual=ref)])
    return copy.deepcopy(regions)


def interaction_prompt(payload, targets, role, feedback=None):
    binding=payload['interaction']
    lines=['Complete only the host tasks. Return one JSON object. No option selection.']
    examples=[]
    if payload.get('coverage_review'):
        lines += ['Audit the supplied core input for the named collection. Existing findings are retained by the host; do not output members or update their identities.',
                  'complete means this sampled input was actually searched and no unresolved search region or missed candidate remains. It does NOT certify every video frame or settle member identity/category.',
                  'If a previously omitted candidate is visible, use partial with reason uninspected and its frame. If a real region obstructs search, localize it. An unknown animal name is a category question, not automatically a whole-window search gap.',
                  'Return coverage, optionally gaps and regions. regions is an array of {set,refs,reason,detail}; refs is an array of core F/T strings. reason: occluded, blurred, too_small, uninspected. Never invent a region to fill the format.']
        sid=next(iter(binding['requirements']))
        examples=[('Imagined search finished, with no unresolved region',{'coverage':'complete','gaps':[]}),
                  ('Imagined tool partly hidden by a hand',{'coverage':'partial','regions':[{'set':sid,'refs':['F1'],'reason':'occluded','detail':'A hand hides the tool area at the lower left.'}]})]
    else:
        if binding['tasks']:
            lines += ['For Q tasks use ONLY status observed/absent/uncertain; yes/no are not Q status values.',
                      'observed means the EXACT requested action/object is directly shown with core evidence. Associated objects, preparation or results alone mean uncertain. absent means inspected but not observed in this input. uncertain means evidence prevents judging.',
                      'Answer each supplied Q ID exactly once. refs is a flat array of frame/text strings, for example ["F2","F3"], never frame/box/label objects. An observed answer also has a record-level note describing the direct evidence.',
                      'For observed also return visible_fact (literal visible action/object, before interpreting the candidate) and evidence_kind: exact/preparation/result/associated/uncertain. A candidate name repeated in note is not a visible fact. Only exact evidence supports observed. Compare the actual fact to the requested candidate, including what distinguishes performance from preparation. Never promote preparation because related objects are present. When there are only Q tasks, omit member and coverage fields.']
            qs=list(binding['tasks'])
            for state in ('observed','absent','uncertain'):
                rows=[{'task':q,'status':state,'refs':['F1'] if state=='observed' else [],
                       **({'note':'The visible act matches the hypothetical requested action.', 'visible_fact':'The person writes letters on paper with a moving pen.', 'evidence_kind':'exact'} if state=='observed' else {})} for q in qs]
                examples.append(('Unrelated hypothetical writing-a-note task, not the current Q definition: '+state,{'answers':rows}))
        if binding['requirements']:
            lines += ['For member tasks, identify actual members and answer their J questions with yes/no/unknown. label must identify the concrete category at query granularity; repeating the collection target or unit does not identify a category.',
                      'For categories, label preserves the observed name/caption; query_value separately names the kind on the supplied count_unit axis. category_basis is specific_kind/umbrella/unknown. mapping_reason explains the visible features or caption supporting the mapping. A video title or upper-level class is umbrella: query_value=null. A decorative name is not a new kind. Read individual labels and visual structure, not just the shared title.',
                      'Member uncertainty and search coverage are separate: a visible but unnamed member can have unknown judgments while the input search is complete. Do not mark the entire window partial just because identity/category remains unknown.',
                      'coverage complete means all supplied core search input was inspected without unresolved search regions. It is not exhaustive video proof. A searched empty input may return members:[] with coverage:complete.',
                      'For a real search gap, give regions:[{set,refs,reason,detail}], with core F/T string refs, reason occluded/blurred/too_small/uninspected and a specific location/cause. Do not copy generic gaps. Preserve real uncertainty.',
                      'Only core evidence establishes membership. Context is for interpretation. O IDs are objects, F/T IDs are evidence. note is optional.']
            if any(t.namespace=='physical_instance' for t in targets):
                lines += ['An entity has 1–3 representative boxes, each {ref:"F1",xyxy:[left,top,right,bottom]}, four positive-area integer coordinates in 0–1000. Keep same-object appearances together when established.',
                          'Independent objects actually coexisting in one frame may be asserted as separate:[{objects:["O1","O2"],ref:"F1"}]. Each needs its own box on that frame. Names or duplicate records alone do not establish independence.']
            if any(t.namespace=='text_value' for t in targets): lines.append('For text values, text preserves the original visible wording; never correct its spelling.')
            rows=[]
            for n,(sid,qs) in enumerate(binding['requirements'].items(),1):
                t=next(t for t in targets if t.set_id==sid)
                stable=next((c['candidate_id'] for c in payload.get('existing',[]) if c['set']==sid),None)
                r={'set':sid,'object':stable if role=='inspect_existing' and stable else 'O'+str(n),
                   'label':'wrench' if t.namespace=='physical_instance' else 'pear' if t.namespace=='semantic_category' else 'OPEN',
                   'judgments':{j:'yes' for j in qs}}
                if t.namespace=='physical_instance': r['boxes']=[{'ref':'F1','xyxy':[120,180,380,620]}]
                else: r['refs']=['F1']
                if t.namespace=='semantic_category' and t.equivalence!='combination':
                    r.update(query_value='pear', category_basis='specific_kind',
                             mapping_reason='Hypothetical fruit-kind task: the visible pear shape identifies pear, not the title fruit.')
                if t.namespace=='text_value': r['text']='OPEN'
                if t.equivalence=='combination': r['attributes']={k:'EXAMPLE' for k in t.attribute_keys}
                validate(r,member_schema(t,binding,role=='inspect_existing')); rows.append(r)
            base={'members':rows,'coverage':'complete','gaps':[]}
            if binding['tasks']: base['answers']=examples[1][1]['answers']
            examples=[('Unrelated imaginary tools, fruit or sign, after a completed search',base)] if not binding['tasks'] else [(e,{**base,**x}) for e,x in examples]
            unknown=copy.deepcopy(base)
            for r in unknown['members']:
                r['label']='unidentified object'; r['judgments']={j:'unknown' for j in r['judgments']}
                if 'query_value' in r: r.update(query_value=None,category_basis='unknown',mapping_reason='The distinguishing features are hidden.')
            examples.append(('Imagined member is located but not identified; no unsearched region',unknown))
            if role=='inspect_existing':
                lines += ['Update each supplied stable candidate ID; do not append discoveries. Resolve the stated review_goal using these images. A context-only old witness needs a new core witness, not relabeling the old evidence.',
                          'When frames carry core_for, they establish core membership ONLY for the listed candidate IDs. For a batch use each candidate\'s own witnesses, never borrow the first candidate\'s frame for the rest.',
                          'For generic target labels, identify the specific kind from the image. If that cannot be determined, keep unknown and describe what prevents identification.']
    for explanation,example in examples:
        # Full runtime wire structure acceptance (reference applicability belongs to the real input).
        if payload.get('coverage_review'):
            validate(example,obj({'coverage':{'enum':['complete','partial','unreadable']},'gaps':array(STR),'regions':array(region_schema(binding),8)},['coverage']))
        else:
            dummy=copy.deepcopy(payload)
            decode_observation(example,dummy,targets,role)
        lines += ['FORMAT ONLY: '+explanation+'. These imagined labels, refs and facts are NOT input evidence. Choose from the actual media, never copy the example.\n```json\n'+json.dumps(example,ensure_ascii=False)+'\n```']
    if feedback:
        lines += ['REOBSERVE the same supplied media; correct the complete requested response. Already committed slots are ignored, never duplicated. Keep Q/object IDs fixed.',
                  'Use only this task branch. Q refs must be strings and its status observed/absent/uncertain; member J judgments alone use yes/no/unknown. Do not copy internal records/checks wrappers.',
                  'Precise validation errors: '+json.dumps(feedback.get('errors',[]),ensure_ascii=False)]
    public={k:payload[k] for k in ('question','existing','catalog','sets','interaction','candidates','check_review','review_goal','coverage_review') if k in payload}
    return '\n'.join(lines)+'\nINPUT\n'+json.dumps(public,ensure_ascii=False)
