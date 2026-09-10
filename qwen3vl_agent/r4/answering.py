"""Bounded best-effort answer, separate from the evidence ledger and its status."""
from types import SimpleNamespace
import json

from .collection_contracts import obj, STR, array, validate
from .contracts import ContractError, issue


def answer_schema(labels, refs):
    return obj({'prediction': {'enum': labels}, 'reason': {**STR, 'minLength': 1},
                'refs': array({'enum': refs} if refs else STR, 8),
                'uncertainties': array(STR, 8)}, ['prediction', 'reason', 'refs', 'uncertainties'])


def answer_prompt(payload):
    return ('Choose the best supported option from the existing evidence even when uncertain. '
            'The ledger contains model claims, NOT verified truth. Reconsider generic category labels, '
            'related actions mistaken for the requested action, and uncertain object identities using the images. '
            'Do not turn missing coverage into proven absence. Do not pick the closest number or default first option. '
            'Preserve the query unit, scope and completed-action conditions in the original question. '
            'A best-effort choice does not certify the ledger or change its counts. Explain any disagreement with '
            'the reduction, name remaining uncertainty, and cite only supplied F/T refs. '
            'Return one JSON object with prediction (one supplied option label), reason (short string), '
            'refs (array of supplied evidence IDs), uncertainties (array of short strings).\nINPUT\n'
            + json.dumps(payload, ensure_ascii=False))


def answer_input(controller, state):
    """Reuse returned observation inputs; never resample or obtain scoring annotations.

    Spread a bounded set across windows and include their final frames (completed
    results/captions often appear there). Keep source IDs in the durable mapping.
    """
    from qwen3vl_agent.coarse_to_fine.types import FrameRef
    calls = [r for r in controller.session.state['calls'].values()
             if r['status'] == 'returned' and r['role'] in ('discover_candidates', 'inspect_existing')]
    windows = {}
    for call in calls:
        if call['role'] == 'discover_candidates':
            windows.setdefault(call.get('root_id') or call['call_id'], call)
    selected = list(windows.values()) or calls
    if len(selected) > 8:
        selected = [selected[round(i * (len(selected)-1) / 7)] for i in range(8)]
    candidates = []
    for call in selected:
        images = [p for m in call['messages'] for p in m['content'] if p['type'] == 'image']
        rows = list(zip(call.get('source_frame_ids', []), images))
        if rows:
            candidates.append(rows[-1])
    # Fill spare slots with accepted witnesses, then central images.
    wanted = {r for c in (controller.store.cards.values() if controller.store else []) for r in c.get('evidence_refs', [])}
    for call in calls:
        images = [p for m in call['messages'] for p in m['content'] if p['type'] == 'image']
        rows = list(zip(call.get('source_frame_ids', []), images))
        candidates.extend((r,p) for r,p in rows if r in wanted)
        if rows: candidates.append(rows[len(rows)//2])
    chosen, seen = [], set()
    for ref, part in candidates:
        if ref not in seen and len(chosen) < 8:
            chosen.append((ref, part)); seen.add(ref)
    frames, sizes, aliases, catalog, public = [], [], {}, {}, {}
    for ref, part in chosen:
        from PIL import Image
        with Image.open(part['image']) as im: size = im.size
        short = 'F'+str(len(frames)+1)
        meta = controller.work['catalog'][ref]
        frames.append(FrameRef(ref, meta['start_sec'], part['image'])); sizes.append(size)
        aliases[short] = ref; catalog[ref] = meta
        public[short] = {'region': 'answer_review', 'source': meta.get('entry_id'),
                         'source_time': meta['start_sec']}
    for ref, meta in controller.work['catalog'].items():
        if meta.get('kind') != 'frame' and meta.get('text') and len(public) < 20:
            short = 'T'+str(sum(k.startswith('T') for k in public)+1)
            aliases[short] = ref; catalog[ref] = meta
            public[short] = {'text': meta['text'][:800], 'source': meta.get('entry_id')}
    store = controller.store
    payload = {'question': controller.request.question,
               'choices': [{'label':c.label,'text':c.text} for c in controller.request.choices],
               'compiled_task': controller.spec.to_dict(), 'catalog': public,
               'reduction': {k:v for k,v in state.get('final', {}).items() if k != 'view'},
               'claims': [{k:c.get(k) for k in ('candidate_id','raw_value','query_value','membership','facts','issues')}
                          for c in list(store.cards.values())[:24]] if store else [],
               'checks': [{k:r.get(k) for k in ('candidate','state','facts','support','needs_review')}
                          for r in list(store.state['checks'].values())[-24:]] if store else [],
               'identity_claims': [{k:r.get(k) for k in ('left','right','relation','reason','active')} for r in store.state['relations'][-24:]] if store else [],
               'stage_failures':controller.work.get('failures', [])[-4:],
               'stop': controller.work.get('stop_detail'),
               'limitations': ['Only selected previously supplied frames/text are shown; other ranges may remain unchecked.']}
    prepared = SimpleNamespace(frames=frames, sizes=sizes, pixels=sum(w*h for w,h in sizes)) if frames else None
    return payload, prepared, aliases, catalog


def best_effort(controller, state):
    payload, prepared, aliases, catalog = answer_input(controller, state)
    if not aliases:
        return None, {'status': 'unavailable', 'reason': 'no_available_evidence'}
    rec = controller.session.call('best_effort:0', 'best_effort', payload, pool='answer',
                                  prepared=prepared, aliases=aliases, catalog=catalog)
    try:
        output = validate(controller.session.parse(rec), answer_schema(
            [c.label for c in controller.request.choices], list(aliases)))
        if not output['refs']:
            raise ContractError([issue('refs','answer_evidence_required','Cite supplied evidence for the best-effort choice')])
        controller.session.validation(rec)
    except ContractError as exc:
        controller.session.validation(rec, exc.errors)
        return None, {'status':'invalid','call_id':rec['call_id'],'errors':exc.errors}
    return output['prediction'], {**output, 'status':'returned', 'call_id':rec['call_id'],
                                  'evidence_refs':[aliases[r] for r in output['refs']],
                                  'answer_mode':'best_effort', 'certainty':'uncertain',
                                  'ledger_modified':False}
