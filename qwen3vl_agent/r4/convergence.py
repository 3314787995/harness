"""Deterministic obligations and progress accounting, never visual truth inference."""
import json
import re


def generic_category(value, target):
    if target.namespace != 'semantic_category' or target.equivalence == 'combination' or not isinstance(value, str):
        return False
    def norm(s):
        return ' '.join(re.sub(r'[^\w\s]', ' ', s.casefold()).split())
    words = set(norm(value).split())
    return any(set(norm(x).split()) <= words for x in (target.target, target.count_unit or '') if norm(x))


def semantic_review(card):
    return (card.get('membership') != 'excluded' and
            ((card.get('namespace') == 'semantic_category' and card.get('query_value') is None) or
             'context_only_member' in card.get('issues', []) or
             any(i.startswith('required_attribute_missing:') for i in card.get('issues', []))))


def progress_snapshot(store, windows):
    """Measure evidence changes separately from the final (possibly open) bounds."""
    cards = sorted((c['candidate_id'], c['membership'], c.get('query_value'), tuple(c.get('issues', [])))
                   for c in store.cards.values() if not c.get('superseded_window'))
    relations = sorted((r['left'], r['right'], r['relation']) for r in store.state['relations']
                       if r.get('active') and r['relation'].lower() != 'unknown')
    coverage = sorted((w['tile_id'], w['status']) for w in windows.values() if not w.get('children'))
    checks = sorted((k, r.get("state"), r.get("facts"), r.get("support"), r.get("reviewed")) for k,r in store.state["checks"].items())
    return json.dumps([cards, relations, coverage, checks], sort_keys=True)


def obligations(store, windows):
    rows = []
    for c in store.cards.values():
        if c['membership'] == 'excluded':
            continue
        if c['namespace'] == 'semantic_category' and c.get('query_value') is None:
            rows.append({'kind':'category_identification', 'candidate_id':c['candidate_id'], 'window_id':c['window_id']})
        if c['membership'] == 'unknown':
            rows.append({'kind':'core_witness' if 'context_only_member' in c.get('issues', []) else 'membership',
                         'candidate_id':c['candidate_id'], 'window_id':c['window_id']})
    for w in windows.values():
        if w.get('children') or w['status'] == 'complete':
            continue
        rows.append({'kind':'coverage', 'window_id':w['tile_id'], 'status':w['status'],
                     'localized':bool(w.get('localized_gaps')), 'gaps':w.get('gaps', []),
                     'audit_attempted':bool(w.get('coverage_audit_id'))})
    return rows
