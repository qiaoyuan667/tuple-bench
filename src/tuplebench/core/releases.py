"""Only specified fields cross the recipient boundary; all views share one run."""
from __future__ import annotations
import copy
import math
from .io import require, finite_number
PROJECTIONS = ('public_only', 'choice', 'choice_confidence', 'choice_max_probability', 'choice_margin', 'choice_top2_ids', 'choice_top2_scores', 'full_probabilities', 'full_native')
SPECIAL = ('backend_action_only', 'oracle_T_only')
ALL_RELEASES = PROJECTIONS + SPECIAL

def validate_response(response: dict, candidate_order: list[str], tolerance: float=1e-06) -> list[str]:
    require(isinstance(response, dict), 'Response must be an object')
    require(response.get('choice') in candidate_order, 'Choice not in candidate set')
    p = response.get('probabilities')
    require(isinstance(p, dict) and set(p) == set(candidate_order), 'Probability keys must exactly match candidates')
    require(all((finite_number(v) and 0 <= v <= 1 for v in p.values())), 'Invalid probability value')
    require(abs(sum(p.values()) - 1) <= tolerance, 'Probability sum outside declared tolerance')
    require(finite_number(response.get('confidence')) and 0 <= response['confidence'] <= 1, 'Native confidence required; it is never imputed from max probability')
    warnings = []
    if p[response['choice']] < max(p.values()) - tolerance:
        warnings.append('native_choice_not_argmax; preserved_as_returned')
    return warnings

def release_fields(response: dict | None, release_id: str, candidate_order: list[str], status: str='completed') -> dict:
    require(release_id in ALL_RELEASES, 'Unknown release mechanism')
    if release_id in ('public_only', 'backend_action_only'):
        return {}
    require(release_id != 'oracle_T_only', 'Oracle reference must be explicitly constructed by evaluator')
    if status != 'completed':
        return {'execution_status': 'unavailable'}
    require(response is not None, 'Missing completed response')
    p = response['probabilities']
    ranks = sorted(candidate_order, key=lambda k: (-p[k], candidate_order.index(k)))
    r = {'choice': response['choice']}
    if release_id == 'choice_confidence':
        r['confidence'] = response['confidence']
    elif release_id == 'choice_max_probability':
        r['max_probability'] = max(p.values())
    elif release_id == 'choice_margin':
        r['margin'] = p[ranks[0]] - p[ranks[1]]
    elif release_id == 'choice_top2_ids':
        r['top2_ids'] = ranks[:2]
    elif release_id == 'choice_top2_scores':
        r['top2'] = [{'id': k, 'score': p[k]} for k in ranks[:2]]
    elif release_id in ('full_probabilities', 'full_native'):
        r['probabilities'] = {k: p[k] for k in candidate_order}
        if release_id == 'full_native':
            r['confidence'] = response['confidence']
    return r

def order_from_case(case: dict) -> list[str]:
    return [c['id'] for c in case['model_view']['question']['candidates']]
