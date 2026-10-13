"""Three downstream tasks with independent realized losses.

Policies only receive a class posterior and public specifications. T is used by
execute_plan AFTER the initial action, for approved observations and scoring.
"""
from __future__ import annotations
import copy
from collections import defaultdict
import numpy as np
from .contracts import class_ids, check_law, cost_vector
from .features import utility_features
from .io import require, digest
from .learning import fit_prior, fit_linear, predict, smooth_posterior
from .releases import PROJECTIONS, release_fields, order_from_case
from .statistics import cluster_mean_ci

def initial_plan(q: np.ndarray, domain: dict, consumer: dict) -> dict:
    labels = class_ids(domain)
    kind = consumer['consumer_id']
    if kind == 'asymmetric_cost_routing':
        mat = np.asarray(consumer['cost_matrix_by_domain'][domain['domain_id']], float)
        action = int(np.argmin(mat @ q))
        return {'kind': 'GENERAL_INSPECTION' if action == len(labels) else 'ROUTE', 'route': None if action == len(labels) else labels[action]}
    route = int(np.argmax(q))
    stop = float(consumer['wrong_cost'] * (1 - q[route]))
    if kind == 'selective_routing':
        if float(consumer['review_cost']) < stop:
            return {'kind': 'REVIEW', 'route': None}
        return {'kind': 'ROUTE', 'route': labels[route]}
    require(kind == 'one_step_clarification', 'Unknown consumer')
    best = {'kind': 'ROUTE', 'route': labels[route]}
    best_cost = stop
    for check in domain['public_checks']:
        expected = float(consumer['check_cost'])
        yes = np.array([check['p_yes_given_T'][t] for t in labels])
        for likelihood in [yes, 1 - yes]:
            joint = q * likelihood
            expected += consumer['wrong_cost'] * (float(joint.sum()) - float(joint.max()))
        if expected < best_cost - 1e-12:
            best = {'kind': 'CHECK', 'check_id': check['check_id'], 'route': None}
            best_cost = expected
    return best

def execute_plan(q: np.ndarray, plan: dict, domain: dict, consumer: dict, T: str, approved_answers: dict, execution_ok: bool=True) -> dict:
    """The environment owns T; it releases it only for a paid trusted review."""
    labels = class_ids(domain)
    kind = consumer['consumer_id']
    trace = {'initial_action': plan['kind'], 'chosen_check_id': None, 'check_answer': None, 'final_route': None, 'execution_status': 'ok'}
    used_review = False
    used_check = False
    if not execution_ok:
        trace.update(initial_action='FALLBACK_REVIEW', final_route=T, execution_status='unavailable')
        return {'trace': trace, 'cost': 2.0, 'final_correct': True, 'automated': False, 'used_check': False, 'used_review': True, 'initial_plan': {'kind': 'FALLBACK_REVIEW'}}
    if plan['kind'] in ('REVIEW', 'GENERAL_INSPECTION'):
        used_review = True
        trace['final_route'] = T
        cost = float(consumer['review_cost']) if kind == 'selective_routing' else float(consumer['cost_matrix_by_domain'][domain['domain_id']][-1][labels.index(T)])
    elif plan['kind'] == 'CHECK':
        used_check = True
        qid = plan['check_id']
        answer = approved_answers[qid]
        require(answer in ('yes', 'no'), 'Invalid check answer')
        likelihood = np.array([check_law(domain, qid, t)[answer] for t in labels])
        posterior = q * likelihood
        require(posterior.sum() > 0, 'Approved answer has zero likelihood under all classes')
        route = labels[int(np.argmax(posterior))]
        trace.update(chosen_check_id=qid, check_answer=answer, final_route=route)
        cost = float(consumer['check_cost'] + (0 if route == T else consumer['wrong_cost']))
    else:
        require(plan['kind'] == 'ROUTE', 'Invalid downstream action')
        route = plan['route']
        trace['final_route'] = route
        cost = cost_vector(consumer, domain, T)['ROUTE:' + route]
    return {'trace': trace, 'cost': float(cost), 'final_correct': trace['final_route'] == T, 'automated': not used_review and (not used_check), 'used_check': used_check, 'used_review': used_review, 'initial_plan': plan}

def _payload(case, run, release):
    return release_fields(run.get('response'), release, order_from_case(case), run['status'])

def model_q(model: dict, cases: list[dict], runs: dict, release: str, labels: list[str], floor: float) -> np.ndarray:
    if model['kind'] == 'native_distribution':
        result = []
        for c in cases:
            r = runs[c['case_id']]
            q = np.array([r['response']['probabilities'][t] for t in labels]) if r['status'] == 'completed' else np.ones(len(labels))
            result.append(smooth_posterior(q, floor))
        return np.array(result)
    features = [utility_features(c['consumer_view'], _payload(c, runs[c['case_id']], release)) for c in cases]
    probs = predict(model, features, labels)
    return np.array([smooth_posterior(q, floor) for q in probs])

def _execute_cases(cases, probs, runs, domain, consumer):
    output = []
    for c, q in zip(cases, probs):
        e = c['evaluator_only']
        plan = initial_plan(q, domain, consumer)
        output.append(execute_plan(q, plan, domain, consumer, e['truth']['T'], e['approved_check_answers'], runs[c['case_id']]['status'] == 'completed'))
    return output

def fit_and_evaluate_utility(ds, runs: dict) -> tuple[list[dict], dict, list[dict], list[dict]]:
    episodes = []
    selected_models = {}
    validation_log = []
    risk_curves = []
    floor = ds.engine['posterior_floor']
    bydomain = {n: [c for c in ds.cases if c['domain_id'] == n] for n in ds.domains}
    for name, domain in ds.domains.items():
        labels = class_ids(domain)
        allcases = bydomain[name]
        train = [c for c in allcases if c['evaluator_only']['split'] == 'train' and runs[c['case_id']]['status'] == 'completed']
        val = [c for c in allcases if c['evaluator_only']['split'] == 'validation']
        require(train and val, 'Need training responses and validation cases in each domain')
        y = [c['evaluator_only']['truth']['T'] for c in train]
        require(set(y) == set(labels), 'Each coarse class needs at least one valid training response')
        for release in PROJECTIONS:
            ftrain = [utility_features(c['consumer_view'], _payload(c, runs[c['case_id']], release)) for c in train]
            candidates = [('empirical_prior', fit_prior(y, labels))]
            for C in ds.engine['utility_regularization_C']:
                candidates.append((f'logistic_C={C}', fit_linear(ftrain, y, labels, C)))
            if release in ('full_probabilities', 'full_native'):
                candidates.append(('native_distribution', {'kind': 'native_distribution', 'classes': labels}))
            val_probs = [model_q(m, val, runs, release, labels, floor) for _, m in candidates]
            for consumer_id, consumer in ds.consumers.items():
                costs = []
                for (mid, model), qp in zip(candidates, val_probs):
                    results = _execute_cases(val, qp, runs, domain, consumer)
                    costs.append(float(np.mean([x['cost'] for x in results])))
                best = int(np.argmin(costs))
                mid, model = candidates[best]
                modelkey = f'{name}/{consumer_id}/{release}'
                selected_models[modelkey] = {'candidate': mid, 'model': model, 'release': release, 'train_ids_sha256': digest(sorted((c['case_id'] for c in train))), 'validation_ids_sha256': digest(sorted((c['case_id'] for c in val))), 'selection': 'minimum validation realized cost, fixed candidate order for ties', 'test_used_for_selection': False}
                validation_log.append({'domain': name, 'consumer': consumer_id, 'release': release, 'candidates': [{'id': i, 'validation_cost': v} for (i, _), v in zip(candidates, costs)], 'selected': mid})
                probs = model_q(model, allcases, runs, release, labels, floor)
                results = _execute_cases(allcases, probs, runs, domain, consumer)
                for c, q, result in zip(allcases, probs, results):
                    run = runs[c['case_id']]
                    e = c['evaluator_only']
                    episode = {'case_id': c['case_id'], 'domain_id': name, 'consumer_id': consumer_id, 'release_id': release, 'released_payload': _payload(c, run, release), 'visible_transcript': result['trace'], 'evaluator_only': {'split': e['split'], 'lineage_id': e['lineage_id'], 'base_id': e['base_id'], 'resampling_cluster': e[ds.engine.get('resampling_unit', 'lineage_id')], 'cost': result['cost'], 'final_correct': result['final_correct'], 'automated': result['automated'], 'used_check': result['used_check'], 'used_review': result['used_review'], 'native_valid': run['status'] == 'completed', 'posterior': q.tolist(), 'utility_policy_key': modelkey, 'policy_selection_split': 'validation'}}
                    episodes.append(episode)
                if consumer_id == 'selective_routing':
                    test = [(c, q) for c, q in zip(allcases, probs) if c['evaluator_only']['split'] == 'test' and runs[c['case_id']]['status'] == 'completed']
                    for threshold in np.linspace(0, 1, 21):
                        accepted = [(c, q) for c, q in test if q.max() >= threshold]
                        mistakes = sum((labels[int(q.argmax())] != c['evaluator_only']['truth']['T'] for c, q in accepted))
                        risk_curves.append({'domain': name, 'release': release, 'threshold': float(threshold), 'coverage': len(accepted) / len(test) if test else None, 'risk': mistakes / len(accepted) if accepted else None, 'n_valid_test': len(test), 'n_automated': len(accepted), 'scope': 'fixed grid on valid responses, descriptive; no test-selected operating point'})
        full = [e for e in episodes if e['domain_id'] == name and e['release_id'] == 'full_native']
        for ep in full:
            action = copy.deepcopy(ep)
            action['release_id'] = 'backend_action_only'
            action['released_payload'] = {}
            action['evaluator_only']['utility_policy_key'] = ep['evaluator_only']['utility_policy_key']
            episodes.append(action)
        for consumer_id, consumer in ds.consumers.items():
            for c in allcases:
                T = c['evaluator_only']['truth']['T']
                e = c['evaluator_only']
                q = np.array([float(t == T) for t in labels])
                result = execute_plan(q, initial_plan(q, domain, consumer), domain, consumer, T, e['approved_check_answers'])
                episodes.append({'case_id': c['case_id'], 'domain_id': name, 'consumer_id': consumer_id, 'release_id': 'oracle_T_only', 'released_payload': {'choice': T}, 'visible_transcript': result['trace'], 'evaluator_only': {'split': e['split'], 'lineage_id': e['lineage_id'], 'base_id': e['base_id'], 'resampling_cluster': e[ds.engine.get('resampling_unit', 'lineage_id')], 'cost': result['cost'], 'final_correct': result['final_correct'], 'automated': result['automated'], 'used_check': result['used_check'], 'used_review': result['used_review'], 'native_valid': runs[c['case_id']]['status'] == 'completed', 'posterior': q.tolist(), 'utility_policy_key': 'EVALUATION_ORACLE_NOT_ASSUMED_DEPLOYABLE', 'policy_selection_split': 'none'}})
    return (episodes, {'selected': selected_models, 'validation_search': validation_log}, risk_curves, utility_baselines(ds, runs))

def utility_baselines(ds, runs):
    """Always native-choice and always trusted-review operating points."""
    rows = []
    for c in ds.cases:
        if c['evaluator_only']['split'] != 'test':
            continue
        d = ds.domains[c['domain_id']]
        T = c['evaluator_only']['truth']['T']
        r = runs[c['case_id']]
        for kind, con in ds.consumers.items():
            for baseline in ['always_native_choice', 'always_trusted_review']:
                if baseline == 'always_trusted_review':
                    loss = 2.0
                    correct = True
                elif r['status'] != 'completed':
                    loss = 2.0
                    correct = True
                else:
                    route = r['response']['choice']
                    loss = cost_vector(con, d, T)['ROUTE:' + route]
                    correct = route == T
                rows.append({'case_id': c['case_id'], 'domain': c['domain_id'], 'consumer': kind, 'baseline': baseline, 'cost': loss, 'final_correct': correct, 'lineage_id': c['evaluator_only']['lineage_id']})
    return rows

def summarize_utility(ds, episodes: list[dict]) -> list[dict]:
    buckets = defaultdict(list)
    for e in episodes:
        if e['evaluator_only']['split'] != 'test':
            continue
        buckets[e['domain_id'], e['consumer_id'], e['release_id']].append(e)
        buckets['ALL', e['consumer_id'], e['release_id']].append(e)
    out = []
    for (domain, consumer, release), rows in buckets.items():
        ev = [r['evaluator_only'] for r in rows]
        auto = [x for x in ev if x['automated']]
        ci = cluster_mean_ci([x['cost'] for x in ev], [x['resampling_cluster'] for x in ev], draws=ds.engine['bootstrap_resamples'], seed=ds.engine['seed'], level=ds.engine['confidence_level'])
        out.append({'domain': domain, 'consumer': consumer, 'release': release, 'cost': ci, 'final_accuracy': float(np.mean([x['final_correct'] for x in ev])), 'automation_coverage': len(auto) / len(ev), 'risk_among_automated': float(np.mean([not x['final_correct'] for x in auto])) if auto else None, 'review_rate': float(np.mean([x['used_review'] for x in ev])), 'check_rate': float(np.mean([x['used_check'] for x in ev])), 'native_valid_fraction': float(np.mean([x['native_valid'] for x in ev])), 'idealized_review_warning': True})
    return out
