"""Conditional closed-set secret inference, using only declared observations.

S labels are supplied to fitting/scoring separately. At inference, attack
features contain X,B,T, the *entire hypothesis set*, and released fields/trace;
never S, another world's response, case IDs, eligibility, or private reports.
"""
from __future__ import annotations
from collections import defaultdict
import numpy as np
from sklearn.metrics import roc_auc_score
from .features import privacy_features
from .io import require, digest
from .learning import fit_prior, fit_linear, predict, nll
from .releases import PROJECTIONS, release_fields, order_from_case
from .statistics import cluster_mean_ci, paired_difference

def _target(c: dict) -> int:
    e = c['evaluator_only']
    return e['hypotheses'].index(e['truth']['S'])

def _features(cases, observations, variant):
    return [privacy_features(c['consumer_view'], observations[c['case_id']]['payload'], c['evaluator_only']['truth']['T'], c['evaluator_only']['hypotheses'], observations[c['case_id']].get('trace'), variant) for c in cases]

def endpoint_observations(ds, runs, episodes):
    endpoints = {}
    for release in PROJECTIONS:
        endpoints['payload', 'none', release] = {c['case_id']: {'payload': release_fields(runs[c['case_id']]['response'], release, order_from_case(c), runs[c['case_id']]['status']), 'trace': None} for c in ds.cases}
    endpoints['payload', 'none', 'oracle_T_only'] = {c['case_id']: {'payload': {'choice': c['evaluator_only']['truth']['T']}, 'trace': None} for c in ds.cases}
    for e in episodes:
        key = ('episode', e['consumer_id'], e['release_id'])
        endpoints.setdefault(key, {})[e['case_id']] = {'payload': e['released_payload'], 'trace': e['visible_transcript']}
    return endpoints

def fit_and_evaluate_privacy(ds, runs: dict, episodes: list[dict]):
    predictions = []
    models = {}
    search = []
    endpoints = endpoint_observations(ds, runs, episodes)
    all_ids = {c['case_id'] for c in ds.cases}
    for (scope, consumer, release), obs in endpoints.items():
        require(set(obs) == all_ids, 'Observation endpoint has missing or duplicate case linkage')
        for domain in ds.domains:
            dc = [c for c in ds.cases if c['domain_id'] == domain]
            tr = [c for c in dc if c['evaluator_only']['split'] == 'train']
            va = [c for c in dc if c['evaluator_only']['split'] == 'validation']
            te = [c for c in dc if c['evaluator_only']['split'] == 'test']
            require(tr and va and te, 'Each privacy endpoint needs all three splits')
            ytr = [_target(c) for c in tr]
            yva = [_target(c) for c in va]
            require(set(ytr) == set(yva) == {0, 1}, 'Both closed-set secret labels are required in train/validation')
            candidates = [('prior', 'linear', fit_prior(ytr, [0, 1]))]
            validation_cache = {}
            for variant in ds.engine['privacy_feature_families']:
                ft = _features(tr, obs, variant)
                fv = _features(va, obs, variant)
                validation_cache[variant] = fv
                for C in ds.engine['privacy_regularization_C']:
                    candidates.append((f'{variant}_C={C}', variant, fit_linear(ft, ytr, [0, 1], C)))
            records = []
            for cid, variant, model in candidates:
                pv = predict(model, validation_cache.get(variant) or _features(va, obs, variant), [0, 1])
                pred = pv.argmax(axis=1)
                acc = float(np.mean(pred == np.array(yva)))
                records.append({'candidate': cid, 'validation_accuracy': acc, 'validation_log_loss': nll(yva, pv, [0, 1])})
            best = min(range(len(candidates)), key=lambda i: (-records[i]['validation_accuracy'], records[i]['validation_log_loss'], i))
            cid, variant, model = candidates[best]
            key = '/'.join([domain, scope, consumer, release])
            models[key] = {'candidate': cid, 'feature_family': variant, 'model': model, 'train_case_ids_sha256': digest(sorted((c['case_id'] for c in tr))), 'validation_case_ids_sha256': digest(sorted((c['case_id'] for c in va))), 'hypothesis_order': 'sorted_by_public_hypothesis_name', 'selection': 'validation accuracy then log-loss then fixed candidate order', 'test_used_for_selection': False}
            search.append({'model_key': key, 'selected': cid, 'candidates': records})
            p = predict(model, _features(te, obs, variant), [0, 1])
            pred = p.argmax(axis=1)
            for c, q, guess in zip(te, p, pred):
                e = c['evaluator_only']
                y = _target(c)
                predictions.append({'case_id': c['case_id'], 'domain': domain, 'scope': scope, 'consumer': consumer, 'release': release, 'lineage_id': e['lineage_id'], 'base_id': e['base_id'], 'resampling_cluster': e[ds.engine.get('resampling_unit', 'lineage_id')], 'truth_T': e['truth']['T'], 'true_secret_index': y, 'predicted_secret_index': int(guess), 'probability_secret_1': float(q[1]), 'correct': int(guess == y), 'model_key': key, 'role': 'evaluator_only_predictions_not_attack_features'})
    return (predictions, {'selected': models, 'validation_search': search}, endpoints)

def population_masks(ds, runs):
    pairs = defaultdict(list)
    for c in ds.cases:
        pairs[c['evaluator_only']['base_id']].append(c)
    valid = set()
    correct = set()
    for worlds in pairs.values():
        if all((runs[c['case_id']]['status'] == 'completed' for c in worlds)):
            valid.update((c['case_id'] for c in worlds))
            if all((runs[c['case_id']]['response']['choice'] == c['evaluator_only']['truth']['T'] for c in worlds)):
                correct.update((c['case_id'] for c in worlds))
    return {'all_worlds': set(ds.case_map), 'valid_pairs': valid, 'both_choices_correct': correct}

def summarize_privacy(ds, runs, predictions):
    masks = population_masks(ds, runs)
    baseline = {p['case_id']: p for p in predictions if p['scope'] == 'payload' and p['release'] == 'public_only'}
    buckets = defaultdict(list)
    for p in predictions:
        buckets[p['domain'], p['scope'], p['consumer'], p['release']].append(p)
        buckets['ALL', p['scope'], p['consumer'], p['release']].append(p)
    out = []
    kw = {'draws': ds.engine['bootstrap_resamples'], 'seed': ds.engine['seed'], 'level': ds.engine['confidence_level']}
    for (domain, scope, consumer, release), allrows in buckets.items():
        for subset, ids in masks.items():
            rows = [r for r in allrows if r['case_id'] in ids]
            clusters = [r['resampling_cluster'] for r in rows]
            successes = [r['correct'] for r in rows]
            base = [baseline[r['case_id']]['correct'] for r in rows]
            acc = cluster_mean_ci(successes, clusters, **kw)
            gain = paired_difference(successes, base, clusters, **kw)
            auc = None
            if rows and len({r['true_secret_index'] for r in rows}) == 2:
                auc = float(roc_auc_score([r['true_secret_index'] for r in rows], [r['probability_secret_1'] for r in rows]))
            out.append({'domain': domain, 'scope': scope, 'consumer': consumer, 'release': release, 'subset': subset, 'accuracy': acc, 'gain_over_X_B_T': gain, 'auc': auc, 'pair_coverage': len(rows) / len(allrows) if allrows else 0, 'n_pairs': len(set((r['base_id'] for r in rows))), 'interpretation': 'finite closed-set decoding, not mutual information or a safety certificate'})
    lookup = {(p['case_id'], p['scope'], p['consumer'], p['release']): p for p in predictions}
    contrasts = [('choice', 'public_only', 'label_increment'), ('choice_confidence', 'choice', 'confidence_after_choice'), ('full_probabilities', 'choice', 'probabilities_after_choice'), ('full_native', 'choice_confidence', 'probabilities_after_choice_and_confidence'), ('full_native', 'full_probabilities', 'native_confidence_after_probabilities')]
    decomposition = []
    for a, b, title in contrasts:
        for domain in list(ds.domains) + ['ALL']:
            rows = [p for p in predictions if p['scope'] == 'payload' and p['release'] == a and (domain == 'ALL' or p['domain'] == domain)]
            delta = [p['correct'] - lookup[p['case_id'], 'payload', 'none', b]['correct'] for p in rows]
            decomposition.append({'domain': domain, 'contrast': title, 'richer': a, 'poorer': b, 'accuracy_difference': cluster_mean_ci(delta, [p['resampling_cluster'] for p in rows], **kw), 'scope': 'all_worlds; separately validation-selected decoders; not an additive MI decomposition; negative differences retained'})
    return (out, decomposition)

def paired_sensitivity(ds, runs) -> dict:
    """Single-response descriptive movement. No statistical secret-dependence claim."""
    pairs = defaultdict(list)
    for c in ds.cases:
        pairs[c['evaluator_only']['base_id']].append(c)
    rows = []
    for bid, worlds in pairs.items():
        if not all((runs[c['case_id']]['status'] == 'completed' for c in worlds)):
            continue
        a, b = worlds
        ra, rb = (runs[a['case_id']]['response'], runs[b['case_id']]['response'])
        ids = order_from_case(a)
        shifts = [abs(ra['probabilities'][k] - rb['probabilities'][k]) for k in ids]
        rows.append({'base_id': bid, 'domain': a['domain_id'], 'split': a['evaluator_only']['split'], 'lineage_id': a['evaluator_only']['lineage_id'], 'probability_L1': float(sum(shifts)), 'probability_Linf': float(max(shifts)), 'confidence_abs_shift': float(abs(ra['confidence'] - rb['confidence'])), 'choice_changed': ra['choice'] != rb['choice']})
    return {'rows': rows, 'scope': 'descriptive paired native response differences; same-input repeats required to separate serving noise', 'is_mutual_information': False, 'is_distribution_equality_test': False}
