"""Grouped OOF simplex stacking for the frozen, known-H privacy audit.

Only declared attacker views enter fits/predictions. Group metadata is used solely
to construct folds, never as a predictive feature. No held-out-fold selection.
"""
from collections import defaultdict, Counter
import numpy as np
from scipy.optimize import minimize
from tuplebench.core.io import require, digest
from . import strong_attacker_v1 as base
FOLD_SEED = 2026100804
PERMUTATION_SEEDS = list(range(2026100820, 2026100840))
LAMBDAS = [0.01, 0.1]
GROUP_KEYS = ['base_id', 'lineage_id', 'source_family_id', 'rendering_family_id']

def fit_base_bank(rows, labels, release, seed):
    """Fixed candidates only: no validation/heldout labels accepted by this API."""
    require(all((r['hypotheses'] == sorted(r['hypotheses']) for r in rows)), 'Canonical training H required')
    rows = [base.project(r, release) for r in rows]
    bank = [('prior', dict(kind='legacy', variant='linear', model=base.fit_prior(labels, [0, 1])))]
    for variant in ['linear', 'quadratic']:
        x = base.legacy_features(rows, variant)
        for C in base.GRID['C']:
            bank.append((f'{variant}_C={C}', dict(kind='legacy', variant=variant, model=base.fit_linear(x, labels, [0, 1], C))))
    for i, params in enumerate(base.GRID['forests']):
        bank.append((f'random_forest_{i}', dict(kind='legacy', variant='linear', model=base.fit_forest(base.legacy_features(rows, 'linear'), labels, [0, 1], params, seed))))
    reps = base.GRID['representations'] if release in ('full_probabilities', 'full_native') else ['raw']
    for rep in reps:
        for C in base.GRID['C']:
            bank.append((f'shared/{rep}/C={C}', base.fit_shared(rows, labels, rep, C)))
        for cov, strength in base.GRID['templates']:
            bank.append((f'template/{rep}/{cov}/mean={strength}', base.fit_template(rows, labels, rep, cov, strength)))
    return bank

def bank_predictions(bank, rows, release):
    rows = [base.project(r, release) for r in rows]
    q = np.stack([base.predict(m, rows) for _, m in bank], axis=1)
    require(np.isfinite(q).all() and (q >= 0).all() and np.allclose(q.sum(axis=2), 1), 'Invalid bank predictions')
    return q

def fold_fit(rows, labels, folds, fold, release, seed):
    require(len(rows) == len(labels) == len(folds) and set(folds) == set(range(4)), 'Invalid OOF partition')
    fit = [i for i, f in enumerate(folds) if f != fold]
    heldout = [i for i, f in enumerate(folds) if f == fold]
    bank = fit_base_bank([rows[i] for i in fit], [labels[i] for i in fit], release, seed)
    q = bank_predictions(bank, [rows[i] for i in heldout], release)
    return dict(fit_indices=fit, heldout_indices=heldout, names=[n for n, _ in bank], q=q.tolist(), model_sha256=[digest(m) for _, m in bank], encoder_sha256=[digest(m.get('encoder', m.get('model', {}))) for _, m in bank], no_heldout_selection=True)

def assemble_oof(parts, n):
    require(len(parts) == 4, 'All four folds required')
    names = parts[0]['names']
    q = np.full((n, len(names), 2), np.nan)
    seen = set()
    for part in parts:
        ids = part['heldout_indices']
        fit = part['fit_indices']
        require(part['names'] == names and part['no_heldout_selection'], 'OOF bank changed')
        require(set(fit).isdisjoint(ids) and set(fit) | set(ids) == set(range(n)), 'OOF overlap or omission')
        require(not seen.intersection(ids), 'OOF row predicted twice')
        q[ids] = np.asarray(part['q'])
        seen.update(ids)
    require(seen == set(range(n)) and np.isfinite(q).all(), 'Incomplete OOF predictions')
    return q

def fit_weights(q, labels, penalty):
    """Convex log loss + fixed L2 shrinkage to uniform on the simplex."""
    q = np.asarray(q, float)
    y = np.asarray(labels, int)
    require(q.ndim == 3 and q.shape[0] == len(y) and (q.shape[2] == 2) and (set(y) <= {0, 1}), 'Bad OOF tensor')
    require(np.isfinite(q).all() and (q >= 0).all() and np.allclose(q.sum(axis=2), 1), 'Bad OOF probabilities')
    require(penalty > 0, 'Positive predeclared shrinkage required')
    truth = q[np.arange(len(y)), :, y]
    uniform = np.full(q.shape[1], 1 / q.shape[1])

    def objective(w):
        mixed = truth @ w
        clipped = np.maximum(mixed, 1e-15)
        loss = -np.log(clipped).mean() + penalty * np.sum((w - uniform) ** 2)
        gradient = -(truth * (mixed > 1e-15)[:, None] / clipped[:, None]).mean(axis=0) + 2 * penalty * (w - uniform)
        return (loss, gradient)
    result = minimize(objective, uniform, jac=True, method='SLSQP', bounds=[(0.0, 1.0)] * len(uniform), constraints=[dict(type='eq', fun=lambda w: w.sum() - 1.0, jac=lambda w: np.ones_like(w))], options=dict(maxiter=2000, ftol=1e-12))
    require(result.success and np.isfinite(result.x).all(), 'Simplex optimization did not converge')
    require(result.x.min() >= -1e-09 and abs(result.x.sum() - 1) < 1e-08, 'Invalid simplex solution')
    w = np.maximum(result.x, 0)
    w /= w.sum()
    require(objective(w)[0] <= objective(uniform)[0] + 1e-09, 'Weights worse than feasible uniform solution')
    return dict(weights=w.tolist(), penalty=penalty, objective=float(objective(w)[0]), uniform_objective=float(objective(uniform)[0]), iterations=int(result.nit), fit_scope='train_oof_only')

def descriptor(source, name, members, weights=None):
    return dict(source=source, name=name, members=list(members), method='mean' if weights is None else 'weighted', weights=None if weights is None else list(weights))

def combine(desc, matrices):
    q = np.asarray(matrices[desc['source']])[:, desc['members'], :]
    if desc['method'] == 'mean':
        return np.mean(q, axis=1)
    require(desc['method'] == 'weighted', 'Unknown blend')
    w = np.asarray(desc['weights'])
    require(len(w) == q.shape[1] and (w >= 0).all() and np.isfinite(w).all() and (abs(w.sum() - 1) < 1e-09), 'Invalid blend weights')
    return np.einsum('nmc,m->nc', q, w)

def score(q, rows, labels):
    return dict(validation_accuracy=float(np.mean(base.guesses(q, rows) == labels)), validation_log_loss=base.nll(labels, q, [0, 1]))

def select(banks, validation_q, validation_rows, validation_labels, fitted_weights):
    """Reproduce v1 and select blends on validation only. No test API argument."""
    exact_v1 = {}
    options = {}
    searches = {}
    result = {}

    def choose(candidates):
        records = [dict(candidate=d['source'] + '/' + d['name'], **score(combine(d, validation_q), validation_rows, validation_labels)) for d in candidates]
        i = base.choose(records)
        return (candidates[i], records)
    for release in base.RELEASES:
        bank = banks[release]
        singles = [descriptor(release, name, [i]) for i, (name, _) in enumerate(bank)]
        families = []
        for kind in ['legacy', 'shared_secret', 'secret_template']:
            ids = [i for i, (_, m) in enumerate(bank) if m['kind'] == kind]
            best, _ = choose([singles[i] for i in ids])
            families.extend(best['members'])
        exact_v1[release], _ = choose(singles + [descriptor(release, 'family_mean', families)])
        members = list(range(len(bank)))
        options[release] = [descriptor(release, 'equal_all', members)] + [descriptor(release, 'oof_lambda=' + str(w['penalty']), members, w['weights']) for w in fitted_weights[release]]
        old, _ = choose([exact_v1[s] for s in base.SUBVIEWS[release]])
        equal, _ = choose([options[s][0] for s in base.SUBVIEWS[release]])
        learned, _ = choose([o for s in base.SUBVIEWS[release] for o in options[s][1:]])
        selected, search = choose([old] + [o for s in base.SUBVIEWS[release] for o in options[s]])
        result[release] = dict(v1=old, equal=equal, learned=learned, selected=selected)
        searches[release] = search
    return (result, searches)

def predict(banks, desc, rows):
    release = desc['source']
    subset = [banks[release][i] for i in desc['members']]
    q = bank_predictions(subset, rows, release)
    return combine(dict(desc, members=list(range(len(subset)))), {release: q})
