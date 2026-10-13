"""Train-only secret-shared decoders for the declared known-hypothesis audit.

No case IDs, private reports, realized secrets or raw responses enter predict.
Models and preprocessing are portable JSON; the historical engine is unchanged.
"""
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, softmax
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import StandardScaler
from tuplebench.core.features import public_features, payload_features, privacy_features
from tuplebench.core.io import require
from tuplebench.core.learning_extended_v1 import fit_prior, fit_linear, predict as old_predict, nll
from tuplebench.core.tree_attack import fit_forest, DEFAULT_CANDIDATES
RELEASES = ['public_only', 'choice', 'choice_confidence', 'full_probabilities', 'full_native']
SUBVIEWS = {'public_only': ['public_only'], 'choice': ['public_only', 'choice'], 'choice_confidence': ['public_only', 'choice', 'choice_confidence'], 'full_probabilities': ['public_only', 'choice', 'full_probabilities'], 'full_native': RELEASES}
GRID = dict(C=[0.1, 1.0, 10.0], representations=['raw', 'log_ratio', 'residual'], templates=[['diagonal', 0.0], ['diagonal', 2.0], ['shared', 2.0]], covariance_floor=0.05, covariance_diagonal_shrinkage=0.75, log_epsilon=0.0001, forests=DEFAULT_CANDIDATES)

def checked(row):
    require(set(row) == {'consumer_view', 'payload', 'T', 'hypotheses'}, 'Undeclared attack input')
    require(len(row['hypotheses']) == 2 and len(set(row['hypotheses'])) == 2, 'Two distinct hypotheses required')
    require(all((isinstance(s, str) for s in row['hypotheses'])), 'Secret identities must be strings')
    payload_features(row['payload'])
    public_features(row['consumer_view'])
    return row

def project(row, release):
    checked(row)
    require(release in RELEASES, 'Unknown release')
    keys = {'public_only': [], 'choice': ['choice'], 'choice_confidence': ['choice', 'confidence'], 'full_probabilities': ['choice', 'probabilities'], 'full_native': ['choice', 'probabilities', 'confidence']}[release]
    return dict(row, payload={k: v for k, v in row['payload'].items() if k in keys + ['execution_status']})

def features(row, representation, include_public=True):
    checked(row)
    require(representation in GRID['representations'], 'Unknown representation')
    p = row['payload']
    f = payload_features(p)
    if include_public:
        f.update(public_features(row['consumer_view']))
    f['allowed_T=' + row['T']] = 1.0
    if representation == 'raw':
        return f
    probs = p.get('probabilities', {})
    names = sorted((c['id'] for c in row['consumer_view']['public_task']['candidates']))
    require(row['T'] in names and set(probs) <= set(names), 'Unexpected coarse probability identities')
    for name, value in probs.items():
        f['zero:p:' + name] = float(value == 0)
    if representation == 'log_ratio':
        ref = names[0]
        for name in names[1:]:
            if ref in probs and name in probs:
                f['logratio:' + name + '/' + ref] = float(np.log(probs[name] + GRID['log_epsilon']) - np.log(probs[ref] + GRID['log_epsilon']))
                f['observed:logratio:' + name] = 1.0
    elif row['T'] in probs:
        mass = max(0.0, 1.0 - probs[row['T']])
        f['residual:mass'] = mass
        f['observed:residual_mass'] = 1.0
        f['residual:zero_mass'] = float(mass == 0)
        f['observed:residual_shape'] = float(mass > 0)
        if mass > 0:
            for name in names:
                if name != row['T'] and name in probs:
                    f['residual:shape:' + name] = probs[name] / mass
                    f['observed:residual_shape:' + name] = 1.0
    return f

def encode_train(rows, representation, include_public):
    vec = DictVectorizer(sparse=False)
    X = vec.fit_transform([features(r, representation, include_public) for r in rows])
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    return (X, dict(names=vec.get_feature_names_out().tolist(), mean=scaler.mean_.tolist(), scale=scaler.scale_.tolist(), representation=representation, include_public=include_public))

def encode(model, rows):
    enc = model['encoder']
    lookup = {k: i for i, k in enumerate(enc['names'])}
    X = np.zeros((len(rows), len(lookup)))
    for i, row in enumerate(rows):
        for k, v in features(row, enc['representation'], enc['include_public']).items():
            if k in lookup:
                X[i, lookup[k]] = v
    return (X - np.array(enc['mean'])) / np.array(enc['scale'])

def fit_shared(rows, labels, representation, C):
    """u(s,x)=w_s.z(x)+b_s; pairwise logistic loss shares w_s across pairs."""
    require(len(rows) == len(labels) > 0 and set(labels) <= {0, 1}, 'Invalid training labels')
    X, enc = encode_train(rows, representation, True)
    X = np.column_stack([X, np.ones(len(X))])
    secrets = sorted({s for r in rows for s in r['hypotheses']})
    ids = {s: i for i, s in enumerate(secrets)}
    a = np.array([ids[r['hypotheses'][0]] for r in rows])
    b = np.array([ids[r['hypotheses'][1]] for r in rows])
    y = np.array(labels)
    shape = (len(secrets), X.shape[1])
    penalty = 1.0 / (C * len(rows))

    def objective(flat):
        W = flat.reshape(shape)
        z = np.sum((W[b] - W[a]) * X, axis=1)
        loss = np.mean(np.logaddexp(0, z) - y * z) + 0.5 * penalty * np.sum(W * W)
        residual = (expit(z) - y)[:, None] * X / len(y)
        grad = penalty * W.copy()
        np.add.at(grad, b, residual)
        np.add.at(grad, a, -residual)
        return (loss, grad.ravel())
    result = minimize(objective, np.zeros(np.prod(shape)), jac=True, method='L-BFGS-B', options=dict(maxiter=2000, gtol=1e-07, ftol=1e-12))
    require(result.success and np.all(np.isfinite(result.x)), 'Shared scorer did not converge')
    return dict(kind='shared_secret', encoder=enc, secrets=secrets, weights=result.x.reshape(shape).tolist(), C=C, n_training_rows=len(rows), fit_scope='train_only', final_loss=float(result.fun))

def fit_template(rows, labels, representation, covariance, mean_strength):
    """Per-secret means with per-T shared covariance and shrinkage to the T mean."""
    X, enc = encode_train(rows, representation, False)
    groups = {}
    for T in sorted({r['T'] for r in rows}):
        indices = [i for i, r in enumerate(rows) if r['T'] == T]
        parent = X[indices].mean(axis=0)
        means = {}
        residual = []
        secrets = sorted({rows[i]['hypotheses'][labels[i]] for i in indices})
        for s in secrets:
            selected = [i for i in indices if rows[i]['hypotheses'][labels[i]] == s]
            mu = X[selected].mean(axis=0)
            means[s] = ((len(selected) * mu + mean_strength * parent) / (len(selected) + mean_strength)).tolist()
            residual.extend(X[selected] - mu)
        E = np.array(residual)
        cov = E.T @ E / max(1, len(E) - len(secrets))
        diagonal = np.diag(np.diag(cov))
        if covariance == 'diagonal':
            cov = diagonal
        else:
            require(covariance == 'shared', 'Unknown covariance')
            shrink = GRID['covariance_diagonal_shrinkage']
            cov = shrink * diagonal + (1 - shrink) * cov
        cov += np.eye(X.shape[1]) * GRID['covariance_floor']
        groups[T] = dict(means=means, parent_mean=parent.tolist(), precision=np.linalg.inv(cov).tolist())
    return dict(kind='secret_template', encoder=enc, groups=groups, covariance=covariance, mean_strength=mean_strength, equal_pair_priors=True, n_training_rows=len(rows), fit_scope='train_only')

def legacy_features(rows, variant):
    return [privacy_features(r['consumer_view'], r['payload'], r['T'], sorted(r['hypotheses']), variant=variant) for r in rows]

def predict(model, rows):
    for row in rows:
        checked(row)
    kind = model['kind']
    if kind == 'ensemble':
        return np.mean([predict(m, rows) for m in model['members']], axis=0)
    if kind == 'view':
        return predict(model['model'], [project(r, model['release']) for r in rows])
    if kind == 'legacy':
        q = old_predict(model['model'], legacy_features(rows, model['variant']), [0, 1])
        return np.array([v if r['hypotheses'] == sorted(r['hypotheses']) else v[::-1] for r, v in zip(rows, q)])
    X = encode(model, rows)
    scores = np.zeros((len(rows), 2))
    if kind == 'shared_secret':
        X = np.column_stack([X, np.ones(len(X))])
        lookup = dict(zip(model['secrets'], model['weights']))
        for i, row in enumerate(rows):
            for j, s in enumerate(row['hypotheses']):
                scores[i, j] = X[i] @ np.array(lookup.get(s, np.zeros(X.shape[1])))
    else:
        require(kind == 'secret_template', 'Unknown strong attacker kind')
        for i, row in enumerate(rows):
            group = model['groups'].get(row['T'])
            if group is None:
                continue
            for j, s in enumerate(row['hypotheses']):
                delta = X[i] - np.array(group['means'].get(s, group['parent_mean']))
                scores[i, j] = -0.5 * delta @ np.array(group['precision']) @ delta
    q = softmax(scores, axis=1)
    require(np.all(np.isfinite(q)) and np.allclose(q.sum(axis=1), 1), 'Invalid attack posterior')
    return q

def guesses(q, rows):
    return np.array([min(range(2), key=lambda j: (-v[j], r['hypotheses'][j])) for v, r in zip(q, rows)])

def metric(model, rows, labels):
    q = predict(model, rows)
    return dict(accuracy=float(np.mean(guesses(q, rows) == labels)), log_loss=nll(labels, q, [0, 1]))

def choose(records):
    return min(range(len(records)), key=lambda i: (-records[i]['validation_accuracy'], records[i]['validation_log_loss'], i))

def fit_bank(tr, ytr, va, yva, release, seed):
    require(all((r['hypotheses'] == sorted(r['hypotheses']) for r in tr + va)), 'Canonical training hypotheses required')
    tr = [project(r, release) for r in tr]
    va = [project(r, release) for r in va]
    candidates = [('prior', dict(kind='legacy', variant='linear', model=fit_prior(ytr, [0, 1])))]
    for variant in ['linear', 'quadratic']:
        ft = legacy_features(tr, variant)
        for C in GRID['C']:
            candidates.append((f'{variant}_C={C}', dict(kind='legacy', variant=variant, model=fit_linear(ft, ytr, [0, 1], C))))
    for i, params in enumerate(GRID['forests']):
        candidates.append((f'random_forest_{i}', dict(kind='legacy', variant='linear', model=fit_forest(legacy_features(tr, 'linear'), ytr, [0, 1], params, seed))))
    representations = GRID['representations'] if release in ('full_probabilities', 'full_native') else ['raw']
    for representation in representations:
        for C in GRID['C']:
            candidates.append((f'shared/{representation}/C={C}', fit_shared(tr, ytr, representation, C)))
        for covariance, strength in GRID['templates']:
            candidates.append((f'template/{representation}/{covariance}/mean={strength}', fit_template(tr, ytr, representation, covariance, strength)))
    records = []
    for name, model in candidates:
        scores = metric(model, va, yva)
        records.append(dict(candidate=name, validation_accuracy=scores['accuracy'], validation_log_loss=scores['log_loss']))
    families = []
    for prefix in ['legacy', 'shared_secret', 'secret_template']:
        ids = [i for i, (_, m) in enumerate(candidates) if m['kind'] == prefix]
        winner = min(ids, key=lambda i: (-records[i]['validation_accuracy'], records[i]['validation_log_loss'], i))
        families.append(candidates[winner][1])
    ensemble = dict(kind='ensemble', members=families)
    scores = metric(ensemble, va, yva)
    candidates.append(('family_mean', ensemble))
    records.append(dict(candidate='family_mean', validation_accuracy=scores['accuracy'], validation_log_loss=scores['log_loss']))
    return (candidates, records)

def fit_selected(tr, ytr, va, yva, seed):
    banks = {}
    searches = {}
    result = {}
    for release in RELEASES:
        banks[release], searches[release] = fit_bank(tr, ytr, va, yva, release, seed)
        choices = []
        records = []
        for source in SUBVIEWS[release]:
            i = choose(searches[source])
            name, model = banks[source][i]
            choices.append(dict(kind='view', release=source, model=model))
            records.append(dict(searches[source][i], candidate=source + '/' + name))
        best = choose(records)
        result[release] = dict(model=choices[best], selected=records[best]['candidate'], validation_search=records, test_used_for_selection=False)
    return (result, searches, banks)
