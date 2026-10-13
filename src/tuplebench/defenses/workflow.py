"""Cached output defenses. Inference accepts public views and response fields only.

Derived scores are deliberately NOT native model responses. All learned tensors
are portable JSON. No dataset, case identifier, T/S, H or paired test response is
accepted by a deployable transformation.
"""
from collections import defaultdict
import importlib.util
from pathlib import Path
import numpy as np
from scipy.special import softmax, expit
from tuplebench.core.contracts import class_ids
from tuplebench.core.features import utility_features, public_features, payload_features, TRACE_KEYS
from tuplebench.core.io import require
from tuplebench.core.learning import smooth_posterior
from tuplebench.core.releases import validate_response
from tuplebench.core.utility import initial_plan

def quantize(response, labels, denominator=20):
    """Largest-remainder simplex quantization, stable candidate-order tie break."""
    require(set(response) == {'choice', 'probabilities', 'confidence'}, 'Only declared response fields may enter defense')
    validate_response(response, labels)
    v = np.array([response['probabilities'][t] for t in labels]) * denominator
    counts = np.floor(v).astype(int)
    remaining = denominator - int(counts.sum())
    require(0 <= remaining <= len(labels), 'Unexpected quantization remainder')
    for j in sorted(range(len(labels)), key=lambda j: (-(v[j] - counts[j]), j))[:remaining]:
        counts[j] += 1
    result = dict(choice=response['choice'], probabilities=dict(zip(labels, (counts / denominator).tolist())), confidence=float(np.floor(response['confidence'] * denominator + 0.5) / denominator))
    validate_response(result, labels)
    return result

def simplex_grid(k=4, denominator=20):

    def compositions(total, count):
        if count == 1:
            yield (total,)
        else:
            for value in range(total + 1):
                for rest in compositions(total - value, count - 1):
                    yield ((value,) + rest)
    rows = [tuple((v / denominator for v in p)) + (confidence,) for p in compositions(denominator, k) for confidence in (0.0, 0.5, 1.0)]
    rows.sort(key=lambda r: (sum(((x - 1 / k) ** 2 for x in r[:-1])), r[-1], r[:-1]))
    return np.array(rows)

def affine_policy(model, public_view, choice, labels):
    """Exactly reconstruct the saved logistic policy's raw-score dependence."""
    if model['kind'] == 'native_distribution':
        return dict(kind='native_distribution')
    require(model['kind'] in ('prior', 'logistic'), 'Unsupported frozen utility model')
    if model['kind'] == 'prior':
        return dict(kind='prior', q=model['probabilities'], classes=model['classes'])
    names = model['feature_names']
    index = {n: i for i, n in enumerate(names)}
    x = np.zeros(len(names))
    fields = dict(choice=choice, probabilities={t: 0.0 for t in labels}, confidence=0.0)
    for name, value in utility_features(public_view, fields).items():
        if name in index:
            x[index[name]] = value
    coef = np.asarray(model['coef'])
    scale = np.asarray(model['scale'])
    offset = coef @ ((x - np.asarray(model['mean'])) / scale) + np.asarray(model['intercept'])
    slopes = np.zeros((len(model['classes']), len(labels) + 1))
    for j, name in enumerate(['release:p:' + t for t in labels] + ['release:confidence']):
        if name in index:
            slopes[:, j] = coef[:, index[name]] / scale[index[name]]
    require(len(model['classes']) == len(labels) and len(labels) > 2, 'This experiment uses four-class policies')
    order = [model['classes'].index(t) for t in labels]
    return dict(kind='logistic', offset=offset[order].tolist(), slopes=slopes[order].tolist())

def policy_q(affine, values, floor):
    values = np.atleast_2d(np.asarray(values, float))
    k = values.shape[1] - 1
    if affine['kind'] == 'native_distribution':
        q = values[:, :k]
    elif affine['kind'] == 'prior':
        q = np.tile(affine['q'], (len(values), 1))
    else:
        q = softmax(values @ np.asarray(affine['slopes']).T + np.asarray(affine['offset']), axis=1)
    return (q + floor) / (q.sum(axis=1, keepdims=True) + floor * k)

def signatures(q, domain, consumer):
    """Initial action, route, check ID and final routes for BOTH check answers.

    Uses the exact engine's strict inequalities and candidate-order ties. No
    realized T or check result is needed to compute this all-branches signature.
    """
    q = np.atleast_2d(q)
    n, k = q.shape
    kind = consumer['consumer_id']
    out = np.full((n, 5), -1, int)
    route = q.argmax(axis=1)
    out[:, 0] = 0
    out[:, 1] = route
    if kind == 'asymmetric_cost_routing':
        cost = q @ np.asarray(consumer['cost_matrix_by_domain'][domain['domain_id']]).T
        action = cost.argmin(axis=1)
        out[:, 0] = np.where(action == k, 2, 0)
        out[:, 1] = np.where(action == k, -1, action)
    else:
        stop = consumer['wrong_cost'] * (1 - q[np.arange(n), route])
        if kind == 'selective_routing':
            review = consumer['review_cost'] < stop
            out[review, 0] = 1
            out[review, 1] = -1
        else:
            require(kind == 'one_step_clarification', 'Unknown consumer')
            for ci, check in enumerate(domain['public_checks']):
                yes = np.array([check['p_yes_given_T'][t] for t in class_ids(domain)])
                y, no = (q * yes, q * (1 - yes))
                expected = consumer['check_cost'] + consumer['wrong_cost'] * (y.sum(axis=1) - y.max(axis=1) + no.sum(axis=1) - no.max(axis=1))
                take = expected < stop - 1e-12
                out[take, 0] = 3
                out[take, 1] = -1
                out[take, 2] = ci
                out[take, 3] = y[take].argmax(axis=1)
                out[take, 4] = no[take].argmax(axis=1)
                stop = np.where(take, expected, stop)
    return out

def scalar_signature(q, domain, consumer):
    """Independent engine reference, including both future check answers."""
    labels = class_ids(domain)
    plan = initial_plan(q, domain, consumer)
    out = np.full(5, -1, int)
    out[0] = {'ROUTE': 0, 'REVIEW': 1, 'GENERAL_INSPECTION': 2, 'CHECK': 3}[plan['kind']]
    if plan['kind'] == 'ROUTE':
        out[1] = labels.index(plan['route'])
    if plan['kind'] == 'CHECK':
        j = next((i for i, c in enumerate(domain['public_checks']) if c['check_id'] == plan['check_id']))
        law = domain['public_checks'][j]
        yes = np.array([law['p_yes_given_T'][t] for t in labels])
        out[2:] = [j, int(np.argmax(q * yes)), int(np.argmax(q * (1 - yes)))]
    return out

def response_values(response, labels):
    require(set(response) == {'choice', 'probabilities', 'confidence'}, 'Only declared response fields may enter defense')
    validate_response(response, labels)
    return np.array([response['probabilities'][t] for t in labels] + [response['confidence']])

def network_input(responses, labels):
    values = np.array([response_values(r, labels) for r in responses])
    onehot = np.eye(len(labels))[[labels.index(r['choice']) for r in responses]]
    return np.column_stack([onehot, values])
