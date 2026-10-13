"""Executable strict and bounded-cost quantization."""
import numpy as np
from tuplebench.core.io import require
from tuplebench.core.contracts import class_ids
from tuplebench.core.releases import validate_response
from . import workflow as old

def loss_vectors(sig, domain, consumer):
    """All four counterfactual class losses; no realized T or answer is accepted."""
    sig = np.atleast_2d(np.asarray(sig, int))
    labels = class_ids(domain)
    result = np.zeros((len(sig), len(labels)))
    for i, (kind, route, check, yes_route, no_route) in enumerate(sig):
        if kind in (0, 2) and consumer['consumer_id'] == 'asymmetric_cost_routing':
            result[i] = consumer['cost_matrix_by_domain'][domain['domain_id']][len(labels) if kind == 2 else route]
        elif kind == 0:
            result[i] = consumer['wrong_cost'] * (np.arange(len(labels)) != route)
        elif kind == 1:
            result[i] = consumer['review_cost']
        else:
            require(kind == 3, 'Unexpected workflow signature')
            law = domain['public_checks'][check]['p_yes_given_T']
            yes = np.array([law[t] for t in labels])
            result[i] = consumer['check_cost'] + consumer['wrong_cost'] * (yes * (np.arange(len(labels)) != yes_route) + (1 - yes) * (np.arange(len(labels)) != no_route))
    return result

def cost_scale(domain, consumer):
    if consumer['consumer_id'] == 'asymmetric_cost_routing':
        return float(np.max(consumer['cost_matrix_by_domain'][domain['domain_id']]))
    return float(consumer['wrong_cost'])

def expanded_candidates(responses, labels):
    """Static grid plus G-only responses and pair midpoints supplied by caller."""
    values = list(map(tuple, old.simplex_grid(len(labels))))
    values += [tuple(old.response_values(r, labels)) for r in responses]
    unique = sorted(set(values), key=lambda x: (sum(((v - 1 / len(labels)) ** 2 for v in x[:-1])), x[-1], x[:-1]))
    return np.asarray(unique)

def feasibility(response, public_view, domain, policies, floor, candidates, fractions):
    """Exact signatures and worst-class budgets for all declared policies."""
    labels = class_ids(domain)
    value = old.response_values(response, labels)
    strict = np.ones(len(candidates), bool)
    budget = {a: np.ones(len(candidates), bool) for a in fractions}
    target_signatures = []
    for item in policies:
        model, consumer = (item['models']['full_native'], item['specification'])
        affine = old.affine_policy(model, public_view, response['choice'], labels)
        q = old.policy_q(affine, candidates, floor)
        before = old.scalar_signature(old.policy_q(affine, value, floor)[0], domain, consumer)
        after = old.signatures(q, domain, consumer)
        target_signatures.append(before.tolist())
        strict &= np.all(after == before, axis=1)
        increase = loss_vectors(after, domain, consumer) - loss_vectors(before, domain, consumer)[0]
        worst = increase.max(axis=1)
        for a in fractions:
            budget[a] &= worst <= a * cost_scale(domain, consumer) + 1e-10
    return (strict, budget, target_signatures)

def greedy_codebook(mask, pairs, paired_weight=0.0, max_codes=128):
    """Learn an ordered codebook using only G feasibility and G pairing.

    Objective is new coverage plus pairs assigned jointly to the SAME code.
    No neural/continuous adversarial guarantee is claimed. Deployed mechanism
    takes the first feasible code in this learned order and is deterministic.
    """
    mask = np.asarray(mask, bool)
    require(mask.ndim == 2, 'Invalid feasibility matrix')
    assigned = np.zeros(len(mask), bool)
    chosen = []
    pairs = np.asarray(pairs, int).reshape(-1, 2)
    for _ in range(min(max_codes, mask.shape[1])):
        score = mask[~assigned].sum(axis=0).astype(float)
        if len(pairs) and paired_weight:
            available = pairs[~assigned[pairs].any(axis=1)]
            if len(available):
                score += paired_weight * (mask[available[:, 0]] & mask[available[:, 1]]).sum(axis=0)
        if chosen:
            score[chosen] = -1
        k = int(np.argmax(score))
        if score[k] <= 0:
            break
        chosen.append(k)
        assigned |= mask[:, k]
        if assigned.all():
            break
    return dict(indices=chosen, train_covered=int(assigned.sum()), train_total=len(mask), pair_weight=paired_weight, max_codes=max_codes, training_objective='coverage_plus_joint_pair_assignment', deployment='deterministic_first_feasible', randomized_privacy_claim=False)

def release(response, public_view, domain, policies, floor, codes, fraction=None):
    """Single-world deployable interface; rejects evaluator metadata."""
    labels = class_ids(domain)
    old.response_values(response, labels)
    require(set(public_view) == {'recipient_role', 'public', 'public_task', 'downstream_contexts'}, 'Public view only')
    codes = np.asarray(codes, float)
    if not len(codes):
        return (dict(response), dict(fallback=True, selected_code=None))
    strict, budgets, _ = feasibility(response, public_view, domain, policies, floor, codes, [] if fraction is None else [fraction])
    valid = strict if fraction is None else budgets[fraction]
    for k in np.flatnonzero(valid):
        result = dict(choice=response['choice'], probabilities=dict(zip(labels, codes[k, :-1].tolist())), confidence=float(codes[k, -1]))
        validate_response(result, labels)
        if verify_release(response, result, public_view, domain, policies, floor, fraction)['passed']:
            return (result, dict(fallback=False, selected_code=int(k)))
    return (dict(response), dict(fallback=True, selected_code=None))

def verify_release(before, after, public_view, domain, policies, floor, fraction):
    labels = class_ids(domain)
    b = old.response_values(before, labels)
    a = old.response_values(after, labels)
    require(before['choice'] == after['choice'], 'Native choice changed')
    same = True
    largest = -float('inf')
    passed = True
    for item in policies:
        consumer = item['specification']
        affine = old.affine_policy(item['models']['full_native'], public_view, before['choice'], labels)
        sb = old.scalar_signature(old.policy_q(affine, b, floor)[0], domain, consumer)
        sa = old.scalar_signature(old.policy_q(affine, a, floor)[0], domain, consumer)
        equal = bool(np.array_equal(sb, sa))
        same &= equal
        increase = loss_vectors(sa, domain, consumer)[0] - loss_vectors(sb, domain, consumer)[0]
        scale = cost_scale(domain, consumer)
        largest = max(largest, float(increase.max() / scale))
        passed &= equal if fraction is None else bool(np.all(increase <= fraction * scale + 1e-10))
    return dict(passed=bool(passed), same_signature=bool(same), max_normalized_cost_increase=largest)

def signature_features(rows, keys, include_choice=False):
    """Declared observable categorical signatures, never a raw posterior."""
    result = {}
    names = ('kind', 'route', 'check', 'yes_route', 'no_route')
    for key, row in zip(keys, rows):
        for name, value in zip(names, row):
            result[f'workflow:{key}:{name}={int(value)}'] = 1.0
    return result
