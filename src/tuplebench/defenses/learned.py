"""Learned quantizers and explicitly adapted score defenses."""
import re
import json
import numpy as np
from scipy.special import softmax, expit
from tuplebench.core.io import require
from tuplebench.core.contracts import class_ids
from . import quantizer as qz

def public_numbers(view):
    require(set(view) == {'recipient_role', 'public', 'public_task', 'downstream_contexts'}, 'Only public context is allowed')
    text = json.dumps(view, sort_keys=True, ensure_ascii=False)
    words = re.findall('[a-z]+', text.lower())
    return [min(len(text), 10000) / 10000, min(len(words), 2000) / 2000, min(len(set(words)), 1000) / 1000, min(sum((c.isdigit() for c in text)), 500) / 500, min(len(view['downstream_contexts']), 20) / 20, 1.0]

def selector_input(responses, views, labels):
    require(len(responses) == len(views), 'Input lengths differ')
    return np.column_stack([qz.old.network_input(responses, labels), np.array([public_numbers(v) for v in views])])

def init_mlp(width, hidden, out, rng):
    return dict(w1=rng.normal(0, 0.1, (width, hidden)), b1=np.zeros(hidden), w2=rng.normal(0, 0.1, (hidden, out)), b2=np.zeros(out))

def forward(net, x):
    h = np.tanh(x @ net['w1'] + net['b1'])
    return (h @ net['w2'] + net['b2'], (x, h))

def backward(net, cache, g):
    x, h = cache
    gh = g @ net['w2'].T * (1 - h * h)
    return dict(w1=x.T @ gh, b1=gh.sum(0), w2=h.T @ g, b2=g.sum(0))

class Adam:

    def __init__(self, net, lr):
        self.m = {k: np.zeros_like(v) for k, v in net.items()}
        self.v = {k: np.zeros_like(v) for k, v in net.items()}
        self.t = 0
        self.lr = lr

    def step(self, net, grad, decay=0.0001):
        self.t += 1
        for k in net:
            g = np.clip(grad[k] + decay * net[k], -5, 5)
            self.m[k] = 0.9 * self.m[k] + 0.1 * g
            self.v[k] = 0.999 * self.v[k] + 0.001 * g * g
            net[k] -= self.lr * (self.m[k] / (1 - 0.9 ** self.t)) / (np.sqrt(self.v[k] / (1 - 0.999 ** self.t)) + 1e-08)

def arrays(obj):
    return {k: np.asarray(v, float) for k, v in obj.items()}

def serial(net):
    return {k: v.tolist() for k, v in net.items()}

def pair_js(distribution, pairs):
    pairs = np.asarray(pairs, int)
    a, b = (distribution[pairs[:, 0]], distribution[pairs[:, 1]])
    mid = (a + b) / 2
    la = np.log(np.maximum(a, 1e-30) / np.maximum(mid, 1e-30))
    lb = np.log(np.maximum(b, 1e-30) / np.maximum(mid, 1e-30))
    loss = float(np.sum(a * la + b * lb) / (2 * len(pairs)))
    grad = np.zeros_like(distribution)
    np.add.at(grad, pairs[:, 0], la / (2 * len(pairs)))
    np.add.at(grad, pairs[:, 1], lb / (2 * len(pairs)))
    return (loss, grad)

def masked_distribution(logits, mask):
    require(np.all(np.any(mask, axis=1)), 'At least one feasible code is needed for training')
    return softmax(np.where(mask, logits, -1e+30), axis=1)

def fit_quantizer(responses, views, labels, T, H, y, pairs, codes, mask, seed, pair_weight, steps=600, keep=64):
    """G-only alternating expected-log-loss game, followed by hard gate pruning.

    Critic is a nonlinear 24-hidden shared-secret scorer plus secret/code
    intercepts. The selector maximizes critic cross entropy and adds paired JS.
    A global trainable log gate controls code preference/support. The published
    mechanism is deterministic argmax; stochastic training claims do not carry
    over to that mechanism. Its independent attackers are refit afterwards.
    """
    x = selector_input(responses, views, labels)
    n = len(x)
    m = len(codes)
    rng = np.random.default_rng(seed)
    secrets = sorted({s for h in H for s in h})
    si = {s: i for i, s in enumerate(secrets)}
    hi = np.array([[si[s] for s in h] for h in H])
    target = np.asarray(y, float)
    t = np.eye(len(labels))[[labels.index(t) for t in T]]
    common = np.column_stack([x[:, :len(labels)], x[:, -6:], t])
    f = np.concatenate([np.repeat(common[:, None, :], m, 1), np.tile(codes, (n, 1, 1))], axis=2)
    critic = init_mlp(f.shape[-1], 24, len(secrets), rng)
    critic['id'] = np.zeros((len(secrets), m))
    net = init_mlp(x.shape[1], 24, m, rng)
    net['gate'] = np.zeros(m)
    ao = Adam(net, 0.01)
    ac = Adam(critic, 0.01)
    history = []
    for step in range(steps):
        logits, cache = forward(net, x)
        logits += np.log(expit(net['gate']) + 1e-12)
        dist = masked_distribution(logits, mask)
        scores, cc = forward(critic, f.reshape(-1, f.shape[-1]))
        scores = scores.reshape(n, m, -1)
        diff = scores[np.arange(n)[:, None], np.arange(m)[None, :], hi[:, 1, None]] - scores[np.arange(n)[:, None], np.arange(m)[None, :], hi[:, 0, None]]
        diff += critic['id'][hi[:, 1]] - critic['id'][hi[:, 0]]
        loss = np.logaddexp(0, diff) - target[:, None] * diff
        residual = (expit(diff) - target[:, None]) * dist / n
        gs = np.zeros_like(scores)
        for i in range(n):
            gs[i, :, hi[i, 1]] += residual[i]
            gs[i, :, hi[i, 0]] -= residual[i]
        gc = backward(critic, cc, gs.reshape(-1, len(secrets)))
        gid = np.zeros_like(critic['id'])
        np.add.at(gid, hi[:, 1], residual)
        np.add.at(gid, hi[:, 0], -residual)
        gc['id'] = gid
        ac.step(critic, gc)
        js, gjs = pair_js(dist, pairs)
        gd = -loss / n + pair_weight * gjs
        gl = dist * (gd - (gd * dist).sum(1, keepdims=True))
        grad = backward(net, cache, gl)
        grad['gate'] = gl.sum(0) * (1 - expit(net['gate'])) + 0.002 * expit(net['gate']) * (1 - expit(net['gate'])) / m
        ao.step(net, grad)
        if step % 100 == 0 or step == steps - 1:
            history.append(dict(step=step, critic_cross_entropy=float(np.mean(np.sum(dist * loss, 1))), pair_JS=js, effective_gate_sum=float(expit(net['gate']).sum())))
    kept = sorted(np.argsort(-net['gate'], kind='stable')[:min(keep, m)].tolist())
    return dict(kind='masked_adversarial_quantizer', network=serial(net), code_indices=kept, codes=np.asarray(codes).tolist(), seed=seed, pair_weight=pair_weight, steps=steps, history=history, critic=serial(critic), deployment='deterministic_masked_argmax_after_global_gate_top64', fit_scope='G_only', selector_input='native choice/p/c plus six public context statistics', private_inference_inputs=False)

def deploy_quantizer(model, response, view, domain, policies, floor, fraction):
    labels = class_ids(domain)
    codes = np.asarray(model['codes'])[model['code_indices']]
    _, budgets, _ = qz.feasibility(response, view, domain, policies, floor, codes, [fraction])
    mask = budgets[fraction]
    if not mask.any():
        return (dict(response), dict(fallback=True, selected_code=None))
    net = arrays(model['network'])
    z, _ = forward(net, selector_input([response], [view], labels))
    z = z[0] + np.log(expit(net['gate']) + 1e-12)
    order = sorted(np.flatnonzero(mask), key=lambda i: (-float(z[model['code_indices'][i]]), int(i)))
    for i in order:
        candidate = as_response(codes[i], response['choice'], labels)
        audit = qz.verify_release(response, candidate, view, domain, policies, floor, fraction)
        if audit['passed']:
            return (candidate, dict(fallback=False, selected_code=int(model['code_indices'][i]), **audit))
    return (dict(response), dict(fallback=True, selected_code=None))

def as_response(vector, choice, labels):
    return dict(choice=choice, probabilities=dict(zip(labels, map(float, vector[:-1]))), confidence=float(vector[-1]))

def fit_continuous_critic(responses, views, labels, T, H, y, seed, steps=600):
    x = selector_input(responses, views, labels)
    t = np.eye(len(labels))[[labels.index(a) for a in T]]
    x = np.column_stack([x, t])
    secrets = sorted({s for h in H for s in h})
    si = {s: i for i, s in enumerate(secrets)}
    hi = np.array([[si[s] for s in h] for h in H])
    rng = np.random.default_rng(seed)
    net = init_mlp(x.shape[1], 24, len(si), rng)
    adam = Adam(net, 0.01)
    y = np.asarray(y)
    for _ in range(steps):
        scores, cache = forward(net, x)
        diff = scores[np.arange(len(x)), hi[:, 1]] - scores[np.arange(len(x)), hi[:, 0]]
        gr = (expit(diff) - y) / len(x)
        gs = np.zeros_like(scores)
        gs[np.arange(len(x)), hi[:, 1]] = gr
        gs[np.arange(len(x)), hi[:, 0]] = -gr
        adam.step(net, backward(net, cache, gs), decay=0.005)
    return dict(network=serial(net), classes=labels, secrets=secrets, fit_scope='G_only')

def memguard_adaptation(model, response, view, labels, steps=80, radius=0.2):
    """Confuse every same-parent secret contrast; deployment knows no true T/H.

    Uses all public coarse conditions and all trained same-parent contrasts,
    not the target's true condition/hypotheses. Continuous score adaptation.
    """
    net = arrays(model['network'])
    r0 = qz.old.response_values(response, labels)
    v = r0.copy()
    k = len(labels)
    contrasts = model['public_contrasts']
    ones = np.eye(k)
    for _ in range(steps):
        rr = as_response(v, response['choice'], labels)
        base = selector_input([rr], [view], labels)[0]
        x = np.column_stack([np.tile(base, (len(contrasts), 1)), ones[[c[0] for c in contrasts]]])
        sc, (xx, h) = forward(net, x)
        a = np.array([c[1] for c in contrasts])
        b = np.array([c[2] for c in contrasts])
        diff = sc[np.arange(len(x)), b] - sc[np.arange(len(x)), a]
        gs = np.zeros_like(sc)
        gs[np.arange(len(x)), b] = diff / len(x)
        gs[np.arange(len(x)), a] = -diff / len(x)
        gx = gs @ net['w2'].T * (1 - h * h) @ net['w1'].T
        g = gx[:, k:k + k + 1].sum(0) + 0.05 * (v - r0)
        v -= 0.025 * np.clip(g, -2, 2)
        v = np.clip(v, r0 - radius, r0 + radius)
        v[:-1] = np.maximum(v[:-1], 1e-06)
        v[:-1] /= v[:-1].sum()
        v[-1] = np.clip(v[-1], 0, 1)
    return as_response(v, response['choice'], labels)

def fit_autoencoder(responses, labels, seed, steps=1500):
    """PURIFIER-inspired reconstruction bottleneck; not a paper replication."""
    values = np.array([qz.old.response_values(r, labels) for r in responses])
    rng = np.random.default_rng(seed)
    enc = init_mlp(values.shape[1], 12, 2, rng)
    dec = init_mlp(2, 12, values.shape[1], rng)
    ae = Adam(enc, 0.005)
    ad = Adam(dec, 0.005)
    for _ in range(steps):
        z, ec = forward(enc, values)
        raw, dc = forward(dec, z)
        p = softmax(raw[:, :-1], axis=1)
        c = expit(raw[:, -1:])
        v = np.column_stack([p, c])
        g = 2 * (v - values) / len(values)
        gr = np.column_stack([p * (g[:, :-1] - (g[:, :-1] * p).sum(1, keepdims=True)), g[:, -1:] * c * (1 - c)])
        gd = backward(dec, dc, gr)
        h = dc[1]
        gz = gr @ dec['w2'].T * (1 - h * h) @ dec['w1'].T
        ge = backward(enc, ec, gz)
        ad.step(dec, gd)
        ae.step(enc, ge)
    return dict(encoder=serial(enc), decoder=serial(dec), bottleneck=2, fit_scope='G_only', kind='reconstruction_adaptation')

def reconstruct(model, values):
    z, _ = forward(arrays(model['encoder']), np.atleast_2d(values))
    raw, _ = forward(arrays(model['decoder']), z)
    return np.column_stack([softmax(raw[:, :-1], axis=1), expit(raw[:, -1:])])

def baseline(kind, model, response, view, labels):
    values = qz.old.response_values(response, labels)
    if kind == 'rounding':
        p = values[:-1] * 20
        v = np.floor(p)
        rest = 20 - int(v.sum())
        order = np.argsort(-(p - v), kind='stable')
        v[order[:rest]] += 1
        return as_response(np.r_[v / 20, np.round(values[-1] * 20) / 20], response['choice'], labels)
    if kind == 'hamp_output':
        ranking = np.argsort(-values[:-1], kind='stable')
        p = np.zeros(len(labels))
        p[ranking] = [0.28, 0.26, 0.24, 0.22]
        return as_response(np.r_[p, 0.28], response['choice'], labels)
    if kind == 'memguard':
        return memguard_adaptation(model, response, view, labels)
    if kind == 'purifier':
        return as_response(reconstruct(model, values)[0], response['choice'], labels)
    raise ValueError(kind)

def guarded(before, proposed, view, domain, policies, floor, fraction):
    audit = qz.verify_release(before, proposed, view, domain, policies, floor, fraction)
    return (proposed, dict(fallback=False, **audit)) if audit['passed'] else (dict(before), dict(fallback=True, **qz.verify_release(before, before, view, domain, policies, floor, fraction)))
