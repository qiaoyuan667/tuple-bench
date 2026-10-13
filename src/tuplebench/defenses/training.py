"""Training-only candidate support and validation-only proxy selection."""
import numpy as np
from collections import defaultdict
from tuplebench.core.io import require
from tuplebench.core.contracts import class_ids
from . import quantizer, learned
from . import workflow
from tuplebench.attacks import strong_attacker_v1 as original_base
from tuplebench.attacks import strong_attacker_v2 as original_attack
BUDGETS = dict(zero=0., low=.04, high=.20)


def fit_wcpq(benchmark, model, domain, *, learned_extension=False, steps=600):
    """Rebuild strict codes from G; optionally fit six bounded learners on G/V.

    This function never requests a test split. Gold T/H/S occur only in the
    training critic and validation proxy targets, never in deployed release.
    """
    train = benchmark.cases(domain, "train")
    require(train and all(c["evaluator_only"]["split"] == "train" for c in train), "Training cases only")
    dom = benchmark.dataset.domains[domain]
    labels = class_ids(dom)
    floor = benchmark.dataset.engine["posterior_floor"]
    policies = benchmark.policies(model, domain)
    responses = [benchmark.responses(model)[c["case_id"]]["response"] for c in train]
    groups = defaultdict(list)
    for i, case in enumerate(train):
        groups[case["evaluator_only"]["base_id"]].append(i)
    pairs = list(groups.values())
    require(all(len(pair) == 2 for pair in pairs), "Complete training pairs required")
    midpoints = []
    for i, j in pairs:
        values = (workflow.response_values(responses[i], labels) +
                  workflow.response_values(responses[j], labels)) / 2
        midpoints.append(learned.as_response(values, responses[i]["choice"], labels))
    candidates = quantizer.expanded_candidates(responses + midpoints, labels)
    result = dict(strict_expanded=dict(codes=candidates.tolist(), fraction=None),
                  fixed_grid_control=dict(codes=workflow.simplex_grid(len(labels)).tolist(), fraction=None))
    if not learned_extension:
        return dict(codebooks=result, learned={}, test_used_for_training=False)
    validation = benchmark.cases(domain, "validation")
    require(all(c["evaluator_only"]["split"] == "validation" for c in validation), "Validation cases only")
    views = [c["consumer_view"] for c in train]
    compatible = [pair for pair in pairs if responses[pair[0]]["choice"] == responses[pair[1]]["choice"]]
    masks = {name: [] for name in BUDGETS}
    for response, view in zip(responses, views):
        _, feasible, _ = quantizer.feasibility(response, view, dom, policies, floor,
                                               candidates, list(BUDGETS.values()))
        for name, fraction in BUDGETS.items():
            masks[name].append(feasible[fraction])
    for name, fraction in BUDGETS.items():
        for paired, weight in (("unpaired", 0), ("paired", 4)):
            selected = quantizer.greedy_codebook(masks[name], compatible, weight, 128)
            result["budget_" + name + "_" + paired] = dict(
                codes=candidates[selected["indices"]].tolist(), fraction=fraction)
    pool = train_pool(train, responses, result, labels)
    truths = [c["evaluator_only"]["truth"]["T"] for c in train]
    hypotheses = [c["evaluator_only"]["hypotheses"] for c in train]
    targets = [h.index(c["evaluator_only"]["truth"]["S"]) for c, h in zip(train, hypotheses)]
    folds = [benchmark.folds[c["case_id"]] for c in train]
    networks = {}
    for name, fraction in BUDGETS.items():
        mask = np.array([quantizer.feasibility(response, view, dom, policies, floor, pool,
                                               [fraction])[1][fraction]
                         for response, view in zip(responses, views)])
        for paired, pair_weight in (("unpaired", 0), ("paired", 2)):
            proposed = []
            for seed in (19, 71):
                fitted = learned.fit_quantizer(responses, views, labels, truths, hypotheses,
                                                targets, pairs, pool, mask, seed, pair_weight, steps=steps)

                def releases(cases):
                    return [learned.deploy_quantizer(fitted, benchmark.responses(model)[c["case_id"]]["response"],
                                                       c["consumer_view"], dom, policies, floor, fraction)[0]
                            for c in cases]

                proposed.append((fitted, releases(train), releases(validation)))
            selection = proxy_select(proposed, train, validation, labels, policies, dom,
                                     floor, folds, benchmark.dataset.engine["seed"])
            selected = proposed[selection["selected_index"]][0]
            method = "privacy_" + paired + "_" + name
            networks[method] = {k: selected[k] for k in ("network", "codes", "code_indices")}
            kept = np.asarray(selected["codes"])[selected["code_indices"]]
            result[method] = dict(codes=kept.tolist(), fraction=fraction)
    return dict(codebooks=result, learned=networks, test_used_for_training=False,
                selection_scope="validation", seeds=[19, 71], updates=steps)

def train_pool(gc, gr, oldbooks, labels):
    vectors = []
    for b in BUDGETS:
        vectors += oldbooks[f'budget_{b}_unpaired']['codes']
        vectors += oldbooks[f'budget_{b}_paired']['codes']
    vectors += workflow.simplex_grid(len(labels))[:32].tolist()
    originals = np.array([workflow.response_values(r, labels) for r in gr])
    chosen = [0]
    distance = np.full(len(originals), np.inf)
    for _ in range(min(64, len(originals))):
        distance = np.minimum(distance, ((originals - originals[chosen[-1]]) ** 2).sum(1))
        distance[chosen] = -1
        j = int(np.argmax(distance))
        if distance[j] <= 0:
            break
        chosen.append(j)
    vectors += originals[chosen].tolist()
    return np.array(list(dict.fromkeys((tuple(v) for v in vectors))))

def proxy_select(candidates, gcases, vcases, labels, policies, dom, floor, folds, seed):
    """Independent linear/shared/template proxy; four source folds, no T."""
    base, attack = original_base, original_attack

    def rows(cs, responses):
        return [dict(consumer_view=c['consumer_view'], payload=r, T=c['evaluator_only']['truth']['T'], hypotheses=c['evaluator_only']['hypotheses']) for c, r in zip(cs, responses)]
    y = lambda cs: np.array([c['evaluator_only']['hypotheses'].index(c['evaluator_only']['truth']['S']) for c in cs])
    yt, yv = (y(gcases), y(vcases))
    records = []
    for ci, (model, gr, vr) in enumerate(candidates):
        tr, va = (rows(gcases, gr), rows(vcases, vr))

        def bank(rr, yy):
            return [('shared', base.fit_shared(rr, yy, 'raw', 1.0)), ('template', base.fit_template(rr, yy, 'log_ratio', 'diagonal', 2.0))]
        oof = np.zeros((len(tr), 2, 2))
        for fold in range(4):
            idx = np.array(folds) != fold
            held = np.flatnonzero(~idx)
            bb = bank([tr[i] for i in np.flatnonzero(idx)], yt[idx].tolist())
            oof[held] = np.stack([base.predict(m, [tr[i] for i in held]) for _, m in bb], axis=1)
        weights = attack.fit_weights(oof, yt.tolist(), 0.01)
        bb = bank(tr, yt.tolist())
        qs = [base.predict(m, va) for _, m in bb]
        ww = np.array(weights['weights'])
        qs.append(np.einsum('nmc,m->nc', np.stack(qs, axis=1), ww))
        metrics = [dict(accuracy=float(np.mean(base.guesses(q, va) == yv)), log_loss=float(-np.log(np.maximum(q[np.arange(len(yv)), yv], 1e-15)).mean())) for q in qs]
        strongest = max(range(len(qs)), key=lambda i: (metrics[i]['accuracy'], -metrics[i]['log_loss'], -i))
        records.append(dict(index=ci, seed=model['seed'], strongest_proxy=strongest, metrics=metrics, weights=weights, privacy_accuracy=metrics[strongest]['accuracy'], privacy_log_loss=metrics[strongest]['log_loss'], fit_scope='G_only', selection_scope='V_only'))
    chosen = min(range(len(records)), key=lambda i: (records[i]['privacy_accuracy'], -records[i]['privacy_log_loss'], i))
    return dict(selected_index=chosen, candidates=records, test_used_for_selection=False)
