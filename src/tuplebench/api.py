"""Public evaluation API for the frozen TUPLE-Bench release."""
from collections import defaultdict
import copy
from functools import lru_cache
from pathlib import Path

import numpy as np

from .attacks.api import fit_decoder, predict_decoder
from .attacks.strong_attacker_v1 import RELEASES
from .core.contracts import class_ids
from .core.data import load_dataset, validate_dataset
from .core.io import digest, read_json, require, write_json
from .core.native import load_runs
from .core.releases import order_from_case, release_fields
from .core.statistics import cluster_mean_ci
from .core.utility import execute_plan, initial_plan, model_q
from .core.utility import fit_and_evaluate_utility, summarize_utility
from .defenses import learned, probability, quantizer, workflow

PROFILES = {
    "authored": (1, 1, 1, False),
    "review_half": (0.5, 1, 1, False),
    "review_double": (2, 1, 1, False),
    "check_quarter": (1, 0.25, 1, False),
    "error_double": (1, 1, 2, False),
    "asymmetric_class0_double": (1, 1, 1, True),
}


def profile_consumers(consumers, profile):
    """Apply the six published cost contexts without modifying any response."""
    require(profile in PROFILES, "Unknown cost profile")
    review, check, error, asymmetric = PROFILES[profile]
    result = copy.deepcopy(consumers)
    selective = result["selective_routing"]
    clarification = result["one_step_clarification"]
    selective["review_cost"] *= review
    selective["wrong_cost"] *= error
    clarification["check_cost"] *= check
    clarification["wrong_cost"] *= error
    for matrix in result["asymmetric_cost_routing"]["cost_matrix_by_domain"].values():
        for i, row in enumerate(matrix):
            for j, value in enumerate(row):
                row[j] = value * (review if i == len(matrix) - 1
                                 else error * (2 if asymmetric and j == 0 else 1))
    return result


class Benchmark:
    """Read and validate a portable release, never a developer's local archive."""

    def __init__(self, root="."):
        self.root = Path(root).resolve()
        self.dataset = load_dataset(self.root / "data/benchmark")
        self.info = read_json(self.root / "data/release.json")
        self.folds = read_json(self.root / "data/folds.json")
        validate_dataset(self.dataset)
        require(len(self.dataset.cases) == 1440 and len(self.dataset.bases) == 720,
                "Unexpected benchmark population")

    @lru_cache(maxsize=None)
    def responses(self, model):
        require(model in self.info["models"], "Unknown model")
        return load_runs(self.dataset, self.root / "data/responses" / (model + ".jsonl"))[0]

    @lru_cache(maxsize=None)
    def consumers(self, model, profile):
        selected = read_json(self.root / "artifacts/consumers" / model / (profile + ".json"))["selected"]
        return {key: dict(value, model=self.consumer_model(value["model_ref"]))
                for key, value in selected.items()}

    @lru_cache(maxsize=None)
    def consumer_model(self, identity):
        require(len(identity) == 64 and all(c in "0123456789abcdef" for c in identity), "Invalid model reference")
        return read_json(self.root / "artifacts/consumer_models" / (identity + ".json"))

    @lru_cache(maxsize=None)
    def policies(self, model, domain):
        result = []
        for profile in PROFILES:
            specs = profile_consumers(self.dataset.consumers, profile)
            selected = self.consumers(model, profile)
            for consumer, specification in specs.items():
                result.append(dict(profile=profile, consumer=consumer, specification=specification,
                                   models={release: selected[domain + "/" + consumer + "/" + release]["model"]
                                           for release in RELEASES}))
        return result

    def cases(self, domain=None, split="test"):
        require(split in ("train", "validation", "test"), "Unknown split")
        require(domain is None or domain in self.dataset.domains, "Unknown domain")
        return [c for c in self.dataset.cases
                if c["evaluator_only"]["split"] == split
                and (domain is None or c["domain_id"] == domain)]

    def primary_rows(self, cases, model, release="full_native"):
        runs = self.responses(model)
        return [dict(consumer_view=c["consumer_view"],
                     payload=release_fields(runs[c["case_id"]]["response"], release,
                                            order_from_case(c), runs[c["case_id"]]["status"]),
                     T=c["evaluator_only"]["truth"]["T"],
                     hypotheses=c["evaluator_only"]["hypotheses"]) for c in cases]


class Defense:
    """Single-response backend API. Gold labels, identifiers and pair mates are excluded."""

    def __init__(self, benchmark, model, domain):
        require(model in benchmark.info["defense_models"], "No frozen defense for this model")
        self.domain = benchmark.dataset.domains[domain]
        self.labels = class_ids(self.domain)
        self.policies = benchmark.policies(model, domain)
        self.floor = benchmark.dataset.engine["posterior_floor"]
        self.package = read_json(benchmark.root / "artifacts/defenses" / model / (domain + ".json"))

    def release(self, response, public_view, method="strict_expanded", *, rng=None):
        require(set(response) == {"choice", "probabilities", "confidence"}, "Native decision fields only")
        require(set(public_view) == {"recipient_role", "public", "public_task", "downstream_contexts"},
                "Public context only")
        if method == "native":
            return copy.deepcopy(response)
        if method in probability.METHODS:
            order = [c["id"] for c in public_view["public_task"]["candidates"]]
            return probability.transform(response, order, method, rng)
        require(method in self.package["codebooks"], "Unknown defense")
        book = self.package["codebooks"][method]
        fraction = book.get("fraction")
        if method in self.package["learned"]:
            result, _ = learned.deploy_quantizer(self.package["learned"][method], response, public_view,
                                                 self.domain, self.policies, self.floor, fraction)
            return result
        for style in ("rounding", "memguard", "purifier", "hamp_output"):
            if method.startswith(style + "_"):
                proposed = learned.baseline(style, self.package["adaptations"].get(style),
                                            response, public_view, self.labels)
                if method.endswith("_standalone"):
                    return proposed
                return learned.guarded(response, proposed, public_view, self.domain,
                                        self.policies, self.floor, fraction)[0]
        return quantizer.release(response, public_view, self.domain, self.policies,
                                 self.floor, book["codes"], fraction)[0]

    def public_codes(self, method):
        codes = self.package["codebooks"].get(method, {}).get("codes", [])
        order = [self.labels.index(t) for t in sorted(self.labels)]
        return [[v[i] for i in order] + [v[-1]] for v in codes]

    def observed_signatures(self, response, view):
        value = workflow.response_values(response, self.labels)
        signatures, keys = [], []
        for policy in self.policies:
            affine = workflow.affine_policy(policy["models"]["full_native"], view,
                                             response["choice"], self.labels)
            q = workflow.policy_q(affine, value, self.floor)[0]
            signatures.append(workflow.scalar_signature(q, self.domain, policy["specification"]))
            keys.append(policy["profile"] + "/" + policy["consumer"])
        return quantizer.signature_features(signatures, keys)


def _truth(cases):
    return [c["evaluator_only"]["hypotheses"].index(c["evaluator_only"]["truth"]["S"]) for c in cases]


def _recovery(benchmark, cases, predictions):
    clusters = [c["evaluator_only"]["rendering_family_id"] for c in cases]
    return cluster_mean_ci(predictions, clusters,
                           draws=benchmark.dataset.engine["bootstrap_resamples"],
                           seed=benchmark.dataset.engine["seed"])


def _save(output, result):
    if output is not None:
        path = Path(output)
        require(not path.exists(), "Choose a new output directory")
        path.mkdir(parents=True)
        write_json(path / "summary.json", result)
    return result


def evaluate_primary(root=".", *, models=None, domains=None, profiles=("authored",),
                     refit=False, output=None):
    """Replay published fits, or freshly train Strong A on G and select on V."""
    benchmark = Benchmark(root)
    models = benchmark.info["models"] if models is None else list(models)
    domains = list(benchmark.dataset.domains) if domains is None else list(domains)
    require(set(models) <= set(benchmark.info["models"]), "Unknown model")
    require(set(domains) <= set(benchmark.dataset.domains), "Unknown domain")
    require(set(profiles) <= set(PROFILES), "Unknown profile")
    records, pooled, coarse = [], defaultdict(list), defaultdict(list)
    for model in models:
        for domain in domains:
            cases = benchmark.cases(domain)
            print("Primary:", model, domain, flush=True)
            runs = benchmark.responses(model)
            coarse[model].extend((c, float(runs[c["case_id"]]["response"]["choice"] ==
                                          c["evaluator_only"]["truth"]["T"])) for c in cases)
            if refit:
                train = benchmark.cases(domain, "train")
                validation = benchmark.cases(domain, "validation")
                fitted, _ = fit_decoder(benchmark.primary_rows(train, model), _truth(train),
                                        benchmark.primary_rows(validation, model), _truth(validation),
                                        [benchmark.folds[c["case_id"]] for c in train])
            for release in RELEASES:
                snapshot = fitted[release] if refit else read_json(
                    benchmark.root / "artifacts/primary" / model / domain / (release + ".json"))
                _, guesses = predict_decoder(snapshot, benchmark.primary_rows(cases, model))
                correct = (guesses == _truth(cases)).astype(float).tolist()
                pooled[model, release].extend(zip(cases, correct))
                costs = []
                for policy in benchmark.policies(model, domain):
                    if policy["profile"] not in profiles:
                        continue
                    model_fit = policy["models"][release]
                    q = model_q(model_fit, cases, benchmark.responses(model), release,
                                class_ids(benchmark.dataset.domains[domain]), benchmark.dataset.engine["posterior_floor"])
                    values = []
                    for c, posterior in zip(cases, q):
                        ev = c["evaluator_only"]
                        plan = initial_plan(posterior, benchmark.dataset.domains[domain], policy["specification"])
                        execution = execute_plan(posterior, plan, benchmark.dataset.domains[domain],
                                                  policy["specification"], ev["truth"]["T"], ev["approved_check_answers"])
                        values.append(execution["cost"])
                    costs.append(dict(profile=policy["profile"], workflow=policy["consumer"],
                                      cost=_recovery(benchmark, cases, values)))
                records.append(dict(model=model, domain=domain, release=release,
                                    recovery=_recovery(benchmark, cases, correct), costs=costs))
    aggregate = [dict(model=model, release=release,
                      recovery=_recovery(benchmark, [c for c, _ in rows], [x for _, x in rows]))
                 for (model, release), rows in pooled.items()]
    coarse_accuracy = [dict(model=model, accuracy=_recovery(benchmark,
                           [c for c, _ in values], [value for _, value in values]))
                       for model, values in coarse.items()]
    return _save(output, dict(protocol="primary_release_audit", refitted=refit,
                             coarse_accuracy=coarse_accuracy,
                             recovery=aggregate, per_domain=records, API_calls=0))


def evaluate_response_file(response_file, root=".", *, profiles=("authored",), output=None):
    """Fit and evaluate a complete canonical response file from an additional model."""
    benchmark = Benchmark(root)
    runs, quality = load_runs(benchmark.dataset, response_file)
    require(set(profiles) <= set(PROFILES), "Unknown cost profile")
    recovery, costs = [], []
    for profile in profiles:
        current = copy.deepcopy(benchmark.dataset)
        current.consumers = profile_consumers(current.consumers, profile)
        episodes, _, _, _ = fit_and_evaluate_utility(current, runs)
        costs.extend(dict(profile=profile, **row) for row in summarize_utility(current, episodes)
                     if row["release"] in RELEASES)
    for domain in benchmark.dataset.domains:
        train = benchmark.cases(domain, "train")
        validation = benchmark.cases(domain, "validation")
        test = benchmark.cases(domain)

        def rows(cases):
            return [dict(consumer_view=c["consumer_view"],
                         payload=release_fields(runs[c["case_id"]]["response"], "full_native",
                                                order_from_case(c), runs[c["case_id"]]["status"]),
                         T=c["evaluator_only"]["truth"]["T"],
                         hypotheses=c["evaluator_only"]["hypotheses"]) for c in cases]

        fitted, _ = fit_decoder(rows(train), _truth(train), rows(validation), _truth(validation),
                                [benchmark.folds[c["case_id"]] for c in train])
        for release, snapshot in fitted.items():
            _, guesses = predict_decoder(snapshot, rows(test))
            correct = (guesses == _truth(test)).astype(float).tolist()
            recovery.append(dict(domain=domain, release=release,
                                 recovery=_recovery(benchmark, test, correct)))
    return _save(output, dict(protocol="additional_model_primary_audit", recovery=recovery,
                             utility=costs, input_validation=quality, API_calls=0))


def evaluate_defenses(root=".", *, models=("typesafe", "openjev", "clef"), domains=None,
                      methods=("native", "quantized_10", "temperature_2", "gaussian_05", "strict_expanded"),
                      refit=False, output=None):
    """Audit all methods with frozen consumers and defense-aware Strong A."""
    benchmark = Benchmark(root)
    domains = list(benchmark.dataset.domains) if domains is None else list(domains)
    pooled, records = defaultdict(list), []
    for model in models:
        for domain in domains:
            cases = benchmark.cases(domain)
            defense = Defense(benchmark, model, domain)
            saved = read_json(benchmark.root / "artifacts/probability_releases" / model / (domain + ".json"))["methods"]
            saved = {m: {r["case_id"]: r["payload"] for r in rows} for m, rows in saved.items()}
            for method in methods:
                print("Defense:", model, domain, method, flush=True)
                components = list(probability.NOISE_SEEDS) if method == "gaussian_05" else [method]
                component_correct, component_cost = [], []
                for component in components:
                    codes = defense.public_codes(component)

                    def rows_for(population):
                        result, releases = [], {}
                        for c in population:
                            original = benchmark.responses(model)[c["case_id"]]["response"]
                            released = (saved[component][c["case_id"]] if component in saved
                                        else defense.release(original, c["consumer_view"], component))
                            releases[c["case_id"]] = dict(status="completed", response=released)
                            ev = c["evaluator_only"]
                            result.append(dict(consumer_view=c["consumer_view"], payload=released,
                                               T=ev["truth"]["T"], hypotheses=ev["hypotheses"],
                                               workflow=defense.observed_signatures(released, c["consumer_view"])))
                        return result, releases

                    test_rows, releases = rows_for(cases)
                    if refit:
                        train, validation = benchmark.cases(domain, "train"), benchmark.cases(domain, "validation")
                        training_rows = rows_for(train)[0]
                        validation_rows = rows_for(validation)[0]
                        fitted, _ = fit_decoder(training_rows, _truth(train), validation_rows,
                                                _truth(validation), [benchmark.folds[c["case_id"]] for c in train],
                                                public_codes=codes, workflow_features=True)
                        snapshot = fitted["full_native"]
                        if component == "temperature_2":
                            inverse_train = [dict(r, payload=probability.inverse_temperature(r["payload"])) for r in training_rows]
                            inverse_validation = [dict(r, payload=probability.inverse_temperature(r["payload"])) for r in validation_rows]
                            inverse_fitted, _ = fit_decoder(inverse_train, _truth(train), inverse_validation,
                                                            _truth(validation), [benchmark.folds[c["case_id"]] for c in train],
                                                            workflow_features=True)
                            candidates = [(snapshot, validation_rows),
                                          (inverse_fitted["full_native"], inverse_validation)]
                            criteria = []
                            truth = np.array(_truth(validation))
                            for fitted_snapshot, observed in candidates:
                                q, predicted = predict_decoder(fitted_snapshot, observed, workflow_features=True)
                                criteria.append((-float(np.mean(predicted == truth)),
                                                 float(-np.log(np.maximum(q[np.arange(len(truth)), truth], 1e-15)).mean())))
                            chosen = min(range(2), key=lambda i: (*criteria[i], i))
                            snapshot = candidates[chosen][0]
                            if chosen == 1:
                                test_rows = [dict(r, payload=probability.inverse_temperature(r["payload"])) for r in test_rows]
                    else:
                        snapshot = read_json(benchmark.root / "artifacts/defense_attacks" / model / domain / (component + ".json"))
                    if component == "temperature_2" and not refit:
                        selection = read_json(benchmark.root / "artifacts/temperature_selection" / model / (domain + ".json"))
                        if selection["selected_track"] == "public_inverse":
                            snapshot = read_json(benchmark.root / "artifacts/defense_attacks" / model / domain / "temperature_2_inverse_known.json")
                            test_rows = [dict(r, payload=probability.inverse_temperature(r["payload"])) for r in test_rows]
                    _, guesses = predict_decoder(snapshot, test_rows, public_codes=codes, workflow_features=True)
                    component_correct.append((guesses == _truth(cases)).astype(float))
                    costs = {}
                    for policy in defense.policies:
                        posteriors = model_q(policy["models"]["full_native"], cases, releases, "full_native",
                                             defense.labels, defense.floor)
                        baseline = model_q(policy["models"]["full_native"], cases, benchmark.responses(model),
                                           "full_native", defense.labels, defense.floor)
                        values = []
                        for c, q, original_q in zip(cases, posteriors, baseline):
                            ev = c["evaluator_only"]
                            if component == "strict_expanded":
                                require(np.array_equal(workflow.scalar_signature(q, defense.domain, policy["specification"]),
                                                       workflow.scalar_signature(original_q, defense.domain, policy["specification"])),
                                        "Strict workflow signature changed")
                            execution = execute_plan(q, initial_plan(q, defense.domain, policy["specification"]),
                                                      defense.domain, policy["specification"], ev["truth"]["T"],
                                                      ev["approved_check_answers"])
                            values.append(execution["cost"])
                        costs[policy["profile"], policy["consumer"]] = values
                    component_cost.append(costs)
                correct = np.mean(component_correct, axis=0).tolist()
                pooled[model, method].extend(zip(cases, correct))
                costs = [dict(profile=profile, workflow=consumer,
                              cost=_recovery(benchmark, cases, np.mean([x[profile, consumer] for x in component_cost], axis=0).tolist()))
                         for profile, consumer in component_cost[0]]
                records.append(dict(model=model, domain=domain, method=method,
                                    recovery=_recovery(benchmark, cases, correct), costs=costs))
    aggregate = [dict(model=model, method=method,
                      recovery=_recovery(benchmark, [c for c, _ in rows], [x for _, x in rows]))
                 for (model, method), rows in pooled.items()]
    return _save(output, dict(protocol="workflow_aware_defense_audit", refitted=refit,
                             recovery=aggregate, per_domain=records, API_calls=0,
                             all_frozen_workflows=18, development_results=True))
