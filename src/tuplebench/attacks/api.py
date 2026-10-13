"""Train/validation separation and isolated defense-aware attack features."""
from contextlib import contextmanager
import importlib.util
from uuid import uuid4

import numpy as np

from tuplebench.core.io import require
from . import strong_attacker_v1 as original_base
from . import strong_attacker_v2 as original_attack


@contextmanager
def attack_engine(public_codes=(), workflow_features=False):
    """Use fresh module state so one evaluation cannot alter another."""
    def isolated(module):
        name = "tuplebench.attacks._isolated_" + uuid4().hex
        spec = importlib.util.spec_from_file_location(name, module.__file__)
        result = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(result)
        return result

    base = isolated(original_base)
    attack = isolated(original_attack)
    attack.base = base
    if workflow_features:
        before_checked = base.checked
        before_features = base.features
        before_legacy = base.legacy_features
        code_set = {tuple(code) for code in public_codes}

        def checked(row):
            require(set(row) <= {"consumer_view", "payload", "T", "hypotheses", "workflow"},
                    "Undeclared attacker input")
            before_checked({k: v for k, v in row.items() if k != "workflow"})
            require(all(k.startswith("workflow:") and v == 1.0
                        for k, v in row.get("workflow", {}).items()),
                    "Invalid observable workflow signature")
            return row

        def extras(row):
            result = dict(row.get("workflow", {}))
            payload = row["payload"]
            if "probabilities" in payload and "confidence" in payload:
                vector = tuple(payload["probabilities"][t]
                               for t in sorted(payload["probabilities"])) + (payload["confidence"],)
                result["release:known_finite_code=" + str(vector)] = 1.0
                result["release:in_public_codebook=" + str(vector in code_set)] = 1.0
            return result

        def features(row, representation, include_public=True):
            checked(row)
            result = before_features({k: v for k, v in row.items() if k != "workflow"},
                                     representation, include_public)
            result.update(extras(row))
            return result

        def legacy(rows, variant):
            result = before_legacy([{k: v for k, v in r.items() if k != "workflow"}
                                    for r in rows], variant)
            for row, feature in zip(rows, result):
                extra = extras(row)
                feature.update(extra)
                condition = row["T"] + "|" + ",".join(sorted(row["hypotheses"]))
                feature.update({"condition:" + condition + "|" + k: v
                                for k, v in extra.items()})
            return result

        base.checked, base.features, base.legacy_features = checked, features, legacy
    yield base, attack


def fit_decoder(train_rows, train_labels, validation_rows, validation_labels,
                training_folds, *, seed=20261009, public_codes=(), workflow_features=False):
    """Refit the full Strong A library. This API deliberately accepts no test data."""
    require(len(training_folds) == len(train_rows), "Training-fold length mismatch")
    require(set(training_folds) == {0, 1, 2, 3}, "Four frozen training folds required")
    with attack_engine(public_codes, workflow_features) as (_, attack):
        banks, predictions, weights = {}, {}, {}
        for release in original_base.RELEASES:
            print("Fitting decoder view:", release, flush=True)
            parts = [attack.fold_fit(train_rows, train_labels, training_folds, fold, release, seed)
                     for fold in range(4)]
            oof = attack.assemble_oof(parts, len(train_rows))
            weights[release] = [attack.fit_weights(oof, train_labels, penalty)
                                for penalty in attack.LAMBDAS]
            banks[release] = attack.fit_base_bank(train_rows, train_labels, release, seed)
            predictions[release] = attack.bank_predictions(banks[release], validation_rows, release)
        selections, search = attack.select(banks, predictions, validation_rows,
                                            validation_labels, weights)
        fitted = {}
        for release, tracks in selections.items():
            descriptor = tracks["selected"]
            subset = [banks[descriptor["source"]][i] for i in descriptor["members"]]
            fitted[release] = dict(bank=subset,
                                   descriptor=dict(descriptor, members=list(range(len(subset)))),
                                   test_used_for_selection=False)
        return fitted, search


def predict_decoder(snapshot, rows, *, public_codes=(), workflow_features=False):
    """Predict only from declared recipient/audit inputs; no gold secret is accepted."""
    require(snapshot["test_used_for_selection"] is False, "Invalid selection record")
    descriptor = snapshot["descriptor"]
    with attack_engine(public_codes, workflow_features) as (base, attack):
        probability = attack.predict({descriptor["source"]: snapshot["bank"]}, descriptor, rows)
        return probability, base.guesses(probability, rows)
