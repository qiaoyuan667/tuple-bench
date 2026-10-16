# Reproduce the released experiments

The GitHub checkout includes data and frozen fitted artifacts. For a
download-only installation, merge the software and dataset ZIPs as described
in the main README. Run all snippets from the resulting repository root.

## Installation and integrity

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[test]"
python -m pytest -q
```

```python
from tuplebench import verify_release
print(verify_release("."))
```

Verification hashes every file named in the software and dataset manifests.
Generated evaluation outputs go under a new directory and do not replace
frozen input artifacts. Windows activation uses `.venv\Scripts\Activate.ps1`;
native Windows execution has not been validated.

## Small offline smoke run

```python
from tuplebench import evaluate_primary

report = evaluate_primary(root=".", models=["typesafe"],
                          domains=["delivery_operations"],
                          output="results/smoke")
```

This runs five release views on 96 test worlds and the three authored-cost
workflows. Expected recovery is 50.00%, 53.125%, 50.00%, 61.458333…%, and
62.50% for no output, choice, choice+confidence, full probabilities, and
full native respectively. Machine-readable no output uses `public_only`.

## Nine-model primary release audit

```python
from tuplebench import evaluate_primary
from tuplebench.api import PROFILES

report = evaluate_primary(root=".", profiles=list(PROFILES),
                          output="results/primary")
```

Default models are TypeSafe, OpenJev, JevK5, Clef, Clef-flash, Decider,
Sage, d1, and Mercury Decide. `data/models.json` records serving versions.
No model is substituted with a newer serving version during replay.

The return value and saved `summary.json` contain:

- `coarse_accuracy`: pooled selected-category accuracy by model;
- `recovery`: pooled attack recovery by model and release;
- `per_domain`: domain-specific recovery and workflow costs;
- `API_calls`: zero for cached evaluation.

In `per_domain`, each `costs` item identifies `profile` and `workflow` and
contains a fixed-fit mean and interval. Pooled business cost is the average
of domain means because every domain has 96 test worlds.

## Three-model defense audit

```python
from tuplebench import evaluate_defenses

report = evaluate_defenses(root=".", output="results/defenses")
```

Defaults are TypeSafe, OpenJev, and Clef, with methods:

| Method ID | Release |
| --- | --- |
| `native` | Unmodified native fields |
| `quantized_10` | Largest-remainder simplex quantization at step 0.1 |
| `temperature_2` | Temperature smoothing with temperature 2 |
| `gaussian_05` | Gaussian standard deviation 0.05, clipping and normalization |
| `strict_expanded` | Strict WCPQ, including native fallback |

Probability baselines independently modify probabilities while retaining
choice and native confidence. Every method is evaluated with the same
eighteen frozen consumers and adapted signature-aware attack inputs.
Gaussian averages outcomes per world across three predeclared realizations;
it represents 288 test worlds, not 864 independent cases.

Additional IDs accepted through `methods=[...]` include:

- `privacy_unpaired_zero`, `privacy_paired_zero`;
- `privacy_unpaired_low`, `privacy_paired_low`;
- `privacy_unpaired_high`, `privacy_paired_high`;
- `fixed_grid_control`;
- `rounding_standalone`, `memguard_standalone`, `purifier_standalone`,
  `hamp_output_standalone`.

Learned suffixes denote public-cost budgets 0, 0.04, and 0.20. The
paired/unpaired ablation changes pair loss while sharing a candidate pool
that already includes pair-derived candidates. Named score-defense
controls are adaptations, not complete replications of the original papers.

## Whole-dataset refusal probe

```python
from tuplebench import evaluate_refusal
report = evaluate_refusal(root=".", output="results/refusal")
```

This uses separate fine-mechanism refusal responses for all 1,440 worlds
per model. Correct direct selection and correct unique residual recovery
after withholding are disjoint routes. Ties, zero fine-score mass, and
wrong winners receive no recovery credit.

## Retrain the privacy attacker

```python
evaluate_primary(root=".", models=["typesafe"], refit=True,
                 output="results/primary_refitted")
evaluate_defenses(root=".", models=["typesafe"],
                  methods=["strict_expanded"], refit=True,
                  output="results/strict_refitted")
```

Refitting trains Strong A using training worlds and four grouped folds,
learns ensemble weights from out-of-fold predictions, selects on
validation, and scores test. It does not refit frozen defense consumers.
For primary replay/refit, utility consumers are also the bundled fits.
For another model, use `evaluate_response_file` as described in
[the API guide](model-api.md).

Default replay runs selected fitted models on saved responses. It is the
fast reproducibility path; refitting is substantially more expensive.
Numerical libraries can affect refitted choices.

## Reference checks

[`validation/`](../validation) records original release checks:

| Check | Recorded result |
| --- | --- |
| Unit and integration suite | 92 passed |
| Independent installation from both extracted ZIPs | Passed |
| Primary recovery estimates | 45 reproduced |
| Main defense recovery estimates | 15 reproduced |
| Refusal models | Nine reproduced |
| Primary domain cost estimates and intervals | 2,430 reproduced |
| Defense domain cost estimates and intervals | 810 reproduced |
| Strict signature comparisons | 15,552 passed |

Reference Python is 3.14.7, with NumPy 2.5.3, SciPy 1.18.1,
scikit-learn 1.9.1, jsonschema 4.26.0, and pytest 9.1.1. A separate extracted
installation with NumPy 2.5.4 reproduced the same cached estimates and
intervals. These describe completed environments, not every platform or
dependency combination.
