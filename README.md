# Run evaluation

Requires Python 3.10 or newer. Extract both `tuple-bench-software.zip` and
`tuple-bench-dataset.zip` into the **same directory**. Both archives contain a
`tuple-bench/` folder, so their contents merge into this layout:

```text
tuple-bench/
  README.md
  pyproject.toml
  src/tuplebench/
  tests/unit/
  tests/integration/
  artifacts/           # frozen policies, defenses and selected decoders
  data/               # database, frozen cases, folds and cached responses
```

From `tuple-bench/`, install and test:

```bash
python3 -m venv .venv
source .venv/bin/activate
# Windows: .venv\Scripts\Activate.ps1
python -m pip install -e ".[test]"
python -m pytest -q
```

Run a small evaluation first:

```python
from tuplebench import evaluate_primary
from tuplebench import verify_release

print(verify_release("."))

result = evaluate_primary(
    root=".",
    models=["typesafe"],
    domains=["delivery_operations"],
    output="results/smoke",
)
for row in result["recovery"]:
    print(row["release"], row["recovery"]["mean"])
```

Run the complete experiments through the Python API:

```python
from tuplebench import evaluate_primary, evaluate_defenses, evaluate_refusal
from tuplebench.api import PROFILES

# Nine models, five release views, three domains and six cost profiles.
evaluate_primary(root=".", profiles=list(PROFILES), output="results/primary")

# Three models; native, 0.1 quantization, temperature 2, Gaussian 0.05 and strict WCPQ.
# Every method uses the same 18 frozen workflow policies and adapted attack features.
evaluate_defenses(root=".", output="results/defenses")

# Nine-model direct/residual refusal probe on all 1,440 worlds per model.
evaluate_refusal(root=".", output="results/refusal")
```

Save these snippets as a Python file or paste them into a Python session.
Each call writes `summary.json` when `output` is supplied. Use a new output
directory for each run. Default evaluation re-executes the saved,
training-fitted and validation-selected models on the bundled responses;
it does not merely copy reported accuracy values. No model API calls occur.

To refit the complete secret-decoder library, set `refit=True`:

```python
evaluate_primary(root=".", models=["typesafe"], refit=True,
                 output="results/primary_refitted")
evaluate_defenses(root=".", models=["typesafe"], methods=["strict_expanded"],
                  refit=True, output="results/strict_refitted")
```

Refitting uses the frozen 864/288/288 train/validation/test split and four
grouped training folds; test labels are used only for scoring. Frozen
workflow policies remain fixed in the defense evaluation. Full refitting
takes longer and numerical results can vary across dependency versions.
The reference environment is recorded in `validation/environment.json`.

Additional learned WCPQ variants can be evaluated using `methods`, for example
`["privacy_unpaired_zero", "privacy_paired_zero", "privacy_unpaired_low",
"privacy_paired_low", "privacy_unpaired_high", "privacy_paired_high"]`.
The suffixes denote public-cost budgets 0, 0.04 and 0.20. `fixed_grid_control`,
`rounding_standalone`, `memguard_standalone`, `purifier_standalone` and
`hamp_output_standalone` are also available. The last three are score-defense
adaptations, rather than full replications of the named training defenses.

`data/benchmark.sqlite3` provides `worlds`, `pairs`, `public_cases` and
`model_inputs`; the JSONL mirror is used by the evaluation API. Review issues
are retained in `data/review_issues.json`. These are synthetic development
results, without independent confirmation or a formal privacy guarantee.

For a new model, supply your own HTTPS typed-decision endpoint using the API:

```python
from tuplebench import Benchmark, TypedDecisionClient

benchmark = Benchmark(".")
client = TypedDecisionClient(
    endpoint="https://YOUR_PROVIDER/YOUR_TYPED_DECISION_ENDPOINT",
    model="YOUR_FIXED_RETURNED_MODEL_VERSION",
    api_key_env="MODEL_API_KEY",
)
# This explicitly sends one paid request. Set MODEL_API_KEY in your environment.
response = client.predict(benchmark.dataset.cases[0]["model_view"])
```

Only the registered model view is sent. The client uses HTTP APIs, sends one
request per call and never retries automatically. A `response_adapter`
callable can translate provider envelopes without changing returned scores.
Store one canonical status row per case using the schema in `data/responses/`,
and keep model versions separate. Then run:

```python
from tuplebench import evaluate_response_file

evaluate_response_file("new_model_responses.jsonl", root=".",
                       output="results/new_model")
```

This fits workflow consumers and secret decoders on training data, selects
on validation, and scores test. Existing bundled results run entirely offline.
