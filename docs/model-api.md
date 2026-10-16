# Connect your typed decision model

Bundled experiments run offline. New-model collection is an explicit
caller action and can incur provider charges. The client sends one
request per `predict` call and never automatically retries.

## Configure an HTTPS endpoint

Install the project, set your provider key in `MODEL_API_KEY`, and use:

```python
from tuplebench import Benchmark, TypedDecisionClient

benchmark = Benchmark(".")
client = TypedDecisionClient(
    endpoint="https://YOUR_PROVIDER/YOUR_TYPED_DECISION_ENDPOINT",
    model="YOUR_REQUEST_SELECTOR",
    expected_version="YOUR_FIXED_RETURNED_MODEL_VERSION",
    api_key_env="MODEL_API_KEY",
)
case = benchmark.dataset.cases[0]
response = client.predict(case["model_view"])
```

Only `model_view` is serialized into the typed request. It contains the
recipient contract, public state, private evidence, and coarse question.
The protected secret is not offered as a production candidate. Evaluator
fields, truth labels, pair IDs, and the paired response are excluded.

This is a typed-decision JSON API, rather than a universal chat API.
`response_adapter` may translate a provider envelope to the expected
schema while preserving returned scores and selected choice.

The accepted response envelope has this structure (illustrative values):

```json
{
  "model": "YOUR_FIXED_RETURNED_MODEL_VERSION",
  "answers": {
    "coarse_cause": {
      "type": "choice",
      "choice": "POWER",
      "probabilities": {"POWER": 0.8, "LINK": 0.1, "SENSOR": 0.05, "MOTION": 0.05},
      "confidence": 0.8
    }
  }
}
```

The returned selected category is authoritative; do not reconstruct it
from the probability argmax. Preserve scores by class identity, not
dictionary order. Inspect invalid vectors rather than silently normalizing
or choosing replacements based on prediction correctness.

## Store canonical per-case responses

`predict` returns only the canonical decision fields. Store them in one
JSONL status record per registered case:

```python
record = {
    "case_id": case["case_id"],
    "model_view_sha256": case["evaluator_only"]["model_view_sha256"],
    "status": "completed",
    "origin": "native_api",
    "model_version": "YOUR_FIXED_RETURNED_MODEL_VERSION",
    "response": response,
}
```

The input hash is grouping/validation metadata read after the model call;
it is not sent to the provider or used as a predictive feature.
Compare the exact wrapper with `data/responses/` before full collection.
Preserve one model version per file and all planned worlds, including
incorrect coarse predictions. Inspect failures before any later retry.

Keep credentials, transport headers, unrelated private logs, and local
developer paths out of published response files. Collection provenance
is separate from prediction features.

## Fit and evaluate

Once a complete canonical file is available:

```python
from tuplebench import evaluate_response_file
from tuplebench.api import PROFILES

report = evaluate_response_file(
    "new_model_responses.jsonl", root=".", profiles=list(PROFILES),
    output="results/new_model",
)
```

This validates coverage, fits consumers and decoders on training worlds,
selects on validation, and evaluates test. Reports include domain
recovery and utility results. Offline fitting makes no model API calls.

## Apply a bundled WCPQ backend

Frozen defenses cover TypeSafe, OpenJev, and Clef:

```python
from tuplebench import Benchmark, Defense

benchmark = Benchmark(".")
case = benchmark.cases("delivery_operations", "test")[0]
native = benchmark.responses("typesafe")[case["case_id"]]["response"]
defense = Defense(benchmark, "typesafe", "delivery_operations")
released = defense.release(native, case["consumer_view"], method="strict_expanded")
```

The mapper receives native fields and public context only. Codebooks and
policies are model-specific; evaluating another model does not
automatically provide a validated defense for it.
