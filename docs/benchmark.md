# Benchmark and data guide

[Code](https://github.com/qiaoyuan667/tuple-bench) · [Hugging Face](https://huggingface.co/datasets/Qiaoyuan/TUPLE-Bench)

## Recipient contract

Let `X` be public context, `B` public downstream costs and permitted checks,
`T` the authorized coarse category, `S` the protected fine mechanism, and
`E` model-visible private evidence. The evaluated model receives `(X, E)`
and coarse candidate definitions. Its native decision contains a selected
category, a probability vector, and a confidence field when available.

At action time, a legitimate recipient has `(X, B, release)` but not gold
`T`, `S`, or `E`. The privacy attacker additionally receives the authorized
`T` and two registered secret hypotheses. Gold `S` is a training target and
evaluation label, never a test feature. The paired world's response is not
an attacker or deployed-defense input.

## Frozen population

| Domain ID | Coarse categories | Pairs | Worlds |
| --- | --- | ---: | ---: |
| `delivery_operations` | DELAY, ADDRESS, DAMAGE, DELIVERY_RECORD | 240 | 480 |
| `equipment_service` | POWER, LINK, SENSOR, MOTION | 240 | 480 |
| `it_operations` | IDENTITY, NETWORK, STORAGE, APPLICATION | 240 | 480 |

Each coarse category contains four protected mechanisms. The complete
ontology, scoped definitions, recipient contracts, costs, and checks are in
`data/benchmark/domain_specs/` and `data/benchmark/consumer_specs/`.

| Split | Pairs | Worlds | Worlds per domain | Worlds per fine mechanism |
| --- | ---: | ---: | ---: | ---: |
| Train | 432 | 864 | 288 | 18 |
| Validation | 144 | 288 | 96 | 6 |
| Test | 144 | 288 | 96 | 6 |

Both worlds of a source family remain in the same split and training fold.
`data/folds.json` stores the four fixed training-fold assignments. Preserve
these assignments when reproducing the released results.

## Complete evaluation dataset

The full dataset ZIP is available at:

https://huggingface.co/datasets/Qiaoyuan/TUPLE-Bench/resolve/main/tuple-bench-dataset.zip

It contains a `tuple-bench/` directory. Extract it alongside the software
ZIP into the same parent directory, or use the GitHub checkout which
already includes the data. A Python download example:

```python
from pathlib import Path
from zipfile import ZipFile
from huggingface_hub import hf_hub_download

archive = hf_hub_download(
    repo_id="Qiaoyuan/TUPLE-Bench", repo_type="dataset",
    filename="tuple-bench-dataset.zip", token=False,
)
with ZipFile(archive) as zipped:
    zipped.extractall(Path("evaluation-release"))
```

Install `huggingface_hub` separately for this optional download path.
The archive's SHA-256 is
`27abf9c9a77efb6b848ffd3ead16ea5577ff8ee91b88993a890f1980d7e38fdd`.
`DATASET_SHA256SUMS.json` checks the 37 dataset payload files. The database
and cached responses are identical to dataset v1.0.0 in the review release.

## Canonical files

| Path within `tuple-bench/` | Purpose |
| --- | --- |
| `data/benchmark/cases.jsonl` | All 1,440 canonical nested worlds |
| `data/benchmark/bases.jsonl` | All 720 paired public specifications |
| `data/benchmark/certificates/equivalence.jsonl` | Executable structural equivalence records |
| `data/benchmark/manifest.json` | Frozen counts and dataset checksums |
| `data/benchmark.sqlite3` | SQLite mirror: `worlds`, `pairs`, `metadata`, `public_cases`, `model_inputs` |
| `data/responses/` | Final coarse native responses for nine models |
| `data/refusal_responses/` | Separate final refusal-probe responses |
| `data/models.json` | Recorded provider model versions and coverage |
| `data/folds.json` | Frozen training folds |
| `data/review_issues.json` | Retained adverse review evidence and relation counts |

Each canonical case contains `case_id`, `domain_id`, `contract_id`,
`model_view`, `consumer_view`, and `evaluator_only`.

- `model_view`: recipient role, permission, public state plus private
  diagnostic evidence, and the coarse question. This is the only view to
  pass to a target model.
- `consumer_view`: recipient-visible public context, coarse candidate
  definitions, and downstream context. This view excludes the private report.
- `evaluator_only`: labels, hypotheses, pair and family IDs, split, check
  answers, and provenance. Use identifiers for grouping, never prediction.

Structural certificates verify shared public specifications and simulator
behavior. They do not establish semantic validity or equal informativeness
of the two private narratives.

## Hugging Face viewer tables

The top-level `train.jsonl`, `validation.jsonl`, and `test.jsonl` are a
lossless projection for the dataset viewer and `datasets.load_dataset`.
They keep the original split and serialize nested objects as JSON strings.

| Field | Meaning |
| --- | --- |
| `schema_version`, `case_id`, `domain_id`, `contract_id` | Original schema, world, and contract identifiers |
| `pair_id`, `source_family_id`, `rendering_family_id` | Grouping/provenance only |
| `split` | Original train/validation/test assignment |
| `authorized_category` | Gold coarse `T`, evaluator-only at action time |
| `protected_mechanism` | Gold fine `S`, training/evaluation target |
| `secret_hypotheses` | Registered pair of mechanism identifiers |
| `model_view_json` | JSON serialization of the original model view |
| `consumer_view_json` | JSON serialization of the original consumer view |
| `evaluator_only_json` | JSON serialization of the complete evaluator-only view |

```python
import json
from datasets import load_dataset

dataset = load_dataset("Qiaoyuan/TUPLE-Bench", token=False)
row = dataset["train"][0]
case = {
    "schema_version": row["schema_version"],
    "case_id": row["case_id"], "domain_id": row["domain_id"],
    "contract_id": row["contract_id"],
    "model_view": json.loads(row["model_view_json"]),
    "consumer_view": json.loads(row["consumer_view_json"]),
    "evaluator_only": json.loads(row["evaluator_only_json"]),
}
```

Use the complete ZIP for `tuplebench` evaluation: the viewer tables alone
do not include native outputs or fitted software artifacts.

## Data quality and intended use

The cases are AI-authored hypothetical incidents. The accepted frozen
development dataset retains three B-review uncertain pairs, one invalid
quote record, two related six-pair groups, and other adverse cross-family
relation evidence. Acceptance for evaluation does not mean every judge
approved every case. These flags are retained rather than relabeled as
either universally invalid or universally passed.

Use this dataset to study recipient-relative disclosure from typed
decision fields and specified downstream workflows. It does not estimate
real incident prevalence, industry loss, or general privacy-norm compliance.
