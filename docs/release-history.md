# Release history

## v1.0.1 — Public documentation and Hugging Face distribution

Public release under the maintainer's GitHub and Hugging Face accounts.
Adds the detailed README, Chinese overview, data card, data viewer tables,
model connection, result guide, and public download links.

The canonical dataset, native responses, frozen models, evaluation
implementation, and numerical results are unchanged from v1.0.0.
The software checksum manifest is regenerated for documentation and
package metadata. The dataset ZIP remains byte-for-byte identical:

```text
27abf9c9a77efb6b848ffd3ead16ea5577ff8ee91b88993a890f1980d7e38fdd  tuple-bench-dataset.zip
```

Hugging Face viewer tables are reversible projections of the original
worlds and preserve the exact split. They do not replace canonical
records used by the evaluation APIs.

## v1.0.0 — Frozen review artifacts

Two archives contain software and the complete dataset. The initial Git
commit used an anonymous author for review. Historical
`validation/anonymity.json` describes that release, not an anonymity
claim for the current public repository.

Original review ZIPs and checksums are preserved. Original installation,
92-test, replay, cost, and fresh-fitting records remain in `validation/`.
Historical commits are retained without rewriting.
