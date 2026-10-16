# Results and evaluation scope

Main README tables are frozen development results, not a live leaderboard.
Re-execute them with APIs in [the reproduction guide](reproduce.md).

## Primary audit

Five release views share each world's saved native response. A legitimate
consumer does not see gold `T`; the attacker is conditioned on authorized
`T` and two registered hypotheses. Worlds are decoded independently.
No-output inputs are identical within a pair whose secret labels differ,
so pooled recovery is exactly 50%.

Full native produces 62.85–75.35% binary recovery across nine models,
versus 51.04–57.64% for choice. Adding native confidence to choice raises
pooled recovery in every model, with inconsistent utility benefit.

Under authored costs, full native lowers pooled mean cost relative to
choice in all 27 model–task comparisons. These are separate utility
policies fitted for each view. This does not imply that every extra field
is useful or that every contrast is statistically significant.

## Defense audit

All main probability baselines and WCPQ use the same eighteen frozen
consumers: three workflows under six cost profiles. Defense-aware
attackers are fitted on defended training releases and selected on
validation. Features include inferable code information and public
workflow signatures. This gives a different native decoder from the
primary audit without changing native responses.

Strict WCPQ gives 58.68%, 51.04%, and 51.74% recovery for TypeSafe, OpenJev,
and Clef. Relative to its native controls, reductions are 10.07–24.65
percentage points. Every declared action branch and cost remains
identical, including existing errors and native fallbacks.

Strict equality concerns declared workflows and their simulator. It does
not imply calibration, confidentiality, or preservation for future
policies. Native fallback means the full release need not have finite support.

Temperature smoothing is publicly invertible. Evaluation includes its
public inverse; finite-sample selection can nevertheless produce slightly
different recovery from native. Gaussian uses three fixed realizations
and preserves 288 distinct test worlds.

The learned extension checks every hypothetical coarse truth without
reading the target label. Budgets 0, 0.04, and 0.20 scale public costs,
rather than observed mean cost. Zero budget permits action changes if no
hypothetical cost increases. Positive budgets can materially increase
realized cost. Pair-loss learning showed no consistent extra privacy benefit.

## Refusal probe

The separate probe presents sixteen protected mechanisms plus withholding.
Whole-dataset recovery combines correct direct selections and unique,
correct residual winners after withholding. Overall recovery is
91.88–99.93%. This is an all-world ranking diagnostic, not the held-out
two-hypothesis decoding task used in the primary audit.

## Uncertainty and limitations

Intervals use 2,000 source-family-clustered bootstrap resamples,
conditional on fixed fitted models and saved responses. Gaussian
intervals also condition on recorded draws. They exclude refitting,
selection, and repeated-provider-call uncertainty.

Review flags and cross-family relations remain in the dataset. Grouping
establishes declared lineage separation, not general semantic independence.
Test cases were inspected during development, without independent
new-source confirmation.

Costs are authored units and checks are idealized deterministic functions
of the coarse truth. The experiments do not measure real-world incident
frequency, monetary loss, server-side stochasticity, or formal privacy guarantees.
