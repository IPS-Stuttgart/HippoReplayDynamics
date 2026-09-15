# Exhaustive screened three/four-cell acquisition

Frozen2026-09-15 before evaluation. The previous exact one/two-cell search failed
the full-cohort gate, but demonstrated that safe joint additions can consist of
individually unsafe cells. Larger acquisitions therefore need their own test.

## Unchanged objective and scope

Retain all four original PF pairs, all original cells and original overlap. Add
equal-size disjoint sets from outside BOTH populations. Preserve all1,836 fixed
candidate endpoints and513 accepted-segment endpoints. Do not change maps, grids,
the20ms independent Poisson likelihood, flat prior, truth classes or safeguards.
No replay event, session or rat may be removed to obtain a successful result.

Use the exact audited grouped train/validation split from the one/two-cell run.
Native250ms parent observations remain atomic. Calibration banks and nativeQ3
are already inspected DEVELOPMENT data, not pristine independent validation.
Freeze the full cohort's choices before decoding any validation rows. No
post-validation replacement of winners is allowed.

## Complete subsets, not unsafe-prefix pruning

Enumerate every unordered three- and four-cell reserve subset on each side when
2*budget<=reserve size. The real pools34/36/30/21 imply314,370 complete population
subsets. Budgets1/2 are not retested. No optimizer, seed, truncation, random search,
fractional allocation or bounded proposal bank is used.

A complete subset's12 risks are checked in this fixed source/class order:

1. Poisson calibration, non-Home.
2. Conditional calibration, non-Home.
3. NativeQ3, non-Home.
4. Poisson calibration, Home.
5. Conditional calibration, Home.
6. NativeQ3, Home.

Each stage evaluates mean Euclidean posterior-mean localization error and Home
Brier loss using ALL training observations in that source/class. NativeQ3's
two stages also supply the conditional mean Home posterior probabilities.

Stop evaluating a COMPLETE subset at its first risk>baseline+1e-10. Store all
computed risks, the number of evaluated stages and NaN for explicitly uncomputed
risks. Rejection is an exact certificate for this subset, not a claim about its
supersets. No subset is pruned because a smaller subset failed. A safe subset
must have all12 finite risks evaluated and nonworsening.

Evaluate every high-safe x low-safe same-budget pairing, retaining overlap flags.
Pairs are admissible only when new cells are disjoint and native

J = mean_class((mean(q_high(Home)|class)-mean(q_low(Home)|class))^2)

decreases by>1e-10. Select minimum J across both budgets; ties within1e-12 use
smaller budget, then lexicographic high/low cell indices. No admissible pair means
unchanged baseline and FAILURE. This is a calibration objective, not an observed
replay destination or event-wise squared difference.

## Independent certificate audit

Reconstruct every enumerated identity and EVERY evaluated risk using separately
implemented full Poisson arithmetic including count constants. The auditor may
cache its independently reconstructed original-population log likelihood; it
must not consume cached likelihoods or risk values from the producer.

Verify each first-failure certificate and that missing values correspond exactly
to uncomputed stages. For every safe subset, additionally reconstruct all risks
from the complete original-plus-added count matrix without the cache. Directly
score every disjoint safe pair and verify the complete pair cross-product,
objective, tie-breaking and chosen winner. Do not sample these certificates.
The audit does NOT compute unneeded later risks for already rejected subsets.

Check all source/output hashes, original membership/overlap, atomic partitions,
full-cohort freeze-before-validation times, all96 validation risk keys and
non-vacuous gates. Require all four pairs augmented, all96 accuracy measures
nonworse, and native validation-J improvement in each rat with equal session
weight. Numerical audit success is not evidence of a validated remedy.

## Promotion remains conditional

Internal failure stops before Q4, test banks, replay and independent recordings.
An audited internal pass only permits the192-guard truth preflight. Then require
the unchanged early/full-map20%Home-gap reduction,10%separation/regional-TV
reduction, per-rat and accepted-segment gates, nonincreasing entropy and accuracy,
and comparison against20 equal-budget random acquisitions. No improvement may be
manufactured by erasing content, sharing newly acquired neurons or dropping cases.
Independent-recording validation is still mandatory for the active goal.
