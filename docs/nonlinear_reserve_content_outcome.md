# Nonlinear reserve acquisition: development screen failed

## Decision

No validated remedy. The nonlinear proposal rule did not find an admissible
integer acquisition in any of the four original PF pairs. All original
populations remained unchanged. Do not advance this rule to Q4, test banks,
replay or independent recordings. The active goal is still unfinished.

This is a bounded development-search failure, not proof that no larger or
different acquisition design can work. Fractional solutions are only proposals;
they do not correspond to the required disjoint neuron samples.

## Provenance and verification

- Server: gpuserver6000.
- Code/protocol commit: a19cc1e8f372f08f437c9c29fb99c72a035477fd.
- Root: /mnt/seagate10tb/florianpfaff/nonlinear-reserve-20260915.
- Producer: nonlinear-reserve-20260915.service, exit0,
  2026-09-15 05:36:21--05:42:20 UTC.
- Independent auditor: nonlinear-reserve-audit-20260915.service, exit0,
  2026-09-15 05:43:52--05:46:15 UTC.
- Both were detached systemd user services, independent of SSH.
- Producer manifest SHA256:
  f073c5240dab87530961eabb591790ac1e00e957c207820a002aabf257d5fa33.
- Audit SHA256:
  e16123e4aa9ef954ed75c47cca51efd2efeebf4c198ddca106e97a4a7d77da4a.
- Relevant tests: 88 passed; Ruff and staged whitespace checks passed.

The manifest records git_dirty=true because unrelated pre-existing regional-bound
files were untracked. They were not edited, committed or used by this experiment.
The tracked experiment code was committed before the run; all registered input
and output hashes were independently checked after completion.

## Frozen design

All four original PF pairs, their original neurons and overlap were retained.
Additional cells came only from the reserve outside both populations and were
assigned disjointly, with equal numbers on the two sides. Budgets were1,2,4 and
the feasible maximum17/18/15/10. This changes acquisition budget, not the cohort
or final improvement requirements.

Selection reused the previously audited atomic-parent train/validation split of
native Q3, Poisson calibration and conditional-count calibration. These banks had
already been inspected in development and are not independent validation data.
All four selections froze before any validation scoring. No replay labels,
endpoints, evidence or validation losses influenced the search.

Fractional likelihood powers and exact derivatives proposed subsets through a
minimax stage and a constrained objective stage. Integer rounding used fixed
seeds and exact assignment with disjoint quotas. Every integer candidate was
rescored with the full likelihood, not an additive single-cell approximation.

Training eligibility required lower native
J = mean_k(mean(q_high(Home)-q_low(Home) | true_class=k)^2)
and no increase in all24 source/side/true-class physical-error and Home-Brier
risks (unchanged1e-10 numerical tolerance).

## Results

| Pair | Named candidates | Distinct allocations | J improved | Minimum failed risks | Admissible |
| --- | ---: | ---: | ---: | ---: | ---: |
| Rat1/Open1 |224|160|132|2|0|
| Rat1/Open2 |224|161|161|2|0|
| Rat2/Open1 |224|162|116|2|0|
| Rat4/Open2 |224|147|91|2|0|
| Total |896|630|500| |0|

Named proposals include duplicate memberships from different seeds/searches;
they are not896 independent statistical observations. The minimum-failure
column includes all candidates, not only J-improving candidates. Among
J-improving candidates the minimum is2,2,3,3 by pair.

Across896 named candidates and24 risks,5,733/21,504 point-estimate guards failed.
Many candidates improved physical localization or true-Home probability while
worsening false Home probability on non-Home observations. For example, the
low-Home-tuning population's non-Home Brier risk increased in758/896 candidates
on native Q3 and768/896 on Poisson calibration. Median increases across named
proposals were approximately0.000042 and0.000029. Here non-Home Brier is
mean(q(Home)^2), not a binary false-positive rate.

Some of the closest failures were numerically small. Rat1/Open1's best
J-improver among two-failure candidates reduced J from0.03185245 to0.03167430
but increased native non-Home Brier by0.00000532 (high population) and0.00001809
(low). This fails the frozen safeguard; it is not a statistical finding of
biological harm. Thresholds were not relaxed after observing the result.

The64 fractional searches recorded2,856 evaluations. Statuses were28 successful,
30 reaching the iteration limit and6 reaching the evaluation limit. Small-budget
fractional feasible solutions therefore did not translate into an admissible
integer allocation in this fixed rounding search. Neither a successful solver
status nor a finite-budget failure proves global optimality or infeasibility.

## Validation and audit

All four choices fell back to their unchanged originals. All96 validation risks
were identical to baseline. The all-pairs-augmented and every-rat native-J
improvement gates failed, and ready_for_truth_preflight=false. Equality with an
unchanged baseline is not a successful remedy.

The independent auditor reconstructed:

- All896 named integer candidates, with all21,504 training risks and objectives.
- All64 selected fractional proposal states using full weighted Poisson constants;
  maximum risk difference6.40e-14.
- All96 validation risks, exact partitions, population membership, original
  overlap, seeds, quotas, candidate coverage, training-only choices and gates.
- Trace allocation feasibility and selected-iterate rules for all recorded
  evaluations. Intermediate unselected fractional losses were not numerically
  reconstructed; optimizer optimality was not certified.

No Q4/test/replay/external evaluation occurred. The original1,836 candidate and
513 accepted endpoints were not replaced, filtered or retimed.

## One-cell boundary check

A supplementary read-only check used the earlier independently reconstructed
complete single-addition tables from guarded-reserve-20260915. Since each
population's accuracy depends only on its own added cell, every safe one-per-side
pair must use individually safe reserve cells on both sides. Across all reserve
cells, the allowable cell INDICES (not external unit IDs) were:

| Pair | Safe high-side cells | Safe low-side cells |
| --- | --- | --- |
| Rat1/Open1 |none|none|
| Rat1/Open2 |8|none|
| Rat2/Open1 |none|59|
| Rat4/Open2 |6|54|

The sole disjoint safe combination, Rat4/Open2 high6/low54, was reconstructed
with the independent full-Poisson implementation on the frozen training banks.
It preserves all24 accuracy risks but raises J from0.0331035121 to0.0332127642.
Consequently budget1 cannot produce an admissible allocation under these exact
training constraints, not merely under the sampled roundings. This finite
boundary result does not extend to multi-cell additions or independent data.

## Next unresolved question

The failed rule removes the additive-approximation explanation as a sufficient
fix, but does not exhaust discrete multi-cell allocations. An exact small-budget
search can exploit the independence of the two population likelihoods: evaluate
complete two-cell additions separately for each population, retain only sets
passing all own-population accuracy risks, then pair disjoint safe sets and
calculate the actual J. This avoids treating gradient/rounding failure as
infeasibility. It is a possible next development test, not an already implemented
or validated remedy. All original accuracy, replay, random-budget and independent
recording safeguards would still apply.
