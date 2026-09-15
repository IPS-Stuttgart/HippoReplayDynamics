# Exact small-budget acquisition: local training solutions, no validated remedy

## Decision

The exhaustive one/two-cell rule fails the full-cohort development gate. It finds
15 training-admissible two-cell pairings in Rat1/Open2, including solutions missed
by the preceding nonlinear-rounding search. However, the single frozen winner
fails six held-out accuracy measures. The other three original pairs have no
training-admissible acquisition at either budget.

Do not promote this rule to Q4, frozen test banks, replay or independent recordings.
The active objective remains unfinished. No cohort, event set or safeguard was
reduced to obtain a successful result.

## Frozen scope

All four original PF pairs and their original cells/overlap were retained. New
cells came from outside BOTH populations and were assigned disjointly, with equal
budgets of one or two per side. The same maps, spatial grid, independent 20-ms
Poisson likelihood and flat prior were used. All 1,836 candidate endpoints and
513 accepted-segment endpoints remain frozen and were not rescored in this run.

The previously audited atomic-parent native-Q3 and two calibration-bank partitions
were reused without change. They are previously inspected development data, not
pristine external validation. Every population subset was chosen using training
rows only; all four choices were frozen before validation decoding.

Each side has 12 accuracy risks: three sources, two true-position classes, and
physical posterior-mean error plus Home Brier score. A safe subset must worsen
none of them (the original 1e-10 numerical tolerance). A disjoint safe pairing
must additionally improve native

J = mean_k((mean(q_high(Home)|class=k)-mean(q_low(Home)|class=k))^2)

by more than 1e-10. The training winner minimizes J, with the frozen budget and
cell-index tie-breaks. This is not an event-wise squared discrepancy or a replay
content estimate.

## Exhaustive results

| Pair | Reserve cells | Individual subsets | Safe high/low, budget 1 | Safe high/low, budget 2 | Disjoint safe pairs, both budgets | Admissible pairs |
| --- | ---: | ---: | --- | --- | ---: | ---: |
| Rat1/Open1 |34|1,190|0 / 0|17 / 0|0|0|
| Rat1/Open2 |36|1,332|1 / 0|21 / 1|19|15|
| Rat2/Open1 |30|930|0 / 1|0 / 1|0|0|
| Rat4/Open2 |21|462|1 / 1|3 / 0|1|0|
| Total | |3,914| | |20|15|

All 242 one-cell and 3,672 two-cell population subsets were evaluated, not sampled.
The two sides' likelihoods and accuracy risks are independent of the OTHER side's
cell assignment. Thus an individually unsafe subset cannot form a safe pair at
that fixed budget. The J objective can be recovered exactly from the two native
conditional mean Home probabilities. This factorization uses full joint likelihoods
within each population, not summed single-cell posterior effects.

For reserve size n and budget b, there are C(n,b)*C(n-b,b) disjoint, side-labeled
pairings. Summing budgets 1 and 2 gives 279,378 / 354,690 / 165,300 / 36,330
pairings in the four sessions, or 835,698 total. The factorization covers this
finite search space by excluding unsafe component subsets exactly. It does NOT
mean 835,698 full paired decodes were computed.

Only 22 cross-products of individually safe subsets remained: 21 for Rat1/Open2
at budget 2 and one for Rat4/Open2 at budget 1. Two Rat1/Open2 products shared
an added cell and were rejected. All 20 disjoint safe pairs were reconstructed
with independent full paired likelihoods. Fifteen improved J, all in Rat1/Open2.

This is a finite completeness result under the frozen samples, maps, reserve
pools and training point-estimate safeguards. No full-cohort one/two-cell rule
can meet those training eligibility conditions. It does not rule out larger
acquisitions, different information or another observation model.

## The frozen winner and held-out failure

Rat1/Open2 added cell INDICES high=[8,192], low=[117,169]. These are indices into
the immutable encoding arrays, not external unit identifiers. The original 106
cells per side were retained, giving 108 per side with unchanged shared cells.

| Readout | Baseline | Selected |
| --- | ---: | ---: |
| Native training J |0.0295698595|0.0293980952|
| Native validation J |0.0152849923|0.0152636764|

Relative reductions are 0.581% in training and 0.139% in validation. These are
small changes in the calibration objective; they are not reductions in the
unscored replay Home-content gap.

Six of this pair's 24 held-out accuracy guards fail:

| Source | Side | True class | Metric | Increase |
| --- | --- | --- | --- | ---: |
| Native Q3 |high|non-Home|Home Brier|0.0000244313|
| Native Q3 |low|non-Home|Home Brier|0.0000028295|
| Poisson calibration |high|non-Home|Home Brier|0.0000091504|
| Conditional calibration |high|Home|Home Brier|0.0002859600|
| Conditional calibration |low|non-Home|Home Brier|0.0000078518|
| Conditional calibration |low|Home|Physical error, cm|0.0731910|

These are small point-estimate increases, not evidence of statistical or
biological harm. They nevertheless fail the frozen safeguards. No threshold was
relaxed, and none of the other 14 training-admissible pairs was selected or
evaluated after observing validation.

The remaining three pairs retained their baselines. Consequently the total
validation table has 96 rows: 72 unchanged, 18 nonworsening rows for the selected
pair, and six failures. The all-pairs-augmented, internal-accuracy and every-rat
native-improvement gates fail. ready_for_truth_preflight=false.

## Independent audit and provenance

- Code/protocol commit: 2460db7f3fb169142ee7ca51fab2e650aa7d5f7e.
- Branch: test-cross-dataset-content-stability.
- Server: gpuserver6000.
- Root: /mnt/seagate10tb/florianpfaff/exact-small-reserve-20260915.
- Producer: exact-small-reserve-20260915.service, exit 0,
  2026-09-15 06:07:06--06:08:16 UTC.
- Auditor: exact-small-reserve-audit-20260915.service, exit 0,
  2026-09-15 06:09:38--06:16:17 UTC.
- Both used detached systemd user services with RemainAfterExit.
- Manifest SHA256:
  fb03a4c950c05c0f561c67b59a1026c1bbec974c3455a8bc68a3b09b42d6c338.
- Audit SHA256:
  ee4c19d3d093b1a70f19215afb21a5112f75ffbe500a8250ceafa6d5209bb379.
- Relevant tests: 98 passed; Ruff and staged whitespace checks passed.

The independent implementation uses the full Poisson likelihood including count
constants, without the producer's cached original likelihood. It reconstructed
every one of the 3,914 population subsets and all 46,968 corresponding accuracy
values (13,635 failed guards). It verified all conditional probability means,
safe-state classifications, combination coverage, all 22 safe cross-products,
20 direct disjoint paired decodes and all 96 validation risks.

The audit also checked input/output hashes, original membership and overlap,
atomic-parent partitions, the full-cohort freeze-before-validation chronology,
the exact training winner and all non-vacuous gates. The numerical audit passes;
the remedy/readiness gate fails. An audit pass is not a scientific success.

The manifest's dirty flag reflects unrelated pre-existing untracked regional-bound
files. They were not edited or committed, and no tracked experiment source was
changed during either run. A pandas warning about equivalent empty CSV values
(NaN versus None) did not affect the current reconstruction or gate result.

## What this changes about the next step

The preceding bounded search missed real training-admissible allocations, so its
failure was not a proof of discrete infeasibility. The exact search now closes
that uncertainty for budgets one and two, while revealing that the recovered
training improvement is small and does not satisfy held-out accuracy.

Importantly, Rat1/Open2's low population has no safe one-cell addition but does
have a safe two-cell addition. Accuracy safety is therefore not monotone in the
number of added cells: an unsafe individual addition cannot be used to prune
all larger sets containing it. Exact pruning here only concerns pairing complete
same-budget high/low population subsets, whose likelihoods are separate.

Do not repeat one/two-cell searches with different seeds or optimizers, and do
not substitute the one locally promising session for the original cohort. Any
continued acquisition test needs a genuinely larger budget or different strategy,
unchanged accuracy/replay/random-budget safeguards and independent-recording
validation. No validated remedy or predictive external diagnostic exists yet.
