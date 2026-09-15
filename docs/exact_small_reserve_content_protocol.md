# Exact small-budget reserve acquisition

Frozen2026-09-15 before evaluation. This is a new development rule, not a
successful reinterpretation of the nonlinear reserve search.

## Unchanged scope and validation

Retain all four original PF pairs, all original cells and their original
interpopulation overlap, and the fixed1,836 candidate/513 accepted endpoints.
Additions must be outside BOTH original populations and disjoint across sides.
Use equal acquisition budgets on the two sides, with budget chosen in TRAIN only.
Use the same maps, spatial grid,20ms independent Poisson decoding and flat prior.
No population, class, session, rat or hard event may be removed to pass.

Reuse exactly the audited native-Q3/cal_poisson_gain1/cal_conditional grouped
train/validation partition. Native250ms parent groups remain atomic. All training
rows are used. These previously inspected development banks are not pristine
independent validation. Freeze all four choices before any validation scoring.

## Exact enumeration and factorization

Candidate budgets are1 and2 when2*budget<=reserve size. For every population,
enumerate ALL unordered subsets of reserve cells of that budget. Retain original
cells unchanged. Each subset is decoded using the FULL joint Poisson likelihood.
Summing log-likelihood contributions is exact; summing changes of posterior loss
would be an approximation and is not used.

Each side has12 accuracy risks:3 sources x2 true classes x2 metrics (mean
Euclidean posterior-mean error and Home Brier). Store every subset's12 risks and
two native conditional mean Home probabilities. A subset is safe only if every
risk is <= its baseline+1e-10. Store unsafe subsets too.

For each budget, compare EVERY high-safe x low-safe combination. Record overlap,
objective and eligibility even when a combination shares a new cell. Disjoint
combinations are admissible if they reduce native

J = mean_k((mean(q_high(Home)|class=k)-mean(q_low(Home)|class=k))^2)

by>1e-10. Per-population likelihoods and risks do not depend on the other side's
added cells. Therefore a pair failing an individual-side risk cannot become safe
through pairing; pruning these pairs is exact. The class-mean decomposition of J
is also exact. This is NOT the mean of squared observation-wise differences.

Choose the admissible minimum J over both budgets. Ties within1e-12 choose smaller
budget, then lexicographic high indices and low indices. None admissible means
unchanged baseline and failure, not a successful zero-cost remedy.

There is no optimizer, stochastic rounding, iteration cap or random search in
selection. For the actual reserve sizes34/36/30/21, enumerate3,914 population
subsets in total (242 one-cell and3,672 two-cell). The completeness certificate
only applies to these frozen budgets, pools, maps, training samples and point
accuracy guards. It does not rule out larger budgets or other modeling choices.

## Independent verification

Test exact factorization against brute-force full pair decoding in synthetic
cases, including beneficial joint additions, missing subset rows, a safe but
non-improving fallback, changed held-out counts, disjointness and tie-breaking.

The serialized auditor must reconstruct EVERY individual subset with an
independent full-Poisson implementation, including count constants. It must
verify exhaustive combination identities, all12 risks, native class means,
individual eligibility, every safe-pair cross-product entry and training winner.
Directly reconstruct the full joint likelihood for every disjoint safe pair to
cross-check J and all24 risks independently of the factorized table. If large,
the finite cross-product still runs to completion; do not silently sample it.

Verify hashes, original membership/overlap, all parent-group partitions,
freeze-before-validation chronology, all96 validation risk keys, and non-vacuous
gates. Require all four pairs augmented, all96 validation risks nonworse and
strict native validation-J improvement in each rat with sessions equally weighted.
Failure stops before Q4/test-bank/replay/independent-recording evaluation.

An independently audited internal pass only permits the separate192-guard truth
preflight. Then all original endpoints, early/full maps and20 equal-budget random
acquisitions must be evaluated. Keep the original20% Home-gap reduction,10%
separation/TV reduction, per-rat, accepted-segment, entropy and known-position
accuracy requirements. Independent recordings are mandatory for the active goal.
No remedy is validated by training agreement or internal feasibility alone.
