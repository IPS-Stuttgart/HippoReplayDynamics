# Dense independent path integration for clock recovery

Frozen before production scoring. Goal: determine whether the large clock-mixture
bias for new paths is reduced by better path integration, without changing the
generator, observations, nuisance controls, or inferential target.

## Inputs remain fixed

Use all 1,267,200 observations of the audited
`unknown-path-clock-populations-all33-20260909` run, all 33 encoders (9 animals),
all 50 replicates, all three mixture fractions and both original teacher banks.
Read source count vectors and metadata, not previous evidence, into the scorer.
Verify parent manifest, audit linkage, every used count/metadata file and source
encoding cache. No real replay event is rescored, no observations are regenerated,
and no easier path subset is selected.

Original `matched`/`independent` teacher labels become `original_bank_0` and
`original_bank_1`. BOTH now have independent scorer libraries. The old word
matched must not describe a new scorer/teacher relationship.

## Scorer geometry and implementation

Two independent path libraries, with nested supports 1,024, 4,096 and 8,192.
The RNG namespace is `dense_path_clocks_v1|20260909|tag|bank|path|geometry`.
It is independent of both original teacher banks. Accidental overlap of paths
is allowed under the finite endpoint prior; true paths are neither injected
nor excluded. Repeated draws retain their Monte Carlo multiplicities.

Retain the original endpoint, straight/curved, coverage, and 801-knot geometry
prior and both literal physical/Hellinger-code clocks. Keep physical, neural,
stationary, physical-reset and neural-reset likelihoods. The stationary bank is
the full source grid. Marginalize, do not maximize, over every sampled path.

Stream paths in chunks of 128. Use exact piecewise-linear rate integration,
reusing antiderivatives across native time-bin counts. Sparse count multiplication
may replace dense multiplication but must produce the same likelihood. Keep
unnormalized log sums across chunks, then divide by the total path count.
Reset sums over paths independently at each bin. Never average per-chunk log
likelihoods, discard unfavorable paths, or condition a library on a true label.

Save all new likelihoods in a memory-mappable array with explicit axis metadata,
source observation identifiers and geometry descriptors. Frozen simulated counts
remain in the parent artifact and are referenced by hash. Save both successful
and failed run manifests. No stopping early on a favorable statistical result.

## Recovery and integration checks

Fit the same five mixture weights and profile intervals as the parent study.
Keep phi=.25,.50,.75, coherent mass .60, stationary .20 and each reset .10.
Population sizes remain 128 events per recording, not actual biological counts.

Recovery targets are unchanged: maximum absolute bias <=.10, minimum interval
coverage >=.90, minimum directional power >=.80, false directional claims at
phi=.50 <=.05. Report Wilson Monte Carlo intervals and the coarseness of 50
replicates, without changing gates after seeing a 3/50 outcome.

At 8,192 paths, numerical stability additionally requires median paired phi
difference between independent banks <=.025; median phi shift from 4,096 to
8,192 <=.025; median maximum profile-interval endpoint shift <=.05. These are
practical approximation tolerances chosen to be smaller than the .10 recovery
bias target, not biological thresholds. Also report p95 discrepancies.
Recovery must still be demonstrated; numerical agreement alone is insufficient.

The primary difficult corpus is original_bank_1, with original_bank_0 providing
a second fixed teacher library. Both datasets and both scorer banks must remain
visible; do not select the bank or support with the best apparent recovery.

## Independent verification and interpretation

Verify geometry from seed and source rates, check analytic integration against
independent knot-splitting/trapezoidal integration, and compare the streaming
sparse scorer against the prior exhaustive dense scorer on synthetic fixtures.
On the production run independently recompute all likelihoods for a fixed
stratified audit subset: 2 events per teacher/scenario/replicate per recording,
first by frozen event index, 600 observations per recording. This audits all
scorer paths, all supports and both banks, across all encoders, but is NOT a
claim that every production likelihood was independently recomputed.
Certify every population/profile optimum and recompute all summary/gate tables.

Even successful recovery is restricted to these rate maps and the straight/
sinusoidal path prior. This is not a fuzzy-continuity validation, a uniform-speed
result, or a biological discovery. If integration remains unstable, record it
and retain the stop on real-event mechanism claims. Dense-path convergence
cannot be replaced by giving the scorer the actual teacher paths.
