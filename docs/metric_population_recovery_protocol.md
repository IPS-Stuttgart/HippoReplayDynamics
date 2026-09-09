# Population Metric Recovery

Frozen before computation on 2026-09-09. The question is whether soft likelihood
aggregation can recover a known population composition despite weak individual
physical/neural classification. This is NOT fuzzy trajectory classification,
constant-speed recovery, held-out prediction, or real replay model selection.

## Design

Start at `2d2a001313265f98cfe1cb46cd40e3e3d468637b` and require the passing
`metric-oracle-recovery-all33-20260909` audit. Reuse its fixed RUN rate maps,
physical/full-neural/training-neural kernels and native full-bin count-profile
library, but not its generated paths, spike draws or likelihood rows.

All 33 recordings and nine animals remain. There are 50 independent Monte Carlo
replicates of three populations. Let phi be the neural fraction among moving
events. Use phi = .25, .50 and .75; moving mass .60, stationary .20, independent
position .20. Draw each event's generator independently from these probabilities.
The shared composition is deliberately identical across recordings; population
heterogeneity would require a separate validation before biological inference.

Each recording provides 128 freshly simulated events per replicate/scenario.
Also analyze the first 32 as a nested smaller cohort. This corresponds to
256/1024 events in Pfeiffer/Foster and 800/3200 in Tanni, with equal event counts
per recording and equal recordings per animal. No evidence or simulated label
selects events. Reusing empirical count profiles does not reuse paths or spikes.
Seed is 20260909 with a new `population_v1` namespace.

Use the original cell partition zero and separately condition multinomial
identities on its training and held-group totals. Both groups contribute to
whole-event likelihoods. Do not call this held-out predictive validation.
Uniform initial position, 20-ms bins, all four models, no path optimization.

Conditions:

- `exact`: matched observations, exact full generating neural metric.
- `train_geometry`: same matched observations, substitute training-cell metric.
- `gain_drift`: per-cell multiplicative lognormal gain SD .35, generated once
  per recording with a new namespace, then fixed across replicate events;
  observations change, but decoding uses unperturbed rates and full geometry.

The gain condition is a misspecification diagnostic, not a different biological
generating mixture. No diagnostic can replace a failed primary exact condition.
Rate-map estimation error and between-recording composition variation are not
tested in this first population screen.

## Estimation And Intervals

Fit nonnegative weights for all four event likelihoods jointly, by maximizing
the sum of log mixture likelihoods. Do not average event posterior class
probabilities or count best-model labels. The moving mass and static/IID balance
are estimated nuisance parameters, not fixed to their true generating values.

Report phi = neural_weight / (physical_weight + neural_weight). Profile out
the two nuisance weights at fixed phi, and invert a likelihood-ratio cutoff
3.841458820694124 for nominal 95% intervals. These intervals are asymptotic;
their coverage MUST be measured in the simulation, not assumed. Keep mixture
weights at least 1e-9 solely for numerical optimization. A KKT certificate is
required for each simplex optimum. Flat profiles give wide/uninformative
intervals, not positive claims.

Each replicate provides one estimate per dataset/cohort/condition. No Monte
Carlo replicate is a biological animal. Native counts and 128 events per
recording are primary; 32 is a sample-size diagnostic. No fourfold count arm.

## Decision Rule

Within each dataset/condition/cohort report bias, RMSE, interval width, coverage,
false directional claims when phi=.5 and power for both phi=.25/.75. Also report
Wilson intervals for the Monte Carlo proportions, since 50 repeats give limited
precision. A practical screen requires absolute bias <= .10, coverage >= .90
for each scenario, false direction <= .05 under the equal mixture, and direction
power >= .80 for each enriched mixture. These finite-simulation criteria are
chosen operating targets, not population guarantees or an equivalence test.

The exact-condition primary requires both datasets to pass at 128. A robustness
flag additionally requires training-geometry and gain-drift conditions to pass.
Even a robust pass only justifies further recovery under heterogeneous and
estimated models; it does not authorize real-data mechanism claims. A failure
must not be repaired by pooling Monte Carlo replicates or dropping null events.

## Audit

Save every simulated path, count array, profile identity, generator label and
four-model likelihood. Verify source hashes, native spike totals, complete
cohorts, and probability normalization. Independently reconstruct likelihoods
by backward recursion, verify all fitted simplex optima with KKT conditions,
check profile boundary likelihood ratios and reconstruct summaries. Preserve
provenance and failed artifacts. No new datasets or biological scoring.
