# Finite-Spike Physical/Neural Metric Recovery

Frozen before new simulations, 2026-09-09. The previous oracle-origin,
single-identity screen passed but measured very small distinctions. This
experiment tests usable recovery, not a new biological result.

## Fixed Design

Reuse all 33 RUN maps, five saved 70/30 partitions, and the independently
audited metric kernels from `physical-neural-metric-screen-all33-20260909`,
manifest SHA256
`71882248599186e0e790b408d1524a98d713d4c3220225004e00aafc05fde937`.
Keep the parent's occupancy/dwell/entropy matching convention unchanged.

For each recording choose 64 duration/spike-count profiles with replacement,
using seed 20260909 and a stable session-specific hash. Eligible profiles
have at least three complete 20-ms bins. Keep full windows; discard only
the original final partial bin. Do not select on scores, continuity, wall
distance, or event strength. Report profile eligibility and reuse.

Generate four latent paths for each profile: physical metric, full-population
neural metric, stationary, and independently uniform positions. Initial states
are uniform. The first two use the frozen parent kernels; controls use I
and uniform transitions. Paths are shared across cell partitions, observation
conditions and spike-support levels. Their physical speeds are stochastic,
not constant. No real replay event is rescored or classified.

Primary support uses the observed train/held total spike counts per bin.
The fourfold-count arm is an information-rich sensitivity, not a substitute
for a failed primary. Generate training and held cell identities separately
conditional on those totals and the latent position. This gives a correctly
specified conditional-identity observation model at matched encoding.
Counts are redrawn for each partition: repeated partitions share latent
paths/count profiles but are not one identical full-population spike sample.
Never multiply likelihoods across these partitions. Median paired scores
within a synthetic path/profile precede recording and animal reductions.

## Observation Conditions

1. `matched`: source RUN rates generate and decode spikes.
2. `gain_drift`: truth rates receive independent cell-wise lognormal gains,
   log SD 0.35 and distributional expected gain one; decoder keeps original RUN maps.
   Latent geometry is unchanged. This tests observation-transfer error,
   not a coupling of instantaneous rate gains to latent neural geometry.
3. `map_error`: truth and spikes are identical to matched; decoder maps are
   perturbed by an independent smooth multiplicative lognormal field, log
   SD 0.35 and correlation kernel scale 16 cm. Preserve each cell's uniform-
   spatial mean rate. Recompute training-cell neural geometry from those
   perturbed maps, retaining the same occupancy/dwell/entropy constraints.

All perturbations are fixed per recording before spike generation. These
are controlled stress levels, not estimates of actual RUN-map uncertainty
or replacement for independently fitted RUN halves. Matched-map generation
remains optimistic. No parameter is tuned to the recovery outcome.

## Prediction

Fixed 40-ms center lag (two bins), no other horizon in this experiment.
The primary origin is independently decoded from training spikes in the
origin bin using a uniform prior. No dynamics prior, earlier bins, intervening
or target training activity, or held spikes update that origin. Compare
physical, training-neural, stationary and iid future kernels. A known-origin
diagnostic uses the true starting position but otherwise identical predictors.

Score held target-bin count vectors with proper multinomial probabilities
mixed over the forecast destination. Zero-spike targets contribute zero.
Sum per-bin predictive scores; these are not joint sequence Bayes factors.
Save observations, paths, origins/targets, parameters, and complete score
tables for independent reconstruction. Verify held/future mutation invariance.

## Readiness And Calibration

Trials 0-31 are calibration; 32-63 are evaluation. Only stationary and iid
calibration trials set detection thresholds, separately by recording,
condition, support and origin arm. Statistic: best physical/neural forecast
score minus best stationary/iid score, after median paired contrasts over
partitions. Set the threshold to the larger of the two null conformal
upper order statistics at alpha .05, rank ceil((n+1)*.95). Use strict >;
numerical score differences <=1e-9 are ties. An event without held target
spikes or, in the decoded arm, training-origin spikes cannot be detected.
An unavailable finite null threshold means abstention, not a vacuous pass.

Report all four model winner fractions, binary physical/neural accuracy,
structured-detection power and each null false-positive rate. Binary ties
receive 0.5 accuracy rather than an arbitrary winner. Report results for all
evaluation trials, not only detected ones. Bootstrap Monte Carlo trials
within fixed recordings (2,000 draws), resampling shared trial/profile indices
jointly across generators, then equal recordings
within animals and equal animals; intervals describe simulation uncertainty
conditional on maps/perturbations/calibration, not biological uncertainty.

The primary native-count, decoded-origin, matched condition is ready only
if both datasets have balanced physical/neural accuracy lower CI > .5,
all animal accuracy estimates > .5, detection power >= .5 for each moving
generator, and empirical false-positive rates <= .05 for each null.
The .5 power target is an explicit practical operating requirement, not a
biological significance threshold or mathematical identifiability theorem.
Robust readiness additionally requires the two mismatch conditions to meet
the same rules. Fourfold support/known origins are diagnostic only.

Failure identifies limited information, estimator error or observation
sensitivity; it does not refute physical/neural propagation in the brain.
Passing still does not establish which geometry real events follow or a
high-importance discovery. Do not change thresholds, select favorable animals
or use sensitivity arms to rescue the primary. Broader novelty constraints
remain those documented in the parent protocol.
