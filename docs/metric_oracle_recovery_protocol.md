# Exact-Generator Oracle Recovery

Frozen before computation, 2026-09-09. The finite 40-ms forecast failed practical
recovery. This follow-up distinguishes teacher separation, observation loss,
and estimator loss. It does not change the previous gates or rescore real events.

## Fixed Inputs

Reuse all 33 recordings, 9 animals, 256 simulated path/profile families per
recording and five observation realizations from
`metric-finite-recovery-all33-20260909`. Its manifest SHA256 is
`ad202b19f9b187f85f83cb5700f46860b43ecd679764d6278cc5cabb0c5099f8`.
Require its passing independent audit. No new paths, spike draws, event
selection or parameter optimization. Use matched-map `base` observations;
native counts are primary and fourfold counts diagnostic. Gain drift/map-error
validation would be a subsequent step if even this optimistic test succeeds.

## Five Comparisons

1. `latent_path`: exact probability of the saved full latent path under the
   physical, full-population neural, stationary and iid-position generators.
   Initial position is uniform. This receives the entire true latent path,
   not neural observations. One score per path, not repeated by split/support.
2. `whole_exact`: exact whole-event spike probability under those generators,
   integrating all latent paths. Use the known RUN maps and all simulated
   identities in every bin; initial position remains unknown/uniform.
   The emission is the product of the separate train/held multinomials,
   conditional on the source generator's exogenous group totals, not a
   pooled multinomial with a different conditioning convention.
3. `whole_train_geometry`: same observations, likelihood and inference as #2,
   but replace the neural transition with the training-cell-only RUN estimate.
4. `lagged_full_geometry`: the previous independent-bin origin decoder and
   40-ms held-cell forecast, replacing only the neural kernel with full RUN
   geometry. Held replay spikes never update the training-origin posterior.
5. `lagged_train_geometry`: the previously saved per-split matched-map,
   decoded-origin forecast scores. Copy, do not rescore.

With equal physical/neural priors and known correct simulated generators,
the maximum-likelihood choice in #2 minimizes expected binary classification
error for those observations; #1 is a richer-information diagnostic. These are
model-specific ceilings, not universal information limits on real replay.
Whole-event likelihood uses future and held spikes for classification and is
NOT held-out predictive validation. Full RUN maps and held replay spikes are
distinct sources: full RUN geometry itself need not leak held replay activity.

## Replication Unit And Calibration

Each split contains a different simulated observation draw, not disjoint
evidence about one observed event. Classify each realization separately;
then average accuracy/detection indicators across the five realizations of a
path/profile. Do not median/sum their likelihoods to improve classification.
This differs from the previous report's median-contrast classifier; compare
to the old *per-realization* scores under the new common reduction explicitly.

Retain calibration trials 0-31 and evaluation trials 32-63. Separately for
each recording, method, support and realization, calibrate the statistic
`max(physical, neural) - max(stationary, iid)` from the two null generators.
Take the larger upper conformal order statistic, alpha .05 and rank
ceil((32+1)*.95), and use strict > with tolerance 1e-9. Missing/nonfinite null
thresholds mean abstention. Numerical binary ties receive half credit.
No-spike observations cannot be detected; lagged arms additionally require
origin training and held-target support. Latent paths are observed even if
their associated spike counts are zero. Stationary path probability is
exactly zero (-infinity log probability) if the path changes location.

All four generators and every evaluation trial remain in the summaries.
Bootstrap trial/profile indices jointly across generators within recording,
after averaging realization indicators (2,000 draws, seed 20260909 with an
oracle namespace). Equal recordings within animals, equal animals. Intervals
are conditional simulation uncertainty, not biological population intervals.

Report binary accuracy, moving-generator detection, null FPR, winner fractions,
per-animal/session results and paired accuracy differences:
whole_exact minus whole_train_geometry; lagged_full_geometry minus
lagged_train_geometry; whole_exact minus lagged_full_geometry; and latent_path
minus whole_exact. The third contrast combines information usage and inference,
not a pure algorithmic effect on identical input subsets. Physical/neural
scores from different target sets are never directly subtracted across methods.

## Decision Boundary

Keep the prior practical operating rule: accuracy CI above .5, every animal
above .5, moving detection >= .5 for both generators, each empirical null FPR
<= .05 and finite calibration. Primary is native-count `whole_exact` in both
datasets. Report latent-path and other method/support gates diagnostically.
Do not let diagnostic successes replace a primary failure. The .5 detection
floor is a chosen operating target, not an identifiability theorem.

If the exact latent paths separate but whole observations do not, the observed
spikes limit this specified classification task. If whole_exact improves over
the forecasts, the prior failure also reflects discarded information/estimation.
If true paths already separate poorly, the matched generators themselves are
too similar for reliable per-event discrimination. Small samples and intervals
must qualify each conclusion; do not infer intrinsic impossibility solely from
a failed practical gate. A positive test only motivates a new, uncertainty-
validated estimator and independent recovery, not a real-data mechanism claim.

## Verification

Freeze code/protocol before scoring. Test forward evidence against explicit
path enumeration and the pinned hmmlearn 0.3.3 implementation; verify static/iid
closed forms, absent-spike behavior, time-index resets, copying of baseline
scores, no latent-path input in observation classifiers, complete factors and
calibration/evaluation separation. Independently reconstruct the actual score
outputs, reductions, thresholds and conditional bootstrap summaries. Preserve
failed artifacts. No fuzzy continuity, constant-speed or speed-uniformity claim
is tested here. The high-importance discovery objective remains open.
