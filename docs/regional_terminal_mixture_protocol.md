# Terminal-segment calibration under unknown dynamics

Frozen before this experiment's readouts, 2026-09-16. Development experiment
on the audited geometry-blind shared PF bank; no real replay inference.

## Inputs and target

Use all eight sessions/four rats, all 1139 common >=100-ms candidate templates,
and the unchanged first-half RUN encoders and 22 population definitions. Verify
the bank audit and hashes. Shorter original events remain excluded, never padded.
Use only calibration and validation panels; reserve the null panels unused.

Target: >=20 ms cumulative represented time in the 20-cm geometric Home disc
within the final Delta=20/40/60/100 ms. The latent path's exact geometric dwell
is truth. It is not instantaneous endpoint, average position, or a 2-ms touch.
Calibration and validation use each Delta's own conditioned bank. Positive
prevalence is held fixed across dynamics strata at each panel's realized quota.

## Readouts, fixed before evaluating error

Primary: independent_bin_any_home, continuous with an explicit zero atom.
Partition the final Delta into non-overlapping 20-ms windows ending at the frozen
native endpoint. Decode each independently with the existing Poisson likelihood
and uniform prior over valid spatial bins. No HMM or dynamical prior. A silent
subwindow has regional BF=0 (neutral), as in the previous diagnostic.

If q_j is each window's Home posterior mass, use q_any=1-product(1-q_j), and
subtract the log prior odds of 1-(1-area_fraction)^K from logit(q_any) to form a
scalar regional readout. This uses an independent-location surrogate inside
the decoder, NOT a claim that real trajectory bins are independent. In particular,
at-least-one decoded bin in Home is only a surrogate for continuous cumulative
20-ms dwell. Calibrate that scalar against the actual geometric truth; do not
declare it a calibrated content probability by construction.

Sensitivity: pooled_counts, one Poisson decode with exposure Delta and all
segment spikes, implicitly a static-location summary. It does not by itself
implement 'anywhere in segment'. Its calibrated errors test that limitation.

For each readout compare continuous_zero_mass (primary) to the unchanged
three-level BF calls using +/-log(3); zero-spike segments are neutral. Continuous
likelihoods retain the prior arcsinh/Scott/min-bandwidth-.05 KDE and explicit
zero-mass convention. No bandwidth, threshold or window tuning using validation.
Both summaries coincide with the previous neutral 20-ms regional BF at Delta20.

Fit calibration once per session/population/Delta/readout: equal weights for
the seven frozen geometry strata (stationary; moving and jumping at .5/1/2).
No simulated generator label is used to select calibration at inference.
Generator-specific continuous calibration is a separately labeled oracle
diagnostic on pure strata only, not an inferred-type method or attainable bound.

## Unknown-mixture stress test

Evaluate every pure stratum at pi=.05/.15/.30/.50 and four independent validation
replicas. All seven panels at a given session/Delta/pi/replica have the same
number of events and same true quota. Form mixture likelihoods by assigning
common nonnegative generator weights w summing to one to their empirical event
distributions. Thus prevalence stays fixed exactly. These are weighted empirical
mixtures, not newly simulated independent N-event panels.

For fixed pooled class likelihoods, each pure panel's prevalence log likelihood
is concave. A mixture's log likelihood is a convex combination of them, so its
MLE lies between the minimum and maximum pure-panel MLE. Consequently maximum
absolute error over ALL common generator weights occurs at a pure stratum.
Use this exact empirical envelope, with unit tests and numerical grid checks.
If a pure panel is unidentified, include [0,1] as its optimizer range and fail
the gate. This is an envelope of fitted estimates conditional on this finite
bank, NOT a confidence or identification interval for real content prevalence.
It does not cover mixtures that also alter within-stratum occupancy, spike
profiles, or class-conditional generator weights beyond the frozen bank.

Numerical verification uses the full quarter-weight simplex grid (210 mixtures)
plus equal seven-stratum and equal three-base-stratum mixtures, on one validation
replica per prevalence for the full population. These use no oracle labels for
calibration; generator identities define the externally specified stress test.

For each session/prevalence/replica report worst empirical mixture absolute error.
Development Delta* is the smallest primary-readout length with (a) maximum
session/prevalence mean worst-mixture error <=5 pp, (b) at least 90% of the 128
session/prevalence/replica estimates within 5 pp, and (c) no unidentified fits.
Report the raw maximum too, rather than hiding tail errors behind the mean.
This is more explicit than conflating maximum error and 90% success.

Repeat descriptively by population, animal, prevalence, dynamics and occupancy
fraction; show unconditioned geometry-reference prevalence beside every Delta.
The iid decoder reference and geometry-reference prevalence are distinct.
Show false-positive readout patterns in [0,.2), [.2,.5), [.5,.8), [.8,1] occupancy
strata, plus late-crossing strata, including empty cells rather than dropping them.

## Uncertainty and limits

Four validation replicas are development only. Report replica dispersion and
per-rat/session heterogeneity, plus descriptive four-rat cluster-bootstrap ranges
for averages. These do not certify nominal interval coverage or a 5% false-flag
rate. Calibration maps, count templates and four calibration replicas are fixed;
do not count synthetic replicas as independent animals.

No real-data Home-rich/Home-poor comparison, Delta* deployment or hc-11 transfer
in this run, even if a development length passes. A passing candidate requires
larger independent simulation confirmation. Failure means these frozen readouts
failed this robustness test, not that every possible terminal estimator is
mathematically non-identifiable or that dynamics inference must solve it.

Retain the bank limits: decoder-derived dynamics anchors from 108 selected
events, overlapping legacy populations, fixed total counts, cumulative-dwell
definition, and arena-limited realization of approximately 31% of requested
2x jump distances. No new bank generation or evidence rescoring.
