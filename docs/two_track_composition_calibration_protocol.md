# Known-composition recovery audit

Frozen after the real-data and known-track readouts, before this audit's results.
This is a measurement diagnostic, not a new confirmatory test of biological bias.
Do not rescore events or change maps, units, sequence thresholds or cohort gates.

Inputs are the six completed `measurement_calibration_v1` sessions. Each anchor
has paired generated tracks with known labels, five cell splits, full/half
coverage and ordered/shuffled copies. Pair everything by anchor, split and track.
The half subset repeat was (anchor rank + split) modulo five; a split does NOT
represent the same half-cell subset across every simulated anchor. This differs
from each fixed repeated-subset realization in the real experiment and must stay
visible. Do not claim exact calibration of real-data confidence intervals.

## Actual selection

For ordered and shuffled simulations separately, calculate true track-2 fractions
for all, full-accepted, retained, half-accepted, lost and gained groups. Compare
with mean evaluation posterior mass on exactly the same events. Report true
fractions for all selected events AND those with evaluation spikes; the latter
is the matched denominator for the posterior readout. Decompose retained-minus-
full, half-minus-retained and half-minus-full for truth and readout. Empty groups
are missing. Equal split summaries do not create additional animals. This is
descriptive, sparse simulation recovery, not a null p-value for real biology.

## Known selection-probability interventions

Use the paired ordered simulations without sequence conditioning. Both tracks
are equally frequent before selection. Apply these deterministic retention
probabilities to track 1 and track 2 respectively:

- (0.5, 0.5): no content selection, true shift 0.
- (0.6, 0.4) and (0.4, 0.6): true shifts -0.10 and +0.10.
- (0.75, 0.25) and (0.25, 0.75): true shifts -0.25 and +0.25.

These are instrument-calibration levels, not biological equivalence bounds.
Weight events analytically, without drawing extra spike or selection samples.
Report expected true shift, expected observed shift, bias, and recovery slope
mean(q_track2 | truth2) - mean(q_track2 | truth1). Use Poisson and conditional-
count readouts separately. Only paired finite evaluation readouts enter each
split. Average split estimates equally. Reconstruct conditional q from stored
signed log odds using the known label solely for orientation, not for decoding;
check Poisson reconstruction against its explicitly saved q column.

Uncertainty resamples count anchors jointly across both true tracks and all cell
splits (2,000 draws). Report 95% conditional calibration intervals only with at
least two informative anchors and >=95% finite draws. These condition on the
existing maps, generated spikes and count anchors. They are NOT animal-level
intervals, real-data confidence-interval coverage, or new independent simulation
replicates. The zero-shift intervention is an algebraic check, not an empirical
false-positive-rate estimate. Shuffled copies have identical sequenceless
readouts and are not added as independent observations.

Primary readout: POST ripple-positive count anchors in both strict RUN-pass
sessions. Keep PRE, MUA-only and all diagnostic sessions visible. Do not use this
audit to rescale the real biological result automatically: model-matched
monotonic simulated paths may not describe latent real replay content.
