# Kleinman short-window content calibration

Frozen before any synthetic recovery or biological reward contrast in this run.
This is a bounded measurement check, not a biological finding or equivalence test.

## Cohort and maps

- First chronological RUN-decoder-pass session from each of the six animals.
  Sort by animal and released session name, not decoder accuracy or replay results.
- Refit the frozen RUN encoding procedure on all eligible RUN traversals, keeping
  directional 2 cm maps, 4 cm smoothing, unit inclusion and raw occupancy support.
- Save exact maps and source hashes. These RUN-derived maps are the known maps
  for simulation and inference. No simulated map-estimation error in this test.
- Do not silently repair unsupported states: report the fraction of each true
  path that is supported. No trajectory is supplied to the decoder.

## Fixed bank

For both track ends, generate reverse paths using incoming-direction fields:
from the visit threshold inward by 25%, 50%, or 75% of the inter-threshold span.
Include stationary controls at the same thresholds. These are threshold-anchored
paths, not a claim to start exactly at the physical reward well.

- Durations: 100, 200, 400 ms.
- Expected event spikes: 24, 48, 96, obtained by one uniform multiplicative gain
  over all selected cells and time points. Actual spike counts remain Poisson.
- Timing: linear, cosine ease-in/ease-out, and pause-step (hold first/last quarter,
  linear travel in the middle half). Static controls have one timing profile.
- 16 repeats per condition, seed 20260923. No extra repeats to seek significance.
- Generate independent Poisson counts in 1 ms intervals with the known discrete
  spatial map at the true position. Replay decodes use wholly contained 40 ms
  windows, advanced by 5 ms. Overlap is not independent replication.
- The 17,280 distinct synthetic events are each decoded in three arms. The same
  generated spike counts are used across arms.

## Decoder arms

1. `matched_gain_poisson`: correct map and generating gain. Oracle gain is known,
   but the path is not; this is a recovery upper-bound, not the real-data method.
2. `run_rate_poisson`: same map, gain fixed to 1. Explicit gain-mismatch stress.
3. `count_conditioned`: multinomial/composition likelihood conditional on each
   window's total count, using state-normalized cell rates. Removes global gain,
   not cell-specific gain or dependence. Zero-count windows are uniform on support.

All arms decode independently with a flat prior over supported position/direction
states. No HMM, motion prior, temporal smoothing or truth-conditioned inference.

## Observable endpoint and diagnostics

Primary calibration endpoint is signed **late-minus-early represented displacement**:
mean posterior-mean position in the last quarter of decoded window centers minus
the first quarter, oriented away from the starting threshold. Compare to the same
functional of true *window-averaged* positions. This is NOT full replay length;
no extrapolation to unseen endpoints and no assumption of constant speed.

Report bias, absolute error, recovery of short vs long paths, gain/count effects,
directional posterior mass and weighted posterior time-position correlation.
An illustrative reverse-content call requires: dominant directional mass >=0.55,
weighted correlation in that direction with absolute value >=0.5, movement
opposite that direction, and signed represented displacement >10% of track span.
No continuity trimming or selection is used to estimate displacement. This call
is an explicit adaptation, not a bit-for-bit reproduction of the authors' rule.
Static false calls are reported, especially because windows overlap.

## Fixed readiness screen

Primary engineering screen uses 200 ms, linear timing (static controls unchanged),
expected 48 and 96 spikes. Evaluate each animal/end separately, with no pooled
rescue of a failing animal. Required in all 12 animal/end strata:

- All outputs finite; all six animals and both ends present.
- True state support >=95% in all primary paths.
- Median absolute displacement error <=10% of inter-threshold span.
- Median decoded displacement for 75% paths exceeds 25% paths at each count.
- Absolute median signed error <=10% of span at each count/length.
- Difference of median signed errors between 48 and 96 expected spikes <=10%
  of span for each path length (including static).
- Static reverse-content call fraction <=10% at each count.

Report each arm separately. The oracle arm alone cannot authorize real-data
inference; at least one non-oracle arm must pass before a reward-content contrast
uses this endpoint. A failing screen means this first-session bank does not
justify a general analysis; it does not prove all 127 sessions unusable.
Other durations/counts/timing profiles remain mandatory stress reports, not data
to optimize thresholds. These engineering tolerances are not biological bounds.

## Claim boundaries and provenance

Reward modulation of replay rate/localization, fidelity and speed already has
substantial prior work. Berners-Lee et al. also tested reward-change effects on
duration/slope. The proposed backward-content-vs-recruitment contrast is not yet
certified novel. This calibration neither tests reward effects nor validates a
Bayesian-smoothing mechanism. Record commit, protocol/input/output SHA256, seed,
host, duration and terminal exit state. Preserve failures and do not retune here.

Sources: https://elifesciences.org/articles/99678 ;
https://doi.org/10.5281/zenodo.10368995 ;
https://pmc.ncbi.nlm.nih.gov/articles/PMC9514662/ .
