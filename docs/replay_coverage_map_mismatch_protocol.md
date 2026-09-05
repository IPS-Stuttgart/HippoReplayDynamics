# Independent RUN-Map and Observation Mismatch

Frozen before production scores. This extends the original study requirements;
it is not a replacement for held-out calibration or real replay ground truth.

## Question and Sources

Does known-path recovery change when the decoder uses an independently estimated
RUN map rather than the exact synthetic generator map? Does a common fluctuating
gain change that comparison under Poisson versus conditional decoding?

Use all 33 cached PF/Tanni sessions and the 971 source duration profiles frozen
in `replay-coverage-recovery-all33-20260905`. Source profiles are deduplicated by
dataset/animal/session/source event. No selection on recovered trajectories.
Root seed 20260912; rate multiplier 3; base speed 1000 cm/s. One new synthetic
draw per source and half direction. The two directions are not new animals.

## Independent Map Construction

Split each session chronologically at the midpoint of its supported RUN extent.
Leave a one-second guard on both sides. Fit decoder A with training-only grid,
occupancy, cell inclusion and split-half stability QC using the existing RUN
validation helper. Fit generator B using only the other half's spikes/position.
Reverse A/B for a second comparison. The full-RUN map and full-RUN QC mask are
never used. Both halves use the existing 8 cm / sigma1.5 bins / >=10 cm/s RUN
encoding settings. At least five training-selected units and two decoder states
are required; failures remain in the planned-session table, not silently dropped.

Only A selects cells. A cell absent from B is retained with the encoding floor;
no B-based stability or peak-rate selection is applied. Record missing cells.
Interpolate B's full rate grid bilinearly, clamping at its grid-center edges as
in the previous benchmark. Evaluate B at A's states for the oracle decoder.
Thus both decoders have IDENTICAL cell identities, spatial states and occupancy
support; only their rate values differ. A's support is not expanded or masked
using B occupancy. Record B support separately and count unsupported truth as a
coverage miss. The oracle refers to rate knowledge, not unlimited spatial support.

Generate paths in the intersection of the two grid-center bounding rectangles,
clipped to native arena bounds where known. This uses both halves' position
extents to define a synthetic test domain, NOT to fit decoder A. It is not a
strict real-data held-out-position test and does not identify PF physical walls.
Do not redraw paths to avoid unoccupied bins or obtain successful decoding.
Half-map differences combine estimation noise, sampling and possible neural
nonstationarity; they are not automatically pure estimation error.

## Paired Observations and Decoding

Truth classes: g=-0.5/0/+0.5 horizontal speed gradients, stationary, independent
5 ms snapshots, and whole-5-ms-block shuffled constant-speed path/count pairs.
Source duration is truncated only to complete 5 ms bins, recorded explicitly.
Path integration and observations use 1 ms resolution.

Generate independent Poisson counts using B rates and a separate shared-gain
stress condition. The gain is constant in each 20 ms block, lognormal with mean
one and coefficient of variation one, independent of path/position. It induces
overdispersion and temporal/population dependence. This is a declared stress
model, not an empirical estimate of replay correlations. Poisson and conditional
decoders both ignore the latent gain. Conditional decoding removes population
intensity information but is not guaranteed exact for a moving path inside a
window. Record realized gains and simulated/source count ratios.

Full and deterministic nested half-cell observations come from each identical
parent array. No restoration or relabeling. Decode every observation using:

- generator-known B versus independent RUN-half A rates;
- Poisson versus conditional multinomial likelihood;
- full versus half training-selected cells;
- MAP versus posterior mean;
- unfiltered versus >=2 cells / >=3 spikes per window.

Use independent uniform spatial priors, 20 ms windows, 5 ms continuity stride,
the fixed <20 cm / >=10 frames / >=40 cm displacement criterion, and
non-overlapping 20 ms speed steps with no bridging excluded windows. No HMM.
No retrospective threshold changes. The two map conditions always see exactly
the same count array and truth, including their support/selection denominators.

## Outcomes, Audit and Interpretation

Report position error, posterior HPD coverage, genuine-path recovery, null
acceptance, speed errors and equal-event gradient response before/after
continuity selection. Report availability and true spatial support. Pair map
contrasts within source/direction/condition, then average directions within
session, sessions within animal, animals equally. Bootstrap animals conditional
on the fixed maps and new synthetic draws. Do not treat repeated windows or
directions as independent biological replicates.

Save half-map models, training/generator hashes, source and cell IDs, path/count
hashes, per-batch metrics, clean commit and dependency/input hashes. Independently
reconstruct models and observations, verify disjoint data and direct support
counts, and check sampled likelihood/metric outputs. Technical gates cannot
require a positive map effect or successful biological decoding. Missing
encoding support is a separate readiness gate, not a zero-valued recovery.

This experiment tests map/observation sensitivity with known synthetic paths.
It does not calibrate replay posterior uncertainty, validate a correction on
unseen animals, establish an optimal threshold, or prove biological uniformity.
Those parts of the original objective remain active.
