# Empirical-Map Known-Path Recovery Protocol

Status: frozen development benchmark specification, before real-map results.
This is one required experiment in `replay_recording_coverage_study.md`, not a
replacement for the full calibration, factor-isolation, and equivalence study.

## Inputs and Denominators

Use all 33 cached sessions (eight PF/four rats; 25 Tanni/five rats, all arena
sizes). Draw 30 source candidates/session uniformly without replacement from
the complete original high-MUA candidate list. Seed 20260907; one nested random
cell ordering/session gives full, half, and quarter QC-unit populations.
No event selection on decoding, continuity, posterior uncertainty, or model
evidence. No decoded-success retries. Record every draw and insufficient-duration
case. Keep complete 5 ms source bins; explicitly report any omitted partial
final bin, source duration and counts, simulated duration and counts.

Rate maps and unit QC are the frozen all-RUN empirical surrogate population.
These are not claimed true biological tuning curves or training-only held-out
RUN fits. Rate evaluation is bilinear between grid centers and clamped at
outer centers. Decoder support is the existing fixed occupied-bin mask with
out-of-arena centers excluded where native boundaries are available. Report
truth outside support; do not reject or redraw those paths.

## Known Paths

Draw an origin uniformly in the rectangular simulation domain and a uniform
initial direction, paired by source candidate across all conditions. PF uses
encoding-grid extent, explicitly NOT verified physical walls. Tanni uses
native arena bounds. Simulate at 1 ms with specular boundary reflections and
midpoint integration, checking half-step convergence in tests.

Five truth conditions:

- continuous, speed v(x) = 1000 * (1 + g*q(x)) cm/s, g = -0.5, 0, +0.5;
- stationary, one fixed position throughout;
- discontinuous, independently drawn position every 5 ms.

q is normalized horizontal position from -1 to +1. This is a spatial-gradient
identifiability benchmark, NOT a wall-distance-gradient experiment. Both signs
are essential. Record fine-time arclength speeds, instantaneous center truth,
and time-averaged window positions/chord speeds separately. A constant-speed
reflection can reduce chord speed without changing distance traveled per time.

## Observation Conditions

1. Poisson oracle-map family: sample independent fine-time cell counts using
   empirical rates multiplied by 3, then remove columns for reduced populations.
   Decoder uses the same rate multiplier. This removes both cells and spikes.
2. Fixed count: preserve each source candidate's full-population total in every
   5 ms bin across all cell fractions. Split totals uniformly among the five
   1 ms intervals, allocate cell labels in proportion to interpolated rates in
   the retained subset. Shared random uniforms pair subset conditions.
   Count-conditioned decoding is primary; unconditional Poisson is deliberately
   misspecified for imposed totals.
3. Shared-gain mismatch: mean-one lognormal gain, coefficient of variation 1,
   shared across cells in 20 ms blocks. Generate full population then remove
   columns. Fixed-rate decoder does not know gain. This is a stress test,
   not an estimate of real replay gain fluctuations.

The Poisson and conditional families match their fine-time generators, but
the static-position decoder applied to a MOVING 20 ms window is not an exact
generative likelihood. Within-window motion, interpolation/discretization,
spatial support, and gain mismatch are explicitly distinct approximation
sources. Oracle maps do not guarantee calibrated instantaneous-position HPD
regions. Do not use this benchmark to certify real uncertainty calibration.

## Decoding and Metrics

Uniform spatial prior, independent windows, no temporal model. Decode 20 ms
windows at 5 ms strides. Both likelihoods, MAP and posterior mean. Report with
and without >=2 cells and >=3 spikes per window. Never bridge unsupported bins.
Continuity uses the fixed <20 cm adjacent jump, >=10-frame longest run and
>=40 cm end-to-end rule. This is geometric screening, not shuffle significance.
Speed uses adjacent NON-overlapping 20 ms windows.

Report continuity fractions for all continuous draws and separately those whose
truth window-mean path passes the same geometric rule. Report null geometric
acceptance for stationary/discontinuous draws, not empirical replay FPR.
Report position error, HPD coverage of instantaneous center truth, arclength
speed error, window-mean chord error, and selection/filter denominators.

For spatial gradients, average sufficient moments within each event so long
events do not dominate. Regress speed/base_speed on q using these event-weighted
moments. Separate known true coordinate from decoded midpoint coordinate, true
arclength from window-mean chord and decoded speed, all supported steps from
selected-continuity-core steps. Abstain if fewer than five events or spatial
variance <0.01; expose these thresholds and omitted session denominators.
Average sessions within animal, animals equally. Animal-bootstrap intervals
are conditional on frozen synthetic draws, not biological speed confidence.

## Verification and Remaining Work

Hash code, source cache, source totals, selected population IDs, path and count
arrays. Verify fixed-count totals in EVERY source bin and native exact column
subsets. Expected with 30 sources/session: 990 source profiles, 4,950 truth
draws, 356,400 metric rows. Repeated rows are not biological replicates.
Production must run from a clean commit; dirty-tree option only for a tiny
technical smoke, never a paper artifact. Smoke may have unavailable gradient
summaries because its samples are deliberately too small.

This is DEVELOPMENT, not independent evaluation or a biological uniformity
test. Any correction chosen from these results requires new held-out paths,
seeds/population draws and mismatch validation. Further required work includes
map-estimation error, more rate scales and observation variability, independent
population replicates, controlled field size/density and arena-size effects,
bin/grid sensitivities, actual wall gradients with verified PF walls, and a
prespecified equivalence bound only where nonuniform gradients are recoverable.
