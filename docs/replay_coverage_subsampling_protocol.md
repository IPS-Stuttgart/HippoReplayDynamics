# Paired Real-Recording Subsampling

Protocol frozen before examining the full-cohort decoding results (2026-09-05).

## Intervention

Use the 33-session all-candidate recording cache. Every one of the 12,141 source
high-MUA candidates remains in every recording condition. The detector is held
fixed: this experiment isolates decoding/continuity sensitivity, not changes in
MUA event detection when fewer cells are recorded.

Primary population conditions: full RUN-QC population plus 75%, 50%, and 25% of
its units, rounding down with a minimum of one unit. Use three deterministic
recording replicates (seed 20260905). Draw a population permutation per session
and replicate, use nested prefixes for the three fractions, and apply that same
cell subset to all events in the session. The full reference is scored once.
The manifest lists the actual retained cell IDs, not just the seed.

This removes both spatial sampling and spikes. It does not isolate cell count
from total spike information; the matched generator experiment must separate
those factors. No cell or event is chosen by decoded outcome.

## Decoder and Readouts

Run the matched baseline flat-prior independent Poisson decoder and an explicit
count-conditioned multinomial sensitivity on the same windows. The latter is
not asserted to be a better observation model for real replay. Rate scale is
one; it is not fitted to make a subset pass. There is no temporal prior.

Spatial states use the cached occupancy mask, fixed across recording subsets.
Where physical walls are known (Tanni), exclude centers of partially intersecting
edge bins that fall outside the actual arena. Record their number. Do not
replace the unknown PF walls with tracking extrema. A later wall-distance study
still needs boundary-centroid/grid-alignment sensitivity and verified PF walls.

Decode 20 ms windows advanced by 5 ms. Evaluate MAP and posterior-mean continuity
separately, with both no spike-support filter and a >=2 active cells / >=3 spikes
per-window filter. The fixed geometric criterion is the longest run with strict
adjacent jumps <20 cm, >=10 frames, and >=40 cm end-to-end displacement. Failed
support breaks a run. This is NOT a shuffle-significance replay classification.

Measure speeds only between adjacent NON-OVERLAPPING 20 ms windows. Report speed
over the full event and within the selected core separately. Never bridge an
unsupported window, count overlapping steps as independent observations, or
assign zero speed to an unmeasurable event. Partial final 5 ms bins are not used
as complete 20 ms observation windows.

## Denominators and Uncertainty

Report continuity pass fractions, losses AND gains relative to each event's
full-population decision, posterior uncertainty, and full-event/selected-core
speed summaries. A gain after subsampling can reflect blurring and does not
identify a newly discovered true replay. Speed changes have no biological
interpretation because the underlying recorded event has not changed.

Average recording replicates within session, sessions within animal, then give
animals equal weight. Bootstrap those animal summaries (5,000 draws); intervals
are conditional on these recorded sessions and limited by four PF/five Tanni
animals. Simulation-population replicates and repeated cell subsets are not
independent animal replication. Tanni's pooled result covers all arena sizes;
arena-stratified and within-animal comparisons remain separately required.

## Remaining Scope

This is a real-data perturbation, not a recovery calibration or a uniformity
test. Subsequent one-draw simulations must include constant-speed and graded
speed paths, stationary/discontinuous controls, actual maps/populations,
likelihood-matched generation, held-out recovery trials, and every failure in
the denominator. Train-only RUN decoder QC is still required. No interpretation
of real speed uniformity is licensed by these subsampling plots.
