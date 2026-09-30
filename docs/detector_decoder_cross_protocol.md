# Detector by decoder: frozen regional sampling diagnostic

2026-09-17. New measurement experiment, not a reopening of the content-correction
stopgate. No calibrated biological prevalence or new replay classifier is claimed.

## Scope and populations

Use all eight PF and 25 Tanni input recordings in the frozen all33 coverage cache.
Verify cache hashes and RUN-only unit QC. The reference population is every
RUN-qualified sorted unit, not every raw recorded cluster; report both counts.
Use the existing full-RUN maps (8 cm grid, 1.5-bin smoothing, 0.05 s occupancy,
10 cm/s minimum running speed; original RUN-only unit QC). No replay outcomes
enter unit inclusion or map estimation. This is not a held-out RUN accuracy test.

The common valid grid is identical for every decoder within a recording. Split
its bounding rectangle into 3 by 3 tiles. The central tile (index 4) is the
predeclared primary region; it is not called Home or reward. Report all nine tile
masses as secondary readouts and the actual valid-grid area fractions.

Rank cells by mean RUN rate inside the central tile divided by mean rate on the
whole valid grid. Highest/lowest floor(N/2) cells define two disjoint targeted
halves, with deterministic ID tie-breaking. Include two seeded random bipartitions
and two seeded whole-tetrode bipartitions. Whole-tetrode assignments minimize
cell-count imbalance among 128 RNG proposals, using no spatial/replay outcomes.
Do not split an electrode, invent tetrode IDs, or silently substitute random cells.
Record individual IDs, tetrode IDs, RUN coverage, counts and unavailable families.

## Crossed design

For each subset S, cross detector population F/S with decoder population F/S.
Use one unchanged spike train and unchanged behavior per source. F detection and
decoding are computed once then reused. A decoder may not change candidate
membership, event boundaries, or the readout support gate.

Reuse the existing PF-style pooled-spike MUA detector: 1 ms bins, 10 ms Gaussian
SD, threshold 3 SD above each population's immobile mean, mean-crossing edges,
50-2000 ms duration, minimum ceil(0.10*N_detector) active cells. Smoothed behavioral
speed <5 cm/s, with 0.1 s speed smoothing. Restrict real detection to existing
tracking-supported RUN intervals; no LFP or replay-content filter. The same
algorithm is applied to both datasets for this diagnostic, not claimed as an
exact reproduction of every historical Tanni/PF candidate list.

Read the last complete 20 ms on the existing 5 ms event grid. Preserve silent
and low-support windows and report their counts; do not move endpoints or
abstain depending on the decoder's cells. Independently decode Poisson likelihood
with uniform valid-grid prior, 20 ms exposure and 1e-4 Hz floor. Conditional
multinomial is a separately labeled likelihood sensitivity, not a correction.

Report candidate counts, window overlap and endpoint shifts. On overlapping
events, also compare population readouts on identical full-detector endpoints.
The primary detector contrast includes candidate selection AND detector boundary
changes; it is not called a pure membership effect.

Let q_D,C be mean central-region posterior mass over endpoints selected by D and
decoded by C. Report detection contrast q_S,F-q_F,F; decoding contrast
q_F,S-q_F,F; interaction q_S,S-q_S,F-q_F,S+q_F,F; total q_S,S-q_F,F.
The identity total=detection+decoding+interaction must hold. Means are conditional
on detected windows, with nonempty denominators required. No zero-event pass.

## Known-content simulations

Generate independent Poisson spike trains from fixed RUN rate maps, NOT fixed
whole-recording totals. Use 2 s latent epochs, region-blind stationary or moving
paths on the largest connected valid-grid component. Moving paths use the
existing region-blind graph-route generator at a declared 500 cm/s. This is a
synthetic stress axis, not an empirically established replay speed or biology.
Rate intensities and represented positions are piecewise constant within 5 ms
generator bins; moving rates interpolate along graph edges at each midpoint.
Each epoch receives the same Gaussian activity gain centered at 1 s (40 ms SD),
with peak gain 3 or 6 and baseline 1. Regions do not enter path or gain generation.
Three replicas of 600 s per generator/gain/recording; seed 2026091701.

Compare all population/detector choices on EXACTLY the same generated spikes.
Use independently fixed 20 ms reference windows centered on each epoch's gain
peak, in addition to detected endpoints. Compute actual geometric regional
occupancy of each selected window from the saved 5 ms truth process; do not use
the full decoder as truth. Report detection probability against true central
content at fixed reference windows, selected true content, timing-induced truth
shifts, and decoded-minus-true content. Stationary epochs separate spatial
selection from within-epoch temporal movement; moving results are a robustness
check with potentially different true selected content.

Do not condition paths on Home, detected-event acceptance, or posterior outcomes.
Record no detections, background detections, multiple detections per epoch, and
windows crossing epoch boundaries rather than altering the generator afterward.
Real-data results are only sensitivity to neuron sampling, never verified errors.

## Analysis, audit and stopping

Synthetic replicas and subset repetitions are Monte Carlo, not animals. Aggregate
within recording, then animal, then dataset with equal animal weight. Keep each
family, likelihood, generator and gain separate. Report per-animal directions;
do not hide PF/Tanni heterogeneity behind pooled windows. A recording without an
estimable region or enough cells/tetrodes stays in the denominator with a reason.

Start with one recording per dataset for interface/runtime checks, then execute
the frozen all33 cohort without tuning to outcomes. Save raw simulation truth,
spikes, subsets, detector events, endpoint counts, readouts, factorial contrasts,
input/source hashes and commands. Independently reconstruct native detection,
counts, likelihood regional sums, truth occupancy and contrast algebra. Verify
plots and non-vacuous technical completeness before interpretation.

This does not claim first-in-literature novelty, simulated biological replay,
uniform replay speed, or a universal coverage correction. If effects are small
or dataset-dependent, report that and stop; do not optimize thresholds/regions
to obtain a preferred result. No paper changes or new real scoring claims are
authorized by a synthetic technical pass.
