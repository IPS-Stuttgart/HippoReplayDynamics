# Frozen empirical hc-11 MAZE transfer control

2026-09-08, before real MAZE scoring. This is the next diagnostic after the
negative sleep conditional-prediction experiment and positive known-generator
recovery. It is not a replacement event selection or a new sleep result.

## Question and scope

Can separately fitted native RUN maps predict actual observed cell identities
in held-out MAZE time? Distinguish (a) behavior-conditioned map prediction,
(b) position inferred only from training neurons, and (c) temporal-model
advantages. Successful prediction of running is not evidence for replay,
learning, or a sleep-specific biological mechanism.

Use the same eight native hc-11 sessions/four animals, native 4 cm
linear/circular geometry, 20 ms count bins, and frozen encoding/dynamics
parameters as `hc11-conditional-cross-cell-prediction-320x5-20260908`.
Verify its manifest, source counts/maps and the live raw spike/position files.
No new dataset or favorable session subset.

## Honest train/test separation

Split each single MAZE epoch at its temporal midpoint. Encode on the first
half and evaluate on the second, and vice versa. Exclude a 5 s guard from
the midpoint and epoch edges. Rate maps and primary unit QC use ONLY the
encoding half's moving samples. RUN speed >=5 cm/s; all other native QC
thresholds unchanged (20 RUN spikes, information >=0.1 bits/spike, peak >=1 Hz,
at least 5 encoding units, 1.5-bin smoothing). Geometry comes from the full
observed MAZE support; that is a fixed behavioral coordinate convention, not
a rate-map fit using test spikes.

Primary population: train-only unit QC. Sensitivity: the previously frozen
sleep-encoder unit IDs, but still fit their rates exclusively in the encoding
half. The sensitivity conditions on prior full-RUN unit selection and is not
a wholly independent unit-selection validation. Report overlap and failures;
do not silently fall back to those units if primary QC fails.

Use five 70/30 neural splits, seed 20260804 + split. Every held-out neuron's
encoding model uses its encoding-half spikes, as usual. None of its
evaluation-window spikes enters latent inference or chooses a model/prior.

## Test windows, chosen without spikes or scores

Nonoverlapping 200 ms windows anchored to the start of the guarded test half.
Every 20 ms bin center must have a finite MAZE position sample within 50 ms,
speed >=5 cm/s, and one common nonzero running direction across the window.
The latter matches the fixed global-direction likelihood convention. These
are locomotion control windows, not ripple or replay candidates.

Uniformly select up to 100 eligible windows per half with a frozen stable
SHA256 seed (20260908, hc11_maze_transfer_windows, session, fold). Retain all
if fewer than 100; record denominators and exclusions. No spike-support gate
selects or drops a scored window. Both halves and all sessions remain visible.

For each selected window score its native population count matrix and a
paired sleep-total-cap sensitivity. Assign target totals by deterministically
shuffling/cycling that session's 20 frozen POST totals. If native N exceeds
the target, sample the target number of actual observed spikes without
replacement from the entire window (multivariate hypergeometric over
time-by-cell counts). Otherwise retain native counts and flag the cap as
not exactly attained. One population draw feeds every neural split/model.
This preserves original spike times/identities for retained spikes, but NOT
the real sleep per-bin count profile or precisely the held-out/active-unit
counts. It is not simulated replay or information matching. Never reject an
insufficient-count window to force a match.

## Scoring and measurement

Proper normalized held-out multinomial cell-identity probabilities given
the observed held-out total, T=1 throughout. Direction-mixture primary,
pooled maps sensitivity. Frozen IID, static, diffusion and IMM kernels; no
prior sweep. Independent bin decoding has a uniform spatial prior.
For the direction mixture, IID positions are independent conditional on a
single direction inferred from the whole window's training-cell data; the
pooled sensitivity has no shared direction variable. Neither uses measured
test position/direction to infer the neural posterior.

Compare against a nonspatial cell-probability vector from encoding-half
pooled RUN rates averaged by that half's occupancy. Also calculate a separate
behavior-conditioned score from the held-out half's measured position and
direction, using the encoding-half map. This supervised positive control is
clearly labeled and is never fed into the neural-inference predictions.
Use one fixed common spatial-bin permutation per session/fold to calculate
the behavior-conditioned wrong-map score.

Primary encoding contrasts: behavior-conditioned minus nonspatial and
behavior-conditioned real minus permuted map. Neural IID minus nonspatial
tests generalization of independent snapshots. IMM-IID, IMM-static,
diffusion-IID and static-IID are descriptive controls, not moving-replay gates:
real running moves much less in 200 ms than typical replay assumptions.

Report train-IID MAP and topology-aware mean position errors and 95% highest
posterior mass set coverage, both over all bins and supported bins (>=2
training spikes and >=2 training active units). Unsupported/zero-held-out bins
remain in predictive scores. Circular mean is undefined at near-zero
resultant; mark it missing rather than substitute a coordinate.

Paired score differences precede median aggregation across splits. Then
average windows within each half, halves equally within session, sessions
equally within animal, and animals equally. Report five thousand
animal-cluster bootstrap draws, seed 20260908, with all fitted encodings and
within-animal observations held fixed. Four animals imply minimum exact
one-sided sign-flip p=0.0625. Intervals are exploratory and do not include
uncertainty from fitting rate maps. Show per-window, half/session/animal and
per-held-out-spike sensitivities. No pooled-window pseudoreplication.

## Required verification and decision

Non-vacuous completeness, true time and cell separation, train-only unit
selection invariance to changed test-half spikes, exact timestamp counts,
thinning without replacement, normalized scores, held-out independence, and
unchanged posterior hashes. Save counts, edges, maps, unit IDs, masks, neural
splits, selected windows and raw-file hashes. Independently reconstruct a
balanced native/capped sample of predictions using the dense log-domain
verifier, including supervised and nonspatial scores and aggregate estimates.

If actual MAZE prediction works, the sleep failure is not a universal
inability of these observed units/maps to support spatial prediction. It
still need not be a biological absence: state-dependent encoding/noise,
event class and dynamics mismatch remain alternatives. If MAZE prediction
also fails, do not interpret the sleep result biologically. Do not use this
control to retune the closed sleep analysis or select its strongest animal.
