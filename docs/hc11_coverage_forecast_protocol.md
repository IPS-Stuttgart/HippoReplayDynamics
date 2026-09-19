# hc-11 independent-detection coverage and future-prediction test

Frozen on 2026-09-19 before new classification or predictive scores. This is
an external extension of the PF coverage result, not an IMM rescue, new-animal
confirmation, or a claim that sleep/1D and awake/2D replay are interchangeable.

## Question

Does removing half of the inference neurons remove geometric trajectory labels
while useful prediction of future spikes from other neurons remains?
The bounded interpretation is that temporal population organization can be
missed by a trajectory screen. It does not establish remembered content,
memory consolidation, or true replay membership.

## Cohort and independent detection

Attempt all eight native hc-11 MAZE sessions (four animals). Use native POST
epoch and NREM intervals, excluding 50 ms from state boundaries. Keep native
CA1 sorted units and the frozen RUN-only encoding recipe: 4 cm bins, speed
>=5 cm/s, smoothing 1.5 bins, >=20 RUN spikes, information >=0.1 bits/spike,
peak >=1 Hz. The old helper's RUN count is a rate-times-occupancy estimate,
not a newly computed raw-spike-count threshold.

Reserve 20% of qualifying neurons for detection using the existing deterministic
partition routine. Exclude these neurons from classification and prediction.
Detect their population activity in 1 ms bins, Gaussian SD 10 ms, peak >3 SD
above POST-NREM baseline, mean-crossing boundaries, duration 50-2000 ms, at least
max(2, ceiling(10% of detector cells)) active. The domain is NREM, not a claim
of independently measured immobility. Do not require ripple overlap, trajectory
evidence, inference-cell counts, or evaluation-cell counts.

Keep all events in the catalog. Bound computation by uniformly sampling at most
200 per recording with fixed seed 20260919, before reading classification or
prediction. Short and zero-target-spike events stay visible. Report overlap
with the previously inspected 320-event cohort without changing selection.
This is not a wholly uninspected dataset or prospective new-animal experiment.

Missing metadata, fewer than eight qualifying units, fewer than five events,
or invalid clocks cause explicit feasibility exclusions, not threshold changes.
No session may be removed because its predictive result is weak.

## Classification and prediction

Five fixed 70/30 inference/evaluation partitions of non-detector neurons.
Compare full inference cells with a deterministic nested half, keeping evaluation
cells, spikes, windows, maps and targets identical.

Use the necessary geometric screen from the PF study: independent flat-prior
Poisson MAP, 20 ms windows advancing 5 ms, supported edges with >=2 spikes,
longest adjacent-jump run <20 cm, >=10 frames and >=40 cm displacement. This
is NOT the full published replay test: the original 5000-shuffle validations
are absent. Circular tracks require shortest-arc jumps and displacement, not
artificial seam jumps. Use pooled native RUN maps, explicitly not the earlier
direction-mixture IMM predictor. Report linear/circular recordings separately.

Forecast nonoverlapping complete 20 ms bins, primary target center 40 ms ahead
with one wholly unobserved intervening bin. Only inference history through the
origin may affect forecasts. Evaluation spike totals define a conditional
cell-identity target, not an inference input. No target event calibrates a fit.
Use five chronological folds with 1 s guards and the existing learned K50
multinomial HMM (two restarts, 500 iterations), against its occupancy/self-
transition-matched maximum-entropy null with its OWN causal filter. Also report
global, no-history and frozen-origin predictions. No outcome-dependent K choice.

Frozen-operator analysis is primary. A reduced-population refit (retained
inference plus the same evaluation cells on OTHER events) is required before
claiming robustness to training coverage. Nonconvergence is a failure, not
permission to change models. This is not evidence for a unique HMM/IMM circuit.

## Estimand and decision

Primary: half-inference, 40 ms future, full-pass/half-fail events with >=10
supported half-population frames, dynamic minus own-filter matched null in
nats/evaluation spike. Report all lost events too. Aggregate split medians per
event, session means, equal-animal means. Include all animals, zero-spike
denominators and leave-one-animal-out effects. Four-rat bootstrap intervals are
descriptive, not a replacement for limited independent-animal inference.

A supportive extension requires non-vacuous label loss, positive excess over
matched-null/global/no-history controls in the lost-supported group, positive
effects across represented animals, and agreement of frozen/refitted analyses.
Missing animals or too few lost-supported events limit replication despite a
positive pooled interval. Retain mixed/negative outcomes. Neither generic
predictive information nor a geometric pass proves experience-specific replay.

## Execution

The first script only creates a hashed bank and reports eligibility and counts;
it cannot classify or score events. Freeze feasibility before scoring. Run
computations detached on a designated server, with clean source commit, input
and output hashes, terminal status and separate verification of raw counts,
partitions, calibration guards and future-information masking.
