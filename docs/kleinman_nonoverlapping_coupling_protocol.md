# Kleinman non-overlapping coupling design audit

Frozen before this audit is run. Coverage only; no spatial-change score,
association, treatment coefficient, confidence interval, or replay claim.

## Cohort and chronology

Keep all 135 Experiment 1 sorted-spike sessions in the inventory. Calculate
packets only in sessions passing the existing frozen blocked-RUN QC, whose
CSV hash and inventory must agree with its manifest and the raw release.
Retain failures and sessions with zero packets in the denominators.

Within each epoch and direction, enumerate six consecutive traversals:
R0,R1,R2 are reference; R3 is past; R4 baseline; R5 target. The predictor
would use eligible native ripple/background exposure between R4 end and
R5 start. A future score would compare R4 to R5, a preceding-RUN control R3
to R4. The reference is earlier than all three readouts.

Select full-span non-overlapping packets across BOTH directions, session-wide.
Sort candidates by (target end, reference start, direction, packet ID);
greedily retain a candidate when its reference start is >= the previous
selected target end. This deterministic earliest-finish rule maximizes packet
count without observing spikes, ripple exposure, readout quality or outcomes.
Never replace a selected packet that subsequently has poor coverage. Keep
selected and overlap-excluded candidates. Direction imbalance is reported,
not optimized after seeing data. Non-overlap is not statistical independence.

## Coverage descriptors

Use existing RUN parameters: 2 cm bins, 4 cm smoothing, reference RUN >8 cm/s,
reference-only cell inclusion >=10 spikes and peak >=1 Hz. Readouts >20 cm/s,
20 cm inside reward thresholds, valid tracking. Report three readout
durations/spike counts and paired occupancy overlap; do not require outcome
spikes for inclusion or silently discard zero counts.

Native ripple and nonripple exposure: valid immobility <=8 cm/s, first 10 s
of each reward visit, clipped to R4-end/R5-start gap. Reference minimum of
five cells is the existing descriptor, not a newly tuned cutoff.
An availability descriptor requires five reference cells, positive ripple AND
background exposure, variable finite reference-cell log-rate enrichment, and
shared occupied bins for both preceding and future comparisons. This is not a
conditional-information or decoder-quality pass. No future spatial scores or
outcome/predictor correlations are computed.

## Interpretation gates

Audit full released metadata fields. Do NOT infer physical track identity from
session names, dates, novelty, reward thresholds or track length. Drug and track
were linked in the published design, except pretraining and specific exceptions:
https://elifesciences.org/articles/99678 (Methods, Task design).
Session ordinals are recorded as filename descriptors, not injection timestamps.

Report all animal x drug x novelty cells including zeros. Six rats provide only
three experimental and three controls. Source familiarity does not prove a
shared track across drug conditions. With no authoritative track IDs, a
within-track pharmacological contrast is not authorized by this audit.

The preceding comparison shares R4 with the future comparison and can have
mathematical anticorrelation, drift and reverse-causality confounds. The
two-readout calibration does not calibrate this three-readout contrast.
A separate frozen joint simulation/calibration is required before inferential
use. Neither a pooled pilot null nor this availability audit rejects a VTA
interaction or establishes plasticity.
