# DANDI000978 agreement-subset RUN pilot

Protocol frozen 2026-09-21 before this pilot's outcomes. This is a feasibility
experiment, not a biological result or an anatomical-label repair.

## Scope and provenance

Use version 0.240511.0307 at the existing read-only dataset path on the dataset
self-hosted runner. Reuse the completed acquisition SHA-256 record and recheck
current sizes. Do not describe this as rehashing the 323 GB dataset. Record the
manifest/record hashes, metadata fingerprint, source hashes, commit, runner,
environment, seeds, trial folds and output hashes. No new dataset download.

Start with ER1 and JS34, chosen before neural outcomes for the previously
audited population availability. ZT2 is excluded: do not concatenate its files
by numeric unit ID. Expansion to other animals is a separate documented run.

## Provisional populations

A: literal NWB row and unconfirmed tetrode-ID interpretations both say CA1.
B: both say PFC. Output A/B, not confirmed anatomical groups. Exclude every
unresolved or disagreeing unit. Freeze membership from metadata alone. Neither
firing rate, tuning nor decoding performance may determine group membership.
A common export error could invalidate both interpretations. This procedure is
NOT evidence that the anatomical problem has been solved.

## Samples, targets and folds

A sample is a complete RUN trial [start, stop), not an independently randomized
time bin. The target is the ordered start-well/end-well pair. Retain error and
correct trials alike; exclude only missing, nonpositive, noninteger or same-well
labels and record exclusions. Do not assume there are four supported classes.
Report every observed route and its trial support. Require at least 30 trials,
three RUN epochs and two observed routes before scoring.

Two prespecified outer CV schemes:

1. Five contiguous trial blocks within each epoch. The same block index is held
   out across epochs. Purge adjacent trials at each held-block boundary.
2. Leave one entire RUN epoch out.

Every included trial is tested exactly once per scheme; train/test populations
are disjoint by trial. Fold membership is identical for both neuron groups,
all models and all nulls. Report any fold with fewer than two training trials
for any observed route. Such a cohort cannot pass the nominal screen; it is
not silently removed from the table. Models remain finite with pseudocounts.

## Readouts and calibration

Three uniform-prior classifiers, independently fitted to each group:

- Conditional neuronal identity: class-conditional multinomial composition,
  Dirichlet pseudocount 0.5 per neuron. Total population count is conditioned
  upon, not included as a class-specific count likelihood. Confidence can
  nevertheless depend on count; this is not complete invariance to firing rate.
- Independent Poisson population counts with trial-duration exposure and a
  fixed 0.5-count / 0.5-second regularizer.
- Count/duration nuisance baseline: diagonal Gaussian on log(1+population count)
  and log(duration), with 10% shrinkage toward training-only pooled variance.
  This is a limited nuisance model, not proof that all behavioral/rate confounds
  have been removed.

For each outer fit, choose a scalar temperature from
[1,2,4,8,16,32,64,128,256,512,1024] by macro log loss on two contiguous
training-only trial blocks. Refit on all outer-training trials. Refit this
calibration inside every null replicate. No outer-test label, held-out epoch,
or sleep data may tune the decoder. No outcome-based hyperparameter search.

Report balanced accuracy, accuracy, micro/macro log loss in nats, confusion
matrices, trial predictions and population counts. No bin/fold/null replicate
is an independent animal. The biological pilot sample is two rats.

## Controls and feasibility screen

For conditional identity, run 199 replicates each of whole-trial label
permutation within each RUN epoch and circular shifts of trial-label sequences
within each epoch. Shift offsets are at least 20% of the epoch's trial sequence
from zero, subject to finite-length constraints. Both preserve within-epoch
label frequencies; neither is a universal null for every possible drift
process. These are RUN label controls, NOT sleep spike-time-shift controls.

Use (1 + number of null statistics at least as favorable as observed)/(K+1),
with the correct tail for each metric. The nominal screen requires complete
training-class support, >=99 nulls, p<=0.05 for both accuracy and macro loss
under both controls, lower macro loss than uniform, and lower macro loss than
the count/duration baseline. Require both outer CV schemes for a group-level
feasibility interpretation. This is an exploratory screen, not a
multiplicity-corrected biological discovery. A negative screen must be retained.

The workflow succeeds when computation/validation completes, even for a negative
scientific result. An invalid input or incomplete subject job fails the workflow
while retaining available diagnostics. Group agreement is not ground truth.

## What RUN success would and would not establish

It would show reproducible route-associated information in these fixed groups.
Whole-trial counts can reflect location, direction, dwell time, movement and
other behavioral variables. It does not demonstrate abstract experience coding,
sorting stability, validated sleep content or compressed replay decoding.
Do not rename groups CA1/PFC without the authors' mapping confirmation.

## Downstream direction (specified, not executed by this workflow)

Keep candidate windows, validation neurons and the independently learned
validation content readout fixed. Reduce only the trajectory-classification
population, initially at fractions 1.0, 0.75, 0.5, 0.25 and 20 prespecified
subset seeds, with count-matched spike thinning as a separate control.
Trajectory thresholds/encoders must be frozen or calibrated on independent
RUN/synthetic material, never on desired replay outcomes.

Separate endpoints:

- Independent content support in newly rejected events versus rate/duration-
  matched and within-state/time-block alignment controls. The validation score
  of an individual event staying fixed is true by construction, not evidence.
- Change in the distribution of independently read-out content among accepted
  events as coverage decreases. Retained prediction alone does not establish
  content-selection bias. Report route-specific acceptance and the complete
  candidate denominator, with animal-level summaries.

For a known-content calibration, use route-balanced surrogates with separate
encoding/decoding estimation data and observation-model mismatch. Measure false
acceptance on stationary/order-disrupted controls, not sensitivity alone.
A confidence-only RUN selector must never be called a replay-trajectory gate.

Extension to existing PF/Tanni ripple cohorts requires their recorded candidate
provenance and matching content definitions; next-visited well is a behavioral
proxy, not replay ground truth. No PF/Tanni analysis or sleep experiment is
implemented or claimed by this RUN-only pilot.
