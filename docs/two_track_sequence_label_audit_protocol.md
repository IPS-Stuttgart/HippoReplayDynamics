# Sequence-label versus population-content audit

## Frozen purpose

The held-out RUN calibration found that thinning can admit windows whose
sequence labels disagree with behavioral context. Audit whether that same
distinction affects the already-scored primary POST ripple candidates.
This is diagnostic, not a replacement criterion or a positive-biological gate.

Use RAT3_SESS2 and RAT5_SESS2 only, exactly the existing primary cohort.
Inputs are the immutable heldout_RUN_selection_v1 experiment (original order,
Poisson sequence decisions only), the original complete replay campaigns,
the frozen candidate banks, and evaluation_identity_control_v1.
Use every selected RUN window, and every originally scored POST ripple
candidate. Do not choose examples by agreement or change source sequence labels.

## Distinct readouts

For each original split and each full/half inference subset, compute the mean
joint-posterior track mass across nonempty time bins, using the same counts and
rates as the original independent-bin decoder. Primary readout is Poisson;
conditional-count readout is a fixed sensitivity on the SAME sequence decisions.
RUN uses actual 100-ms exposure, rest the existing 20-ms exposure. No new spikes,
trajectory rescoring, shuffles or p-values are generated.

The sequence classifier normalizes its correlation within each track. A large
correlation on a track with little total posterior mass is therefore possible;
sequence evidence and context evidence must not be conflated.

Keep evaluation cells fixed and reuse their original count-conditional track
mass and RUN-rate-matched identity-null z. For RUN, translate the existing
truth-oriented readouts back to the common track-2 coordinate. For rest, reuse
the hashed evaluation control without recomputation. Never use B to choose
windows or change labels. Behavioral labels exist for RUN ONLY.

Track preference is defined by mass above/below 0.5, with a numerical tie band
of 1e-12; ties and missing spikes remain explicitly unassigned. This is an
agreement diagnostic, not a confidence gate. Mean per-bin posterior mass is
NOT a calibrated whole-event probability or a biological experience fraction.

## Paired groups and estimands

Pair full and half scores for identical windows/splits. Groups: full, half,
retained, lost and gained. Full and lost use the full-reference label/readout;
half, retained and gained use the half label/readout. For retained windows also
report the fraction whose sequence label changes after thinning.

Report selection weight, label agreement with each inference readout, label
agreement with B, label opposition by both populations, signed B identity-null
z, mean A/B mass assigned to the sequence label, and (RUN only) behavioral-label
agreement. Expose finite/tied denominators. Report label confusion matrices and
behavioral-label agreement separately from sleep-population agreement.

Average split/repeat contributions inside each original event before aggregation.
RUN: paired 2,000 block bootstraps, 10-s blocks stratified by track/fold,
preserving behavioral balance. POST: paired 2,000 occupied 60-s block bootstraps;
events and all cell-subset copies remain together. No pooled animal inference.
Intervals require >=95% finite draws and at least two informative blocks for
the metric. RUN also requires two observed blocks in each track/fold stratum.

Do not infer bias solely from agreement, disagreement, or changed classifier
label counts. This audit assesses whether a detector label is interpretable as
experience identity; it does not identify true sleep replay or correct its
experience proportions. Preserve all previous null findings and source hashes.
