# Uncertainty-aware CA1/PFC content calibration

Frozen 2026-09-23 before computing new-readout outcomes. Exploratory measurement
development on an existing calibration bank, not independent confirmation or
sleep evidence. No event selection, smoothing parameter or threshold changes.

## Fixed Inputs

Use every verified fold of the synchronized RUN calibration: 14 whole-epoch
holdouts, two source-supported animals (JS14 and ZT2), identical maps, counts,
three donor conditions, primary target lists and common-support masks.
Native-all, native-common and sleep-matched-common remain separate summaries.
The original primary lists stay unsupported if any target is unmatched.

## Readouts

Let c_r be the flat-prior CA1 route posterior and L_r the count-conditioned PFC
population likelihood for route r, with four routes. These are fixed RUN maps.

Primary uncertainty-aware score, in nats per population window:

    log(sum_r c_r L_r) - log(sum_r L_r / 4)

This integrates the latent route before taking the log. It is equivalently
log(4 * dot(c, p)), where p is PFC's flat-prior route posterior. Use log-space
arithmetic; no temperature tuning, posterior clipping or new learned parameters.
The score is bounded above by log(4), not an arbitrary trajectory threshold.

Compare with the same predictive score using a point mass at CA1's MAP route,
and a known-true-route oracle. Also retain the historical per-spike centered log
score and its posterior-weighted expectation. Those last two have different
units/baselines and are NOT themselves the posterior-predictive mixture. Do not
compare their raw numerical magnitudes to the predictive scores.

If CA1 is uninformative the soft predictive score is zero. If PFC has no spikes,
every predictive contrast is zero. Report ambiguous/tied route estimates rather
than interpreting tie-breaking as route information. The hard-score comparison
keeps the old first-argmax convention solely for reproducibility.

## Calibration And Failure Checks

Same-route RUN donors are known-content controls. Random PFC donor-route draws
are the null. Reuse 500 virtual studies, 199 null draws, alpha .05, study seed
20260924, and identical rows/routes/null draws across all readouts and donors.
Unique count targets per study; sizes 10/25/50 plus unchanged original primary
lists. Repeats, donor reuse and studies do not create biological replicates.

Primary planning adequacy remains synchronous soft-predictive detection >= .80
with null false-positive rate <= .075 in BOTH animals at n=50. This is a
conditional calibration criterion, not actual sleep power or a replay gate.
Report paired changes relative to hard predictive scoring, all sizes and donors,
and Monte Carlo precision rather than biological confidence intervals.

Also partition diagonal same-route scores into both estimates correct, CA1 only
correct, PFC only correct, both wrong but agreeing, both wrong and disagreeing,
and tied/ambiguous top choices. Positive overlap of two wrong estimates is not
validated route content. Partition contributions must sum to the total; weight
routes/repeats within count targets, targets within epochs, epochs within animal.

## Stop Rule

No outcome validates RUN-to-NREM transfer or the original underpowered sleep
comparison. This reused-bank exercise cannot serve as external validation.
If the soft readout does not meet the existing two-animal planning criterion,
stop incremental tuning of this four-route assay rather than trying more priors
or selecting a more favorable animal. A different content assay or additional
source-confirmed data would need a separate, justified protocol.
Even a planning pass leaves paper_ready=false and sleep_content_validated=false.
