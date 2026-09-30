# Regional occupancy calibration: first falsification experiment

Frozen 2026-09-16 before reading occupancy-calibration results.

Question: can unchanged mean regional posterior mass be calibrated to the
fraction of terminal represented time at Home using dynamics-invariant s/f?
This is a new estimand, not endpoint recovery or the prevalence of >=20-ms visits.

## Data and readout

Use the audited geometry-blind shared PF bank, all eight sessions/four rats,
1,139 common >=100-ms count-trace templates, all seven generator/scale strata,
all frozen populations, and Delta=20/40/60/100 ms. Preserve the fixed-count
allocation, candidate detector, endpoint and population definitions. No new
paths, event selection, thresholds, null panels or real-data inference.

Decode each non-overlapping 20-ms bin with the frozen independent flat-prior
Poisson likelihood. Primary: silent-bin mass equals spatial-prior area, matching
the preceding experiment. Sensitivity: retain the native Poisson silence term.
Average bin masses; do not take a maximum/union or apply a dynamics prior.

Compute exact geometric Home dwell per bin by clipping each saved path segment
to the bin and integrating its intersection with the 20-cm disc. Average dwell
fractions to obtain segment occupancy. Preserve the same events at all lengths.

## Two explicit definitions of calibration means

For mass q_w and true fractional occupancy o_w, sample represented time uniformly
within equal-length windows. The exact time-conditional means are:

    s_time = sum(o_w*q_w)/sum(o_w)
    f_time = sum((1-o_w)*q_w)/sum(1-o_w)

They satisfy mean(q)=mean(o)*s_time+(1-mean(o))*f_time within a sample by algebra.
That identity is not validation. Estimate on calibration replicas only and test
invariance/transfer on separate validation replicas.

Separately report s_pure=mean(q | o>=1-1e-8) and
f_pure=mean(q | o<=1e-8), excluding mixed windows. This is the within-bin-motion
diagnostic; excluded-window counts and missing support must remain visible.
These pure-window means need not reconstruct mixed-window occupancy.

Linearity alone does not establish invariance: conditioning on a region still
averages over locations, count traces, within-bin occupancy and firing-rate
patterns. Even a perfect readout q=o has distribution-dependent time-conditional
s/f when o is fractional. Include that counterexample in tests; a failed s/f
gate rejects this two-mean calibration, not occupancy as an estimand.

## Frozen first gate

For each session/population/Delta/silence policy/definition, pool the four
calibration replicas within each of the seven strata. Require BOTH
max(s)-min(s)<=0.02 and max(f)-min(f)<=0.02 across all seven strata. Missing
support is not a pass. Report leave-one-calibration-replica-out ranges and
the same check on validation data, including validation-prevalence strata.
Full-population primary eligibility requires all eight sessions to pass;
matched-population rows are reported separately. This point-estimate tolerance
is a development screen, not a statistical equivalence proof.

## Transfer diagnostics, not authorization

Fit one pooled calibration from all seven strata (all windows pooled with the
same number of events per stratum), plus a known-generator oracle from that
stratum's calibration replicas. Apply (mean(q)-f)/(s-f) to each independent
validation panel. Retain unbounded estimates, out-of-range flags and s-f;
do not clip failures into [0,1]. Missing or nonpositive discrimination fails.
Report naive mean(q), true occupancy, raw/calibrated/oracle errors, and low-true-
occupancy 2-10% panels actually supported by the bank. Do not pretend the binary
label quotas 5/15/30/50% are occupancy quotas, or fabricate a 2% validation panel.

No new occupancy-accuracy budget is copied from the binary prevalence task.
The 40-ms binary 5.68-pp result is contextual only, since the target differs.
If invariance fails, stop deployment and tests of the real 11.38%/2.80% gap:
those aggregate endpoint masses cannot simply be relabeled segment occupancies.
The real matched-event comparison and nonspatial perturbation/held-out-cell
diagnostic remain downstream and untested in this experiment.

## Audit and limits

Check source hashes, exact bin-dwell averages against saved bank truth, posterior
masses against a direct normalized Poisson calculation, and calibration means
against the saved event/bin arrays. Independently approximate sampled path dwell
numerically. Tests must cover constant paths, moving crossings, jumps, silence,
exact arithmetic identity, missing support, non-vacuous seven-stratum gates,
and a perfect fractional-occupancy readout that fails time-conditional invariance.

No biological interpretation or claim that inferred dynamics is the only remedy.
Report the selected decoder-derived anchors, fixed spike totals, legacy overlap,
arena-limited large jumps and four-replica development limitation unchanged.
