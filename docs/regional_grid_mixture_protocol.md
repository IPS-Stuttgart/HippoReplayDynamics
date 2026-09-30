# Grid-level mixture inference: capped synthetic falsification screen

Frozen 2026-09-17 before fitting bank content. No real-data correction authorized.

## Question and estimand

Infer mixing weights pi over the existing valid spatial grid before summing Home
bins. The target is mean geometric Home occupancy over the final 40 ms, not
endpoint prevalence or the binary >=20-ms visit label. Window likelihoods are
20-ms, non-overlapping, independent flat-map observation likelihoods. No HMM or
trajectory prior. The first unpenalized EM iterate from flat pi must equal the
mean flat-prior posterior; this identity alone does not establish bias direction,
regional identification, or agreement of population-specific optima.

Use all eight sessions/four rats and all 22 frozen population definitions from
the audited shared bank. Use the same 1,139 eligible count templates and original
geometry labels; exclude no events. The experiment is capped at 40 ms and frozen
validation replica 0 at all four binary-label quotas, across seven pure dynamics/
scale strata plus their equal-weight empirical mixture. Calibration replica 0
is used only for event-fold penalty selection; no truth labels enter fitting or
penalty choice. Remaining replicas and null panels stay reserved.

The eight tested mixtures are not every possible dynamics mixture. A pure-stratum
failure is a counterexample; a pass would require a broader test. Grid-level
regional estimates need not have extrema at pure mixtures, unlike the earlier
one-dimensional mixture-weight estimator. Binary quotas are not occupancy quotas.

## Observation models and consistency limits

Primary: conditional multinomial within the given cell subset.
Sensitivity: ordinary 20-ms independent Poisson, including silence.
Diagnostic: fixed-total censored multinomial, with excluded cells represented by
one 'other cells' category. It uses the observed whole-population total and the
summed RUN rates of excluded cells, not their replay identities. This extra
information means it is not the same estimator available to an isolated subset.

The bank fixes whole-recording spike totals, not subset totals. Even the frozen
'full' decoder population excludes some recorded cells. Subset counts therefore
contain position-dependent participation information; conditioning them away
does not generally yield a common mixture prior independent of count. Censoring
is a matching diagnostic before the native active-cell acceptance rule, not an
assertion that the bank is exactly model-consistent after selection. Report the
bank's spike-support rejection counts. Stationary is the strongest consistency
check; both moving AND jumping can change position within a decoding bin.

## Objective, smoothing and numerical validation

Maximize mean_w log(sum_x L_wx*pi_x) - lambda*R(pi) on the probability simplex.
R(pi)=K^2/(2*E) times the sum of squared pi differences over E nearest-neighbor
grid edges. This smooths a marginal spatial distribution, not a temporal path.
Compare lambda in {0, .001, .01, .1}. Choose ONE shared lambda using three-fold
held-out window log likelihood with folds assigned by original event ID. All
windows, strata and generated copies of one template stay in one fold. Average
scores over populations within session, sessions within rat, then rats. Select
highest score; ties within 1e-6 nats/window prefer the stronger penalty.
Use conditional-multinomial CV to choose lambda, unchanged across populations,
sessions and diagnostic observation models. No Home mask or occupancy truth is
used for CV. Truth is used only in subsequent evaluation.

Report first EM, 20th EM, unpenalized NPMLE and selected-penalty estimates. Use
20 EM steps to initialize SciPy constrained nonnegative-coordinate optimization;
normalizing coordinates enforces the simplex and a radial term fixes scale.
Check the final convex simplex dual gap <=1e-6 independently of optimizer exit
status. The unpenalized result is additionally checked for the EM fixed-point
residual. Compare small fixtures with independent SLSQP. Flat likelihoods and
nonconverged fits are explicit failures, never interpreted as regional absence.
Numerical optimization does not certify that regional content is identified.

## First two screens and stopping

For each population/session/observation, report absolute truth errors for every
tested condition and their maximum. Report high/low gaps only on the exact same
events, separately for targeted and whole-tetrode pairs. The five-percentage-
point budget applies to BOTH regional truth error and absolute high/low gap.
Report error/gap reductions relative to the first EM step, not just agreement.
All relevant fits must be numerically valid; no missing/boundary fit is dropped.

One held-out replica is a falsification pilot, not a bias confidence interval or
certification of a worst-case population error. Small first-step gaps need not
reproduce the old real 8.58-pp gap because the cohort and likelihood differ.
If either primary screen fails, do not run real reconciliation, event bootstraps
for real claims, perturbations or population-specific detection. Those remain
subsequent experiments. In any later bootstrap, the event is the sampling unit;
inferences across recordings use animals, not windows or repeated simulation fits.

Failure here rejects the tested likelihood/regularization/sample-size combination,
not every estimator on the encoding model or the possibility of better recording
design. A pass cannot establish correct spatial truth under shared misspecification.

## Audit and provenance

Retain source/input hashes, frozen CV choice, event-fold assignments, per-fit
objective/dual gap, grid weights and observation model. Recompute first-step
identity, likelihoods and regional truth against saved bank paths. Independently
reconstruct selected fits' objective/gradient/dual gap and all summary tables.
Leave the previous binary and occupancy experiments unchanged. No simulation
truth is fed into the optimizer, CV selection or smoothing graph construction.

## Coordinate reconstruction preflight correction

The first attempt stopped before selecting a penalty: legacy matched populations
have larger native grids than the simulation grid. Preserve each population's
original grid, reconstructing its coordinates from the hashed matched archive
and its all-cell rates from the hashed training encoder. Check every frozen rate
entry and Home mask by exact coordinate/identity matching. Compute smoothing and
censored totals on that native grid; never equate column indices across grids.
Report decoder/generator grid overlap. No outcome threshold, CV rule, likelihood
or population was changed. The rerun is v2; the initial preflight output is not
an experiment result.

