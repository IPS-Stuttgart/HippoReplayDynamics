# Regional readout and endpoint ambiguity diagnostic

Frozen before results, 2026-09-15. This is development on known-truth PF simulations,
not real-data content estimation and not external validation. Do not run hc-11.

## Inputs and estimand

Reuse selection-matched-regional-pf-v3-20260915, all eight sessions/four rats,
200 frozen candidate templates per session and all 22 population definitions.
Verify source hashes. No new real candidates, maps, event selection or evidence
scoring. The binary truth is position at the END of the last complete window,
inside the frozen 20-cm Home region. It is not average window content.

Calibration uses only the saved calibration replicas. Validation uses the
independent saved validation replicas at prevalences .05, .15, .30 and .50.
Null/perturbation and .38 panels are not used to fit or choose methods here.
This is a post-failure development diagnostic on an existing validation bank,
not a new confirmatory holdout. Replicas/events are not independent animals.

## Factor 1: readout compression, original 20 ms

Compare frozen three-category calls with a continuous regional log-Bayes-factor
mixture fit. Primary continuous scores set silent windows to BF=0, preserving
the neutral-silence convention. Sensitivity retains native silent Poisson BFs.
Transform continuous scores by arcsinh, fit class-conditional Gaussian KDEs
using Scott bandwidths and a minimum transformed bandwidth .05. No validation
bandwidth search. Evaluate KDEs on a 2049-point calibration-only grid extending
eight bandwidths beyond calibration support; evaluate outside-grid queries
directly. The transformation Jacobian cancels in the class density ratio.
Use equal .5 class smoothing only for the existing ternary emissions.

## Factor 2: generator uncertainty

Compare the original pooled stationary/moving/late-jump calibration with
generator-specific calibration. For mixed validation panels, the latter fits
prevalence separately within the KNOWN simulated generator, then aggregates
by its accepted event count. This is an oracle diagnostic, unavailable on real
replay; it is not a proposed generator classifier or deployable remedy.

Report raw BF AUC and calibration likelihood-ratio AUC as well as prevalence
error. An inverted raw AUC can still contain information. In particular, do
not infer no information from AUC below .5 or claim the preceding stationary
simulation AUC establishes strict-terminal identifiability for late jumps.

## Factor 3: temporal aggregation and spike loss

For stationary and late-jump generators, redraw their original calibration
and validation panels using exact saved seeds and the unchanged generator.
Verify identities reproduce saved 20-ms counts, targets, labels and active
gates exactly. From the SAME spikes, count the last 5 ms and decode with .005-s
Poisson exposure; compare to the original 20 ms with .020-s exposure. Keep the
original accepted event set, including zero-spike short windows; no filtering
that silently changes the target population. Full population only for this
temporal arm. Fit independent, generator-matched calibrations at each duration.
The late-jump generator occupies its terminal state only during the final 5 ms;
stationary is the spike-loss control. Boundaries are half-open and use the
existing endpoint_interval timing convention.

## Outputs and interpretation

Write per-panel estimates, session/population/prevalence summaries, equal-rat
descriptive summaries, paired readout and temporal contrasts, endpoint spike
support, technical audit gates, manifest with exact source/input/output hashes,
markdown and static figure. Report NaN/unidentified fits as failures, not drop
them from denominators. Retain the prior criterion of absolute error <=.05 in
at least 90% of replicas for each population/generator/prevalence.

Improvement with continuous scores implicates compression. Improvement with
oracle generator calibration implicates generator dependence. Improvement for
late jumps but deterioration for stationary in 5 ms implicates temporal mixing
versus spike loss. These are conditional diagnostic interpretations, not proof
of a uniquely isolated cause. KDE and finite calibration error remain possible.
No method is promoted to a real content estimator based on this diagnostic.
