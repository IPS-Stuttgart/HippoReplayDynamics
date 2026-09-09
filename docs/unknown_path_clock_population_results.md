# Unknown-path population clock recovery: matched signal, geometry confounding

## Status

This changes the measurement diagnosis, not the biological claim. Weak
single-event clock recovery does not imply that population inference is
impossible. Under the matched finite path library, both datasets recover known
mixture fractions with small bias. Under independently generated geometry,
the current finite-library estimator becomes strongly biased. Do not apply
this estimator to real events or interpret it as evidence of constant speed.

This experiment did not test fuzzy continuity, rescore real events, prove
biological equivalence, or establish a novel high-importance-paper finding.

## Frozen run and verification

- Producer: `dac19196eae951ef831dc3d1f75854323fa24b63`.
- Run: `/mnt/seagate10tb/florianpfaff/unknown-path-clock-populations-all33-20260909`.
- Manifest SHA256: `c808d0658ba0c3557d3dd8c0d762c372bdccee261e865cef1ab0f2975df3a5ba`.
- Independent audit: the same path with `-audit/unknown_path_clock_audit.json`.
- Non-rescoring report: the same path with `-report/unknown_path_clock_population_report.md`.
- 33 encoders, 9 animals, 1,267,200 fresh observations, 2,534,400 score rows,
  1,200 population fits; simulation runtime 117.83 seconds.
- Independent regeneration checked all observations, all 12,672,000
  likelihoods and all 1,200 mixture/profile optimality certificates.
- Maximum score disagreement: 1.05e-9 log-score units. Audit passed.
- Regression suite: 197 tests passed; Ruff check and format passed.

The 1.27 million observations are Monte Carlo replicates, NOT real replay
events or independent animal samples. Population size was fixed at 128 events
per recording: 1,024 for PF, 3,200 for Tanni. Native source profiles with at
least 3 complete 20-ms bins were sampled with replacement; even recordings
with few native candidates supplied 128 synthetic observations per replicate.
This design is a calibration, not a claim that each recording has that many
usable biological events.

## Primary 256-path results

Target is the neural-clock fraction among coherent events. The generating
mixture also contains 20% stationary, 10% physical-reset, and 10% neural-reset
events. All five weights were fitted, not supplied to inference.

| Dataset | Generating paths | True fractions | Mean estimated fractions | Worst bias | Minimum directional power | False direction at 50% |
| --- | --- | --- | --- | ---: | ---: | ---: |
| PF | Scorer's same library | .25, .50, .75 | .258, .504, .753 | .008 | 94% | 1/50 (2%) |
| Tanni | Scorer's same library | .25, .50, .75 | .245, .491, .748 | .009 | 94% | 3/50 (6%) |
| PF | Independent library | .25, .50, .75 | .600, .629, .641 | .350 | 0% | 22/50 (44%) |
| Tanni | Independent library | .25, .50, .75 | .578, .657, .713 | .328 | 0% | 27/50 (54%) |

Matched PF passes all frozen practical gates. Matched Tanni passes bias,
coverage and power targets but misses the <=5% false-direction target:
3/50 versus at most 2/50. The Monte Carlo Wilson interval for 3/50 is
approximately 2.1%-16.2%, so this alone is not compelling evidence that the
nominal interval procedure is miscalibrated. Keep the failed gate visible;
do not change its threshold or treat it as a catastrophic loss of information.

Both independent-path conditions fail decisively. At a true 25% neural-clock
fraction, the estimator instead reports roughly 58%-60%; interval coverage is
0% for PF and 2% for Tanni. At a true 50% fraction, false-direction rates are
44% and 54%, far above the nominal target. Coverage and false-direction errors
coincide here because an interval excluding 50% also misses the true fraction.

The 128-path sensitivity is also poor. In the matched-teacher condition it
deliberately omits half the generating support and is not exact-model
calibration. Increasing 128 to 256 candidates does not resolve the independent
path failure. The fitted coherent mass falls from the true .60 to about .43
(PF) and .49 (Tanni) for the independent/256 condition, indicating that path
mismatch also changes allocation to nuisance components.

## What the experiment identifies

The scorer did not receive true path indices. It marginalized over a common
candidate library for both clocks. Matched/256 is nevertheless favorable:
the true path is somewhere in the library and the finite generative prior is
exactly represented. Independent/256 tests new geometry from the same path
sampling rule and exposes finite support/prior approximation problems.

Therefore the result is neither "there is no information" nor "the neural
clock is supported." It identifies a representation/calibration problem in
this estimator. The real-event mechanistic comparison remains unvalidated.
The clock is specifically arc length in the conditional rate-code Hellinger
metric, not a complete model of neural-space motion.

## Next safe experiment

Improve path integration and verify convergence on independently generated
paths, keeping the same two clocks, nuisance components, source profiles,
mixture scenarios, and false-claim checks. Importance sampling or substantially
denser candidate support must be normalized correctly and tested for numerical
convergence; using a best-fit path is not a substitute for marginal likelihood.

Do not achieve apparent recovery by narrowing the generator to paths already
in the scorer's library, dropping difficult controls, or changing the true
mixtures. A held-out geometry recovery pass is required before any real-event
mechanism estimate. Once that exists, compare physical and neural clocks
across the two datasets with animal/session robustness; do not infer a
high-importance-paper conclusion from this synthetic result alone.
