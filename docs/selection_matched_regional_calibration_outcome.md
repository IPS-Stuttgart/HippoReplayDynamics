# Selection-matched calibration: development stop

2026-09-15. Final measurement: selection-matched-regional-pf-v3-20260915.
No hc-11 transfer; no calibrated real regional-content claim.

## Executed design

Eight PF sessions, four rats, 200 frozen candidate time profiles per session.
Exact spike timestamps and whole-population count traces were preserved; cell
identities were generated from fixed first-half RUN maps. Original detection
boundaries and the active-cell support gate were verified by native-detector
reruns. There were 5120 simulated panels, each based on 200 templates, and 22
population definitions: eight full, eight targeted high/low, six whole-tetrode.

Calibration, null-threshold, validation and perturbation draws used separate
seeds, apart from the deliberately identical global-gain negative control.
This is conditional validation on fixed real candidate profiles, not a new
unconditional replay-detection experiment. Simulated terminal content is known;
real terminal content is not.

## What passed and failed

The technical audit passed. The proposed calibrated estimator did not.

Full-population, genuinely matched generator-mix validation:
- Mean absolute prevalence error: 8.72 percentage points.
- Fraction within the predeclared 5-point budget: 40%, versus required 90%.
- Mean false-flag rate: 2.66%; some individual population/prevalence checks
  nevertheless exceeded the predeclared 10% finite-pilot tolerance.

Pure-generator stress tests use the SAME frozen mixture calibration, not
generator-specific calibration: stationary error 13.54 pp, moving 12.85 pp,
late-jump 62.30 pp. The late jump occurs in the last 5 ms of a 20-ms readout;
earlier spikes can describe the other region. These stress results must not
be described as failures of a correctly generator-specific decoder/calibrator.

No population passed the full requirements (0/22). No matched high/low group
passed agreement (0/7) or eight-point power (0/7). Targeted matched groups agreed
within 5 pp in 35-38.75% of mixed validation panels, not 90%. Detection of a true
8-point difference was only 5-40% across targeted and tetrode comparisons.
That power check uses separate .30/.38 panels, not inconsistent simultaneous
truth assignments to cells shared by the original matched populations.

## Why a small mismatch is not enough

Adding a 2-Hz nonspatial component to Home-peak cells produced mean absolute
prevalence error 20.29 pp for the full population, but flags occurred in only
1/160 perturbed panels. Thus a low-false-flag fit diagnostic need not detect a
large content-estimation error. The perturbation sensitivity is insufficient
to certify real content, even in a real session that is not flagged.

Interpretation: the mixture fit can absorb a changed readout distribution as a
different prevalence, without necessarily leaving a large goodness-of-fit
residual. This is an explanatory inference from the controlled counterexample,
not evidence that these perturbations occurred in the real recordings.

Together with the preceding selection null, this establishes two distinct
possibilities: large calibration mismatch with small prevalence error, and
small mismatch with large prevalence error. Neither mismatch nor non-mismatch
alone validates the regional estimate.

## Real-data boundary

Seven of eight full-population real sessions exceeded their matched-null TV
cutoff; Rat1/Open2 did not. This was not a Rat3-only result. None passed the
synthetic calibration requirements, so no real enrichment or reconciliation
conclusion is licensed. Descriptive nuisance mixture estimates, including
boundary estimates at zero, must not be reported as true Home prevalence.

The result rejects this frozen ternary, equal-generator-mixture calibration
as a 5-point-accuracy remedy at this endpoint resolution and sample size. It
does not reject every continuous-readout method or prove real content is
universally unidentifiable. Distinguishing within-window terminal ambiguity,
generator-mixture uncertainty, and information lost by ternary thresholding
would be a separate, predeclared next diagnostic, not a reason to loosen gates.

## Verification and retained attempts

- 55 relevant tests pass; Ruff passes.
- Independent audit checks input/output hashes, exact endpoint totals, active
  support, truth, seed separation, redraws, every saved likelihood and call,
  calibration counts, gain invariance and 24 full native-detector reruns.
- First attempt stopped on a Rat4 clock assertion; 1e-9-second completion
  tolerance was aligned with the original cache and covered by a regression test.
- v2 completed simulations; the CSV token `null` was parsed as missing in pooled
  reports. v3 preserves stage strings. All 16 simulation/readout archives are
  byte-identical between v2 and v3. No scientific parameters or seeds changed.
- Failed/intermediate artifacts were retained; no existing research work reverted.

Server code: /home/florianpfaff/HippoReplayDynamics-content-stability-20260914
Raw: /mnt/seagate10tb/florianpfaff/selection-matched-regional-pf-v3-20260915
Audit: /mnt/seagate10tb/florianpfaff/selection-matched-regional-pf-v3-20260915-independent-audit.json
Report: /mnt/seagate10tb/florianpfaff/selection-matched-regional-pf-report-v2-20260915

Code remains uncommitted; source hashes supplement the base commit and dirty flag.
