# Held-Out-Population Calibration: Broad Coverage, Limited Information

Status: completed retrospective transfer test at the frozen source panel
budget. It does not establish a useful general-purpose correction or
biological uniform speed. A panel-size sensitivity is required before
generalizing the primary abstention result beyond this small-panel benchmark.

## Provenance and Verification

- Server: gpuserver6000.
- Artifact: `/mnt/seagate10tb/florianpfaff/replay-speed-population-transfer-all33-20260905/`.
- Source: audited `replay-speed-identifiability-all33-20260905`, unchanged
  panel arrays, schedules, baseline calibration code and local-reference tables.
- Scorer/auditor commit: `e385f374e51b4ec1c672d9360e5b38f4166abada`, clean.
- Manifest SHA256:
  `efccd0cbb2cf893d121329ba1d7b3ef4bba8289e1f902bb77b7a9d1f23d769f0`.
- 33 sessions, nine animals; 72 excluded-animal/readout fits and 422,400 test
  interval rows. These are synthetic panel predictions, not newly scored
  biological replay events. Runtime 37.20 s, BLAS/OMP single-threaded.
- All technical and full-scope gates pass. The independent auditor reconstructs
  all fits using least squares, all residual ranks, interval endpoints and
  decisions, and grouped counts/means. Seeded bootstrap reconstruction shares
  the summarizer implementation. No new raw-spike reconstruction is claimed;
  the consumed panels link to the earlier successful simulation audit.
- The original report remains at `report/`; `report-v2/` adds a post-inspection
  availability diagnosis using the unchanged five-event statistic requirement.
  No fit, interval, model or selection threshold changes in that update.
- 199 relevant tests pass, including 11 new tests for animal exclusion, test
  contamination, input/interval corruption, source completeness, abstention,
  scalar/vector agreement, reference pairing and report provenance. Ruff passes.

Protocol: `replay_speed_population_transfer_protocol.md`. All source panels
and code have recorded hashes and input snapshots.

## What Was Held Out

Within each dataset, exclude one animal and all its sessions from inverse
fitting and residual calibration. Fit the existing scalar inverse regression
on the other animals' A/Poisson fit panels; compute the existing pooled
conformal radius on their separate A/Poisson calibration panels. Evaluate all
test panels of the excluded animal without using any of its panels to learn
the calibration. Source RUN maps still belong to the decoded animal: cell
identities or rate maps are not transferred between animals.

This is a retrospective transfer baseline, not a prospective external-animal
study. Training panels are pooled with the existing equal-row weighting;
animals with more sessions and sessions with more finite fit rows can have
more influence. No population-adaptive covariates or new calibration algorithm
were optimized. Both source phase/test outcomes and the broader study had
already been inspected, so this must not be called a blinded replication.

Conformal exchangeability across new animals is not established. Consequently,
95% is a target evaluated empirically, not a finite-sample guarantee. The
leave-one-out folds also share training animals. Bootstrap intervals below are
descriptive animal-level summaries conditional on these fitted folds, not
refitting uncertainty or proof of universal new-animal coverage.

## Before Selection: Transfer Is Not Uniformly Worse

Posterior mean, >=2 cells/3 spikes, all-data diagnostic, uniform-g test panels.
Percentages average sessions within animal and animals equally.

| Dataset / test condition | Local conformal coverage | Excluded-animal conformal coverage [95% CI] | Excluded-animal finite output | Mean session median finite width, g |
|---|---:|---:|---:|---:|
| PF A/Poisson | 95.50% | 92.75% [84.00, 98.50] | 100.00% | 1.265 |
| PF B/shared gain | 76.75% | 86.00% [73.50, 94.50] | 100.00% | 1.265 |
| Tanni A/Poisson | 95.08% | 95.60% [93.80, 97.04] | 99.96% | 1.419 |
| Tanni B/shared gain | 89.40% | 91.88% [89.44, 94.28] | 99.24% | 1.419 |

A is the matched first RUN-half map; B is a disjoint RUN-half generator. Shared
gain is the frozen declared observation stress, not fitted replay covariance.

The Tanni combined-stress coverage improvement over local calibration is
+2.48 percentage points [1.04, 3.92], positive in all five animals. The PF
improvement is +9.25 points but its interval [-12.50, 29.00] crosses zero.
Other pooled-versus-local coverage differences also have intervals crossing
zero. It would therefore be incorrect to summarize this experiment as
"excluding animals always destroys calibration."

However, the intervals are broad relative to the generating g range [-0.75,
0.75], whose total width is 1.5. The pooled Gaussian widths are broader still
in the Tanni all-data diagnostic (about 1.57). Coverage alone cannot establish
that a recording distinguishes meaningful gradients. This experiment tests
normalized horizontal-coordinate speed modulation, not physical wall distance.

## Primary Selected-Event Result

The primary readout uses the selected continuous run, posterior mean and
per-bin support. Every excluded-animal regression is mathematically fitted,
but every primary conformal radius is infinite. All primary conformal
predictions therefore abstain in both datasets and all test conditions.

The Gaussian baseline produces some finite predictions, so do not call
the entire model stack numerically broken:

| Matching A/Poisson, primary | PF | Tanni |
|---|---:|---:|
| Conformal finite fraction | 0.00% | 0.00% |
| Conformal all-panel coverage | 100.00% | 100.00% |
| Gaussian finite fraction | 62.63% | 4.88% |
| Gaussian finite-only coverage, animal-balanced | 90.08% | 83.96% |
| Gaussian mean session median finite width, g | 1.205 | 1.325 |

Under B/shared gain, Gaussian finite availability falls to 32.63% PF and
0.32% Tanni. The latter's 100% finite-only coverage describes very few
measurable panels in only three animals and is not strong validation.

Neither inverse method makes a strict +/-0.25 equivalence claim anywhere in
the new 422,400 rows, including every estimator, support setting, selection
and fixed-gradient stratum. This includes constant-speed truth. The declared
band is a benchmark tolerance, not an established biological effect size.

## The Important Availability Diagnosis

The source panels contain 30 duration profiles per PF session and 11-30 per
Tanni session. The decoded-gradient statistic requires at least five
contributing selected events. Inspecting the primary A/Poisson calibration
panels gives:

| Dataset | Finite statistic | Fewer than five contributing events | Missing despite at least five events |
|---|---:|---:|---:|
| PF | 62.25% | 37.75% | 0.00% |
| Tanni | 5.62% | 94.38% | 0.00% |

Every missing primary calibration statistic in this source benchmark is
explained by the event-count gate, not by the remaining spatial-variance gate.
For PF, Rat1 has 100% finite calibration panels, Rat2 92.93%, Rat3 3.54%,
Rat4 52.53%. Tanni per-animal finite fractions range from 0.61% to 10.51%.
Thus pooling across other animals cannot make the 95th residual rank finite
while this many calibration panels contribute infinite residuals.

This diagnosis was added after inspecting the transfer result, using the
frozen threshold. It is not a new event selection. The per-session decomposition
is retained in the second report, including actual source/contributing counts.

## What This Resolves, and What It Does Not

The declared held-out-population test is completed: the calibration code,
source exclusions and empirical outcomes are verified. Matching coverage can
transfer in the all-data diagnostic, but useful selected-event equivalence is
not established. Broad or unbounded intervals should not be labeled success.

The abstention result is also partly an analysis-budget result. It would be
misleading to turn "too few trajectories in an up-to-30-path panel" into
"this recording can never identify speed modulation." The real recordings
contain more candidates than these simulated panels. Before the final paper
claim, quantify recovery/calibration versus candidate-panel size using larger,
paired known-path panels, keeping all current criteria and parameter choices
fixed. This asks whether more events restore measurability and informative
gradient recovery, rather than tuning a threshold to rescue uniformity.

The final methods/results/limitations pack and claim audit remain unfinished.
No biological uniform-speed claim follows from the completed transfer test.
