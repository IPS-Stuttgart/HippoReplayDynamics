# Training-Only RUN Validation Results

Status: completed prerequisite; the full recording-coverage study remains active.

## Provenance and Verification

- Server: gpuserver6000 (`workstation2`).
- Worktree: `/home/florianpfaff/HippoReplayDynamics-recording-coverage`.
- Scoring commit: `2126310afbbd3f4c33903b0e3b27c0d3ded5aa3a`, clean at invocation.
- Artifact: `/mnt/seagate10tb/florianpfaff/replay-coverage-RUN-validation-all33-20260906`.
- Invocation UTC: `2026-09-05T00:28:36.080322+00:00`. The `20260906` suffix is
  an experiment/seed label, not the actual date of invocation.
- Audit code commit: `5223334a`; no evidence or position decoding was rerun by
  the independent audit.
- All six technical gates passed: 33 sessions, 165 folds, 41,250 test centers,
  and 165,000 prediction rows (two durations times two likelihoods per center).
  Every fold retained its predeclared 250 behavior-selected test windows.
- The independent audit verified all prediction rows against raw spike counts,
  active-unit identities, tracked center positions, training guard intervals,
  and model/cache hashes. Zero discrepancies.
- Eight hashed inputs are frozen in `input_snapshot/`; the complete scoring
  commit is archived as `scoring-code-2126310a.tar.gz`.
- The focused test suite has 54 passing tests; Ruff and whitespace checks pass.
  A synthetic leakage test modifies all held-out/guard coordinates and spikes,
  including introducing new test-only cells, without changing the trained maps
  or selected units. The test caught an inclusive-endpoint issue before any
  production validation; the committed scorer excludes both guard endpoints.

Five chronological folds, one-second guards, and fresh training-only grid,
occupancy, firing maps, and unit QC were used. No all-RUN rate/stability mask
was reused. Training populations ranged from 63-216 PF units and 46-170 Tanni
units. See `replay_coverage_run_validation_protocol.md` for the frozen design.

## Spatial Information and Uncertainty Are Distinct

Primary analysis: independent uniform-prior Poisson decoding; all
behavior-selected windows, without any held-out spike-support exclusion.
Error entries are session medians, averaged within animal and then equally
across animals. Coverage is likewise equally animal-weighted. These are not
pooled errors from 165,000 independent observations; there are only nine animals.

### 250 ms RUN Windows

| Cohort | Sessions / animals | Posterior-mean error | MAP error | Wrong-cell-map mean error | Nominal 95% region coverage |
|---|---:|---:|---:|---:|---:|
| PF | 8 / 4 | 11.34 cm | 11.94 cm | 96.65 cm | 50.61% |
| Tanni, all arenas | 25 / 5 | 18.81 cm | 19.01 cm | 91.84 cm | 60.66% |
| Tanni, 8.75 m2 arenas | 5 / 5 | 27.57 cm | 26.10 cm | 166.57 cm | 54.16% |

The paired improvement over the wrong-cell-map control is positive in all 33
sessions. The control uses five cell-identity permutations, summarized
descriptively, not a permutation-significant biological claim.

However, the nominal 95% highest-density region contains tracked position only
39.52-70.56% of the time across individual 250 ms session results. Location
information is present, but the uncorrected posterior is not a calibrated
uncertainty estimate for tracked RUN position.

This is not explained primarily by unvisited test locations: true locations lie
in training support 99.15% of the time in PF and 99.42% in Tanni. Conditioning on
spatial support changes 95% coverage only to 51.05% and 61.02%, respectively.
Using integrated window-mean truth rather than tracked center position also
changes the aggregate results very little. Neither check identifies the
remaining cause; rate variability, correlations, map estimation, spatial
discretization, and represented-versus-physical position remain possible factors.

Count-conditioned decoding is not a remedy for this RUN miscalibration:
250 ms 95% coverage is 52.35% in PF and 62.88% in Tanni, with errors of 11.26 cm
and 18.91 cm. This sensitivity must not be confused with the earlier known
fixed-count simulation/Poisson-likelihood mismatch.

### 20 ms RUN Windows

| Cohort | Mean-position error, all windows | Windows with >=2 cells / >=3 spikes | Error within that supported subset | 95% coverage, all windows |
|---|---:|---:|---:|---:|
| PF | 40.81 cm | 43.95% | 23.88 cm | 86.46% |
| Tanni, all arenas | 61.33 cm | 26.17% | 40.02 cm | 92.02% |
| Tanni, 8.75 m2 arenas | 114.21 cm | 29.54% | 77.72 cm | 90.56% |

The higher short-window coverage is not evidence of better spatial resolution:
the estimates are much less precise, and most windows have little spike
support. Restricting to supported windows changes the target population; it
does not repair the omitted windows. RUN firing rates and theta-related
representations differ from replay, so these values must not be assigned to
replay events as their error bars.

## Arena-Size Readout

Each of the five Tanni animals has higher absolute 250 ms error in its large
arena than in the average of its two small-arena sessions. This is a descriptive
within-animal comparison, not a randomized isolation of area or cell coverage.
The larger absolute error does not show that decoding is proportionally worse
relative to arena dimensions, nor that this explains the replay-yield contrast.
Native PF wall coordinates are still unverified; no wall-distance result is
derived from tracking extrema here.

## Consequence for the Study

Do not promote posterior RMS or nominal credible mass directly into an adaptive
continuity allowance. The uncertainty itself needs validation, and calibration
on RUN cannot automatically be transferred to replay. See
`replay_coverage_novelty_scope.md`: neural-decoder overconfidence and calibration
are already published topics, not a stand-alone novelty claim for this project.

Next, the ground-truth recovery experiments must separate an oracle,
likelihood-matched Poisson baseline from map-estimation/observation-mismatch
sensitivities. Use all candidate support profiles, one draw per trial, and
independent evaluation seeds. Isolate spatial cell coverage from spike-count
information; measure false retention of stationary/discontinuous paths as well
as rejection of continuous paths. Recover positive and negative speed gradients
before interpreting any real-data near-zero association as uniformity.

The full goal is not complete. This validation establishes that the maps carry
held-out RUN position information and exposes a substantial uncertainty problem;
it does not establish replay speed, biological uniformity, or a new mechanism.

## Figure

`coverage_RUN_validation.png` is the original scoring-run figure. Its lower
titles were too long and clipped. `coverage_RUN_validation_display.png` is a
non-rescoring rerender from the same session CSV with shorter titles; its
separate manifest records the source table, rendering code, and image hashes.
No data, threshold, likelihood, or metric changed for that display correction.
