# Speed-Interval Calibration: Recovery, Transfer, and Abstention

Status: completed and reconstructed conditional surrogate benchmark. Not a
biological speed-equivalence result or a validated universal correction method.

## Provenance and Verification

- Server: gpuserver6000.
- Artifact: `/mnt/seagate10tb/florianpfaff/replay-speed-identifiability-all33-20260905/`.
- Scorer: clean commit `2d6a0f082fd4da4aa9ab89735bfb52cc9596fe53`.
- Reconstruction/reporter: clean commit `ac1f39b9f292f8f1a7ee34063e3ea2dce8aef4fe`.
- Production runtime: 220.916 s, 16 session workers, BLAS/OMP single-threaded.
- Python 3.12.3, NumPy 2.4.6, SciPy 1.17.1, pandas 3.0.5.
- 33 sessions: eight PF/four rats, 25 Tanni/five rats, all arena sizes.
- 971 frozen source duration profiles: 240 PF, 731 Tanni. Up to 30 per session,
  from the earlier uniformly sampled high-MUA candidates, not continuity winners.
- 247,896 readout rows; 633,600 test interval decisions. These are repeated
  synthetic panels/readouts, not additional biological events or animals.
- All technical gates pass. Every saved interval/decision and all input/output
  hashes verified; all 33 A maps refitted exactly. Sampled reconstruction covers
  264 draws and 6,864 condition/readout rows, including each fixed-gradient
  stratum in every session. It reuses the simulator/decoder implementation and
  is not an independent implementation or a reconstruction of every simulation.
- All seven reporter outputs hash-verified; both figures visually inspected.
- 140 relevant tests pass and Ruff passes. Sequential/parallel two-session
  smoke panel, fit, decision and schedule tables match exactly. Smoke results
  were not used to tune the frozen production thresholds.

Protocol: `replay_speed_identifiability_protocol.md`. Manifests include input
snapshots/hashes, exact CLI, library versions, seeds, commit and working tree.

## What Was Tested

Generate speed `v(x)=1000*(1+g*q(x))` cm/s, with horizontal coordinate q in
[-1,1]. This is not a physical wall-distance test. Fit an inverse scalar map
from the decoded speed-gradient statistic T to imposed g on 40 A/Poisson
panels; calibrate with 99 new panels; test 100 new uniform-g panels plus 20
each at g=-0.5,-0.25,0,+0.25,+0.5. Each panel uses every frozen session profile.

A is the first RUN-half population; B is the disjoint second-half generator.
All four test conditions decode with A. Shared gain is a declared mean-one,
CV=1, 20 ms block stress, not measured replay covariance. No B/gain test data
enter inverse fitting or calibration. Both empirical halves constrain the
simulation domain; no new biological animals are held out.

Compare raw decoded-slope event-bootstrap intervals, inverse OLS Gaussian
prediction intervals and inverse split-conformal intervals. The latter are
marginal prediction intervals for a randomly drawn surrogate g, NOT confidence
intervals with guaranteed coverage at every fixed biological gradient. Fixed-g
panels explicitly examine that limitation. These are standard baseline methods,
not a new conformal algorithm.

The primary readout is posterior mean, >=2 active cells/3 spikes per bin,
within the selected continuous run. The statistic requires >=5 contributing
events and spatial variance >=0.01. Inverse fits require >=20 finite fit panels.
Missing calibration statistics contribute infinite residuals; missing fits or
test statistics abstain with unbounded intervals. No interval is clipped to
the generating parameter range. All-data readouts are diagnostic, not a
post-hoc replacement for the primary selected-event analysis.

## Matching Calibration Does Not Ensure Transfer

Below: posterior mean, bin-supported, before continuity selection, uniform-g
test panels. Percentages average sessions within animal and animals equally.
Intervals in brackets are animal-bootstrap intervals, with only four/five
animals; they condition on the fitted calibration and simulated populations.

| Generator / observation | PF conformal coverage | Tanni conformal coverage |
|---|---:|---:|
| A / Poisson (matching) | 95.50% [94.00, 97.13] | 95.08% [93.84, 96.24] |
| A / shared gain | 85.50% [76.63, 92.50] | 92.52% [91.92, 93.12] |
| B / Poisson | 89.50% [87.38, 91.38] | 93.44% [92.00, 94.92] |
| B / shared gain | 76.75% [65.50, 86.75] | 89.40% [87.68, 91.08] |

Finite availability is 100% PF and 99.24-99.96% Tanni in this diagnostic, so
these drops are not explained by counting unbounded intervals as covered.
Combined B/gain coverage is lower than matching A/Poisson in every animal.
Matching raw bootstrap coverage is only 78.13% PF/79.04% Tanni; inverse
Gaussian gives 96.88%/96.08%. Thus calibration improves matching-simulation
coverage, but the improvement does not certify real-data transfer.

Conformal intervals are broad: animal-averaged session median widths are
1.118 PF and 1.421 Tanni in g units under matching conditions. The entire
simulated uniform-g range has width 1.5. Nominal coverage alone is not useful
identifiability. The unfiltered diagnostic figure shows the same qualitative
transfer failure, with different numbers; do not substitute it for this table.

## Primary Selected-Event Result: Predominantly Abstention

| Matching A/Poisson, primary readout | PF | Tanni |
|---|---:|---:|
| Finite conformal interval fraction | 25.00% | 0.00% |
| All-panel coverage, including unbounded intervals | 98.38% | 100.00% |
| Finite-only coverage | 93.50%, Rat1 only | unavailable |
| Finite inverse-Gaussian interval fraction | 60.13% | 0.00% |
| Finite raw-bootstrap interval fraction | 56.13% | 3.24% |

Only Rat1/Open1 and Rat1/Open2 provide finite primary conformal radii, 0.421
and 0.301 respectively. Rat2's two sessions and Rat4/Open2 fit the inverse
regression but have too many missing calibration statistics for a finite
95% conformal radius. The remaining three PF sessions fail the minimum fit
availability. All 25 Tanni sessions fail that fit requirement (0-11 finite
fit panels out of 40). Their 100% all-panel coverage is entirely vacuous.

Under B/shared gain, PF finite primary conformal availability is 23.63% and
finite-only coverage is 69.80%, still Rat1 only. No cross-rat reliable primary
calibration result can be claimed. This is a limit for the declared up-to-30
profile panels, speed/encoding settings and criteria, not proof that additional
data or a different validated estimator could never recover the gradient.

## Equivalence and False Claims

Predeclared primary equivalence is strictly |g|<0.25, with 0.10/0.50 bands
reported as sensitivity. This is a benchmark tolerance, not a biologically
established effect size. Boundary g=+/-0.25 is outside equivalence.

Neither calibrated method makes any +/-0.25 equivalence claim in any tested
readout/condition/gradient. This includes zero-gradient tests. It demonstrates
lack of informative equivalence, not success in proving uniform speed.

The primary PF raw-bootstrap baseline claims equivalence in 17.50% of matching
g=0 panels, but also in 6.88% at g=+0.25 and 7.50% at g=-0.25, which are false
claims under the prespecified strict band. These are animal-balanced
descriptions from 20 fixed-g panels/session, not precise biological error
rates. Per-session Wilson Monte Carlo intervals and exact denominators are
in `report/speed_identifiability_session_monte_carlo.csv`; they condition on
the learned fit/radius rather than accounting for refitting uncertainty.

## Implication for the Paper

The constructive result is a measured inference boundary: continuity selection
can leave too little data for speed-gradient inference, and a simulation
correction that calibrates under its own observation model may not transfer.
An honest procedure can abstain instead of translating a flat decoded profile
into biological uniformity. This strengthens the proposed measurement study,
but does not establish a universal abstention/calibration solution.

Remaining work is not another identical draw expansion. It is real
event-definition sensitivity, comparison to established replay-detection
baselines, and an integrated limits-of-inference report. New-population
transfer remains necessary before claiming a generalizable calibrated method.
No real replay equivalence test, new replay-event selection or biological
scoring run was authorized by these surrogate results.
