# Fixed-window screening bound: outcome

## Status

All 384 numerical bounds are independently certified. No observable screening
rule has been learned or independently validated. The user's remedy/diagnostic
goal remains OPEN. This is not a replacement for that goal.

Original baseline data only: four matched PF population pairs across three rats,
1,836 original candidate endpoints and 513 accepted-segment endpoints. No cells,
event windows, maps, likelihoods or original readouts were changed or rescored.

## Primary half-retention result

Progress t=1 requires 20% reduction in the absolute mean Home-content gap and
10% reductions in mean positional separation and regional TV, without raising
either posterior entropy. Known-position truth guards additionally forbid
increased class-balanced position error and Home Brier loss for either population.

| Source | Same-count oracle cases reaching t>=1 | Range of optimal t |
|---|---:|---:|
| Original candidates, early maps, no unknown-truth guard | 4/4 | 3.722-4.653 |
| Accepted endpoints, early maps, no unknown-truth guard | 4/4 | 3.890-5.000 |
| Native held-out Q4, truth guarded | 4/4 | 3.782-4.909 |
| Matched Poisson, truth guarded | 4/4 | 1.379-2.746 |
| Gain-4 Poisson, truth guarded | 1/4 | 0.851-1.166 |
| Conditional total counts, truth guarded | 4/4 | 1.161-1.546 |
| Conditional map drift, truth guarded | 4/4 | 1.262-1.870 |
| Conditional shared assembly, truth guarded | 4/4 | 1.157-2.150 |

Both full-map real-data sensitivities also reach the numerical targets for 4/4
pairs. These are separate optimized selection rules, not one common solution.
Native Q4 refers to original 20-ms RUN observations with known physical positions,
not to replay ground truth.

The three gain-4 failures are Rat1/Open1 (t=0.853576), Rat1/Open2 (0.851469),
and Rat4/Open2 (0.894322). Even the unrestricted free-event oracle fails there.
At 25% retention all 24 same-count truth-guarded cases reach target. At 75%,
only 9/24 do. These are predeclared sensitivities, not a switch to a passing
primary retained fraction.

## What changed our next action

Exact-count indistinguishability is not the dominant restriction in these banks.
At half retention the largest free-versus-same-count truth-guarded difference
in t is 0.144; the real-data bounds agree to numerical precision. However,
95.7-100% of original candidate count vectors are unique within their sessions.
Thus this tied oracle can still memorize almost every real observation. It
provides no evidence of out-of-sample prediction or a reliable feature space.

Known-truth safeguards are materially restrictive. For conditional-count
simulations, same-count t falls from 2.430-3.347 without accuracy guards to
1.161-1.546 with guards; drift shows the same pattern. Improving agreement
alone must not be counted as a remedy.

The next aligned experiment would learn ONE observable selection policy using
development RUN/simulation data with the accuracy and agreement requirements
jointly represented, freeze it, and evaluate its transfer on untouched
observations and independent recordings. Oracle probabilities and target-bank
labels must not be supplied to a claimed independent predictor. A simple
agreement threshold cannot inherit these oracle guarantees. No new external
scoring or biological scaling was performed here.

## Scope of the bound

Selection weights are fractional, not a realized integer cohort. The LP enforces
exact retention per true Home/non-Home class and per session, stricter than
earlier empirical per-rat/pooled screens. The truth sources also receive their
own agreement-reduction targets, not merely the no-harm guard required by the
original real-data question. Therefore gain-4 infeasibility does NOT prove the
user's entire requested remedy impossible. Conversely, real-data feasibility
does not certify the unknown true replay content.

No decision threshold changed after seeing outcomes. The original protocol was
committed at 5c6e2df4; implementation at 28d93063. The initial run failed a
numerical constraint audit, was preserved, and was repaired only by equivalent
constraint scaling (e3f4b4ed). Independent auditing still uses the original
unscaled constraints and original tolerances. See the numerical-repair note.

## Verification and artifacts

Authoritative run:
`/mnt/seagate10tb/florianpfaff/content-screening-bound-v2-20260915`
on gpuserver6000, detached user service `content-screening-bound-v2-20260915`.
Terminal exit 0, no live worker. Measurement plus audit: 33.38 seconds.

Independent reconstruction checks all source identities/hashes, 384 primal/dual
certificates, exact-count groups and recomputed weighted readouts. Maximum
constraint residual 5.48e-12; stationarity residual 9.26e-12; duality gap
3.86e-12; complementarity residual 4.60e-11. Independent auditor does not import
the producer or optimization solver.

Measurement manifest SHA256:
`102d4e237dcc8dd8aff717675bf8a2abab012f16c1a8b84398cf741f0d21f4a8`.
Related synthetic and artifact tests: 66 passed; Ruff passed. The known optional
Matplotlib Axes3D warning has no effect on the inspected 2D figure.

Compact report: `report-final/report.md`, `screening_bound.png`,
`attainability_summary.csv`, `oracle_comparison.csv`, `primary_cases.csv`,
`report_manifest.json`. The original failed numerical run is separately retained
at `content-screening-bound-20260915`; never use it as a certified result.
