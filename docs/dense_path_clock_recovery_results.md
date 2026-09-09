# Dense-path clock recovery: 2026-09-10

## Decision

Numerically verified, but recovery and path-integration convergence fail.
Do not apply this estimator to real replay events to claim a physical versus
neural clock. This is not a uniform-speed result or a fuzzy-continuity test.

Increasing independent candidate-path support substantially reduces the earlier
positive bias in estimated neural-clock fraction. It does not remove the bias,
restore interval coverage/power, or stabilize estimates between independent
scorer libraries. This implicates inadequate path integration as part of the
measurement problem; it does not prove an absence of biological information.

## Frozen inputs and provenance

- Server: gpuserver6000; worktree
  `/home/florianpfaff/HippoReplayDynamics-dense-path-clock-recovery`.
- Branch: `test-dense-path-clock-recovery`.
- Producer/protocol: `654b09af90ca9315b8fc84df084bc233b8ab20b9`.
- Non-rescoring reporter: `1f6951d1`.
- Independent vectorized audit integration: `f3165050`.
- Parent observations:
  `/mnt/seagate10tb/florianpfaff/unknown-path-clock-populations-all33-20260909`.
- Run:
  `/mnt/seagate10tb/florianpfaff/dense-path-clock-recovery-all33-20260910`.
- Audit/report/validation directories use the same prefix followed by
  `-audit`, `-report`, and `-validation-final`.
- Run manifest SHA256:
  `81aae0a1e74c8325864c15f8886ed32335e3095c7beee5d337ae0a8e12d4ce3e`.
- Audit SHA256:
  `8eaea876dc390a0029ace37fe01da981ac72b61c158699612ba56560c303145a`.

All 1,267,200 frozen simulated observations were reused unchanged: 33 encoders,
nine animals, 50 replicates, three true clock-mixture fractions, and two original
teacher banks. Two new independent scorer banks each use nested supports of
1,024, 4,096 and 8,192 paths. Both are independent of both teachers; no true path
was supplied or deliberately inserted. Native count profiles and the original
straight/sinusoidal geometry prior were retained.

Five component weights are fitted: physical, neural, stationary, physical-reset
and neural-reset. The target fraction is neural / (physical + neural), not the
fraction of all observations. True coherent mass is 0.60; stationary mass is
0.20 and each reset mass is 0.10. Reset models redraw a path each time bin.

## Primary difficult corpus

Mean fraction estimates, teacher `original_bank_1`:

| Dataset | True neural fraction | Previous 256-path estimate | New 8,192-path bank 0 | New 8,192-path bank 1 |
| --- | ---: | ---: | ---: | ---: |
| Pfeiffer/Foster | 0.25 | 0.600 | 0.447 | 0.450 |
| Pfeiffer/Foster | 0.50 | 0.629 | 0.590 | 0.582 |
| Pfeiffer/Foster | 0.75 | 0.641 | 0.705 | 0.721 |
| Tanni | 0.25 | 0.578 | 0.355 | 0.345 |
| Tanni | 0.50 | 0.657 | 0.560 | 0.558 |
| Tanni | 0.75 | 0.713 | 0.740 | 0.713 |

The previous 256-path scorer used a different independent bank. This is a
same-observation comparison, not a claim that the old library is nested inside
the new one. Within the new experiment, supports are nested in each bank.

At 8,192 paths the primary-corpus recovery gates fail in both banks/datasets:

| Dataset / bank | Worst absolute bias | Minimum coverage | Minimum directional power | False direction at 50% |
| --- | ---: | ---: | ---: | ---: |
| Pfeiffer/Foster / 0 | 0.197 | 0.38 | 0.14 | 0.22 |
| Pfeiffer/Foster / 1 | 0.200 | 0.38 | 0.06 | 0.16 |
| Tanni / 0 | 0.105 | 0.84 | 0.26 | 0.06 |
| Tanni / 1 | 0.095 | 0.86 | 0.30 | 0.08 |

The second teacher corpus also fails every full recovery gate. In particular,
passing one bias threshold is not enough when interval coverage and power fail.
Fifty simulations give coarse false-direction estimates; Wilson Monte Carlo
intervals are retained, and 3/50 is not treated as decisive evidence by itself.

All eight integration-stability rows fail. Median paired bank disagreement is
0.053-0.070, above the frozen 0.025 limit. Median estimate shifts from 4,096 to
8,192 paths are 0.040-0.047, also above 0.025. Close bank-average estimates do
not imply agreement on the same individual population replicates.

## Verification

- All 7,603,200 score rows and 3,600 population fits completed; production
  runtime 865.5 seconds. No biological events were rescored.
- All 540,672 scorer paths independently reconstructed from seeds/source maps.
- All five likelihoods at all supports/banks independently recomputed for the
  first two frozen observations per teacher/scenario/replicate/recording:
  19,800 observations, 594,000 likelihoods. Maximum discrepancy
  `1.024e-10`. This is a stratified audit, not a full likelihood re-computation.
- Every mixture/profile optimum certified independently; all summary, recovery
  and convergence tables recomputed.
- 103 production output hashes and six report output hashes checked.
- 208 focused/regression tests passed; Ruff check/format checks passed.
- The 2040 x 1615 recovery figure was visually inspected; nonblank pixel checks
  passed and no text overlap was observed.

## Scope and next methodological decision

The result supports keeping the biological mechanism inference stopped. More
paths help, so the current failure must not be summarized as proof that these
recordings contain no clock information. Conversely, modestly improved estimates
do not establish a publishable biological result.

A future integration method would need to locate plausible paths efficiently
without receiving the generating path, preserve importance weights or an equally
valid marginal-likelihood calculation, and pass recovery on these same frozen
observations. Giving the scorer the true path, selecting an easier observation
subset, loosening the gates, or interpreting a nonsignificant speed correlation
as uniformity would not solve this measurement problem.
