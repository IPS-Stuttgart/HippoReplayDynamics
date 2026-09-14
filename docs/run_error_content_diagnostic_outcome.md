# RUN-error-trained content diagnostic: failed external screen

## Decision

The frozen PF-trained diagnostic is NOT a validated remedy for population-content
instability. It improves known-position accuracy slightly in a retained subset,
but does not reduce disagreement between disjoint populations on Tanni. Its
absolute error predictions also fail external transfer. Do not replace the full
predictor, change retention or choose another split after seeing this outcome.

This turn made concrete progress: recovered both completed external applications,
expanded independent numerical reconstruction to all cases, completed a tested
non-rescoring reporter, and independently checked its primary aggregates. The
user objective remains active and unachieved, not blocked.

## Frozen experiment

- Training: 7,559 third-quarter RUN windows from 8 PF sessions / 4 rats.
- Target: log(1+B known-position error / arena diagonal), not A/B agreement.
- Only population A observations plus A/B RUN maps are predictors. No B replay
  spikes, posterior, coordinates, agreement or truth enter prediction/selection.
- First-half encoding; disjoint equal cell halves; flat-prior Poisson, 20 ms.
- Ridge alpha=10. Constant and spikes+entropy baselines frozen alongside full.
- PF uses leave-one-rat-out models; Tanni receives fixed all-PF coefficients.
- External: all 25 Tanni recordings / 5 rats, 8,072 original candidate endpoints
  and 19,596 held-out RUN windows in primary split0. Other splits sensitivity only.
- Select the lowest predicted risk 50% per session/source/split/draw (rounded up).
  Baseline is the exact expected mean under random fixed-size retention, not an
  actual randomly selected half. Animal and session contributions are equalized.

Tanni is external to training, but these recordings have been inspected under
earlier failed diagnostics. This is external development validation, not pristine
confirmation. Full-RUN unit eligibility and spatial-mask construction remain
limitations inherited from the original caches.

## Primary external results

| Outcome | Random-retention expectation | Retained full-risk half | Change |
|---|---:|---:|---:|
| Real endpoint A/B separation (cm) | 42.3752 | 42.9922 | +0.6170, worse |
| Real nine-tile posterior TV | 0.445197 | 0.469179 | +0.023981, worse |
| Real mean A/B normalized entropy | 0.861441 | 0.847972 | -0.013469, sharper |
| Known RUN mean A/B position error (cm) | 66.3148 | 63.2128 | -3.1020 |
| Matched-simulation mean A/B error (cm) | 55.4573 | 51.9984 | -3.4589 |
| Drift-simulation mean A/B error (cm) | 56.8051 | 53.1924 | -3.6127 |

Known-error improvements remain after matching the selected true-tile distribution:
-2.2523 cm RUN, -2.5772 cm matched simulation, -2.5252 cm drift simulation.
Neither population's known accuracy is made worse on average. This differs from
the earlier agreement-trained policy, which selected broader, less accurate
posteriors. But improved accuracy here is NOT improved cross-population stability.

External forecast log-MSE: constant 0.0112050, spikes+entropy 0.0112210, full
0.0650285. Full is approximately 5.80 times the constant baseline, rather than
the required >=5% improvement. Predicted risk versus physical B error correlation
is negative in all five animals when combining their sessions with equal-session
weights. The alternative log-normalized target correlations also do not provide
a consistent rescue; they remain available in the table.

Regional TV improved in only 1/5 animals; equal-animal relative change is +5.39%
(worse). Its descriptive bootstrap interval for baseline-minus-selected is
[-0.04019, -0.00653]. Endpoint separation improved in 3/5 animals but worsened
1.46% overall; reduction interval [-2.4579, +0.9956] cm. Both predeclared >=10%
stability gates fail. Accuracy and non-broadening gates pass; the overall screen
fails. These five-animal intervals are descriptive, not population guarantees.

## Exploratory failure localization, not a repaired result

All Tanni arena-size values fall outside the PF training range. The logarithmic
arena-size feature spans 3.5038-3.5455 during training, versus 2.8919-3.9705 in
Tanni. Its training SD is only 0.01450; its external linear contribution ranges
from -0.2216 to +0.2993 in predicted log-risk units. This is large extrapolation.
The largest Tanni arenas receive near-zero clipped risk despite high true error.

Within individual sessions, risk/error correlation is positive in 23/25 cases,
but mostly small (approximately -0.020 to +0.233 across all sessions). This is
consistent with the severe pooled failure involving between-environment transfer.
It does not rescue calibration or the failed real-content stability gates.
No feature was removed, model refitted or result relabeled following inspection.

This experiment therefore does not show that true position error is intrinsically
unpredictable. It shows that this particular fixed cross-dataset predictor and
retention rule do not solve the requested problem. Agreement and known accuracy
must continue to be evaluated separately.

## Verification and reproducibility

- Producer frozen in 4eae4db8 before application. Audit b2662668; reporter
  64c78ce9; report audit / constant-correlation fix 7319d8ec.
- Independent audit checks all 33 sessions: raw RUN and endpoint counts, native
  truth coordinates, frozen cell IDs/maps, all real and synthetic posterior
  moments/regions/errors/true tiles, coefficients, features and predictions.
- Independent aggregate audit reconstructs 53 primary metric/source groups and
  the three forecast MSEs through separate aggregation and selection code.
- An initial report gave tiny finite correlations to a mathematically constant
  baseline due to floating-point centering. Report-v2 correctly returns undefined.
  No producer output changed; original report remains preserved. Primary gate
  CSVs are byte-identical before/after this correction.
- All services completed with exit 0, detached on gpuserver6000. Original raw
  recordings and previous failed experiments remain unchanged.

Authoritative root:
`/mnt/seagate10tb/florianpfaff/run-error-content-diagnostic-20260914/`
Subdirectories: `frozen/`, `pf/`, `tanni/`, `audit/`, `report-v2/`, `report-audit/`.
Code: `/home/florianpfaff/HippoReplayDynamics-content-stability-20260914`.

The original targeted matched-population Home-content contrast has not been
corrected by this method. Its additional transfer gate was not run because this
predeclared external screen failed. Do not mark the broader objective achieved.
