# Terminal-segment mixture calibration: outcome

Date: 2026-09-16. Known-truth synthetic development experiment, not real replay inference.

## Frozen question

Does extending the terminal segment to 20, 40, 60 or 100 ms yield an estimator of
regional-content prevalence that tolerates unknown stationary/moving/jumping
mixtures? Home content means at least 20 ms cumulative geometric dwell inside
the Home disc during that segment. It does not mean the endpoint is at Home.

The protocol was written before running the experiment. The geometry-blind
shared bank supplies 1,139 common candidate-count templates across eight
Pfeiffer/Foster sessions and four rats. All 461 shorter original candidates
remain excluded. Seven dynamics/scale strata are tested at fixed positive-label
quotas, with four calibration replicas and four validation replicas at each of
four prevalences. Null panels were not used.

The primary method decodes non-overlapping 20-ms bins independently with a flat
spatial prior, combines their regional masses into an independent-bin union
score, and calibrates that scalar against the actual geometric dwell label.
This is a surrogate readout, not an exact posterior for cumulative dwell.
Silent bins are neutral and the calibration explicitly represents the zero
atom. There is no HMM, inferred dynamics type, or readout threshold tuning.

## Decision

**No tested segment length meets the frozen development budget.** No Delta* is
selected and no calibration is authorized for deployment on real replay events.

| Segment | Mean worst-mixture error | Worst session/prevalence mean error | Validation cases within 5 pp |
| --- | ---: | ---: | ---: |
| 20 ms | 6.41 pp | 10.32 pp | 34.4% |
| 40 ms | 5.68 pp | 10.01 pp | 45.3% |
| 60 ms | 6.58 pp | 13.70 pp | 37.5% |
| 100 ms | 9.12 pp | 19.17 pp | 21.1% |

Each row has 128 session/prevalence/replica validation cases. Within each case,
the error is the largest error over common-weight empirical dynamics mixtures
at fixed truth. The first numerical column averages those worst errors. The
second averages replicas within each session/prevalence and then takes the
maximum over the 32 cells. The budget requires the second column <=5 pp, at
least 90% within 5 pp, and no unidentified fits. It is not just a mean-error gate.

The pooled-spike sensitivity readout and ternary versions also fail. None of
the tested population/method/length rows passes. This is not evidence that
every possible terminal estimator must fail.

## Diagnostic: why longer is not automatically better

The full-population mean spike count rises from 4.32 (20 ms) to 33.05 (100 ms),
with silent-segment fraction falling from 2.72% to zero (equal-session means).
Thus the 100-ms failure cannot be attributed simply to fewer observed spikes.
Longer segments also change the content estimand and its occupancy distribution.

At 100 ms, the pooled primary calibration has opposite signed biases against
actual geometric label quotas:

| Simulated stratum | Mean estimated minus true prevalence |
| --- | ---: |
| Stationary | +4.01 pp |
| Jumping, fitted scale | -6.05 pp |
| Jumping, 2x scale | -7.87 pp |

This is consistent with dynamics/occupancy-dependent readout sensitivity, not a
universal fix obtained by collecting a longer segment. Occupancy and the
late-crossing adversarial stratum are reported separately in the detailed CSV.
The 2x jumping result retains the bank's arena-truncation caveat.

## Oracle comparison

Known-generator calibration uses simulation truth that would not be available
on real data. These are equal-stratum mean absolute errors, NOT the worst-mixture
quantity in the decision table above:

| Segment | Pooled calibration | Known-generator oracle |
| --- | ---: | ---: |
| 20 ms | 2.94 pp | 3.19 pp |
| 40 ms | 2.58 pp | 2.51 pp |
| 60 ms | 3.10 pp | 2.31 pp |
| 100 ms | 4.27 pp | 2.45 pp |

The oracle advantage becomes appreciable at longer segments. At 20 ms there is
no oracle improvement, so unknown generator identity is not a sufficient
explanation for all budget failures. Finite calibration/validation sampling and
the worst-over-seven requirement also matter. The pooled estimator has more
calibration observations than each oracle fit; this comparison is diagnostic,
not a data-size-matched causal isolation of generator information. An oracle
mean below 5 pp does not establish the robust 90% criterion.

## Verification

- All 32 session/length tasks completed.
- 59,136 pure validation estimates, 5,632 empirical envelopes and 27,136
  numerical simplex-grid checks were produced.
- The independent audit passed all 32 tasks, comparing 39,424 calibration
  likelihood pairs with established helpers and independently refitting 3,584
  pure and 640 mixed cases with a separate optimizer.
- All 5,632 envelope rows were checked. 1,344 readouts were reconstructed from
  saved spike identities; maximum numerical discrepancy was 2.84e-14.
- The focused bank/calibration/experiment regression suite passed 107 tests.

Every common-weight empirical mixture MLE is between the pure-stratum MLEs
under a single fixed calibration. This is an envelope of estimates on this
finite validation bank, not a real-prevalence confidence interval or guarantee
over unseen/class-dependent mixtures. Four replicas and four rats do not
certify tail coverage. Monte Carlo repeats are not independent animals.

The frozen bank's limitations remain: decoder-derived dynamics anchors from a
selected event subset, fixed-total spike allocation, legacy overlapping
population definitions, and arena-limited realization of some 2x jumps. No
biological Home-content inference, new event scoring or hc-11 transfer was run.

## Artifacts

Server repository:
`/home/florianpfaff/HippoReplayDynamics-content-stability-20260914`

Base commit: `8b8473018da6ab5c293b13600399be41686e6c15`; branch
`test-cross-dataset-content-stability`. Research additions are uncommitted;
the experiment and audit manifests record source/input hashes and dirty state.

Under `/mnt/seagate10tb/florianpfaff/`:

- `regional-terminal-mixture-pf-20260916/`: raw estimates, saved scores and
  likelihoods, task manifests and input hashes.
- `regional-terminal-mixture-pf-audit-20260916/`: independent technical audit.
- `regional-terminal-mixture-pf-report-20260916/`: report, tables, decision,
  figure and manifest.

The defensible conclusion is: longer terminal windows alone did not make these
frozen regional readouts robust to the tested unknown dynamics mixtures.
Known-generator calibration helps at longer windows, but a deployable,
assumption-explicit regional-content bound remains unvalidated.
