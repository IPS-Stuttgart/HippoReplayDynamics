# Linear terminal occupancy: first invariance experiment

2026-09-16. Synthetic development result, not real replay content calibration.

## Decision

**The proposed two-mean calibration did not pass its first invariance screen.**
Do not use these coefficients to correct the real Home-rich/Home-poor gap.
No real-population reconciliation, new perturbation panel or held-out-cell
diagnostic was run, because the prerequisite failed.

Occupancy remains a potentially useful estimand. This result rejects neither
occupancy itself nor every possible estimator, and does not establish that
explicit dynamics inference is the only remedy.

## What was tested

The audited geometry-blind shared bank: 1,139 common count-trace templates,
eight PF sessions, four rats, seven dynamics/scale strata and four terminal
lengths. The detector, endpoint, maps, fixed-total allocation and all 22 frozen
population definitions were unchanged. Calibration and validation replicas
were disjoint; null panels stayed untouched.

Each 20-ms bin was independently Poisson decoded with a flat prior. Its Home
mass q was averaged over the terminal segment. Exact simulated path geometry
provided the fraction o of represented time inside the Home disc in each bin.
No union, maximum, binary visit label or HMM was used for the new estimand.

Two explicit calibration definitions were tested:

- Time-conditional: s=sum(o*q)/sum(o), f=sum((1-o)*q)/sum(1-o). These correspond
  to conditioning on Home at a uniformly sampled represented instant.
- Pure-window diagnostic: s/f are mean masses in fully-inside/fully-outside
  bins. Mixed bins are excluded from this calibration only, not from validation.

Primary silent bins retain prior mass, as in the previous experiment. Keeping
the actual Poisson silence likelihood was also tested, with similar results.

## First gate: mean-mass tolerance 0.02

For each session, the four calibration replicas were pooled within a stratum.
Both s and f had to vary by no more than 0.02 across all seven strata.
Full-population, time-conditional results:

| Terminal length | Sessions passing both tolerances | Median s range across strata | Median f range across strata |
| --- | ---: | ---: | ---: |
| 20 ms | 0/8 | 0.0371 | 0.0135 |
| 40 ms | 0/8 | 0.0469 | 0.0393 |
| 60 ms | 0/8 | 0.0527 | 0.0463 |
| 100 ms | 0/8 | 0.0633 | 0.0469 |

Pure-window calibration also passed 0/8 sessions at every length. Excluding
mixed windows reduces some f variation but does not make the coefficients
invariant under this tolerance. No support was missing in these full-population
comparisons. Leaving out each calibration replica in turn yielded 0/32 passes
per length for the primary definition. On the separate validation pools, the
primary screen passed 1/8 sessions at 20 ms and 0/8 at the other lengths.

These are point-estimate development screens with four calibration replicas,
not equivalence confidence intervals. Sampling uncertainty and the maximum over
seven strata remain relevant; the screen alone does not identify a biological
or generative cause of the differences.

## Transfer: some improvement, no validated remedy

Mean absolute occupancy error across 896 independent-validation panels per
length (eight sessions x seven strata x four quotas x four replicas):

| Length | Uncorrected mean mass | Pooled pure-window calibration | Pooled time-conditional calibration |
| --- | ---: | ---: | ---: |
| 20 ms | 15.86 pp | 3.28 pp | 3.34 pp |
| 40 ms | 11.72 pp | 2.50 pp | 4.77 pp |
| 60 ms | 9.05 pp | 2.70 pp | 5.31 pp |
| 100 ms | 6.52 pp | 3.08 pp | 5.20 pp |

Thus calibration is useful on average in this unchanged-tuning synthetic bank.
But the attractive 2.50-pp mean is not an error bound: the worst 40-ms pure-window
panel had 16.52-pp error. Among the 224 supported low-occupancy panels at 40 ms,
its mean error was 1.62 pp and maximum 7.48 pp; their actual truths ranged from
3.26% to 6.70%. This is not a complete validation of every occupancy in 2-10%.

Unbounded estimates were retained rather than clipped to hide failures. At
40 ms, 2/896 pure-window corrections and 158/896 time-conditional corrections
were outside [0,1]. All full-population fits had positive s-f and finite outputs.

Knowing the simulated generator does not uniformly solve this: oracle
time-conditional mean errors at 20/40/60/100 ms are 3.56/4.65/4.84/4.47 pp;
oracle pure-window errors are 3.50/2.52/2.24/2.39 pp. The oracle is diagnostic
and has fewer calibration observations than the pooled fit.

The earlier 40-ms binary 5.68-pp worst-mixture error is a different target and
aggregation. Comparing it directly with these average occupancy errors would
not establish an improvement in the same scientific quantity. No occupancy
budget was borrowed from the prevalence experiment and no full mixture-bound
claim is made after the first screen failed.

## Why linearity is insufficient

Within one sample, mean(q)=mean(o)*s_time+(1-mean(o))*f_time is an identity.
Transfer requires the same s/f on a new occupancy/path/count distribution.
Conditioning on the whole Home disc still mixes positions and firing patterns;
even a fully Home window may represent different within-disc locations.

A decisive algebraic counterexample uses a perfect readout q=o. With only pure
o=0/1 windows, s=1 and f=0. With equally frequent o=0.2/0.8 windows, s=0.68 and
f=0.32, even though q estimates occupancy perfectly in both cases. The
time-conditional means depend on within-window occupancy variance. Their
invariance is therefore not a necessary property of every good occupancy
estimator. A failure cannot by itself imply that occupancy is unidentifiable.

The pure-window diagnostic avoids that specific algebraic dependence, but still
failed the empirical screen. Further work would need to separate finite-sample
variation, within-region spatial sampling, count support and the response to
partial occupancy. Inferred dynamics is one possible approach, not the only one.

## Verification and provenance

- All 32 session/length tasks completed.
- Independent audit checked 24,640 emission rows, 3,520 invariance rows and
  78,848 validation corrections directly against saved event/bin arrays.
- It reconstructed 3,696 regional masses from raw spike identities with maximum
  discrepancy 3.11e-15. An independent 20-microsecond numerical dwell calculation
  agreed with sampled analytic occupancies within 0.000495.
- Every cumulative bin-dwell target was checked against the frozen bank truth.
- 116 focused regression tests and Ruff passed.

The frozen bank's limitations remain: selected decoder-derived dynamics anchors,
fixed spike totals, legacy population overlap, arena-limited large jumps, and
four-replica development sampling. The real 11.38%/2.80% endpoint summaries were
not relabeled as terminal occupancy. No statement about spatial truth, biological
Home content or population-coverage invariance is licensed by this experiment.

Server repository:
`/home/florianpfaff/HippoReplayDynamics-content-stability-20260914`

Base commit: `8b8473018da6ab5c293b13600399be41686e6c15`, branch
`test-cross-dataset-content-stability`, with uncommitted research additions.
Manifests record input/source hashes and dirty state.

Artifacts under `/mnt/seagate10tb/florianpfaff/`:

- `regional-occupancy-invariance-pf-20260916/`: readouts, truth, calibration,
  validation, protocol provenance, report and figure.
- `regional-occupancy-invariance-pf-audit-20260916/`: independent audit.
